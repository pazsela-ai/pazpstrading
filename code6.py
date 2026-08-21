import os
import json
import asyncio
import logging
import warnings
import feedparser
import requests
import yfinance as yf
from flask import Flask
from threading import Thread
from google import genai
from google.genai import types
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)

# השתקת אזהרות
warnings.filterwarnings("ignore")

# ==========================================
# 1. הגדרות יומנים ומפתחות API ממשתני סביבה
# ==========================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "5021033385")  # מזהה הצ'אט שלך

DEFAULT_BUDGET_USD = 1000
ATR_MULTIPLIER = 1.5

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
seen_articles = set()

GLOBAL_NEWS_FEEDS = [
    "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTR1dvSUwyMHZNRGx6TVRydUVnVkdVM1J5S0FBUAE?hl=en-US&gl=US&ceid=US:en",
    "https://finance.yahoo.com/news/rssindex",
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom"
]

# ==========================================
# 2. ניתוח אירועים באמצעות Gemini API
# ==========================================
def analyze_event_with_gemini(headline: str, summary: str) -> dict:
    if not client:
        return {"is_opportunity": False}

    prompt = f"""
    אתה אנליסט מסחר ואסטרטג מאקרו-כלכלי קפדני. סרוק את הידיעה הבאה ומצא אם יש בה אירוע מפתח 
    (ביטחוני, גיאופוליטי, חוזה עסקי, אישור רגולטורי/FDA, מיזוגים, שינוי סחר):

    כותרת: {headline}
    תקציר: {summary}

    אם הידיעה מכילה אירוע מפתח שיכול ליצור פוטנציאל פריצה חיובי לחברה ספציפית:
    1. חלץ את סימול המניה (Ticker). אם החברה נסחרת בלעדית בתל אביב, הוסף את הסיומת .TA (למשל LUMI.TA).
    2. תן את שם החברה בעברית.
    3. תן נימוק תמציתי במשפט אחד בלבד למה האירוע מייצר פוטנציאל צמיחה.
    4. תן הרחבה מפורטת של הידיעה עצמה (2-4 משפטים) הכוללת את הרקע, העובדות והפרטים המלאים של הידיעה עבור בלחיצה על "מידע נוסף".

    אם הידיעה כללית, שגרתית, או עוסקת במניה שכבר עשתה את כל הדרך למעלה - החזר is_opportunity = false.

    החזר בפורמט JSON בלבד:
    {{
        "is_opportunity": true/false,
        "ticker": "TICKER",
        "company_name": "שם החברה",
        "event_summary": "תיאור האירוע במשפט אחד",
        "reasoning": "הנימוק הכלכלי/עסקי בשורה אחת",
        "article_expansion": "הרחבה מפורטת של הידיעה והאירוע עצמו בלבד"
    }}
    """
    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                tools=[]
            )
        )
        data = json.loads(res.text)

        if not data.get("is_opportunity") or not data.get("ticker"):
            return {"is_opportunity": False}

        return data
    except Exception as e:
        logging.error(f"שגיאה בניתוח Gemini: {e}")
        return {"is_opportunity": False}

# ==========================================
# 3. חישוב כרטיסיית מסחר וניהול סיכונים
# ==========================================
def calculate_trade_card(ticker_symbol: str, budget_usd: float = DEFAULT_BUDGET_USD):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="2mo")

        if df.empty or len(df) < 20:
            return None

        currency_symbol = "₪" if ticker_symbol.endswith(".TA") else "$"

        # חישוב RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))

        if rsi >= 70:
            return None

        # חישוב ATR
        df["High-Low"] = df["High"] - df["Low"]
        df["High-PC"] = abs(df["High"] - df["Close"].shift(1))
        df["Low-PC"] = abs(df["Low"] - df["Close"].shift(1))
        df["TR"] = df[["High-Low", "High-PC", "Low-PC"]].max(axis=1)
        atr = df["TR"].rolling(14).mean().iloc[-1]

        entry_price = round(df["Close"].iloc[-1], 2)
        stop_distance = round(atr * ATR_MULTIPLIER, 2)
        stop_loss_price = round(entry_price - stop_distance, 2)

        effective_budget = budget_usd * (3.7 if currency_symbol == "₪" else 1)
        shares_to_buy = int(effective_budget // entry_price)

        if shares_to_buy < 1:
            return None

        total_investment = round(shares_to_buy * entry_price, 2)
        max_risk_usd = round(shares_to_buy * stop_distance, 2)

        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss_price,
            "stop_distance": stop_distance,
            "shares": shares_to_buy,
            "total_investment": total_investment,
            "max_risk": max_risk_usd,
            "currency": currency_symbol,
            "budget_usd": budget_usd
        }
    except Exception as e:
        logging.error(f"שגיאה בחישוב ניהול סיכונים עבור {ticker_symbol}: {e}")
        return None

def format_alert_message(analyzed: dict, trade_data: dict) -> str:
    curr = trade_data["currency"]
    return f"""🚨 **התראת פוטנציאל פריצה (Event-Driven)**

* **אירוע מפתח:** {analyzed['event_summary']}
* **מניה מזוהה:** {analyzed['company_name']} (`{analyzed['ticker']}`)
* **תמצית הנימוק:** {analyzed['reasoning']}

🎯 **כרטיסיית עבודה (תקציב ${trade_data['budget_usd']}):**
• **מחיר כניסה מומלץ:** {curr}{trade_data['entry_price']}
• **סטופ-לוס (Stop-Loss):** {curr}{trade_data['stop_loss']}
• **מרחק סיכון למניה:** {curr}{trade_data['stop_distance']}
• **כמות מניות לקנייה:** {trade_data['shares']} מניות
• **סך השקעה בפועל:** {curr}{trade_data['total_investment']}
• **הפסד מקסימלי מחושב:** **{curr}{trade_data['max_risk']} בלבד**
"""

def build_keyboard(ticker: str, current_budget: float) -> InlineKeyboardMarkup:
    clean_ticker = ticker.replace(".TA", "")
    tradingview_url = f"https://www.tradingview.com/symbols/{clean_ticker}/"

    budget_btn_text = "🔄 עדכן לתקציב $3,000" if current_budget == 1000 else "🔄 החזר לתקציב $1,000"
    target_budget = 3000 if current_budget == 1000 else 1000

    keyboard = [
        [InlineKeyboardButton(budget_btn_text, callback_data=f"recalc:{ticker}:{target_budget}")],
        [
            InlineKeyboardButton("📊 פתח גרף", url=tradingview_url),
            InlineKeyboardButton("ℹ️ מידע נוסף", callback_data=f"info:{ticker}")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

# ==========================================
# 4. סריקת הפידים העולמיים
# ==========================================
def scan_global_feeds():
    events = []
    headers = {
        "User-Agent": "MyMarketBot mybotuser@gmail.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }

    if len(seen_articles) > 1000:
        seen_articles.clear()

    for feed_url in GLOBAL_NEWS_FEEDS:
        try:
            req_headers = headers if "sec.gov" in feed_url else {"User-Agent": "Mozilla/5.0"}
            res = requests.get(feed_url, headers=req_headers, timeout=20)
            feed = feedparser.parse(res.content)
            
            for entry in feed.entries[:5]:
                link = entry.get("link", entry.get("id", ""))
                if link in seen_articles:
                    continue
                seen_articles.add(link)
                headline = entry.get("title", "")
                summary = entry.get("summary", "")
                events.append((headline, summary))
        except Exception as e:
            logging.error(f"שגיאה בסריקת פיד {feed_url}: {e}")
    return events

# ==========================================
# 5. טיפול באירועים ופקודות טלגרם
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("היי! הבוט מחובר, סורק אירועים גלובליים וישלח התראות בזמן אמת.")

async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data.split(":")
    action = data[0]
    ticker = data[1]

    stored_info = context.bot_data.get(ticker, {})

    if action == "recalc":
        new_budget = float(data[2])
        event_summary = stored_info.get("event", "אירוע מפתח דרמטי")
        company_name = stored_info.get("comp", ticker)
        reasoning = stored_info.get("reason", "פוטנציאל צמיחה בעקבות אירוע")

        trade_data = calculate_trade_card(ticker, budget_usd=new_budget)
        if not trade_data:
            await query.message.reply_text("⚠️ לא ניתן לחשב מחדש (הנתונים לא זמינים או המחיר זינק).")
            return

        analyzed = {
            "ticker": ticker,
            "company_name": company_name,
            "event_summary": event_summary,
            "reasoning": reasoning
        }

        updated_msg = format_alert_message(analyzed, trade_data)
        reply_markup = build_keyboard(ticker, new_budget)

        await query.message.edit_text(
            text=updated_msg,
            parse_mode="Markdown",
            reply_markup=reply_markup
        )

    elif action == "info":
        article_expansion = stored_info.get("expansion", "אין פרטים נוספים זמינים על הידיעה.")
        comp = stored_info.get("comp", ticker)
        
        info_msg = f"📰 **הרחבת הידיעה - {comp} (`{ticker}`):**\n\n{article_expansion}"
        
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=info_msg,
            parse_mode="Markdown"
        )

async def background_monitoring(app: Application):
    logging.info("לולאת הניטור והסריקה הופעלה...")
    while True:
        try:
            raw_events = scan_global_feeds()

            for headline, summary in raw_events:
                analyzed = analyze_event_with_gemini(headline, summary)

                if analyzed.get("is_opportunity"):
                    ticker = analyzed["ticker"]
                    trade_data = calculate_trade_card(ticker, budget_usd=DEFAULT_BUDGET_USD)

                    if trade_data:
                        app.bot_data[ticker] = {
                            "event": analyzed["event_summary"],
                            "comp": analyzed["company_name"],
                            "reason": analyzed["reasoning"],
                            "expansion": analyzed.get("article_expansion", "אין פרטים נוספים זמינים.")
                        }

                        msg = format_alert_message(analyzed, trade_data)
                        reply_markup = build_keyboard(ticker, DEFAULT_BUDGET_USD)

                        await app.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=msg,
                            parse_mode="Markdown",
                            reply_markup=reply_markup
                        )

            await asyncio.sleep(600)  # סריקה מדי 10 דקות
        except Exception as e:
            logging.error(f"שגיאה בלולאת הרקע: {e}")
            await asyncio.sleep(60)

# ==========================================
# 6. הרצת שרת הבוט ב-Thread נפרד
# ==========================================
def run_telegram_bot():
    if not TELEGRAM_BOT_TOKEN:
        logging.error("TELEGRAM_BOT_TOKEN missing!")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def post_init(application: Application):
        asyncio.create_task(background_monitoring(application))

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(handle_button_click))

    logging.info("Starting Telegram Bot Polling...")
    application.run_polling(stop_signals=None)

# הפעלת הבוט ברקע
bot_thread = Thread(target=run_telegram_bot, daemon=True)
bot_thread.start()

# ==========================================
# 7. שרת Flask ראשי עבור Render
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot & Market Scanner Active"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
