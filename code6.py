import os
import logging
import threading
import time
import re
import feedparser
import json
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from flask import Flask
from telebot import TeleBot, types
from apscheduler.schedulers.background import BackgroundScheduler
from google import genai

# הגדרת לוגים
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# הגדרות משתני סביבה
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# אתחול לקוח AI (Gemini)
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# מסד נתונים פשוט בזיכרון לסימולציות
simulated_trades = []
last_scans = {"news": "טרם בוצעה", "tech": "טרם בוצעה"}

# רשימת מעקב לדוגמה למניות במדדים מובילים (ת"א 125, S&P 500, Nasdaq)
WATCHLIST = [
    {"symbol": "ELAL.TA", "name": "אל על", "market": "TASE"},
    {"symbol": "TEVA.TA", "name": "טבע", "market": "TASE"},
    {"symbol": "NVDA", "name": "NVIDIA", "market": "NASDAQ"},
    {"symbol": "AAPL", "name": "Apple", "market": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft", "market": "NASDAQ"},
]

# ---------------------------------------------------------
# 1. מנוע ניתוח חדשות ואירועים (AI Catalyst Engine)
# ---------------------------------------------------------

def analyze_news_with_ai(headline, summary):
    """שולח את הידיעה ל-AI לניתוח סנטימנט, זיהוי מניה ונימוק"""
    if not ai_client:
        return None

    prompt = f"""
    נתח את הידיעה הכלכלית/חדשותית הבאה:
    כותרת: {headline}
    תקציר: {summary}

    אם הידיעה מצביעה על קטליזטור חיובי משמעותי עבור מניה מסוימת במדדים מובילים (ת"א 125, S&P500, Nasdaq):
    1. החזר את סימול המניה (למשל ELAL.TA, TEVA, NVDA).
    2. נימוק קצר וברור (משפט 1-2) מדוע הידיעה צפויה להשפיע לחיוב על מניית החברה או מתחרותיה.
    3. עוצמת סנטימנט (HIGH / MEDIUM).

    החזר תשובה בפורמט JSON בלבד:
    {{"is_relevant": true, "ticker": "ELAL.TA", "reason": "נימוק...", "conviction": "HIGH"}}
    אם אינה רלוונטית:
    {{"is_relevant": false}}
    """
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        res_text = response.text.strip()
        # ניקוי מוגן של תגיות JSON כדי למנוע SyntaxError בשרת
        res_text = re.sub(r'^```json\s*', '', res_text)
        res_text = re.sub(r'^```\s*', '', res_text)
        res_text = re.sub(r'\s*```$', '', res_text)
        
        return json.loads(res_text)
    except Exception as e:
        logging.error(f"Error in AI news analysis: {e}")
        return None

def scan_news_feed():
    """סורק עדכוני RSS מ-Google News עבור חדשות שוק ההון"""
    logging.info("Starting news scan...")
    last_scans["news"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    rss_urls = [
        "[https://news.google.com/rss/search?q=שוק+ההון+מניות+אל+על+טבע+דוחות&hl=he&gl=IL&ceid=IL:he](https://news.google.com/rss/search?q=שוק+ההון+מניות+אל+על+טבע+דוחות&hl=he&gl=IL&ceid=IL:he)",
        "[https://news.google.com/rss/search?q=stock+market+fda+approval+acquisition&hl=en-US&gl=US&ceid=US:en](https://news.google.com/rss/search?q=stock+market+fda+approval+acquisition&hl=en-US&gl=US&ceid=US:en)"
    ]

    found_any = False
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            ai_res = analyze_news_with_ai(entry.title, entry.get("summary", ""))
            if ai_res and ai_res.get("is_relevant"):
                ticker = ai_res["ticker"]
                reason = ai_res["reason"]
                send_alert(ticker=ticker, trigger_type="NEWS", news_reason=reason)
                found_any = True
    return found_any

# ---------------------------------------------------------
# 2. מנוע ניתוח טכני (Technical Screener)
# ---------------------------------------------------------

def analyze_technical(ticker):
    """מבצע ניתוח טכני ומחזיר אינדיקטורים ומחירי עבודה"""
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df.empty or len(df) < 20:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else current_price * 0.03

        # תנאי פריצה טכנית: מחיר מעל EMA20 ו-RSI בעלייה מעל 50
        is_breakout = (current_price > latest['EMA20']) and (latest['RSI'] > 50) and (latest['RSI'] > prev['RSI'])

        tp = round(current_price + (atr_val * 2), 2)
        sl = round(current_price - (atr_val * 1.2), 2)

        return {
            "is_breakout": is_breakout,
            "price": round(current_price, 2),
            "tp": tp,
            "sl": sl,
            "rsi": round(latest['RSI'], 1)
        }
    except Exception as e:
        logging.error(f"Technical analysis error for {ticker}: {e}")
        return None

def scan_technical_market():
    """סורק את רשימת המעקב לאיתור פריצות טכניות"""
    logging.info("Starting technical scan...")
    last_scans["tech"] = time.strftime("%Y-%m-%d %H:%M:%S")
    for item in WATCHLIST:
        ticker = item["symbol"]
        tech_res = analyze_technical(ticker)
        if tech_res and tech_res["is_breakout"]:
            send_alert(ticker=ticker, trigger_type="TECHNICAL", tech_data=tech_res)

# ---------------------------------------------------------
# 3. יצירת הודעות וכפתורים בטלגרם
# ---------------------------------------------------------

def get_tradingview_link(ticker):
    """יוצר קישור ישיר לגרף ב-TradingView"""
    clean_ticker = ticker.replace(".TA", "")
    exchange = "TASE" if ".TA" in ticker else "NASDAQ"
    return f"[https://www.tradingview.com/chart/?symbol=](https://www.tradingview.com/chart/?symbol=){exchange}:{clean_ticker}"

def send_alert(ticker, trigger_type="TECHNICAL", news_reason="", tech_data=None, target_chat_id=None):
    dest_id = target_chat_id or CHAT_ID
    if not dest_id:
        logging.warning("No CHAT_ID available for alert.")
        return

    if not tech_data:
        tech_data = analyze_technical(ticker)

    price = tech_data["price"] if tech_data else "N/A"
    tp = tech_data["tp"] if tech_data else "N/A"
    sl = tech_data["sl"] if tech_data else "N/A"
    tv_link = get_tradingview_link(ticker)

    if trigger_type == "NEWS":
        header = f"🚨 **התראת חדשות וקטליזטור - {ticker}**"
        body = f"🗞️ **ידיעה:** {news_reason}\n"
    elif trigger_type == "TECHNICAL":
        header = f"📊 **התראת פריצה טכנית - {ticker}**"
        body = f"📈 **אינדיקטור:** פריצת מומנטום (RSI: {tech_data.get('rsi', 'N/A')}) מעל ממוצעים נעים.\n"
    else:
        header = f"🔥 **התראה משולבת: חדשות + טכני - {ticker}**"
        body = f"🗞 **ידיעה:** {news_reason}\n📈 **ניתוח טכני:** פריצת מומנטום בגרף יומי.\n⭐ **עוצמת איתות:** גבוהה מאוד.\n"

    currency = "₪" if ".TA" in ticker else "$"
    msg = f"{header}\n\n{body}\n" \
          f"🎯 **מחירי עבודה מומלצים:**\n" \
          f"• מחיר כניסה: {currency}{price}\n" \
          f"• יעד רווח (TP): {currency}{tp}\n" \
          f"• סטופ לוס (SL): {currency}{sl}\n"

    keyboard = types.InlineKeyboardMarkup()
    sim_btn = types.InlineKeyboardButton("📈 בצע סימולציית קנייה", callback_data=f"sim_{ticker}_{price}_{tp}_{sl}")
    tv_btn = types.InlineKeyboardButton("📊 פתח גרף ב-TradingView", url=tv_link)
    keyboard.add(sim_btn, tv_btn)

    bot.send_message(dest_id, msg, parse_mode="Markdown", reply_markup=keyboard)

# ---------------------------------------------------------
# 4. ניהול פקודות וכפתורי טלגרם (Start, Controls, Status)
# ---------------------------------------------------------

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "🟢 **אפליקציית PazPSTrading פעילה ועובדת!**\n\n"
        "הבוט מריץ סריקות אוטומטיות ברקע לאיתור קטליזטורים חדשותיים ופריצות טכניות "
        "במדדים המובילים (ת\"א 125, S&P 500, Nasdaq).\n\n"
        "🛠️ **פקודות בקרה ובדיקה זמינות:**\n"
        "• `/status` - בדיקת תקינות הבוט וזמני הסריקות האוטומטיות\n"
        "• `/test_news` - הרצת סורק חדשות מיידית לבדיקה\n"
        "• `/test_tech` - הרצת סורק טכני מרוכז מיידית\n"
        "• `/news_scan <TICKER>` - ניתוח חדשותי נקודתי (למשל `/news_scan ELAL.TA`)\n"
        "• `/tech <TICKER>` - ניתוח טכני נקודתי (למשל `/tech NVDA`)\n"
        "• `/portfolio` - צפייה בתיק סימולציות העסקאות הווירטואליות"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def handle_status(message):
    status_msg = (
        "⚙️️ **סטטוס מערכת:**\n\n"
        f"• חיבור ל-AI: {'✅ תקין' if ai_client else '❌ לא מחובר'}\n"
        f"• סריקת חדשות אחרונה: `{last_scans['news']}`\n"
        f"• סריקה טכנית אחרונה: `{last_scans['tech']}`\n"
        f"• מספר עסקאות בסימולטור: `{len(simulated_trades)}`\n"
        f"• Chat ID מוגדר: `{CHAT_ID or 'לא מוגדר (השתמש בפקודות הידניות)'}`"
    )
    bot.reply_to(message, status_msg, parse_mode="Markdown")

@bot.message_handler(commands=['test_news'])
def handle_test_news(message):
    bot.reply_to(message, "🔎 מריץ סורק חדשות מבוסס AI בלייב...")
    found = scan_news_feed()
    if not found:
        bot.send_message(message.chat.id, "ℹ️ הסורק הסתיים: לא נמצאו כרגע חדשות חריגות עם קטליזטור חיובי במדדים.")

@bot.message_handler(commands=['test_tech'])
def handle_test_tech(message):
    bot.reply_to(message, "🔎 מריץ סורק טכני על רשימת המעקב...")
    scan_technical_market()
    bot.send_message(message.chat.id, "✅ הסריקה הטכנית הושלמה.")

@bot.message_handler(commands=['news_scan'])
def handle_news_scan_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "יש לציין סימול מניה. לדוגמה: `/news_scan ELAL.TA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    bot.reply_to(message, f"📰 מחפש ומנתח חדשות עבור `{ticker}`...", parse_mode="Markdown")
    tech_data = analyze_technical(ticker)
    send_alert(ticker=ticker, trigger_type="NEWS", news_reason="סריקת חדשות ידנית לבקשת המשתמש.", tech_data=tech_data, target_chat_id=message.chat.id)

@bot.message_handler(commands=['tech'])
def handle_tech_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "יש לציין סימול מניה. לדוגמה: `/tech TEVA.TA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    bot.reply_to(message, f"📊 מנתח אינדיקטורים טכניים עבור `{ticker}`...", parse_mode="Markdown")
    tech_data = analyze_technical(ticker)
    if tech_data:
        send_alert(ticker=ticker, trigger_type="TECHNICAL", tech_data=tech_data, target_chat_id=message.chat.id)
    else:
        bot.reply_to(message, f"❌ לא ניתן היה לשלוף נתונים טכניים עבור `{ticker}`.")

@bot.callback_query_handler(func=lambda call: call.data.startswith('sim_'))
def handle_simulation_callback(call):
    _, ticker, price, tp, sl = call.data.split('_')
    simulated_trades.append({
        "ticker": ticker,
        "entry_price": float(price),
        "tp": float(tp),
        "sl": float(sl),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    bot.answer_callback_query(call.id, text=f"✅ עסקה וירטואלית על {ticker} נרשמה!")
    bot.send_message(call.message.chat.id, f"📝 **עסקה נרשמה בסימולטור!**\nמניה: `{ticker}`\nכניסה: `{price}` | TP: `{tp}` | SL: `{sl}`", parse_mode="Markdown")

@bot.message_handler(commands=['portfolio'])
def handle_portfolio(message):
    if not simulated_trades:
        bot.reply_to(message, "אין עסקאות פעילות כרגע בסימולטור.")
        return
    text = "💼 **תיק סימולציות פעיל:**\n\n"
    for idx, trade in enumerate(simulated_trades, 1):
        text += f"{idx}. **{trade['ticker']}** | כניסה: {trade['entry_price']} | TP: {trade['tp']} | SL: {trade['sl']}\n"
    bot.reply_to(message, text, parse_mode="Markdown")

# ---------------------------------------------------------
# 5. Flask, Scheduler והפעלת Polling ברקע
# ---------------------------------------------------------

@app.route('/')
def home():
    return "PazPSTrading Bot is active and running!"

def start_background_tasks():
    # 1. תזמון משימות אוטומטיות
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_technical_market, 'interval', minutes=30)
    scheduler.add_job(scan_news_feed, 'interval', minutes=15)
    scheduler.start()

    # 2. הרצת הבוט בלולאה נפרדת
    def run_bot():
        try:
            bot.remove_webhook()
            logging.info("Starting Telegram Bot Polling...")
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except Exception as e:
            logging.error(f"Error running bot polling: {e}")

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

# הפעלה אוטומטית ברגע שהאפליקציה עולה (מבטיח עבודה ב-Gunicorn)
start_background_tasks()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
