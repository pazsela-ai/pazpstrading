import os
import logging
import threading
import time
import feedparser
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
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # מזהה הצ'אט לשליחת התראות אוטומטיות
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# אתחול לקוח AI (Gemini)
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# מסד נתונים פשוט בזיכרון/קובץ לסימולטור עסקאות
simulated_trades = []

# רשימת מעקב לדוגמה (תל אביב 125, S&P 500, Nasdaq)
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
    1. הטריידר/סימול המניה (למשל ELAL.TA, TEVA, NVDA).
    2. נימוק קצר וברור (משפט 1-2) מדוע הידיעה צפויה להשפיע לחיוב.
    3. סנטימנט (HIGH_CONVICTION / MEDIUM_CONVICTION).

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
        import json
        res_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(res_text)
    except Exception as e:
        logging.error(f"Error in AI news analysis: {e}")
        return None

def scan_news_feed():
    """סורק עדכוני RSS מ-Google News עבור חדשות שוק ההון"""
    logging.info("Starting news scan...")
    rss_urls = [
        "https://news.google.com/rss/search?q=שוק+ההון+מניות+מנכ\"ל+דוחות&hl=he&gl=IL&ceid=IL:he",
        "https://news.google.com/rss/search?q=stock+market+fda+approval+acquisition&hl=en-US&gl=US&ceid=US:en"
    ]

    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:  # 5 הידיעות האחרונות
            ai_res = analyze_news_with_ai(entry.title, entry.get("summary", ""))
            if ai_res and ai_res.get("is_relevant"):
                ticker = ai_res["ticker"]
                reason = ai_res["reason"]
                send_alert(ticker=ticker, trigger_type="NEWS", news_reason=reason)

# ---------------------------------------------------------
# 2. מנוע ניתוח טכני (Technical Screener)
# ---------------------------------------------------------

def analyze_technical(ticker):
    """מבצע ניתוח טכני ומחזיר אינדיקטורים ומחירי עבודה"""
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df.empty or len(df) < 20:
            return None

        # תיקון MultiIndex של yfinance אם קיים
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

        # תנאי פריצה טכנית: מחיר מעל EMA20 ו-EMA50 + RSI במגמה עולה
        is_breakout = (current_price > latest['EMA20']) and (latest['RSI'] > 55) and (latest['RSI'] > prev['RSI'])

        # חישוב יעד רווח (TP) וסטופ לוס (SL)
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
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{clean_ticker}"

def send_alert(ticker, trigger_type="TECHNICAL", news_reason="", tech_data=None):
    if not CHAT_ID:
        logging.warning("CHAT_ID not configured, skipping alert.")
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
    else:  # COMBINED
        header = f"🔥 **התראה משולבת: חדשות + טכני - {ticker}**"
        body = f"🗞️️ **ידיעה:** {news_reason}\n📈 **ניתוח טכני:** פריצת מומנטום בגרף יומי.\n⭐ **עוצמת איתות:** גבוהה מאוד (הצלבה).\n"

    msg = f"{header}\n\n{body}\n" \
          f"🎯 **מחירי עבודה מומלצים:**\n" \
          f"• מחיר כניסה: ₪{price} / ${price}\n" \
          f"• יעד רווח (TP): ₪{tp} / ${tp}\n" \
          f"• סטופ לוס (SL): ₪{sl} / ${sl}\n"

    # יצירת כפתורים אינטראקטיביים
    keyboard = types.InlineKeyboardMarkup()
    sim_btn = types.InlineKeyboardButton("📈 בצע סימולציית קנייה", callback_data=f"sim_{ticker}_{price}_{tp}_{sl}")
    tv_btn = types.InlineKeyboardButton("📊 פתח גרף ב-TradingView", url=tv_link)
    keyboard.add(sim_btn, tv_btn)

    bot.send_message(CHAT_ID, msg, parse_mode="Markdown", reply_markup=keyboard)

# ---------------------------------------------------------
# 4. ניהול כפתורים ופקודות טלגרם
# ---------------------------------------------------------

@bot.callback_query_handler(func=lambda call: call.data.startswith('sim_'))
def handle_simulation_callback(call):
    """טיפול בלחיצה על כפתור סימולציית קנייה"""
    _, ticker, price, tp, sl = call.data.split('_')
    simulated_trades.append({
        "ticker": ticker,
        "entry_price": float(price),
        "tp": float(tp),
        "sl": float(sl),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    bot.answer_callback_query(call.id, text=f"✅ עסקה וירטואלית על {ticker} נרשמה בסימולטור!")
    bot.send_message(call.message.chat.id, f"📝 **סימולציית קנייה נפלטה בהצלחה!**\nמניה: `{ticker}`\nמחיר כניסה: `{price}`\nיעד רווח: `{tp}`\nסטופ לוס: `{sl}`", parse_mode="Markdown")

@bot.message_handler(commands=['scan'])
def handle_manual_scan(message):
    """פקודה לסריקה ידנית של מניה: /scan TEVA"""
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "נא לציין סימול מניה. לדוגמה: `/scan TEVA` או `/scan ELAL.TA`", parse_mode="Markdown")
            return
        
        ticker = parts[1].upper()
        bot.reply_to(message, f"🔍 מבצע ניתוח טכני וחדשותי עבור `{ticker}`...", parse_mode="Markdown")
        
        tech_res = analyze_technical(ticker)
        if tech_res:
            send_alert(ticker=ticker, trigger_type="TECHNICAL", tech_data=tech_res)
        else:
            bot.reply_to(message, f"❌ לא ניתן היה לשלוף נתונים עבור `{ticker}`. ודא שהסימול תקין.")
    except Exception as e:
        bot.reply_to(message, f"שגיאה בביצוע הסריקה: {e}")

@bot.message_handler(commands=['portfolio'])
def handle_portfolio(message):
    """צפייה בתיק הסימולציות הפעיל"""
    if not simulated_trades:
        bot.reply_to(message, "אין עסקאות פעילות כרגע בסימולטור.")
        return
    
    text = "💼 **תיק סימולציות פעיל:**\n\n"
    for idx, trade in enumerate(simulated_trades, 1):
        text += f"{idx}. **{trade['ticker']}** | כניסה: {trade['entry_price']} | TP: {trade['tp']} | SL: {trade['sl']}\n"
    
    bot.reply_to(message, text, parse_mode="Markdown")

# ---------------------------------------------------------
# 5. הגדרת Flask ושרת ה-Web
# ---------------------------------------------------------

@app.route('/')
def home():
    return "PazPSTrading Bot is running smoothly!"

def start_scheduler():
    scheduler = BackgroundScheduler()
    # הרצת סריקה טכנית מדי 30 דקות
    scheduler.add_job(scan_technical_market, 'interval', minutes=30)
    # הרצת סריקת חדשות מדי 15 דקות
    scheduler.add_job(scan_news_feed, 'interval', minutes=15)
    scheduler.start()

def run_bot():
    bot.infinity_polling()

if __name__ == '__main__':
    start_scheduler()
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.start()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
