import os
import logging
import time
import requests
import re
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser
import threading
from google import genai
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request
from telebot import TeleBot, types
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
RENDER_EXTERNAL_URL = (os.getenv("RENDER_EXTERNAL_URL") or "").strip()

bot = TeleBot(TELEGRAM_TOKEN, threaded=False)
app = Flask(__name__)

# שימוש במודל הנתמך
GEMINI_MODEL = "gemini-1.5-flash"

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
        logging.info("Gemini Client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini Client: {e}")

last_processed_news_titles = set()
last_scans = {"news": "טרם בוצעה", "tech": "טרם בוצעה"}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

# ---------------------------------------------------------
# 1. WEBHOOK & COMMANDS
# ---------------------------------------------------------

def setup_bot_commands():
    try:
        commands = [
            types.BotCommand("start", "הפעלת הבוט ותפריט ראשי"),
            types.BotCommand("status", "בדיקת סטטוס מערכת"),
            types.BotCommand("test_tech", "הרצת סריקה טכנית בלייב"),
            types.BotCommand("test_news", "סריקת אירועי AI חדשותיים"),
            types.BotCommand("tech", "ניתוח טכני מעמיק (למשל /tech NVDA)"),
            types.BotCommand("news_scan", "ניתוח חדשות AI (למשל /news_scan ELAL.TA)")
        ]
        bot.set_my_commands(commands)
    except Exception as e:
        logging.error(f"Failed to update bot commands: {e}")

@app.route('/')
@app.route('/health')
def home():
    return "OK - Advanced Event & Technical Scanner Active!", 200

@app.route('/init_webhook', methods=['GET', 'POST'])
def init_webhook():
    if not TELEGRAM_TOKEN or not RENDER_EXTERNAL_URL:
        return "Missing variables", 400
    url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_TOKEN}"
    try:
        bot.remove_webhook()
        time.sleep(1)
        res = bot.set_webhook(url=url)
        setup_bot_commands()
        return f"Success! Webhook set to {url}" if res else "Telegram rejected webhook", 200
    except Exception as e:
        return f"Error: {e}", 500

@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def telegram_webhook():
    if request.headers.get('content-type') == 'application/json':
        try:
            json_string = request.get_data().decode('utf-8')
            update = types.Update.de_json(json_string)
            bot.process_new_updates([update])
            return 'OK', 200
        except Exception as e:
            logging.error(f"Webhook error: {e}")
            return 'OK', 200
    return 'Forbidden', 403

# ---------------------------------------------------------
# 2. ADVANCED TECHNICAL ENGINE (MULTI-INDICATOR SCORE)
# ---------------------------------------------------------

def analyze_technical_deep(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="100d", interval="1d")
        if df.empty or len(df) < 50:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # חישוב אינדיקטורים מרובים
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200) if len(df) >= 200 else df['EMA50']
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # MACD & Volatility
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
        
        # Volume
        df['VOL_SMA20'] = ta.sma(df['Volume'], length=20)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else current_price * 0.03
        rsi_val = float(latest['RSI'])
        ema20 = float(latest['EMA20'])
        ema50 = float(latest['EMA50'])
        vol_now = float(latest['Volume'])
        vol_avg = float(latest['VOL_SMA20']) if not pd.isna(latest['VOL_SMA20']) else 1

        score = 0
        reasons = []

        # 1. מגמה ראשית
        if current_price > ema20 > ema50:
            score += 25
            reasons.append(f"מגמה עולה חזקה (מחיר > EMA20 > EMA50)")

        # 2. RSI ומומנטום
        if 52 <= rsi_val <= 68:
            score += 25
            reasons.append(f"מומנטום בריא: RSI ברמה של {rsi_val:.1f}")
        elif rsi_val > 70:
            reasons.append(f"⚠️ RSI גבוה ({rsi_val:.1f}) - קרוב לאזור קניית יתר")

        # 3. אישור נפח מסחר (Volume Confirmation)
        if vol_now > (vol_avg * 1.3):
            score += 25
            reasons.append(f"זינוק בנפח המסחר ({int(vol_now/vol_avg*100)}% ממוצע)")

        # 4. MACD Cross
        if 'MACD' in df and latest['MACD'] > latest['MACD_SIGNAL']:
            score += 25
            reasons.append("אישור MACD עולה (MACD מעל Signal)")

        is_quality_breakout = score >= 75
        recommendation = "🟢 **מומלץ לכניסה (איתות איכותי)**" if is_quality_breakout else "🔴 **המתנה (איכות איתות נמוכה)**"

        return {
            "ticker": ticker,
            "score": score,
            "is_breakout": is_quality_breakout,
            "recommendation": recommendation,
            "price": round(current_price, 2),
            "tp": round(current_price + (atr_val * 2.2), 2),
            "sl": round(current_price - (atr_val * 1.2), 2),
            "rsi": round(rsi_val, 1),
            "reasons": reasons
        }
    except Exception as e:
        logging.error(f"Error deep analyzing {ticker}: {e}")
        return None

# ---------------------------------------------------------
# 3. EVENT-DRIVEN NEWS SCANNER (DYNAMIC & FIXED)
# ---------------------------------------------------------

GLOBAL_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=stock+market+pharma+aviation+oil+gas&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%D7%91%D7%95%D7%A8%D7%A1%D7%94+%D7%AA%D7%A2%D7%95%D7%A4%D7%94+%D7%A4%D7%90%D7%A8%D7%9E%D7%94+%D7%A0%D7%A4%D7%90&hl=he&gl=IL&ceid=IL:he"
]

def scan_breaking_news_events():
    last_scans["news"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not ai_client or not CHAT_ID:
        return

    logging.info("Starting breaking news scan...")
    collected_articles = []

    for feed_url in GLOBAL_NEWS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=HEADERS, timeout=6)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:8]:
                if entry.title not in last_processed_news_titles:
                    collected_articles.append(entry.title)
                    last_processed_news_titles.add(entry.title)
        except Exception as e:
            logging.error(f"Error fetching feed: {e}")

    if not collected_articles:
        return

    prompt = (
        "אתה אנליסט מסחר מבוסס אירועים. נתח את הידיעות החדשותיות הבאות:\n"
        + "\n".join([f"- {t}" for t in collected_articles]) +
        "\n\nאם יש אירוע קריטי המשפיע על מניות/סקטורים (ניסוי רפואי, אירוע ביטחוני/תעופה, מחירי נפט/גז):\n"
        "1. תמצת את האירוע בקצרה.\n"
        "2. ציין סימולי מניות רלוונטיים באנגלית (למשל: MRNA, TEVA, ELAL.TA, XLE).\n"
        "3. רשום המלצת פעולה.\n"
        "אם אין אירוע חריג, ענה 'אין אירוע קריטי'."
    )

    try:
        response = ai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        if response and response.text and "אין אירוע קריטי" not in response.text:
            msg = f"🚨 **איתות אירוע מתפרץ (Event-Driven)!**\n\n{response.text.strip()}"
            bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
            
            # חילוץ טיקרים אוטומטי מהטקסט של AI והרצת בדיקה טכנית
            found_tickers = re.findall(r'\b[A-Z]{2,5}(?:\.TA)?\b', response.text)
            for tick in set(found_tickers):
                tech_data = analyze_technical_deep(tick)
                if tech_data:
                    send_alert(tick, tech_data)

    except Exception as e:
        logging.error(f"AI Event Scan error: {e}")

def analyze_single_ticker_news(ticker):
    if not ai_client:
        return "❌ מנוע AI אינו מחובר."

    query = f"{ticker.replace('.TA', '')}+stock"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=6)
        feed = feedparser.parse(resp.content)
        items = [f"• {e.title}" for e in feed.entries[:5]]

        if not items:
            return f"ℹ️ לא נמצאו כתבות חדשותיות עבור `{ticker}`."

        prompt = f"נתח בקצרה בעברית את הידיעות עבור {ticker} ותן המלצה (חיובי/שלילי/ניטרלי):\n" + "\n".join(items)
        response = ai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        
        if response and response.text:
            return f"📰 **ניתוח חדשות AI עבור {ticker}:**\n\n{response.text.strip()}"
        return "❌ לא התקבל מענה מ-Gemini AI."
    except Exception as e:
        return f"❌ שגיאה: {e}"

# ---------------------------------------------------------
# 4. TELEGRAM HANDLERS
# ---------------------------------------------------------

def send_alert(ticker, tech_data=None, target_chat_id=None):
    dest_id = target_chat_id or CHAT_ID
    if not dest_id:
        return

    if not tech_data:
        tech_data = analyze_technical_deep(ticker)

    if not tech_data:
        bot.send_message(dest_id, f"❌ לא ניתן לשלוף נתונים עבור `{ticker}`.")
        return

    price = tech_data["price"]
    tp = tech_data["tp"]
    sl = tech_data["sl"]
    rec = tech_data["recommendation"]
    score = tech_data["score"]
    currency = "₪" if ".TA" in ticker else "$"

    reasons_text = "\n".join([f"  • {r}" for r in tech_data["reasons"]]) if tech_data["reasons"] else "  • ללא אינדיקטור מיוחד"

    msg = (
        f"📊 **ניתוח טכני מעמיק - {ticker}** (ציון: {score}/100)\n\n"
        f"📣 **המלצה:** {rec}\n\n"
        f"💡 **פרמטרים שנבדקו:**\n"
        f"{reasons_text}\n\n"
        f"🎯 **תכנית עבודה מוצעת:**\n"
        f"• מחיר: {currency}{price}\n"
        f"• יעד רווח (TP): {currency}{tp}\n"
        f"• סטופ לוס (SL): {currency}{sl}\n"
    )

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(
        types.InlineKeyboardButton("📈 TradingView", url=f"https://www.tradingview.com/chart/?symbol={ticker.replace('.TA','')}")
    )

    bot.send_message(dest_id, msg, parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "🟢 **הבוט PazPSTrading מחובר ופעיל במתכונת מעמיקה!**", parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def handle_status(message):
    status_msg = (
        "⚙ **סטטוס מערכת:**\n\n"
        f"• AI Engine: {'✅ פעיל (' + GEMINI_MODEL + ')' if GEMINI_API_KEY else '❌ לא מחובר'}\n"
        f"• סריקת אירועים אוטומטית: 🟢 מופעלת (כל 15 דק')\n"
        f"• סריקת חדשות אחרונה: `{last_scans['news']}`"
    )
    bot.reply_to(message, status_msg, parse_mode="Markdown")

@bot.message_handler(commands=['tech'])
def handle_tech_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "ציין סימול. לדוגמה: `/tech NVDA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    tech_data = analyze_technical_deep(ticker)
    send_alert(ticker=ticker, tech_data=tech_data, target_chat_id=message.chat.id)

@bot.message_handler(commands=['news_scan'])
def handle_news_scan_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "ציין סימול. לדוגמה: `/news_scan ELAL.TA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    bot.reply_to(message, "🔎 מריץ ניתוח חדשות ב-AI...")
    res = analyze_single_ticker_news(ticker)
    bot.reply_to(message, res, parse_mode="Markdown")

# ---------------------------------------------------------
# 5. BACKGROUND SCHEDULER
# ---------------------------------------------------------

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
scheduler.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
