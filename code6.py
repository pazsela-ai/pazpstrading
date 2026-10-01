import os
import logging
import time
import requests
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

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
        logging.info("Gemini Client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini Client: {e}")

last_processed_news_titles = set()
simulated_trades = []
last_scans = {"news": "טרם בוצעה", "tech": "טרם בוצעה"}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

# ---------------------------------------------------------
# 1. WEBHOOK & BOT COMMANDS SETUP
# ---------------------------------------------------------

def setup_bot_commands():
    try:
        commands = [
            types.BotCommand("start", "הפעלת הבוט ותפריט ראשי"),
            types.BotCommand("status", "בדיקת סטטוס מערכת וחיבורים"),
            types.BotCommand("test_tech", "הרצת סריקה טכנית בלייב"),
            types.BotCommand("test_news", "בדיקת סריקת אירועים ב-AI"),
            types.BotCommand("tech", "ניתוח טכני למניה/קרן (למשל /tech QQQ)"),
            types.BotCommand("news_scan", "ניתוח חדשות למניה (למשל /news_scan NVDA)")
        ]
        bot.set_my_commands(commands)
    except Exception as e:
        logging.error(f"Failed to update bot commands: {e}")

@app.route('/')
@app.route('/health')
def home():
    return "OK - Full Automatic Market Scanner Active!", 200

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
# 2. TICKERS LIST (STOCKS + ETFS)
# ---------------------------------------------------------

STATIC_TICKERS = [
    # ETFs
    "QQQ", "SPY", "IWM", "TQQQ", "SQQQ", "SOXX", "SMH", "XLK", "XLF", "XLE",
    "XLV", "XLY", "XLP", "XLI", "XLU", "ARKK", "ARKG", "BITO", "GLD", "SLV",

    # US Tech & Pharma & Energy
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "MRNA",
    "PFE", "JNJ", "LLY", "ABBV", "XOM", "CVX", "COP", "SLB", "BA", "LMT",

    # TASE / Israel
    "ELAL.TA", "TEVA.TA", "ICL.TA", "NICE.TA", "LUMI.TA", "POLI.TA", "ESLT.TA",
    "DSCT.TA", "FIBI.TA", "AZRG.TA", "MVRN.TA", "DELTG.TA", "ENLT.TA", "ORA.TA"
]

def get_all_market_tickers():
    return list(set(STATIC_TICKERS))

# ---------------------------------------------------------
# 3. TECHNICAL ANALYSIS ENGINE
# ---------------------------------------------------------

def analyze_technical(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="60d", interval="1d")
        if df.empty or len(df) < 20:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else current_price * 0.03
        rsi_val = float(latest['RSI'])
        ema20 = float(latest['EMA20'])
        ema50 = float(latest['EMA50'])

        reasons = []
        if current_price > ema20:
            reasons.append(f"מחיר (${current_price:.2f}) מעל EMA20 (${ema20:.2f})")
        if ema20 > ema50:
            reasons.append("מגמה עולה: EMA20 מעל EMA50")
        if 50 <= rsi_val <= 70:
            reasons.append(f"מומנטום חיובי: RSI ברמה של {rsi_val:.1f}")

        is_breakout = (current_price > ema20) and (rsi_val >= 50)
        recommendation = "🟢 **מומלץ לכניסה (איתות פריצה)**" if is_breakout else "🔴 **לא מומלץ לכניסה כעת**"

        return {
            "ticker": ticker,
            "is_breakout": is_breakout,
            "recommendation": recommendation,
            "price": round(current_price, 2),
            "tp": round(current_price + (atr_val * 2.0), 2),
            "sl": round(current_price - (atr_val * 1.2), 2),
            "rsi": round(rsi_val, 1),
            "reasons": reasons
        }
    except Exception as e:
        logging.error(f"Error analyzing {ticker}: {e}")
        return None

# ---------------------------------------------------------
# 4. EVENT-DRIVEN NEWS AI ENGINE
# ---------------------------------------------------------

GLOBAL_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=stock+market+pharma+aviation+oil+gas&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%D7%91%D7%95%D7%A8%D7%A1%D7%94+%D7%AA%D7%A2%D7%95%D7%A4%D7%94+%D7%A4%D7%90%D7%A8%D7%9E%D7%94+%D7%A0%D7%A4%D7%90&hl=he&gl=IL&ceid=IL:he"
]

def scan_breaking_news_events():
    last_scans["news"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not ai_client or not CHAT_ID:
        return

    logging.info("Starting fast breaking news scan...")
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
            logging.error(f"Error fetching news feed: {e}")

    if not collected_articles:
        return

    prompt = (
        "אתה אנליסט מסחר מבוסס אירועים (Event-Driven Trading). נתח את החדשות הבאות שנאספו כעת:\n"
        + "\n".join([f"- {t}" for t in collected_articles]) +
        "\n\nאם יש אירוע קריטי המשפיע על מניה/קרן סל (כמו ניסוי רפואי, אירוע בטחוני/תעופתי, שינוי במחירי נפט/גז), רשום:\n"
        "1. האירוע שזוקק.\n"
        "2. מניות מומלצות לכניסה/יציאה (למשל ELAL.TA, MRNA, TEVA, XLE).\n"
        "3. המלצת פעולה ברורה.\n"
        "אם אין אירוע מספיק קריטי, ענה 'אין אירוע קריטי'."
    )

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        if response and response.text and "אין אירוע קריטי" not in response.text:
            msg = f"🚨 **איתות אירוע מתפרץ בזמן אמת!**\n\n{response.text.strip()}"
            bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
    except Exception as e:
        logging.error(f"AI Event Scan error: {e}")

def analyze_single_ticker_news(ticker):
    if not ai_client:
        return "❌ מנוע ה-AI אינו מחובר (חסר GEMINI_API_KEY ב-Render)."

    query = f"{ticker.replace('.TA', '')}+stock" if ".TA" not in ticker else f"{ticker.replace('.TA', '')}+מניה"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=he&gl=IL&ceid=IL:he"

    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=6)
        feed = feedparser.parse(resp.content)
        items = [f"• {e.title}" for e in feed.entries[:5]]

        if not items:
            return f"ℹ️ לא נמצאו כתבות חדשותיות עדכניות עבור `{ticker}`."

        prompt = f"נתח בקצרה בעברית את הידיעות החדשותיות עבור {ticker} ותן המלצה (חיובי/שלילי/ניטרלי):\n" + "\n".join(items)
        response = ai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        
        if response and response.text:
            return f"📰 **סיכום חדשות AI עבור {ticker}:**\n\n{response.text.strip()}"
        return "❌ לא התקבל מענה מ-Gemini AI."
    except Exception as e:
        return f"❌ שגיאה: {e}"

# ---------------------------------------------------------
# 5. AUTOMATED SCANNER TASKS
# ---------------------------------------------------------

def scan_single_ticker_task(ticker):
    tech_res = analyze_technical(ticker)
    if tech_res and tech_res["is_breakout"]:
        send_alert(ticker=ticker, tech_data=tech_res)

def run_automatic_technical_scan():
    logging.info("Starting automatic technical scan...")
    last_scans["tech"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tickers = get_all_market_tickers()
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(scan_single_ticker_task, tickers)

# ---------------------------------------------------------
# 6. TELEGRAM MESSAGING & HANDLERS
# ---------------------------------------------------------

def get_tradingview_link(ticker):
    clean_ticker = ticker.replace(".TA", "")
    exchange = "TASE" if ".TA" in ticker else "NASDAQ"
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{clean_ticker}"

def send_alert(ticker, tech_data=None, target_chat_id=None):
    dest_id = target_chat_id or CHAT_ID
    if not dest_id:
        return

    if not tech_data:
        tech_data = analyze_technical(ticker)

    if not tech_data:
        bot.send_message(dest_id, f"❌ לא ניתן לשלוף נתונים עבור `{ticker}`.")
        return

    price = tech_data["price"]
    tp = tech_data["tp"]
    sl = tech_data["sl"]
    rec = tech_data["recommendation"]
    currency = "₪" if ".TA" in ticker else "$"

    reasons_text = "\n".join([f"  • {r}" for r in tech_data["reasons"]]) if tech_data["reasons"] else "  • אין כרגע איתות פריצה משמעותי"

    msg = (
        f"📊 **ניתוח טכני - {ticker}**\n\n"
        f"📣 **המלצה:** {rec}\n\n"
        f"💡 **נימוקי הניתוח:**\n"
        f"{reasons_text}\n\n"
        f"🎯 **תכנית עבודה מוצעת:**\n"
        f"• מחיר נוכחי: {currency}{price}\n"
        f"• יעד רווח (TP): {currency}{tp}\n"
        f"• סטופ לוס (SL): {currency}{sl}\n"
    )

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(
        types.InlineKeyboardButton("📈 סימולציית קנייה", callback_data=f"sim_{ticker}_{price}_{tp}_{sl}"),
        types.InlineKeyboardButton("📊 TradingView", url=get_tradingview_link(ticker))
    )

    bot.send_message(dest_id, msg, parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "🟢 **הבוט PazPSTrading מחובר ופעיל!**\n\n"
        "הבוט מנטר ברקע סריקות טכניות ואירועי חדשות בזמן אמת.\n\n"
        "פקודות לבקרה ידנית:\n"
        "• `/status` - בדיקת סטטוס מערכת\n"
        "• `/test_tech` - הרצת סריקה טכנית בלייב\n"
        "• `/test_news` - הרצת סריקת אירועי AI בלייב\n"
        "• `/tech <TICKER>` - ניתוח טכני (למשל `/tech QQQ`)\n"
        "• `/news_scan <TICKER>` - ניתוח חדשות (למשל `/news_scan NVDA`)"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def handle_status(message):
    total_tickers = len(get_all_market_tickers())
    status_msg = (
        "⚙ **סטטוס מערכת:**\n\n"
        f"• AI Engine: {'✅ פעיל' if GEMINI_API_KEY else '❌ חסר מפתח'}\n"
        f"• מעקב דינמי (מניות + ETFs): `{total_tickers}`\n"
        f"• סריקה טכנית אחרונה: `{last_scans['tech']}`\n"
        f"• סריקת חדשות אחרונה: `{last_scans['news']}`\n"
        f"• ניטור אוטומטי ברקע: 🟢 מופעל"
    )
    bot.reply_to(message, status_msg, parse_mode="Markdown")

@bot.message_handler(commands=['test_tech'])
def handle_test_tech(message):
    bot.reply_to(message, "🔎 מריץ סריקה טכנית בלייב כעת...")
    threading.Thread(target=run_automatic_technical_scan, daemon=True).start()

@bot.message_handler(commands=['test_news'])
def handle_test_news(message):
    bot.reply_to(message, "📰 מריץ בדיקת אירועי AI בחדשות...")
    threading.Thread(target=scan_breaking_news_events, daemon=True).start()

@bot.message_handler(commands=['tech'])
def handle_tech_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "ציין סימול. לדוגמה: `/tech QQQ`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    tech_data = analyze_technical(ticker)
    send_alert(ticker=ticker, tech_data=tech_data, target_chat_id=message.chat.id)

@bot.message_handler(commands=['news_scan'])
def handle_news_scan_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "ציין סימול. לדוגמה: `/news_scan NVDA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    res = analyze_single_ticker_news(ticker)
    bot.reply_to(message, res, parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('sim_'))
def handle_simulation_callback(call):
    _, ticker, price, tp, sl = call.data.split('_')
    simulated_trades.append({"ticker": ticker, "price": price, "tp": tp, "sl": sl})
    bot.answer_callback_query(call.id, text=f"✅ עסקה על {ticker} נרשמה בסימולטור!")

# ---------------------------------------------------------
# 7. BACKGROUND SCHEDULER
# ---------------------------------------------------------

scheduler = BackgroundScheduler(daemon=True)
# סריקת אירועים בחדשות כל 15 דקות
scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
# סריקה טכנית אוטומטית כל שעה
scheduler.add_job(run_automatic_technical_scan, 'interval', hours=1)
scheduler.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
