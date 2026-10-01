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
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

# ---------------------------------------------------------
# 1. WEBHOOK SETUP
# ---------------------------------------------------------

@app.route('/')
@app.route('/health')
def home():
    return "OK - Event-Driven Market Scanner Active!", 200

@app.route('/init_webhook', methods=['GET', 'POST'])
def init_webhook():
    if not TELEGRAM_TOKEN or not RENDER_EXTERNAL_URL:
        return "Missing variables", 400
    url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_TOKEN}"
    try:
        bot.remove_webhook()
        time.sleep(1)
        res = bot.set_webhook(url=url)
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
# 2. FAST EVENT-DRIVEN NEWS SCANNER (EVERY 20 MINS)
# ---------------------------------------------------------

GLOBAL_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=stock+market+pharma+aviation+oil+gas&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%D7%91%D7%95%D7%A8%D7%A1%D7%94+%D7%AA%D7%A2%D7%95%D7%A4%D7%94+%D7%A4%D7%90%D7%A8%D7%9E%D7%94+%D7%A0%D7%A4%D7%90&hl=he&gl=IL&ceid=IL:he"
]

def scan_breaking_news_events():
    """סריקה מהירה של חדשות מתפרצות וחיבורן למניות רלוונטיות באמצעות AI"""
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
        logging.info("No new news events found in this cycle.")
        return

    prompt = (
        "אתה אנליסט מסחר מבוסס אירועים (Catalyst Trading). נתח את החדשות הבאות שנאספו כעת:\n"
        + "\n".join([f"- {t}" for t in collected_articles]) +
        "\n\nאם יש בין הידיעות אירוע משמעותי שמשפיע ישירות על סקטור או מניה (למשל: ניסוי קליני מוצלח/כשל בפארמה, מתיחות במצר מים/נפט, אירוע תעופתי, דוחות כספיים), ציין:\n"
        "1. מהו האירוע.\n"
        "2. אילו מניות/קרנות סל מושפעות (למשל: MRNA, TEVA, ELAL.TA, XLE).\n"
        "3. המלצת פעולה ברורה: מומלץ לכניסה / לא מומלץ לכניסה.\n"
        "אם אין אירוע מספיק קריטי, ענה 'אין אירוע קריטי'."
    )

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        if response and response.text and "אין אירוע קריטי" not in response.text:
            msg = f"🚨 **איתות חדשותי/אירוע מתפרץ בזמן אמת!**\n\n{response.text.strip()}"
            bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
    except Exception as e:
        logging.error(f"AI Event Scan error: {e}")

# ---------------------------------------------------------
# 3. TELEGRAM HANDLERS
# ---------------------------------------------------------

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "🟢 **בוט מסחר מבוסס אירועים (Event-Driven) פעיל!**\nהבוט סורק אירועי חדשות מתפרצים כל 20 דקות ומדווח בזמן אמת.", parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def handle_status(message):
    bot.reply_to(message, f"⚙ **סטטוס מערכת:**\n• AI Engine: {'✅ פעיל' if GEMINI_API_KEY else '❌ לא פעיל'}\n• ניטור אירועים בזמן אמת: 🟢 מופעל (כל 20 דקות)", parse_mode="Markdown")

# ---------------------------------------------------------
# 4. BACKGROUND SCHEDULER
# ---------------------------------------------------------

scheduler = BackgroundScheduler(daemon=True)
# מורץ כל 20 דקות לזיהוי אירועים בזמן אמת
scheduler.add_job(scan_breaking_news_events, 'interval', minutes=20)
scheduler.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
