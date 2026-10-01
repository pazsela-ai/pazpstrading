import os
import logging
import time
import json
import re
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

# הגדרת לוגים
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# הגדרת משתני סביבה מנוקים מרווחים
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
RENDER_EXTERNAL_URL = (os.getenv("RENDER_EXTERNAL_URL") or "").strip()

if not TELEGRAM_TOKEN:
    logging.error("CRITICAL: TELEGRAM_BOT_TOKEN is missing!")

bot = TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# אתחול מנוע ה-AI
ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini Client: {e}")

simulated_trades = []
last_scans = {"news": "טרם בוצעה", "tech": "טרם בוצעה"}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

# ---------------------------------------------------------
# 1. WEBHOOK & HEALTH CHECK
# ---------------------------------------------------------

@app.route('/')
@app.route('/health')
def home():
    return "OK - Market Scanner Bot Active!", 200

@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def telegram_webhook():
    """קבלת עדכונים בלייב מטלגרם דרך Webhook"""
    try:
        if request.headers.get('content-type') == 'application/json':
            json_string = request.get_data().decode('utf-8')
            update = types.Update.de_json(json_string)
            bot.process_new_updates([update])
            return 'OK', 200
    except Exception as e:
        logging.error(f"Error processing update: {e}")
    return 'OK', 200

def setup_webhook():
    if RENDER_EXTERNAL_URL and TELEGRAM_TOKEN:
        webhook_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_TOKEN}"
        try:
            bot.remove_webhook()
            time.sleep(1)
            bot.set_webhook(url=webhook_url)
            logging.info(f"Webhook set successfully to: {webhook_url}")
        except Exception as e:
            logging.error(f"Failed to set Webhook: {e}")

# ---------------------------------------------------------
# 2. TICKER FETCHERS
# ---------------------------------------------------------

def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        tables = pd.read_html(resp.text)
        return [str(t).replace('.', '-') for t in tables[0]['Symbol'].tolist()]
    except Exception as e:
        logging.error(f"Error SP500 fetch: {e}")
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"]

def get_nasdaq100_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        tables = pd.read_html(resp.text)
        for t in tables:
            for col in ['Ticker', 'Symbol']:
                if col in t.columns:
                    return [str(x).replace('.', '-').strip() for x in t[col].dropna().tolist() if len(str(x)) <= 5]
        raise ValueError("Nasdaq format error")
    except Exception as e:
        logging.error(f"Error Nasdaq fetch: {e}")
        return ["QQQ", "AVGO", "COST", "QCOM", "ADBE", "PANW", "NFLX"]

def get_ta125_tickers():
    try:
        url = "https://he.wikipedia.org/wiki/%D0%A0%D7%A9%D7%99%D7%9E%D7%AA_%D7%97%D7%91%D7%A8%D7%95%D7%AA_%D7%91%D7%9E%D7%93%D7%93_%D7%AA%22%D7%90-125"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        tables = pd.read_html(resp.text)
        df = tables[0]
        for col in df.columns:
            if any(k in str(col) for k in ['סימול', 'Ticker', 'סמל']):
                return [f"{str(t).strip().upper()}.TA" for t in df[col].dropna().tolist() if str(t).strip()]
        raise ValueError("TA125 format error")
    except Exception as e:
        logging.error(f"Error TA125 fetch: {e}")
        return ["ELAL.TA", "TEVA.TA", "ICL.TA", "NICE.TA", "LUMI.TA", "POLI.TA"]

def get_all_market_tickers():
    return list(set(get_sp500_tickers() + get_nasdaq100_tickers() + get_ta125_tickers()))

# ---------------------------------------------------------
# 3. TECHNICAL ANALYSIS & NEWS ENGINE
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
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else current_price * 0.03

        is_breakout = (current_price > latest['EMA20']) and (50 <= latest['RSI'] <= 70) and (latest['RSI'] > prev['RSI'])

        return {
            "ticker": ticker,
            "is_breakout": is_breakout,
            "price": round(current_price, 2),
            "tp": round(current_price + (atr_val * 2.0), 2),
            "sl": round(current_price - (atr_val * 1.2), 2),
            "rsi": round(latest['RSI'], 1)
        }
    except Exception:
        return None

def scan_single_ticker_task(ticker):
    tech_res = analyze_technical(ticker)
    if tech_res and tech_res["is_breakout"]:
        send_alert(ticker=ticker, trigger_type="TECHNICAL", tech_data=tech_res)

def _async_technical_scan():
    logging.info("Starting background technical scan...")
    last_scans["tech"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tickers = get_all_market_tickers()
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(scan_single_ticker_task, tickers)
    logging.info("Background technical scan completed.")

def scan_technical_market():
    # הרצת הסריקה ב-Thread נפרד כדי שלא לחסום את השרת
    threading.Thread(target=_async_technical_scan, daemon=True).start()

def analyze_ticker_specific_news(ticker):
    if not ai_client:
        return "❌ מנוע ה-AI אינו מחובר (חסר GEMINI_API_KEY)."

    clean_ticker = ticker.replace(".TA", "")
    query = f"{clean_ticker}+בורסה" if ".TA" in ticker else f"{clean_ticker}+stock"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=he&gl=IL&ceid=IL:he"
    
    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=8)
        feed = feedparser.parse(resp.content)
        items = [f"• {e.title}" for e in feed.entries[:4]]
        if not items:
            return f"ℹ️ לא נמצאו כתבות חדשות עדכניות עבור `{ticker}`."

        prompt = f"נתח בקצרה בעברית את הידיעות הבאות עבור מניית {ticker}:\n" + "\n".join(items)
        response = ai_client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return response.text.strip() if response and response.text else "❌ לא התקבלה תשובה מ-AI."
    except Exception as e:
        return f"❌ שגיאה בניתוח חדשות: {e}"

# ---------------------------------------------------------
# 4. TELEGRAM BOT HANDLERS
# ---------------------------------------------------------

def get_tradingview_link(ticker):
    clean_ticker = ticker.replace(".TA", "")
    exchange = "TASE" if ".TA" in ticker else "NASDAQ"
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{clean_ticker}"

def send_alert(ticker, trigger_type="TECHNICAL", news_reason="", tech_data=None, target_chat_id=None):
    dest_id = target_chat_id or CHAT_ID
    if not dest_id:
        return

    if not tech_data:
        tech_data = analyze_technical(ticker)

    price = tech_data["price"] if tech_data else "N/A"
    tp = tech_data["tp"] if tech_data else "N/A"
    sl = tech_data["sl"] if tech_data else "N/A"
    currency = "₪" if ".TA" in ticker else "$"

    msg = (
        f"📊 **התראת פריצה - {ticker}**\n\n"
        f"• מחיר כניסה: {currency}{price}\n"
        f"• יעד (TP): {currency}{tp}\n"
        f"• סטופ (SL): {currency}{sl}\n"
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
        "פקודות זמינות:\n"
        "• `/status` - בדיקת סטטוס מערכת\n"
        "• `/tech <TICKER>` - ניתוח טכני למניה (למשל: `/tech NVDA` או `/tech ELAL.TA`)\n"
        "• `/news_scan <TICKER>` - ניתוח חדשות AI למניה"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def handle_status(message):
    total_tickers = len(get_all_market_tickers())
    status_msg = (
        "⚙ **סטטוס מערכת:**\n\n"
        f"• AI Engine: {'✅ פעיל' if ai_client else '❌ חסר מפתח'}\n"
        f"• סריקה טכנית אחרונה: `{last_scans['tech']}`\n"
        f"• מניות במעקב דינמי: `{total_tickers}`\n"
        f"• עסקאות בסימולטור: `{len(simulated_trades)}`"
    )
    bot.reply_to(message, status_msg, parse_mode="Markdown")

@bot.message_handler(commands=['tech'])
def handle_tech_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "ציין סימול מניה. לדוגמה: `/tech NVDA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    tech_data = analyze_technical(ticker)
    if tech_data:
        send_alert(ticker=ticker, trigger_type="TECHNICAL", tech_data=tech_data, target_chat_id=message.chat.id)
    else:
        bot.reply_to(message, f"❌ לא נמצאו נתונים עבור `{ticker}`.")

@bot.message_handler(commands=['news_scan'])
def handle_news_scan_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "ציין סימול מניה. לדוגמה: `/news_scan NVDA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    res = analyze_ticker_specific_news(ticker)
    bot.reply_to(message, res, parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('sim_'))
def handle_simulation_callback(call):
    _, ticker, price, tp, sl = call.data.split('_')
    simulated_trades.append({"ticker": ticker, "price": price, "tp": tp, "sl": sl})
    bot.answer_callback_query(call.id, text=f"✅ עסקה על {ticker} נרשמה!")

# ---------------------------------------------------------
# 5. SCHEDULER & STARTUP
# ---------------------------------------------------------

scheduler = BackgroundScheduler()
scheduler.add_job(scan_technical_market, 'interval', minutes=60)
scheduler.start()

setup_webhook()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
