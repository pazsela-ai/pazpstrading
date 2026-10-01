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
# 2. EXPANDED TICKER LISTS
# ---------------------------------------------------------

STATIC_TICKERS = [
    # US Tech & Major S&P 500
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "NFLX",
    "MS", "JPM", "BAC", "V", "MA", "UNH", "PG", "HD", "DIS", "PYPL", "COST", "CSCO",
    "ORCL", "CRM", "PEP", "KO", "XOM", "CVX", "NKE", "LLY", "AVGO", "QCOM", "TXN",
    # TASE / Israel
    "ELAL.TA", "TEVA.TA", "ICL.TA", "NICE.TA", "LUMI.TA", "POLI.TA", "ESLT.TA",
    "DSCT.TA", "FIBI.TA", "AZRG.TA", "MVRN.TA", "DELTG.TA", "ENLT.TA", "ORA.TA"
]

def get_all_market_tickers():
    tickers = set(STATIC_TICKERS)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=HEADERS, timeout=5)
        tables = pd.read_html(resp.text)
        wiki_sp = [str(t).replace('.', '-') for t in tables[0]['Symbol'].tolist()]
        tickers.update(wiki_sp)
    except Exception as e:
        logging.warning(f"Wikipedia fetch skipped: {e}")
    return list(tickers)

# ---------------------------------------------------------
# 3. TECHNICAL ANALYSIS ENGINE (ENHANCED REASONING)
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
        df['Vol_SMA'] = ta.sma(df['Volume'], length=20)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else current_price * 0.03
        rsi_val = float(latest['RSI'])
        ema20 = float(latest['EMA20'])
        ema50 = float(latest['EMA50'])

        # נימוקים טכניים
        reasons = []
        if current_price > ema20:
            reasons.append(f"מחיר (${current_price:.2f}) מעל ממוצע נע EMA20 (${ema20:.2f})")
        if ema20 > ema50:
            reasons.append("מגמה עולה: EMA20 נמצא מעל EMA50")
        if 50 <= rsi_val <= 70:
            reasons.append(f"מומנטום חיובי בריא: RSI ברמה של {rsi_val:.1f}")
        elif rsi_val > 70:
            reasons.append(f"מומנטום עוצמתי (קניית יתר): RSI ברמה של {rsi_val:.1f}")
        if latest['Volume'] > prev['Vol_SMA']:
            reasons.append("נפח מסחר גבוה מהממוצע ב-20 הימים האחרונים")

        is_breakout = (current_price > ema20) and (rsi_val >= 50)

        return {
            "ticker": ticker,
            "is_breakout": is_breakout,
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
# 4. NEWS ANALYSIS (UPDATED TO GEMINI-2.5-FLASH)
# ---------------------------------------------------------

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
        
        # עדכון שם המודל לגרסה הנתמכת
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip() if response and response.text else "❌ לא התקבלה תשובה מ-AI."
    except Exception as e:
        return f"❌ שגיאה בניתוח חדשות: {e}"

# ---------------------------------------------------------
# 5. TELEGRAM BOT HANDLERS
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
    currency = "₪" if ".TA" in ticker else "$"

    reasons_text = "\n".join([f"  • {r}" for r in tech_data["reasons"]]) if tech_data["reasons"] else "  • פריצת מומנטום כללית"

    msg = (
        f"📊 **ניתוח טכני מפורט - {ticker}**\n\n"
        f"💡 **נימוקי הניתוח והפריצה:**\n"
        f"{reasons_text}\n\n"
        f"🎯 **תכנית עבודה מומלצת:**\n"
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
        "פקודות זמינות:\n"
        "• `/status` - בדיקת סטטוס מערכת\n"
        "• `/tech <TICKER>` - ניתוח טכני מנומק למניה\n"
        "• `/news_scan <TICKER>` - ניתוח חדשות AI למניה"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def handle_status(message):
    total_tickers = len(get_all_market_tickers())
    status_msg = (
        "⚙ **סטטוס מערכת:**\n\n"
        f"• AI Engine: {'✅ פעיל' if ai_client else '❌ חסר מפתח'}\n"
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
    send_alert(ticker=ticker, tech_data=tech_data, target_chat_id=message.chat.id)

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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
