import os
import sys
import time
import sqlite3
import logging
import threading
import requests
import datetime
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from flask import Flask
from telebot import TeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler
from deep_translator import GoogleTranslator

# ------------------------------------------------------------------------------
# 1. הגדרות לוגים, סביבה וקונפיגורציה
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger("trading_bot")

CONFIG = {
    "MIN_PRICE": 1.0,               # מחיר מינימלי למניה
    "MIN_DAILY_VALUE_USD": 500000,  # מחזור כספי יומי ממוצע מינימלי
    "ALERT_COOLDOWN_HOURS": 4
}

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
DEFAULT_CHAT_ID = os.environ.get("DEFAULT_CHAT_ID", None)
PORT = int(os.environ.get("PORT", 5000))
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")

bot = TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)
translator = GoogleTranslator(source='auto', target='iw')

USER_CALC_STATE = {}
DB_LOCK = threading.Lock()
SCAN_LOCK = threading.Lock()
TICKERS_CACHE: Dict[str, Any] = {"tickers": [], "fetched_at": 0.0}

SCAN_STATS = {"is_running": False, "last_run_start": None}

# ------------------------------------------------------------------------------
# 2. מסד נתונים SQLite (bot_database.db)
# ------------------------------------------------------------------------------
DB_FILE = "bot_database.db"

def init_db():
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS users (chat_id INTEGER PRIMARY KEY)")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sent_signals (
                    fingerprint TEXT PRIMARY KEY,
                    symbol TEXT,
                    alert_time TEXT
                )
            """)
            conn.commit()

def add_user(chat_id: int):
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO users (chat_id) VALUES (?)", (chat_id,))
            conn.commit()

def get_all_users() -> list:
    users = []
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chat_id FROM users")
            users = [r[0] for r in cursor.fetchall()]

    if DEFAULT_CHAT_ID:
        try:
            def_id = int(DEFAULT_CHAT_ID)
            if def_id not in users: users.append(def_id)
        except ValueError: pass
    return users

def is_in_cooldown(symbol: str) -> bool:
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT alert_time FROM sent_signals WHERE symbol = ? ORDER BY alert_time DESC LIMIT 1", (symbol,))
            row = cursor.fetchone()
            if not row: return False
            alert_time = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            return (datetime.datetime.now() - alert_time).total_seconds() < (CONFIG["ALERT_COOLDOWN_HOURS"] * 3600)

def record_signal(symbol: str):
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fp = f"{symbol}_{now_str}"
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO sent_signals (fingerprint, symbol, alert_time) VALUES (?, ?, ?)", (fp, symbol, now_str))
            conn.commit()

init_db()

# ------------------------------------------------------------------------------
# 3. רשימת נכסים (S&P 500, NASDAQ 100, TA-125)
# ------------------------------------------------------------------------------
TA_125_TICKERS = [
    "TEVA.TA", "ICL.TA", "NICE.TA", "LUMI.TA", "POLI.TA", "MZR.TA", "FIBI.TA",
    "DSCT.TA", "AZRG.TA", "ESLT.TA", "BEZQ.TA", "HARL.TA", "CLIS.TA", "MNTV.TA"
]

def fetch_all_index_tickers() -> List[dict]:
    now = time.time()
    if TICKERS_CACHE["tickers"] and (now - TICKERS_CACHE["fetched_at"] < 43200):
        return TICKERS_CACHE["tickers"]

    results = [{"symbol": sym, "index": "תל אביב 125", "currency": "ILS"} for sym in TA_125_TICKERS]

    try:
        url_sp = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp = pd.read_csv(url_sp)
        for _, row in df_sp.iterrows():
            sym = str(row['Symbol']).replace('.', '-').strip()
            results.append({"symbol": sym, "index": "S&P 500 / NASDAQ 100", "currency": "USD"})
    except Exception as e:
        logger.warning(f"Failed loading S&P500/NASDAQ: {e}")

    TICKERS_CACHE["tickers"] = results
    TICKERS_CACHE["fetched_at"] = now
    return results

# ------------------------------------------------------------------------------
# 4. מנוע ניתוח חדשות מתורגם (News & Catalyst Filter)
# ------------------------------------------------------------------------------
HIGH_IMPACT_CATALYSTS = {
    "ביטחוני / גיאופוליטי": ["military", "defense", "pentagon", "contract", "war", "sanctions", "army"],
    "אישור/ניסוי FDA": ["fda", "approval", "phase 2", "phase 3", "clinical trial", "patent"],
    "עסקאות / דוחות": ["acquisition", "merger", "buyout", "earnings", "eps", "revenue", "investment"]
}

def analyze_news_catalysts(symbol: str) -> dict:
    raw_articles = []
    clean_symbol = symbol.replace(".TA", "")

    if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
        try:
            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={from_date}&to={today.strftime('%Y-%m-%d')}&token={FINNHUB_API_KEY}"
            res = requests.get(news_url, timeout=3)
            if res.status_code == 200:
                for item in res.json():
                    if item.get("headline"): raw_articles.append(item.get("headline"))
        except Exception: pass

    if not raw_articles:
        try:
            news_items = yf.Ticker(symbol).news
            if news_items:
                for item in news_items:
                    title = item.get("title", "")
                    if title: raw_articles.append(title)
        except Exception: pass

    matched_categories = set()
    meaningful_headlines = []

    for headline in raw_articles[:4]:
        headline_lower = headline.lower()
        for cat, keywords in HIGH_IMPACT_CATALYSTS.items():
            if any(kw in headline_lower for kw in keywords):
                matched_categories.add(cat)
                meaningful_headlines.append(headline)

    translated_headlines = []
    for h in (meaningful_headlines[:2] if meaningful_headlines else raw_articles[:1]):
        try:
            trans = translator.translate(h)
            translated_headlines.append(trans)
        except Exception:
            translated_headlines.append(h)

    return {
        "category": ", ".join(matched_categories) if matched_categories else "כללי / ללא זרז חריג",
        "headlines": translated_headlines
    }

# ------------------------------------------------------------------------------
# 5. מנוע ניתוח טכני וסינון פריצות (Technical Analysis Engine)
# ------------------------------------------------------------------------------
def analyze_stock_breakout(symbol: str, ignore_cooldown: bool = False) -> Optional[dict]:
    try:
        if not ignore_cooldown and is_in_cooldown(symbol):
            return None

        df = yf.Ticker(symbol).history(period="6m")
        if df.empty or len(df) < 50: return None

        curr_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change_pct = ((curr_price - prev_price) / prev_price) * 100

        if curr_price < CONFIG["MIN_PRICE"]: return None

        avg_vol_20 = df['Volume'].iloc[-21:-1].mean()
        curr_vol = df['Volume'].iloc[-1]
        vol_ratio = (curr_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0

        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        ema20 = float(df['EMA20'].iloc[-1])
        ema50 = float(df['EMA50'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
        atr = float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else (curr_price * 0.02)

        high_20 = float(df['High'].iloc[-21:-1].max())
        high_50 = float(df['High'].iloc[-51:-1].max())

        # קריטריוני הליבה:
        # 1. פריצת שיא 20 או 50 ימים
        is_breakout = curr_price >= high_20 or curr_price >= high_50
        # 2. מגמה עולה: Price > EMA20 > EMA50
        is_uptrend = curr_price > ema20 > ema50
        # 3. RVOL >= 1.2
        is_high_volume = vol_ratio >= 1.2

        if ignore_cooldown:
            if not (is_breakout or (is_uptrend and is_high_volume)):
                return None
        else:
            if not (is_breakout and is_uptrend and is_high_volume):
                return None

        news_data = analyze_news_catalysts(symbol)

        stop_loss = round(curr_price - (1.5 * atr), 2)
        if stop_loss >= curr_price: stop_loss = round(curr_price * 0.95, 2)
        risk = curr_price - stop_loss

        return {
            "symbol": symbol,
            "price": round(curr_price, 2),
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi, 1),
            "vol_ratio": round(vol_ratio, 2),
            "high_type": "50 ימים 🚀" if curr_price >= high_50 else "20 ימים 📈",
            "news": news_data,
            "entry": round(curr_price, 2),
            "stop_loss": stop_loss,
            "tp1": round(curr_price + (1.5 * risk), 2),
            "tp2": round(curr_price + (2.5 * risk), 2)
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

# ------------------------------------------------------------------------------
# 6. בניית הודעת התראה
# ------------------------------------------------------------------------------
def build_breakout_report(ticker_info: dict, data: dict) -> Tuple[str, InlineKeyboardMarkup]:
    symbol = ticker_info["symbol"]
    curr_symbol = "₪" if symbol.endswith(".TA") else "$"

    news_str = "\n".join([f"• {h}" for h in data["news"]["headlines"]]) if data["news"]["headlines"] else "• לא זוהה זרז חדשותי חריג."

    msg = f"""
🌟 <b>התראת פריצה טכנית - {symbol}</b>
<b>מדד:</b> {ticker_info.get('index', 'כללי')}

---
📈 <b>נתונים טכניים:</b>
• סוג פריצה: <b>פריצת שיא {data['high_type']}</b>
• מחיר נוכחי: <code>{curr_symbol}{data['price']}</code> ({'+' if data['change_pct']>0 else ''}{data['change_pct']}%)
• נפח מסחר יחסי (RVOL): <b>{data['vol_ratio']}x</b>
• RSI: <code>{data['rsi']}</code>
• מגמה: <b>Price > EMA20 > EMA50 🟢</b>

---
📰 <b>חדשות וקטליזטורים (מתורגם):</b>
• סוג זרז: <b>{data['news']['category']}</b>
{news_str}

---
🎯 <b>תוכנית מסחר מוצעת:</b>
• 🎯 מחיר כניסה: <code>{curr_symbol}{data['entry']}</code>
• 🛑 סטופ לוס (1.5xATR): <code>{curr_symbol}{data['stop_loss']}</code>
• 🚀 יעד 1 (TP1 - 1.5R): <code>{curr_symbol}{data['tp1']}</code>
• 🚀 יעד 2 (TP2 - 2.5R): <code>{curr_symbol}{data['tp2']}</code>
"""

    markup = InlineKeyboardMarkup(row_width=2)
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף", url=f"https://www.tradingview.com/chart/?symbol={symbol}")
    btn_calc = InlineKeyboardButton("💰 חישוב עסקה", callback_data=f"calc_{symbol}_{data['entry']}_{data['stop_loss']}")
    markup.add(btn_chart, btn_calc)

    return msg, markup

# ------------------------------------------------------------------------------
# 7. מנוע סריקה
# ------------------------------------------------------------------------------
def run_scan_process(target_chat_id: Optional[int] = None):
    with SCAN_LOCK:
        if SCAN_STATS["is_running"]: 
            if target_chat_id:
                bot.send_message(target_chat_id, "⏳ סריקה כבר רצה ברקע, אנא המתיני לסיום.")
            return
        SCAN_STATS["is_running"] = True
        SCAN_STATS["last_run_start"] = datetime.datetime.now()

    logger.info("🚀 מתחיל סריקת מניות ברקע...")
    found_count = 0
    
    try:
        tickers = fetch_all_index_tickers()
        
        def check_and_send(item):
            nonlocal found_count
            res = analyze_stock_breakout(item["symbol"], ignore_cooldown=(target_chat_id is not None))
            if res:
                found_count += 1
                msg, markup = build_breakout_report(item, res)
                
                recipients = [target_chat_id] if target_chat_id else get_all_users()
                for cid in recipients:
                    try:
                        bot.send_message(cid, msg, parse_mode="HTML", reply_markup=markup)
                    except Exception as e:
                        logger.error(f"Error sending to {cid}: {e}")
                
                if not target_chat_id:
                    record_signal(item["symbol"])

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(check_and_send, tickers)

        if target_chat_id and found_count == 0:
            bot.send_message(target_chat_id, "🔍 הסריקה הידנית הושלמה. לא נמצאו מניות שענו באופן מלא על כל תנאי הפריצה כעת.")

    except Exception as e:
        logger.error(f"Error during scan: {e}")
        if target_chat_id:
            bot.send_message(target_chat_id, "❌ אירעה שגיאה במהלך הסריקה.")
    finally:
        with SCAN_LOCK:
            SCAN_STATS["is_running"] = False

# תזמון סריקה אוטומטית ברקע (APScheduler)
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(lambda: run_scan_process(), 'interval', minutes=15)
scheduler.start()

# ------------------------------------------------------------------------------
# 8. שרת Web (Flask Keep-Alive)
# ------------------------------------------------------------------------------
@app.route('/')
def home():
    return "Bot is running 24/7", 200

def keep_alive_ping():
    while True:
        time.sleep(600)
        try:
            if "localhost" not in SELF_URL: requests.get(SELF_URL, timeout=10)
        except Exception: pass

# ------------------------------------------------------------------------------
# 9. פקודות טלגרם ואינטראקטיביות (סדר Handlers מוקפד לבקשתך!)
# ------------------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def cmd_start(message):
    add_user(message.chat.id)
    welcome_text = (
        "👋 <b>ברוכים הבאים לבוט סורק המניות האוטומטי!</b>\n\n"
        "הבוט סורק בזמן אמת את המדדים <b>S&P 500, NASDAQ 100 ו-תל אביב 125</b> "
        "ומאתר פריצות טכניות (Price > EMA20 > EMA50, RVOL > 1.2x ושיאי 20/50 ימים).\n\n"
        "📋 <b>פקודות זמינות:</b>\n"
        "• /scan - הפעלת סריקה ידנית מיידית ברקע (Non-blocking)\n"
        "• /tech <SYMBOL> - ניתוח טכני וחדשותי ממוקד למניה (לדוגמה: <code>/tech AAPL</code> או <code>/tech TEVA.TA</code>)\n"
        "• /status - בדיקת סטטוס סורק הרקע"
    )
    bot.reply_to(message, welcome_text, parse_mode="HTML")

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    add_user(message.chat.id)
    bot.reply_to(message, "🔍 <b>סריקה ידנית הופעלה ברקע!</b>\nהמערכת סורקת כעת את הנכסים ותשלח התראות במידה ותזהה הזדמנויות...", parse_mode="HTML")
    threading.Thread(target=run_scan_process, args=(message.chat.id,), daemon=True).start()

@bot.message_handler(commands=['tech'])
def cmd_tech(message):
    add_user(message.chat.id)
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "⚠️ נא להזין סימול מניה. לדוגמה: <code>/tech AAPL</code>", parse_mode="HTML")
        return

    symbol = parts[1].upper().strip()
    bot.reply_to(message, f"🔍 מריץ ניתוח עבור <b>{symbol}</b>...", parse_mode="HTML")

    def run_single():
        res = analyze_stock_breakout(symbol, ignore_cooldown=True)
        if not res:
            bot.send_message(message.chat.id, f"❌ לא זוהתה פריצה טכנית מובהקת עבור <b>{symbol}</b> ברגע זה.", parse_mode="HTML")
            return

        item = {
            "symbol": symbol,
            "index": "תל אביב 125" if symbol.endswith(".TA") else "S&P 500 / NASDAQ 100",
            "currency": "ILS" if symbol.endswith(".TA") else "USD"
        }
        msg, markup = build_breakout_report(item, res)
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)

    threading.Thread(target=run_single, daemon=True).start()

@bot.message_handler(commands=['status'])
def cmd_status(message):
    add_user(message.chat.id)
    with SCAN_LOCK:
        is_run = SCAN_STATS["is_running"]
        last = SCAN_STATS["last_run_start"]
    last_str = last.strftime('%Y-%m-%d %H:%M:%S') if last else "טרם בוצעה"
    bot.reply_to(message, f"🩺 <b>סטטוס מערכת:</b>\n• סורק פעיל ברגע זה: {'כן ⏳' if is_run else 'לא 🟢'}\n• זמן ריצה אחרון: {last_str}", parse_mode="HTML")

# ------------------------------------------------------------------------------
# לוכדי אירועים פנימיים (Callback & Text Inputs)
# ------------------------------------------------------------------------------
@bot.callback_query_handler(func=lambda call: call.data.startswith("calc_"))
def handle_calc_callback(call):
    try:
        _, symbol, entry, sl = call.data.split("_")
        USER_CALC_STATE[call.message.chat.id] = {"symbol": symbol, "entry": float(entry), "sl": float(sl)}
        curr_symbol = "₪" if symbol.endswith(".TA") else "$"
        
        bot.send_message(
            call.message.chat.id,
            f"💰 <b>מחשבון סימולציית עסקה עבור {symbol}</b>\n\n"
            f"מחיר כניסה: <code>{curr_symbol}{entry}</code> | סטופ לוס: <code>{curr_symbol}{sl}</code>\n\n"
            f"אנא הזיני את תקציב ההשקעה בדולרים/שקלים (לדוגמה: <code>2500</code>):",
            parse_mode="HTML"
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Callback error: {e}")

@bot.message_handler(func=lambda msg: msg.chat.id in USER_CALC_STATE and USER_CALC_STATE[msg.chat.id] is not None and not msg.text.startswith("/"))
def handle_calc_input(message):
    try:
        state = USER_CALC_STATE.pop(message.chat.id)
        budget = float(message.text.replace("$", "").replace("₪", "").strip())
        entry, sl = state["entry"], state["sl"]

        shares = int(budget // entry)
        if shares == 0:
            bot.reply_to(message, "⚠️ התקציב שהוזן נמוך ממחיר מניה אחת.")
            return

        total_inv = shares * entry
        risk_per_share = entry - sl
        total_risk = shares * risk_per_share
        
        tp1 = entry + (1.5 * risk_per_share)
        tp2 = entry + (2.5 * risk_per_share)

        curr_symbol = "₪" if state['symbol'].endswith(".TA") else "$"

        reply = f"""
📐 <b>תוצאות סימולציית קנייה עבור {state['symbol']}</b>

• <b>כמות מניות לקנייה:</b> <code>{shares}</code> מניות
• <b>סך השקעה בפועל:</b> {curr_symbol}{total_inv:,.2f}
• <b>סיכון כולל בסטופ לוס:</b> -{curr_symbol}{total_risk:,.2f}

---
<b>🎯 צפי רווח ביעדים:</b>
• <b>יעד 1 ({curr_symbol}{tp1:.2f}):</b> רווח של <b>+{curr_symbol}{(shares * (tp1 - entry)):,.2f}</b>
• <b>יעד 2 ({curr_symbol}{tp2:.2f}):</b> רווח של <b>+{curr_symbol}{(shares * (tp2 - entry)):,.2f}</b>
"""
        bot.send_message(message.chat.id, reply, parse_mode="HTML")
    except ValueError:
        bot.reply_to(message, "⚠️ נא להזין מספר בלבד.")

# ------------------------------------------------------------------------------
# 10. הרצה ראשית
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()
    logger.info("🤖 Stock Breakout Bot is active and running...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
