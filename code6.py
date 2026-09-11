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

bot = TeleBot(TELEGRAM_BOT_TOKEN, threaded=True)
app = Flask(__name__)

USER_CALC_STATE = {}
DB_LOCK = threading.Lock()
SCAN_LOCK = threading.Lock()
TICKERS_CACHE: Dict[str, Any] = {"tickers": [], "fetched_at": 0.0}

SCAN_STATS = {"is_running": False, "last_run_start": None}

def safe_translate(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    try:
        translated = GoogleTranslator(source='auto', target='iw').translate(text)
        return translated if translated else text
    except Exception as e:
        logger.warning(f"Translation error for text '{text[:20]}...': {e}")
        return text

# ------------------------------------------------------------------------------
# 2. מסד נתונים SQLite
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
    try:
        with DB_LOCK:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO users (chat_id) VALUES (?)", (chat_id,))
                conn.commit()
    except Exception as e:
        logger.error(f"Error adding user {chat_id}: {e}")

def get_all_users() -> list:
    users = []
    try:
        with DB_LOCK:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chat_id FROM users")
                users = [r[0] for r in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching users: {e}")

    if DEFAULT_CHAT_ID:
        try:
            def_id = int(DEFAULT_CHAT_ID)
            if def_id not in users: 
                users.append(def_id)
        except ValueError: 
            pass
    return list(set(users))

def is_in_cooldown(symbol: str) -> bool:
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT alert_time FROM sent_signals WHERE symbol = ? ORDER BY alert_time DESC LIMIT 1", (symbol,))
            row = cursor.fetchone()
            if not row: 
                return False
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
# 3. רשימת נכסים
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
# 4. מנוע ניתוח חדשות
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
            res = requests.get(news_url, timeout=4)
            if res.status_code == 200 and isinstance(res.json(), list):
                for item in res.json():
                    if isinstance(item, dict) and item.get("headline"):
                        raw_articles.append(str(item.get("headline")))
        except Exception as e:
            logger.warning(f"Finnhub fetch error for {symbol}: {e}")

    if not raw_articles:
        try:
            news_items = yf.Ticker(symbol).news
            if news_items and isinstance(news_items, list):
                for item in news_items:
                    if isinstance(item, dict):
                        title = item.get("title")
                        if not title and "content" in item and isinstance(item["content"], dict):
                            title = item["content"].get("title")
                        if title:
                            raw_articles.append(str(title))
        except Exception as e:
            logger.warning(f"yFinance news error for {symbol}: {e}")

    matched_categories = set()
    meaningful_headlines = []

    for headline in raw_articles[:5]:
        headline_lower = headline.lower()
        for cat, keywords in HIGH_IMPACT_CATALYSTS.items():
            if any(kw in headline_lower for kw in keywords):
                matched_categories.add(cat)
                meaningful_headlines.append(headline)

    translated_headlines = []
    selected_headlines = meaningful_headlines[:3] if meaningful_headlines else raw_articles[:3]

    for h in selected_headlines:
        translated = safe_translate(h)
        if translated:
            translated_headlines.append(translated)

    return {
        "category": ", ".join(matched_categories) if matched_categories else "כללי / ללא זרז חריג",
        "headlines": translated_headlines
    }

# ------------------------------------------------------------------------------
# 5. מנוע ניתוח טכני
# ------------------------------------------------------------------------------
def analyze_stock_breakout(symbol: str, ignore_cooldown: bool = False) -> Optional[dict]:
    try:
        if not ignore_cooldown and is_in_cooldown(symbol):
            return None

        df = yf.Ticker(symbol).history(period="6m")
        if df.empty or len(df) < 50: 
            return {"error": f"לא נמצאו מספיק נתוני מסחר היסטוריים עבור הסימול {symbol}."}

        curr_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change_pct = ((curr_price - prev_price) / prev_price) * 100

        avg_vol_20 = df['Volume'].iloc[-21:-1].mean()
        curr_vol = df['Volume'].iloc[-1]
        vol_ratio = (curr_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0

        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        ema20 = float(df['EMA20'].iloc[-1]) if not pd.isna(df['EMA20'].iloc[-1]) else curr_price
        ema50 = float(df['EMA50'].iloc[-1]) if not pd.isna(df['EMA50'].iloc[-1]) else curr_price
        rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
        atr = float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else (curr_price * 0.02)

        high_20 = float(df['High'].iloc[-21:-1].max())
        high_50 = float(df['High'].iloc[-51:-1].max())

        is_breakout = curr_price >= high_20 or curr_price >= high_50
        is_uptrend = curr_price > ema20 > ema50
        is_high_volume = vol_ratio >= 1.2
        is_valid_price = curr_price >= CONFIG["MIN_PRICE"]

        has_passed_all = is_breakout and is_uptrend and is_high_volume and is_valid_price

        reasons = []
        if not is_breakout:
            reasons.append(f"❌ **אין פריצת שיא:** המחיר ({curr_price:.2f}) נמוך משיא 20 ימים ({high_20:.2f}) ושיא 50 ימים ({high_50:.2f}).")
        else:
            reasons.append(f"✅ **זוהתה פריצה:** המחיר פרץ שיא {'50' if curr_price >= high_50 else '20'} ימים.")

        if not is_uptrend:
            reasons.append(f"❌ **מגמה שאינה עולה:** לא מתקיים הכלל Price > EMA20 > EMA50 (מחיר: {curr_price:.2f}, EMA20: {ema20:.2f}, EMA50: {ema50:.2f}).")
        else:
            reasons.append(f"✅ **מגמה עולה:** המחיר נמצא מעל הממוצעים הנעים (EMA20 & EMA50).")

        if not is_high_volume:
            reasons.append(f"❌ **נפח מסחר חלש:** יחס נפח המסחר (RVOL) הינו {vol_ratio:.2f}x (נדרש לפחות 1.2x).")
        else:
            reasons.append(f"✅ **נפח מסחר חזק:** RVOL עומד על {vol_ratio:.2f}x.")

        if not is_valid_price:
            reasons.append(f"❌ **מחיר נמוך מדי:** {curr_price:.2f} (נדרש לפחות {CONFIG['MIN_PRICE']}$).")

        news_data = analyze_news_catalysts(symbol)

        stop_loss = round(curr_price - (1.5 * atr), 2)
        if stop_loss >= curr_price: 
            stop_loss = round(curr_price * 0.95, 2)
        risk = curr_price - stop_loss

        return {
            "symbol": symbol,
            "is_valid": has_passed_all,
            "reasons": reasons,
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
        return {"error": f"אירעה שגיאה בעיבוד הנתונים הטכניים עבור הסימול {symbol}."}

# ------------------------------------------------------------------------------
# 6. בניית הודעת דוח
# ------------------------------------------------------------------------------
def build_breakout_report(ticker_info: dict, data: dict) -> Tuple[str, InlineKeyboardMarkup]:
    symbol = ticker_info["symbol"]
    curr_symbol = "₪" if symbol.endswith(".TA") else "$"

    news_str = "\n".join([f"• {h}" for h in data["news"]["headlines"]]) if data["news"]["headlines"] else "• לא נמצאו חדשות חריגות כעת."

    if data.get("is_valid", False):
        msg = f"""
🌟 **התראת פריצה טכנית - {symbol}**
**מדד:** {ticker_info.get('index', 'כללי')}
