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

# ------------------------------------------------------------------------------
# 1. הגדרות לוגים, סביבה וקונפיגורציה
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger("trading_bot")

CONFIG = {
    "MIN_PRICE": 1.0,
    "ALERT_COOLDOWN_HOURS": 4,
    "MIN_RVOL": 1.2,
    "MIN_RSI": 40.0,
    "MAX_RSI": 75.0
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

SCAN_STATS = {"is_running": False, "last_run_start": None, "last_run_status": "טרם הופעל"}

# ------------------------------------------------------------------------------
# 2. מסד נתונים SQLite
# ------------------------------------------------------------------------------
DB_FILE = "bot_database.db"

def init_db():
    with DB_LOCK:
        with sqlite3.connect(DB_FILE, timeout=10) as conn:
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
            with sqlite3.connect(DB_FILE, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO users (chat_id) VALUES (?)", (chat_id,))
                conn.commit()
    except Exception as e:
        logger.error(f"Error adding user {chat_id}: {e}")

def get_all_users() -> list:
    users = []
    try:
        with DB_LOCK:
            with sqlite3.connect(DB_FILE, timeout=10) as conn:
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
        with sqlite3.connect(DB_FILE, timeout=10) as conn:
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
        with sqlite3.connect(DB_FILE, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO sent_signals (fingerprint, symbol, alert_time) VALUES (?, ?, ?)", (fp, symbol, now_str))
            conn.commit()

init_db()

# ------------------------------------------------------------------------------
# 3. רשימת נכסים מורחבת (S&P 500, NASDAQ 100, תל אביב 125)
# ------------------------------------------------------------------------------
TA_125_TICKERS = [
    "TEVA.TA", "ICL.TA", "NICE.TA", "LUMI.TA", "POLI.TA", "MZR.TA", "FIBI.TA",
    "DSCT.TA", "AZRG.TA", "ESLT.TA", "BEZQ.TA", "HARL.TA", "CLIS.TA", "MNTV.TA",
    "ENLT.TA", "ENOG.TA", "NSTR.TA", "PZOL.TA", "SAEN.TA", "DELT.TA", "DIMO.TA"
]

FALLBACK_US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "PG", "HD", "COST", "JNJ", "ABBV", "MRK", "NFLX",
    "AMD", "PEP", "KO", "BAC", "WMT", "CVX", "TSM", "CRM", "ADBE", "ORCL", "QCOM"
]

def fetch_all_index_tickers() -> List[dict]:
    now = time.time()
    if TICKERS_CACHE["tickers"] and (now - TICKERS_CACHE["fetched_at"] < 43200):
        return TICKERS_CACHE["tickers"]

    results = []
    seen = set()

    # 1. Tel Aviv 125
    for sym in TA_125_TICKERS:
        if sym not in seen:
            results.append({"symbol": sym, "index": "תל אביב 125", "currency": "ILS"})
            seen.add(sym)

    # 2. S&P 500
    try:
        url_sp = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp = pd.read_csv(url_sp, timeout=5)
        for _, row in df_sp.iterrows():
            sym = str(row['Symbol']).replace('.', '-').strip().upper()
            if sym not in seen:
                results.append({"symbol": sym, "index": "S&P 500", "currency": "USD"})
                seen.add(sym)
        logger.info(f"Loaded S&P 500 tickers successfully.")
    except Exception as e:
        logger.warning(f"Failed loading S&P 500 dynamically: {e}")

    # 3. NASDAQ 100
    try:
        url_nasdaq = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq100/nasdaq100.csv"
        df_nasdaq = pd.read_csv(url_nasdaq, timeout=5)
        for _, row in df_nasdaq.iterrows():
            sym = str(row['symbol']).replace('.', '-').strip().upper() if 'symbol' in row else str(row.iloc[0]).strip().upper()
            if sym not in seen:
                results.append({"symbol": sym, "index": "NASDAQ 100", "currency": "USD"})
                seen.add(sym)
        logger.info(f"Loaded NASDAQ 100 tickers successfully.")
    except Exception as e:
        logger.warning(f"Failed loading NASDAQ 100 dynamically: {e}")

    # Fallback במקרה של כישלון בטעינת ארה"ב
    if len(results) <= len(TA_125_TICKERS):
        logger.warning("Using Fallback US list.")
        for sym in FALLBACK_US_TICKERS:
            if sym not in seen:
                results.append({"symbol": sym, "index": "S&P 500 / NASDAQ 100", "currency": "USD"})
                seen.add(sym)

    TICKERS_CACHE["tickers"] = results
    TICKERS_CACHE["fetched_at"] = now
    logger.info(f"Total tickers loaded across indices: {len(results)}")
    return results

# ------------------------------------------------------------------------------
# 4. מנוע ניתוח חדשות ואימות זיקה מתוחכם
# ------------------------------------------------------------------------------
HIGH_IMPACT_CATALYSTS = {
    "Defense / Geopolitical": ["military", "defense", "pentagon", "contract", "war", "sanctions", "army", "weapon"],
    "FDA / Clinical Trials": ["fda", "approval", "phase 2", "phase 3", "clinical trial", "patent", "drug"],
    "Earnings / M&A / Financials": ["acquisition", "merger", "buyout", "earnings", "eps", "revenue", "guidance", "investment", "profit"]
}

def is_headline_relevant(headline: str, symbol: str, company_name: str = "") -> bool:
    clean_sym = symbol.replace(".TA", "").upper()
    h_upper = headline.upper()
    
    words = [w.strip(":,.-()[]\"'") for w in h_upper.split()]
    if clean_sym in words:
        return True
    
    if company_name and len(company_name) > 3:
        first_word = company_name.split()[0].upper()
        if len(first_word) > 2 and first_word in h_upper:
            return True
            
    return False

def analyze_news_catalysts(symbol: str) -> dict:
    raw_articles = []
    clean_symbol = symbol.replace(".TA", "")
    company_name = ""

    try:
        ticker_obj = yf.Ticker(symbol)
        company_name = ticker_obj.info.get("shortName", "") or ticker_obj.info.get("longName", "")
    except Exception:
        company_name = ""

    # 1. Finnhub API
    if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
        try:
            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=4)).strftime('%Y-%m-%d')
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={from_date}&to={today.strftime('%Y-%m-%d')}&token={FINNHUB_API_KEY}"
            res = requests.get(news_url, timeout=4)
            if res.status_code == 200 and isinstance(res.json(), list):
                for item in res.json():
                    if isinstance(item, dict) and item.get("headline"):
                        hl = str(item.get("headline")).strip()
                        if is_headline_relevant(hl, symbol, company_name):
                            raw_articles.append(hl)
        except Exception as e:
            logger.warning(f"Finnhub error for {symbol}: {e}")

    # 2. yFinance News Fallback
    if len(raw_articles) < 3:
        try:
            news_items = yf.Ticker(symbol).news
            if news_items and isinstance(news_items, list):
                for item in news_items:
                    if isinstance(item, dict):
                        title = item.get("title")
                        if not title and "content" in item and isinstance(item["content"], dict):
                            title = item["content"].get("title")
                        if title:
                            title_str = str(title).strip()
                            if is_headline_relevant(title_str, symbol, company_name) and title_str not in raw_articles:
                                raw_articles.append(title_str)
        except Exception as e:
            logger.warning(f"yFinance news error for {symbol}: {e}")

    matched_categories = set()
    meaningful_headlines = []

    for headline in raw_articles[:7]:
        headline_lower = headline.lower()
        for cat, keywords in HIGH_IMPACT_CATALYSTS.items():
            if any(kw in headline_lower for kw in keywords):
                matched_categories.add(cat)
                if headline not in meaningful_headlines:
                    meaningful_headlines.append(headline)

    selected_headlines = meaningful_headlines[:3] if meaningful_headlines else raw_articles[:3]

    return {
        "category": ", ".join(matched_categories) if matched_categories else "General / No major catalyst detected",
        "headlines": selected_headlines
    }

# ------------------------------------------------------------------------------
# 5. מנוע ניתוח טכני משולב ומדויק
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

        # סינון ימים ללא מסחר בפועל (Volume = 0)
        valid_vol_df = df[df['Volume'] > 0]
        if valid_vol_df.empty:
            return {"error": "נפח מסחר אפסי לאורך כל התקופה שנבדקה."}

        avg_vol_20 = float(valid_vol_df['Volume'].iloc[-21:-1].mean())
        curr_vol = float(valid_vol_df['Volume'].iloc[-1])
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
        is_high_volume = vol_ratio >= CONFIG["MIN_RVOL"]
        is_valid_rsi = CONFIG["MIN_RSI"] <= rsi <= CONFIG["MAX_RSI"]
        is_valid_price = curr_price >= CONFIG["MIN_PRICE"]

        has_passed_all = is_breakout and is_uptrend and is_high_volume and is_valid_rsi and is_valid_price

        reasons = []
        if is_breakout:
            reasons.append(f"✅ **זוהתה פריצה:** המחיר ({curr_price:.2f}) פרץ שיא {'50' if curr_price >= high_50 else '20'} ימים.")
        else:
            reasons.append(f"❌ **אין פריצת שיא:** המחיר ({curr_price:.2f}) נמוך משיא 20 ימים ({high_20:.2f}) ושיא 50 ימים ({high_50:.2f}).")

        if is_uptrend:
            reasons.append(f"✅ **מגמה עולה:** Price ({curr_price:.2f}) > EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}).")
        else:
            reasons.append(f"❌ **מגמה שאינה עולה:** לא מתקיים Price > EMA20 > EMA50 (EMA20: {ema20:.2f}, EMA50: {ema50:.2f}).")

        if is_high_volume:
            reasons.append(f"✅ **נפח מסחר חזק:** RVOL עומד על {vol_ratio:.2f}x (סף מינימלי: {CONFIG['MIN_RVOL']}x).")
        else:
            reasons.append(f"❌ **נפח מסחר חלש:** RVOL עומד על {vol_ratio:.2f}x (נדרש לפחות {CONFIG['MIN_RVOL']}x).")

        if is_valid_rsi:
            reasons.append(f"✅ **מדד RSI תקין:** עומד על {rsi:.1f} (בטווח המותר {CONFIG['MIN_RSI']}-{CONFIG['MAX_RSI']}).")
        else:
            reasons.append(f"❌ **מדד RSI חורג:** עומד על {rsi:.1f} (נדרש בין {CONFIG['MIN_RSI']} ל-{CONFIG['MAX_RSI']}).")

        if is_valid_price:
            reasons.append(f"✅ **מחיר תקין:** {curr_price:.2f} (מעל סף {CONFIG['MIN_PRICE']}$).")
        else:
            reasons.append(f"❌ **מחיר נמוך מדי:** {curr_price:.2f} (סף מינימום: {CONFIG['MIN_PRICE']}$).")

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
            "high_20": round(high_20, 2),
            "high_50": round(high_50, 2),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "rsi": round(rsi, 1),
            "vol_ratio": round(vol_ratio, 2),
            "high_type": "50 ימים 🚀" if curr_price >= high_50 else ("20 ימים 📈" if curr_price >= high_20 else "ללא פריצה"),
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
# 6. בניית הודעת דוח מפורטת + קישור תקין ל-TradingView
# ------------------------------------------------------------------------------
def build_breakout_report(ticker_info: dict, data: dict) -> Tuple[str, InlineKeyboardMarkup]:
    symbol = ticker_info["symbol"]
    curr_symbol = "₪" if symbol.endswith(".TA") else "$"

    headlines = data["news"]["headlines"]
    news_str = "\n".join(["• " + h for h in headlines]) if headlines else "• No recent ticker-specific news found."

    chg_sign = "+" if data["change_pct"] > 0 else ""
    
    # בניית קישור מדויק ל-TradingView (כולל TASE מותאם)
    tv_symbol = symbol.replace(".TA", "") if not symbol.endswith(".TA") else f"TASE:{symbol.replace('.TA', '')}"
    chart_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

    if data.get("is_valid", False):
        lines = [
            f"🌟 **התראת פריצה טכנית - {symbol}**",
            f"**מדד:** {ticker_info.get('index', 'כללי')}",
            "",
            "---",
            "📈 **נתונים טכניים:**",
            f"• סוג פריצה: **פריצת שיא {data['high_type']}**",
            f"• מחיר נוכחי: `{curr_symbol}{data['price']}` ({chg_sign}{data['change_pct']}%)",
            f"• נפח מסחר יחסי (RVOL): **{data['vol_ratio']}x**",
            f"• מדד חוזק (RSI): `{data['rsi']}`",
            f"• ממוצעים נעים: `EMA20: {data['ema20']}` | `EMA50: {data['ema50']}`",
            "",
            "---",
            "📰 **חדשות וקטליזטורים (באנגלית מקורית):**",
            f"• סיווג: **{data['news']['category']}**",
            news_str,
            "",
            "---",
            "🎯 **תוכנית מסחר מוצעת:**",
            f"• 🎯 מחיר כניסה: `{curr_symbol}{data['entry']}`",
            f"• 🛑 סטופ לוס (1.5xATR): `{curr_symbol}{data['stop_loss']}`",
            f"• 🚀 יעד 1 (TP1 1.5R): `{curr_symbol}{data['tp1']}`",
            f"• 🚀 יעד 2 (TP2 2.5R): `{curr_symbol}{data['tp2']}`"
        ]
        msg = "\n".join(lines)
        markup = InlineKeyboardMarkup(row_width=2)
        btn_chart = InlineKeyboardButton("📈 צפייה בגרף חי (TradingView)", url=chart_url)
        btn_calc = InlineKeyboardButton("💰 חישוב עסקה", callback_data=f"calc_{symbol}_{data['entry']}_{data['stop_loss']}")
        markup.add(btn_chart, btn_calc)
    else:
        reasons_text = "\n".join(data["reasons"])
        lines = [
            f"🔎 **ניתוח טכני מלא עבור {symbol}**",
            f"**מדד:** {ticker_info.get('index', 'כללי')}",
            "",
            "📋 **סטטוס התאמה לפריצה:**",
            reasons_text,
            "",
            "---",
            "📊 **פירוט אינדיקטורים מלא:**",
            f"• מחיר נוכחי: `{curr_symbol}{data['price']}` ({chg_sign}{data['change_pct']}%)",
            f"• שיא 20 ימים: `{curr_symbol}{data['high_20']}` | שיא 50 ימים: `{curr_symbol}{data['high_50']}`",
            f"• ממוצעים נעים: `EMA20: {data['ema20']}` | `EMA50: {data['ema50']}`",
            f"• נפח מסחר יחסי (RVOL): **{data['vol_ratio']}x**",
            f"• מדד חוזק יחסי (RSI): `{data['rsi']}`",
            "",
            "---",
            "🎯 **רמות מפתח מחושבות (למקרה של כניסה):**",
            f"• 🛑 סטופ לוס מוצע: `{curr_symbol}{data['stop_loss']}`",
            f"• 🚀 יעד רווח TP1: `{curr_symbol}{data['tp1']}`",
            "",
            "---",
            "📰 **חדשות וקטליזטורים (באנגלית מקורית):**",
            news_str
        ]
        msg = "\n".join(lines)
        markup = InlineKeyboardMarkup()
        btn_chart = InlineKeyboardButton("📈 צפייה בגרף חי (TradingView)", url=chart_url)
        markup.add(btn_chart)

    return msg, markup

# ------------------------------------------------------------------------------
# 7. מנוע סריקה אוטומטית ברקע
# ------------------------------------------------------------------------------
def run_scan_process(target_chat_id: Optional[int] = None):
    with SCAN_LOCK:
        if SCAN_STATS["is_running"]: 
            if target_chat_id:
                bot.send_message(target_chat_id, "⏳ סריקה כבר מורצת ברקע, אנא המתיני לסיומה.")
            return
        SCAN_STATS["is_running"] = True
        SCAN_STATS["last_run_start"] = datetime.datetime.now()
        SCAN_STATS["last_run_status"] = "סורק כעת..."

    logger.info("🚀 מתחיל סריקת מניות ברקע (S&P 500, NASDAQ 100, ת"א 125)...")
    found_count = 0
    
    try:
        tickers = fetch_all_index_tickers()
        
        def check_and_send(item):
            nonlocal found_count
            res = analyze_stock_breakout(item["symbol"], ignore_cooldown=(target_chat_id is not None))
            if res and res.get("is_valid", False):
                found_count += 1
                msg, markup = build_breakout_report(item, res)
                
                recipients = [target_chat_id] if target_chat_id else get_all_users()
                for cid in recipients:
                    try:
                        bot.send_message(cid, msg, parse_mode="Markdown", reply_markup=markup)
                    except Exception as e:
                        logger.error(f"Error sending alert to {cid}: {e}")
                
                if not target_chat_id:
                    record_signal(item["symbol"])

        with ThreadPoolExecutor(max_workers=6) as executor:
            executor.map(check_and_send, tickers)

        status_msg = f"סריקה הושלמה: אותרו {found_count} מניות פוטנציאליות."
        SCAN_STATS["last_run_status"] = status_msg
        logger.info(f"✅ {status_msg}")

        if target_chat_id and found_count == 0:
            bot.send_message(target_chat_id, "🔍 הסריקה הידנית הושלמה. לא נמצאו מניות שענו על כל תנאי הפריצה ברגע זה.")

    except Exception as e:
        logger.error(f"Error during scan: {e}")
        SCAN_STATS["last_run_status"] = f"שגיאה בסריקה: {e}"
        if target_chat_id:
            bot.send_message(target_chat_id, "❌ אירעה שגיאה במהלך ביצוע הסריקה.")
    finally:
        with SCAN_LOCK:
            SCAN_STATS["is_running"] = False

# הגדרת Scheduler לסריקה מחזורית אוטומטית בכל 15 דקות
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(lambda: run_scan_process(), 'interval', minutes=15)
scheduler.start()

# ------------------------------------------------------------------------------
# 8. שרת Web (Flask Keep-Alive)
# ------------------------------------------------------------------------------
@app.route('/')
def home():
    return "Bot is running 24/7 with automatic scanning background tasks", 200

def keep_alive_ping():
    while True:
        time.sleep(600)
        try:
            if "localhost" not in SELF_URL: 
                requests.get(SELF_URL, timeout=10)
        except Exception: 
            pass

# ------------------------------------------------------------------------------
# 9. פקודות טלגרם
# ------------------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def handle_start(message):
    add_user(message.chat.id)
    welcome_text = (
        "👋 **ברוכים הבאים לבוט סריקת המניות והפריצות הטכניות!**\n\n"
        "הבוט סורק באופן רציף וברקע את מדדי **S&P 500, NASDAQ 100 ו-תל אביב 125** "
        "כדי לאתר מניות המציגות פריצה טכנית, נפח מסחר חורג (RVOL) וקטליזטורים חדשותיים.\n\n"
        "🛠️ **רשימת הפקודות הזמינות בבוט:**\n\n"
        "• /start - הצגת הודעת פתיחה זו והסבר על הפיצ'רים\n"
        "• /scan - הפעלת סריקה ידנית מיידית ברקע על כל המדדים\n"
        "• /tech <SYMBOL> - ניתוח טכני מלא וצפייה בגרף (לדוגמה: `/tech AAPL` או `/tech TEVA.TA`)\n"
        "• /news <SYMBOL> - סריקת חדשות וכותרות באנגלית מקורית (לדוגמה: `/news TSLA`)\n"
        "• /status - בדיקת סטטוס סורק הרקע האוטומטי"
    )
    bot.send_message(message.chat.id, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['scan'])
def handle_scan(message):
    add_user(message.chat.id)
    bot.reply_to(message, "🔍 **סריקה ידנית הופעלה ברקע!**\nהמערכת מעבדת כעת את כל המדדים ותשלח התראות במידה ותזהה הזדמנויות...", parse_mode="Markdown")
    threading.Thread(target=run_scan_process, args=(message.chat.id,), daemon=True).start()

@bot.message_handler(commands=['tech'])
def handle_tech(message):
    add_user(message.chat.id)
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "⚠️ נא להזין סימול מניה. לדוגמה: `/tech AAPL` או `/tech TEVA.TA`", parse_mode="Markdown")
        return

    symbol = parts[1].upper().strip()
    bot.reply_to(message, f"🔍 מריץ ניתוח טכני מלא עבור **{symbol}**...", parse_mode="Markdown")

    def run_single():
        res = analyze_stock_breakout(symbol, ignore_cooldown=True)
        if not res or "error" in res:
            err_msg = res.get('error', 'לא התקבלו נתונים') if res else 'לא התקבלו נתונים'
            bot.send_message(message.chat.id, f"❌ שגיאה בניתוח **{symbol}**: {err_msg}", parse_mode="Markdown")
            return

        item = {
            "symbol": symbol,
            "index": "תל אביב 125" if symbol.endswith(".TA") else "S&P 500 / NASDAQ 100",
            "currency": "ILS" if symbol.endswith(".TA") else "USD"
        }
        msg, markup = build_breakout_report(item, res)
        bot.send_message(message.chat.id, msg, parse_mode="Markdown", reply_markup=markup)

    threading.Thread(target=run_single, daemon=True).start()

@bot.message_handler(commands=['news'])
def handle_news(message):
    add_user(message.chat.id)
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "⚠️ נא להזין סימול מניה לסריקת חדשות. לדוגמה: `/news AAPL`", parse_mode="Markdown")
        return

    symbol = parts[1].upper().strip()
    bot.reply_to(message, f"📰 סורק חדשות וזרזים רלוונטיים עבור **{symbol}**...", parse_mode="Markdown")

    def run_news_single():
        try:
            news_data = analyze_news_catalysts(symbol)
            headlines = news_data["headlines"]
            news_str = "\n".join(["• " + h for h in headlines]) if headlines else "• No recent news found for this ticker."

            tv_symbol = symbol.replace(".TA", "") if not symbol.endswith(".TA") else f"TASE:{symbol.replace('.TA', '')}"
            chart_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

            lines = [
                f"📰 **סריקת חדשות עבור {symbol}**",
                "",
                f"• **סיווג זרז:** {news_data['category']}",
                "",
                "**כותרות אחרונות (באנגלית מקורית):**",
                news_str
            ]
            msg = "\n".join(lines)
            markup = InlineKeyboardMarkup()
            btn_chart = InlineKeyboardButton("📈 צפייה בגרף ב-TradingView", url=chart_url)
            markup.add(btn_chart)

            bot.send_message(message.chat.id, msg, parse_mode="Markdown", reply_markup=markup)
        except Exception as e:
            logger.error(f"Error handling /news command for {symbol}: {e}")
            bot.send_message(message.chat.id, f"❌ אירעה שגיאה בעת שליפת החדשות עבור **{symbol}**.", parse_mode="Markdown")

    threading.Thread(target=run_news_single, daemon=True).start()

@bot.message_handler(commands=['status'])
def handle_status(message):
    add_user(message.chat.id)
    with SCAN_LOCK:
        is_run = SCAN_STATS["is_running"]
        last = SCAN_STATS["last_run_start"]
        status_text = SCAN_STATS["last_run_status"]
    last_str = last.strftime('%Y-%m-%d %H:%M:%S') if last else "טרם בוצעה"
    
    msg = (
        f"🩺 **סטטוס מערכת וסורק אוטומטי:**\n"
        f"• סורק מורץ ברגע זה: {'כן ⏳' if is_run else 'לא (ממתין למחזור הבא) 🟢'}\n"
        f"• זמן ריצה אחרון: `{last_str}`\n"
        f"• סטטוס ריצה: {status_text}\n"
        f"• תדר סריקה אוטומטית: כל 15 דקות"
    )
    bot.reply_to(message, msg, parse_mode="Markdown")

# ------------------------------------------------------------------------------
# 10. אירועי אינטראקציה ומחשבון ניהול סיכונים בטוח
# ------------------------------------------------------------------------------
@bot.callback_query_handler(func=lambda call: call.data.startswith("calc_"))
def handle_calc_callback(call):
    try:
        _, symbol, entry, sl = call.data.split("_")
        USER_CALC_STATE[call.message.chat.id] = {"symbol": symbol, "entry": float(entry), "sl": float(sl)}
        curr_symbol = "₪" if symbol.endswith(".TA") else "$"
        
        bot.send_message(
            call.message.chat.id,
            f"💰 **מחשבון סימולציית עסקה עבור {symbol}**\n\n"
            f"מחיר כניסה: `{curr_symbol}{entry}` | סטופ לוס: `{curr_symbol}{sl}`\n\n"
            f"אנא הזיני את תקציב ההשקעה בדולרים/שקלים (לדוגמה: `2500`):",
            parse_mode="Markdown"
        )
        bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Callback error: {e}")

@bot.message_handler(func=lambda msg: not msg.text.startswith("/") and msg.chat.id in USER_CALC_STATE and USER_CALC_STATE[msg.chat.id] is not None)
def handle_calc_input(message):
    state = USER_CALC_STATE.get(message.chat.id)
    if not state:
        return

    try:
        budget = float(message.text.replace("$", "").replace("₪", "").strip())
        entry, sl = state["entry"], state["sl"]

        shares = int(budget // entry)
        if shares == 0:
            bot.reply_to(message, "⚠️ התקציב שהוזן נמוך ממחיר מניה אחת.")
            return

        # ניקוי בטוח של ה-State
        USER_CALC_STATE.pop(message.chat.id, None)

        total_inv = shares * entry
        risk_per_share = entry - sl
        total_risk = shares * risk_per_share
        
        tp1 = entry + (1.5 * risk_per_share)
        tp2 = entry + (2.5 * risk_per_share)

        curr_symbol = "₪" if state['symbol'].endswith(".TA") else "$"

        lines = [
            f"📐 **תוצאות סימולציית קנייה עבור {state['symbol']}**",
            "",
            f"• **כמות מניות לקנייה:** `{shares}` מניות",
            f"• **סך השקעה בפועל:** {curr_symbol}{total_inv:,.2f}",
            f"• **סיכון כולל בסטופ לוס:** -{curr_symbol}{total_risk:,.2f}",
            "",
            "---",
            "**🎯 צפי רווח ביעדים:**",
            f"• **יעד 1 ({curr_symbol}{tp1:.2f}):** רווח של **+{curr_symbol}{(shares * (tp1 - entry)):,.2f}**",
            f"• **יעד 2 ({curr_symbol}{tp2:.2f}):** רווח של **+{curr_symbol}{(shares * (tp2 - entry)):,.2f}**"
        ]
        reply = "\n".join(lines)
        bot.send_message(message.chat.id, reply, parse_mode="Markdown")
    except ValueError:
        bot.reply_to(message, "⚠️ נא להזין מספר בלבד (תקציב).")

# ------------------------------------------------------------------------------
# 11. הרצה ראשית וסריקה ראשונית
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()
    
    # הרצת סריקה ראשונית ברקע עם עלייתו של הבוט
    threading.Thread(target=run_scan_process, daemon=True).start()
    
    logger.info("🤖 Bot is active and running continuously...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
