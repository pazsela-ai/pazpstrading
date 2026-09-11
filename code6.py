import os
import sys
import time
import sqlite3
import logging
import threading
import requests
import datetime
import re
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from flask import Flask
from telebot import TeleBot, types
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler
from deep_translator import GoogleTranslator

# ------------------------------------------------------------------------------
# 1. הגדרות לוגים, סביבה ומשתנים גלובליים
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("trading_bot")

CONFIG = {
    "MIN_PRICE": 2.0,
    "MIN_AVG_VOLUME": 100000,
    "MIN_RVOL": 1.2,
    "LIMITS": {
        "ALERT_COOLDOWN_HOURS": 4
    }
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
TICKERS_CACHE: Dict[str, Any] = {"tickers": [], "fetched_at": 0.0}
TICKERS_CACHE_TTL = 12 * 3600

SCAN_STATS = {
    "is_running": False,
    "last_run_start": None,
    "last_alerts_count": 0
}
SCAN_LOCK = threading.Lock()

# ------------------------------------------------------------------------------
# 2. מסד נתונים (SQLite)
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

def remove_user(chat_id: int):
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE chat_id = ?", (chat_id,))
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
            return (datetime.datetime.now() - alert_time).total_seconds() < (CONFIG["LIMITS"]["ALERT_COOLDOWN_HOURS"] * 3600)

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
# 3. טעינת נכסים: S&P 500, NASDAQ 100, תל אביב 125
# ------------------------------------------------------------------------------
TA_125_TICKERS = [
    "TEVA.TA", "ICL.TA", "NICE.TA", "LUMI.TA", "POLI.TA", "MZR.TA", "FIBI.TA",
    "DSCT.TA", "AZRG.TA", "ESLT.TA", "BEZQ.TA", "HARL.TA", "CLIS.TA", "MNTV.TA",
    "ENLT.TA", "ENOG.TA", "SAEN.TA", "DSRG.TA", "DELT.TA", "BIG.TA", "ORL.TA"
]

def fetch_all_index_tickers() -> List[dict]:
    now = time.time()
    if TICKERS_CACHE["tickers"] and (now - TICKERS_CACHE["fetched_at"] < TICKERS_CACHE_TTL):
        return TICKERS_CACHE["tickers"]

    results = []

    # 1. תל אביב 125
    for sym in TA_125_TICKERS:
        results.append({"symbol": sym, "index": "תל אביב 125", "currency": "ILS"})

    # 2. S&P 500
    try:
        url_sp = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp = pd.read_csv(url_sp)
        for _, row in df_sp.iterrows():
            sym = str(row['Symbol']).replace('.', '-').strip()
            results.append({"symbol": sym, "index": "S&P 500", "currency": "USD"})
    except Exception as e:
        logger.warning(f"Failed loading S&P500: {e}")

    # 3. NASDAQ 100 Top
    nasdaq_list = ["AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", "META", "GOOGL", "NFLX", "AVGO", "COST", "PLTR", "QCOM", "INTC", "AMAT"]
    for sym in nasdaq_list:
        if not any(r["symbol"] == sym for r in results):
            results.append({"symbol": sym, "index": "NASDAQ 100", "currency": "USD"})

    TICKERS_CACHE["tickers"] = results
    TICKERS_CACHE["fetched_at"] = now
    return results

# ------------------------------------------------------------------------------
# 4. מנוע זיהוי תבניות מחיר ונרות יפניים (Patterns Engine)
# ------------------------------------------------------------------------------
def detect_candlestick_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    if len(df) < 5:
        return patterns

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    open_p, close_p, high_p, low_p = curr['Open'], curr['Close'], curr['High'], curr['Low']
    body = abs(close_p - open_p)
    candle_range = high_p - low_p
    
    # מניעת חלוקה באפס בנרות דוג'י
    safe_body = max(body, 0.0001)

    upper_shade = high_p - max(open_p, close_p)
    lower_shade = min(open_p, close_p) - low_p

    if candle_range > 0:
        # פטיש (Hammer)
        if lower_shade >= (2 * safe_body) and upper_shade <= (0.2 * safe_body):
            patterns.append("פטיש (Hammer) 🔨")
        # פטיש הפוך (Inverted Hammer)
        if upper_shade >= (2 * safe_body) and lower_shade <= (0.2 * safe_body):
            patterns.append("פטיש הפוך (Inverted Hammer) ⛏️")

    # בליעה שורית (Bullish Engulfing)
    prev_open, prev_close = prev['Open'], prev['Close']
    if prev_close < prev_open and close_p > open_p:
        if open_p <= prev_close and close_p >= prev_open:
            patterns.append("בליעה שורית (Bullish Engulfing) 🟢")

    # כוכב שחר (Morning Star)
    p2_open, p2_close = prev2['Open'], prev2['Close']
    p2_body = abs(p2_close - p2_open)
    prev_body = abs(prev_close - prev['Open'])
    
    # נר 1: אדום חזק, נר 2: גוף קטן/דוג'י בתחתית, נר 3: ירוק שנכנס לפחות לחצי מהנר הראשון
    if p2_close < p2_open and p2_body > 0 and (prev_body / p2_body < 0.4):
        if close_p > open_p and close_p >= (p2_close + (p2_body * 0.5)):
            patterns.append("כוכב שחר (Morning Star) 🌅")

    return patterns

def detect_chart_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    if len(df) < 60:
        return patterns

    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values

    # 1. ספל וידית (Cup and Handle)
    left_rim = max(highs[-60:-30])
    cup_bottom = min(lows[-45:-15])
    right_rim = max(highs[-20:-5])
    handle_low = min(lows[-10:])
    
    if (left_rim * 0.95 <= right_rim <= left_rim * 1.05) and (cup_bottom < left_rim * 0.85):
        if handle_low > cup_bottom and closes[-1] >= right_rim * 0.98:
            patterns.append("ספל וידית (Cup and Handle) ☕")

    # 2. משולש עולה (Ascending Triangle)
    recent_highs = highs[-30:]
    recent_lows = lows[-30:]
    resistance = np.max(recent_highs)
    
    lows_part1 = np.min(recent_lows[:15])
    lows_part2 = np.min(recent_lows[15:])
    if (lows_part2 > lows_part1 * 1.01) and (abs(np.max(recent_highs[:15]) - resistance) / resistance < 0.02):
        patterns.append("משולש עולה (Ascending Triangle) 📐")

    # 3. דגל שורי (Bull Flag)
    pole_move = (closes[-20] - closes[-40]) / closes[-40]
    if pole_move > 0.08:
        flag_consolidation = (max(highs[-15:]) - min(lows[-15:])) / closes[-1]
        if flag_consolidation < 0.05 and closes[-1] > max(highs[-5:]):
            patterns.append("דגל שורי (Bull Flag) 🚩")

    # 4. תחתית כפולה / משולשת (Double / Triple Bottom)
    l1 = min(lows[-60:-40])
    l2 = min(lows[-40:-20])
    l3 = min(lows[-20:])
    
    # תחתית כפולה
    if abs(l2 - l3) / l2 < 0.025 and closes[-1] > max(highs[-20:]) * 0.97:
        patterns.append("תחתית כפולה (Double Bottom) 👥")
    # תחתית משולשת
    elif abs(l1 - l2) / l1 < 0.025 and abs(l2 - l3) / l2 < 0.025 and closes[-1] > max(highs[-20:]) * 0.97:
        patterns.append("תחתית משולשת (Triple Bottom) 🔱")

    return patterns

# ------------------------------------------------------------------------------
# 5. מנוע ניתוח חדשותי
# ------------------------------------------------------------------------------
KEYWORD_CATEGORIES = {
    "ביטחוני/גיאופוליטי": ["war", "defense", "military", "geopolitical", "sanctions", "contract", "pentagon", "מלחמה", "ביטחוני", "עסקה ביטחונית"],
    "ניסויים רפואיים/FDA": ["fda", "phase 3", "phase 2", "clinical trial", "approval", "drug", "patent", "ניסוי קליני", "אישור fda"],
    "עסקאות כלכליות/מיזוגים": ["acquisition", "merger", "buyout", "partnership", "deal", "earnings beat", "guidance", "מיזוג", "רכישה", "דוחות"]
}

def analyze_news_catalysts(symbol: str) -> dict:
    raw_articles = []
    clean_symbol = symbol.replace(".TA", "")

    if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
        try:
            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=4)).strftime('%Y-%m-%d') # הרחבה ל-4 ימים למניעת פספוס בסופ"ש
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={from_date}&to={today.strftime('%Y-%m-%d')}&token={FINNHUB_API_KEY}"
            res = requests.get(news_url, timeout=4)
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
                    if not title and "content" in item and isinstance(item["content"], dict):
                        title = item["content"].get("title", "")
                    if title: raw_articles.append(title)
        except Exception: pass

    detected_categories = set()
    translated_headlines = []

    for headline in raw_articles[:3]:
        headline_lower = headline.lower()
        for cat, keywords in KEYWORD_CATEGORIES.items():
            if any(kw in headline_lower for kw in keywords):
                detected_categories.add(cat)
        try:
            translated_headlines.append(translator.translate(headline))
        except Exception:
            translated_headlines.append(headline)

    cat_str = ", ".join(detected_categories) if detected_categories else "חדשות שוטפות / כללי"
    return {
        "headlines": translated_headlines,
        "category": cat_str,
        "has_major_event": len(detected_categories) > 0
    }

# ------------------------------------------------------------------------------
# 6. מנוע ניתוח טכני משולב
# ------------------------------------------------------------------------------
def analyze_stock_technical(symbol: str) -> Optional[dict]:
    try:
        df = yf.Ticker(symbol).history(period="1y")
        if df.empty or len(df) < 50: return None

        # מניעת שגיאת מניות ישראליות ב-TA שנסחרות באגורות
        scale_factor = 100.0 if symbol.endswith(".TA") else 1.0

        curr_price = float(df['Close'].iloc[-1]) / scale_factor
        prev_price = float(df['Close'].iloc[-2]) / scale_factor
        change_pct = ((curr_price - prev_price) / prev_price) * 100

        # התאמת היסטוריית המחירים המנורמלת
        scaled_close = df['Close'] / scale_factor
        scaled_high = df['High'] / scale_factor
        scaled_low = df['Low'] / scale_factor

        df['RSI'] = ta.rsi(scaled_close, length=14)
        df['EMA20'] = ta.ema(scaled_close, length=20)
        df['EMA50'] = ta.ema(scaled_close, length=50)
        df['ATR'] = ta.atr(scaled_high, scaled_low, scaled_close, length=14)

        rsi = float(df['RSI'].dropna().iloc[-1]) if not df['RSI'].dropna().empty else 50.0
        ema20 = float(df['EMA20'].dropna().iloc[-1])
        ema50 = float(df['EMA50'].dropna().iloc[-1])
        atr = float(df['ATR'].dropna().iloc[-1]) if not df['ATR'].dropna().empty else (curr_price * 0.02)

        avg_vol = df['Volume'].iloc[-21:-1].mean()
        curr_vol = df['Volume'].iloc[-1]
        vol_ratio = (curr_vol / avg_vol) if avg_vol > 0 else 1.0

        high_20 = float(scaled_high.iloc[-21:-1].max())
        high_50 = float(scaled_high.iloc[-51:-1].max())

        is_uptrend = curr_price > ema20 > ema50
        is_breakout = (curr_price > high_20) and (vol_ratio >= CONFIG["MIN_RVOL"])
        is_major_breakout = curr_price > high_50

        chart_patterns = detect_chart_patterns(df)
        candle_patterns = detect_candlestick_patterns(df)
        all_detected_patterns = chart_patterns + candle_patterns

        if is_uptrend and (is_breakout or is_major_breakout or len(all_detected_patterns) > 0):
            stop_loss = round(curr_price - (1.5 * atr), 2)
            if stop_loss >= curr_price: stop_loss = round(curr_price * 0.95, 2)

            breakout_label = "50 ימים 🚀" if is_major_breakout else ("20 ימים 📈" if is_breakout else "מבוסס תבנית 🎯")

            return {
                "symbol": symbol,
                "price": round(curr_price, 2),
                "change_pct": round(change_pct, 2),
                "rsi": round(rsi, 1),
                "vol_ratio": round(vol_ratio, 2),
                "breakout_type": breakout_label,
                "patterns": all_detected_patterns,
                "entry": round(curr_price, 2),
                "stop_loss": stop_loss,
                "tp1": round(curr_price + (1.5 * (curr_price - stop_loss)), 2),
                "tp2": round(curr_price + (2.5 * (curr_price - stop_loss)), 2)
            }
        return None
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

# ------------------------------------------------------------------------------
# 7. יצירת הודעת איתות
# ------------------------------------------------------------------------------
def build_breakout_report(ticker_info: dict, tech: dict, news: dict) -> Tuple[str, InlineKeyboardMarkup]:
    symbol = ticker_info["symbol"]
    curr_symbol = "$" if ticker_info.get("currency") == "USD" else "₪"
    
    headlines_formatted = "\n".join([f"• {h}" for h in news["headlines"]]) if news["headlines"] else "• אין דיווחים יוצאי דופן ב-96 שעות האחרונות."
    patterns_formatted = "\n".join([f"  - {p}" for p in tech["patterns"]]) if tech["patterns"] else "  - לא זוהתה תבנית ספציפית"

    msg = f"""
🚨 <b>איתות פריצה ותבנית - {symbol}</b>
<b>מדד שיוך:</b> {ticker_info.get('index', 'כללי')}

---
<b>📈 ניתוח טכני ותבניות שנמצאו:</b>
• מחיר: <code>{curr_symbol}{tech['price']}</code> ({'+' if tech['change_pct']>0 else ''}{tech['change_pct']}%)
• סוג פריצה: <b>{tech['breakout_type']}</b>
• נפח מסחר יחסי (RVOL): <b>{tech['vol_ratio']}x</b> מהממוצע 🟢
• RSI: <code>{tech['rsi']}</code>
<b>תבניות מחיר ונרות יפניים:</b>
{patterns_formatted}

---
<b>📰 קטליזטור וחדשות מהותיות:</b>
• סיווג אירוע: <b>{news['category']}</b>
<b>כותרות אחרונות:</b>
{headlines_formatted}

---
<b>🎯 תוכנית מסחר מוצעת:</b>
• 🎯 מחיר כניסה: <code>{curr_symbol}{tech['entry']}</code>
• 🛑 סטופ לוס: <code>{curr_symbol}{tech['stop_loss']}</code>
• 🚀 יעד 1 (TP1): <code>{curr_symbol}{tech['tp1']}</code>
• 🚀 יעד 2 (TP2): <code>{curr_symbol}{tech['tp2']}</code>
"""

    markup = InlineKeyboardMarkup(row_width=2)
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף", url=f"https://www.tradingview.com/chart/?symbol={symbol}")
    btn_calc = InlineKeyboardButton("💰 חישוב עסקה", callback_data=f"calc_{symbol}_{tech['entry']}_{tech['stop_loss']}")
    markup.add(btn_chart, btn_calc)

    return msg, markup

# ------------------------------------------------------------------------------
# 8. מנוע הסריקה האוטומטי
# ------------------------------------------------------------------------------
def process_single_stock(item: dict):
    symbol = item["symbol"]
    if is_in_cooldown(symbol): return

    tech = analyze_stock_technical(symbol)
    if tech:
        news = analyze_news_catalysts(symbol)
        msg, markup = build_breakout_report(item, tech, news)
        record_signal(symbol)
        
        users = get_all_users()
        for chat_id in users:
            try:
                bot.send_message(chat_id, msg, parse_mode="HTML", reply_markup=markup)
            except Exception as e:
                logger.error(f"Failed sending alert to {chat_id}: {e}")
                # ניקוי משתמש שחסם את הבוט
                if "forbidden" in str(e).lower() or "blocked" in str(e).lower():
                    remove_user(chat_id)

def run_auto_scan_job():
    with SCAN_LOCK:
        if SCAN_STATS["is_running"]: return
        SCAN_STATS["is_running"] = True
        SCAN_STATS["last_run_start"] = datetime.datetime.now()

    logger.info("🚀 מפעיל סריקה אוטומטית ברקע עכשיו...")
    try:
        tickers = fetch_all_index_tickers()
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(process_single_stock, tickers)
    except Exception as e:
        logger.error(f"Error during auto scan: {e}")
    finally:
        with SCAN_LOCK:
            SCAN_STATS["is_running"] = False

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(run_auto_scan_job, 'interval', minutes=15)
scheduler.start()

# ------------------------------------------------------------------------------
# 9. טיפול במחשבון ופקודות טלגרם
# ------------------------------------------------------------------------------
@bot.callback_query_handler(func=lambda call: True)
def handle_callbacks(call):
    try:
        if call.data.startswith("calc_"):
            _, symbol, entry, sl = call.data.split("_")
            USER_CALC_STATE[call.message.chat.id] = {"symbol": symbol, "entry": float(entry), "sl": float(sl)}
            
            curr_symbol = "₪" if symbol.endswith(".TA") else "$"
            
            bot.send_message(
                call.message.chat.id,
                f"💰 <b>מחשבון סימולציית עסקה עבור {symbol}</b>\n\n"
                f"מחיר כניסה: <code>{curr_symbol}{entry}</code> | סטופ לוס: <code>{curr_symbol}{sl}</code>\n\n"
                f"הכנס את תקציב ההשקעה בדולרים/שקלים (לדוגמה: <code>2000</code>):",
                parse_mode="HTML"
            )
            bot.answer_callback_query(call.id)
    except Exception as e:
        logger.error(f"Callback error: {e}")

@bot.message_handler(func=lambda msg: msg.chat.id in USER_CALC_STATE and USER_CALC_STATE[msg.chat.id] is not None)
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

@bot.message_handler(commands=['start'])
def cmd_start(message):
    add_user(message.chat.id)
    bot.reply_to(
        message,
        "<b>בוט סריקה אוטומטי לפריצות ותבניות מחיר פעיל! 🚀</b>\n\n"
        "המערכת סורקת ברציפות את המדדים S&P 500, NASDAQ 100 ותל אביב 125, ומזהה:\n"
        "• תבניות מחיר: ספל וידית, משולש עולה, דגל שורי, תחתית כפולה/משולשת.\n"
        "• נרות יפניים: פטיש, פטיש הפוך, בליעה שורית, כוכב שחר.\n"
        "• אירועים וחדשות מהותיות בזמן אמת.\n\n"
        "💡 <b>פקודות זמינות:</b>\n"
        "/scan - הרצת סריקה ידנית ברקע\n"
        "/tech SYMBOL - ניתוח טכני וחדשותי למניה בודדת (למשל: <code>/tech AAPL</code> או <code>/tech TEVA.TA</code>)\n"
        "/status - בדיקת סטטוס הסורק",
        parse_mode="HTML"
    )

@bot.message_handler(commands=['tech'])
def cmd_tech(message):
    add_user(message.chat.id)
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "⚠️ נא להזין סימול מניה. לדוגמה: <code>/tech AAPL</code> או <code>/tech TEVA.TA</code>", parse_mode="HTML")
        return

    symbol = parts[1].upper().strip()
    bot.reply_to(message, f"🔍 מנתח את מניית <b>{symbol}</b>...", parse_mode="HTML")

    def run_single_tech():
        tech = analyze_stock_technical(symbol)
        if not tech:
            bot.send_message(message.chat.id, f"❌ לא נמצאו נתונים מספקים או מגמה עולה/תבנית פעילה עבור <b>{symbol}</b>.", parse_mode="HTML")
            return

        news = analyze_news_catalysts(symbol)
        item = {
            "symbol": symbol,
            "index": "תל אביב 125" if symbol.endswith(".TA") else "ארה\"ב",
            "currency": "ILS" if symbol.endswith(".TA") else "USD"
        }
        msg, markup = build_breakout_report(item, tech, news)
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)

    threading.Thread(target=run_single_tech, daemon=True).start()

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    add_user(message.chat.id)
    bot.reply_to(message, "🔍 מפעיל בדיקה ידנית של הסורק האוטומטי ברקע...")
    threading.Thread(target=run_auto_scan_job, daemon=True).start()

@bot.message_handler(commands=['status'])
def cmd_status(message):
    add_user(message.chat.id)
    with SCAN_LOCK:
        is_run = SCAN_STATS["is_running"]
        last = SCAN_STATS["last_run_start"]
    bot.reply_to(message, f"🩺 <b>מצב סורק אוטומטי:</b>\n• סריקה פעילה כעת: {'כן ⏳' if is_run else 'לא'}\n• ריצה אחרונה: {last or 'טרם רצה'}", parse_mode="HTML")

# ------------------------------------------------------------------------------
# 10. הרצת השרת והבוט
# ------------------------------------------------------------------------------
def keep_alive_ping():
    while True:
        time.sleep(600)
        try:
            if "localhost" not in SELF_URL: requests.get(SELF_URL, timeout=10)
        except Exception: pass

if __name__ == "__main__":
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()
    
    logger.info("🤖 Auto-Scan Engine v2.0 started successfully...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
