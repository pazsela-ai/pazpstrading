import os
import sys
import time
import sqlite3
import logging
import threading
import requests
import datetime
import pytz
import re
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
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
# -1. LOGGING - בלי זה אי אפשר לדעת בזמן אמת מה קורה בסריקות ברקע
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("trading_bot")

# ------------------------------------------------------------------------------
# 0. תצורה מרכזית וסף הגדרות (CONFIG)
# ------------------------------------------------------------------------------
CONFIG = {
    "MIN_PRICE": 5.0,
    "MIN_AVG_VOLUME": 150000,
    "MIN_DOLLAR_VOLUME": 1000000.0,
    "MIN_RR": 2.0,
    "MIN_RVOL": 1.3,
    "MAX_BREAKOUT_DIST_ATR_MULT": 1.5,
    "MAX_BREAKOUT_DIST_PCT": 2.0,
    "IGNORE_MARKET_HOURS": False,  # שנה ל-True אם ברצונך שהסורק האוטומטי ירוץ גם בסופ"ש/לילה
    "SCORES": {
        "BULLISH_MARKET_MIN_TECH": 75,
        "NEUTRAL_MARKET_MIN_TECH": 80,
        "BEARISH_MARKET_MIN_TECH": 85,
        "MIN_COMPOSITE_BUY": 75.0,
        "MIN_COMPOSITE_READY": 70.0  # סף מינימלי לרמזור צהוב (SETUP READY)
    },
    "LIMITS": {
        "MAX_ALERTS_PER_SCAN": 5,
        "MAX_ALERTS_PER_DAY": 15,
        "ALERT_COOLDOWN_HOURS": 4
    }
}

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
DEFAULT_CHAT_ID = os.environ.get("DEFAULT_CHAT_ID", None) # מזהה הצ'אט שלך כגיבוי לשליחה
PORT = int(os.environ.get("PORT", 5000))
# --- תיקון: הוחזר SELF_URL שהיה קיים בגרסה קודמת ונשמט - בלעדיו אי אפשר להריץ keep-alive אמיתי ---
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")

bot = TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)
translator = GoogleTranslator(source='auto', target='iw')

CACHE_LOCK = threading.Lock()
DB_LOCK = threading.Lock()

BENCHMARK_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
BENCHMARK_CACHE_TTL = 1800 

KNOWN_TICKERS_DICT: Dict[str, dict] = {}

# --- תיקון: קאש לרשימת הטיקרים כדי לא למשוך CSV מגיטהאב ולהעמיס על yfinance בכל סריקה ---
TICKERS_CACHE: Dict[str, Any] = {"tickers": [], "fetched_at": 0.0}
TICKERS_CACHE_TTL = 6 * 3600  # 6 שעות

# --- תיקון: מעקב אחרי מצב הסריקה האחרונה, זמין דרך /status בלי לחפור בלוגים ---
SCAN_STATS: Dict[str, Any] = {
    "last_run_start": None,
    "last_run_end": None,
    "tickers_scanned": 0,
    "tickers_with_errors": 0,
    "buy_candidates": 0,
    "ready_candidates": 0,
    "alerts_sent": 0,
    "last_error": None,
    "is_running": False,
}
SCAN_STATS_LOCK = threading.Lock()

# ------------------------------------------------------------------------------
# 1. מודלים של נתונים (Data Structures)
# ------------------------------------------------------------------------------
@dataclass
class PatternResult:
    score: int
    is_valid: bool
    label: str
    is_bullish: bool
    meta: dict

@dataclass
class SignalResult:
    symbol: str
    market: str
    setup_state: str       # NO_SETUP, READY (צהוב), TRIGGERED (ירוק)
    technical_score: float
    composite_score: float
    is_buy: bool           # True עבור ירוק
    is_ready: bool         # True עבור צהוב (מותנה)
    rejection_reasons: List[str]
    trade_plan: Optional[dict]
    tech_details: dict
    news_details: dict
    fingerprint: str
    timestamp: datetime.datetime

# ------------------------------------------------------------------------------
# 2. PATTERN DETECTORS (משמשים כראיות בלבד - Setup Evidence)
# ------------------------------------------------------------------------------
def detect_hammer(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    if len(df) < 20: return PatternResult(0, False, "נר פטיש", True, {})
    o1, h1, l1, c1 = float(df['Open'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
    body = abs(c1 - o1)
    lower_shadow = min(o1, c1) - l1
    upper_shadow = h1 - max(o1, c1)
    range_total = h1 - l1

    if range_total == 0 or body == 0:
        return PatternResult(0, False, "נר פטיש", True, {})

    ratio = lower_shadow / body
    score = 0
    if ratio >= 2.0 and upper_shadow <= (0.3 * body):
        score = 60
        if ratio >= 3.0: score += 15
        if volume_spike: score += 15
        if l1 <= df['Low'].iloc[-20:].min(): score += 10

    return PatternResult(score, score >= 60, "נר פטיש היפוכי (Hammer)", True, {"ratio": round(ratio, 2)})

def detect_bullish_engulfing(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    if len(df) < 2: return PatternResult(0, False, "בליעה שורית", True, {})
    o1, c1 = float(df['Open'].iloc[-1]), float(df['Close'].iloc[-1])
    o2, c2 = float(df['Open'].iloc[-2]), float(df['Close'].iloc[-2])

    is_engulfing = (c2 < o2) and (c1 > o1) and (c1 >= o2) and (o1 <= c2)
    if not is_engulfing: return PatternResult(0, False, "בליעה שורית", True, {})

    score = 60
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    if body2 > 0 and (body1 / body2) >= 1.5: score += 15
    if volume_spike: score += 15

    return PatternResult(score, score >= 60, "בליעה שורית (Bullish Engulfing)", True, {"body_ratio": round(body1 / max(body2, 0.01), 2)})

def detect_cup_and_handle(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    if len(df) < 60: return PatternResult(0, False, "ספל וידית", True, {})
    high_60 = float(df['High'].iloc[-60:-15].max())
    cup_bottom = float(df['Low'].iloc[-60:-10].min())
    cup_depth = (high_60 - cup_bottom) / max(high_60, 0.01)

    if not (0.08 <= cup_depth <= 0.38): return PatternResult(0, False, "ספל וידית", True, {})

    handle_df = df.iloc[-12:-2]
    handle_vol_avg = handle_df['Volume'].mean()
    prior_vol_avg = df['Volume'].iloc[-30:-12].mean()
    handle_retrace = (high_60 - handle_df['Low'].min()) / max(high_60, 0.01)
    has_valid_handle = (handle_retrace <= (cup_depth * 0.5)) and (handle_vol_avg < prior_vol_avg)

    score = 50 + (30 if has_valid_handle else 0) + (20 if volume_spike else 0)
    return PatternResult(score, score >= 70, "ספל וידית (Cup & Handle)", True, {"cup_depth": round(cup_depth, 2)})

def detect_double_bottom(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    if len(df) < 50: return PatternResult(0, False, "תחתית כפולה", True, {})
    lows = df['Low'].iloc[-50:-10]
    first_bottom = float(lows.min())
    second_candidates = df['Low'].iloc[-20:-2]
    if second_candidates.empty: return PatternResult(0, False, "תחתית כפולה", True, {})

    second_bottom = float(second_candidates.min())
    diff = abs(first_bottom - second_bottom) / max(first_bottom, 0.01)
    if diff > 0.025: return PatternResult(0, False, "תחתית כפולה", True, {})

    score = 60 + (20 if volume_spike else 0) + (20 if diff < 0.01 else 0)
    return PatternResult(score, score >= 70, "תחתית כפולה (Double Bottom)", True, {"diff_pct": round(diff * 100, 2)})

def detect_triangles(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    if len(df) < 30: return PatternResult(0, False, "משולש", True, {})
    sub_df = df.iloc[-30:]
    x = np.arange(len(sub_df))
    mean_price = float(sub_df['Close'].mean())
    if mean_price == 0: return PatternResult(0, False, "משולש", True, {})

    raw_slope_high, _ = np.polyfit(x, sub_df['High'].values, 1)
    raw_slope_low, _ = np.polyfit(x, sub_df['Low'].values, 1)
    norm_slope_high = raw_slope_high / mean_price
    norm_slope_low = raw_slope_low / mean_price

    if abs(norm_slope_high) < 0.0015 and norm_slope_low > 0.002:
        score = 70 + (15 if volume_spike else 0)
        return PatternResult(score, True, "משולש עולה (Ascending Triangle)", True, {"norm_slope_low": round(norm_slope_low, 4)})

    return PatternResult(0, False, "משולש", True, {})

def detect_flags_pennants(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    if len(df) < 25: return PatternResult(0, False, "דגל", True, {})
    pole_df = df.iloc[-20:-10]
    pole_return = (pole_df['Close'].iloc[-1] - pole_df['Close'].iloc[0]) / max(pole_df['Close'].iloc[0], 0.01)
    if pole_return < 0.06: return PatternResult(0, False, "דגל", True, {})

    flag_df = df.iloc[-10:-1]
    if flag_df['Volume'].mean() < pole_df['Volume'].mean():
        score = 75 + (15 if volume_spike else 0)
        return PatternResult(score, True, "דגל שורי (Bullish Flag)", True, {"pole_return_pct": round(pole_return * 100, 1)})

    return PatternResult(0, False, "דגל", True, {})

PATTERN_DETECTORS = [
    detect_hammer, detect_bullish_engulfing, detect_cup_and_handle,
    detect_double_bottom, detect_triangles, detect_flags_pennants
]

# ------------------------------------------------------------------------------
# 3. מסד נתונים וניהול משתמשים
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
                    alert_time TEXT,
                    breakout_level REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    tp1 REAL,
                    tp2 REAL,
                    technical_score REAL,
                    composite_score REAL,
                    outcome TEXT DEFAULT 'PENDING'
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
            if def_id not in users:
                users.append(def_id)
        except ValueError: pass
    return users

def is_signal_in_cooldown(fingerprint: str) -> bool:
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT alert_time FROM sent_signals WHERE fingerprint = ?", (fingerprint,))
            row = cursor.fetchone()
            if not row: return False
            alert_time = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            return (datetime.datetime.now() - alert_time).total_seconds() < (CONFIG["LIMITS"]["ALERT_COOLDOWN_HOURS"] * 3600)

def count_today_alerts() -> int:
    today_prefix = datetime.date.today().strftime('%Y-%m-%d')
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sent_signals WHERE alert_time LIKE ?", (f"{today_prefix}%",))
            return cursor.fetchone()[0]

def record_sent_signal(signal: SignalResult):
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tp1 = signal.trade_plan['tp1'] if signal.trade_plan else 0.0
    tp2 = signal.trade_plan['tp2'] if signal.trade_plan else 0.0
    sl = signal.trade_plan['stop_loss'] if signal.trade_plan else 0.0
    entry = signal.trade_plan['entry'] if signal.trade_plan else 0.0
    breakout = signal.tech_details.get('breakout_details', {}).get('breakout_level', 0.0)

    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sent_signals 
                (fingerprint, symbol, alert_time, breakout_level, entry_price, stop_loss, tp1, tp2, technical_score, composite_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (signal.fingerprint, signal.symbol, now_str, breakout, entry, sl, tp1, tp2, signal.technical_score, signal.composite_score))
            conn.commit()

init_db()

# ------------------------------------------------------------------------------
# 4. ניהול UNIVERSE (S&P 500, NASDAQ-100, TA-125)
# ------------------------------------------------------------------------------
def fetch_market_tickers(force_refresh: bool = False) -> List[dict]:
    global KNOWN_TICKERS_DICT

    # --- תיקון: קאש של 6 שעות - אין טעם למשוך את כל רשימת ה-S&P500 מחדש בכל סריקה ---
    now = time.time()
    if not force_refresh and TICKERS_CACHE["tickers"] and (now - TICKERS_CACHE["fetched_at"] < TICKERS_CACHE_TTL):
        logger.info(f"fetch_market_tickers: using cached list ({len(TICKERS_CACHE['tickers'])} tickers)")
        return TICKERS_CACHE["tickers"]

    results = []

    # 1. S&P 500
    try:
        url_sp = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp = pd.read_csv(url_sp)
        for _, row in df_sp.iterrows():
            sym = str(row['Symbol']).replace('.', '-').strip()
            item = {"symbol": sym, "market": "US", "index": "SP500", "name": str(row.get('Security', sym))}
            results.append(item)
            KNOWN_TICKERS_DICT[sym] = item
        logger.info(f"fetch_market_tickers: loaded {len(df_sp)} S&P500 symbols")
    except Exception as e:
        logger.warning(f"fetch_market_tickers: S&P500 fetch failed ({e}) - falling back to hardcoded lists only")

    # 2. Nasdaq 100 Fallback/Add
    nasdaq_top = ["QQQ", "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX", "COST", "TMUS", "CSCO", "PLTR", "ARM"]
    for sym in nasdaq_top:
        if sym not in KNOWN_TICKERS_DICT:
            item = {"symbol": sym, "market": "US", "index": "NASDAQ100", "name": sym}
            results.append(item)
            KNOWN_TICKERS_DICT[sym] = item

    # 3. TA-125 (TASE - Israeli Market)
    tase_sample = ["NICE.TA", "TEVA.TA", "LUMI.TA", "POLI.TA", "ICL.TA", "AZRG.TA"]
    for sym in tase_sample:
        item = {"symbol": sym, "market": "IL", "index": "TA125", "name": sym.replace('.TA', '')}
        results.append(item)
        KNOWN_TICKERS_DICT[sym] = item

    if not results:
        logger.error("fetch_market_tickers: results list is EMPTY - scan will find nothing this round")

    TICKERS_CACHE["tickers"] = results
    TICKERS_CACHE["fetched_at"] = now
    logger.info(f"fetch_market_tickers: total universe = {len(results)} symbols")
    return results

# ------------------------------------------------------------------------------
# 5. רכיבי ניתוח טכני מתקדמים (Detailed Technical Subroutines)
# ------------------------------------------------------------------------------
def fetch_benchmark_data(symbol_bm: str = "SPY") -> pd.DataFrame:
    now = time.time()
    if symbol_bm in BENCHMARK_CACHE:
        cached_time, df_bm = BENCHMARK_CACHE[symbol_bm]
        if now - cached_time < BENCHMARK_CACHE_TTL:
            return df_bm
    try:
        df_bm = yf.Ticker(symbol_bm).history(period="1y")
        BENCHMARK_CACHE[symbol_bm] = (now, df_bm)
        return df_bm
    except Exception:
        return pd.DataFrame()

def analyze_market_structure(df: pd.DataFrame) -> dict:
    if len(df) < 50:
        return {"trend_structure": "NEUTRAL", "structure_score": 5, "resistance_level": float(df['High'].iloc[-1]) if not df.empty else 0.0, "support_level": float(df['Low'].iloc[-1]) if not df.empty else 0.0}

    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    swing_highs = []
    swing_lows = []
    for i in range(10, n - 1):
        if highs[i] == max(highs[i-5:i+1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i-5:i+1]):
            swing_lows.append(lows[i])

    hh = len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]
    hl = len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]

    resistance = max(swing_highs[-2:]) if len(swing_highs) >= 2 else float(df['High'].iloc[-20:].max())
    support = min(swing_lows[-2:]) if len(swing_lows) >= 2 else float(df['Low'].iloc[-20:].min())

    if hh and hl:
        trend_struct = "BULLISH"
        structure_score = 15
    elif hl:
        trend_struct = "MODERATE_BULLISH"
        structure_score = 10
    else:
        trend_struct = "NEUTRAL"
        structure_score = 5

    return {
        "trend_structure": trend_struct,
        "higher_highs": hh,
        "higher_lows": hl,
        "resistance_level": resistance,
        "support_level": support,
        "structure_score": structure_score
    }

def detect_breakout_quality(df: pd.DataFrame, current_price: float, resistance: float, atr: float) -> dict:
    if len(df) < 2:
        return {"is_breakout": False, "breakout_confirmed": False, "breakout_level": resistance, "breakout_score": 0, "distance_pct": 0.0}

    prev_close = float(df['Close'].iloc[-2])
    breakout_buffer = max(resistance * 0.002, atr * 0.10)
    required_level = resistance + breakout_buffer

    distance_pct = ((current_price - resistance) / max(resistance, 0.01)) * 100
    max_dist_atr = atr * CONFIG["MAX_BREAKOUT_DIST_ATR_MULT"]
    max_dist_pct = CONFIG["MAX_BREAKOUT_DIST_PCT"]
    too_far = (current_price - resistance > max_dist_atr) or (distance_pct > max_dist_pct)

    is_breakout = (prev_close <= (resistance + breakout_buffer)) and (current_price > required_level) and not too_far
    holds_above = current_price >= resistance

    last_candle = df.iloc[-1]
    c_open, c_high, c_low, c_close = float(last_candle['Open']), float(last_candle['High']), float(last_candle['Low']), float(last_candle['Close'])
    candle_range = max(c_high - c_low, 0.01)
    body = abs(c_close - c_open)
    body_ratio = body / candle_range
    close_location = (c_close - c_low) / candle_range

    strong_candle = (body_ratio >= 0.55) and (close_location >= 0.70)

    score = 0
    if is_breakout:
        score += 12
        if strong_candle: score += 8
        if holds_above: score += 5
    elif current_price >= (resistance * 0.98): # קירבה משמעותית לפריצה (READY)
        score += 10

    return {
        "is_breakout": is_breakout,
        "breakout_confirmed": is_breakout and holds_above and strong_candle,
        "breakout_level": resistance,
        "too_far": too_far,
        "candle_strength": strong_candle,
        "distance_from_breakout_pct": round(distance_pct, 2),
        "breakout_score": score
    }

def analyze_volume_metrics(df: pd.DataFrame) -> dict:
    curr_vol = float(df['Volume'].iloc[-1])
    avg_vol_20 = float(df['Volume'].iloc[-21:-1].mean()) if len(df) >= 21 else float(df['Volume'].mean())
    avg_vol_20 = max(avg_vol_20, 1.0)

    rvol = curr_vol / avg_vol_20
    price_change = float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2]) if len(df) >= 2 else 0.0

    vol_score = 0
    if rvol >= 2.0: vol_score = 15
    elif rvol >= 1.5: vol_score = 12
    elif rvol >= 1.3: vol_score = 8
    elif rvol >= 1.0: vol_score = 4

    vol_supports_price = (price_change > 0) and (rvol >= CONFIG["MIN_RVOL"])
    return {
        "rvol": round(rvol, 2),
        "curr_volume": curr_vol,
        "avg_volume_20": avg_vol_20,
        "volume_supports_price": vol_supports_price,
        "volume_score": vol_score
    }

def analyze_momentum_and_rsi(df: pd.DataFrame) -> dict:
    rsi = float(df['RSI'].iloc[-1])
    rsi_score = 0
    if 50 <= rsi <= 68: rsi_score = 10
    elif 68 < rsi <= 75: rsi_score = 6
    elif 40 <= rsi < 50: rsi_score = 3

    return {
        "rsi": round(rsi, 1),
        "rsi_overextended": rsi > 75,
        "momentum_score": rsi_score
    }

def calculate_relative_strength(df: pd.DataFrame, df_bm: pd.DataFrame) -> dict:
    if df.empty or df_bm.empty or len(df) < 20 or len(df_bm) < 20:
        return {"rs_score": 5, "relative_return_20d": 0.0}

    stock_ret = (float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-20])) / max(float(df['Close'].iloc[-20]), 0.01)
    bm_ret = (float(df_bm['Close'].iloc[-1]) - float(df_bm['Close'].iloc[-20])) / max(float(df_bm['Close'].iloc[-20]), 0.01)

    rel_perf = (stock_ret - bm_ret) * 100
    rs_score = 10 if rel_perf >= 6.0 else (7 if rel_perf >= 2.0 else (4 if rel_perf >= -2.0 else 0))

    return {"rs_score": rs_score, "relative_return_20d": round(rel_perf, 2)}

def detect_volatility_compression(df: pd.DataFrame) -> dict:
    if len(df) < 20: return {"is_compressed": False, "compression_score": 0}
    atr_now = float(df['ATR'].iloc[-1])
    atr_10d_ago = float(df['ATR'].iloc[-10]) if len(df) >= 10 else atr_now
    vol_recent = df['Volume'].iloc[-10:-1].mean()
    vol_prior = df['Volume'].iloc[-30:-10].mean() if len(df) >= 30 else vol_recent

    is_compressed = (atr_now < atr_10d_ago) and (vol_recent < vol_prior)
    return {"is_compressed": is_compressed, "compression_score": 5 if is_compressed else 0}

def detect_multi_timeframe_confirmation(ticker: yf.Ticker, current_price: float) -> dict:
    try:
        df_1h = ticker.history(period="1mo", interval="1h")
        if df_1h.empty or len(df_1h) < 20:
            return {"mtf_status": "UNKNOWN", "mtf_score": 2, "confirmed": False}

        df_1h['EMA20'] = ta.ema(df_1h['Close'], length=20)
        ema20_1h = float(df_1h['EMA20'].iloc[-1])
        confirmed = current_price > ema20_1h

        return {
            "mtf_status": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
            "mtf_score": 5 if confirmed else 0,
            "confirmed": confirmed
        }
    except Exception:
        return {"mtf_status": "UNKNOWN", "mtf_score": 2, "confirmed": False}

def detect_market_regime() -> dict:
    df_sp = fetch_benchmark_data("SPY")
    if df_sp.empty or len(df_sp) < 50:
        return {"regime": "NEUTRAL", "min_tech_score": CONFIG["SCORES"]["NEUTRAL_MARKET_MIN_TECH"]}

    close_sp = float(df_sp['Close'].iloc[-1])
    ema50_sp = float(ta.ema(df_sp['Close'], length=50).iloc[-1])
    ema200_sp = float(ta.ema(df_sp['Close'], length=200).iloc[-1]) if len(df_sp) >= 200 else ema50_sp

    if close_sp > ema50_sp > ema200_sp:
        return {"regime": "BULLISH", "min_tech_score": CONFIG["SCORES"]["BULLISH_MARKET_MIN_TECH"]}
    elif close_sp < ema50_sp < ema200_sp:
        return {"regime": "BEARISH", "min_tech_score": CONFIG["SCORES"]["BEARISH_MARKET_MIN_TECH"]}
    else:
        return {"regime": "NEUTRAL", "min_tech_score": CONFIG["SCORES"]["NEUTRAL_MARKET_MIN_TECH"]}

# ------------------------------------------------------------------------------
# 6. מחשב ניקוד טכני מלא (Technical Score Engine)
# ------------------------------------------------------------------------------
def calculate_technical_score(df: pd.DataFrame, ticker: yf.Ticker, live_price: Optional[float] = None) -> dict:
    close_price = float(df['Close'].iloc[-1])
    entry_price = live_price if (live_price and live_price > 0 and not np.isnan(live_price)) else close_price

    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    atr = float(df['ATR'].dropna().iloc[-1]) if not df['ATR'].dropna().empty else 1.0
    ema20 = float(df['EMA20'].dropna().iloc[-1]) if not df['EMA20'].dropna().empty else entry_price
    ema50 = float(df['EMA50'].dropna().iloc[-1]) if not df['EMA50'].dropna().empty else entry_price

    struct = analyze_market_structure(df)
    bk = detect_breakout_quality(df, entry_price, struct["resistance_level"], atr)
    vol = analyze_volume_metrics(df)
    mom = analyze_momentum_and_rsi(df)
    df_bm = fetch_benchmark_data("SPY")
    rs = calculate_relative_strength(df, df_bm)

    trend_score = 10 if entry_price > ema20 > ema50 else (5 if entry_price > ema20 else 0)
    comp = detect_volatility_compression(df)
    mtf = detect_multi_timeframe_confirmation(ticker, entry_price)

    # Risk / Reward חישוב אמיתי ודינמי
    stop_loss = struct["support_level"] - (0.5 * atr) if struct["support_level"] < entry_price else entry_price - (1.5 * atr)
    risk = entry_price - stop_loss

    if risk <= 0: risk = 0.01

    tp1 = entry_price + (2.0 * risk)
    tp2 = entry_price + (3.5 * risk)
    rr_ratio = (tp1 - entry_price) / risk
    rr_score = 5 if rr_ratio >= CONFIG["MIN_RR"] else 0

    total_tech_score = (
        struct["structure_score"] +
        bk["breakout_score"] +
        vol["volume_score"] +
        mom["momentum_score"] +
        rs["rs_score"] +
        trend_score +
        comp["compression_score"] +
        mtf["mtf_score"] +
        rr_score
    )

    dist_ema20 = ((entry_price - ema20) / max(ema20, 0.01)) * 100
    if dist_ema20 > 8.0:
        total_tech_score = max(0, total_tech_score - 15)

    return {
        "technical_score": float(total_tech_score),
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "atr": round(atr, 2),
        "structure": struct,
        "breakout_details": bk,
        "volume_details": vol,
        "momentum_details": mom,
        "relative_strength": rs,
        "compression": comp,
        "mtf": mtf,
        "rr_ratio": round(rr_ratio, 2),
        "distance_from_ema20": round(dist_ema20, 2)
    }

# ------------------------------------------------------------------------------
# 7. ניתוח חדשות ואימות תוכן (News Subsystem)
# ------------------------------------------------------------------------------
HIGH_IMPACT_CATALYSTS = {
    r"\bfda\b|\btrial\b|\bphase\b|\bapproval\b": ("אישור/ניסוי קליני (FDA/Pharma)", 25),
    r"\bearnings\b|\bbeat\b|\brevenue beat\b": ("דוחות כספיים / תוצאות שיא 📈", 20),
    r"\bguidance\b|\braises outlook\b": ("עדכון תחזית צמיחה כלפי מעלה 🚀", 20),
    r"\bmerger\b|\bacquisition\b|\bbuyout\b": ("עסקת מיזוג / רכישה דרמטית 🤝", 20),
    r"\bcontract\b|\bdeal\b|\bpartnership\b": ("חתימת חוזה אסטרטגי 📝", 15)
}

BEARISH_NEWS_PATTERNS = r"\blawsuit\b|\binvestigation\b|\bdowngrade\b|\boffering\b|\bdilution\b|\bmissed\b|\bfail\b"

def analyze_news_catalyst(symbol: str) -> dict:
    headlines = []
    if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
        try:
            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={today.strftime('%Y-%m-%d')}&token={FINNHUB_API_KEY}"
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                for item in res.json():
                    if item.get("headline"): headlines.append(item.get("headline"))
        except Exception: pass

    if not headlines:
        try:
            news = yf.Ticker(symbol).news
            if news:
                for item in news:
                    if item.get("title"): headlines.append(item.get("title"))
        except Exception: pass

    news_score = 0
    catalyst_label = "ללא אירוע חדשותי מיוחד"
    valid_headline = ""
    has_bearish_news = False

    for h in headlines[:5]:
        h_lower = h.lower()
        if re.search(BEARISH_NEWS_PATTERNS, h_lower):
            has_bearish_news = True
            break
        for pattern, (label, score) in HIGH_IMPACT_CATALYSTS.items():
            if re.search(pattern, h_lower):
                news_score = score
                catalyst_label = label
                valid_headline = h
                break
        if news_score > 0: break

    return {
        "news_score": news_score,
        "catalyst_label": catalyst_label,
        "headline": valid_headline,
        "has_bearish_news": has_bearish_news,
        "news_available": len(headlines) > 0
    }

# ------------------------------------------------------------------------------
# 8. מחולל איתותים אחיד ו-BUY / READY Gate מרכזי
# ------------------------------------------------------------------------------
def generate_signal(symbol: str, df: pd.DataFrame = None, live_price: float = None) -> SignalResult:
    rejection_reasons = []
    meta_info = KNOWN_TICKERS_DICT.get(symbol, {"market": "US", "name": symbol})

    try:
        ticker = yf.Ticker(symbol)
        if df is None or df.empty:
            df = ticker.history(period="1y")

        if df.empty or len(df) < 60:
            return SignalResult(symbol, meta_info["market"], "NO_SETUP", 0, 0, False, False, ["אין מספיק נתונים היסטוריים"], None, {}, {}, "", datetime.datetime.now())

        if live_price is None:
            live_price = float(df['Close'].iloc[-1])

        # 1. סינון נזילות (Liquidity Filter)
        close_p = float(df['Close'].iloc[-1])
        avg_vol = float(df['Volume'].iloc[-20:].mean())
        dollar_vol = close_p * avg_vol

        if close_p < CONFIG["MIN_PRICE"]: rejection_reasons.append(f"מחיר נמוך מ-${CONFIG['MIN_PRICE']}")
        if avg_vol < CONFIG["MIN_AVG_VOLUME"]: rejection_reasons.append("נפח מסחר ממוצע נמוך")
        if dollar_vol < CONFIG["MIN_DOLLAR_VOLUME"]: rejection_reasons.append("נפח כספי יומי נמוך")

        # 2. ניתוח טכני וחדשות
        tech = calculate_technical_score(df, ticker, live_price)
        tech_score = tech["technical_score"]
        news = analyze_news_catalyst(symbol)

        # 3. חישוב Composite Score
        if news["news_available"]:
            composite_score = (tech_score * 0.75) + (news["news_score"] * 0.25)
        else:
            composite_score = tech_score

        # 4. זיהוי תבניות למתן Evidence בלבד
        avg_vol_20 = float(df['Volume'].iloc[-21:-1].mean()) if len(df) >= 21 else float(df['Volume'].mean())
        vol_spike = float(df['Volume'].iloc[-1]) > (avg_vol_20 * 1.4)
        found_patterns = [p.label for detector in PATTERN_DETECTORS if (p := detector(df, tech["entry_price"], vol_spike)).is_valid]

        # 5. הגדרת Setup State המדויק
        bk_confirmed = tech["breakout_details"]["breakout_confirmed"]
        is_breakout = tech["breakout_details"]["is_breakout"]
        res_level = tech["structure"]["resistance_level"]

        if bk_confirmed: 
            setup_state = "TRIGGERED" # רמזור ירוק (פרצה כעת)
        elif is_breakout or (tech["entry_price"] >= res_level * 0.98): 
            setup_state = "READY"     # רמזור צהוב (ממתינה מותנית)
        else: 
            setup_state = "NO_SETUP"

        # 6. בדיקת Conflicts
        if tech["momentum_details"]["rsi_overextended"]: rejection_reasons.append("Soft Conflict: RSI במצב קניות יתר (>75)")
        if tech["distance_from_ema20"] > 8.0: rejection_reasons.append("Soft Conflict: התרחקות יתר מ-EMA20")
        if news["has_bearish_news"]: rejection_reasons.append("Hard Reject: קיימות חדשות שליליות דומיננטיות")
        if tech["breakout_details"]["too_far"]: rejection_reasons.append("Hard Reject: המחיר התרחק מדי מרמת הפריצה")

        # 7. יחס סיכון/תשואה ($R:R \ge 2.0$)
        entry = tech["entry_price"]
        sl = tech["stop_loss"]
        risk = entry - sl
        tp1 = entry + (2.0 * risk)
        tp2 = entry + (3.5 * risk)
        rr_ratio = (tp1 - entry) / max(risk, 0.01)

        if rr_ratio < CONFIG["MIN_RR"]: rejection_reasons.append(f"יחס סיכון/תשואה נמוך מ-1:{CONFIG['MIN_RR']}")

        # מחיר טריגר מותנה לכניסה (Buy Stop Trigger) עבור SETUP READY
        buy_trigger = round(res_level * 1.002, 2)

        trade_plan = {
            "entry": round(entry, 2),
            "buy_trigger": buy_trigger,
            "stop_loss": round(sl, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "risk_reward": round(rr_ratio, 2)
        }

        # 8. BUY & READY Gates מרכזיים
        market_regime = detect_market_regime()
        min_required_tech = market_regime["min_tech_score"]

        has_hard_reject = len([r for r in rejection_reasons if "Hard Reject" in r or "נמוך" in r]) > 0

        # רמזור ירוק - קנייה מיידית (פעל בזמן אמת)
        is_buy = (
            setup_state == "TRIGGERED" and
            tech_score >= min_required_tech and
            composite_score >= CONFIG["SCORES"]["MIN_COMPOSITE_BUY"] and
            tech["volume_details"]["volume_supports_price"] and
            rr_ratio >= CONFIG["MIN_RR"] and
            not has_hard_reject
        )

        # רמזור צהוב - תוכנית עבודה מותנית לפריצה קרובה
        is_ready = (
            not is_buy and
            setup_state == "READY" and
            tech_score >= (min_required_tech - 5) and
            composite_score >= CONFIG["SCORES"]["MIN_COMPOSITE_READY"] and
            rr_ratio >= CONFIG["MIN_RR"] and
            not has_hard_reject
        )

        fp_raw = f"{symbol}_{res_level}_{datetime.date.today()}"
        fingerprint = hashlib.md5(fp_raw.encode()).hexdigest()

        tech["found_patterns"] = found_patterns
        tech["market_regime"] = market_regime["regime"]

        return SignalResult(
            symbol=symbol, market=meta_info["market"], setup_state=setup_state,
            technical_score=round(tech_score, 1), composite_score=round(composite_score, 1),
            is_buy=is_buy, is_ready=is_ready, rejection_reasons=rejection_reasons, trade_plan=trade_plan,
            tech_details=tech, news_details=news, fingerprint=fingerprint, timestamp=datetime.datetime.now()
        )

    except Exception as e:
        return SignalResult(symbol, meta_info["market"], "NO_SETUP", 0, 0, False, False, [f"שגיאה בניתוח: {e}"], None, {}, {}, "", datetime.datetime.now())

# ------------------------------------------------------------------------------
# 9. עיצוב הודעת איתות ברורה (רמזור ירוק וצהוב)
# ------------------------------------------------------------------------------
def build_alert_message(sig: SignalResult) -> Tuple[str, InlineKeyboardMarkup]:
    plan = sig.trade_plan
    tech = sig.tech_details
    news = sig.news_details

    headline_tr = news["headline"]
    if headline_tr:
        try: headline_tr = translator.translate(headline_tr)
        except Exception: pass

    if sig.is_buy:
        header = "🟢 <b>HIGH CONVICTION BUY (פריצה פעילה)</b>"
        entry_text = f"<b>Entry Price:</b> <code>${plan['entry']}</code> (כניסה בשוק)"
    else:
        header = "🟡 <b>WATCHLIST / SETUP READY (תוכנית עבודה מותנית)</b>"
        entry_text = f"🎯 <b>Buy Trigger (הוראת Stop Buy):</b> <code>${plan['buy_trigger']}</code>\n<i>*להיכנס אך ורק אם המחיר סוגר/חוצה מעל סכום זה!</i>"

    msg = f"""{header}

<b>SYMBOL: {sig.symbol} ({sig.market})</b>
<b>Setup State:</b> {sig.setup_state}
<b>Technical Score:</b> {sig.technical_score}/100 | <b>Composite:</b> {sig.composite_score}/100

{entry_text}
<b>Breakout Level:</b> <code>${tech['breakout_details']['breakout_level']}</code>
<b>Stop Loss:</b> <code>${plan['stop_loss']}</code>
<b>Target 1 (TP1):</b> <code>${plan['tp1']}</code>
<b>Target 2 (TP2):</b> <code>${plan['tp2']}</code>
<b>Risk/Reward:</b> <code>1:{plan['risk_reward']}</code>

<b>RVOL:</b> {tech['volume_details']['rvol']}x | <b>RS vs SPY:</b> +{tech['relative_strength']['relative_return_20d']}%
<b>Market Regime:</b> {tech['market_regime']}

<b>Catalyst / החדשות שהשפיעו:</b>
{news['catalyst_label']}
<i>{headline_tr if headline_tr else ''}</i>
"""

    markup = InlineKeyboardMarkup()
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף (TradingView)", url=f"https://www.tradingview.com/chart/?symbol={sig.symbol}")
    markup.add(btn_chart)

    return msg, markup

# ------------------------------------------------------------------------------
# 10. ניהול שעות מסחר ואסטרטגיית סריקה
# ------------------------------------------------------------------------------
def is_market_open(market: str = "US") -> bool:
    if CONFIG.get("IGNORE_MARKET_HOURS", False):
        return True # עוקף שעות מסחר אם מוגדר ב-CONFIG

    if market == "IL":
        tz = pytz.timezone('Asia/Jerusalem')
        now = datetime.datetime.now(tz)
        if now.weekday() in (4, 5): return False
        start = now.replace(hour=10, minute=0, second=0, microsecond=0)
        end = now.replace(hour=17, minute=25, second=0, microsecond=0)
        return start <= now <= end
    else:
        tz = pytz.timezone('America/New_York')
        now = datetime.datetime.now(tz)
        if now.weekday() in (5, 6): return False
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now <= end

def scan_symbol_worker(args: Tuple[dict, bool]) -> Optional[SignalResult]:
    item, force_scan = args
    symbol = item["symbol"]
    try:
        if not force_scan and not is_market_open(item["market"]):
            return None
        sig = generate_signal(symbol)
        if sig.rejection_reasons and sig.rejection_reasons[0].startswith("שגיאה בניתוח"):
            with SCAN_STATS_LOCK:
                SCAN_STATS["tickers_with_errors"] += 1
            logger.warning(f"[{symbol}] error during signal generation: {sig.rejection_reasons[0]}")
        if sig.is_buy:
            logger.info(f"[{symbol}] 🟢 BUY candidate - tech={sig.technical_score} composite={sig.composite_score}")
            return sig
        if sig.is_ready:
            logger.info(f"[{symbol}] 🟡 READY candidate - tech={sig.technical_score} composite={sig.composite_score}")
            return sig
        return None
    except Exception as e:
        with SCAN_STATS_LOCK:
            SCAN_STATS["tickers_with_errors"] += 1
        logger.exception(f"[{symbol}] unexpected exception in scan_symbol_worker: {e}")
        return None

def execute_global_market_scan(force_scan: bool = False) -> List[SignalResult]:
    # --- תיקון: לא מריצים סריקה חדשה אם אחת כבר רצה (מונע חפיפה/תקיעה) ---
    with SCAN_STATS_LOCK:
        if SCAN_STATS["is_running"]:
            logger.warning("execute_global_market_scan: previous scan still running - skipping this trigger")
            return []
        SCAN_STATS["is_running"] = True
        SCAN_STATS["last_run_start"] = datetime.datetime.now()
        SCAN_STATS["tickers_scanned"] = 0
        SCAN_STATS["tickers_with_errors"] = 0
        SCAN_STATS["buy_candidates"] = 0
        SCAN_STATS["ready_candidates"] = 0
        SCAN_STATS["alerts_sent"] = 0
        SCAN_STATS["last_error"] = None

    logger.info(f"execute_global_market_scan: starting scan cycle (force_scan={force_scan})")
    sent_signals_list: List[SignalResult] = []
    try:
        if not force_scan and count_today_alerts() >= CONFIG["LIMITS"]["MAX_ALERTS_PER_DAY"]:
            logger.info("execute_global_market_scan: daily alert limit already reached - skipping")
            return []

        tickers = fetch_market_tickers()
        if not tickers:
            logger.error("execute_global_market_scan: empty ticker universe - aborting scan")
            return []

        with SCAN_STATS_LOCK:
            SCAN_STATS["tickers_scanned"] = len(tickers)

        candidates: List[SignalResult] = []
        worker_args = [(item, force_scan) for item in tickers]

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(scan_symbol_worker, worker_args)
            for sig in results:
                if sig is not None: candidates.append(sig)

        buy_n = len([s for s in candidates if s.is_buy])
        ready_n = len([s for s in candidates if s.is_ready])
        logger.info(f"execute_global_market_scan: {buy_n} BUY + {ready_n} READY candidates out of {len(tickers)} scanned")
        with SCAN_STATS_LOCK:
            SCAN_STATS["buy_candidates"] = buy_n
            SCAN_STATS["ready_candidates"] = ready_n

        if not candidates:
            return []

        candidates.sort(key=lambda x: (
            x.composite_score * 0.50 +
            x.tech_details["breakout_details"]["breakout_score"] * 0.25 +
            x.tech_details["volume_details"]["volume_score"] * 0.15 +
            x.trade_plan["risk_reward"] * 0.10
        ), reverse=True)

        sent_count = 0
        users = get_all_users()
        if not users:
            logger.warning("execute_global_market_scan: no registered users and no DEFAULT_CHAT_ID - nobody will receive the alert")

        for sig in candidates:
            if sent_count >= CONFIG["LIMITS"]["MAX_ALERTS_PER_SCAN"]: break
            if not force_scan and count_today_alerts() >= CONFIG["LIMITS"]["MAX_ALERTS_PER_DAY"]: break
            if not force_scan and is_signal_in_cooldown(sig.fingerprint): continue

            msg, markup = build_alert_message(sig)
            for chat_id in users:
                try:
                    bot.send_message(chat_id, msg, parse_mode="HTML", reply_markup=markup)
                except Exception as e:
                    logger.warning(f"failed to send alert to chat_id={chat_id}: {e}")

            if not force_scan:
                record_sent_signal(sig)
            sent_signals_list.append(sig)
            sent_count += 1

        with SCAN_STATS_LOCK:
            SCAN_STATS["alerts_sent"] = sent_count
        logger.info(f"execute_global_market_scan: sent {sent_count} alert(s)")
        return sent_signals_list

    except Exception as e:
        logger.exception(f"execute_global_market_scan: fatal error in scan cycle: {e}")
        with SCAN_STATS_LOCK:
            SCAN_STATS["last_error"] = str(e)
        return sent_signals_list
    finally:
        with SCAN_STATS_LOCK:
            SCAN_STATS["is_running"] = False
            SCAN_STATS["last_run_end"] = datetime.datetime.now()

# ------------------------------------------------------------------------------
# 10.1 KEEP-ALIVE אמיתי - תיקון: SELF_URL היה קיים בעבר אך לא חובר לשום פינג.
# בלי זה, שירותי אחסון חינמיים מרדימים את השרת בין בקשות, וה-scheduler נעצר איתו.
# ------------------------------------------------------------------------------
def keep_alive_loop():
    if "localhost" in SELF_URL:
        logger.info("keep_alive_loop: SELF_URL points to localhost - skipping self-ping (not deployed remotely)")
        return
    while True:
        time.sleep(600)  # כל 10 דקות
        try:
            r = requests.get(SELF_URL, timeout=10)
            logger.info(f"keep_alive_loop: self-ping to {SELF_URL} -> {r.status_code}")
        except Exception as e:
            logger.warning(f"keep_alive_loop: self-ping failed: {e}")

# ------------------------------------------------------------------------------
# 10.2 SCHEDULER - תיקון: next_run_time מריץ סריקה ראשונה מיד עם עליית התהליך
# (במקום לחכות 15 דקות), max_instances מונע חפיפה בין סריקות.
# ------------------------------------------------------------------------------
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(
    lambda: execute_global_market_scan(force_scan=False),
    'interval',
    minutes=15,
    next_run_time=datetime.datetime.now(),
    max_instances=1,
    coalesce=True,
    misfire_grace_time=120,
)
scheduler.start()

# ------------------------------------------------------------------------------
# 11. פקודות TELEGRAM BOT (כולל /scan לסריקה ידנית)
# ------------------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def cmd_start(message):
    add_user(message.chat.id)
    bot.reply_to(
        message, 
        "<b>מערכת איתותים ותוכניות עבודה (v5.5 Precision Engine) פעילה! 🚀</b>\n\n"
        "פקודות זמינות:\n"
        "• <code>/scan</code> - הרצת סריקה ידנית יזומה על כל השוק (רצה ברקע).\n"
        "• <code>/status</code> - מצב הסריקה האחרונה, כמה מניות נסרקו וכמה עברו את השער.\n"
        "• <code>/tech AAPL</code> - ניתוח טכני ותוכנית עבודה מפורטת למניה ספציפית.\n"
        "• <code>/backtest NVDA</code> - הרצת בדיקה היסטורית.", 
        parse_mode="HTML"
    )

@bot.message_handler(commands=['scan', 'testscan'])
def cmd_scan(message):
    """הרצת סריקה ידנית יזומה (עוקף שעות מסחר לבדיקות)"""
    add_user(message.chat.id)

    if SCAN_STATS["is_running"]:
        bot.reply_to(message, "⏳ סריקה כבר רצה כרגע ברקע. אפשר לבדוק את ההתקדמות עם /status, ולנסות /scan שוב אחרי שהיא תסתיים.")
        return

    # --- תיקון קריטי: הסריקה על 500+ טיקרים עלולה לקחת הרבה יותר מ-15-30 שניות.
    # אם היא רצה ישירות בתוך ה-handler, היא חוסמת את thread הבוט וגורמת לכל פקודה
    # אחרת (tech/status/start) "לא להגיב" עד שהסריקה תסתיים. לכן מריצים ב-thread נפרד
    # ומדווחים על ההתחלה/סיום בנפרד. ---
    bot.reply_to(
        message,
        "🔎 <b>מתחילה סריקה ידנית יזומה על כל המניות במערכת ברקע...</b>\n"
        "זה יכול לקחת כמה דקות בהתאם לגודל היקום הנסרק. אפשר לבדוק התקדמות עם /status, "
        "ואת שאר הפקודות (למשל /tech) אפשר להמשיך להשתמש בינתיים.",
        parse_mode="HTML"
    )

    def _run():
        chat_id = message.chat.id
        try:
            found_signals = execute_global_market_scan(force_scan=True)
            if not found_signals:
                bot.send_message(chat_id, "✅ הסריקה הסתיימה: לא נמצאו מניות העונות על סף האיכות ברגע זה (נסי /status לפרטים).", parse_mode="HTML")
            else:
                bot.send_message(chat_id, f"✅ הסריקה הסתיימה! נשלחו {len(found_signals)} איתותים/תוכניות עבודה למשתמשים.", parse_mode="HTML")
        except Exception as e:
            logger.exception(f"cmd_scan background thread failed: {e}")
            try:
                bot.send_message(chat_id, f"❌ שגיאה בביצוע הסריקה הידנית: {e}")
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()

@bot.message_handler(commands=['status'])
def cmd_status(message):
    """תיקון: פקודת אבחון - כדי לדעת בלי לחפור בלוגים אם הסריקה בכלל רצה, על כמה מניות, וכמה עברו את השערים."""
    add_user(message.chat.id)
    with SCAN_STATS_LOCK:
        s = dict(SCAN_STATS)

    def fmt(dt):
        return dt.strftime('%Y-%m-%d %H:%M:%S') if dt else "טרם רצה"

    regime = detect_market_regime()
    reply = (
        f"<b>🩺 מצב מערכת</b>\n\n"
        f"<b>סריקה פעילה כרגע:</b> {'כן ⏳' if s['is_running'] else 'לא'}\n"
        f"<b>תחילת סריקה אחרונה:</b> {fmt(s['last_run_start'])}\n"
        f"<b>סיום סריקה אחרונה:</b> {fmt(s['last_run_end'])}\n"
        f"<b>מניות שנסרקו:</b> {s['tickers_scanned']}\n"
        f"<b>שגיאות בסריקה:</b> {s['tickers_with_errors']}\n"
        f"<b>🟢 מועמדות BUY:</b> {s['buy_candidates']} | <b>🟡 מועמדות READY:</b> {s['ready_candidates']}\n"
        f"<b>איתותים שנשלחו:</b> {s['alerts_sent']}\n"
        f"<b>שגיאה כללית אחרונה:</b> {s['last_error'] or 'אין'}\n\n"
        f"<b>משטר שוק נוכחי (SPY):</b> {regime['regime']} (סף טכני BUY: {regime['min_tech_score']})\n"
        f"<b>איתותים היום:</b> {count_today_alerts()}/{CONFIG['LIMITS']['MAX_ALERTS_PER_DAY']}\n"
        f"<b>משתמשים רשומים:</b> {len(get_all_users())}\n"
        f"<b>IGNORE_MARKET_HOURS:</b> {CONFIG['IGNORE_MARKET_HOURS']}"
    )
    bot.send_message(message.chat.id, reply, parse_mode="HTML")

@bot.message_handler(commands=['tech'])
def cmd_tech(message):
    add_user(message.chat.id)
    try:
        symbol = message.text.split()[1].upper()
        sig = generate_signal(symbol)

        if sig.is_buy or sig.is_ready:
            msg, markup = build_alert_message(sig)
            bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
        else:
            reasons = "\n".join([f"• {r}" for r in sig.rejection_reasons]) if sig.rejection_reasons else "• לא התקיימה תבנית פריצה או כניסה מותנית"
            reply = (
                f"<b>📊 תוצאת ניתוח עבור {symbol}</b>\n\n"
                f"<b>State:</b> <code>{sig.setup_state}</code>\n"
                f"<b>Technical Score:</b> <code>{sig.technical_score}/100</code>\n"
                f"<b>Composite Score:</b> <code>{sig.composite_score}/100</code>\n\n"
                f"❌ <b>סיבות לדחיית איתות/תוכנית עבודה:</b>\n{reasons}"
            )
            bot.send_message(message.chat.id, reply, parse_mode="HTML")
    except IndexError:
        bot.reply_to(message, "⚠️ נא לציין סימול מניה: <code>/tech AAPL</code>", parse_mode="HTML")

@bot.message_handler(commands=['backtest'])
def cmd_backtest(message):
    try:
        symbol = message.text.split()[1].upper()
        bot.reply_to(message, f"⏳ מריץ Backtest היסטורי נקי עבור {symbol}...", parse_mode="HTML")

        df = yf.Ticker(symbol).history(period="2y")
        if len(df) < 150:
            bot.send_message(message.chat.id, f"❌ אין מספיק נתונים עבור {symbol}.")
            return

        total_signals = 0
        tp1_hits = 0
        sl_hits = 0
        r_multiples = []

        for i in range(100, len(df) - 15):
            sub_df = df.iloc[:i]
            curr_price = float(sub_df['Close'].iloc[-1])

            sig = generate_signal(symbol, df=sub_df, live_price=curr_price)

            if (sig.is_buy or sig.is_ready) and sig.trade_plan:
                total_signals += 1
                target_tp1 = sig.trade_plan['tp1']
                target_sl = sig.trade_plan['stop_loss']

                future_df = df.iloc[i:i+15]

                for _, row in future_df.iterrows():
                    high = float(row['High'])
                    low = float(row['Low'])

                    if low <= target_sl:
                        sl_hits += 1
                        r_multiples.append(-1.0)
                        break
                    elif high >= target_tp1:
                        tp1_hits += 1
                        r_multiples.append(2.0)
                        break

        win_rate = round((tp1_hits / max(total_signals, 1)) * 100, 1)
        expectancy = round(np.mean(r_multiples), 2) if r_multiples else 0.0

        reply = (
            f"<b>🔬 תוצאות Backtest עבור {symbol} (שנתיים אחורה):</b>\n\n"
            f"• סה\"כ איתותים שנמצאו: <code>{total_signals}</code>\n"
            f"• פגיעות ביעד (TP1): <code>{tp1_hits}</code>\n"
            f"• פגיעות בסטופ (SL): <code>{sl_hits}</code>\n"
            f"• תוחלת ב-R (Expectancy): <code>{expectancy}R</code>\n"
            f"• 🏆 <b>אחוז הצלחה: {win_rate}%</b>"
        )
        bot.send_message(message.chat.id, reply, parse_mode="HTML")
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ שגיאה בהרצת Backtest: {e}")

# ------------------------------------------------------------------------------
# 12. UNIT TESTS
# ------------------------------------------------------------------------------
def run_unit_tests():
    logger.info("🧪 מריץ Unit Tests מקיפים...")
    dates = pd.date_range(start="2023-01-01", periods=100)

    data = {
        'Open': [100.0]*98 + [100.0, 101.0],
        'High': [101.0]*98 + [101.0, 107.0],
        'Low': [99.0]*98 + [99.5, 100.5],
        'Close': [100.0]*98 + [100.5, 106.5],
        'Volume': [1000000]*98 + [1000000, 3000000]
    }
    df_test = pd.DataFrame(data, index=dates)
    df_test['ATR'] = 2.0
    bk = detect_breakout_quality(df_test, 106.5, 101.0, 2.0)
    assert bk["is_breakout"], "Unit Test Failed: Valid Breakout Not Detected"

    logger.info("✅ כל בדיקות היחידה (Unit Tests) עברו בהצלחה!")

# ------------------------------------------------------------------------------
# 13. FLASK & KEEP-ALIVE
# ------------------------------------------------------------------------------
@app.route('/')
def health_check():
    return "OK - Precision Trading Engine Active", 200

if __name__ == "__main__":
    run_unit_tests()
    logger.info(f"SELF_URL = {SELF_URL}")
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()
    threading.Thread(target=keep_alive_loop, daemon=True).start()  # תיקון: מפעיל בפועל את מנגנון ה-keep-alive
    logger.info("🤖 Precision Trading Engine Bot Is Ready... (use /status to check the scanner, /scan to trigger one manually)")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
