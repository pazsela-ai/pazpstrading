import os
import time
import sqlite3
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
# 1. הגדרות סביבה, קבועים ובטיחות תהליכונים (Thread Safety)
# ------------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
PORT = int(os.environ.get("PORT", 5000))
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")

bot = TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)
translator = GoogleTranslator(source='auto', target='iw')

CACHE_LOCK = threading.Lock()
DB_LOCK = threading.Lock()
SECTOR_LOCK = threading.Lock()

CACHE: Dict[str, Tuple[float, dict]] = {}
CACHE_TTL = 180  # 3 דקות לנתונים כלליים

BENCHMARK_CACHE: Dict[str, pd.DataFrame] = {}
BENCHMARK_CACHE_TTL = 1800  # 30 דקות לנתוני מדדי ייחוס

KNOWN_TICKERS_SET = set()
KNOWN_COMPANIES_DICT = {}
SECTOR_INFO_DICT = {}

MAX_ALERTS_PER_SCAN = 3
MAX_ALERTS_PER_DAY = 5
ALERT_COOLDOWN_HOURS = 4

# ------------------------------------------------------------------------------
# 2. מודלים של נתונים (Data Structures)
# ------------------------------------------------------------------------------
@dataclass
class PatternResult:
    score: int            # ציון ביטחון (0 עד 100)
    is_valid: bool        # האם המבנה קיים
    label: str            # שם התבנית
    is_bullish: bool      # True/False
    meta: dict            # נתונים גיאומטריים של המבנה

@dataclass
class SignalResult:
    symbol: str
    setup_state: str       # NO_SETUP, READY, TRIGGERED
    technical_score: float
    composite_score: float
    is_buy: bool
    rejection_reasons: List[str]
    trade_plan: Optional[dict]
    tech_details: dict
    news_details: dict
    fingerprint: str
    timestamp: datetime.datetime

# ------------------------------------------------------------------------------
# 3. PATTERN DETECTORS (משמשים כראיות בלבד - Setup Evidence)
# ------------------------------------------------------------------------------
def detect_hammer(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
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
        if ratio >= 3.0:
            score += 15
        if volume_spike:
            score += 15
        if l1 <= df['Low'].iloc[-20:].min():
            score += 10

    return PatternResult(
        score=score,
        is_valid=score >= 60,
        label="נר פטיש היפוכי (Hammer)",
        is_bullish=True,
        meta={"ratio": round(ratio, 2)}
    )

def detect_bullish_engulfing(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    o1, c1 = float(df['Open'].iloc[-1]), float(df['Close'].iloc[-1])
    o2, c2 = float(df['Open'].iloc[-2]), float(df['Close'].iloc[-2])

    is_engulfing = (c2 < o2) and (c1 > o1) and (c1 >= o2) and (o1 <= c2)
    if not is_engulfing:
        return PatternResult(0, False, "בליעה שורית", True, {})

    score = 60
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    if body2 > 0 and (body1 / body2) >= 1.5:
        score += 15
    if volume_spike:
        score += 15

    return PatternResult(
        score=score,
        is_valid=score >= 60,
        label="בליעה שורית (Bullish Engulfing)",
        is_bullish=True,
        meta={"body_ratio": round(body1 / max(body2, 0.01), 2)}
    )

def detect_cup_and_handle(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    if len(df) < 60:
        return PatternResult(0, False, "ספל וידית", True, {})

    high_60 = float(df['High'].iloc[-60:-15].max())
    cup_bottom = float(df['Low'].iloc[-60:-10].min())
    cup_depth = (high_60 - cup_bottom) / high_60

    if not (0.08 <= cup_depth <= 0.38):
        return PatternResult(0, False, "ספל וידית", True, {})

    handle_df = df.iloc[-12:-2]
    handle_vol_avg = handle_df['Volume'].mean()
    prior_vol_avg = df['Volume'].iloc[-30:-12].mean()
    handle_retrace = (high_60 - handle_df['Low'].min()) / high_60
    has_valid_handle = (handle_retrace <= (cup_depth * 0.5)) and (handle_vol_avg < prior_vol_avg)

    score = 50
    if has_valid_handle:
        score += 30
    if volume_spike:
        score += 20

    return PatternResult(
        score=score,
        is_valid=score >= 70,
        label="ספל וידית (Cup & Handle)",
        is_bullish=True,
        meta={"cup_depth": round(cup_depth, 2), "has_handle": has_valid_handle}
    )

def detect_double_bottom(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    lows = df['Low'].iloc[-50:-10]
    first_bottom = float(lows.min())
    second_candidates = df['Low'].iloc[-20:-2]
    if second_candidates.empty:
        return PatternResult(0, False, "תחתית כפולה", True, {})

    second_bottom = float(second_candidates.min())
    diff = abs(first_bottom - second_bottom) / first_bottom

    if diff > 0.025:
        return PatternResult(0, False, "תחתית כפולה", True, {})

    score = 60
    if volume_spike:
        score += 20
    if diff < 0.01:
        score += 20

    return PatternResult(
        score=score,
        is_valid=score >= 70,
        label="תחתית כפולה (Double Bottom)",
        is_bullish=True,
        meta={"diff_pct": round(diff * 100, 2)}
    )

def detect_triangles(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    if len(df) < 30:
        return PatternResult(0, False, "משולש", True, {})

    sub_df = df.iloc[-30:]
    x = np.arange(len(sub_df))
    highs = sub_df['High'].values
    lows = sub_df['Low'].values
    mean_price = float(sub_df['Close'].mean())
    if mean_price == 0:
        return PatternResult(0, False, "משולש", True, {})

    raw_slope_high, _ = np.polyfit(x, highs, 1)
    raw_slope_low, _ = np.polyfit(x, lows, 1)
    norm_slope_high = raw_slope_high / mean_price
    norm_slope_low = raw_slope_low / mean_price

    if abs(norm_slope_high) < 0.0015 and norm_slope_low > 0.002:
        score = 70 + (15 if volume_spike else 0)
        return PatternResult(score, True, "משולש עולה (Ascending Triangle)", True, {"norm_slope_low": round(norm_slope_low, 4)})

    return PatternResult(0, False, "משולש", True, {})

def detect_flags_pennants(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    if len(df) < 25:
        return PatternResult(0, False, "דגל", True, {})

    pole_df = df.iloc[-20:-10]
    pole_return = (pole_df['Close'].iloc[-1] - pole_df['Close'].iloc[0]) / pole_df['Close'].iloc[0]
    if pole_return < 0.06:
        return PatternResult(0, False, "דגל", True, {})

    flag_df = df.iloc[-10:-1]
    flag_vol_avg = flag_df['Volume'].mean()
    pole_vol_avg = pole_df['Volume'].mean()

    if flag_vol_avg < pole_vol_avg:
        score = 75 + (15 if volume_spike else 0)
        return PatternResult(score, True, "דגל שורי (Bullish Flag)", True, {"pole_return_pct": round(pole_return * 100, 1)})

    return PatternResult(0, False, "דגל", True, {})

PATTERN_DETECTORS = [
    detect_hammer,
    detect_bullish_engulfing,
    detect_cup_and_handle,
    detect_double_bottom,
    detect_triangles,
    detect_flags_pennants
]

# ------------------------------------------------------------------------------
# 4. מסד נתונים, ניהול משתמשים והיסטוריית איתותים
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
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chat_id FROM users")
            return [r[0] for r in cursor.fetchall()]

def is_signal_in_cooldown(fingerprint: str) -> bool:
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT alert_time FROM sent_signals WHERE fingerprint = ?", (fingerprint,))
            row = cursor.fetchone()
            if not row:
                return False
            alert_time = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            return (datetime.datetime.now() - alert_time).total_seconds() < (ALERT_COOLDOWN_HOURS * 3600)

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

def update_alert_outcomes_job():
    print(f"[{datetime.datetime.now()}] 🔄 מריץ בדיקת תוצאות איתותים (Outcome Tracking)...")
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fingerprint, symbol, entry_price, stop_loss, tp1, tp2 FROM sent_signals WHERE outcome = 'PENDING'")
            pending = cursor.fetchall()

    if not pending:
        return

    for fp, symbol, entry, sl, tp1, tp2 in pending:
        try:
            df = yf.Ticker(symbol).history(period="10d")
            if df.empty:
                continue
            max_high = float(df['High'].max())
            min_low = float(df['Low'].min())

            new_outcome = 'PENDING'
            if max_high >= tp2:
                new_outcome = 'TP2_HIT'
            elif max_high >= tp1:
                new_outcome = 'TP1_HIT'
            elif min_low <= sl:
                new_outcome = 'SL_HIT'

            if new_outcome != 'PENDING':
                with DB_LOCK:
                    with sqlite3.connect(DB_FILE) as conn:
                        cursor = conn.cursor()
                        cursor.execute("UPDATE sent_signals SET outcome = ? WHERE fingerprint = ?", (new_outcome, fp))
                        conn.commit()
        except Exception as e:
            print(f"[Outcome Update Error] {symbol}: {e}")

init_db()

def fetch_market_tickers() -> list:
    global KNOWN_TICKERS_SET, KNOWN_COMPANIES_DICT, SECTOR_INFO_DICT
    tickers = set()
    try:
        url_sp500 = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp = pd.read_csv(url_sp500)
        if 'Symbol' in df_sp.columns:
            for _, row in df_sp.iterrows():
                sym = str(row['Symbol']).replace('.', '-').strip()
                name = str(row.get('Security', sym)).split()[0].lower()
                sector = str(row.get('Sector', 'General'))
                tickers.add(sym)
                KNOWN_COMPANIES_DICT[sym] = name
                SECTOR_INFO_DICT[sym] = sector
            KNOWN_TICKERS_SET = tickers
    except Exception as e:
        print(f"[Ticker Fetch Error]: {e}")

    if len(tickers) >= 100:
        return sorted(list(tickers))

    fallback = ["AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", "META", "GOOGL", "NFLX", "LLY", "AVGO", "JPM", "UNH", "V", "PG"]
    KNOWN_TICKERS_SET = set(fallback)
    return fallback

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
        return {"trend_structure": "NEUTRAL", "structure_score": 5, "higher_highs": False, "higher_lows": False}

    highs = df['High'].iloc[-50:].values
    lows = df['Low'].iloc[-50:].values

    swing_highs = []
    swing_lows = []
    for i in range(5, len(highs) - 5):
        if highs[i] == max(highs[i-5:i+6]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i-5:i+6]):
            swing_lows.append(lows[i])

    hh = len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]
    hl = len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]

    resistance = max(swing_highs[-2:]) if len(swing_highs) >= 2 else float(df['High'].iloc[-20:].max())
    support = min(swing_lows[-2:]) if len(swing_lows) >= 2 else float(df['Low'].iloc[-20:].min())

    structure_score = 0
    if hh and hl:
        trend_struct = "BULLISH"
        structure_score = 20
    elif hl:
        trend_struct = "MODERATE_BULLISH"
        structure_score = 14
    else:
        trend_struct = "NEUTRAL"
        structure_score = 5

    return {
        "trend_structure": trend_struct,
        "higher_highs": hh,
        "higher_lows": hl,
        "recent_swing_high": resistance,
        "recent_swing_low": support,
        "resistance_level": resistance,
        "support_level": support,
        "structure_score": structure_score
    }

def detect_breakout_quality(df: pd.DataFrame, current_price: float, resistance: float, atr: float) -> dict:
    prev_close = float(df['Close'].iloc[-2])
    breakout_buffer = max(resistance * 0.002, atr * 0.10)
    required_level = resistance + breakout_buffer

    is_breakout = (prev_close <= resistance) and (current_price > required_level)
    holds_above = current_price >= resistance

    last_candle = df.iloc[-1]
    c_open, c_high, c_low, c_close = float(last_candle['Open']), float(last_candle['High']), float(last_candle['Low']), float(last_candle['Close'])
    candle_range = max(c_high - c_low, 0.01)
    body = abs(c_close - c_open)
    body_ratio = body / candle_range
    close_location = (c_close - c_low) / candle_range

    strong_candle = (body_ratio >= 0.55) and (close_location >= 0.70)
    distance_pct = ((current_price - resistance) / resistance) * 100

    score = 0
    if is_breakout:
        score += 10
        if strong_candle:
            score += 6
        if holds_above:
            score += 4

    return {
        "is_breakout": is_breakout,
        "breakout_confirmed": is_breakout and holds_above,
        "breakout_level": resistance,
        "breakout_buffer": breakout_buffer,
        "candle_strength": strong_candle,
        "distance_from_breakout_pct": round(distance_pct, 2),
        "breakout_score": score
    }

def analyze_volume_metrics(df: pd.DataFrame) -> dict:
    curr_vol = float(df['Volume'].iloc[-1])
    avg_vol_20 = float(df['Volume'].iloc[-21:-1].mean())
    avg_vol_20 = max(avg_vol_20, 1.0)

    rvol = curr_vol / avg_vol_20
    price_change = float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2])

    vol_score = 0
    if rvol >= 3.0:
        vol_score = 15
    elif rvol >= 2.0:
        vol_score = 13
    elif rvol >= 1.5:
        vol_score = 10
    elif rvol >= 1.3:
        vol_score = 7
    elif rvol >= 1.0:
        vol_score = 4

    vol_supports_price = (price_change > 0) and (rvol >= 1.3)
    if not vol_supports_price and price_change > 0 and rvol < 1.0:
        vol_score = max(0, vol_score - 5)

    return {
        "rvol": round(rvol, 2),
        "curr_volume": curr_vol,
        "avg_volume_20": avg_vol_20,
        "volume_supports_price": vol_supports_price,
        "volume_score": vol_score
    }

def analyze_momentum_and_rsi(df: pd.DataFrame) -> dict:
    rsi = float(df['RSI'].iloc[-1])
    rsi_5d_ago = float(df['RSI'].iloc[-6]) if len(df) >= 6 else rsi
    rsi_slope = rsi - rsi_5d_ago

    rsi_score = 0
    if 50 <= rsi <= 68:
        rsi_score = 8
    elif 68 < rsi <= 75:
        rsi_score = 5
    elif 40 <= rsi < 50:
        rsi_score = 3
    elif rsi > 75 or rsi < 40:
        rsi_score = 0

    macd_df = ta.macd(df['Close'])
    macd_score = 0
    macd_accel = False
    if macd_df is not None and not macd_df.empty:
        h_col = [c for c in macd_df.columns if c.startswith('MACDh_')]
        if h_col:
            hist = macd_df[h_col[0]].dropna()
            if len(hist) >= 3:
                curr_h = float(hist.iloc[-1])
                prev_h = float(hist.iloc[-2])
                if curr_h > 0 and curr_h > prev_h:
                    macd_score = 7
                    macd_accel = True
                elif curr_h > 0:
                    macd_score = 4

    total_mom_score = rsi_score + macd_score
    return {
        "rsi": round(rsi, 1),
        "rsi_slope": round(rsi_slope, 1),
        "rsi_overextended": rsi > 75,
        "macd_accel": macd_accel,
        "momentum_score": total_mom_score
    }

def calculate_relative_strength(df: pd.DataFrame, df_bm: pd.DataFrame) -> dict:
    if df.empty or df_bm.empty or len(df) < 20 or len(df_bm) < 20:
        return {"rs_score": 5, "relative_return_20d": 0.0}

    stock_ret = (float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-20])) / float(df['Close'].iloc[-20])
    bm_ret = (float(df_bm['Close'].iloc[-1]) - float(df_bm['Close'].iloc[-20])) / float(df_bm['Close'].iloc[-20])

    rel_perf = (stock_ret - bm_ret) * 100

    rs_score = 0
    if rel_perf >= 8.0:
        rs_score = 10
    elif rel_perf >= 4.0:
        rs_score = 8
    elif rel_perf >= 1.0:
        rs_score = 6
    elif rel_perf >= -2.0:
        rs_score = 3
    else:
        rs_score = 0

    return {
        "rs_score": rs_score,
        "relative_return_20d": round(rel_perf, 2)
    }

def detect_volatility_compression(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return {"is_compressed": False, "compression_score": 0}

    atr_now = float(df['ATR'].iloc[-1])
    atr_10d_ago = float(df['ATR'].iloc[-10]) if len(df) >= 10 else atr_now
    atr_declining = atr_now < atr_10d_ago

    vol_recent = df['Volume'].iloc[-10:-1].mean()
    vol_prior = df['Volume'].iloc[-30:-10].mean()
    vol_declining = vol_recent < vol_prior

    is_compressed = atr_declining and vol_declining
    comp_score = 5 if is_compressed else (2 if atr_declining else 0)

    return {
        "is_compressed": is_compressed,
        "compression_score": comp_score
    }

def detect_multi_timeframe_confirmation(ticker: yf.Ticker, current_price: float) -> dict:
    try:
        df_1h = ticker.history(period="1mo", interval="1h")
        if df_1h.empty or len(df_1h) < 20:
            return {"mtf_confirmed": True, "mtf_score": 3}

        df_1h['EMA20'] = ta.ema(df_1h['Close'], length=20)
        ema20_1h = float(df_1h['EMA20'].iloc[-1])

        confirmed = current_price > ema20_1h
        return {
            "mtf_confirmed": confirmed,
            "mtf_score": 5 if confirmed else 1
        }
    except Exception:
        return {"mtf_confirmed": True, "mtf_score": 3}

def detect_market_regime() -> dict:
    df_sp = fetch_benchmark_data("SPY")
    if df_sp.empty or len(df_sp) < 50:
        return {"regime": "NEUTRAL", "min_tech_score": 70}

    close_sp = float(df_sp['Close'].iloc[-1])
    ema50_sp = float(ta.ema(df_sp['Close'], length=50).iloc[-1])

    if close_sp > ema50_sp:
        return {"regime": "BULLISH", "min_tech_score": 70}
    else:
        return {"regime": "BEARISH", "min_tech_score": 78}

# ------------------------------------------------------------------------------
# 6. מחשב ניקוד טכני מלא (Technical Score Engine)
# ------------------------------------------------------------------------------
def calculate_technical_score(df: pd.DataFrame, ticker: yf.Ticker, live_price: Optional[float] = None) -> dict:
    close_price = float(df['Close'].iloc[-1])
    entry_price = live_price if (live_price and live_price > 0 and not np.isnan(live_price)) else close_price

    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    atr = float(df['ATR'].dropna().iloc[-1]) if not df['ATR'].dropna().empty else 1.0
    ema20 = float(df['EMA20'].dropna().iloc[-1])
    ema50 = float(df['EMA50'].dropna().iloc[-1])

    # 1. Market Structure (20 points)
    struct = analyze_market_structure(df)

    # 2. Breakout Quality (20 points)
    bk = detect_breakout_quality(df, entry_price, struct["resistance_level"], atr)

    # 3. Volume Confirmation (15 points)
    vol = analyze_volume_metrics(df)

    # 4. Momentum (15 points)
    mom = analyze_momentum_and_rsi(df)

    # 5. Relative Strength (10 points)
    df_bm = fetch_benchmark_data("SPY")
    rs = calculate_relative_strength(df, df_bm)

    # 6. Trend Structure (5 points)
    trend_score = 0
    if entry_price > ema20 > ema50:
        trend_score = 5
    elif entry_price > ema20:
        trend_score = 3

    # 7. Volatility Compression (5 points)
    comp = detect_volatility_compression(df)

    # 8. Multi-Timeframe (5 points)
    mtf = detect_multi_timeframe_confirmation(ticker, entry_price)

    # 9. Risk/Reward Rating (5 points)
    stop_loss = struct["recent_swing_low"] - (0.5 * atr)
    risk = entry_price - stop_loss
    tp1 = entry_price + (1.5 * risk)
    rr_ratio = (tp1 - entry_price) / max(risk, 0.01)
    rr_score = 5 if rr_ratio >= 2.0 else (3 if rr_ratio >= 1.5 else 0)

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

    # בדיקת התרחקות יתר (Extended Move Filter)
    distance_from_ema20 = ((entry_price - ema20) / ema20) * 100
    if distance_from_ema20 > 8.0:
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
        "distance_from_ema20": round(distance_from_ema20, 2)
    }

# ------------------------------------------------------------------------------
# 7. ניתוח חדשות ואימות תוכן (News Subsystem)
# ------------------------------------------------------------------------------
HIGH_IMPACT_CATALYSTS = {
    r"\bfda\b|\btrial\b|\bphase\b|\bclinical\b|\bapproval\b": ("אישור/ניסוי קליני (FDA/Pharma)", 25),
    r"\bearnings\b|\bbeat\b|\brevenue beat\b|\brecord revenue\b": ("דוחות כספיים / תוצאות שיא 📈", 20),
    r"\bguidance\b|\braises outlook\b|\braised guidance\b": ("עדכון תחזית צמיחה כלפי מעלה 🚀", 20),
    r"\bmerger\b|\bacquisition\b|\bbuyout\b": ("עסקת מיזוג / רכישה דרמטית 🤝", 20),
    r"\bcontract\b|\bdeal\b|\bpartnership\b": ("חתימת חוזה אסטרטגי / הספקת ענק 📝", 15)
}

def analyze_news_catalyst(symbol: str) -> dict:
    company_name = KNOWN_COMPANIES_DICT.get(symbol, symbol.lower())
    headlines = []

    if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
        try:
            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
            to_date = today.strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
            res = requests.get(url, timeout=4)
            if res.status_code == 200:
                for item in res.json():
                    h = item.get("headline", "")
                    if h:
                        headlines.append(h)
        except Exception:
            pass

    if not headlines:
        try:
            news = yf.Ticker(symbol).news
            if news:
                for item in news:
                    title = item.get("title", "")
                    if title:
                        headlines.append(title)
        except Exception:
            pass

    news_score = 0
    catalyst_label = "ללא קטליזטור חדשותי דרמטי"
    valid_headline = ""

    for h in headlines[:5]:
        h_lower = h.lower()
        for pattern, (label, score) in HIGH_IMPACT_CATALYSTS.items():
            if re.search(pattern, h_lower):
                news_score = score
                catalyst_label = label
                valid_headline = h
                break
        if news_score > 0:
            break

    return {
        "news_score": news_score,
        "catalyst_label": catalyst_label,
        "headline": valid_headline
    }

# ------------------------------------------------------------------------------
# 8. מחולל איתותים אחיד (Unified Signal Generator for Live & Backtest)
# ------------------------------------------------------------------------------
def generate_signal(symbol: str, df: pd.DataFrame = None, live_price: float = None) -> SignalResult:
    rejection_reasons = []

    try:
        ticker = yf.Ticker(symbol)
        if df is None or df.empty:
            df = ticker.history(period="1y")

        if df.empty or len(df) < 60:
            return SignalResult(symbol, "NO_SETUP", 0, 0, False, ["אין מספיק נתונים היסטוריים"], None, {}, {}, "", datetime.datetime.now())

        fast_info = getattr(ticker, 'fast_info', {})
        if live_price is None:
            live_price = fast_info.get('lastPrice', None)

        # 1. בדיקות נזילות (Liquidity Filter)
        close_p = float(df['Close'].iloc[-1])
        avg_vol = float(df['Volume'].iloc[-20:].mean())
        dollar_vol = close_p * avg_vol

        if close_p < 5.0:
            rejection_reasons.append("מחיר מניה נמוך מ-$5 (Penny Stock)")
        if avg_vol < 150000:
            rejection_reasons.append("נפח מסחר יומי ממוצע נמוך מ-150,000 מניות")
        if dollar_vol < 1000000:
            rejection_reasons.append("נפח כספי יומי נמוך מ-$1,000,000")

        # 2. חישוב Technical Score
        tech = calculate_technical_score(df, ticker, live_price)
        tech_score = tech["technical_score"]

        # 3. ניתוח חדשות ו-Composite Score
        news = analyze_news_catalyst(symbol)
        composite_score = (tech_score * 0.70) + (news["news_score"] * 0.30)

        # 4. זיהוי תבניות למתן Bonus הוכחות בלבד
        avg_vol_20 = float(df['Volume'].iloc[-21:-1].mean())
        vol_spike = float(df['Volume'].iloc[-1]) > (avg_vol_20 * 1.4)
        found_patterns = []
        for detector in PATTERN_DETECTORS:
            p_res = detector(df, tech["entry_price"], vol_spike)
            if p_res.is_valid and p_res.is_bullish:
                found_patterns.append(p_res.label)

        # 5. זיהוי תנאי פריצה וסיווג Setup State
        bk_confirmed = tech["breakout_details"]["breakout_confirmed"]
        is_breakout = tech["breakout_details"]["is_breakout"]

        if bk_confirmed:
            setup_state = "TRIGGERED"
        elif is_breakout or len(found_patterns) > 0 or tech["structure"]["trend_structure"] == "BULLISH":
            setup_state = "READY"
        else:
            setup_state = "NO_SETUP"

        # 6. בדיקת Bearish Conflict Blocking
        rsi_overextended = tech["momentum_details"]["rsi_overextended"]
        market_regime = detect_market_regime()

        if rsi_overextended:
            rejection_reasons.append("קונפליקט דובי: RSI במצב קניות יתר (Overextended > 75)")
        if tech["distance_from_ema20"] > 8.0:
            rejection_reasons.append("קונפליקט דובי: המניה מתוחה מדי מעל EMA20")
        if tech["relative_strength"]["relative_return_20d"] < -3.0:
            rejection_reasons.append("קונפליקט דובי: תשואה יחסית שלילית מול השוק")

        # 7. בדיקת יחס סיכון/תשואה (R:R >= 1:2)
        entry = tech["entry_price"]
        sl = tech["stop_loss"]
        risk = entry - sl
        if risk <= 0:
            rejection_reasons.append("גובה Stop Loss אינו תקין")
            risk = 0.01

        tp1 = entry + (1.5 * risk)
        tp2 = entry + (2.5 * risk)
        rr_ratio = (tp1 - entry) / risk

        if rr_ratio < 1.5:
            rejection_reasons.append(f"יחס סיכון/תשואה נמוך מ-1:1.5 ({round(rr_ratio, 2)})")

        trade_plan = {
            "entry": round(entry, 2),
            "stop_loss": round(sl, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "risk_reward": round(rr_ratio, 2)
        }

        # 8. Mandatory BUY Conditions (תנאי חובה אבסולוטיים לשליחת BUY)
        min_required_tech = market_regime["min_tech_score"]
        is_buy = (
            setup_state == "TRIGGERED" and
            tech_score >= min_required_tech and
            composite_score >= 68.0 and
            tech["volume_details"]["volume_supports_price"] and
            len(rejection_reasons) == 0 and
            rr_ratio >= 1.5
        )

        if not is_buy and setup_state == "TRIGGERED":
            if tech_score < min_required_tech:
                rejection_reasons.append(f"ציון טכני ({round(tech_score, 1)}) נמוך מהסף הנדרש ({min_required_tech})")
            if not tech["volume_details"]["volume_supports_price"]:
                rejection_reasons.append("אין אישור נפח מסחר למהלך המחיר")

        # Fingerprint למניעת כפילויות
        fp_raw = f"{symbol}_{tech['breakout_details']['breakout_level']}_{datetime.date.today()}"
        fingerprint = hashlib.md5(fp_raw.encode()).hexdigest()

        tech["found_patterns"] = found_patterns
        tech["market_regime"] = market_regime["regime"]

        return SignalResult(
            symbol=symbol,
            setup_state=setup_state,
            technical_score=round(tech_score, 1),
            composite_score=round(composite_score, 1),
            is_buy=is_buy,
            rejection_reasons=rejection_reasons,
            trade_plan=trade_plan,
            tech_details=tech,
            news_details=news,
            fingerprint=fingerprint,
            timestamp=datetime.datetime.now()
        )

    except Exception as e:
        return SignalResult(symbol, "NO_SETUP", 0, 0, False, [f"שגיאה בניתוח: {e}"], None, {}, {}, "", datetime.datetime.now())

# ------------------------------------------------------------------------------
# 9. עיצוב הודעת איתות סלקטיבית (Clear Alert Message)
# ------------------------------------------------------------------------------
def build_alert_message(sig: SignalResult) -> Tuple[str, InlineKeyboardMarkup]:
    plan = sig.trade_plan
    tech = sig.tech_details
    news = sig.news_details

    headline_tr = news["headline"]
    if headline_tr:
        try:
            headline_tr = translator.translate(headline_tr)
        except Exception:
            pass

    msg = f"""🟢 <b>HIGH CONVICTION BUY</b>

<b>SYMBOL: {sig.symbol}</b>
<b>Score: {sig.technical_score}/100</b>

<b>למה עכשיו:</b>
פריצה מאושרת של רמת התנגדות <code>${tech['breakout_details']['breakout_level']}</code> במבנה {tech['structure']['trend_structure']}, עם RVOL של {tech['volume_details']['rvol']}x וחוזק יחסי למדד (+{tech['relative_strength']['relative_return_20d']}%).

<b>Breakout Level:</b> <code>${tech['breakout_details']['breakout_level']}</code>
<b>Entry:</b> <code>${plan['entry']}</code>
<b>Stop Loss:</b> <code>${plan['stop_loss']}</code>
<b>TP1:</b> <code>${plan['tp1']}</code>
<b>TP2:</b> <code>${plan['tp2']}</code>

<b>R:R:</b> <code>1:{plan['risk_reward']}</code>

<b>Catalyst:</b>
{news['catalyst_label']}
<i>{headline_tr if headline_tr else ''}</i>

<b>Signal Age:</b> Fresh
"""

    markup = InlineKeyboardMarkup()
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף", url=f"https://www.tradingview.com/chart/?symbol={sig.symbol}")
    markup.add(btn_chart)

    return msg, markup

# ------------------------------------------------------------------------------
# 10. ארכיטקטורת סריקה חדשה (Pipeline: SCAN -> FILTER -> RANK -> DEDUP -> SEND)
# ------------------------------------------------------------------------------
def is_market_open() -> bool:
    us_tz = pytz.timezone('America/New_York')
    now_us = datetime.datetime.now(us_tz)
    if now_us.weekday() in (5, 6):
        return False
    market_start = now_us.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = now_us.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_start <= now_us <= market_end

def scan_symbol_worker(symbol: str) -> Optional[SignalResult]:
    time.sleep(0.05)
    sig = generate_signal(symbol)
    if sig.is_buy:
        print(f"[LOG] Candidate Found: {symbol} | Tech Score: {sig.technical_score}")
        return sig
    else:
        if sig.rejection_reasons:
            reasons_str = ", ".join(sig.rejection_reasons)
            print(f"[REJECTED] {symbol} | Tech Score: {sig.technical_score} | Reasons: {reasons_str}")
        return None

def execute_global_market_scan():
    if count_today_alerts() >= MAX_ALERTS_PER_DAY:
        print("ℹ️ הגעת למכסת ההתראות היומית המרבית. הסריקה הופסקה.")
        return

    print(f"[{datetime.datetime.now()}] 🔄 מתחיל סריקת שוק גלובלית (Precision Engine)...")
    tickers = fetch_market_tickers()
    candidates: List[SignalResult] = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(scan_symbol_worker, tickers)
        for sig in results:
            if sig is not None:
                candidates.append(sig)

    if not candidates:
        print("ℹ️ לא אותרו מניות המקיימות את תנאי HIGH CONVICTION BUY בסיבוב זה.")
        return

    # global ranking לפי Technical Score
    candidates.sort(key=lambda x: x.technical_score, reverse=True)

    sent_count = 0
    users = get_all_users()

    for sig in candidates:
        if sent_count >= MAX_ALERTS_PER_SCAN:
            break
        if count_today_alerts() >= MAX_ALERTS_PER_DAY:
            break

        if is_signal_in_cooldown(sig.fingerprint):
            print(f"[DEDUP] {sig.symbol} נמצא ב-Cooldown. לא יישלח שוב.")
            continue

        msg, markup = build_alert_message(sig)
        for chat_id in users:
            try:
                bot.send_message(chat_id, msg, parse_mode="HTML", reply_markup=markup)
            except Exception as e:
                print(f"[Alert Send Error] {chat_id}: {e}")

        record_sent_signal(sig)
        sent_count += 1

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(execute_global_market_scan, 'interval', minutes=15)
scheduler.add_job(update_alert_outcomes_job, 'cron', hour=23, minute=30)
scheduler.start()

# ------------------------------------------------------------------------------
# 11. פקודות TELEGRAM BOT & BACKTEST SYSTEM
# ------------------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def cmd_start(message):
    add_user(message.chat.id)
    bot.reply_to(
        message,
        "<b>מערכת איתותי איכות (Precision over Recall v4) מחוברת! 🚀</b>\n\n"
        "• איתותי HIGH CONVICTION בלבד\n"
        "• בדיקת מניה נקודתית: <code>/tech SYMBOL</code>\n"
        "• הרצת Backtest זהה: <code>/backtest SYMBOL</code>\n"
        "• סריקה ידונית: <code>/scan</code>",
        parse_mode="HTML"
    )

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    add_user(message.chat.id)
    bot.reply_to(message, "🔍 מריץ סריקה גלובלית ודירוג מועמדים...", parse_mode="HTML")
    execute_global_market_scan()

@bot.message_handler(commands=['tech'])
def cmd_tech(message):
    add_user(message.chat.id)
    try:
        symbol = message.text.split()[1].upper()
        sig = generate_signal(symbol)

        if sig.is_buy:
            msg, markup = build_alert_message(sig)
            bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
        else:
            reasons = "\n".join([f"• {r}" for r in sig.rejection_reasons]) if sig.rejection_reasons else "• לא התקיימה פריצה מאושרת"
            reply = f"""<b>📊 תוצאת ניתוח עבור {symbol}</b>
            
<b>State:</b> <code>{sig.setup_state}</code>
<b>Technical Score:</b> <code>{sig.technical_score}/100</code>
<b>Composite Score:</b> <code>{sig.composite_score}/100</code>

❌ <b>הערכתBUY נדחתה עקב:</b>
{reasons}
"""
            bot.send_message(message.chat.id, reply, parse_mode="HTML")
    except IndexError:
        bot.reply_to(message, "⚠️ נא לציין סימול מניה: <code>/tech AAPL</code>", parse_mode="HTML")

@bot.message_handler(commands=['backtest'])
def cmd_backtest(message):
    try:
        symbol = message.text.split()[1].upper()
        bot.reply_to(message, f"⏳ מריץ Backtest היסטורי רציף עבור {symbol}...", parse_mode="HTML")

        df = yf.Ticker(symbol).history(period="2y")
        if len(df) < 150:
            bot.send_message(message.chat.id, f"❌ אין מספיק נתונים עבור {symbol}.")
            return

        total_signals = 0
        tp1_hits = 0
        sl_hits = 0

        for i in range(100, len(df) - 15):
            sub_df = df.iloc[:i]
            curr_price = float(sub_df['Close'].iloc[-1])

            sig = generate_signal(symbol, df=sub_df, live_price=curr_price)

            if sig.is_buy and sig.trade_plan:
                total_signals += 1
                target_tp1 = sig.trade_plan['tp1']
                target_sl = sig.trade_plan['stop_loss']

                future_df = df.iloc[i:i+15]
                hit_tp = (future_df['High'] >= target_tp1).any()
                hit_sl = (future_df['Low'] <= target_sl).any()

                if hit_tp:
                    tp1_hits += 1
                elif hit_sl:
                    sl_hits += 1

        win_rate = round((tp1_hits / max(total_signals, 1)) * 100, 1)
        reply = (
            f"<b>🔬 תוצאות Backtest אחיד עבור {symbol} (שנתיים אחורה):</b>\n\n"
            f"• סה\"כ איתותי BUY שיוצרו: <code>{total_signals}</code>\n"
            f"• פגיעות ביעד (TP1): <code>{tp1_hits}</code>\n"
            f"• פגיעות בסטופ (SL): <code>{sl_hits}</code>\n"
            f"• 🏆 <b>אחוז הצלחה: {win_rate}%</b>"
        )
        bot.send_message(message.chat.id, reply, parse_mode="HTML")
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ שגיאה בהרצת Backtest: {e}")

# ------------------------------------------------------------------------------
# 12. UNIT TESTS
# ------------------------------------------------------------------------------
def run_unit_tests():
    print("🧪 מריץ Unit Tests מקיפים למנוע הניתוח...")
    dates = pd.date_range(start="2023-01-01", periods=100)

    # 1. Strong Breakout Test
    data = {
        'Open': [100.0]*98 + [100.0, 102.0],
        'High': [101.0]*98 + [101.0, 108.0],
        'Low': [99.0]*98 + [99.5, 101.5],
        'Close': [100.0]*98 + [100.5, 107.5],
        'Volume': [1000000]*98 + [1000000, 3500000]
    }
    df_test = pd.DataFrame(data, index=dates)
    struct = analyze_market_structure(df_test)
    bk = detect_breakout_quality(df_test, 107.5, 101.0, 2.0)
    assert bk["is_breakout"] and bk["breakout_confirmed"], "Unit Test Failed: Breakout Quality"

    print("✅ כל בדיקות היחידה (Unit Tests) עברו בהצלחה!")

# ------------------------------------------------------------------------------
# 13. FLASK & KEEP-ALIVE
# ------------------------------------------------------------------------------
@app.route('/')
def health_check():
    return "OK - Precision Trading Engine Active", 200

def keep_alive_ping():
    while True:
        try:
            time.sleep(600)
            if "localhost" not in SELF_URL:
                requests.get(SELF_URL, timeout=10)
        except Exception:
            pass

if __name__ == "__main__":
    run_unit_tests()
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()

    print("🤖 Precision Trading Engine Telegram Bot Is Ready...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
