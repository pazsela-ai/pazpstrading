import os
import time
import sqlite3
import threading
import requests
import datetime
import pytz
import re
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
# 1. הגדרות סביבה ובטיחות תהליכונים (Thread Safety)
# ------------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
PORT = int(os.environ.get("PORT", 5000))
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")

bot = TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)
translator = GoogleTranslator(source='auto', target='iw')

# מנגנוני נעילה לגישה בטוחה מרובת תהליכונים
CACHE_LOCK = threading.Lock()
DB_LOCK = threading.Lock()

CACHE: Dict[str, Tuple[float, dict]] = {}
CACHE_TTL = 300  # 5 דקות

KNOWN_TICKERS_SET = set()
KNOWN_COMPANIES_DICT = {}

# ------------------------------------------------------------------------------
# 2. מודל נתונים ואחדות ניקוד תבניות (Pattern Registry Framework)
# ------------------------------------------------------------------------------
@dataclass
class PatternResult:
    score: int            # ציון ביטחון (0 עד 100)
    is_valid: bool        # האם התבנית עברה את סף האישור
    label: str            # שם התבנית לתצוגה
    is_bullish: bool      # True = איתות שורי, False = איתות דובי
    meta: dict            # נתונים גיאומטריים של התבנית

# ------------------------------------------------------------------------------
# 3. פונקציות זיהוי תבניות עצמאיות (Pattern Detectors)
# ------------------------------------------------------------------------------
def detect_hammer(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """נר פטיש (Hammer) היפוכי - ניקוד דינמי"""
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
        if l1 <= df['Low'].iloc[-20:].min():  # בתחתית מקומית
            score += 10

    return PatternResult(
        score=score,
        is_valid=score >= 70,
        label=f"נר פטיש היפוכי (Hammer) [ביטחון: {score}%]",
        is_bullish=True,
        meta={"ratio": round(ratio, 2), "lower_shadow": lower_shadow}
    )

def detect_bullish_engulfing(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """בליעה שורית (Bullish Engulfing) - ניקוד דינמי"""
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
    if c1 > df['High'].iloc[-5:-1].max():
        score += 10

    return PatternResult(
        score=score,
        is_valid=score >= 70,
        label=f"בליעה שורית (Bullish Engulfing) [ביטחון: {score}%]",
        is_bullish=True,
        meta={"body_ratio": round(body1 / max(body2, 0.01), 2)}
    )

def detect_cup_and_handle(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """ספל וידית (Cup & Handle) - כולל בדיקת אורך בסיס ובדיקת ידית קפדנית"""
    if len(df) < 60:
        return PatternResult(0, False, "ספל וידית", True, {})

    high_60 = float(df['High'].iloc[-60:-15].max())
    cup_bottom = float(df['Low'].iloc[-60:-10].min())
    cup_depth = (high_60 - cup_bottom) / high_60

    if not (0.08 <= cup_depth <= 0.38):
        return PatternResult(0, False, "ספל וידית", True, {})

    # בדיקת אורך בסיס הספל (מספר הנרות בין השיא לתחתית)
    idx_high = df['High'].iloc[-60:-15].idxmax()
    idx_low = df['Low'].iloc[-60:-10].idxmin()
    base_length = abs((idx_low - idx_high).days) if hasattr((idx_low - idx_high), 'days') else 20

    # בדיקת הידית (Handle): 5 עד 12 נרות אחרונים של דשדוש בנפח יורד
    handle_df = df.iloc[-12:-2]
    handle_vol_avg = handle_df['Volume'].mean()
    prior_vol_avg = df['Volume'].iloc[-30:-12].mean()
    
    handle_retrace = (high_60 - handle_df['Low'].min()) / high_60
    has_valid_handle = (handle_retrace <= (cup_depth * 0.5)) and (handle_vol_avg < prior_vol_avg)

    # פריצה
    prev_price = float(df['Close'].iloc[-2])
    is_breakout = (current_price > high_60) and (prev_price <= high_60)

    if not is_breakout:
        return PatternResult(0, False, "ספל וידית", True, {})

    score = 50
    if has_valid_handle:
        score += 25
    if base_length >= 15:
        score += 15
    if volume_spike:
        score += 10

    return PatternResult(
        score=score,
        is_valid=score >= 70,
        label=f"פריצת ספל וידית (Cup & Handle) [ביטחון: {score}%]",
        is_bullish=True,
        meta={"cup_depth": round(cup_depth, 2), "base_days": base_length, "has_handle": has_valid_handle}
    )

def detect_double_bottom(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """תחתית כפולה (Double Bottom)"""
    lows = df['Low'].iloc[-50:-10]
    first_bottom = float(lows.min())
    second_candidates = df['Low'].iloc[-20:-2]
    
    if second_candidates.empty:
        return PatternResult(0, False, "תחתית כפולה", True, {})

    second_bottom = float(second_candidates.min())
    diff = abs(first_bottom - second_bottom) / first_bottom

    if diff > 0.025:
        return PatternResult(0, False, "תחתית כפולה", True, {})

    neckline = float(df['High'].iloc[-40:-5].max())
    prev_price = float(df['Close'].iloc[-2])
    is_breakout = (current_price > neckline) and (prev_price <= neckline)

    if not is_breakout:
        return PatternResult(0, False, "תחתית כפולה", True, {})

    score = 60
    if volume_spike:
        score += 20
    if diff < 0.01:
        score += 20

    return PatternResult(
        score=score,
        is_valid=score >= 70,
        label=f"פריצת תחתית כפולה (Double Bottom) [ביטחון: {score}%]",
        is_bullish=True,
        meta={"neckline": neckline, "diff_pct": round(diff * 100, 2)}
    )

def detect_double_top(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """פסגה כפולה (Double Top) - איתות דובי"""
    highs = df['High'].iloc[-50:-10]
    first_top = float(highs.max())
    second_candidates = df['High'].iloc[-20:-2]
    
    if second_candidates.empty:
        return PatternResult(0, False, "פסגה כפולה", False, {})

    second_top = float(second_candidates.max())
    diff = abs(first_top - second_top) / first_top

    if diff > 0.025:
        return PatternResult(0, False, "פסגה כפולה", False, {})

    support_line = float(df['Low'].iloc[-40:-5].min())
    prev_price = float(df['Close'].iloc[-2])
    is_breakdown = (current_price < support_line) and (prev_price >= support_line)

    if not is_breakdown:
        return PatternResult(0, False, "פסגה כפולה", False, {})

    score = 65
    if volume_spike:
        score += 20
    if diff < 0.01:
        score += 15

    return PatternResult(
        score=score,
        is_valid=score >= 70,
        label=f"שבירת פסגה כפולה (Double Top) [ביטחון: {score}%]",
        is_bullish=False,
        meta={"support_line": support_line, "diff_pct": round(diff * 100, 2)}
    )

def detect_ma_crosses(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """חיתוכי ממוצעים נעים (Golden Cross / Death Cross) בחלון של 5 ימים"""
    if 'EMA50' not in df.columns or 'EMA200' not in df.columns:
        return PatternResult(0, False, "חיתוך ממוצעים", True, {})

    ema50 = df['EMA50'].dropna()
    ema200 = df['EMA200'].dropna()

    if len(ema50) < 5 or len(ema200) < 5:
        return PatternResult(0, False, "חיתוך ממוצעים", True, {})

    # חיתוך מוזהב ב-5 הנרות האחרונים
    golden_cross = (ema50.iloc[-5] < ema200.iloc[-5]) and (ema50.iloc[-1] > ema200.iloc[-1])
    death_cross = (ema50.iloc[-5] > ema200.iloc[-5]) and (ema50.iloc[-1] < ema200.iloc[-1])

    if golden_cross:
        return PatternResult(
            score=90,
            is_valid=True,
            label="חיתוך מוזהב (Golden Cross - EMA50 חצה מעל EMA200) [ביטחון: 90%]",
            is_bullish=True,
            meta={"ema50": round(float(ema50.iloc[-1]), 2), "ema200": round(float(ema200.iloc[-1]), 2)}
        )
    elif death_cross:
        return PatternResult(
            score=90,
            is_valid=True,
            label="חיתוך מוות (Death Cross - EMA50 חצה מתחת ל-EMA200) [ביטחון: 90%]",
            is_bullish=False,
            meta={"ema50": round(float(ema50.iloc[-1]), 2), "ema200": round(float(ema200.iloc[-1]), 2)}
        )

    return PatternResult(0, False, "חיתוך ממוצעים", True, {})

def detect_macd_cross(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """חיתוך קו MACD ואינדיקטור Signal"""
    macd_df = ta.macd(df['Close'])
    if macd_df is None or macd_df.empty:
        return PatternResult(0, False, "MACD Cross", True, {})

    macd_line = macd_df.iloc[:, 0]
    signal_line = macd_df.iloc[:, 2]

    bull_cross = (macd_line.iloc[-2] < signal_line.iloc[-2]) and (macd_line.iloc[-1] > signal_line.iloc[-1])
    bear_cross = (macd_line.iloc[-2] > signal_line.iloc[-2]) and (macd_line.iloc[-1] < signal_line.iloc[-1])

    if bull_cross:
        score = 75 + (10 if volume_spike else 0)
        return PatternResult(
            score=score,
            is_valid=score >= 70,
            label=f"חיתוך MACD שורי (MACD Bullish Crossover) [ביטחון: {score}%]",
            is_bullish=True,
            meta={"macd": round(float(macd_line.iloc[-1]), 3)}
        )
    elif bear_cross:
        score = 75 + (10 if volume_spike else 0)
        return PatternResult(
            score=score,
            is_valid=score >= 70,
            label=f"חיתוך MACD דובי (MACD Bearish Crossover) [ביטחון: {score}%]",
            is_bullish=False,
            meta={"macd": round(float(macd_line.iloc[-1]), 3)}
        )

    return PatternResult(0, False, "MACD Cross", True, {})

def detect_triangles(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """זיהוי משולשים (Ascending/Descending/Symmetrical) באמצעות רגרסיה ליניארית"""
    if len(df) < 30:
        return PatternResult(0, False, "משולש", True, {})

    sub_df = df.iloc[-30:]
    x = np.arange(len(sub_df))
    highs = sub_df['High'].values
    lows = sub_df['Low'].values

    slope_high, _ = np.polyfit(x, highs, 1)
    slope_low, _ = np.polyfit(x, lows, 1)

    prev_price = float(df['Close'].iloc[-2])
    res_line = float(highs[-1])

    # משולש עולה: התנגדות אופקית (שיפוע קרוב ל-0) ותמיכה עולה
    if abs(slope_high) < 0.05 and slope_low > 0.1:
        if current_price > res_line and prev_price <= res_line:
            score = 70 + (15 if volume_spike else 0)
            return PatternResult(
                score=score,
                is_valid=score >= 70,
                label=f"פריצת משולש עולה (Ascending Triangle) [ביטחון: {score}%]",
                is_bullish=True,
                meta={"slope_low": round(slope_low, 3)}
            )

    return PatternResult(0, False, "משולש", True, {})

def detect_flags_pennants(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """תבניות המשך מגמה: דגל / מטוס נייר (Flag/Pennant)"""
    if len(df) < 25:
        return PatternResult(0, False, "דגל", True, {})

    # זיהוי תורן: עלייה של מעל 6% ב-5 נרות בנפח גבוה
    pole_df = df.iloc[-20:-10]
    pole_return = (pole_df['Close'].iloc[-1] - pole_df['Close'].iloc[0]) / pole_df['Close'].iloc[0]

    if pole_return < 0.06:
        return PatternResult(0, False, "דגל", True, {})

    # דגל: דשדוש מתון 5-10 נרות בנפח יורד
    flag_df = df.iloc[-10:-1]
    flag_vol_avg = flag_df['Volume'].mean()
    pole_vol_avg = pole_df['Volume'].mean()

    if flag_vol_avg >= pole_vol_avg:
        return PatternResult(0, False, "דגל", True, {})

    flag_high = flag_df['High'].max()
    if current_price > flag_high:
        score = 75 + (15 if volume_spike else 0)
        return PatternResult(
            score=score,
            is_valid=score >= 70,
            label=f"פריצת תבנית דגל (Bullish Flag Breakout) [ביטחון: {score}%]",
            is_bullish=True,
            meta={"pole_return_pct": round(pole_return * 100, 1)}
        )

    return PatternResult(0, False, "דגל", True, {})

def detect_wedges(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """זיהוי טריז עולה/יורד (Rising/Falling Wedge)"""
    if len(df) < 30:
        return PatternResult(0, False, "טריז", True, {})

    sub_df = df.iloc[-30:]
    x = np.arange(len(sub_df))
    slope_high, _ = np.polyfit(x, sub_df['High'].values, 1)
    slope_low, _ = np.polyfit(x, sub_df['Low'].values, 1)

    # טריז יורד שורי: שני הקווים יורדים ומתכנסים, והמחיר פורץ כלפי מעלה
    if slope_high < -0.05 and slope_low < -0.05 and slope_high > slope_low:
        res_line = sub_df['High'].iloc[-3:-1].max()
        if current_price > res_line:
            score = 70 + (10 if volume_spike else 0)
            return PatternResult(
                score=score,
                is_valid=score >= 70,
                label=f"פריצת טריז יורד שורי (Falling Wedge) [ביטחון: {score}%]",
                is_bullish=True,
                meta={"slope_high": round(slope_high, 3)}
            )

    return PatternResult(0, False, "טריז", True, {})

def detect_head_and_shoulders(df: pd.DataFrame, current_price: float, volume_spike: bool) -> PatternResult:
    """ראש וכתפיים (Head & Shoulders + Inverse)"""
    if len(df) < 60:
        return PatternResult(0, False, "ראש וכתפיים", True, {})

    # זיהוי ראש וכתפיים הפוך (Inverse - שורי)
    lows = df['Low'].iloc[-50:-2]
    head_idx = lows.idxmin()
    head_val = lows.min()

    left_shoulder = df['Low'].loc[:head_idx].iloc[-15:-2].min() if len(df['Low'].loc[:head_idx]) > 15 else None
    right_shoulder = df['Low'].loc[head_idx:].iloc[2:15].min() if len(df['Low'].loc[head_idx:]) > 15 else None

    if left_shoulder and right_shoulder:
        if head_val < left_shoulder and head_val < right_shoulder:
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff <= 0.08:
                neckline = df['High'].loc[head_idx:].iloc[:15].max()
                if current_price > neckline:
                    score = 80 + (10 if volume_spike else 0)
                    return PatternResult(
                        score=score,
                        is_valid=score >= 70,
                        label=f"פריצת ראש וכתפיים הפוך (Inverse H&S) [ביטחון: {score}%]",
                        is_bullish=True,
                        meta={"neckline": neckline, "shoulder_diff_pct": round(shoulder_diff * 100, 1)}
                    )

    return PatternResult(0, False, "ראש וכתפיים", True, {})

# ------------------------------------------------------------------------------
# REGISTRY PATTERN REGISTER
# ------------------------------------------------------------------------------
PATTERN_DETECTORS = [
    detect_hammer,
    detect_bullish_engulfing,
    detect_cup_and_handle,
    detect_double_bottom,
    detect_double_top,
    detect_ma_crosses,
    detect_macd_cross,
    detect_triangles,
    detect_flags_pennants,
    detect_wedges,
    detect_head_and_shoulders
]

# ------------------------------------------------------------------------------
# 4. שליפת מניות וניהול מסד נתונים (SQLite + Outcome Tracking)
# ------------------------------------------------------------------------------
DB_FILE = "bot_database.db"

def init_db():
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS users (chat_id INTEGER PRIMARY KEY)")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_alerts (
                    symbol TEXT,
                    alert_date TEXT,
                    PRIMARY KEY (symbol, alert_date)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    alert_time TEXT,
                    entry_price REAL,
                    stop_loss REAL,
                    tp1 REAL,
                    tp2 REAL,
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

def has_alerted_today(symbol: str) -> bool:
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM daily_alerts WHERE symbol = ? AND alert_date = ?", (symbol, today_str))
            return cursor.fetchone() is not None

def mark_alerted_today(symbol: str):
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO daily_alerts (symbol, alert_date) VALUES (?, ?)", (symbol, today_str))
            conn.commit()

def log_alert_history(symbol: str, entry: float, sl: float, tp1: float, tp2: float):
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with DB_LOCK:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alert_history (symbol, alert_time, entry_price, stop_loss, tp1, tp2)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, now_str, entry, sl, tp1, tp2))
            conn.commit()

init_db()

def fetch_market_tickers() -> list:
    global KNOWN_TICKERS_SET, KNOWN_COMPANIES_DICT
    tickers = set()

    try:
        url_sp500 = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp = pd.read_csv(url_sp500)
        if 'Symbol' in df_sp.columns and 'Security' in df_sp.columns:
            for _, row in df_sp.iterrows():
                sym = str(row['Symbol']).replace('.', '-').strip()
                name = str(row['Security']).split()[0].lower()
                tickers.add(sym)
                KNOWN_COMPANIES_DICT[sym] = name
            KNOWN_TICKERS_SET = tickers
    except Exception as e:
        print(f"[Ticker Fetch Error]: {e}")

    if len(tickers) >= 100:
        return sorted(list(tickers))

    fallback = [
        "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", "META", "GOOGL", "NFLX",
        "LLY", "AVGO", "JPM", "UNH", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV"
    ]
    KNOWN_TICKERS_SET = set(fallback)
    return fallback

# ------------------------------------------------------------------------------
# 5. FLASK SERVER & KEEP-ALIVE
# ------------------------------------------------------------------------------
@app.route('/')
def health_check():
    return "OK - Trading System Engine v2 Active!", 200

def keep_alive_ping():
    while True:
        try:
            time.sleep(600)
            if "localhost" not in SELF_URL:
                requests.get(SELF_URL, timeout=10)
        except Exception as e:
            print(f"[Keep-Alive Error]: {e}")

# ------------------------------------------------------------------------------
# 6. מנוע ניתוח טכני מעודכן ומקור מחיר חי
# ------------------------------------------------------------------------------
def fetch_ticker_data_with_retry(symbol: str, retries: int = 3) -> Optional[yf.Ticker]:
    """שליפה מוגנת מרשת בגישת Retries ו-Rate Limit Control"""
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol)
            return ticker
        except Exception:
            time.sleep(1 + attempt)
    return None

def analyze_technical_patterns(symbol: str) -> Optional[dict]:
    now = time.time()
    with CACHE_LOCK:
        if symbol in CACHE:
            cached_time, cached_data = CACHE[symbol]
            if now - cached_time < CACHE_TTL:
                return cached_data

    try:
        ticker = fetch_ticker_data_with_retry(symbol)
        if not ticker:
            return None

        df = ticker.history(period="1y")
        if df.empty or len(df) < 100:
            return None

        # מועדף: מחיר בזמן אמת מ-fast_info
        fast_info = getattr(ticker, 'fast_info', {})
        live_price = fast_info.get('lastPrice', None)
        close_price = float(df['Close'].iloc[-1])

        if live_price and not np.isnan(live_price) and live_price > 0:
            entry_price = float(live_price)
            price_source = "מחיר בזמן אמת (Live)"
        else:
            entry_price = close_price
            price_source = "מחיר סגירה אחרון"

        price_discrepancy = abs(entry_price - close_price) / close_price
        price_warning = price_discrepancy > 0.008  # אזהרה מעל 0.8% פער

        data_timestamp = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')
        prev_price = float(df['Close'].iloc[-2])
        change_pct = ((entry_price - prev_price) / prev_price) * 100

        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        rsi = float(df['RSI'].dropna().iloc[-1]) if not df['RSI'].dropna().empty else 50.0
        ema20 = float(df['EMA20'].dropna().iloc[-1])
        ema50 = float(df['EMA50'].dropna().iloc[-1])
        ema200 = float(df['EMA200'].dropna().iloc[-1]) if not df['EMA200'].dropna().empty else ema50
        atr = float(df['ATR'].dropna().iloc[-1])

        is_uptrend = entry_price > ema20 > ema50 and entry_price > ema200

        avg_vol_20 = df['Volume'].iloc[-21:-1].mean()
        curr_vol = df['Volume'].iloc[-1]
        volume_spike = curr_vol > (avg_vol_20 * 1.4)

        # הרצת כל התבניות מה-Registry
        detected_patterns = []
        has_bullish_breakout = False

        for detector in PATTERN_DETECTORS:
            res: PatternResult = detector(df, entry_price, volume_spike)
            if res.is_valid:
                detected_patterns.append(res.label)
                if res.is_bullish:
                    has_bullish_breakout = True

        stop_loss = round(entry_price - (1.5 * atr), 2)

        result = {
            "symbol": symbol,
            "data_timestamp": data_timestamp,
            "price_source": price_source,
            "price_warning": price_warning,
            "current_price": round(entry_price, 2),
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi, 1),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "ema200": round(ema200, 2),
            "volume_accumulating": volume_spike,
            "patterns": detected_patterns,
            "has_breakout": has_bullish_breakout,
            "is_uptrend": is_uptrend,
            "rsi_valid": 45 <= rsi <= 68,
            "entry_price": round(entry_price, 2),
            "stop_loss": stop_loss,
            "atr": round(atr, 2)
        }

        with CACHE_LOCK:
            CACHE[symbol] = (now, result)
        return result

    except Exception as e:
        print(f"[Tech Analysis Error] {symbol}: {e}")
        return None

# ------------------------------------------------------------------------------
# 7. ניתוח חדשות ואימות תוכן קפדני (Entity Detection & Anti-Contamination)
# ------------------------------------------------------------------------------
HIGH_IMPACT_CATALYSTS = {
    r"\bfda\b|\btrial\b|\bphase\b|\bclinical\b|\bapproval\b": ("אישור/ניסוי קליני (FDA/Pharma)", 10),
    r"\bearnings\b|\bbeat\b|\brevenue beat\b|\brecord revenue\b": ("דוחות כספיים / תוצאות שיא 📈", 9),
    r"\bguidance\b|\braises outlook\b|\braised guidance\b": ("עדכון תחזית צמיחה כלפי מעלה 🚀", 9),
    r"\bmerger\b|\bacquisition\b|\bbuyout\b": ("עסקת מיזוג / רכישה דרמטית 🤝", 9),
    r"\bcontract\b|\bdeal\b|\bpartnership\b": ("חתימת חוזה אסטרטגי / הספקת ענק 📝", 8),
    r"\bshare buyback\b|\brepurchase program\b": ("תוכנית רכישה עצמית (Buyback) 💵", 8)
}

INVALID_PATTERNS = [
    r"error 50x", r"server error", r"404 not found", r"403 forbidden",
    r"that's an error", r"please try again later", r"<html", r"javascript"
]

EXCLUDED_WORDS_SET = {"AI", "CEO", "IPO", "CFO", "SEC", "FDA", "US", "USA", "ETF", "FED", "GDP"}

def is_valid_news_text(text: str) -> bool:
    if not text or len(text.strip()) < 15:
        return False
    text_lower = text.lower()
    return not any(err in text_lower for err in INVALID_PATTERNS)

def is_headline_relevant_and_fresh(headline: str, symbol: str, company_name: str, pub_date: Optional[datetime.datetime]) -> Tuple[int, Optional[str], bool]:
    if not is_valid_news_text(headline):
        return 0, None, False

    if pub_date:
        now_utc = datetime.datetime.now(pytz.utc)
        if (now_utc - pub_date).total_seconds() > 172800:
            return 0, None, False

    h_lower = headline.lower()
    sym_lower = symbol.lower()
    comp_lower = company_name.lower() if company_name else sym_lower

    has_exact_symbol = bool(re.search(r'\b' + re.escape(sym_lower) + r'\b', h_lower))
    has_company_name = len(comp_lower) > 3 and comp_lower in h_lower

    if not (has_exact_symbol or has_company_name):
        return 0, None, False

    # זיהוי ישויות קפדני המונע False Positives ממילים כגון AI, CEO
    tokens = set(re.findall(r'\b[A-Z]{2,5}\b', headline))
    valid_other_tickers = set()
    for t in tokens:
        if t not in EXCLUDED_WORDS_SET and t in KNOWN_TICKERS_SET and t != symbol:
            valid_other_tickers.add(t)

    is_multi_company = len(valid_other_tickers) > 0

    for pattern, (label, score) in HIGH_IMPACT_CATALYSTS.items():
        if re.search(pattern, h_lower):
            final_score = score - 2 if is_multi_company else score
            final_label = f"{label} (אזכור משני)" if is_multi_company else label
            return final_score, final_label, not is_multi_company

    return 0, None, False

def fetch_finnhub_data(symbol: str) -> dict:
    raw_articles = []
    company_name = KNOWN_COMPANIES_DICT.get(symbol, symbol.lower())

    if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
        try:
            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
            to_date = today.strftime('%Y-%m-%d')
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
            res = requests.get(news_url, timeout=5)
            if res.status_code == 200:
                for item in res.json():
                    h = item.get("headline")
                    ts = item.get("datetime")
                    pub_dt = datetime.datetime.fromtimestamp(ts, tz=pytz.utc) if ts else None
                    if h and is_valid_news_text(h):
                        raw_articles.append((h, pub_dt))
        except Exception as e:
            print(f"[Finnhub Error] {symbol}: {e}")

    if not raw_articles:
        try:
            t = yf.Ticker(symbol)
            news_items = t.news
            if news_items:
                for item in news_items:
                    title = item.get("title", "")
                    if not title and "content" in item and isinstance(item["content"], dict):
                        title = item["content"].get("title", "")
                    pub_ts = item.get("providerPublishTime")
                    pub_dt = datetime.datetime.fromtimestamp(pub_ts, tz=pytz.utc) if pub_ts else None
                    if title and is_valid_news_text(title):
                        raw_articles.append((title, pub_dt))
        except Exception as e:
            print(f"[YFinance News Error] {symbol}: {e}")

    scored_headlines = []
    found_catalysts = set()
    dedicated_catalyst_found = False

    for headline, pub_date in raw_articles:
        score, catalyst_label, is_dedicated = is_headline_relevant_and_fresh(headline, symbol, company_name, pub_date)
        if score >= 6:
            scored_headlines.append((score, headline))
            found_catalysts.add(catalyst_label)
            if is_dedicated:
                dedicated_catalyst_found = True

    scored_headlines.sort(key=lambda x: x[0], reverse=True)
    top_headlines = [h[1] for h in scored_headlines[:2]]

    translated_headlines = []
    for h in top_headlines:
        try:
            translated_headlines.append(translator.translate(h))
        except Exception:
            translated_headlines.append(h)

    catalyst_str = " | ".join(found_catalysts) if found_catalysts else "לא אותרו קטליזטורים דרמטיים ב-48 השעות האחרונות"
    sentiment = "חיובי חזק 🟢" if dedicated_catalyst_found else ("ניטרלי-חיובי 🟡" if found_catalysts else "ללא חדשות מהותיות ⚪")

    return {
        "headlines": translated_headlines,
        "catalyst": catalyst_str,
        "sentiment": sentiment,
        "has_valid_news": dedicated_catalyst_found
    }

# ------------------------------------------------------------------------------
# 8. ניהול סיכונים ובניית הדוח המאוחד
# ------------------------------------------------------------------------------
def calculate_trade_plan(entry_price: float, stop_loss: float) -> dict:
    risk_per_share = entry_price - stop_loss
    tp1_price = entry_price + (risk_per_share * 1.5)
    tp2_price = entry_price + (risk_per_share * 2.5)
    risk_reward_ratio = round((tp1_price - entry_price) / max(risk_per_share, 0.01), 2)

    return {
        "entry": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "tp1": round(tp1_price, 2),
        "tp1_pct": round(((tp1_price - entry_price) / entry_price) * 100, 1),
        "tp2": round(tp2_price, 2),
        "tp2_pct": round(((tp2_price - entry_price) / entry_price) * 100, 1),
        "risk_reward": risk_reward_ratio,
        "max_position_pct": "2-4% מתיק ההשקעות"
    }

def create_report_message(symbol: str) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
    tech = analyze_technical_patterns(symbol)
    if not tech:
        return f"❌ לא ניתן היה לשלוף נתונים עבור המנייה <b>{symbol}</b>.", None

    finnhub = fetch_finnhub_data(symbol)

    checks = [
        tech["is_uptrend"],
        tech["rsi_valid"],
        tech["has_breakout"] or tech["volume_accumulating"],
        finnhub["has_valid_news"]
    ]
    supported_checks = sum(1 for c in checks if c)

    if supported_checks >= 4:
        recommendation_level = "🟢 <b>המלצת קנייה חזקה (4/4 אינדיקטורים תומכים)</b>"
        signal_type = "BUY"
    elif supported_checks == 3:
        recommendation_level = "🟡 <b>המלצת קנייה בינונית (3/4 אינדיקטורים תומכים - זהירות)</b>"
        signal_type = "BUY"
    elif supported_checks == 2:
        recommendation_level = "🟠 <b>איתות חלש / בהתהוות (2/4 אינדיקטורים - למעקב בלבד)</b>"
        signal_type = "HOLD"
    else:
        recommendation_level = "🔴 <b>לא מומלץ כעת (פחות מ-2 אינדיקטורים תומכים)</b>"
        signal_type = "NEUTRAL"

    patterns_str = ", ".join(tech["patterns"]) if tech["patterns"] else "לא זוהו תבניות מיוחדות"
    headlines_str = "\n".join([f"• {h}" for h in finnhub["headlines"]]) if finnhub["headlines"] else "• לא אותרו כותרות איכותיות ב-48h האחרונות"
    warning_text = "\n⚠️ <b>אזהרה:</b> זוהה פער בין מקור המחיר החי למחיר הסגירה. לוודא מחיר פתיחה.\n" if tech["price_warning"] else ""

    msg = f"""
<b>📊 דוח ניתוח מקיף עבור {symbol}</b>
<i>זמן עדכון: {tech['data_timestamp']} | מקור: {tech['price_source']}</i>
{warning_text}
<b>💡 דירוג המלצה:</b>
{recommendation_level}
<b>שקיפות מדדים:</b> {supported_checks}/4 קטגוריות תומכות באיתות.

---
<b>📈 נתונים טכניים ותבניות שנמצאו:</b>
• מחיר נוכחי: <code>${tech['current_price']}</code> ({'+' if tech['change_pct']>0 else ''}{tech['change_pct']}%)
• RSI: <code>{tech['rsi']}</code> | EMA20: <code>${tech['ema20']}</code> | EMA50: <code>${tech['ema50']}</code>
• ATR (14D): <code>${tech['atr']}</code>
• <b>תבניות שזוהו:</b> {patterns_str}
• נפח מסחר חורג (Volume Spike): {'כן 🟢' if tech['volume_accumulating'] else 'רגיל ⚪'}

---
<b>📰 חדשות וקטליזטורים מסוננים:</b>
• קטליזטור: <b>{finnhub['catalyst']}</b>
• סנטימנט: <b>{finnhub['sentiment']}</b>
<b>כותרות נבחרות:</b>
{headlines_str}

---
"""

    if signal_type == "BUY":
        plan = calculate_trade_plan(tech["entry_price"], tech["stop_loss"])
        msg += f"""<b>🎯 תוכנית מסחר וניהול סיכונים:</b>
• 🎯 <b>מחיר כניסה:</b> <code>${plan['entry']}</code>
• 🛑 <b>Stop Loss (ATR):</b> <code>${plan['stop_loss']}</code>
• 🎯 <b>יעד 1 (TP1):</b> <code>${plan['tp1']}</code> (+{plan['tp1_pct']}%)
• 🚀 <b>יעד 2 (TP2):</b> <code>${plan['tp2']}</code> (+{plan['tp2_pct']}%)
• ⚖️ <b>יחס סיכון/סיכוי:</b> <code>1:{plan['risk_reward']}</code>
• 🛡️ <b>גודל פוזיציה מומלץ:</b> <code>{plan['max_position_pct']}</code>
"""
        log_alert_history(symbol, plan['entry'], plan['stop_loss'], plan['tp1'], plan['tp2'])

    markup = InlineKeyboardMarkup(row_width=2)
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף", url=f"https://www.tradingview.com/chart/?symbol={symbol}")
    if signal_type == "BUY":
        plan = calculate_trade_plan(tech["entry_price"], tech["stop_loss"])
        btn_calc = InlineKeyboardButton("💰 חישוב עסקה", callback_data=f"calc_{symbol}_{plan['entry']}_{plan['stop_loss']}")
        markup.add(btn_chart, btn_calc)
    else:
        markup.add(btn_chart)

    return msg, markup

# ------------------------------------------------------------------------------
# 9. סורק אוטומטי וטיפול בפקודות טלגרם
# ------------------------------------------------------------------------------
def is_market_open() -> bool:
    israel_tz = pytz.timezone('Asia/Jerusalem')
    now = datetime.datetime.now(israel_tz)
    if now.weekday() in (5, 6):
        return False
    start_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
    end_time = now.replace(hour=23, minute=0, second=0, microsecond=0)
    return start_time <= now <= end_time

def scan_worker_auto(symbol: str):
    if has_alerted_today(symbol):
        return

    tech = analyze_technical_patterns(symbol)
    if tech and (tech["has_breakout"] or tech["is_uptrend"]):
        users = get_all_users()
        if not users:
            return

        msg, markup = create_report_message(symbol)
        if "המלצת קנייה" in msg:
            alert_msg = f"🚨 <b>התראת איתות בזמן אמת!</b>\n{msg}"
            for chat_id in users:
                try:
                    bot.send_message(chat_id, alert_msg, parse_mode="HTML", reply_markup=markup)
                except Exception as e:
                    print(f"[Alert Error] {chat_id}: {e}")

            mark_alerted_today(symbol)

def scheduled_market_scan():
    if not is_market_open():
        return

    print(f"[{datetime.datetime.now()}] 🔄 מתחיל סריקת שוק אופטימלית...")
    tickers = fetch_market_tickers()

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(scan_worker_auto, tickers)

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(scheduled_market_scan, 'interval', minutes=15)
scheduler.start()

@bot.message_handler(commands=['start'])
def cmd_start(message):
    add_user(message.chat.id)
    bot.reply_to(message, "<b>ברוכים הבאים למערכת איתותי המסחר המשופרת (v2)! 🚀</b>\n\nמנוע זיהוי תבניות מתקדם ואימות צולב בשידור חי.\nהשתמש ב- /scan לסריקה מקיפה או ב- /tech [SYMBOL] לבדיקת מניה.", parse_mode="HTML")

def scan_worker_manual(symbol: str) -> Tuple[str, Optional[Tuple[str, Any]]]:
    tech = analyze_technical_patterns(symbol)
    if tech:
        msg, markup = create_report_message(symbol)
        if "המלצת קנייה" in msg:
            return symbol, (msg, markup)
    return symbol, None

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    add_user(message.chat.id)
    tickers = fetch_market_tickers()
    bot.reply_to(message, f"🔍 מתחיל סריקה מרוכזת של <b>{len(tickers)} מניות</b> במנוע Pattern Registry...", parse_mode="HTML")

    found_any = False
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(scan_worker_manual, tickers)
        for symbol, report in results:
            if report and report[1] is not None:
                msg, markup = report
                bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
                found_any = True

    if not found_any:
        bot.send_message(message.chat.id, "ℹ️ לא אותרו כעת מניות העונות על קריטריוני האימות הקפדניים.")

@bot.message_handler(commands=['tech'])
def cmd_tech(message):
    add_user(message.chat.id)
    try:
        symbol = message.text.split()[1].upper()
        msg, markup = create_report_message(symbol)
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
    except IndexError:
        bot.reply_to(message, "⚠️ נא לציין סימול מנייה. לדוגמה: <code>/tech AAPL</code>", parse_mode="HTML")

# ------------------------------------------------------------------------------
# 10. הרצת השרת והבוט
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()

    print("🤖 Telegram Trading System Engine v2 is fully active...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
