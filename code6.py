import os
import time
import sqlite3
import threading
import requests
import datetime
import pytz
import re
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from flask import Flask
from telebot import TeleBot, types
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler
from deep_translator import GoogleTranslator

# ------------------------------------------------------------------------------
# 1. הגדרות סביבה ומשתנים גלובליים
# ------------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
PORT = int(os.environ.get("PORT", 5000))
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")

bot = TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)
translator = GoogleTranslator(source='auto', target='iw')

USER_CALC_STATE = {}
CACHE = {}
CACHE_TTL = 300  # 5 דקות

DEFAULT_SCAN_TICKERS = [
    "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", "META", "GOOGL", "NFLX", "BRK-B",
    "LLY", "AVGO", "JPM", "UNH", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "COST",
    "PEP", "ADBE", "WMT", "CRM", "BAC", "ACN", "MCD", "CSCO", "ORCL", "LIN", "ABT",
    "INTC", "QCOM", "TXN", "AMAT", "MU", "PANW", "SNPS", "CDNS", "SMCI", "ARM", "MRNA",
    "LMT", "RTX", "PLTR", "NOC"
]

# Header מותאם למניעת חסימות HTTP 500/403
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# ------------------------------------------------------------------------------
# 2. ניהול בסיס נתונים (SQLite Database + מעקב התראות יומי)
# ------------------------------------------------------------------------------
DB_FILE = "bot_database.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                chat_id INTEGER PRIMARY KEY
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_alerts (
                symbol TEXT,
                alert_date TEXT,
                PRIMARY KEY (symbol, alert_date)
            )
        """)
        conn.commit()

def add_user(chat_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO users (chat_id) VALUES (?)", (chat_id,))
        conn.commit()

def get_all_users() -> list:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chat_id FROM users")
        rows = cursor.fetchall()
        return [r[0] for r in rows]

def has_alerted_today(symbol: str) -> bool:
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM daily_alerts WHERE symbol = ? AND alert_date = ?", (symbol, today_str))
        return cursor.fetchone() is not None

def mark_alerted_today(symbol: str):
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO daily_alerts (symbol, alert_date) VALUES (?, ?)", (symbol, today_str))
        conn.commit()

init_db()

# ------------------------------------------------------------------------------
# 3. שרת FLASK ומנגנון KEEP-ALIVE
# ------------------------------------------------------------------------------
@app.route('/')
def health_check():
    return "OK - Telegram Trading Bot is Running!", 200

def keep_alive_ping():
    while True:
        try:
            time.sleep(600)
            if "localhost" not in SELF_URL:
                requests.get(SELF_URL, timeout=10, headers=HEADERS)
        except Exception as e:
            print(f"[Keep-Alive Error]: {e}")

# ------------------------------------------------------------------------------
# 3.5. מנגנון בקרה ואימות נתונים (VALIDATION LAYER)
# ------------------------------------------------------------------------------
def validate_technical_data(tech: dict) -> bool:
    """
    בקרת איכות נתונים בסיסית: לוודא שהנתונים שהתקבלו תקינים, שלמים וסבירים.
    """
    if not tech:
        return False

    # 1. אימות מחירי שוק
    current_price = tech.get("current_price", 0)
    if pd.isna(current_price) or current_price <= 0:
        return False

    # 2. אימות מדד RSI (חייב להיות בטווח 0-100)
    rsi = tech.get("rsi", 50)
    if pd.isna(rsi) or not (0 <= rsi <= 100):
        return False

    # 3. אימות סטופ-לוס ומחיר כניסה
    if tech.get("signal") == "BUY":
        entry = tech.get("entry_price")
        stop = tech.get("stop_loss")
        if not entry or not stop or pd.isna(entry) or pd.isna(stop):
            return False
        if stop >= entry:  # בלונג - סטופ לוס חייב להיות נמוך ממחיר הכניסה
            return False

    return True

def validate_technical_signal(tech: dict) -> bool:
    """
    בקרת לוגיקה פיננסית: לוודא שאיתות קנייה עומד בקריטריונים קשיחים של סיכון/סיכוי ומניעת איתותי שווא.
    """
    if not tech or tech.get("signal") != "BUY":
        return True  # ניטרלי/המתנה אינו דורש סינון קשיח

    entry = tech.get("entry_price")
    stop = tech.get("stop_loss")
    rsi = tech.get("rsi", 50)
    
    # 1. אימות יחס סיכון-סיכוי (R:R Ratio) - יעד 1 חייב לשקף לפחות 1:1.5
    risk = entry - stop
    if risk <= 0:
        return False
        
    tp1 = entry + (risk * 2)
    if (tp1 - entry) / risk < 1.5:
        return False

    # 2. מניעת מתיחת יתר (RSI Overbought) - כניסה מעל RSI 75 היא בסיכון גבוה
    if rsi > 75:
        return False

    return True

def validate_and_clean_news(headlines: list) -> list:
    """
    בקרת איכות תוכן ותרגום: מנקה כותרות ריקות, כפילויות ושגיאות תרגום.
    """
    cleaned = []
    seen = set()
    for h in headlines:
        if not h or not isinstance(h, str):
            continue
        h_clean = h.strip()
        if len(h_clean) < 10 or h_clean in seen:
            continue
        seen.add(h_clean)
        cleaned.append(h_clean)
    return cleaned

# ------------------------------------------------------------------------------
# 4. מנוע ניתוח טכני משולב (ממוצעים נעים + תבניות + נרות + ווליום)
# ------------------------------------------------------------------------------
def analyze_technical_patterns(symbol: str) -> dict:
    now = time.time()
    if symbol in CACHE:
        cached_time, cached_data = CACHE[symbol]
        if now - cached_time < CACHE_TTL:
            return cached_data

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")
        
        # ניקוי ערכי NaN במחיר הסגירה
        if not df.empty:
            df = df.dropna(subset=['Close'])

        if df.empty or len(df) < 50:
            return None

        # טיפול בערכי NaN פוטנציאליים בשורות האחרונות
        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
        
        # Fallback למקרה ש-Close עדיין מחזיר NaN או 0
        if pd.isna(current_price) or current_price == 0:
            current_price = float(ticker.fast_info.get('lastPrice', 0))
            prev_price = float(ticker.fast_info.get('previousClose', current_price))

        change_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0.0

        # --- א. חישוב אינדיקטורים וממוצעים נעים ---
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        rsi_valid = df['RSI'].dropna()
        rsi = float(rsi_valid.iloc[-1]) if not rsi_valid.empty else 50.0

        ema20_valid = df['EMA20'].dropna()
        ema20 = float(ema20_valid.iloc[-1]) if not ema20_valid.empty else current_price

        ema50_valid = df['EMA50'].dropna()
        ema50 = float(ema50_valid.iloc[-1]) if not ema50_valid.empty else current_price

        ema200_valid = df['EMA200'].dropna()
        ema200 = float(ema200_valid.iloc[-1]) if not ema200_valid.empty else ema50

        atr_valid = df['ATR'].dropna()
        atr = float(atr_valid.iloc[-1]) if not atr_valid.empty else (current_price * 0.02)

        # בדיקת מגמה לפי ממוצעים נעים
        is_uptrend = current_price > ema20 > ema50 and current_price > ema200

        # ניתוח נפח מסחר (Volume Accumulation)
        avg_vol_20 = df['Volume'].iloc[-21:-1].mean()
        curr_vol = df['Volume'].iloc[-1]
        volume_spike = bool(curr_vol > (avg_vol_20 * 1.3)) if avg_vol_20 > 0 else False

        patterns = []
        candlesticks = []

        # --- ב. זיהוי נרות היפוך/אישור (Candlesticks) ---
        o1, h1, l1, c1 = float(df['Open'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
        o2, c2 = float(df['Open'].iloc[-2]), float(df['Close'].iloc[-2])

        body1 = abs(c1 - o1)
        lower_shadow1 = min(o1, c1) - l1
        upper_shadow1 = h1 - max(o1, c1)

        # 1. נר פטיש (Hammer)
        if lower_shadow1 >= (2 * body1) and upper_shadow1 <= (0.3 * body1) and body1 > 0:
            candlesticks.append("נר פטיש היפוכי (Hammer)")

        # 2. בליעה שורית (Bullish Engulfing)
        if c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2:
            candlesticks.append("בליעה שורית (Bullish Engulfing)")

        # --- ג. זיהוי תבניות מחיר (Chart Patterns) ---
        high_30 = float(df['High'].iloc[-30:].max())
        low_30 = float(df['Low'].iloc[-30:].min())
        recent_support = float(df['Low'].tail(10).min())

        # 1. ספל וידית (Cup & Handle)
        if (high_30 - current_price) / high_30 < 0.04 and current_price > ema20:
            patterns.append("ספל וידית (Cup & Handle)")

        # 2. תחתית כפולה (Double Bottom)
        lows = df['Low'].iloc[-30:]
        double_bottom_hits = lows[lows <= low_30 * 1.025]
        if len(double_bottom_hits) >= 2 and (current_price > low_30 * 1.03):
            patterns.append("תחתית כפולה (Double Bottom)")

        # 3. משולש עולה (Ascending Triangle)
        highs_flat = abs(df['High'].iloc[-15:].max() - df['High'].iloc[-5:].max()) / current_price < 0.02
        lows_rising = df['Low'].iloc[-15] < df['Low'].iloc[-8] < df['Low'].iloc[-1]
        if highs_flat and lows_rising:
            patterns.append("משולש עולה (Ascending Triangle)")

        # 4. דגל שורי (Bullish Flag)
        recent_runup = (df['High'].iloc[-5:].max() - df['Low'].iloc[-15:].min()) / df['Low'].iloc[-15:].min()
        if recent_runup > 0.10 and abs(current_price - df['Close'].iloc[-3:].mean()) / current_price < 0.02:
            patterns.append("דגל שורי (Bullish Flag)")

        # --- ד. מודל הצטלבות קריטריונים (Confluence Signal Engine) ---
        has_pattern_or_candle = len(patterns) > 0 or len(candlesticks) > 0
        
        # תנאי סף לאיתות כניסה חזק (BUY): מגמה עולה + תבנית/נר היפוך + נפח מסחר/RSI תקין
        is_strong_buy = is_uptrend and has_pattern_or_candle and (volume_spike or (45 <= rsi <= 70))

        if is_strong_buy:
            signal = "BUY"
            entry_price = round(current_price, 2)
            stop_loss = round(max(recent_support * 0.99, entry_price - (1.5 * atr)), 2)
            if stop_loss >= entry_price:
                stop_loss = round(entry_price - (1.5 * atr), 2)
        elif has_pattern_or_candle or is_uptrend:
            signal = "HOLD"
            entry_price = None
            stop_loss = None
        else:
            signal = "NEUTRAL"
            entry_price = None
            stop_loss = None

        all_patterns_combined = patterns + candlesticks
        if is_uptrend:
            all_patterns_combined.append("מגמה שורית (EMA20 > EMA50 > EMA200)")

        result = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi, 1),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "ema200": round(ema200, 2),
            "volume_accumulating": volume_spike,
            "patterns": all_patterns_combined,
            "pattern_depth": 0.15,
            "signal": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "atr": round(atr, 2)
        }

        # --- ה. הפעלת מנגנון הבקרה והאימות ---
        if not validate_technical_data(result):
            print(f"[Validation Warning] {symbol}: הנתונים שנשלפו אינם תקינים.")
            return None

        if not validate_technical_signal(result):
            # במידה והאיתות לא עבר אימות איכות טכני - משנמכים ל-HOLD
            result["signal"] = "HOLD"
            result["entry_price"] = None
            result["stop_loss"] = None

        CACHE[symbol] = (now, result)
        return result

    except Exception as e:
        print(f"[Tech Analysis Error] {symbol}: {e}")
        return None

# ------------------------------------------------------------------------------
# 5. ניתוח חדשות מורחב (Comprehensive Catalyst NLP Engine)
# ------------------------------------------------------------------------------
NOISE_PATTERNS = [
    r"stock(s)? to watch", r"market recap", r"top gainers", r"weekly roundup",
    r"dow jones update", r"indexes market", r"here is why", r"options volume",
    r"what to expect", r"stocks making moves", r"movers today", r"daily brief",
    r"s&p 500 update", r"nasdaq analysis", r"analyst rating update"
]

HIGH_IMPACT_CATALYSTS = {
    r"\bfda\b|\btrial\b|\bphase\b|\bclinical\b|\bapproval\b": 
        ("אישור/ניסוי קליני מפתח (FDA/Pharma)", 10),
    r"\bdefense contract\b|\bmilitary deal\b|\bpentagon\b|\barmy contract\b": 
        ("חוזה הצטיידות ביטחוני / פנטגון 🛡️", 10),
    r"\bsanctions\b|\btariff exemption\b|\bexport license\b": 
        ("שינוי מדיניות סחר / הקלת סנקציות / מכסים 🌐", 9),
    r"\bgeopolitical\b|\bwar demand\b|\bsecurity crisis\b|\bsupply disruption\b": 
        ("אירוע גיאופוליטי / הסטת ביקוש מלחמתית ⚠️", 8),
    r"\bearnings\b|\bbeat\b|\brevenue beat\b|\brecord revenue\b": 
        ("דוחות כספיים / תוצאות שיא 📈", 9),
    r"\bguidance\b|\braises outlook\b|\braised guidance\b": 
        ("עדכון תחזית צמיחה כלפי מעלה 🚀", 9),
    r"\bmerger\b|\bacquisition\b|\bbuyout\b|\btakeover\b": 
        ("עסקת מיזוג / רכישה דרמטית 🤝", 9),
    r"\bcontract\b|\bdeal\b|\bpartnership\b|\bsupply agreement\b": 
        ("חתימת חוזה אסטרטגי / הספקת ענק 📝", 8),
    r"\bgovernment grant\b|\bsubsidy\b|\bchips act\b|\bfederal funding\b": 
        ("מענק ממשלתי / סובסידיה אסטרטגית 🏛️", 8),
    r"\bftc approval\b|\bdoj approval\b|\bregulatory clearance\b": 
        ("אישור רגולטורי / יציאה מסיכון משפטי ⚖️", 8),
    r"\binsider buy(ing)?\b|\bceo bought\b|\bdirector purchased\b": 
        ("רכישת מניות מאסיבית ע\"י הנהלה (Insider Buying) 💎", 9),
    r"\bshare buyback\b|\brepurchase program\b": 
        ("תוכנית רכישה עצמית של מניות (Buyback) 💵", 8),
    r"\bactivist investor\b|\bboard seat\b": 
        ("כניסת משקיע אקטיביסט להצפת ערך 💥", 8),
    r"\bpatent\b|\bbreakthrough\b|\binnovation\b": 
        ("פריצת דרך טכנולוגית / פטנט 🔬", 8)
}

def evaluate_headline_impact(headline: str) -> tuple:
    h_lower = headline.lower()

    for noise in NOISE_PATTERNS:
        if re.search(noise, h_lower):
            return 0, None

    for pattern, (label, score) in HIGH_IMPACT_CATALYSTS.items():
        if re.search(pattern, h_lower):
            return score, label

    return 4, "ידיעה חברתית/סקטוריאלית כללית"

def fetch_finnhub_data(symbol: str) -> dict:
    raw_headlines = []
    
    # 1. ניסיון שליפה מ-Finnhub
    if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
        try:
            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
            to_date = today.strftime('%Y-%m-%d')
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
            
            res = requests.get(news_url, headers=HEADERS, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list):
                    for item in data:
                        h = item.get("headline")
                        if h:
                            raw_headlines.append(h)
            else:
                print(f"[Finnhub Status Warning] {symbol}: HTTP {res.status_code}")
        except Exception as e:
            print(f"[Finnhub Fetch Error] {symbol}: {e}")

    # 2. ניסיון שליפה מ-YFinance במקרה של חוסר נתונים
    if not raw_headlines:
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news
            if news_items:
                for item in news_items:
                    title = item.get("title", "")
                    if not title and "content" in item and isinstance(item["content"], dict):
                        title = item["content"].get("title", "")
                    if title:
                        raw_headlines.append(title)
        except Exception as e:
            print(f"[YFinance News Error] {symbol}: {e}")

    scored_headlines = []
    found_catalysts = set()

    for headline in raw_headlines:
        score, catalyst_label = evaluate_headline_impact(headline)
        if score >= 5:
            scored_headlines.append((score, headline))
            if catalyst_label and score >= 8:
                found_catalysts.add(catalyst_label)

    scored_headlines.sort(key=lambda x: x[0], reverse=True)
    top_headlines = [h[1] for h in scored_headlines[:3]]

    translated_headlines = []
    for h in top_headlines:
        try:
            translated = translator.translate(h)
            translated_headlines.append(translated)
        except Exception:
            translated_headlines.append(h)

    # בקרת איכות שניקוי כותרות ותרגום
    translated_headlines = validate_and_clean_news(translated_headlines)

    catalyst_str = " | ".join(found_catalysts) if found_catalysts else "לא אותרו קטליזטורים דרמטיים"
    
    if len(found_catalysts) > 0:
        sentiment = "חיובי חזק (אותרו קטליזטורים מהותיים) 🟢"
    elif len(translated_headlines) > 0:
        sentiment = "ניטרלי / מעקב בלבד 🟡"
    else:
        sentiment = "ללא חדשות מהותיות ⚪"

    return {
        "headlines": translated_headlines,
        "catalyst": catalyst_str,
        "sentiment": sentiment,
        "analyst_ratings": "זמין בדוח המלא"
    }

# ------------------------------------------------------------------------------
# 6. מחשבון עסקאות ושער דולר
# ------------------------------------------------------------------------------
def get_usd_ils_rate() -> float:
    try:
        usd_ticker = yf.Ticker("USDILS=X")
        df = usd_ticker.history(period="1d")
        if not df.empty and not pd.isna(df['Close'].iloc[-1]):
            rate = float(df['Close'].iloc[-1])
            if rate > 0:
                return round(rate, 2)
        return 3.70
    except Exception:
        return 3.70

def calculate_trade_plan(entry_price: float, stop_loss: float, pattern_depth_pct: float = 0.15) -> dict:
    risk_per_share = entry_price - stop_loss
    
    tp1_price = entry_price + (risk_per_share * 2)
    tp1_pct = ((tp1_price - entry_price) / entry_price) * 100
    
    tp2_price = entry_price + (risk_per_share * 3.5)
    tp2_pct = ((tp2_price - entry_price) / entry_price) * 100
    
    return {
        "entry": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "tp1": round(tp1_price, 2),
        "tp1_pct": round(tp1_pct, 1),
        "tp2": round(tp2_price, 2),
        "tp2_pct": round(tp2_pct, 1),
        "new_stop_after_tp1": round(entry_price, 2)
    }

def generate_calculator_response(amount: float, is_usd: bool, usd_rate: float, trade_plan: dict) -> str:
    currency_symbol = "$" if is_usd else "₪"
    amount_in_usd = amount if is_usd else amount / usd_rate
    
    entry = trade_plan["entry"]
    stop = trade_plan["stop_loss"]
    tp1 = trade_plan["tp1"]
    tp2 = trade_plan["tp2"]
    
    total_shares = int(amount_in_usd / entry)
    if total_shares < 1:
        return "⚠️ סכום ההשקעה נמוך מדי לרכישת מנייה אחת לפחות במחיר הכניסה."
    
    half_shares = total_shares // 2
    rem_shares = total_shares - half_shares
    
    profit_tp1_usd = half_shares * (tp1 - entry)
    profit_tp2_usd = rem_shares * (tp2 - entry)
    total_profit_usd = profit_tp1_usd + profit_tp2_usd
    max_risk_usd = total_shares * (entry - stop)
    
    profit_tp1_ils = profit_tp1_usd * usd_rate
    profit_tp2_ils = profit_tp2_usd * usd_rate
    total_profit_ils = total_profit_usd * usd_rate
    max_risk_ils = max_risk_usd * usd_rate

    msg = f"""
<b>💰 תוכנית מודולרית לניהול העסקה ({amount:,.0f}{currency_symbol}):</b>
<i>(שער המרה: ₪{usd_rate})</i>

<b>📌 כניסה וכמות:</b>
• מחיר כניסה: <code>${entry:.2f}</code>
• כמות מניות: <b>{total_shares} מניות</b>

---
<b>🎯 שלב 1: מימוש 50% ואיפוס סיכון (TP1)</b>
• מחיר יעד 1: <code>${tp1:.2f}</code> (+{trade_plan['tp1_pct']}%)
• רווח ננעל: ${profit_tp1_usd:.2f} ({profit_tp1_ils:.2f} ₪)
• 🛡️ להעלות סטופ לוס ל-<code>${trade_plan['new_stop_after_tp1']:.2f}</code> (מחיר הכניסה).

---
<b>🚀 שלב 2: יעד מורחב (TP2)</b>
• מחיר יעד 2: <code>${tp2:.2f}</code> (+{trade_plan['tp2_pct']}%)
• רווח נוסף: ${profit_tp2_usd:.2f} ({profit_tp2_ils:.2f} ₪)

---
<b>📊 סיכום עמדה:</b>
🟢 <b>סך רווח צפוי:</b> ${total_profit_usd:.2f} ({total_profit_ils:.2f} ₪)
🔴 <b>סיכון מרבי בסטופ (${stop:.2f}):</b> ${max_risk_usd:.2f} ({max_risk_ils:.2f} ₪)
"""
    return msg

# ------------------------------------------------------------------------------
# 7. מחולל הדוחות והמקלדת
# ------------------------------------------------------------------------------
def create_report_message(symbol: str) -> tuple:
    tech = analyze_technical_patterns(symbol)
    if not tech:
        return f"❌ לא ניתן היה לשלוף נתונים תקינים עבור המנייה <b>{symbol}</b>.", None

    finnhub = fetch_finnhub_data(symbol)

    if tech["signal"] == "BUY":
        recommendation = "🟢 <b>מומלץ להיכנס להשקעה (מגמה עולה + תבנית/נר היפוך)</b>"
    elif tech["signal"] == "HOLD":
        recommendation = "🟡 <b>להמתין ולעקוב (קיימת תבנית אך חסר אישור נפח/מגמה)</b>"
    else:
        recommendation = "🔴 <b>לא מומלץ כעת (ללא איתות פריצה)</b>"

    patterns_str = ", ".join(tech["patterns"]) if tech["patterns"] else "לא זוהו תבניות מיוחדות"
    headlines_str = "\n".join([f"• {h}" for h in finnhub["headlines"]]) if finnhub["headlines"] else "• לא אותרו כותרות איכותיות"

    msg = f"""
<b>📊 דוח ניתוח מקיף עבור {symbol}</b>

<b>💡 המלצה:</b>
{recommendation}

---
<b>📈 נתונים טכניים, ממוצעים ותבניות:</b>
• מחיר נוכחי: <code>${tech['current_price']}</code> ({'+' if tech['change_pct']>0 else ''}{tech['change_pct']}%)
• RSI: <code>{tech['rsi']}</code> | EMA20: <code>${tech['ema20']}</code> | EMA50: <code>${tech['ema50']}</code>
• תבניות, נרות ומגמה: <b>{patterns_str}</b>
• נפח מסחר חורג (Volume Spike): {'כן 🟢' if tech['volume_accumulating'] else 'רגיל ⚪'}

---
<b>📰 חדשות וקטליזטורים מסוננים (NLP Scoring):</b>
• קטליזטור שנמצא: <b>{finnhub['catalyst']}</b>
• סנטימנט חדשותי: <b>{finnhub['sentiment']}</b>
<b>כותרות איכותיות שנבחרו:</b>
{headlines_str}

---
"""

    if tech["signal"] == "BUY":
        plan = calculate_trade_plan(tech["entry_price"], tech["stop_loss"], tech["pattern_depth"])
        msg += f"""<b>🎯 תוכנית מסחר מומלצת:</b>
• 🎯 <b>מחיר כניסה:</b> <code>${plan['entry']}</code>
• 🛑 <b>Stop Loss:</b> <code>${plan['stop_loss']}</code>
• 🎯 <b>יעד 1 (TP1):</b> <code>${plan['tp1']}</code> (+{plan['tp1_pct']}%)
• 🚀 <b>יעד 2 (TP2):</b> <code>${plan['tp2']}</code> (+{plan['tp2_pct']}%)
"""

    markup = InlineKeyboardMarkup(row_width=2)
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף", url=f"https://www.tradingview.com/chart/?symbol={symbol}")
    btn_news = InlineKeyboardButton("📰 חדשות", callback_data=f"news_{symbol}")
    btn_tech = InlineKeyboardButton("📊 ניתוח טכני", callback_data=f"tech_{symbol}")
    
    if tech["signal"] == "BUY":
        btn_calc = InlineKeyboardButton("💰 חישוב עסקה", callback_data=f"calc_{symbol}_{plan['entry']}_{plan['stop_loss']}_{tech['pattern_depth']}")
        markup.add(btn_chart, btn_calc)
    else:
        markup.add(btn_chart)
        
    markup.add(btn_news, btn_tech)

    return msg, markup

# ------------------------------------------------------------------------------
# 8. ניהול פקודות ואירועים (Telegram Handlers)
# ------------------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def cmd_start(message):
    add_user(message.chat.id)
    welcome_text = """
<b>ברוכים הבאים לסורק השוק והסייען הפיננסי! 🚀</b>

הבוט משלב ממוצעים נעים (EMA20/50/200), תבניות מחיר ונרות היפוך, לצד סינון חדשות חכם מבוסס אימפקט ומערכת בקרת איכות נתונים.

<b>פקודות זמינות:</b>
/scan - סריקת שוק לזיהוי פריצות בזמן אמת
/tech &lt;SYMBOL&gt; - ניתוח טכני למנייה (למשל: <code>/tech TSLA</code>)
/news &lt;SYMBOL&gt; - חדשות ואירועים למנייה (למשל: <code>/news MRNA</code>)
"""
    bot.reply_to(message, welcome_text, parse_mode="HTML")

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    add_user(message.chat.id)
    bot.reply_to(message, "🔍 מתחיל סריקה מורחבת בשוק... אנא המתן.")
    
    found_any = False
    for sym in DEFAULT_SCAN_TICKERS:
        tech = analyze_technical_patterns(sym)
        if tech and tech["signal"] == "BUY":
            msg, markup = create_report_message(sym)
            bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
            found_any = True

    if not found_any:
        bot.send_message(message.chat.id, "ℹ️ לא אותרו כעת מניות העונות על כלל הקריטריונים (מגמה + תבנית + ווליום + אימות בקרת איכות).")

@bot.message_handler(commands=['tech'])
def cmd_tech(message):
    add_user(message.chat.id)
    try:
        symbol = message.text.split()[1].upper()
        msg, markup = create_report_message(symbol)
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
    except IndexError:
        bot.reply_to(message, "⚠️ נא לציין סימול מנייה. לדוגמה: <code>/tech AAPL</code>", parse_mode="HTML")

@bot.message_handler(commands=['news'])
def cmd_news(message):
    add_user(message.chat.id)
    try:
        symbol = message.text.split()[1].upper()
        msg, markup = create_report_message(symbol)
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
    except IndexError:
        bot.reply_to(message, "⚠️ נא לציין סימול מנייה. לדוגמה: <code>/news NVDA</code>", parse_mode="HTML")

# ------------------------------------------------------------------------------
# 9. טיפול בלחיצות כפתורים (Callback Queries)
# ------------------------------------------------------------------------------
@bot.callback_query_handler(func=lambda call: True)
def handle_callbacks(call):
    data = call.data
    chat_id = call.message.chat.id

    if data.startswith("calc_"):
        _, symbol, entry, stop, depth = data.split("_")
        USER_CALC_STATE[chat_id] = {
            "symbol": symbol,
            "entry": float(entry),
            "stop": float(stop),
            "pattern_depth": float(depth)
        }
        markup = InlineKeyboardMarkup()
        markup.add(
            InlineKeyboardButton("דולר ($)", callback_data="curr_USD"),
            InlineKeyboardButton("שקל (₪)", callback_data="curr_ILS")
        )
        bot.send_message(chat_id, f"💰 <b>מחשבון עסקה עבור {symbol}</b>\nבחר באיזה מטבע תרצה להזין את סכום ההשקעה:", parse_mode="HTML", reply_markup=markup)

    elif data.startswith("curr_"):
        is_usd = (data == "curr_USD")
        if chat_id in USER_CALC_STATE:
            USER_CALC_STATE[chat_id]["is_usd"] = is_usd
            curr_str = "דולר ($)" if is_usd else "שקלים (₪)"
            msg = bot.send_message(chat_id, f"רשום כעת את סכום ההשקעה המבוקש ב-{curr_str}: (למשל: 5000)")
            bot.register_next_step_handler(msg, process_calculator_amount)

    elif data.startswith("news_"):
        symbol = data.split("_")[1]
        finnhub = fetch_finnhub_data(symbol)
        headlines_str = "\n".join([f"• {h}" for h in finnhub['headlines']]) if finnhub['headlines'] else "אין כותרות איכותיות"
        text = f"<b>📰 חדשות עבור {symbol}:</b>\n\n• קטליזטור: <b>{finnhub['catalyst']}</b>\n• סנטימנט: <b>{finnhub['sentiment']}</b>\n\n<b>כותרות:</b>\n{headlines_str}"
        bot.send_message(chat_id, text, parse_mode="HTML")

    elif data.startswith("tech_"):
        symbol = data.split("_")[1]
        tech = analyze_technical_patterns(symbol)
        if tech:
            patterns_str = ", ".join(tech["patterns"]) if tech["patterns"] else "אין"
            text = f"<b>📊 ניתוח טכני עבור {symbol}:</b>\n\n• מחיר שוק: <code>${tech['current_price']}</code>\n• RSI: <code>{tech['rsi']}</code> | ATR: <code>${tech['atr']}</code>\n• תבניות ומגמה: <b>{patterns_str}</b>\n• איתות: <b>{tech['signal']}</b>"
            if tech["signal"] == "BUY":
                text += f"\n• מחיר כניסה: <code>${tech['entry_price']}</code>\n• סטופ לוס: <code>${tech['stop_loss']}</code>"
            bot.send_message(chat_id, text, parse_mode="HTML")

def process_calculator_amount(message):
    chat_id = message.chat.id
    if chat_id not in USER_CALC_STATE:
        bot.reply_to(message, "❌ פג תוקף החישוב, אנא לחץ שוב על 'חישוב עסקה'.")
        return

    try:
        amount = float(message.text.replace(",", "").strip())
        state = USER_CALC_STATE[chat_id]
        usd_rate = get_usd_ils_rate()
        plan = calculate_trade_plan(state["entry"], state["stop"], state["pattern_depth"])
        response_text = generate_calculator_response(amount, state["is_usd"], usd_rate, plan)
        
        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton("📈 צפייה בגרף ב-TradingView", url=f"https://www.tradingview.com/chart/?symbol={state['symbol']}"))
        bot.send_message(chat_id, response_text, parse_mode="HTML", reply_markup=markup)
        del USER_CALC_STATE[chat_id]
    except ValueError:
        bot.reply_to(message, "⚠️ נא להזין מספר תקין בלבד (לדוגמה: 5000). נסה שוב:")
        bot.register_next_step_handler(message, process_calculator_amount)

# ------------------------------------------------------------------------------
# 10. מנוע סריקה אוטומטית (עם מנגנון סינון התראות יומי)
# ------------------------------------------------------------------------------
def is_market_open() -> bool:
    israel_tz = pytz.timezone('Asia/Jerusalem')
    now = datetime.datetime.now(israel_tz)
    
    if now.weekday() > 4:
        return False

    start_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
    end_time = now.replace(hour=23, minute=0, second=0, microsecond=0)
    
    return start_time <= now <= end_time

def scheduled_market_scan():
    if not is_market_open():
        return

    users = get_all_users()
    if not users:
        return

    for sym in DEFAULT_SCAN_TICKERS:
        if has_alerted_today(sym):
            continue

        tech = analyze_technical_patterns(sym)
        if tech and tech["signal"] == "BUY":
            msg, markup = create_report_message(sym)
            for chat_id in users:
                try:
                    bot.send_message(chat_id, f"🚨 <b>התראת איתות פריצה בזמן אמת!</b>\n{msg}", parse_mode="HTML", reply_markup=markup)
                except Exception as e:
                    print(f"[Scheduled Notification Error] to {chat_id}: {e}")

            mark_alerted_today(sym)

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(scheduled_market_scan, 'interval', minutes=15)
scheduler.start()

# ------------------------------------------------------------------------------
# 11. הרצת השרת והבוט (Main Execution)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()
    
    print("🤖 Telegram Trading Bot is active...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
