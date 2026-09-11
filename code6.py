import os
import sys
import time
import sqlite3
import logging
import threading
import requests
import datetime
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
    "MIN_PRICE": 3.0,               # מחיר מינימלי למניה
    "MIN_DAILY_VALUE_USD": 2000000, # מחזור כספי יומי ממוצע מינימלי (2 מיליון $)
    "SCORE_THRESHOLD": 75,          # רף סף קשיח לשליחת התראה (מתוך 100)
    "ALERT_COOLDOWN_HOURS": 6
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
# 2. מסד נתונים
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
            results.append({"symbol": sym, "index": "S&P 500", "currency": "USD"})
    except Exception as e:
        logger.warning(f"Failed loading S&P500: {e}")

    TICKERS_CACHE["tickers"] = results
    TICKERS_CACHE["fetched_at"] = now
    return results

# ------------------------------------------------------------------------------
# 4. מנוע זיהוי תבניות מתקדם
# ------------------------------------------------------------------------------
def detect_candlestick_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    if len(df) < 5: return patterns
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    body = abs(curr['Close'] - curr['Open'])
    candle_range = curr['High'] - curr['Low']
    safe_body = max(body, 0.0001)

    upper_shade = curr['High'] - max(curr['Open'], curr['Close'])
    lower_shade = min(curr['Open'], curr['Close']) - curr['Low']

    if candle_range > 0 and lower_shade >= (2 * safe_body) and upper_shade <= (0.2 * safe_body):
        patterns.append("פטיש (Hammer) 🔨")

    if prev['Close'] < prev['Open'] and curr['Close'] > curr['Open']:
        if curr['Open'] <= prev['Close'] and curr['Close'] >= prev['Open']:
            patterns.append("בליעה שורית (Bullish Engulfing) 🟢")

    return patterns

def detect_chart_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    if len(df) < 60: return patterns
    closes, highs, lows = df['Close'].values, df['High'].values, df['Low'].values

    left_rim = max(highs[-60:-30])
    cup_bottom = min(lows[-45:-15])
    right_rim = max(highs[-20:-5])
    handle_low = min(lows[-10:])
    if (left_rim * 0.95 <= right_rim <= left_rim * 1.05) and (cup_bottom < left_rim * 0.85):
        if handle_low > cup_bottom and closes[-1] >= right_rim * 0.98:
            patterns.append("ספל וידית (Cup and Handle) ☕")

    pole_move = (closes[-20] - closes[-40]) / closes[-40]
    if pole_move > 0.08 and ((max(highs[-15:]) - min(lows[-15:])) / closes[-1]) < 0.05:
        patterns.append("דגל שורי (Bull Flag) 🚩")

    return patterns

# ------------------------------------------------------------------------------
# 5. ניתוח חדשות ממוקד קטליזטורים (קטגוריות איכות בלבד)
# ------------------------------------------------------------------------------
HIGH_IMPACT_CATALYSTS = {
    "אישור/ניסוי FDA": ["fda approval", "phase 3", "clinical trial results", "fda grants", "breakthrough therapy"],
    "דוחות מעל התחזיות": ["earnings beat", "eps beat", "raises guidance", "record revenue", "q1 beat", "q2 beat", "q3 beat", "q4 beat"],
    "עסקאות ענק / מיזוגים": ["to be acquired", "merger agreement", "buyout", "billion contract", "pentagon contract"]
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

    news_score = 0
    if len(matched_categories) >= 2: news_score = 20
    elif len(matched_categories) == 1: news_score = 15

    translated_headlines = []
    for h in meaningful_headlines[:2]:
        try:
            trans = translator.translate(h)
            translated_headlines.append(h if ("500" in trans or "Error" in trans) else trans)
        except Exception:
            translated_headlines.append(h)

    return {
        "score": news_score,
        "category": ", ".join(matched_categories) if matched_categories else "ללא זרז חדשותי חריג",
        "headlines": translated_headlines
    }

# ------------------------------------------------------------------------------
# 6. מנוע ניתוח ושיקלול נקודות (Scoring Engine)
# ------------------------------------------------------------------------------
def analyze_and_score_stock(symbol: str) -> Optional[dict]:
    try:
        df = yf.Ticker(symbol).history(period="1y")
        if df.empty or len(df) < 100: return None

        scale_factor = 100.0 if symbol.endswith(".TA") else 1.0

        curr_price = float(df['Close'].iloc[-1]) / scale_factor
        prev_price = float(df['Close'].iloc[-2]) / scale_factor
        change_pct = ((curr_price - prev_price) / prev_price) * 100

        if curr_price < CONFIG["MIN_PRICE"]: return None

        avg_vol_20 = df['Volume'].iloc[-21:-1].mean()
        curr_vol = df['Volume'].iloc[-1]
        daily_value_usd = (avg_vol_20 * curr_price)

        if daily_value_usd < CONFIG["MIN_DAILY_VALUE_USD"]: return None

        vol_ratio = (curr_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0

        scaled_close = df['Close'] / scale_factor
        scaled_high = df['High'] / scale_factor
        scaled_low = df['Low'] / scale_factor

        df['RSI'] = ta.rsi(scaled_close, length=14)
        df['EMA20'] = ta.ema(scaled_close, length=20)
        df['EMA50'] = ta.ema(scaled_close, length=50)
        df['EMA200'] = ta.ema(scaled_close, length=200)
        df['ATR'] = ta.atr(scaled_high, scaled_low, scaled_close, length=14)

        ema20 = float(df['EMA20'].iloc[-1])
        ema50 = float(df['EMA50'].iloc[-1])
        ema200 = float(df['EMA200'].iloc[-1]) if not pd.isna(df['EMA200'].iloc[-1]) else 0
        rsi = float(df['RSI'].iloc[-1])
        atr = float(df['ATR'].iloc[-1])

        high_20 = float(scaled_high.iloc[-21:-1].max())
        high_50 = float(scaled_high.iloc[-51:-1].max())

        # --- חישוב ניקוד משוקלל ---
        total_score = 0
        score_breakdown = []

        # 1. מגמה טכנית (עד 15 נק')
        if curr_price > ema20 > ema50 > ema200:
            total_score += 15
            score_breakdown.append("מגמה שורית מושלמת (15/15)")
        elif curr_price > ema20 > ema50:
            total_score += 10
            score_breakdown.append("מגמה שורית בינונית (10/15)")

        # 2. איכות פריצה (עד 20 נק')
        is_breakout = curr_price > high_20
        is_major_breakout = curr_price > high_50
        
        candle_body = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
        candle_range = df['High'].iloc[-1] - df['Low'].iloc[-1]
        is_strong_green_candle = (candle_range > 0) and (candle_body / candle_range >= 0.65) and (df['Close'].iloc[-1] > df['Open'].iloc[-1])

        if is_major_breakout and is_strong_green_candle:
            total_score += 20
            score_breakdown.append("פריצת שיא 50 ימים בנר עוצמתי (20/20)")
        elif is_breakout and is_strong_green_candle:
            total_score += 15
            score_breakdown.append("פריצת שיא 20 ימים בנר עוצמתי (15/20)")
        elif is_breakout:
            total_score += 8
            score_breakdown.append("פריצה טכנית קלה (8/20)")

        # 3. תבניות משלימות (עד 15 נק')
        chart_p = detect_chart_patterns(df)
        candle_p = detect_candlestick_patterns(df)
        all_patterns = chart_p + candle_p

        if len(chart_p) > 0 and len(candle_p) > 0:
            total_score += 15
            score_breakdown.append("תבנית צ'ארט + תבנית נרות (15/15)")
        elif len(all_patterns) > 0:
            total_score += 8
            score_breakdown.append("תבנית ניתוח טכני בודדת (8/15)")

        # 4. נפח מסחר יחסי - RVOL (עד 30 נק')
        if vol_ratio >= 2.5:
            total_score += 30
            score_breakdown.append(f"נפח חריג מאוד RVOL {vol_ratio:.1f}x (30/30)")
        elif vol_ratio >= 1.8:
            total_score += 20
            score_breakdown.append(f"נפח מסחר חזק RVOL {vol_ratio:.1f}x (20/30)")
        elif vol_ratio >= 1.3:
            total_score += 10
            score_breakdown.append(f"נפח מסחר בינוני RVOL {vol_ratio:.1f}x (10/30)")

        # 5. ניקוד חדשותי (עד 20 נק')
        news_data = analyze_news_catalysts(symbol)
        total_score += news_data["score"]
        if news_data["score"] > 0:
            score_breakdown.append(f"זרז חדשותי: {news_data['category']} ({news_data['score']}/20)")

        # 🎯 רף סף קשיח להתרעה!
        if total_score >= CONFIG["SCORE_THRESHOLD"]:
            stop_loss = round(curr_price - (1.5 * atr), 2)
            if stop_loss >= curr_price: stop_loss = round(curr_price * 0.95, 2)

            return {
                "symbol": symbol,
                "score": total_score,
                "score_breakdown": score_breakdown,
                "price": round(curr_price, 2),
                "change_pct": round(change_pct, 2),
                "rsi": round(rsi, 1),
                "vol_ratio": round(vol_ratio, 2),
                "patterns": all_patterns,
                "news": news_data,
                "entry": round(curr_price, 2),
                "stop_loss": stop_loss,
                "tp1": round(curr_price + (1.5 * (curr_price - stop_loss)), 2),
                "tp2": round(curr_price + (2.5 * (curr_price - stop_loss)), 2)
            }
        return None
    except Exception as e:
        logger.error(f"Error scoring {symbol}: {e}")
        return None

# ------------------------------------------------------------------------------
# 7. יצירת דוח התראה משוקלל
# ------------------------------------------------------------------------------
def build_breakout_report(ticker_info: dict, data: dict) -> Tuple[str, InlineKeyboardMarkup]:
    symbol = ticker_info["symbol"]
    curr_symbol = "$" if ticker_info.get("currency") == "USD" else "₪"
    
    breakdown_str = "\n".join([f"• {b}" for b in data["score_breakdown"]])
    news_str = "\n".join([f"• {h}" for h in data["news"]["headlines"]]) if data["news"]["headlines"] else "• לא זוהה זרז חדשותי חריג ב-72 השעות האחרונות."

    msg = f"""
🌟 <b>התראת איכות גבוהה - {symbol}</b> (ציון: <b>{data['score']}/100</b>)
<b>מדד שיוך:</b> {ticker_info.get('index', 'כללי')}

---
🏆 <b>שילוב פרמטרים שנמצאו:</b>
{breakdown_str}

---
📈 <b>נתונים טכניים ברגע הפריצה:</b>
• מחיר: <code>{curr_symbol}{data['price']}</code> ({'+' if data['change_pct']>0 else ''}{data['change_pct']}%)
• נפח מסחר יחסי (RVOL): <b>{data['vol_ratio']}x</b>
• RSI: <code>{data['rsi']}</code>

---
📰 <b>חדשות ואירועים:</b>
• קטגוריית זרז: <b>{data['news']['category']}</b>
{news_str}

---
🎯 <b>תוכנית מסחר מוצעת:</b>
• 🎯 מחיר כניסה: <code>{curr_symbol}{data['entry']}</code>
• 🛑 סטופ לוס: <code>{curr_symbol}{data['stop_loss']}</code>
• 🚀 יעד 1 (TP1): <code>{curr_symbol}{data['tp1']}</code>
• 🚀 יעד 2 (TP2): <code>{curr_symbol}{data['tp2']}</code>
"""

    markup = InlineKeyboardMarkup(row_width=2)
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף", url=f"https://www.tradingview.com/chart/?symbol={symbol}")
    btn_calc = InlineKeyboardButton("💰 חישוב עסקה", callback_data=f"calc_{symbol}_{data['entry']}_{data['stop_loss']}")
    markup.add(btn_chart, btn_calc)

    return msg, markup

# ------------------------------------------------------------------------------
# 8. מנוע סריקה
# ------------------------------------------------------------------------------
def process_single_stock(item: dict):
    symbol = item["symbol"]
    if is_in_cooldown(symbol): return

    result = analyze_and_score_stock(symbol)
    if result:
        msg, markup = build_breakout_report(item, result)
        record_signal(symbol)
        
        for chat_id in get_all_users():
            try:
                bot.send_message(chat_id, msg, parse_mode="HTML", reply_markup=markup)
            except Exception as e:
                logger.error(f"Failed sending alert to {chat_id}: {e}")

def run_auto_scan_job():
    with SCAN_LOCK:
        if SCAN_STATS["is_running"]: return
        SCAN_STATS["is_running"] = True
        SCAN_STATS["last_run_start"] = datetime.datetime.now()

    logger.info("🚀 מפעיל סריקה משוקללת ברקע...")
    try:
        tickers = fetch_all_index_tickers()
        with ThreadPoolExecutor(max_workers=5) as executor:
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
# 9. מחשבון ופקודות טלגרם
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
        "<b>בוט סריקת מניות מבוסס מנוע ניקוד (Scoring Engine) פעיל! 🚀</b>\n\n"
        "המערכת מתריעה רק על מניות שקיבלו ציון משוקלל של <b>75 מתוך 100</b> ומעלה ברמת איכות גבוהה.\n\n"
        "💡 <b>פקודות:</b>\n"
        "/scan - הרצת סריקה ידנית ברקע\n"
        "/tech SYMBOL - ניתוח ושיקלול למניה בודדת (למשל: <code>/tech AAPL</code>)\n"
        "/status - בדיקת סטטוס המערכת",
        parse_mode="HTML"
    )

@bot.message_handler(commands=['tech'])
def cmd_tech(message):
    add_user(message.chat.id)
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "⚠️ נא להזין סימול מניה. לדוגמה: <code>/tech AAPL</code>", parse_mode="HTML")
        return

    symbol = parts[1].upper().strip()
    bot.reply_to(message, f"🔍 מריץ מנוע ניקוד מקיף עבור <b>{symbol}</b>...", parse_mode="HTML")

    def run_single():
        res = analyze_and_score_stock(symbol)
        if not res:
            bot.send_message(message.chat.id, f"❌ המניה <b>{symbol}</b> לא עברה את רף האיכות (קיבלה מתחת ל-75 נקודות או שאינה נזילה מספיק).", parse_mode="HTML")
            return

        item = {
            "symbol": symbol,
            "index": "תל אביב 125" if symbol.endswith(".TA") else "ארה\"ב",
            "currency": "ILS" if symbol.endswith(".TA") else "USD"
        }
        msg, markup = build_breakout_report(item, res)
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)

    threading.Thread(target=run_single, daemon=True).start()

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    add_user(message.chat.id)
    bot.reply_to(message, "🔍 מפעיל סריקה משוקללת ברקע... התראות יישלחו רק על מניות בציון 75+.")
    threading.Thread(target=run_auto_scan_job, daemon=True).start()

@bot.message_handler(commands=['status'])
def cmd_status(message):
    add_user(message.chat.id)
    with SCAN_LOCK:
        is_run = SCAN_STATS["is_running"]
        last = SCAN_STATS["last_run_start"]
    bot.reply_to(message, f"🩺 <b>סטטוס סורק:</b>\n• סריקה ברקע כעת: {'כן ⏳' if is_run else 'לא'}\n• ריצה אחרונה: {last or 'טרם רצה'}", parse_mode="HTML")

# ------------------------------------------------------------------------------
# 10. הרצה
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
    logger.info("🤖 Scoring-Engine Bot started successfully...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
