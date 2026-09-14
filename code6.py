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

translator = GoogleTranslator(source='auto', target='he')

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
# 3. רשימת נכסים (S&P 500, NASDAQ 100, תל אביב 125)
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

    for sym in TA_125_TICKERS:
        if sym not in seen:
            results.append({"symbol": sym, "index": "תל אביב 125", "currency": "ILS"})
            seen.add(sym)

    try:
        url_sp = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp = pd.read_csv(url_sp, timeout=5)
        for _, row in df_sp.iterrows():
            sym = str(row['Symbol']).replace('.', '-').strip().upper()
            if sym not in seen:
                results.append({"symbol": sym, "index": "S&P 500", "currency": "USD"})
                seen.add(sym)
    except Exception as e:
        logger.warning(f"Failed loading S&P 500 dynamically: {e}")

    try:
        url_nasdaq = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq100/nasdaq100.csv"
        df_nasdaq = pd.read_csv(url_nasdaq, timeout=5)
        for _, row in df_nasdaq.iterrows():
            sym = str(row['symbol']).replace('.', '-').strip().upper() if 'symbol' in row else str(row.iloc[0]).strip().upper()
            if sym not in seen:
                results.append({"symbol": sym, "index": "NASDAQ 100", "currency": "USD"})
                seen.add(sym)
    except Exception as e:
        logger.warning(f"Failed loading NASDAQ 100 dynamically: {e}")

    if len(results) <= len(TA_125_TICKERS):
        for sym in FALLBACK_US_TICKERS:
            if sym not in seen:
                results.append({"symbol": sym, "index": "S&P 500 / NASDAQ 100", "currency": "USD"})
                seen.add(sym)

    TICKERS_CACHE["tickers"] = results
    TICKERS_CACHE["fetched_at"] = now
    return results

# ------------------------------------------------------------------------------
# 4. מנוע ניתוח חדשות, תרגום וסיווג פוטנציאל השקעה
# ------------------------------------------------------------------------------
HIGH_IMPACT_CATALYSTS = {
    "ביטחוני / גיאופוליטי 🛡️": ["military", "defense", "pentagon", "contract", "war", "sanctions", "army", "weapon"],
    "אישורי FDA / ניסויים קליניים 🔬": ["fda", "approval", "phase 2", "phase 3", "clinical trial", "patent", "drug"],
    "דוחות כספיים / מיזוגים ורכישות 💰": ["acquisition", "merger", "buyout", "earnings", "eps", "revenue", "guidance", "investment", "profit", "growth", "surpassed", "beat"]
}

BULLISH_KEYWORDS = ["beat", "growth", "approval", "contract", "surge", "record", "profit", "bullish", "higher", "partnership", "upgrade"]
BEARISH_KEYWORDS = ["miss", "drop", "loss", "decline", "investigation", "lawsuit", "down", "downgrade", "slash", "fail", "warning"]

def translate_to_hebrew(text: str) -> str:
    try:
        if not text:
            return ""
        return translator.translate(text)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def analyze_news_catalysts(symbol: str) -> dict:
    raw_articles = []
    clean_symbol = symbol.replace(".TA", "")

    if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
        try:
            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={from_date}&to={today.strftime('%Y-%m-%d')}&token={FINNHUB_API_KEY}"
            res = requests.get(news_url, timeout=4)
            if res.status_code == 200 and isinstance(res.json(), list):
                for item in res.json()[:5]:
                    if isinstance(item, dict) and item.get("headline"):
                        raw_articles.append(str(item.get("headline")).strip())
        except Exception as e:
            logger.warning(f"Finnhub error for {symbol}: {e}")

    if len(raw_articles) < 3:
        try:
            news_items = yf.Ticker(symbol).news
            if news_items and isinstance(news_items, list):
                for item in news_items[:5]:
                    if isinstance(item, dict):
                        title = item.get("title") or (item.get("content", {}).get("title") if isinstance(item.get("content"), dict) else None)
                        if title and str(title).strip() not in raw_articles:
                            raw_articles.append(str(title).strip())
        except Exception as e:
            logger.warning(f"yFinance news error for {symbol}: {e}")

    matched_categories = set()
    translated_headlines = []
    bullish_score = 0
    bearish_score = 0

    for headline in raw_articles[:5]:
        headline_lower = headline.lower()
        
        for cat, keywords in HIGH_IMPACT_CATALYSTS.items():
            if any(kw in headline_lower for kw in keywords):
                matched_categories.add(cat)

        for kw in BULLISH_KEYWORDS:
            if kw in headline_lower:
                bullish_score += 1
        for kw in BEARISH_KEYWORDS:
            if kw in headline_lower:
                bearish_score += 1

        hebrew_title = translate_to_hebrew(headline)
        translated_headlines.append({"original": headline, "hebrew": hebrew_title})

    if bullish_score > bearish_score:
        potential = "🟢 **פוטנציאל חיובי (Bullish):** החדשות מעידות על זרז חיובי (גידול, חוזים או דוחות טובים) שעשוי לתמוך בעליית המניה."
    elif bearish_score > bullish_score:
        potential = "🔴 **פוטנציאל שלילי (Bearish):** החדשות כוללות סימנים מחלישים (ירידה, פספוס תחזיות או תביעות). מומלץ להיזהר."
    elif matched_categories:
        potential = "🟡 **פוטנציאל מעורב / ניטרלי:** קיימות ידיעות מהותיות על החברה, אך יש להמתין לאישור טכני בגרף."
    else:
        potential = "⚪ **ללא זרז חדשותי מהותי:** לא אותרו כותרות חריגות בימים האחרונים. הניתוח נשען בעיקרו על האינדיקטורים הטכניים."

    return {
        "category": ", ".join(matched_categories) if matched_categories else "כללי / ללא זרז מיוחד",
        "headlines": translated_headlines,
        "potential_analysis": potential
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
            reasons.append(f"✅ **זיהוי פריצה:** המחיר ({curr_price:.2f}) פרץ שיא {'50' if curr_price >= high_50 else '20'} ימים.")
        else:
            reasons.append(f"❌ **אין פריצה:** המחיר ({curr_price:.2f}) מתחת לשיא 20 ימים ({high_20:.2f}) ו-50 ימים ({high_50:.2f}).")

        if is_uptrend:
            reasons.append(f"✅ **מגמה עולה:** מחיר ({curr_price:.2f}) > EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}).")
        else:
            reasons.append(f"❌ **אין מגמה עולה ברורה:** לא מתקיים Price > EMA20 > EMA50.")

        if is_high_volume:
            reasons.append(f"✅ **נפח מסחר חזק (RVOL):** {vol_ratio:.2f}x מעל ממוצע 20 יום.")
        else:
            reasons.append(f"❌ **נפח מסחר חלש:** RVOL עומד על {vol_ratio:.2f}x (נדרש מעל {CONFIG['MIN_RVOL']}x).")

        if is_valid_rsi:
            reasons.append(f"✅ **מדד RSI תקין:** {rsi:.1f} (בטווח המומלץ {CONFIG['MIN_RSI']}-{CONFIG['MAX_RSI']}).")
        else:
            reasons.append(f"❌ **מדד RSI חורג:** {rsi:.1f} (מחוץ לטווח הרצוי).")

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
            "news": news_data,
            "entry": round(curr_price, 2),
            "stop_loss": stop_loss,
            "tp1": round(curr_price + (1.5 * risk), 2),
            "tp2": round(curr_price + (2.5 * risk), 2)
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {"error": f"אירעה שגיאה בעיבוד הנתונים עבור {symbol}."}

# ------------------------------------------------------------------------------
# 6. בניית דוחות וקישורים ל-TradingView
# ------------------------------------------------------------------------------
def build_breakout_report(ticker_info: dict, data: dict) -> Tuple[str, InlineKeyboardMarkup]:
    symbol = ticker_info["symbol"]
    curr_symbol = "₪" if symbol.endswith(".TA") else "$"

    headlines_list = data["news"]["headlines"]
    if headlines_list:
        news_str = "\n".join([f"• {item['hebrew']}\n  _(מקור: {item['original']})_" for item in headlines_list[:3]])
    else:
        news_str = "• לא נמצאו חדשות עדכניות."

    chg_sign = "+" if data["change_pct"] > 0 else ""
    
    tv_symbol = symbol.replace(".TA", "") if not symbol.endswith(".TA") else f"TASE:{symbol.replace('.TA', '')}"
    chart_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

    is_recommended = data.get("is_valid", False)
    recommendation_text = "💡 **החלטה: שווה לבחון השקעה!** המניה עונה על כל הקריטריונים הטכניים לפריצה." if is_recommended else "⚠️ **החלטה: לא מומלץ להשקיע כעת.** המניה אינה מציגה תבנית פריצה מושלמת."

    reasons_text = "\n".join(data["reasons"])

    lines = [
        f"🔎 **ניתוח מניה מקיף - {symbol}**",
        f"**מדד:** {ticker_info.get('index', 'כללי')}",
        "",
        f"📌 {recommendation_text}",
        "",
        "📋 **נימוקי ההחלטה:**",
        reasons_text,
        "",
        "---",
        "📊 **נתונים טכניים:**",
        f"• מחיר נוכחי: `{curr_symbol}{data['price']}` ({chg_sign}{data['change_pct']}%)",
        f"• נפח מסחר יחסי (RVOL): **{data['vol_ratio']}x**",
        f"• מדד חוזק (RSI): `{data['rsi']}`",
        f"• ממוצעים נעים: `EMA20: {data['ema20']}` | `EMA50: {data['ema50']}`",
        "",
        "---",
        "📰 **פוטנציאל וזרזים חדשותיים:**",
        f"• **סיווג:** {data['news']['category']}",
        f"• {data['news']['potential_analysis']}",
        news_str,
        "",
        "---",
        "🎯 **תוכנית מסחר פוטנציאלית:**",
        f"• 🎯 מחיר כניסה: `{curr_symbol}{data['entry']}`",
        f"• 🛑 סטופ לוס (1.5xATR): `{curr_symbol}{data['stop_loss']}`",
        f"• 🚀 יעד 1 (TP1): `{curr_symbol}{data['tp1']}`",
        f"• 🚀 יעד 2 (TP2): `{curr_symbol}{data['tp2']}`"
    ]
    
    msg = "\n".join(lines)
    markup = InlineKeyboardMarkup(row_width=2)
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף ב-TradingView", url=chart_url)
    btn_calc = InlineKeyboardButton("💰 חישוב עסקה פוטנציאלית", callback_data=f"calc_{symbol}_{data['entry']}_{data['stop_loss']}")
    markup.add(btn_chart, btn_calc)

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

    logger.info("🚀 מתחיל סריקת מניות ברקע...")
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

        if target_chat_id and found_count == 0:
            bot.send_message(target_chat_id, "🔍 הסריקה הושלמה. לא נמצאו מניות חדשות שענו על כל תנאי הפריצה ברגע זה.")

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
        "👋 **ברוכים הבאים לסורק המניות האוטומטי!**\n\n"
        "הבוט סורק באופן רציף את מדדי **S&P 500, NASDAQ 100 ו-תל אביב 125** "
        "כדי לאתר מניות בפריצה טכנית יחד עם ניתוח חדשותי בעברית.\n\n"
        "🛠️ **פקודות זמינות:**\n"
        "• /start - הצגת הודעה זו\n"
        "• /scan - הרצת סריקה ידנית מיידית ברקע\n"
        "• /tech <SYMBOL> - ניתוח טכני, נימוקי קנייה/אי-קנייה, גרף ומחשבון עסקה\n"
        "• /news <SYMBOL> - ניתוח פוטנציאל חדשותי מתורגם לעברית\n"
        "• /status - בדיקת סטטוס סורק הרקע האוטומטי"
    )
    bot.send_message(message.chat.id, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['scan'])
def handle_scan(message):
    add_user(message.chat.id)
    bot.reply_to(message, "🔍 **סריקה ידנית הופעלה ברקע!**\nהמערכת מעבדת את המדדים ותשלח התראות...", parse_mode="Markdown")
    threading.Thread(target=run_scan_process, args=(message.chat.id,), daemon=True).start()

@bot.message_handler(commands=['tech'])
def handle_tech(message):
    add_user(message.chat.id)
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "⚠️ נא להזין סימול מניה. לדוגמה: `/tech AAPL` או `/tech TEVA.TA`", parse_mode="Markdown")
        return

    symbol = parts[1].upper().strip()
    bot.reply_to(message, f"🔍 מריץ ניתוח מקיף ונימוק עבור **{symbol}**...", parse_mode="Markdown")

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
        bot.reply_to(message, "⚠️ נא להזין סימול מניה. לדוגמה: `/news TSLA`", parse_mode="Markdown")
        return

    symbol = parts[1].upper().strip()
    bot.reply_to(message, f"📰 סורק ומנתח חדשות בעברית עבור **{symbol}**...", parse_mode="Markdown")

    def run_news_single():
        try:
            news_data = analyze_news_catalysts(symbol)
            headlines = news_data["headlines"]
            
            if headlines:
                news_str = "\n\n".join([f"• **{item['hebrew']}**\n  _(מקור: {item['original']})_" for item in headlines])
            else:
                news_str = "• לא נמצאו ידיעות חדשותיות אחרונות עבור מניה זו."

            tv_symbol = symbol.replace(".TA", "") if not symbol.endswith(".TA") else f"TASE:{symbol.replace('.TA', '')}"
            chart_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

            lines = [
                f"📰 **ניתוח פוטנציאל חדשותי - {symbol}**",
                "",
                f"🏷️ **סיווג זרז:** {news_data['category']}",
                "",
                f"💡 **ניתוח פוטנציאל השקעה:**\n{news_data['potential_analysis']}",
                "",
                "---",
                "<b>כותרות אחרונות (מתורגמות לעברית):</b>",
                news_str
            ]
            msg = "\n".join(lines)
            markup = InlineKeyboardMarkup()
            btn_chart = InlineKeyboardButton("📈 צפייה בגרף חי ב-TradingView", url=chart_url)
            markup.add(btn_chart)

            bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
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
        f"🩺 **סטטוס מערכת:**\n"
        f"• סורק מורץ כעת: {'כן ⏳' if is_run else 'לא (ממתין) 🟢'}\n"
        f"• זמן ריצה אחרון: `{last_str}`\n"
        f"• סטטוס: {status_text}\n"
        f"• תדר סריקה אוטומטית: כל 15 דקות"
    )
    bot.reply_to(message, msg, parse_mode="Markdown")

# ------------------------------------------------------------------------------
# 10. מחשבון סימולציית עסקה
# ------------------------------------------------------------------------------
@bot.callback_query_handler(func=lambda call: call.data.startswith("calc_"))
def handle_calc_callback(call):
    try:
        _, symbol, entry, sl = call.data.split("_")
        USER_CALC_STATE[call.message.chat.id] = {"symbol": symbol, "entry": float(entry), "sl": float(sl)}
        curr_symbol = "₪" if symbol.endswith(".TA") else "$"
        
        bot.send_message(
            call.message.chat.id,
            f"💰 **מחשבון סימולציית עסקה פוטנציאלית עבור {symbol}**\n\n"
            f"מחיר כניסה: `{curr_symbol}{entry}` | סטופ לוס: `{curr_symbol}{sl}`\n\n"
            f"אנא הזיני את תקציב ההשקעה המתוכנן (לדוגמה: `3000`):",
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

        USER_CALC_STATE.pop(message.chat.id, None)

        total_inv = shares * entry
        risk_per_share = entry - sl
        total_risk = shares * risk_per_share
        
        tp1 = entry + (1.5 * risk_per_share)
        tp2 = entry + (2.5 * risk_per_share)

        curr_symbol = "₪" if state['symbol'].endswith(".TA") else "$"

        lines = [
            f"📐 **תוצאות סימולציית עסקה עבור {state['symbol']}**",
            "",
            f"• **כמות מניות לקנייה:** `{shares}` מניות",
            f"• **סך השקעה בפועל:** {curr_symbol}{total_inv:,.2f}",
            f"• **סיכון כולל בסטופ לוס:** -{curr_symbol}{total_risk:,.2f}",
            "",
            "---",
            "**🎯 צפי רווח ביעדים:**",
            f"• **יעד 1 ({curr_symbol}{tp1:.2f}):** רווח צפוי של **+{curr_symbol}{(shares * (tp1 - entry)):,.2f}**",
            f"• **יעד 2 ({curr_symbol}{tp2:.2f}):** רווח צפוי של **+{curr_symbol}{(shares * (tp2 - entry)):,.2f}**"
        ]
        reply = "\n".join(lines)
        bot.send_message(message.chat.id, reply, parse_mode="Markdown")
    except ValueError:
        bot.reply_to(message, "⚠️ נא להזין מספר בלבד (תקציב).")

# ------------------------------------------------------------------------------
# 11. הרצה
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()
    
    threading.Thread(target=run_scan_process, daemon=True).start()
    
    logger.info("🤖 Bot active...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
