import os
import sqlite3
import datetime
import pytz
import logging
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import telebot
from telebot.types import BotCommand
from apscheduler.schedulers.background import BackgroundScheduler
from deep_translator import GoogleTranslator
from flask import Flask

# try importing psycopg2 for PostgreSQL support in production
try:
    import psycopg2
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FLASK APP FOR RENDER HEALTH CHECK ---
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Bot is running successfully!", 200

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
DATABASE_URL = os.getenv("DATABASE_URL")  # PostgreSQL URL in production (e.g., Render/Supabase)
MAX_USERS = 50

# Init Telegram Bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# --- DATABASE MANAGEMENT ---
def get_db_connection():
    """Returns a connection to PostgreSQL if DATABASE_URL is present, otherwise SQLite."""
    if DATABASE_URL and HAS_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        return conn, "postgres"
    else:
        conn = sqlite3.connect("bot_database.db")
        return conn, "sqlite"

def init_db():
    """Initializes tables for registered users, alerts cooldown, and personal watchlists."""
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    # Registered Users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            chat_id BIGINT PRIMARY KEY,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Alert History Cooldown
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            symbol VARCHAR(20) PRIMARY KEY,
            last_alert_time TIMESTAMP,
            breakout_type VARCHAR(20)
        )
    """)
    
    # User Watchlist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlists (
            chat_id BIGINT,
            symbol VARCHAR(20),
            PRIMARY KEY (chat_id, symbol)
        )
    """)
    
    conn.commit()
    conn.close()

# --- MARKET HOURS CHECKING ---
def is_market_open():
    """Checks if US or TASE markets are currently open."""
    israel_tz = pytz.timezone('Asia/Jerusalem')
    now = datetime.datetime.now(israel_tz)
    weekday = now.weekday()  # Monday = 0, ..., Sunday = 6
    current_time = now.time()

    # TASE: Sunday (6) to Thursday (3), 10:00 to 17:25
    tase_open = datetime.time(10, 0)
    tase_close = datetime.time(17, 25)
    is_tase_open = (weekday in [6, 0, 1, 2, 3]) and (tase_open <= current_time <= tase_close)

    # US (NYSE/NASDAQ): Monday (0) to Friday (4), 16:30 to 23:00 (Israel Time)
    us_open = datetime.time(16, 30)
    us_close = datetime.time(23, 0)
    is_us_open = (weekday in [0, 1, 2, 3, 4]) and (us_open <= current_time <= us_close)

    return is_tase_open or is_us_open

# --- TICKERS & SCRAPING ---
def get_tase_tickers():
    """Fetches TASE tickers using official TASE API or fallback list."""
    try:
        url = "https://api.tase.co.il/api/content/securities"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            tickers = [f"{item['Symbol']}.TA" for item in data if 'Symbol' in item]
            if len(tickers) > 10:
                return tickers
    except Exception as e:
        logging.warning(f"TASE API fetch failed ({e}). Falling back to static list.")
    
    return ["NICE.TA", "TEVA.TA", "LUMI.TA", "POLI.TA", "ICL.TA", "ALHE.TA", "ELBT.TA", "MZTF.TA"]

def get_us_sp500_tickers():
    """Scrapes S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception as e:
        logging.error(f"Error scraping S&P 500 tickers: {e}")
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

def get_all_tickers():
    """Combines TASE and S&P 500 tickers."""
    tase = get_tase_tickers()
    sp500 = get_us_sp500_tickers()
    return list(set(tase + sp500))

# --- TECHNICAL ANALYSIS & SIGNALS ---
def analyze_stock_data(symbol, df):
    if df is None or len(df) < 50:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=1, axis=1) if symbol in df.columns.levels[1] else df

    df = df.dropna()
    if len(df) < 50:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    current_price = close.iloc[-1]
    current_volume = volume.iloc[-1]
    avg_volume_20 = volume.iloc[-21:-1].mean()

    if current_volume < (1.5 * avg_volume_20):
        return None

    high_20 = high.iloc[-21:-1].max()
    high_50 = high.iloc[-51:-1].max()

    breakout_type = None
    if current_price > high_50:
        breakout_type = "50_DAY"
    elif current_price > high_20:
        breakout_type = "20_DAY"

    if not breakout_type:
        return None

    # Calculate ATR (14-period)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    lowest_low_10 = low.iloc[-10:].min()
    atr_stop = current_price - (1.5 * atr)
    
    stop_loss = min(atr_stop, lowest_low_10)
    risk = current_price - stop_loss
    if risk <= 0:
        return None

    take_profit = current_price + (2.0 * risk)

    return {
        "symbol": symbol,
        "price": current_price,
        "breakout_type": breakout_type,
        "volume_ratio": current_volume / avg_volume_20,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": 2.0
    }

# --- COOLDOWN LOGIC ---
def should_send_alert(symbol, breakout_type, volume_ratio):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    param = "%s" if db_type == "postgres" else "?"

    cursor.execute(f"SELECT last_alert_time, breakout_type FROM alert_history WHERE symbol = {param}", (symbol,))
    row = cursor.fetchone()
    
    now = datetime.datetime.now()
    if row:
        last_time, prev_breakout = row[0], row[1]
        if isinstance(last_time, str):
            last_time = datetime.datetime.fromisoformat(last_time)
            
        time_diff = (now - last_time).total_seconds() / 3600.0

        if time_diff < 4.0:
            if prev_breakout == "20_DAY" and breakout_type == "50_DAY" and volume_ratio >= 2.5:
                logging.info(f"Dynamic Cooldown Override triggered for {symbol}!")
            else:
                conn.close()
                return False

    if db_type == "postgres":
        cursor.execute("""
            INSERT INTO alert_history (symbol, last_alert_time, breakout_type)
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE 
            SET last_alert_time = EXCLUDED.last_alert_time, breakout_type = EXCLUDED.breakout_type
        """, (symbol, now, breakout_type))
    else:
        cursor.execute("""
            INSERT OR REPLACE INTO alert_history (symbol, last_alert_time, breakout_type)
            VALUES (?, ?, ?)
        """, (symbol, now, breakout_type))

    conn.commit()
    conn.close()
    return True

# --- SCANNING ENGINE ---
def run_market_scan():
    if not is_market_open():
        logging.info("Markets are currently closed. Skipping scan.")
        return

    logging.info("Starting batch market scan...")
    tickers = get_all_tickers()
    
    try:
        data = yf.download(tickers, period="60d", interval="1d", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        logging.error(f"Batch download failed: {e}")
        return

    translator = GoogleTranslator(source='auto', target='he')
    alerts_to_send = []

    for symbol in tickers:
        try:
            symbol_df = data[symbol] if symbol in data else None
            signal = analyze_stock_data(symbol, symbol_df)
            
            if signal and should_send_alert(symbol, signal["breakout_type"], signal["volume_ratio"]):
                alerts_to_send.append(signal)
        except Exception:
            continue

    if alerts_to_send:
        broadcast_alerts(alerts_to_send, translator)

def broadcast_alerts(signals, translator):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT chat_id FROM users")
    users = [row[0] for row in cursor.fetchall()]

    for sig in signals:
        breakout_str = "פריצת שיא 50 ימים 🚀" if sig["breakout_type"] == "50_DAY" else "פריצת שיא 20 ימים 📈"
        
        msg = (
            f"🚨 **אות פריצה זוהה!**\n\n"
            f"📌 **מניה:** {sig['symbol']}\n"
            f"📊 **סוג פריצה:** {breakout_str}\n"
            f"💵 **מחיר נוכחי:** ${sig['price']:.2f}\n"
            f"🔊 **נפח מסחר:** פי {sig['volume_ratio']:.1f} מהממוצע\n\n"
            f"🛡 **Stop Loss (משולב ATR ושפל):** ${sig['stop_loss']:.2f}\n"
            f"🎯 **Take Profit (יחס 1:2):** ${sig['take_profit']:.2f}\n"
            f"⚖ **יחס סיכון/סיכוי:** 1:2\n\n"
            f"⚠️ *דיסקליימר: אין לראות באמור המלצה לביצוע פעולות מסחר.*"
        )

        for chat_id in users:
            try:
                bot.send_message(chat_id, msg, parse_mode="Markdown")
            except Exception as e:
                logging.error(f"Failed to send alert to {chat_id}: {e}")

    conn.close()

# --- TELEGRAM BOT HANDLERS ---
@bot.message_handler(commands=['start'])
def handle_start(message):
    chat_id = message.chat.id
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    param = "%s" if db_type == "postgres" else "?"

    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]

    cursor.execute(f"SELECT chat_id FROM users WHERE chat_id = {param}", (chat_id,))
    already_reg = cursor.fetchone()

    if already_reg:
        bot.reply_to(message, "אתה כבר רשום למערכת! תקבל התראות בזמן אמת. 📈")
    elif user_count >= MAX_USERS:
        bot.reply_to(message, "מצטערים, המערכת הגיעה למכסת המשתמשים המרבית (50 משתמשים).")
    else:
        cursor.execute(f"INSERT INTO users (chat_id) VALUES ({param})", (chat_id,))
        conn.commit()
        bot.reply_to(message, "ברוך הבא! נרשמת בהצלחה לקבלת התראות פריצה בזמן אמת. 🚀")

    conn.close()

@bot.message_handler(commands=['watch'])
def handle_watch(message):
    chat_id = message.chat.id
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "אנא ציין סימול מניה. לדוגמה:\n`/watch NVDA` או `/watch TEVA.TA`", parse_mode="Markdown")
        return

    symbol = args[1].upper()
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    param = "%s" if db_type == "postgres" else "?"

    try:
        cursor.execute(f"INSERT INTO watchlists (chat_id, symbol) VALUES ({param}, {param})", (chat_id, symbol))
        conn.commit()
        bot.reply_to(message, f"המניה **{symbol}** נוספה בהצלחה לרשימת המעקב האישית שלך! 👁", parse_mode="Markdown")
    except Exception:
        bot.reply_to(message, f"המניה {symbol} כבר נמצאת ברשימת המעקב שלך.")
    finally:
        conn.close()

@bot.message_handler(commands=['unwatch'])
def handle_unwatch(message):
    chat_id = message.chat.id
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "אנא ציין סימול מניה להסרה. לדוגמה:\n`/unwatch NVDA`", parse_mode="Markdown")
        return

    symbol = args[1].upper()
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    param = "%s" if db_type == "postgres" else "?"

    cursor.execute(f"DELETE FROM watchlists WHERE chat_id = {param} AND symbol = {param}", (chat_id, symbol))
    conn.commit()
    conn.close()
    bot.reply_to(message, f"המניה **{symbol}** הוסרה מרשימת המעקב שלך.", parse_mode="Markdown")

@bot.message_handler(commands=['mywatchlist'])
def handle_my_watchlist(message):
    chat_id = message.chat.id
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    param = "%s" if db_type == "postgres" else "?"

    cursor.execute(f"SELECT symbol FROM watchlists WHERE chat_id = {param}", (chat_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        bot.reply_to(message, "רשימת המעקב שלך ריקה. להוספה השתמש ב-`/watch SYMBOL`", parse_mode="Markdown")
    else:
        symbols_str = "\n".join([f"• `{r[0]}`" for r in rows])
        bot.reply_to(message, f"📋 **רשימת המעקב האישית שלך:**\n\n{symbols_str}", parse_mode="Markdown")

def setup_bot_commands():
    commands = [
        BotCommand("start", "הרשמה לקבלת התראות פריצה"),
        BotCommand("watch", "הוספת מניה לרשימת המעקב האישית"),
        BotCommand("unwatch", "הסרת מניה מרשימת המעקב האישית"),
        BotCommand("mywatchlist", "הצגת רשימת המעקב האישית שלי")
    ]
    try:
        bot.set_my_commands(commands)
    except Exception as e:
        logging.error(f"Failed to set bot commands: {e}")

def run_bot():
    bot.infinity_polling()

# --- INITIALIZATION ---
init_db()
setup_bot_commands()

scheduler = BackgroundScheduler(timezone="Asia/Jerusalem")
scheduler.add_job(run_market_scan, 'interval', minutes=15)
scheduler.start()

# Start bot polling in a separate background thread
bot_thread = threading.Thread(target=run_bot, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    # Local execution fallback
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
