import os
import time
import sqlite3
import threading
import requests
import datetime
import pytz
import re
from concurrent.futures import ThreadPoolExecutor
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

# ------------------------------------------------------------------------------
# 2. שליפת רשימת מניות מורחבת (S&P 500 + NASDAQ 100)
# ------------------------------------------------------------------------------
def fetch_market_tickers() -> list:
    tickers = set()
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers.update(sp500_table['Symbol'].str.replace('.', '-').tolist())
    except Exception as e:
        print(f"[Ticker Fetch Error - S&P500]: {e}")

    try:
        nasdaq100_table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
        tickers.update(nasdaq100_table['Ticker'].str.replace('.', '-').tolist())
    except Exception as e:
        print(f"[Ticker Fetch Error - NASDAQ100]: {e}")

    if not tickers:
        tickers = {
            "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", "META", "GOOGL", "NFLX",
            "LLY", "AVGO", "JPM", "UNH", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV"
        }
    return sorted(list(tickers))

# ------------------------------------------------------------------------------
# 3. ניהול בסיס נתונים (SQLite Database + מעקב התראות יומי)
# ------------------------------------------------------------------------------
DB_FILE = "bot_database.db"

def init_db():
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
        return [r[0] for r in cursor.fetchall()]

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
# 4. שרת FLASK ומנגנון KEEP-ALIVE
# ------------------------------------------------------------------------------
@app.route('/')
def health_check():
    return "OK - Telegram Trading Bot is Running!", 200

def keep_alive_ping():
    while True:
        try:
            time.sleep(600)
            if "localhost" not in SELF_URL:
                requests.get(SELF_URL, timeout=10)
        except Exception as e:
            print(f"[Keep-Alive Error]: {e}")

# ------------------------------------------------------------------------------
# 5. מנוע ניתוח טכני
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
        if df.empty or len(df) < 100:
            return None

        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change_pct = ((current_price - prev_price) / prev_price) * 100

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

        is_uptrend = current_price > ema20 > ema50 and current_price > ema200

        avg_vol_20 = df['Volume'].iloc[-21:-1].mean()
        curr_vol = df['Volume'].iloc[-1]
        volume_spike = curr_vol > (avg_vol_20 * 1.4)

        patterns, candlesticks = [], []

        o1, h1, l1, c1 = float(df['Open'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
        o2, c2 = float(df['Open'].iloc[-2]), float(df['Close'].iloc[-2])
        body1 = abs(c1 - o1)
        lower_shadow1 = min(o1, c1) - l1
        upper_shadow1 = h1 - max(o1, c1)

        if lower_shadow1 >= (2.5 * body1) and upper_shadow1 <= (0.2 * body1) and body1 > 0:
            candlesticks.append("נר פטיש היפוכי (Hammer)")

        if c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2 and volume_spike:
            candlesticks.append("בליעה שורית מאושרת נפח (Bullish Engulfing)")

        high_50 = float(df['High'].iloc[-50:-5].max())
        recent_support = float(df['Low'].tail(15).min())

        lows = df['Low'].iloc[-50:-10]
        first_bottom = lows.min()
        second_bottom_candidates = df['Low'].iloc[-20:-2]
        if len(second_bottom_candidates) > 0:
            second_bottom = second_bottom_candidates.min()
            if abs(first_bottom - second_bottom) / first_bottom < 0.02:
                neckline = df['High'].loc[df['Low'].iloc[-50:].idxmin():].max()
                if current_price > neckline and prev_price <= neckline:
                    patterns.append("פריצת תחתית כפולה (Double Bottom Breakout)")

        if current_price > high_50 and prev_price <= high_50 and volume_spike:
            patterns.append("פריצת ספל וידית (Cup & Handle Breakout)")

        has_breakout = len(patterns) > 0 or len(candlesticks) > 0
        is_strong_buy = is_uptrend and has_breakout and volume_spike and (45 <= rsi <= 68)

        if is_strong_buy:
            signal = "BUY"
            entry_price = round(current_price, 2)
            stop_loss = round(max(recent_support * 0.99, entry_price - (1.5 * atr)), 2)
            if stop_loss >= entry_price:
                stop_loss = round(entry_price - (1.5 * atr), 2)
        elif has_breakout or is_uptrend:
            signal = "HOLD"
            entry_price, stop_loss = None, None
        else:
            signal = "NEUTRAL"
            entry_price, stop_loss = None, None

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
            "signal": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "atr": round(atr, 2)
        }

        CACHE[symbol] = (now, result)
        return result

    except Exception as e:
        print(f"[Tech Analysis Error] {symbol}: {e}")
        return None

# ------------------------------------------------------------------------------
# 6. מנוע ניתוח חדשות (עדכניות 48h, איכות וסנטימנט)
# ------------------------------------------------------------------------------
HIGH_IMPACT_CATALYSTS = {
    r"\bfda\b|\btrial\b|\bphase\b|\bclinical\b|\bapproval\b": ("אישור/ניסוי קליני (FDA/Pharma)", 10),
    r"\bearnings\b|\bbeat\b|\brevenue beat\b|\brecord revenue\b": ("דוחות כספיים / תוצאות שיא 📈", 9),
    r"\bguidance\b|\braises outlook\b|\braised guidance\b": ("עדכון תחזית צמיחה כלפי מעלה 🚀", 9),
    r"\bmerger\b|\bacquisition\b|\bbuyout\b": ("עסקת מיזוג / רכישה דרמטית 🤝", 9),
    r"\bcontract\b|\bdeal\b|\bpartnership\b": ("חתימת חוזה אסטרטגי / הספקת ענק 📝", 8),
    r"\bshare buyback\b|\brepurchase program\b": ("תוכנית רכישה עצמית (Buyback) 💵", 8)
}

def is_headline_relevant_and_fresh(headline: str, symbol: str, company_name: str, pub_date: datetime.datetime = None) -> tuple:
    if pub_date:
        now_utc = datetime.datetime.now(pytz.utc)
        if (now_utc - pub_date).total_seconds() > 172800:
            return 0, None

    h_lower = headline.lower()
    sym_lower = symbol.lower()
    comp_lower = company_name.lower() if company_name else sym_lower

    has_entity = (re.search(r'\b' + re.escape(sym_lower) + r'\b', h_lower) or 
                  (len(comp_lower) > 3 and comp_lower in h_lower))
    if not has_entity:
        return 0, None

    for pattern, (label, score) in HIGH_IMPACT_CATALYSTS.items():
        if re.search(pattern, h_lower):
            return score, label

    return 0, None

def fetch_finnhub_data(symbol: str) -> dict:
    raw_articles = []
    company_name = symbol

    try:
        t_info = yf.Ticker(symbol).info
        company_name = t_info.get("shortName", symbol).split()[0]
    except Exception:
        pass

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
                    if h:
                        raw_articles.append((h, pub_dt))
        except Exception as e:
            print(f"[Finnhub Fetch Error] {symbol}: {e}")

    if not raw_articles:
        try:
            news_items = yf.Ticker(symbol).news
            if news_items:
                for item in news_items:
                    title = item.get("title", "")
                    if not title and "content" in item and isinstance(item["content"], dict):
                        title = item["content"].get("title", "")
                    pub_ts = item.get("providerPublishTime")
                    pub_dt = datetime.datetime.fromtimestamp(pub_ts, tz=pytz.utc) if pub_ts else None
                    if title:
                        raw_articles.append((title, pub_dt))
        except Exception as e:
            print(f"[YFinance News Error] {symbol}: {e}")

    scored_headlines = []
    found_catalysts = set()

    for headline, pub_date in raw_articles:
        score, catalyst_label = is_headline_relevant_and_fresh(headline, symbol, company_name, pub_date)
        if score >= 8:
            scored_headlines.append((score, headline))
            found_catalysts.add(catalyst_label)

    scored_headlines.sort(key=lambda x: x[0], reverse=True)
    top_headlines = [h[1] for h in scored_headlines[:2]]

    translated_headlines = []
    for h in top_headlines:
        try:
            translated_headlines.append(translator.translate(h))
        except Exception:
            translated_headlines.append(h)

    catalyst_str = " | ".join(found_catalysts) if found_catalysts else "לא אותרו קטליזטורים דרמטיים ב-48 השעות האחרונות"
    sentiment = "חיובי חזק 🟢" if found_catalysts else "ללא חדשות מהותיות ⚪"

    return {
        "headlines": translated_headlines,
        "catalyst": catalyst_str,
        "sentiment": sentiment
    }

# ------------------------------------------------------------------------------
# 7. חישוב יעדים וניהול סיכונים
# ------------------------------------------------------------------------------
def get_usd_ils_rate() -> float:
    try:
        usd_ticker = yf.Ticker("USDILS=X")
        rate = usd_ticker.history(period="1d")['Close'].iloc[-1]
        return round(float(rate), 2)
    except Exception:
        return 3.70

def calculate_trade_plan(entry_price: float, stop_loss: float) -> dict:
    risk_per_share = entry_price - stop_loss
    tp1_price = entry_price + (risk_per_share * 1.5)
    tp2_price = entry_price + (risk_per_share * 2.5)

    return {
        "entry": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "tp1": round(tp1_price, 2),
        "tp1_pct": round(((tp1_price - entry_price) / entry_price) * 100, 1),
        "tp2": round(tp2_price, 2),
        "tp2_pct": round(((tp2_price - entry_price) / entry_price) * 100, 1),
    }

def create_report_message(symbol: str) -> tuple:
    tech = analyze_technical_patterns(symbol)
    if not tech:
        return f"❌ לא ניתן היה לשלוף נתונים עבור המנייה <b>{symbol}</b>.", None

    finnhub = fetch_finnhub_data(symbol)

    if tech["signal"] == "BUY":
        recommendation = "🟢 <b>מומלץ להיכנס להשקעה (מגמה עולה + פריצה/נר מאושר)</b>"
    elif tech["signal"] == "HOLD":
        recommendation = "🟡 <b>להמתין ולעקוב (מגמה/תבנית בהתהוות ללא אישור נפח)</b>"
    else:
        recommendation = "🔴 <b>לא מומלץ כעת (ללא איתות פריצה)</b>"

    patterns_str = ", ".join(tech["patterns"]) if tech["patterns"] else "לא זוהו תבניות מיוחדות"
    headlines_str = "\n".join([f"• {h}" for h in finnhub["headlines"]]) if finnhub["headlines"] else "• לא אותרו כותרות איכותיות/עדכניות ב-48h האחרונות"

    msg = f"""
<b>📊 דוח ניתוח מקיף עבור {symbol}</b>

<b>💡 המלצה:</b>
{recommendation}

---
<b>📈 נתונים טכניים ומגמה:</b>
• מחיר נוכחי: <code>${tech['current_price']}</code> ({'+' if tech['change_pct']>0 else ''}{tech['change_pct']}%)
• RSI: <code>{tech['rsi']}</code> | EMA20: <code>${tech['ema20']}</code> | EMA50: <code>${tech['ema50']}</code>
• תבניות ומגמה: <b>{patterns_str}</b>
• נפח מסחר חורג (Volume Spike): {'כן 🟢' if tech['volume_accumulating'] else 'רגיל ⚪'}

---
<b>📰 חדשות וקטליזטורים מסוננים (48 שעות):</b>
• קטליזטור: <b>{finnhub['catalyst']}</b>
• סנטימנט: <b>{finnhub['sentiment']}</b>
<b>כותרות רלוונטיות שנבחרו:</b>
{headlines_str}

---
"""

    if tech["signal"] == "BUY":
        plan = calculate_trade_plan(tech["entry_price"], tech["stop_loss"])
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
        btn_calc = InlineKeyboardButton("💰 חישוב עסקה", callback_data=f"calc_{symbol}_{plan['entry']}_{plan['stop_loss']}")
        markup.add(btn_chart, btn_calc)
    else:
        markup.add(btn_chart)
        
    markup.add(btn_news, btn_tech)
    return msg, markup

# ------------------------------------------------------------------------------
# 8. מנוע סריקה אוטומטית שוטפת ברקע (Automated Market Scanner)
# ------------------------------------------------------------------------------
def is_market_open() -> bool:
    israel_tz = pytz.timezone('Asia/Jerusalem')
    now = datetime.datetime.now(israel_tz)
    
    # בורסות ארה"ב סגורות בסופי שבוע (שבת וראשון)
    if now.weekday() in (5, 6):
        return False

    # שעות מסחר לפי שעון ישראל: 16:30 עד 23:00
    start_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
    end_time = now.replace(hour=23, minute=0, second=0, microsecond=0)
    
    return start_time <= now <= end_time

def scan_worker_auto(symbol: str):
    """בודק מניה ושולח התראה מיידית אם זוהה BUY ולא הציג התראה היום"""
    if has_alerted_today(symbol):
        return

    tech = analyze_technical_patterns(symbol)
    if tech and tech["signal"] == "BUY":
        users = get_all_users()
        if not users:
            return

        msg, markup = create_report_message(symbol)
        alert_msg = f"🚨 <b>התראת איתות פריצה בזמן אמת!</b>\n{msg}"
        
        for chat_id in users:
            try:
                bot.send_message(chat_id, alert_msg, parse_mode="HTML", reply_markup=markup)
            except Exception as e:
                print(f"[Alert Error] Chat {chat_id}: {e}")

        mark_alerted_today(symbol)

def scheduled_market_scan():
    """ריצה מחזורית של הסורק ברקע ברציפות במהלך שעות המסחר"""
    if not is_market_open():
        return

    print(f"[{datetime.datetime.now()}] 🔄 מתחיל סריקה אוטומטית ברקע...")
    tickers = fetch_market_tickers()
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        executor.map(scan_worker_auto, tickers)

# הגדרת תזמון סריקה אוטומטית - ריצה כל 15 דקות
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(scheduled_market_scan, 'interval', minutes=15)
scheduler.start()

# ------------------------------------------------------------------------------
# 9. טיפול בפקודות Telegram (סריקות ידניות לפי דרישה)
# ------------------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def cmd_start(message):
    add_user(message.chat.id)
    bot.reply_to(message, "<b>ברוכים הבאים לסורק המניות האוטומטי! 🚀</b>\n\nהבוט סורק באופן רציף את שוק ההון וישלח לך התראות פריצה בזמן אמת.\nבנוסף, ניתן לבצע סריקה ידנית בכל עת בעזרת הפקודה /scan.", parse_mode="HTML")

def scan_worker_manual(symbol: str) -> tuple:
    tech = analyze_technical_patterns(symbol)
    if tech and tech["signal"] == "BUY":
        return symbol, create_report_message(symbol)
    return symbol, None

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    add_user(message.chat.id)
    tickers = fetch_market_tickers()
    bot.reply_to(message, f"🔍 מתחיל סריקה ידנית מורחבת על <b>{len(tickers)} מניות</b>... אנא המתן כ-30 שניות.", parse_mode="HTML")

    found_any = False
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = executor.map(scan_worker_manual, tickers)
        for symbol, report in results:
            if report and report[1] is not None:
                msg, markup = report
                bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
                found_any = True

    if not found_any:
        bot.send_message(message.chat.id, "ℹ️ לא אותרו כעת מניות העונות על קריטריוני הפריצה הקפדניים.")

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
    
    print("🤖 Telegram Trading Bot with Background Scanner is active...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
