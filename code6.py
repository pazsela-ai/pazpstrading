import os
import time
import sqlite3
import threading
import requests
import datetime
import pytz
import yfinance as yf
import pandas as pd
from flask import Flask
from telebot import TeleBot, types
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler
from deep_translator import GoogleTranslator

# ------------------------------------------------------------------------------
# 1. הגדרות סביבה ומשתנים גלובליים (Environment Setup)
# ------------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
PORT = int(os.environ.get("PORT", 5000))
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")

bot = TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)
translator = GoogleTranslator(source='auto', target='iw')

USER_CALC_STATE = {}
CACHE = {}  # מנגנון מטמון לתוצאות ניתוח {symbol: (timestamp, data)}
CACHE_TTL = 180  # 3 דקות

# ------------------------------------------------------------------------------
# 2. ניהול בסיס נתונים (SQLite Database for Registered Users)
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
            time.sleep(600)  # כל 10 דקות
            if "localhost" not in SELF_URL:
                requests.get(SELF_URL, timeout=10)
        except Exception as e:
            print(f"[Keep-Alive Error]: {e}")

# ------------------------------------------------------------------------------
# 4. מנוע ניתוח טכני + ATR + מחירי כניסה חכמים
# ------------------------------------------------------------------------------
def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr)

def analyze_technical_patterns(symbol: str) -> dict:
    # בדיקת Cache מוקדמת
    now = time.time()
    if symbol in CACHE:
        cached_time, cached_data = CACHE[symbol]
        if now - cached_time < CACHE_TTL:
            return cached_data

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")
        if df.empty or len(df) < 30:
            return None

        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change_pct = ((current_price - prev_price) / prev_price) * 100

        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs.iloc[-1])))

        # SMA20 ונפח מסחר
        sma20 = float(df['Close'].rolling(window=20).mean().iloc[-1])
        avg_volume = df['Volume'].iloc[-21:-1].mean()
        current_volume = df['Volume'].iloc[-1]
        volume_accumulating = current_volume > (avg_volume * 1.2)

        # תמיכה, שיא ו-ATR
        recent_support = float(df['Low'].tail(10).min())
        recent_high = float(df['High'].tail(5).max())
        atr = calculate_atr(df, 14)

        patterns = []
        pattern_depth = 0.15

        high_30 = df['High'].iloc[-30:].max()
        low_30 = df['Low'].iloc[-30:].min()

        if (high_30 - current_price) / high_30 < 0.05 and current_price > sma20:
            patterns.append("ספל וידית (Cup & Handle)")
            pattern_depth = max(pattern_depth, float((high_30 - low_30) / high_30))

        recent_gain = (df['High'].iloc[-5:].max() - df['Low'].iloc[-15:].min()) / df['Low'].iloc[-15:].min()
        if recent_gain > 0.12 and abs(current_price - df['Close'].iloc[-3:].mean()) / current_price < 0.02:
            patterns.append("דגל שורי (Bullish Flag)")
            pattern_depth = max(pattern_depth, float(recent_gain))

        lows = df['Low'].iloc[-20:]
        if len(lows[lows <= low_30 * 1.02]) >= 2:
            patterns.append("תחתית כפולה (Double Bottom)")

        if df['Low'].iloc[-10] < df['Low'].iloc[-5] < df['Low'].iloc[-1] and abs(df['High'].iloc[-10:].max() - current_price) / current_price < 0.03:
            patterns.append("משולש עולה (Ascending Triangle)")

        # נרות יפניים
        open_p, close_p, high_p, low_p = float(df['Open'].iloc[-1]), float(df['Close'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1])
        body = abs(close_p - open_p)
        lower_shadow = min(open_p, close_p) - low_p
        
        if lower_shadow > 2 * body and body > 0:
            patterns.append("נר פטיש (Hammer)")
        
        prev_open, prev_close = float(df['Open'].iloc[-2]), float(df['Close'].iloc[-2])
        if prev_close < prev_open and close_p > open_p and close_p > prev_open and open_p < prev_close:
            patterns.append("בליעה שורית (Bullish Engulfing)")

        # לוגיקת איתות ומחירי כניסה מבוססי ATR
        is_strong_buy = len(patterns) > 0 and (volume_accumulating or rsi < 65)

        if is_strong_buy:
            signal = "BUY"
            if "דגל שורי" in patterns or "משולש עולה" in patterns:
                entry_price = round(recent_high * 1.002, 2)
            else:
                entry_price = round(recent_support * 1.005, 2)

            # סטופ לוס מבוסס ATR (תנודתיות): מחיר הכניסה פחות 1.5 ATR
            atr_stop = entry_price - (1.5 * atr)
            support_stop = recent_support * 0.98
            stop_loss = round(min(atr_stop, support_stop), 2)

            if stop_loss >= entry_price:
                stop_loss = round(entry_price - (1.5 * atr), 2)
        else:
            signal = "HOLD" if len(patterns) > 0 else "NEUTRAL"
            entry_price = None
            stop_loss = None

        result = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi, 1),
            "sma20": round(sma20, 2),
            "volume_accumulating": volume_accumulating,
            "patterns": patterns,
            "pattern_depth": pattern_depth,
            "signal": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "atr": round(atr, 2)
        }

        # שמירה ב-Cache
        CACHE[symbol] = (now, result)
        return result

    except Exception as e:
        print(f"[Tech Analysis Error] {symbol}: {e}")
        return None

# ------------------------------------------------------------------------------
# 5. אינטגרציית FINNHUB (חדשות בזמן אמת + דירוגי אנליסטים)
# ------------------------------------------------------------------------------
def fetch_finnhub_data(symbol: str) -> dict:
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY":
        return fetch_fallback_news(symbol)

    try:
        # 1. שליפת חדשות מ-Finnhub
        today = datetime.date.today()
        from_date = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        news_res = requests.get(news_url, timeout=5).json()

        translated_headlines = []
        catalysts_found = []

        if isinstance(news_res, list):
            for item in news_res[:3]:
                headline = item.get("headline", "")
                if not headline:
                    continue
                try:
                    translated = translator.translate(headline)
                except Exception:
                    translated = headline
                translated_headlines.append(translated)

                h_lower = headline.lower()
                if any(k in h_lower for k in ["merger", "acquisition", "buyout"]):
                    catalysts_found.append("עסקת מיזוג/רכישה")
                elif any(k in h_lower for k in ["fda", "approval", "phase"]):
                    catalysts_found.append("אישור פארמה/FDA")
                elif any(k in h_lower for k in ["contract", "deal", "partner"]):
                    catalysts_found.append("חוזה/שותפות אסטרטגית")
                elif any(k in h_lower for k in ["earnings", "beat", "guidance"]):
                    catalysts_found.append("עדכון תחזית / דוחות חיוביים")

        # 2. שליפת דירוגי אנליסטים מ-Finnhub
        recom_url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_API_KEY}"
        recom_res = requests.get(recom_url, timeout=5).json()
        analyst_str = "אין מידע זמין"

        if isinstance(recom_res, list) and len(recom_res) > 0:
            latest = recom_res[0]
            buy = latest.get("buy", 0) + latest.get("strongBuy", 0)
            hold = latest.get("hold", 0)
            sell = latest.get("sell", 0) + latest.get("strongSell", 0)
            analyst_str = f"🟢 קנייה: {buy} | 🟡 החזק: {hold} | 🔴 מכירה: {sell}"

        catalyst_str = ", ".join(set(catalysts_found)) if catalysts_found else "אין קטליזטור ישיר מעבר למומנטום"
        sentiment = "חיובי 🟢" if (catalysts_found or len(translated_headlines) > 0) else "ניטרלי 🟡"

        return {
            "headlines": translated_headlines,
            "catalyst": catalyst_str,
            "sentiment": sentiment,
            "analyst_ratings": analyst_str
        }

    except Exception as e:
        print(f"[Finnhub Error] {symbol}: {e}")
        return fetch_fallback_news(symbol)

def fetch_fallback_news(symbol: str) -> dict:
    """שליפת חדשות גיבוי דרך yfinance אם Finnhub לא מוגדר"""
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news
        headlines = []
        if news_items:
            for item in news_items[:3]:
                title = item.get("title", "")
                if not title and "content" in item and isinstance(item["content"], dict):
                    title = item["content"].get("title", "")
                if title:
                    try:
                        headlines.append(translator.translate(title))
                    except Exception:
                        headlines.append(title)
        return {
            "headlines": headlines,
            "catalyst": "לא אותרו אירועים מיוחדים",
            "sentiment": "ניטרלי 🟡",
            "analyst_ratings": "לא מוגדר (חסר API Key של Finnhub)"
        }
    except Exception:
        return {"headlines": [], "catalyst": "שגיאה בשליפת חדשות", "sentiment": "לא ידוע", "analyst_ratings": "לא זמין"}

# ------------------------------------------------------------------------------
# 6. מחשבון עסקאות ושער דולר
# ------------------------------------------------------------------------------
def get_usd_ils_rate() -> float:
    try:
        usd_ticker = yf.Ticker("USDILS=X")
        rate = usd_ticker.history(period="1d")['Close'].iloc[-1]
        return round(float(rate), 2)
    except Exception:
        return 3.60

def calculate_trade_plan(entry_price: float, stop_loss: float, pattern_depth_pct: float = 0.15) -> dict:
    risk_per_share = entry_price - stop_loss
    risk_pct = risk_per_share / entry_price
    
    tp1_price = entry_price + (risk_per_share * 2)
    tp1_pct = ((tp1_price - entry_price) / entry_price) * 100
    
    tp2_pct_calculated = max(pattern_depth_pct * 100, (risk_pct * 3.5) * 100)
    tp2_price = entry_price * (1 + (tp2_pct_calculated / 100))
    
    return {
        "entry": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "tp1": round(tp1_price, 2),
        "tp1_pct": round(tp1_pct, 1),
        "tp2": round(tp2_price, 2),
        "tp2_pct": round(tp2_pct_calculated, 1),
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
<i>(לפי שער דולר: ₪{usd_rate})</i>

<b>📌 כניסה וכמות:</b>
• מחיר כניסה מומלץ: <code>${entry:.2f}</code>
• כמות מניות כוללת לרכישה: <b>{total_shares} מניות</b>

---
<b>🎯 שלב 1: מימוש ראשוני ואיפוס סיכון (TP1)</b>
• מחיר יעד 1: <code>${tp1:.2f}</code> (+{trade_plan['tp1_pct']}%)
• <b>פעולה לבצוע:</b> למכור <b>{half_shares} מניות</b> (50% מהכמות).
• <b>רווח ננעל במזומן:</b> ${profit_tp1_usd:.2f} ({profit_tp1_ils:.2f} ₪)
• 🛡️ <b>הוראת סטופ לוס חדשה:</b> להעלות מיד ל-<code>${trade_plan['new_stop_after_tp1']:.2f}</code> (מחיר הכניסה).

---
<b>🚀 שלב 2: הרצת השארית ליעד התבנית המלא (TP2)</b>
• מחיר יעד 2: <code>${tp2:.2f}</code> (+{trade_plan['tp2_pct']}%)
• <b>פעולה לבצוע:</b> למכור את <b>{rem_shares} המניות הנותרות</b>.
• <b>רווח נוסף בשלב זה:</b> ${profit_tp2_usd:.2f} ({profit_tp2_ils:.2f} ₪)

---
<b>📊 סיכום כספי כולל לעסקה:</b>
🟢 <b>סך רווח צפוי:</b> ${total_profit_usd:.2f} ({total_profit_ils:.2f} ₪)
🔴 <b>סיכון מרבי (בסטופ המקורי של ${stop:.2f}):</b> ${max_risk_usd:.2f} ({max_risk_ils:.2f} ₪)
"""
    return msg

# ------------------------------------------------------------------------------
# 7. מחולל הדוחות והמקלדת
# ------------------------------------------------------------------------------
def create_report_message(symbol: str) -> tuple:
    tech = analyze_technical_patterns(symbol)
    if not tech:
        return f"❌ לא ניתן היה לשלוף נתונים עבור המנייה <b>{symbol}</b>.", None

    finnhub = fetch_finnhub_data(symbol)

    if tech["signal"] == "BUY":
        recommendation = "🟢 <b>מומלץ להיכנס להשקעה</b> (זיהוי פריצה טכנית/איתות ירוק)"
    elif tech["signal"] == "HOLD":
        recommendation = "🟡 <b>להמתין ולעקוב</b> (קיימת תבנית אך אין איתות כניסה מיידי)"
    else:
        recommendation = "🔴 <b>לא מומלץ כעת</b> (לא זוהו תבניות פריצה או איתותי קנייה)"

    patterns_str = ", ".join(tech["patterns"]) if tech["patterns"] else "לא זוהו תבניות מיוחדות"
    headlines_str = "\n".join([f"• {h}" for h in finnhub["headlines"]]) if finnhub["headlines"] else "• אין כותרות חדשות מהותיות"

    msg = f"""
<b>📊 דוח ניתוח מקיף עבור {symbol}</b>

<b>💡 המלצה סופית:</b>
{recommendation}

---
<b>📈 נתונים טכניים בשוק:</b>
• מחיר שוק נוכחי: <code>${tech['current_price']}</code> ({'+' if tech['change_pct']>0 else ''}{tech['change_pct']}%)
• מדד RSI: <code>{tech['rsi']}</code> | SMA20: <code>${tech['sma20']}</code> | ATR: <code>${tech['atr']}</code>
• תבניות שנמצאו: <b>{patterns_str}</b>
• איסוף נפח מסחר: {'כן 🟢' if tech['volume_accumulating'] else 'רגיל ⚪'}

---
<b>📰 חדשות, סנטימנט ואנליסטים (Finnhub):</b>
• דירוג אנליסטים: <b>{finnhub['analyst_ratings']}</b>
• קטליזטור שנמצא: <b>{finnhub['catalyst']}</b>
• סנטימנט חדשותי: <b>{finnhub['sentiment']}</b>
<b>כותרות אחרונות:</b>
{headlines_str}

---
"""

    if tech["signal"] == "BUY":
        plan = calculate_trade_plan(tech["entry_price"], tech["stop_loss"], tech["pattern_depth"])
        msg += f"""<b>🎯 תוכנית מסחר מומלצת (איתות ירוק):</b>
• 🎯 <b>מחיר כניסה חכם מומלץ:</b> <code>${plan['entry']}</code>
• 🛑 <b>Stop Loss (ATR):</b> <code>${plan['stop_loss']}</code>
• 🎯 <b>יעד 1 (TP1 - למימוש 50% ואיפוס סיכון):</b> <code>${plan['tp1']}</code> (+{plan['tp1_pct']}%)
• 🚀 <b>יעד 2 (TP2 - יעד תבנית מורחב):</b> <code>${plan['tp2']}</code> (+{plan['tp2_pct']}%)
"""
    else:
        msg += f"""ℹ️ <b>אין המלצת כניסה ירוקה כעת – לא מוגדרים מחירי כניסה וסטופ לוס.</b>
"""

    markup = InlineKeyboardMarkup(row_width=2)
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף", url=f"https://www.tradingview.com/chart/?symbol={symbol}")
    btn_news = InlineKeyboardButton("📰 חדשות בלבד", callback_data=f"news_{symbol}")
    btn_tech = InlineKeyboardButton("📊 ניתוח טכני בלבד", callback_data=f"tech_{symbol}")
    
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

הבוט מנטר את הבורסה האמריקאית, שולף נתונים מ-Finnhub, ומחשב נקודות כניסה חכמות וסטופ לוס (ATR) רק כשיש איתות קנייה ירוק.

<b>פקודות זמינות:</b>
/scan - סריקה ידנית מורחבת
/tech &lt;SYMBOL&gt; - ניתוח טכני למנייה (למשל: <code>/tech TSLA</code>)
/news &lt;SYMBOL&gt; - ניתוח חדשות ואנליסטים (למשל: <code>/news NVDA</code>)
"""
    bot.reply_to(message, welcome_text, parse_mode="HTML")

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    add_user(message.chat.id)
    bot.reply_to(message, "🔍 מתחיל סריקה מורחבת בשוק, אנא המתן...")
    top_symbols = ["AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", "META"]
    found = False
    for sym in top_symbols:
        tech = analyze_technical_patterns(sym)
        if tech and tech["signal"] == "BUY":
            msg, markup = create_report_message(sym)
            bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
            found = True
    if not found:
        msg, markup = create_report_message("NVDA")
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)

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
        headlines_str = "\n".join([f"• {h}" for h in finnhub['headlines']]) if finnhub['headlines'] else "אין כותרות"
        text = f"<b>📰 חדשות ואנליסטים עבור {symbol}:</b>\n\n• אנליסטים: <b>{finnhub['analyst_ratings']}</b>\n• קטליזטור: <b>{finnhub['catalyst']}</b>\n• סנטימנט: <b>{finnhub['sentiment']}</b>\n\n<b>כותרות:</b>\n{headlines_str}"
        bot.send_message(chat_id, text, parse_mode="HTML")

    elif data.startswith("tech_"):
        symbol = data.split("_")[1]
        tech = analyze_technical_patterns(symbol)
        if tech:
            patterns_str = ", ".join(tech["patterns"]) if tech["patterns"] else "אין"
            text = f"<b>📊 ניתוח טכני עבור {symbol}:</b>\n\n• מחיר שוק: <code>${tech['current_price']}</code>\n• RSI: <code>{tech['rsi']}</code> | ATR: <code>${tech['atr']}</code>\n• תבניות: <b>{patterns_str}</b>\n• איתות: <b>{tech['signal']}</b>"
            if tech["signal"] == "BUY":
                text += f"\n• מחיר כניסה חכם: <code>${tech['entry_price']}</code>\n• סטופ לוס: <code>${tech['stop_loss']}</code>"
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
# 10. מנוע סריקה אוטומטית (מסונן לפי שעות המסחר בארה"ב)
# ------------------------------------------------------------------------------
def is_market_open() -> bool:
    """בדיקה האם הבורסה בארה"ב פתוחה (שני-שישי, 16:30 עד 23:00 שעון ישראל)"""
    israel_tz = pytz.timezone('Asia/Jerusalem')
    now = datetime.datetime.now(israel_tz)
    
    # שני = 0, שישי = 4
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

    watch_list = ["NVDA", "TSLA", "AAPL", "AMD", "AMZN"]
    for sym in watch_list:
        tech = analyze_technical_patterns(sym)
        if tech and tech["signal"] == "BUY":
            msg, markup = create_report_message(sym)
            for chat_id in users:
                try:
                    bot.send_message(chat_id, f"🚨 <b>התראת איתות קנייה ירוק!</b>\n{msg}", parse_mode="HTML", reply_markup=markup)
                except Exception as e:
                    print(f"[Scheduled Notification Error] to {chat_id}: {e}")

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(scheduled_market_scan, 'interval', minutes=15)
scheduler.start()

# ------------------------------------------------------------------------------
# 11. הרצת השרת והבוט (Main Execution)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()
    
    print("🤖 Telegram Trading Bot with Finnhub & SQLite is active...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
