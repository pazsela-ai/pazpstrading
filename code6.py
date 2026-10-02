import os
import logging
import time
import requests
import re
import urllib.parse
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser
from flask import Flask, request
from telebot import TeleBot, types
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
RENDER_EXTERNAL_URL = (os.getenv("RENDER_EXTERNAL_URL") or "").strip()

bot = TeleBot(TELEGRAM_TOKEN, threaded=False)
app = Flask(__name__)

user_states = {}
last_processed_news_titles = set()
last_scans = {"news": "טרם בוצעה", "tech": "טרם בוצעה"}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}

# רשימת מעקב טכנית דינמית (Watchlist) למניות תנודתיות ובעלות נזילות
WATCHLIST = [
    "ELAL.TA", "ISRA.TA", "CAMT.TA", "NICE.TA", "TLRD.TA", "ENLT.TA", "NWM.TA", "ESLT.TA",
    "NVDA", "TSLA", "AMD", "MRNA", "PFE", "DAL", "LMT", "AAPL", "MSFT", "AMZN", "META"
]

# ---------------------------------------------------------
# 1. FIXED TRIPLE FAILOVER AI ENGINE
# ---------------------------------------------------------

def ask_gemini_direct(prompt):
    """קריאה יציבה ל-Gemini API דרך נקודות קצה נתמכות בלבד"""
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY is missing!")
        return None

    endpoints = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
    ]
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    for url in endpoints:
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=12)
            if res.status_code == 200:
                data = res.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    return text.strip()
            else:
                logging.warning(f"Gemini API Error [{res.status_code}]: {res.text[:150]}")
        except Exception as e:
            logging.error(f"Error calling Gemini REST API: {e}")

    return None

def ask_groq_direct(prompt):
    """גיבוי ראשון: Groq API"""
    if not GROQ_API_KEY:
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    
    for model in groq_models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        try:
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                logging.warning(f"Groq API Error [{res.status_code}] ({model}): {res.text[:150]}")
        except Exception as e:
            logging.error(f"Error calling Groq API ({model}): {e}")

    return None

def ask_openai_direct(prompt):
    """גיבוי שני: OpenAI API"""
    if not OPENAI_API_KEY:
        return None

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data['choices'][0]['message']['content'].strip()
        else:
            logging.warning(f"OpenAI API Error [{res.status_code}]: {res.text[:150]}")
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")

    return None

def ask_ai_with_failover(prompt):
    """מנגנון קריאה מרכזי המעביר בקשות בין ספקים באופן שקוף"""
    res = ask_gemini_direct(prompt)
    if res:
        return res

    logging.info("Gemini failed. Falling back to Groq...")
    res = ask_groq_direct(prompt)
    if res:
        return res

    logging.info("Groq failed. Falling back to OpenAI...")
    res = ask_openai_direct(prompt)
    if res:
        return res

    return "❌ שגיאה: כל ספקי ה-AI (Gemini, Groq, OpenAI) אינם זמינים כעת. אנא בדוק/י מפתחות API ומכסות."

# ---------------------------------------------------------
# 2. WEBHOOK & COMMANDS SETUP
# ---------------------------------------------------------

def setup_bot_commands():
    try:
        commands = [
            types.BotCommand("start", "הפעלת הבוט ותפריט ראשי"),
            types.BotCommand("status", "בדיקת סטטוס מערכת"),
            types.BotCommand("tech", "ניתוח טכני מעמיק (למשל /tech NVDA)"),
            types.BotCommand("news_scan", "ניתוח חדשות AI (למשל /news_scan NVDA)"),
            types.BotCommand("test_news", "הרצת בדיקת חדשות לייב"),
            types.BotCommand("test_tech", "הרצת בדיקה טכנית בלייב"),
            types.BotCommand("portfolio", "מעקב תיק מניות")
        ]
        bot.set_my_commands(commands)
    except Exception as e:
        logging.error(f"Failed to update bot commands: {e}")

@app.route('/')
@app.route('/health')
def home():
    return "OK - Event & Dynamic Risk Calculator Active!", 200

@app.route('/init_webhook', methods=['GET', 'POST'])
def init_webhook():
    if not TELEGRAM_TOKEN or not RENDER_EXTERNAL_URL:
        return "Missing variables", 400
    url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_TOKEN}"
    try:
        bot.remove_webhook()
        time.sleep(1)
        res = bot.set_webhook(url=url)
        setup_bot_commands()
        return f"Success! Webhook set to {url}" if res else "Telegram rejected webhook", 200
    except Exception as e:
        return f"Error: {e}", 500

@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def telegram_webhook():
    if request.headers.get('content-type') == 'application/json':
        try:
            json_string = request.get_data().decode('utf-8')
            update = types.Update.de_json(json_string)
            bot.process_new_updates([update])
            return 'OK', 200
        except Exception as e:
            logging.error(f"Webhook error: {e}")
            return 'OK', 200
    return 'Forbidden', 403

# ---------------------------------------------------------
# 3. TECHNICAL ENGINE & WATCHLIST SCANNER
# ---------------------------------------------------------

def analyze_technical_deep(ticker):
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        df = stock.history(period="100d", interval="1d")
        if df.empty or len(df) < 20:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=min(50, len(df)-1))
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['VOL_SMA20'] = ta.sma(df['Volume'], length=20)

        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if ('ATR' in df and not pd.isna(latest['ATR'])) else current_price * 0.03
        rsi_val = float(latest['RSI']) if ('RSI' in df and not pd.isna(latest['RSI'])) else 50.0
        ema20 = float(latest['EMA20']) if ('EMA20' in df and not pd.isna(latest['EMA20'])) else current_price
        ema50 = float(latest['EMA50']) if ('EMA50' in df and not pd.isna(latest['EMA50'])) else current_price
        vol_now = float(latest['Volume']) if 'Volume' in df else 0
        vol_avg = float(latest['VOL_SMA20']) if ('VOL_SMA20' in df and not pd.isna(latest['VOL_SMA20'])) else 1

        score = 0
        reasons = []

        if current_price >= ema20 >= ema50:
            score += 30
            reasons.append("מגמה עולה: מחיר מעל EMA20 ומעל EMA50")
        if 48 <= rsi_val <= 68:
            score += 35
            reasons.append(f"מומנטום בריא: RSI ברמה של {rsi_val:.1f}")
        if vol_avg > 0 and vol_now > (vol_avg * 1.1):
            score += 35
            reasons.append(f"נפח מסחר מוגבר ({int(vol_now/vol_avg*100)}% מהממוצע)")

        entry_price = round(current_price * 1.005, 2)
        tp_price = round(entry_price + (atr_val * 2.0), 2)
        sl_price = round(entry_price - (atr_val * 1.2), 2)

        is_quality_breakout = score >= 60
        recommendation = "🟢 **מומלץ לכניסה (איתות חיובי)**" if is_quality_breakout else "🟡 **ניטרלי / המתנה**"

        return {
            "ticker": ticker,
            "score": score,
            "is_breakout": is_quality_breakout,
            "recommendation": recommendation,
            "current_price": round(current_price, 2),
            "entry_price": entry_price,
            "tp": tp_price,
            "sl": sl_price,
            "rsi": round(rsi_val, 1),
            "reasons": reasons
        }
    except Exception as e:
        logging.error(f"Error deep analyzing {ticker}: {e}")
        return None

def scan_watchlist_technical():
    """סריקה טכנית אוטומטית ברקע לרשימת ה-Watchlist"""
    last_scans["tech"] = time.strftime("%Y-%m-%d %H:%M:%S")
    logging.info("Starting background technical scan for Watchlist...")
    for ticker in WATCHLIST:
        try:
            tech_data = analyze_technical_deep(ticker)
            if tech_data and tech_data["is_breakout"]:
                send_alert(ticker, tech_data)
        except Exception as e:
            logging.error(f"Error in watchlist scan for {ticker}: {e}")

# ---------------------------------------------------------
# 4. EXPANDED BROAD EVENT-DRIVEN NEWS ENGINE
# ---------------------------------------------------------

BROAD_NEWS_FEEDS = [
    "https://www.businesswire.com/rss/home/?rss=G1NSRmRZXVR3e1xRWA==",
    "https://www.globenewswire.com/rss/feed/subject/pharmaceuticals",
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&datea=&dateb=&owner=include&start=0&count=40&output=atom",
    "https://news.google.com/rss/search?q=pharma+FDA+clinical+trial+cancer+acquisition+merger&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=aviation+airline+defense+conflict+war+sanctions&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%D7%AA%D7%A2%D7%95%D7%A4%D7%94+%D7%90%D7%9C+%D7%A2%D7%9C+%D7%91%D7%99%D7%98%D7%97%D7%95%D7%9F+%D7%A4%D7%90%D7%A8%D7%9E%D7%94+%D7%91%D7%95%D7%A8%D7%A1%D7%94+%D7%92%D7%96&hl=he&gl=IL&ceid=IL:he"
]

def scan_breaking_news_events():
    last_scans["news"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not CHAT_ID:
        return

    collected_articles = []
    for feed_url in BROAD_NEWS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=HEADERS, timeout=8)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:6]:
                title = getattr(entry, 'title', '')
                if title and title not in last_processed_news_titles:
                    collected_articles.append(title)
                    last_processed_news_titles.add(title)
        except Exception as e:
            logging.error(f"Error fetching feed {feed_url}: {e}")

    if not collected_articles:
        return

    prompt = (
        "אתה אנליסט פיננסי בכיר ומסוחר אירועים (Event-Driven Trader).\n"
        "קרא את כותרות החדשות והדיווחים הבאים שנאספו כעת בזמן אמת:\n"
        + "\n".join([f"- {t}" for t in collected_articles]) +
        "\n\nתפקידך לסווג את הידיעות ולחלץ המלצות מסחר מעשיות:\n"
        "1. **סיווג אירועים קריטיים:** זהה אם יש אירוע מהותי (ניסוי קליני/FDA, עסקאות M&A, אירועים ביטחוניים/תעופתיים, רגולציה/דוחות 8-K, סייבר, שרשרת אספקה).\n"
        "2. **זיהוי מניות מושפעות:** רשום מפורשות סימולי מניות באנגלית (לדוגמה: MRNA, ELAL.TA, ESLT.TA, DAL, NVDA, PFE).\n"
        "3. **המלצת השקעה ומסחר:** לכל מניה שזוהתה, ספק המלצה ברורה:\n"
        "   - **כיוון פוזיציה:** (קנייה / שורט / מעקב בלבד)\n"
        "   - **אופק זמן:** (מסחר יומי, סווינג לטווח קצר, השקעה לטווח בינוני)\n"
        "   - **רציונל וטריגר לכניסה:** הסבר קצר מדוע האירוע יוצר הזדמנות מסחר ומהו התנאי לכניסה.\n\n"
        "אם אין אף אירוע דרמטי בעל השפעה מסחרית ישירה, ענה בדיוק: 'אין אירוע קריטי'."
    )

    ai_text = ask_ai_with_failover(prompt)
    if "אין אירוע קריטי" not in ai_text and "❌" not in ai_text:
        msg = f"🚨 **איתות אירוע מתפרץ + המלצת השקעה!**\n\n{ai_text}"
        bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
        
        # חילוץ מניות מהטקסט והרצת ניתוח טכני משלים
        found_tickers = re.findall(r'\b[A-Z]{2,5}(?:\.TA)?\b', ai_text)
        for tick in set(found_tickers):
            tech_data = analyze_technical_deep(tick)
            if tech_data:
                send_alert(tick, tech_data)

def analyze_single_ticker_news(ticker):
    ticker = ticker.strip().upper()
    clean_ticker = ticker.replace('.TA', '')
    
    if ticker.endswith('.TA'):
        search_query = f"{clean_ticker} stock ISRAEL"
        hl_param, gl_param, ceid_param = "he", "IL", "IL:he"
    else:
        search_query = f"{clean_ticker} stock news"
        hl_param, gl_param, ceid_param = "en-US", "US", "US:en"

    encoded_query = urllib.parse.quote(search_query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl={hl_param}&gl={gl_param}&ceid={ceid_param}"

    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=10)
        
        if resp.status_code != 200:
            logging.error(f"Google News RSS status code: {resp.status_code}")
            return f"⚠️ לא ניתן לשלוף חדשות מ-Google News כרגע (קוד שגיאה: {resp.status_code})."

        feed = feedparser.parse(resp.content)
        items = [f"• {e.title}" for e in feed.entries[:5] if hasattr(e, 'title')]

        if not items:
            return f"ℹ️ לא נמצאו כתבות חדשותיות אחרונות עבור `{ticker}`."

        prompt = (
            f"אתה אנליסט פיננסי. נתח בקצרה בעברית את הידיעות החדשותיות הבאות עבור מניית {ticker}:\n"
            + "\n".join(items) +
            "\n\nספק סיכום קצר, הערכת השפעה על המחיר, והמלצת מסחר/השקעה מפורשת (קנייה / שורט / המתנה)."
        )
        
        ai_text = ask_ai_with_failover(prompt)
        
        if ai_text and "❌" not in ai_text:
            return f"📰 **סיכום חדשות והמלצת השקעה עבור {ticker}:**\n\n{ai_text}"
        return ai_text

    except Exception as e:
        logging.error(f"Error in analyze_single_ticker_news: {e}")
        return f"❌ שגיאה בניתוח חדשות: {e}"

# ---------------------------------------------------------
# 5. ALERT MESSAGING & CALLBACKS
# ---------------------------------------------------------

def send_alert(ticker, tech_data=None, target_chat_id=None):
    dest_id = target_chat_id or CHAT_ID
    if not dest_id:
        return

    if not tech_data:
        tech_data = analyze_technical_deep(ticker)

    if not tech_data:
        bot.send_message(dest_id, f"❌ לא ניתן לשלוף נתונים עבור `{ticker}`.")
        return

    entry = tech_data["entry_price"]
    tp = tech_data["tp"]
    sl = tech_data["sl"]
    rec = tech_data["recommendation"]
    score = tech_data["score"]
    currency = "₪" if ".TA" in ticker.upper() else "$"

    reasons_text = "\n".join([f"  • {r}" for r in tech_data["reasons"]]) if tech_data["reasons"] else "  • ללא אינדיקטור מיוחד"

    msg = (
        f"📊 **ניתוח איתות - {ticker.upper()}** (ציון איכות: {score}/100)\n\n"
        f"📣 **המלצה:** {rec}\n\n"
        f"💡 **פרמטרים שנבדקו:**\n"
        f"{reasons_text}\n\n"
        f"🎯 **תכנית עבודה מוצעת:**\n"
        f"• מחיר נוכחי: {currency}{tech_data['current_price']}\n"
        f"• מחיר כניסה מומלץ (Limit): {currency}{entry}\n"
        f"• יעד רווח (TP): {currency}{tp}\n"
        f"• סטופ לוס (SL): {currency}{sl}\n"
    )

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"trade_{ticker.upper()}_{entry}_{sl}_{tp}"))
    keyboard.add(types.InlineKeyboardButton("📈 TradingView", url=f"https://www.tradingview.com/chart/?symbol={ticker.upper().replace('.TA','')}") )

    bot.send_message(dest_id, msg, parse_mode="Markdown", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data.startswith('trade_'))
def handle_trade_click(call):
    _, ticker, entry, sl, tp = call.data.split('_')
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    buttons = [
        types.InlineKeyboardButton("💵 חישוב בדולרים ($)", callback_data=f"currency_{ticker}_{entry}_{sl}_{tp}_USD"),
        types.InlineKeyboardButton("₪ חישוב בשקלים (₪)", callback_data=f"currency_{ticker}_{entry}_{sl}_{tp}_ILS")
    ]
    keyboard.add(*buttons)
    bot.edit_message_reply_markup(chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=keyboard)
    bot.answer_callback_query(call.id, text="בחר מטבע לחישוב הסיכון")

@bot.callback_query_handler(func=lambda call: call.data.startswith('currency_'))
def handle_currency_select(call):
    _, ticker, entry, sl, tp, curr = call.data.split('_')
    
    user_states[call.from_user.id] = {
        "action": "awaiting_risk_amount",
        "ticker": ticker,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "curr": curr
    }

    symbol = "$" if curr == "USD" else "₪"
    bot.send_message(call.message.chat.id, f"✍️ **אנא הקלד/י כעת בטרמינל את סכום הסיכון המבוקש ב-{symbol}:**\n(לדוגמה: 150 או 500)", parse_mode="Markdown")
    bot.answer_callback_query(call.id)

# ---------------------------------------------------------
# 6. TELEGRAM MESSAGES & USER INPUT HANDLER
# ---------------------------------------------------------

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    user_id = message.from_user.id
    text = message.text.strip()

    if user_id in user_states and user_states[user_id].get("action") == "awaiting_risk_amount":
        try:
            risk_amount = float(text)
            state = user_states.pop(user_id)

            entry = state["entry"]
            sl = state["sl"]
            tp = state["tp"]
            ticker = state["ticker"]
            curr = state["curr"]

            risk_per_share = entry - sl
            if risk_per_share <= 0:
                bot.reply_to(message, "❌ שגיאה: סטופ לוס גבוה/שווה למחיר הכניסה.")
                return

            shares_count = int(risk_amount / risk_per_share)
            total_cost = round(shares_count * entry, 2)
            potential_profit = round(shares_count * (tp - entry), 2)
            curr_symbol = "$" if curr == "USD" else "₪"

            calc_msg = (
                f"📐 **חישוב פוזיציה מותאם אישית עבור {ticker}:**\n\n"
                f"• **סכום סיכון מוגדר:** {curr_symbol}{risk_amount}\n"
                f"• **מחיר כניסה מומלץ:** {curr_symbol}{entry}\n"
                f"• **סטופ לוס (SL):** {curr_symbol}{sl} (סיכון של {curr_symbol}{risk_per_share:.2f} למניה)\n\n"
                f"👉 **כדי לסכן בדיוק {curr_symbol}{risk_amount} - עליך לקנות:** `{shares_count}` מניות\n"
                f"• **שווי פוזיציה כולל:** {curr_symbol}{total_cost}\n"
                f"• **רווח פוטנציאלי ביעד (TP):** {curr_symbol}{potential_profit}\n"
            )
            bot.reply_to(message, calc_msg, parse_mode="Markdown")
            return
        except ValueError:
            bot.reply_to(message, "⚠️ אנא הזן מספר תקין בלבד (למשל: 200). נסה שוב:")
            return

    if text.startswith('/start') or text.startswith('/help'):
        bot.reply_to(message, "🟢 **הבוט PazPSTrading מחובר ופעיל!**\nהקש `/tech NVDA` או `/news_scan NVDA` לבדיקה.", parse_mode="Markdown")
    elif text.startswith('/status'):
        active_providers = []
        if GEMINI_API_KEY: active_providers.append("Gemini")
        if GROQ_API_KEY: active_providers.append("Groq")
        if OPENAI_API_KEY: active_providers.append("OpenAI")

        status_msg = (
            "⚙ **סטטוס מערכת:**\n\n"
            f"• AI Engine: {'✅ מחובר (' + ', '.join(active_providers) + ')' if active_providers else '❌ ללא מפתח פעיל'}\n"
            f"• סריקת אירועים אוטומטית: 🟢 מופעלת (כל 15 דק')\n"
            f"• סריקת חדשות אחרונה: `{last_scans['news']}`\n"
            f"• סריקה טכנית אחרונה: `{last_scans['tech']}`"
        )
        bot.reply_to(message, status_msg, parse_mode="Markdown")
    elif text.startswith('/tech'):
        parts = text.split()
        if len(parts) < 2:
            bot.reply_to(message, "⚠ יש לציין סימול מניה. לדוגמה: `/tech NVDA`", parse_mode="Markdown")
            return
        ticker = parts[1]
        bot.reply_to(message, f"🔍 מריץ ניתוח טכני עבור `{ticker.upper()}`...", parse_mode="Markdown")
        tech_data = analyze_technical_deep(ticker)
        send_alert(ticker=ticker, tech_data=tech_data, target_chat_id=message.chat.id)
    elif text.startswith('/news_scan'):
        parts = text.split()
        if len(parts) < 2:
            bot.reply_to(message, "⚠️ יש לציין סימול מניה. לדוגמה: `/news_scan NVDA`", parse_mode="Markdown")
            return
        ticker = parts[1]
        bot.reply_to(message, f"🔎 מריץ ניתוח חדשות ב-AI עבור `{ticker.upper()}`...", parse_mode="Markdown")
        res = analyze_single_ticker_news(ticker)
        bot.reply_to(message, res, parse_mode="Markdown")
    elif text.startswith('/test_news'):
        bot.reply_to(message, "📰 מריץ סריקת אירועים חדשותיים בלייב...")
        scan_breaking_news_events()
    elif text.startswith('/test_tech'):
        bot.reply_to(message, "📈 מריץ בדיקה טכנית לדוגמה (NVDA)...")
        tech_data = analyze_technical_deep("NVDA")
        send_alert("NVDA", tech_data, message.chat.id)
    elif text.startswith('/portfolio'):
        bot.reply_to(message, "💼 **תיק מעקב מניות:** כרגע אין מניות רשומות במעקב הפעיל.", parse_mode="Markdown")

# ---------------------------------------------------------
# 7. BACKGROUND SCHEDULER & RUNNER
# ---------------------------------------------------------

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
scheduler.add_job(scan_watchlist_technical, 'interval', hours=1)
scheduler.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
