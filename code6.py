import os
import logging
import time
import requests
import re
import urllib.parse
import json
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser
from datetime import datetime
import pytz
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
sent_ticker_cooldowns = {}  # {ticker: timestamp} לשמירת צינון התראות (6 שעות)
MIN_DAILY_VOLUME_USD = 500000  # סף נזילות מינימלי בדולרים/שקלים

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

BROAD_NEWS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=airline+flight+cancellation+conflict+defense+war&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%D7%AA%D7%A2%D7%95%D7%A4%D7%94+%D7%91%D7%99%D7%98%D7%95%D7%9C+%D7%90%D7%9C+%D7%A2%D7%9C+%D7%91%D7%99%D7%91%D7%97%D7%95%D7%9F+%D7%92%D7%96&hl=he&gl=IL&ceid=IL:he",
    "https://www.globenewswire.com/rss/feed/subject/pharmaceuticals",
    "https://news.google.com/rss/search?q=clinical+trial+FDA+approval+phase+cancer+vaccine&hl=en-US&gl=US&ceid=US:en"
]

# ---------------------------------------------------------
# 1. AI ENGINES
# ---------------------------------------------------------

def ask_gemini_direct(prompt):
    if not GEMINI_API_KEY:
        return None
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    ]
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for url in endpoints:
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=12)
            if res.status_code == 200:
                data = res.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    parts = data['candidates'][0].get('content', {}).get('parts', [])
                    if parts and 'text' in parts[0]:
                        return parts[0]['text'].strip()
        except Exception as e:
            logging.error(f"Error calling Gemini API: {e}")
    return None

def ask_groq_direct(prompt):
    if not GROQ_API_KEY:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            res = requests.post(url, json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}, headers=headers, timeout=10)
            if res.status_code == 200:
                return res.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logging.error(f"Error calling Groq API ({model}): {e}")
    return None

def ask_openai_direct(prompt):
    if not OPENAI_API_KEY:
        return None
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        res = requests.post(url, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
    return None

def ask_ai_with_failover(prompt):
    res = ask_gemini_direct(prompt)
    if res: return res
    res = ask_groq_direct(prompt)
    if res: return res
    res = ask_openai_direct(prompt)
    if res: return res
    return None

def safe_send_message(chat_id, text, reply_markup=None):
    try:
        bot.send_message(chat_id, text, parse_mode="Markdown", reply_markup=reply_markup, disable_web_page_preview=True)
    except Exception as e:
        logging.warning(f"Failed to send Markdown message: {e}")
        bot.send_message(chat_id, text, reply_markup=reply_markup, disable_web_page_preview=True)

# ---------------------------------------------------------
# 2. MARKET HOURS & LIQUIDITY FILTERS
# ---------------------------------------------------------

def check_market_status(ticker):
    """בדיקת שעות וימי מסחר מדויקים לפי שעון ישראל עבור תל אביב וניו יורק"""
    tz_il = pytz.timezone('Asia/Jerusalem')
    now = datetime.now(tz_il)
    weekday = now.weekday() # 0=שני, 1=שלישי ... 5=שישי, 6=שבת
    current_time = now.time()

    # 1. הבורסה בתל אביב (TASE)
    if ticker.upper().endswith(".TA"):
        if weekday in [4, 5]: # שישי/שבת סגור
            return "🔒 הבורסה בתל אביב סגורה (סוף שבוע)"
        
        if weekday == 6: # יום ראשון
            start_time = datetime.strptime("10:00", "%H:%M").time()
            end_time = datetime.strptime("16:30", "%H:%M").time()
        elif weekday == 3: # יום חמישי
            start_time = datetime.strptime("09:59", "%H:%M").time()
            end_time = datetime.strptime("16:45", "%H:%M").time()
        else: # ימים שני, שלישי, רביעי
            start_time = datetime.strptime("09:59", "%H:%M").time()
            end_time = datetime.strptime("17:15", "%H:%M").time()

        if start_time <= current_time <= end_time:
            return "🟢 המסחר בתל אביב פעיל כעת (רציף)"
        return "🌙 הבורסה בתל אביב סגורה כעת (הוראה ממתינה לפתיחה)"

    # 2. הבורסה בארה"ב (NYSE / NASDAQ)
    else:
        if weekday in [5, 6]: # שבת/ראשון סגור
            return "🔒 הבורסה בארה\"ב סגורה (סוף שבוע)"

        pre_market_start = datetime.strptime("11:00", "%H:%M").time()
        main_market_start = datetime.strptime("16:30", "%H:%M").time()
        main_market_end = datetime.strptime("23:00", "%H:%M").time()

        if main_market_start <= current_time <= main_market_end:
            return "🟢 המסחר בארה\"ב פעיל כעת (שעות רגילות)"
        elif pre_market_start <= current_time < main_market_start:
            return "🟡 מסחר מוקדם בארה\"ב (Pre-Market פעיל)"
        return "🌙 הבורסה בארה\"ב סגורה כעת (הוראה ממתינה ל-Pre-Market / פתיחה)"

def check_liquidity_and_price(ticker, impact_level="MEDIUM"):
    """בדיקת נזילות וחישוב מחירי יעד דינמיים לפי עוצמת האירוע"""
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        df = stock.history(period="60d", interval="1d")
        if df.empty or len(df) < 5:
            return None, "נתונים לא מספיקים"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        avg_volume = df['Volume'].tail(20).mean()
        dollar_volume = avg_volume * current_price

        if dollar_volume < MIN_DAILY_VOLUME_USD:
            return None, f"נזילות נמוכה מדי (מחזור יומי ממוצע: {int(dollar_volume):,} בלבד)"

        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        atr_val = float(df['ATR'].iloc[-1]) if ('ATR' in df and not pd.isna(df['ATR'].iloc[-1])) else current_price * 0.03

        entry_price = round(current_price * 1.002, 2)
        sl_price = round(entry_price - (atr_val * 1.2), 2)

        if impact_level == "HIGH":
            tp1_price = round(entry_price + (atr_val * 2.0), 2)
            tp2_price = round(entry_price + (atr_val * 4.0), 2)
        else:
            tp1_price = round(entry_price + (atr_val * 2.0), 2)
            tp2_price = None

        return {
            "current_price": round(current_price, 2),
            "entry_price": entry_price,
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": tp2_price,
            "atr": atr_val,
            "dollar_volume": dollar_volume
        }, "OK"
    except Exception as e:
        logging.error(f"Error checking liquidity for {ticker}: {e}")
        return None, str(e)

# ---------------------------------------------------------
# 3. PER-TICKER ISOLATED NEWS SCANNER
# ---------------------------------------------------------

def is_in_cooldown(ticker):
    """מניעת התראות חוזרות באותו חלון זמן (6 שעות)"""
    now = time.time()
    if ticker in sent_ticker_cooldowns:
        if now - sent_ticker_cooldowns[ticker] < 21600:
            return True
    return False

def scan_breaking_news_events():
    if not CHAT_ID:
        return

    articles = []
    for feed_url in BROAD_NEWS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=HEADERS, timeout=8)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:5]:
                title = getattr(entry, 'title', '')
                link = getattr(entry, 'link', '')
                if title and title not in last_processed_news_titles:
                    articles.append({"title": title, "link": link})
                    last_processed_news_titles.add(title)
        except Exception as e:
            logging.error(f"Feed fetch error {feed_url}: {e}")

    if not articles:
        return

    prompt = (
        "אנליסט פיננסי, עיין ברשימת הידיעות החדשותיות:\n"
        + json.dumps(articles, ensure_ascii=False) +
        "\n\nזהה מניות ספציפיות (סימולים באנגלית, למשל ELAL.TA, MRNA, LMT, XOM) שיש לגביהן אירוע משמעותי.\n"
        "דרג את עוצמת האירוע (impact_level) כ- HIGH (אם מדובר בדרמה אקטואלית/ניסוי קריטי/אירוע מלחמה) או MEDIUM/LOW.\n"
        "החזר JSON בלבד במבנה הבא (ללא טקסט נוסף):\n"
        "{\n"
        '  "tickers": [\n'
        '    {\n'
        '      "ticker": "ELAL.TA",\n'
        '      "impact_level": "HIGH",\n'
        '      "relevant_news": [{"title": "כותרת שרלוונטית רק לה", "link": "קישור"}],\n'
        '      "analysis": "ניתוח ממוקד בעברית מדוע המניה ספציפית זו מושפעת",\n'
        '      "action": "כניסה / המתנה / יציאה",\n'
        '      "reason": "הסבר מפורט כולל אזהרת FOMO במידת הצורך"\n'
        '    }\n'
        '  ]\n'
        "}\n"
    )

    ai_res = ask_ai_with_failover(prompt)
    if not ai_res:
        return

    try:
        json_match = re.search(r'\{.*\}', ai_res, re.DOTALL)
        if not json_match:
            return
        parsed_data = json.loads(json_match.group(0))

        for item in parsed_data.get("tickers", []):
            ticker = item["ticker"].upper().strip()

            if is_in_cooldown(ticker):
                logging.info(f"Skipping {ticker} due to active 6h cooldown.")
                continue

            send_per_ticker_news_alert(
                ticker=ticker,
                impact_level=item.get("impact_level", "MEDIUM"),
                relevant_news=item.get("relevant_news", []),
                analysis=item.get("analysis", ""),
                action=item.get("action", "המתנה"),
                reason=item.get("reason", ""),
                target_chat_id=CHAT_ID
            )
    except Exception as e:
        logging.error(f"Failed to parse news JSON response: {e}")

def send_per_ticker_news_alert(ticker, impact_level, relevant_news, analysis, action, reason, target_chat_id):
    ticker = ticker.upper().strip()

    price_data, status_msg = check_liquidity_and_price(ticker, impact_level)

    if not price_data:
        logging.info(f"Filtered out {ticker}: {status_msg}")
        return

    sent_ticker_cooldowns[ticker] = time.time()
    market_status = check_market_status(ticker)
    currency = "₪" if ".TA" in ticker else "$"

    msg = f"📰 **ניתוח אירוע חדשותי ותכנית מסחר: {ticker}**\n"
    msg += f"⏰ **מצב שוק:** {market_status}\n"
    msg += f"🔥 **עוצמת אירוע:** `{impact_level}`\n\n"

    msg += "🔗 **אסמכתאות וידיעות רלוונטיות:**\n"
    for news in relevant_news:
        msg += f"• [{news.get('title', 'ידיעה חדשותית')}]({news.get('link', '#')})\n"
    msg += "\n"

    msg += f"💡 **ניתוח המשמעות למניה:**\n{analysis}\n\n"

    msg += f"📋 **תכנית עבודה ייעודית:**\n"
    msg += f"• **המלצה:** {action}\n"
    msg += f"• **נימוק:** {reason}\n\n"

    entry = price_data["entry_price"]
    tp1 = price_data["tp1"]
    tp2 = price_data["tp2"]
    sl = price_data["sl"]

    msg += (
        f"🎯 **פרמטרי פוזיציה מוצעים (R:R דינמי):**\n"
        f"• מחיר נוכחי: {currency}{price_data['current_price']}\n"
        f"• כניסה מומלץ (Limit): {currency}{entry}\n"
        f"• יעד רווח 1 (TP1): {currency}{tp1}\n"
    )

    if tp2:
        msg += f"• יעד רווח 2 מורחב (TP2): {currency}{tp2} (למיקסום גל עליות)\n"

    msg += f"• סטופ לוס (SL): {currency}{sl}\n"

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"trade_{ticker}_{entry}_{sl}_{tp1}"))
    keyboard.add(types.InlineKeyboardButton("📈 TradingView", url=f"https://www.tradingview.com/chart/?symbol={ticker.replace('.TA','')}") )

    safe_send_message(target_chat_id, msg, reply_markup=keyboard)

# ---------------------------------------------------------
# 4. TELEGRAM CALLBACKS & BOT HANDLERS
# ---------------------------------------------------------

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
    safe_send_message(call.message.chat.id, f"✍ **אנא הקלד/י כעת את סכום הסיכון המבוקש ב-{symbol}:**\n(לדוגמה: 150 או 500)")
    bot.answer_callback_query(call.id)

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    user_id = message.from_user.id
    text = message.text.strip()

    if user_id in user_states and user_states[user_id].get("action") == "awaiting_risk_amount":
        try:
            risk_amount = float(text)
            state = user_states.pop(user_id)

            entry, sl, tp, ticker, curr = state["entry"], state["sl"], state["tp"], state["ticker"], state["curr"]
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
                f"• **רווח פוטנציאלי ביעד (TP1):** {curr_symbol}{potential_profit}\n"
            )
            safe_send_message(message.chat.id, calc_msg)
            return
        except ValueError:
            bot.reply_to(message, "⚠ אנא הזן מספר תקין בלבד (למשל: 200). נסה שוב:")
            return

    if text.startswith('/start'):
        safe_send_message(message.chat.id, "🟢 **הבוט PazPSTrading פעיל ומעודכן כולל מסנני נזילות ושעות מסחר!**")
    elif text.startswith('/test_news'):
        safe_send_message(message.chat.id, "📰 מריץ סריקת אירועים מעודכנת...")
        scan_breaking_news_events()

# ---------------------------------------------------------
# 5. WEBHOOK & SCHEDULER
# ---------------------------------------------------------

@app.route('/')
def home():
    return "OK - Advanced Event Trading Bot Active!", 200

@app.route('/init_webhook', methods=['GET', 'POST'])
def init_webhook():
    if not TELEGRAM_TOKEN or not RENDER_EXTERNAL_URL:
        return "Missing variables", 400
    url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_TOKEN}"
    try:
        bot.remove_webhook()
        time.sleep(1)
        res = bot.set_webhook(url=url)
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

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
scheduler.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
