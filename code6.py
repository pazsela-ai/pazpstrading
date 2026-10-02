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

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

WATCHLIST = [
    "ELAL.TA", "ISRA.TA", "CAMT.TA", "NICE.TA", "TLRD.TA", "ENLT.TA", "NWM.TA", "ESLT.TA",
    "NVDA", "TSLA", "AMD", "MRNA", "PFE", "DAL", "LMT", "AAPL", "MSFT", "AMZN", "META"
]

# ---------------------------------------------------------
# 1. AI ENGINES (FAILOVER)
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
# 2. TECHNICAL & PRICE ENGINE
# ---------------------------------------------------------

def get_stock_price_data(ticker):
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        df = stock.history(period="60d", interval="1d")
        if df.empty or len(df) < 5:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if ('ATR' in df and not pd.isna(latest['ATR'])) else current_price * 0.03

        entry_price = round(current_price * 1.002, 2)
        tp_price = round(entry_price + (atr_val * 2.0), 2)
        sl_price = round(entry_price - (atr_val * 1.2), 2)

        return {
            "current_price": round(current_price, 2),
            "entry_price": entry_price,
            "tp": tp_price,
            "sl": sl_price,
            "atr": atr_val
        }
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return None

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

# ---------------------------------------------------------
# 3. PER-TICKER ISOLATED NEWS SCANNER & AI ENGINE
# ---------------------------------------------------------

BROAD_NEWS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=airline+flight+cancellation+conflict+defense+war&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%D7%AA%D7%A2%D7%95%D7%A4%D7%94+%D7%91%D7%99%D7%98%D7%95%D7%9C+%D7%90%D7%9C+%D7%A2%D7%9C+%D7%91%D7%99%D7%91%D7%97%D7%95%D7%9F+%D7%92%D7%96&hl=he&gl=IL&ceid=IL:he",
    "https://www.globenewswire.com/rss/feed/subject/pharmaceuticals",
    "https://news.google.com/rss/search?q=clinical+trial+FDA+approval+phase+cancer+vaccine&hl=en-US&gl=US&ceid=US:en"
]

def scan_breaking_news_events():
    """סריקת חדשות, חילוץ מניות, ויצירת ניתוח + תכנית עבודה נפרדת לחלוטין לכל מניה"""
    last_scans["news"] = time.strftime("%Y-%m-%d %H:%M:%S")
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

    # שלב 1: ה-AI מזהה מניות מושפעות ומקשר בינן לבין הידיעות הרלוונטיות בלבד (בפורמט JSON)
    prompt = (
        "אנליסט פיננסי, עיין ברשימת הידיעות החדשותיות:\n"
        + json.dumps(articles, ensure_ascii=False) +
        "\n\nזהה מניות ספציפיות (סימולים באנגלית, למשל ELAL.TA, MRNA, LMT, XOM) שיש לגביהן אירוע משמעותי.\n"
        "החזר JSON בלבד במבנה הבא (ללא טקסט נוסף):\n"
        "{\n"
        '  "tickers": [\n'
        '    {\n'
        '      "ticker": "ELAL.TA",\n'
        '      "relevant_news": [{"title": "כותרת שרלוונטית רק לה", "link": "קישור"}],\n'
        '      "analysis": "ניתוח ממוקד בעברית מדוע המניה ספציפית זו מושפעת",\n'
        '      "action": "כניסה / המתנה / יציאה",\n'
        '      "reason": "הסבר מפורט למה להיכנס או להמתין כולל אזהרת FOMO במידת הצורך"\n'
        '    }\n'
        '  ]\n'
        "}\n"
        "אם אין מניות שמשופעות באופן ישיר, החזר: {\"tickers\": []}"
    )

    ai_res = ask_ai_with_failover(prompt)
    if not ai_res:
        return

    try:
        # חילוץ קוד ה-JSON מתוך התגובה
        json_match = re.search(r'\{.*\}', ai_res, re.DOTALL)
        if not json_match:
            return
        parsed_data = json.loads(json_match.group(0))
        
        for item in parsed_data.get("tickers", []):
            send_per_ticker_news_alert(
                ticker=item["ticker"],
                relevant_news=item.get("relevant_news", []),
                analysis=item.get("analysis", ""),
                action=item.get("action", "המתנה"),
                reason=item.get("reason", ""),
                target_chat_id=CHAT_ID
            )
    except Exception as e:
        logging.error(f"Failed to parse news JSON response: {e}")

def send_per_ticker_news_alert(ticker, relevant_news, analysis, action, reason, target_chat_id):
    """שליחת התראה נפרדת ומבודדת למניה יחידה עם תכנית עבודה ואסמכתאות"""
    ticker = ticker.upper().strip()
    price_data = get_stock_price_data(ticker)
    currency = "₪" if ".TA" in ticker else "$"

    # 1. כותרת ייעודית למניה
    msg = f"📰 **ניתוח אירוע חדשותי ותכנית מסחר: {ticker}**\n\n"

    # 2. אסמכתאות (קישורים לידיעות החדשותיות שהשפיעו עליה)
    msg += "🔗 **אסמכתאות וידיעות רלוונטיות:**\n"
    for news in relevant_news:
        title = news.get("title", "ידיעה חדשותית")
        link = news.get("link", "#")
        msg += f"• [{title}]({link})\n"
    msg += "\n"

    # 3. ניתוח AI ממוקד למניה הזו בלבד
    msg += f"💡 **ניתוח המשמעות למניה:**\n{analysis}\n\n"

    # 4. תכנית עבודה ייעודית (האם להיכנס/לצאת/מחירים)
    msg += f"📋 **תכנית עבודה ייעודית:**\n"
    msg += f"• **המלצה:** {action}\n"
    msg += f"• **נימוק:** {reason}\n\n"

    if price_data:
        entry = price_data["entry_price"]
        tp = price_data["tp"]
        sl = price_data["sl"]

        msg += (
            f"🎯 **פרמטרי פוזיציה מוצעים:**\n"
            f"• מחיר שוק נוכחי: {currency}{price_data['current_price']}\n"
            f"• מחיר כניסה מומלץ (Limit): {currency}{entry}\n"
            f"• יעד רווח (TP): {currency}{tp}\n"
            f"• סטופ לוס (SL): {currency}{sl}\n"
        )

        keyboard = types.InlineKeyboardMarkup()
        keyboard.add(types.InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"trade_{ticker}_{entry}_{sl}_{tp}"))
        keyboard.add(types.InlineKeyboardButton("📈 TradingView", url=f"https://www.tradingview.com/chart/?symbol={ticker.replace('.TA','')}") )
        safe_send_message(target_chat_id, msg, reply_markup=keyboard)
    else:
        keyboard = types.InlineKeyboardMarkup()
        keyboard.add(types.InlineKeyboardButton("📈 TradingView", url=f"https://www.tradingview.com/chart/?symbol={ticker.replace('.TA','')}") )
        safe_send_message(target_chat_id, msg, reply_markup=keyboard)

def analyze_single_ticker_news(ticker, chat_id=None):
    """ניתוח חדשות לפי דרישת משתמש (/news_scan TICKER)"""
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
        feed = feedparser.parse(resp.content)
        
        relevant_news = [{"title": e.title, "link": e.link} for e in feed.entries[:4] if hasattr(e, 'title')]

        if not relevant_news:
            return f"ℹ️ לא נמצאו כתבות חדשותיות אחרונות עבור `{ticker}`."

        prompt = (
            f"אתה אנליסט פיננסי. נתח את הידיעות החדשותיות הבאות עבור מניית {ticker}:\n"
            + json.dumps(relevant_news, ensure_ascii=False) +
            "\n\nהחזר תשובה ב-JSON בלבד במבנה הבא:\n"
            "{\n"
            f'  "analysis": "ניתוח קצר וממוקד בעברית למניית {ticker}",\n'
            '  "action": "כניסה / המתנה / יציאה",\n'
            '  "reason": "נימוק האם להכנס/לצאת ואזהרת FOMO במידת הצורך"\n'
            "}"
        )
        
        ai_res = ask_ai_with_failover(prompt)
        if ai_res:
            json_match = re.search(r'\{.*\}', ai_res, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                send_per_ticker_news_alert(
                    ticker=ticker,
                    relevant_news=relevant_news,
                    analysis=data.get("analysis", ""),
                    action=data.get("action", "המתנה"),
                    reason=data.get("reason", ""),
                    target_chat_id=chat_id or CHAT_ID
                )
                return None
        return "❌ לא ניתן היה להשלים את הניתוח כעת."
    except Exception as e:
        logging.error(f"Error in analyze_single_ticker_news: {e}")
        return f"❌ שגיאה בניתוח חדשות: {e}"

# ---------------------------------------------------------
# 4. TELEGRAM CALLBACKS & BOT HANDLERS
# ---------------------------------------------------------

def send_alert(ticker, tech_data=None, target_chat_id=None):
    dest_id = target_chat_id or CHAT_ID
    if not dest_id:
        return

    if not tech_data:
        tech_data = analyze_technical_deep(ticker)

    if not tech_data:
        safe_send_message(dest_id, f"❌ לא ניתן לשלוף נתונים עבור `{ticker}`.")
        return

    entry = tech_data["entry_price"]
    tp = tech_data["tp"]
    sl = tech_data["sl"]
    rec = tech_data["recommendation"]
    score = tech_data["score"]
    currency = "₪" if ".TA" in ticker.upper() else "$"

    reasons_text = "\n".join([f"  • {r}" for r in tech_data["reasons"]]) if tech_data["reasons"] else "  • ללא אינדיקטור מיוחד"

    msg = (
        f"📊 **ניתוח איתות טכני - {ticker.upper()}** (ציון: {score}/100)\n\n"
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

    safe_send_message(dest_id, msg, reply_markup=keyboard)

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
                f"• **רווח פוטנציאלי ביעד (TP):** {curr_symbol}{potential_profit}\n"
            )
            safe_send_message(message.chat.id, calc_msg)
            return
        except ValueError:
            bot.reply_to(message, "⚠ אנא הזן מספר תקין בלבד (למשל: 200). נסה שוב:")
            return

    if text.startswith('/start'):
        safe_send_message(message.chat.id, "🟢 **הבוט PazPSTrading פעיל ומעודכן!**\nהקש `/tech NVDA` או `/news_scan ELAL.TA` לבדיקה.")
    elif text.startswith('/news_scan'):
        parts = text.split()
        if len(parts) < 2:
            safe_send_message(message.chat.id, "⚠ יש לציין סימול מניה. לדוגמה: `/news_scan ELAL.TA`")
            return
        safe_send_message(message.chat.id, f"🔎 מריץ ניתוח חדשות מבודד עבור `{parts[1].upper()}`...")
        res = analyze_single_ticker_news(parts[1], chat_id=message.chat.id)
        if res: safe_send_message(message.chat.id, res)
    elif text.startswith('/test_news'):
        safe_send_message(message.chat.id, "📰 מריץ סריקת אירועים מבודדת לכל מניה...")
        scan_breaking_news_events()

# ---------------------------------------------------------
# 5. WEBHOOK & SCHEDULER
# ---------------------------------------------------------

@app.route('/')
def home():
    return "OK - Per-Ticker Isolated Event Trading Bot Active!", 200

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
