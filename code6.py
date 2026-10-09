import os
import re
import json
import logging
import time
from datetime import datetime, timedelta
import pytz
import feedparser
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from flask import Flask, request
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler

# AI SDKs with safe imports
import google.generativeai as genai

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------------------------------------------------------------------------
# 0. Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("PazPSTradingBot")

# ---------------------------------------------------------------------------
# 1. Environment & API Initialization
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, threaded=False) if TELEGRAM_BOT_TOKEN else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if Groq and GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI and OPENAI_API_KEY else None

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 2. Global State & Configuration
# ---------------------------------------------------------------------------
sent_ticker_cooldowns = {}
seen_articles = set()

WATCHLIST = [
    # --- ארה"ב: טכנולוגיה, AI וסמיקונדקטורס ---
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "INTC",
    "QCOM", "ARM", "MU", "SMCI", "AMAT", "LRCX", "KLAC", "PANW", "CRWD", "PLTR",
    "SNOW", "ORCL", "IBM", "NOW", "DDOG", "NET", "ZS",

    # --- ארה"ב: פינטק, קריפטו ופיננסים ---
    "COIN", "MSTR", "MARA", "RIOT", "PYPL", "SQ", "HOOD", "V", "MA", "JPM", "BAC", "GS",

    # --- ארה"ב: תעופה, ביטחון, אנרגיה וקמעונאות ---
    "BA", "LMT", "NOC", "RTX", "DAL", "UAL", "AAL", "XOM", "CVX", "F", "GM", "RIVN", "LCID",

    # --- ארה"ב: פארמה, בריאות וביוטק ---
    "PFE", "MRNA", "LLY", "NVO", "JNJ", "ABBV", "WMT", "COST", "TGT", "DIS", "NFLX",

    # --- ישראל (בורסת תל אביב - .TA) ---
    "TEVA.TA", "NICE.TA", "ICL.TA", "TSEM.TA", "ELBT.TA", "CAMT.TA", "PRIO.TA",
    "POLI.TA", "LUMI.TA", "DISB.TA", "MZTF.TA", "FIBI.TA", "ENLT.TA", "ENOG.TA",
    "AZRG.TA", "NMVG.TA", "DELT.TA", "DNYA.TA", "PHRE.TA"
]

BROAD_NEWS_FEEDS = [
    # --- חדשות כלליות ועולמיות ---
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.ynet.co.il/Integration/StoryRss1854.xml",

    # --- תעופה, ביטחון, מזרח תיכון ---
    "https://news.google.com/rss/search?q=aviation+airline+incident+flight&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=war+military+strike+tensions+Middle+East&hl=en-US&gl=US&ceid=US:en",

    # --- פארמה, ניסויים קליניים, FDA ---
    "https://news.google.com/rss/search?q=pharma+FDA+approval+clinical+trial+phase&hl=en-US&gl=US&ceid=US:en",

    # --- אנרגיה, נפט, גז ---
    "https://news.google.com/rss/search?q=oil+gas+strait+hormuz+pipeline+energy&hl=en-US&gl=US&ceid=US:en",

    # --- עסקאות, רכישות, מיזוגים, חוזי ענק ---
    "https://news.google.com/rss/search?q=acquisition+merger+deal+contract+partnership&hl=en-US&gl=US&ceid=US:en",

    # --- מאקרו, מדיניות גאו-פוליטית, סנקציות, מכסים, שבבים, ריבית ---
    "https://news.google.com/rss/search?q=sanctions+tariffs+semiconductor+export+policy&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Federal+Reserve+interest+rates+inflation+CPI&hl=en-US&gl=US&ceid=US:en",

    # --- סייבר, פריצות ואבטחה ---
    "https://news.google.com/rss/search?q=cyberattack+data+breach+cybersecurity&hl=en-US&gl=US&ceid=US:en",

    # --- שינויי מנכ"לים, הגבלים עסקיים ותביעות ---
    "https://news.google.com/rss/search?q=CEO+steps+down+resigns+activist+investor&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=antitrust+lawsuit+DOJ+FTC+investigation&hl=en-US&gl=US&ceid=US:en",

    # --- דיווחי SEC 8-K, דוחות כספיים ומאיה ---
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom",
    "https://news.google.com/rss/search?q=earnings+report+quarterly+results+revenue+EPS+beat+miss&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=site:maya.tase.co.il+דוח+מיידי+OR+דוח+כספי+OR+תוצאות&hl=he&gl=IL&ceid=IL:he",
    "https://news.google.com/rss/search?q=PR+Newswire+earnings+release+quarterly&hl=en-US&gl=US&ceid=US:en"
]

# ---------------------------------------------------------------------------
# 3. AI Failover Engine
# ---------------------------------------------------------------------------
def call_ai_failover(prompt: str) -> str:
    if GEMINI_API_KEY:
        for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"Gemini {model_name} failed: {e}")

    if groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            res_text = completion.choices[0].message.content
            if res_text:
                return res_text
        except Exception as e:
            logger.warning(f"Groq failed: {e}")

    if openai_client:
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            res_text = completion.choices[0].message.content
            if res_text:
                return res_text
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")

    return ""

# ---------------------------------------------------------------------------
# 4. Market Status & Helper Functions
# ---------------------------------------------------------------------------
def check_market_status(ticker: str) -> dict:
    is_israel = ticker.endswith(".TA")
    tz_il = pytz.timezone("Asia/Jerusalem")
    now_il = datetime.now(tz_il)
    weekday = now_il.weekday()
    time_str = now_il.strftime("%H:%M")

    if is_israel:
        if weekday in [6, 0, 1, 2, 3]:
            if "09:59" <= time_str <= "17:30":
                return {"status": "OPEN", "market": "Israel (TASE)"}
        return {"status": "CLOSED", "market": "Israel (TASE)"}
    else:
        if weekday in [0, 1, 2, 3, 4]:
            if "11:00" <= time_str < "16:30":
                return {"status": "PRE_MARKET", "market": "US Markets"}
            elif "16:30" <= time_str <= "23:00":
                return {"status": "OPEN", "market": "US Markets"}
        return {"status": "CLOSED", "market": "US Markets"}

def check_liquidity_and_price(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="60d", interval="1d")
        if df.empty or len(df) < 20:
            return {"valid": False, "reason": "Insufficient market data"}

        df.ta.atr(length=14, append=True)
        atr_col = [c for c in df.columns if c.startswith("ATRr_")]
        atr_val = df[atr_col[0]].iloc[-1] if atr_col else (df["High"].iloc[-1] - df["Low"].iloc[-1])

        current_price = df["Close"].iloc[-1]
        avg_vol_20 = df["Volume"].tail(20).mean()
        dollar_volume = current_price * avg_vol_20

        sl = max(0.01, current_price - (1.2 * atr_val))
        tp1 = current_price + (2.0 * atr_val)
        tp2 = current_price + (4.0 * atr_val)

        return {
            "valid": True,
            "current_price": float(current_price),
            "atr": float(atr_val),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "dollar_volume": float(dollar_volume)
        }
    except Exception as e:
        return {"valid": False, "reason": str(e)}

def is_in_cooldown(ticker: str) -> bool:
    if ticker in sent_ticker_cooldowns:
        last_sent = sent_ticker_cooldowns[ticker]
        if datetime.now() - last_sent < timedelta(hours=6):
            return True
    return False

def set_cooldown(ticker: str):
    sent_ticker_cooldowns[ticker] = datetime.now()

# ---------------------------------------------------------------------------
# 5. Telegram Alert Generator
# ---------------------------------------------------------------------------
def send_telegram_alert(ticker: str, alert_type: str, analysis: str, recommendation: str,
                        intensity: str, price_data: dict, market_info: dict, news_title: str = ""):
    if not bot or not TELEGRAM_CHAT_ID:
        return

    price = price_data.get("current_price", 0.0)
    sl = price_data.get("sl", 0.0)
    tp1 = price_data.get("tp1", 0.0)
    tp2 = price_data.get("tp2", 0.0)

    m_status = market_info.get("status", "N/A")
    m_name = market_info.get("market", "N/A")

    if "EARNINGS" in alert_type or "REPORT" in alert_type:
        badge = "📑 איתות דיווח חברה / דוח כספי"
    elif "NEWS" in alert_type:
        badge = "🚨 איתות חדשותי מתפרץ"
    else:
        badge = "📊 איתות סורק טכני"

    intensity_badge = "🔥 HIGH" if intensity.upper() == "HIGH" else "⚡ MEDIUM"

    msg = (
        f"{badge}\n"
        f"<b>מניה:</b> <code>{ticker}</code> | <b> עוצמה:</b> {intensity_badge}\n"
        f"<b>מצב שוק:</b> {m_name} ({m_status})\n"
    )

    if news_title:
        msg += f"<b>כותרת/דיווח:</b> {news_title}\n"

    msg += (
        f"\n<b>💡 ניתוח והסקה:</b>\n{analysis}\n\n"
        f"<b>📌 המלצה:</b> {recommendation}\n\n"
        f"<b>🎯 פרמטרי פוזיציה:</b>\n"
        f"• מחיר כניסה משוער: <b>${price:.2f}</b>\n"
        f"• Stop Loss (SL): <b>${sl:.2f}</b>\n"
        f"• Target 1 (TP1): <b>${tp1:.2f}</b>\n"
    )

    if intensity.upper() == "HIGH" and tp2 > 0:
        msg += f"• Target 2 (TP2): <b>${tp2:.2f}</b>\n"

    tv_ticker = ticker.replace(".TA", "")
    tv_url = f"https://www.tradingview.com/symbols/{tv_ticker}/"

    keyboard = InlineKeyboardMarkup()
    btn_calc = InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"calc_{ticker}_{price:.2f}_{sl:.2f}_{tp1:.2f}")
    btn_tv = InlineKeyboardButton("📈 TradingView", url=tv_url)
    keyboard.add(btn_calc, btn_tv)

    try:
        bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="HTML", reply_markup=keyboard)
        set_cooldown(ticker)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

# ---------------------------------------------------------------------------
# 6. Scanning Logic Engines
# ---------------------------------------------------------------------------
def scan_breaking_news_events(is_test: bool = False):
    logger.info("Starting scan_breaking_news_events...")
    global seen_articles

    for feed_url in BROAD_NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                art_id = entry.get("id", entry.get("link", entry.get("title", "")))
                if not is_test and art_id in seen_articles:
                    continue
                seen_articles.add(art_id)
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))

                prompt = f"""
אתה מנוע AI אנליטי חופשי ומבריק למסחר פיננסי. תפקידך לקרוא את הכתבה/הדיווח ולהפעיל ניתוח והסקה כלכלית עצמאית (Deductive Reasoning):

כותרת הידיעה: {title}
תקציר הידיעה: {summary}

משימה והנחיות:
1. קבע אם הידיעה כוללת אירוע פנדמנטלי/מסחרי/מאקרו בעל אימפקט ממשי:
   - עסקאות, מיזוגים, רכישות, חוזי ענק, שותפויות.
   - ניסויים קליניים, אישורי FDA, התפתחויות פארמה.
   - שינויים גאו-פוליטיים, מכסים, סנקציות, מדיניות שבבים/טכנולוגיה, ריבית.
   - מתקפות סייבר, שיבושים בשרשרת אספקה/ספנות.
   - דוחות כספיים, תוצאות רבעוניות, שינויי הנהלה/מנכ"לים.
   - אירועים ביטחוניים/תעופתיים/אנרגטיים.
2. זהה באופן עצמאי לחלוטין איזו מניה נסחרת (בארה"ב או בישראל עם סיומת .TA) מושפעת ביותר מהאירוע הזה. אל תגביל את עצמך לרשימה סגורה!
3. אם הידיעה היא חדשות כלליות, רכילות, פלילים או ללא השפעה מסחרית ישירה שניתן להסיק ממנה - החזר "NONE".

החזר JSON בלבד, ללא שום טקסט נוסף:
{{
  "ticker": "NONE",
  "category": "NEWS_EVENT",
  "intensity": "HIGH",
  "analysis": "הסבר מפורט בעברית על ההסקה וההשפעה המסחרית",
  "recommendation": "המלצת מסחר קצרה בעברית"
}}
"""
                ai_raw = call_ai_failover(prompt)
                if not ai_raw:
                    continue

                clean_json = ai_raw.replace("```json", "").replace("```", "").strip()
                try:
                    data = json.loads(clean_json)
                except Exception:
                    continue

                ticker = str(data.get("ticker", "")).strip().upper()
                if not ticker or ticker in ["NONE", "NULL", "N/A"]:
                    continue

                market_info = check_market_status(ticker)
                # סינון שוק סגור (רץ בלייב בלבד, עוקף בטסט)
                if market_info["status"] == "CLOSED" and not is_test:
                    logger.info(f"Market for {ticker} is CLOSED. Skipping automatic alert.")
                    continue

                if not is_test and is_in_cooldown(ticker):
                    continue

                price_data = check_liquidity_and_price(ticker)
                if not price_data.get("valid"):
                    continue

                send_telegram_alert(
                    ticker=ticker,
                    alert_type=data.get("category", "BREAKING_NEWS"),
                    analysis=data.get("analysis", "ניתוח אירוע / דוח כספי"),
                    recommendation=data.get("recommendation", "קנייה/מעקב"),
                    intensity=data.get("intensity", "MEDIUM"),
                    price_data=price_data,
                    market_info=market_info,
                    news_title=title
                )
        except Exception as e:
            logger.error(f"Error scanning feed {feed_url}: {e}")

def scan_watchlist_technical(is_test: bool = False):
    logger.info("Starting scan_watchlist_technical...")
    for ticker in WATCHLIST:
        try:
            if not is_test and is_in_cooldown(ticker):
                continue

            market_info = check_market_status(ticker)
            if market_info["status"] == "CLOSED" and not is_test:
                continue

            t = yf.Ticker(ticker)
            df = t.history(period="60d", interval="1d")
            if df.empty or len(df) < 50:
                continue

            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.rsi(length=14, append=True)

            close = df["Close"].iloc[-1]
            ema20 = df["EMA_20"].iloc[-1]
            ema50 = df["EMA_50"].iloc[-1]
            rsi = df["RSI_14"].iloc[-1]
            vol_curr = df["Volume"].iloc[-1]
            vol_avg20 = df["Volume"].tail(20).mean()

            score = 0
            reasons = []

            if close > ema20:
                score += 25
                reasons.append("מחיר מעל EMA20")
            if close > ema50:
                score += 25
                reasons.append("מחיר מעל EMA50")
            if 48 <= rsi <= 68:
                score += 25
                reasons.append(f"RSI בריא ({rsi:.1f})")
            if vol_curr > (1.1 * vol_avg20):
                score += 25
                reasons.append("נפח מסחר גבוה")

            if score >= 65:
                p_data = check_liquidity_and_price(ticker)
                if not p_data.get("valid"):
                    continue

                send_telegram_alert(
                    ticker=ticker,
                    alert_type="TECHNICAL_SCAN",
                    analysis=f"איתות טכני חיובי זוהה! ציון: {score}/100. פרמטרים: {', '.join(reasons)}.",
                    recommendation="מומנטום חיובי לפריצה.",
                    intensity="HIGH" if score >= 85 else "MEDIUM",
                    price_data=p_data,
                    market_info=market_info
                )
        except Exception as e:
            logger.error(f"Error scanning technical {ticker}: {e}")

# ---------------------------------------------------------------------------
# 7. Telegram Bot Commands & Callbacks
# ---------------------------------------------------------------------------
if bot:
    @bot.message_handler(commands=['start'])
    def cmd_start(message):
        bot.reply_to(message, "👋 PazPSTrading Bot פעיל וזמין! שלח /test_news או /test_tech לבדיקה.")

    @bot.message_handler(commands=['test_news', 'news_scan'])
    def cmd_test_news(message):
        bot.reply_to(message, "🧪 מריץ סריקת חדשות ודוחות כספיים בזמן אמת...")
        scan_breaking_news_events(is_test=True)
        bot.send_message(message.chat.id, "✅ סריקת החדשות הסתיימה.")

    @bot.message_handler(commands=['test_tech'])
    def cmd_test_tech(message):
        bot.reply_to(message, "🧪 מריץ סריקת Watchlist טכנית...")
        scan_watchlist_technical(is_test=True)
        bot.send_message(message.chat.id, "✅ סריקת Watchlist הסתיימה.")

    @bot.message_handler(commands=['tech'])
    def cmd_tech(message):
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "נא להזין סימול מניה. לדוגמה: <code>/tech NVDA</code>", parse_mode="HTML")
            return
        ticker = parts[1].upper()
        bot.reply_to(message, f"🔍 מריץ ניתוח טכני יזום על {ticker}...")
        m_info = check_market_status(ticker)
        p_data = check_liquidity_and_price(ticker)
        if not p_data.get("valid"):
            bot.reply_to(message, f"❌ שגיאה בשליפת {ticker}: {p_data.get('reason')}")
            return
        send_telegram_alert(ticker=ticker, alert_type="TECHNICAL_MANUAL", analysis=f"ניתוח יזום עבור {ticker}.", recommendation="סקירה ישירה.", intensity="MEDIUM", price_data=p_data, market_info=m_info)

    user_calc_state = {}

    @bot.callback_query_handler(func=lambda call: call.data.startswith("calc_"))
    def handle_calc_callback(call):
        _, ticker, price, sl, tp1 = call.data.split("_")
        user_calc_state[call.from_user.id] = {"ticker": ticker, "price": float(price), "sl": float(sl), "tp1": float(tp1)}
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("💵 דולר ($)", callback_data="curr_USD"), InlineKeyboardButton("₪ שקל (ILS)", callback_data="curr_ILS"))
        bot.send_message(call.message.chat.id, "בחר מטבע סיכון:", reply_markup=keyboard)

    @bot.callback_query_handler(func=lambda call: call.data.startswith("curr_"))
    def handle_currency_callback(call):
        curr = "USD" if "USD" in call.data else "ILS"
        user_id = call.from_user.id
        if user_id in user_calc_state:
            user_calc_state[user_id]["currency"] = curr
            bot.send_message(call.message.chat.id, "הכנס את סכום הסיכון הכספי (לדוגמה: 200):")

    @bot.message_handler(func=lambda msg: msg.from_user.id in user_calc_state and "currency" in user_calc_state[msg.from_user.id] and "amount" not in user_calc_state[msg.from_user.id])
    def handle_risk_amount_input(message):
        user_id = message.from_user.id
        try:
            risk_amount = float(message.text.strip())
            state = user_calc_state[user_id]
            price, sl, curr = state["price"], state["sl"], state["currency"]
            symbol = "$" if curr == "USD" else "₪"
            risk_per_share = abs(price - sl)
            shares = 1 if risk_per_share == 0 else int(risk_amount / risk_per_share)
            total_pos = shares * price
            profit = shares * (state["tp1"] - price)

            res = (
                f"📊 <b>חישוב גודל פוזיציה עבור {state['ticker']}</b>\n\n"
                f"• סיכון: <b>{symbol}{risk_amount:,.2f}</b>\n"
                f"• מניות לקנייה: <b>{shares:,} מניות</b>\n"
                f"• שווי פוזיציה: <b>{symbol}{total_pos:,.2f}</b>\n"
                f"• רווח צפוי ב-TP1: <b>+{symbol}{profit:,.2f}</b>\n"
            )
            bot.send_message(message.chat.id, res, parse_mode="HTML")
            del user_calc_state[user_id]
        except ValueError:
            bot.send_message(message.chat.id, "הזן מספר תקין.")

# ---------------------------------------------------------------------------
# 8. Flask Server & Unified Webhook Endpoint
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "HEAD"])
def index():
    return "PazPSTrading Bot is running!", 200

@app.route("/status", methods=["GET"])
def status():
    return f"Bot Token Configured: {bool(TELEGRAM_BOT_TOKEN)}, External URL: {RENDER_EXTERNAL_URL}", 200

@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    if request.headers.get("content-type") == "application/json":
        json_string = request.get_data().decode("utf-8")
        update = telebot.types.Update.de_json(json_string)
        if bot:
            bot.process_new_updates([update])
        return "OK", 200
    return "Forbidden", 403

def setup_webhook():
    if bot and RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/webhook"
        try:
            bot.remove_webhook()
            time.sleep(1)
            bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook explicitly configured to: {webhook_url}")
        except Exception as e:
            logger.error(f"Failed to set Webhook: {e}")

scheduler = BackgroundScheduler(timezone="Asia/Jerusalem")
scheduler.add_job(scan_breaking_news_events, "interval", minutes=15)
scheduler.add_job(scan_watchlist_technical, "interval", minutes=30)
scheduler.start()

setup_webhook()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
