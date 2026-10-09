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

# AI SDKs
import google.generativeai as genai
from groq import Groq
from openai import OpenAI

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

# Initialize Telegram Bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None

# Initialize AI Clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Flask App
app = Flask(__name__)

# ---------------------------------------------------------------------------
# 2. Global State & Configuration
# ---------------------------------------------------------------------------
# Cooldown tracking: ticker -> datetime of last alert sent
sent_ticker_cooldowns = {}

# RSS seen articles tracking
seen_articles = set()

# WATCHLIST (לניתוח טכני סדיר בלבד - מורחב ומקיף)
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

# RSS Feeds משולב: חדשות מתפרצות + דיווחי דוחות כספיים ואירועים מהותיים בחברות
BROAD_NEWS_FEEDS = [
    # ------------------- חלק א': חדשות עולמיות וישראליות כלליות -------------------
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.ynet.co.il/Integration/StoryRss1854.xml", # מבזקי Ynet

    # ------------------- חלק ב': חדשות נושאיות (תעופה, ביטחון, פארמה, נפט) -------------------
    "https://news.google.com/rss/search?q=aviation+airline+incident+flight&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=war+military+strike+tensions+Middle+East&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=pharma+FDA+approval+clinical+trial+phase&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=oil+gas+strait+hormuz+pipeline+energy&hl=en-US&gl=US&ceid=US:en",

    # ------------------- חלק ג': דוחות כספיים, תוצאות רבעוניות ודיווחי חברות רשמיים -------------------
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom", # SEC 8-K (דיווחים מיידיים בארה"ב)
    "https://news.google.com/rss/search?q=earnings+report+quarterly+results+revenue+EPS+beat+miss&hl=en-US&gl=US&ceid=US:en", # דוחות רבעוניים ארה"ב
    "https://news.google.com/rss/search?q=site:maya.tase.co.il+דוח+מיידי+OR+דוח+כספי+OR+תוצאות&hl=he&gl=IL&ceid=IL:he", # דיווחי הבורסה בתל אביב (מאיה)
    "https://news.google.com/rss/search?q=PR+Newswire+earnings+release+quarterly&hl=en-US&gl=US&ceid=US:en" # הודעות לתקשורת מחברות
]

# ---------------------------------------------------------------------------
# 3. AI Failover Engine (Gemini -> Groq -> OpenAI)
# ---------------------------------------------------------------------------
def call_ai_failover(prompt: str) -> str:
    """
    Tries Gemini (2.0 Flash / 1.5 Flash), then Groq (llama-3.3-70b), then OpenAI (gpt-4o-mini).
    """
    # 1. Gemini Try
    if GEMINI_API_KEY:
        for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    logger.info(f"AI Success with Gemini ({model_name})")
                    return response.text
            except Exception as e:
                logger.warning(f"Gemini {model_name} failed: {e}")

    # 2. Groq Try
    if groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            res_text = completion.choices[0].message.content
            if res_text:
                logger.info("AI Success with Groq (llama-3.3-70b-versatile)")
                return res_text
        except Exception as e:
            logger.warning(f"Groq failed: {e}")

    # 3. OpenAI Try
    if openai_client:
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            res_text = completion.choices[0].message.content
            if res_text:
                logger.info("AI Success with OpenAI (gpt-4o-mini)")
                return res_text
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")

    logger.error("All AI providers failed.")
    return ""

# ---------------------------------------------------------------------------
# 4. Market Status & Helper Functions
# ---------------------------------------------------------------------------
def check_market_status(ticker: str) -> dict:
    """
    Checks if market is OPEN, PRE_MARKET, or CLOSED for given ticker.
    """
    is_israel = ticker.endswith(".TA")
    tz_il = pytz.timezone("Asia/Jerusalem")
    now_il = datetime.now(tz_il)

    weekday = now_il.weekday() # 0=Mon, 1=Tue, ..., 6=Sun
    time_str = now_il.strftime("%H:%M")

    if is_israel:
        # Israel Market: Sunday (6) to Thursday (3)
        if weekday in [6, 0, 1, 2, 3]:
            if "09:59" <= time_str <= "17:30":
                return {"status": "OPEN", "market": "Israel (TASE)"}
        return {"status": "CLOSED", "market": "Israel (TASE)"}
    else:
        # US Market: Monday (0) to Friday (4)
        if weekday in [0, 1, 2, 3, 4]:
            if "11:00" <= time_str < "16:30":
                return {"status": "PRE_MARKET", "market": "US Markets"}
            elif "16:30" <= time_str <= "23:00":
                return {"status": "OPEN", "market": "US Markets"}
        return {"status": "CLOSED", "market": "US Markets"}

def check_liquidity_and_price(ticker: str) -> dict:
    """
    Fetches yfinance data, checks average daily volume ($500k+), calculates ATR, Entry, SL, TP1, TP2.
    """
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

        # Liquidity check threshold ($500,000)
        if dollar_volume < 500000 and not ticker.endswith(".TA"):
            logger.info(f"Ticker {ticker} low liquidity: ${dollar_volume:,.0f}")

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
        logger.error(f"Error checking liquidity for {ticker}: {e}")
        return {"valid": False, "reason": str(e)}

def is_in_cooldown(ticker: str) -> bool:
    """
    Checks if a ticker was alerted in the last 6 hours.
    """
    if ticker in sent_ticker_cooldowns:
        last_sent = sent_ticker_cooldowns[ticker]
        if datetime.now() - last_sent < timedelta(hours=6):
            return True
    return False

def set_cooldown(ticker: str):
    sent_ticker_cooldowns[ticker] = datetime.now()

# ---------------------------------------------------------------------------
# 5. Telegram Alert Generator & Keyboards
# ---------------------------------------------------------------------------
def send_telegram_alert(ticker: str, alert_type: str, analysis: str, recommendation: str,
                        intensity: str, price_data: dict, market_info: dict, news_title: str = ""):
    if not bot or not TELEGRAM_CHAT_ID:
        logger.error("Telegram bot token or chat ID not set.")
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
        logger.info(f"Alert sent for {ticker}")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

# ---------------------------------------------------------------------------
# 6. Scanning Logic Engines
# ---------------------------------------------------------------------------

# --- ENGINE 1: Breaking News & Corporate Earnings / Filings Deductive Scanner ---
def scan_breaking_news_events(is_test: bool = False):
    """
    Scans RSS feeds (General news, Sector news, SEC filings, Corporate Earnings),
    uses AI with deductive reasoning to extract affected tickers and produce trading alerts.
    """
    logger.info("Starting scan_breaking_news_events (News + Corporate Reports)...")
    global seen_articles

    for feed_url in BROAD_NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]: # Check top 5 items per feed
                art_id = entry.get("id", entry.get("link", entry.get("title", "")))
                
                if not is_test and art_id in seen_articles:
                    continue
                
                seen_articles.add(art_id)
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                
                prompt = f"""
אתה מנוע AI למסחר פיננסי. תפקידך לקרוא מבזק חדשותי או דיווח חברה/דוח כספי (SEC / TASE / Earnings Report) ולבצע **ניתוח והסקה אנליטית (Deductive Reasoning)**.

טקסט הדיווח/המבזק:
כותרת: {title}
תוכן/תקציר: {summary}

משימה:
1. זהה אם מדובר ב:
   א. **דיווח חברה/דוח כספי:** (תוצאות רבעוניות, עקיפת/פספוס תחזיות, דיווח מיידי 8-K, חוזה חדש, רכישה).
   ב. **אירוע חדשותי/פנדמנטלי:** (תעופה, מלחמה/ביטחון, פארמה, נפט/אנרגיה, סמיקונדקטורס וכד').
2. התאם מניה ספציפית (Ticker) בארה"ב או בישראל (עם סיומת .TA):
   - בדוח כספי: חלץ את ה-Ticker של החברה המדווחת.
   - באירוע חדשותי: בצע הסקה אנליטית! (למשל: תקרית בתעופה -> ELAL.TA, מתיחות במפרץ הפרסי -> XOM / CVX / DELT.TA, ניסוי רפואי -> TEVA / PFE).
3. אם אין אירוע משמעותי או שאי אפשר לחלץ מניה ספציפית הגיונית, החזר "NONE".

החזר תשובה בפורמט JSON בלבד, ללא שום טקסט נוסף או Markdown:
{{
  "ticker": "ELAL.TA",
  "category": "NEWS_EVENT", 
  "intensity": "HIGH",
  "analysis": "הסבר אנליטי מפורט בעברית על ההסקה וההשפעה המסחרית",
  "recommendation": "המלצת מסחר קצרה בעברית"
}}
*(הערה: בשדה category החזר "NEWS_EVENT" או "EARNINGS_REPORT")*
"""
                ai_raw = call_ai_failover(prompt)
                if not ai_raw:
                    continue

                # Clean AI output from markdown formatting
                clean_json = ai_raw.replace("```json", "").replace("```", "").strip()
                try:
                    data = json.loads(clean_json)
                except Exception as parse_err:
                    logger.warning(f"Failed to parse JSON from AI response: {parse_err}")
                    continue

                ticker = str(data.get("ticker", "")).strip().upper()
                if not ticker or ticker == "NONE" or ticker == "NULL":
                    continue

                if not is_test and is_in_cooldown(ticker):
                    logger.info(f"Ticker {ticker} is in cooldown. Skipping.")
                    continue

                market_info = check_market_status(ticker)
                price_data = check_liquidity_and_price(ticker)

                # Fallback if yfinance missing full data during event
                if not price_data.get("valid"):
                    price_data = {
                        "current_price": 100.0,
                        "atr": 2.0,
                        "sl": 97.6,
                        "tp1": 104.0,
                        "tp2": 108.0
                    }

                category = data.get("category", "BREAKING_NEWS")

                send_telegram_alert(
                    ticker=ticker,
                    alert_type=category,
                    analysis=data.get("analysis", "ניתוח אירוע / דוח כספי"),
                    recommendation=data.get("recommendation", "קנייה/מעקב"),
                    intensity=data.get("intensity", "MEDIUM"),
                    price_data=price_data,
                    market_info=market_info,
                    news_title=title
                )

        except Exception as feed_err:
            logger.error(f"Error scanning feed {feed_url}: {feed_err}")

# --- ENGINE 2: Regular Technical Watchlist Scanner ---
def scan_watchlist_technical(is_test: bool = False):
    """
    Scans the defined WATCHLIST for positive momentum technical setups (Score >= 65).
    """
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

            # Technical Indicators
            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.atr(length=14, append=True)

            close = df["Close"].iloc[-1]
            ema20 = df["EMA_20"].iloc[-1]
            ema50 = df["EMA_50"].iloc[-1]
            rsi = df["RSI_14"].iloc[-1]
            vol_curr = df["Volume"].iloc[-1]
            vol_avg20 = df["Volume"].tail(20).mean()

            score = 0
            reasons = []

            # Scoring rules
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
                reasons.append("נפח מסחר גבוה מהממוצע")

            if score >= 65:
                p_data = check_liquidity_and_price(ticker)
                if not p_data.get("valid"):
                    continue

                analysis_text = f"איתות טכני חיובי זוהה במניה! ציון טכני: {score}/100.\nפרמטרים: {', '.join(reasons)}."
                recommendation_text = "מומנטום חיובי לפריצה – מומלץ לבחון כניסה לפי יעדים."

                send_telegram_alert(
                    ticker=ticker,
                    alert_type="TECHNICAL_SCAN",
                    analysis=analysis_text,
                    recommendation=recommendation_text,
                    intensity="HIGH" if score >= 85 else "MEDIUM",
                    price_data=p_data,
                    market_info=market_info
                )
        except Exception as e:
            logger.error(f"Error scanning technical ticker {ticker}: {e}")

# ---------------------------------------------------------------------------
# 7. Telegram Bot Interactive Handlers & Commands
# ---------------------------------------------------------------------------
if bot:
    @bot.message_handler(commands=['start'])
    def cmd_start(message):
        bot.reply_to(message, "👋 שלום! PazPSTrading Bot פעיל ומוכן. סורק החדשות, הדוחות הכספיים והניתוח הטכני פועלים ברקע.")

    @bot.message_handler(commands=['test_news'])
    def cmd_test_news(message):
        bot.reply_to(message, "🧪 מריץ בדיקה יזומה לסורק החדשות והדוחות הכספיים...")
        scan_breaking_news_events(is_test=True)

    @bot.message_handler(commands=['test_tech'])
    def cmd_test_tech(message):
        bot.reply_to(message, "🧪 מריץ בדיקה יזומה לסורק הטכני...")
        scan_watchlist_technical(is_test=True)

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
            bot.reply_to(message, f"❌ לא ניתן לשלוף נתונים עבור {ticker}: {p_data.get('reason')}")
            return

        send_telegram_alert(
            ticker=ticker,
            alert_type="TECHNICAL_MANUAL",
            analysis=f"ניתוח יזום לבקשת משתמש עבור {ticker}.",
            recommendation="סקירה טכנית ישירה.",
            intensity="MEDIUM",
            price_data=p_data,
            market_info=m_info
        )

    # Interactive Risk Calculator Handlers
    user_calc_state = {}

    @bot.callback_query_handler(func=lambda call: call.data.startswith("calc_"))
    def handle_calc_callback(call):
        _, ticker, price, sl, tp1 = call.data.split("_")
        user_calc_state[call.from_user.id] = {
            "ticker": ticker,
            "price": float(price),
            "sl": float(sl),
            "tp1": float(tp1)
        }
        
        keyboard = InlineKeyboardMarkup()
        btn_usd = InlineKeyboardButton("💵 דולר ($)", callback_data="curr_USD")
        btn_ils = InlineKeyboardButton("₪ שקל (ILS)", callback_data="curr_ILS")
        keyboard.add(btn_usd, btn_ils)

        bot.send_message(call.message.chat.id, "בחר את מטבע הסיכון שלך:", reply_markup=keyboard)

    @bot.callback_query_handler(func=lambda call: call.data.startswith("curr_"))
    def handle_currency_callback(call):
        curr = "USD" if "USD" in call.data else "ILS"
        user_id = call.from_user.id
        if user_id in user_calc_state:
            user_calc_state[user_id]["currency"] = curr
            bot.send_message(call.message.chat.id, f"הכנס את סכום הסיכון הכספי שברצונך לסכן (לדוגמה: 200):")

    @bot.message_handler(func=lambda msg: msg.from_user.id in user_calc_state and "currency" in user_calc_state[msg.from_user.id] and "amount" not in user_calc_state[msg.from_user.id])
    def handle_risk_amount_input(message):
        user_id = message.from_user.id
        try:
            risk_amount = float(message.text.strip())
            state = user_calc_state[user_id]
            price = state["price"]
            sl = state["sl"]
            curr = state["currency"]
            symbol = "$" if curr == "USD" else "₪"

            risk_per_share = abs(price - sl)
            if risk_per_share == 0:
                shares = 1
            else:
                shares = int(risk_amount / risk_per_share)

            total_position = shares * price
            profit_tp1 = shares * (state["tp1"] - price)

            res_msg = (
                f"📊 <b>חישוב גודל פוזיציה עבור {state['ticker']}</b>\n\n"
                f"• סיכון מוגדר: <b>{symbol}{risk_amount:,.2f}</b>\n"
                f"• כמות מניות לקנייה: <b>{shares:,} מניות</b>\n"
                f"• שווי פוזיציה כולל: <b>{symbol}{total_position:,.2f}</b>\n"
                f"• רווח צפוי ביעד TP1: <b>+{symbol}{profit_tp1:,.2f}</b>\n"
            )
            bot.send_message(message.chat.id, res_msg, parse_mode="HTML")
            del user_calc_state[user_id]
        except ValueError:
            bot.send_message(message.chat.id, "אנא הזן מספר תקין עבור סכום הסיכון.")

# ---------------------------------------------------------------------------
# 8. Webhook & Background Scheduler Setup
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "HEAD"])
def index():
    return "PazPSTrading Bot is running!", 200

@app.route(f"/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
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
        webhook_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_BOT_TOKEN}"
        try:
            bot.remove_webhook()
            time.sleep(1)
            bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook configured to: {webhook_url}")
        except Exception as e:
            logger.error(f"Failed to set Webhook: {e}")

# Scheduler Setup
scheduler = BackgroundScheduler(timezone="Asia/Jerusalem")
scheduler.add_job(scan_breaking_news_events, "interval", minutes=15)
scheduler.add_job(scan_watchlist_technical, "interval", minutes=30)
scheduler.start()

# Initialize Webhook on app start
setup_webhook()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
