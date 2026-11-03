import os
import re
import json
import logging
import pytz
import datetime
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import feedparser

# OpenAI & Alternative AI Providers
import openai
import google.generativeai as genai

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Initialize API Clients
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logging.warning(f"Groq initialization failed: {e}")

# Initialize Telegram Bot & Flask App
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
app = Flask(__name__)

# Cooldown Tracker (In-Memory)
sent_ticker_cooldowns = {}
processed_news_links = set()

# Broad News Feeds (RSS)
BROAD_NEWS_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://search.cnbc.com/rs/search/combinedrender?source=yahoo&partnerId=2001&collection=finance&limit=25",
    "https://www.google.com/finance/news/rss",
    "https://feeds.feedburner.com/realmoney",
    "https://www.sec.gov/news/pressreleases.rss"
]

# Watchlist for Technical Scanner (~50 Tickers)
WATCHLIST = [
    # US Stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO",
    "BAC", "COST", "MCD", "WMT", "CSCO", "AMD", "CRM", "INTC", "NFLX", "QCOM",
    # Israeli Stocks (.TA)
    "TEVA.TA", "NICE.TA", "ICL.TA", "MZR.TA", "LUMI.TA", "POLI.TA", "DSCT.TA", "FIBI.TA",
    "HARL.TA", "PHOE.TA", "ENLT.TA", "ENOG.TA", "BEZQ.TA", "AZRG.TA", "BIG.TA", "SAEN.TA"
]

# Timezones
TZ_IL = pytz.timezone("Asia/Jerusalem")
TZ_US = pytz.timezone("America/New_York")


# ==========================================
# 1. FAILOVER AI ENGINE
# ==========================================

def analyze_with_ai(prompt: str) -> str:
    """
    Executes AI requests with direct failover: OpenAI -> Gemini -> Groq.
    """
    # 1. Try OpenAI
    if OPENAI_API_KEY:
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}. Falling back to Gemini...")

    # 2. Try Gemini
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {e}. Falling back to Groq...")

    # 3. Try Groq
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API error: {e}.")

    return ""


# ==========================================
# 2. MARKET HOURS & LIQUIDITY CHECKS
# ==========================================

def check_market_status(ticker: str) -> str:
    """
    Determines market state based on exchange rules.
    """
    now_il = datetime.datetime.now(TZ_IL)
    weekday = now_il.weekday()  # Mon=0, ..., Sun=6

    if ticker.endswith(".TA"):
        # Israeli Market: Sun-Thu
        if weekday in [6, 0, 1, 2, 3]:  # Sunday to Thursday
            current_time = now_il.time()
            if datetime.time(10, 0) <= current_time <= datetime.time(17, 25):
                return "IL_REGULAR"
        return "CLOSED"
    else:
        # US Market: Mon-Fri
        now_us = datetime.datetime.now(TZ_US)
        us_weekday = now_us.weekday()
        if us_weekday < 5:  # Mon-Fri
            us_time = now_us.time()
            if datetime.time(4, 0) <= us_time < datetime.time(9, 30):
                return "US Markets (PRE_MARKET)"
            elif datetime.time(9, 30) <= us_time <= datetime.time(16, 0):
                return "US Markets (OPEN)"
            elif datetime.time(16, 0) < us_time <= datetime.time(20, 0):
                return "US Markets (AFTER_HOURS)"
        return "CLOSED"


def check_liquidity_and_price(ticker: str) -> tuple[bool, float, float]:
    """
    Validates minimum average volume ($500K daily liquidity).
    Returns (is_liquid, current_price, avg_volume_usd).
    """
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="20d")
        if hist.empty or len(hist) < 5:
            return False, 0.0, 0.0

        current_price = float(hist['Close'].iloc[-1])
        avg_volume_20d = float(hist['Volume'].mean())
        daily_liquidity_usd = current_price * avg_volume_20d

        if daily_liquidity_usd >= 500000:
            return True, current_price, daily_liquidity_usd
        return False, current_price, daily_liquidity_usd
    except Exception as e:
        logging.error(f"Liquidity check error for {ticker}: {e}")
        return False, 0.0, 0.0


def is_in_cooldown(ticker: str) -> bool:
    """Checks if ticker was alerted within the last 6 hours."""
    now = datetime.datetime.now()
    if ticker in sent_ticker_cooldowns:
        last_sent = sent_ticker_cooldowns[ticker]
        if (now - last_sent).total_seconds() < 21600:  # 6 hours
            return True
    return False


def set_cooldown(ticker: str):
    sent_ticker_cooldowns[ticker] = datetime.datetime.now()


# ==========================================
# 3. BREAKING NEWS & EARNINGS FILTER
# ==========================================

def parse_ai_json(text: str) -> dict:
    """Extracts valid JSON from AI string response."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return {}


def scan_breaking_news_events():
    """
    Scans RSS feeds for high-conviction positive earnings/news.
    Filters out previews, mixed reports, and low-conviction signals.
    """
    logging.info("Starting Breaking News Scan...")
    
    for feed_url in BROAD_NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                link = entry.get("link", "")
                if link in processed_news_links:
                    continue
                
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                content_text = f"{title}\n{summary}"

                # Strict AI Prompting for News Filtering
                prompt = f"""
                You are a senior hedge fund analyst filtering financial news for high-probability BUY trade setups.
                Analyze the following news item:

                Headline/Content: "{content_text}"

                CRITICAL FILTERING RULES:
                1. PREVIEW / EXPECTATIONS FILTER: If this is an earnings preview, upcoming event prediction, or "what to expect" article (e.g., "earnings preview: what to look for"), set "is_earnings_actual_report" to false. We ONLY trade actual released reports.
                2. MIXED SIGNALS FILTER: If the company beat earnings but missed revenue, disappointed on network/product volume, or showed key operational weaknesses, set "mixed_or_negative_signals" to true.
                3. GUIDANCE RAISE: Did the company raise full-year/next-quarter guidance or report overwhelmingly strong actual results? Set "guidance_raised" accordingly.
                4. QUALITY SCORE: Rate the net impact from 1 to 10 (10 = massive clean beat & raised guidance like Bloom Energy; 1 = negative/disappointing).

                Return ONLY a JSON response in the exact format:
                {{
                  "is_relevant_trade": true/false,
                  "ticker": "SYMBOL" or null,
                  "is_earnings_actual_report": true/false,
                  "mixed_or_negative_signals": true/false,
                  "guidance_raised": true/false,
                  "earnings_quality_score": number_between_1_and_10,
                  "summary_hebrew": "ניתוח והסקה קצרים בעברית",
                  "recommendation_hebrew": "המלצה מפורטת בעברית"
                }}
                """

                ai_response = analyze_with_ai(prompt)
                parsed = parse_ai_json(ai_response)

                processed_news_links.add(link)

                # Strictly Validate Parameters
                if not parsed.get("is_relevant_trade") or not parsed.get("ticker"):
                    continue

                ticker = parsed.get("ticker", "").upper().strip()
                score = parsed.get("earnings_quality_score", 0)
                mixed = parsed.get("mixed_or_negative_signals", True)
                is_actual = parsed.get("is_earnings_actual_report", False)

                # SUPER FILTER CONDITIONS:
                # - Must NOT have mixed/negative signals
                # - Must be actual earnings/news (not preview)
                # - Quality score must be >= 8.5
                if mixed or not is_actual or score < 8.5:
                    logging.info(f"Skipped {ticker}: Score={score}, Mixed={mixed}, Actual={is_actual}")
                    continue

                if is_in_cooldown(ticker):
                    continue

                market_status = check_market_status(ticker)
                if market_status == "CLOSED":
                    continue

                is_liquid, price, _ = check_liquidity_and_price(ticker)
                if not is_liquid or price <= 0:
                    continue

                # Calculate Risk Parameters (3% SL, 6% TP1, 12% TP2)
                sl = round(price * 0.97, 2)
                tp1 = round(price * 1.06, 2)
                tp2 = round(price * 1.12, 2)

                # Send Telegram Alert
                msg = (
                    f"🚨 **איתות חדשותי מתפרץ (קנייה)**\n"
                    f"**מניה:** {ticker} | **עוצמה:** HIGH 🔥\n"
                    f"**מצב שוק:** {market_status}\n"
                    f"**כותרת/דיווח:** {title}\n\n"
                    f"💡 **ניתוח והסקה:**\n{parsed.get('summary_hebrew', '')}\n\n"
                    f"📌 **המלצת קנייה:** {parsed.get('recommendation_hebrew', '')}\n\n"
                    f"🎯 **פרמטרי פוזיציה (BUY):**\n"
                    f"• מחיר כניסה משוער: ${price:.2f}\n"
                    f"• Stop Loss (3%): ${sl:.2f}\n"
                    f"• Target 1 (6%): ${tp1:.2f}\n"
                    f"• Target 2 (12%): ${tp2:.2f}"
                )

                markup = InlineKeyboardMarkup()
                markup.add(
                    InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"calc_{ticker}_{price}_{sl}"),
                    InlineKeyboardButton("📈 TradingView", url=f"https://www.tradingview.com/symbols/{ticker}")
                )

                if bot and TELEGRAM_CHAT_ID:
                    bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown", reply_markup=markup)
                    set_cooldown(ticker)

        except Exception as e:
            logging.error(f"Error processing feed {feed_url}: {e}")


# ==========================================
# 4. TECHNICAL WATCHLIST SCANNER (MA20 PULLBACK)
# ==========================================

def scan_watchlist_technical():
    """
    Scans Watchlist against the 'MA20 Pullback Strategy' (שייטת ממוצע 20).
    Requires ALL 4 threshold conditions to pass.
    """
    logging.info("Starting Technical Watchlist Scan...")

    for ticker in WATCHLIST:
        try:
            if is_in_cooldown(ticker):
                continue

            market_status = check_market_status(ticker)
            if market_status == "CLOSED":
                continue

            is_liquid, price, _ = check_liquidity_and_price(ticker)
            if not is_liquid or price <= 0:
                continue

            df = yf.download(ticker, period="60d", interval="1d", progress=False)
            if df.empty or len(df) < 30:
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['SMA20'] = ta.sma(df['Close'], length=20)
            df['SMA40'] = ta.sma(df['Close'], length=40)
            df['SMA50'] = ta.sma(df['Close'], length=50)

            curr_sma20 = float(df['SMA20'].iloc[-1])
            curr_sma40 = float(df['SMA40'].iloc[-1]) if 'SMA40' in df and not pd.isna(df['SMA40'].iloc[-1]) else curr_sma20
            curr_sma50 = float(df['SMA50'].iloc[-1]) if 'SMA50' in df and not pd.isna(df['SMA50'].iloc[-1]) else curr_sma20

            # Condition 1: General Uptrend
            trend_ok = curr_sma20 > curr_sma40 or curr_sma20 > curr_sma50
            if not trend_ok:
                continue

            # Condition 2: Proximity to MA20 (0% - 1.5% distance, reject > 2%)
            dist_percent = ((price - curr_sma20) / curr_sma20) * 100
            if not (0.0 <= dist_percent <= 1.5):
                continue

            # Condition 3: Confirmation Candle (Hammer or strong green after touch)
            open_price = float(df['Open'].iloc[-1])
            high_price = float(df['High'].iloc[-1])
            low_price = float(df['Low'].iloc[-1])
            close_price = float(df['Close'].iloc[-1])

            body = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            is_hammer = (lower_shadow >= 2 * body) and (low_price <= curr_sma20 * 1.005)
            is_strong_green = (close_price > open_price) and (close_price >= (high_price + open_price) / 2)

            if not (is_hammer or is_strong_green):
                continue

            # Condition 4: 3% Stop Loss sits below MA20
            sl_price = round(price * 0.97, 2)
            if sl_price >= curr_sma20:
                continue

            tp1_price = round(price * 1.06, 2)

            pattern_desc = "נר פטיש שנגע ב-MA20" if is_hammer else "נר ירוק חזק בעקבות בדיקת MA20"

            msg = (
                f"🎯 **התראת כניסה טכנית: {ticker}**\n"
                f"• מחיר נוכחי: ${price:.2f}\n"
                f"• מיקום ממוצע 20: ${curr_sma20:.2f} (מרחק: {dist_percent:.2f}%)\n"
                f"• תבנית שנצפתה: {pattern_desc}\n"
                f"• מחיר כניסה מומלץ: ${price:.2f}\n"
                f"• סטופ לוס (3%): ${sl_price:.2f} (ממוקם מתחת לתמיכת MA20)\n"
                f"• יעד רווח ראשון (6%): ${tp1_price:.2f}"
            )

            markup = InlineKeyboardMarkup()
            markup.add(
                InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"calc_{ticker}_{price}_{sl_price}"),
                InlineKeyboardButton("📈 TradingView", url=f"https://www.tradingview.com/symbols/{ticker}")
            )

            if bot and TELEGRAM_CHAT_ID:
                bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown", reply_markup=markup)
                set_cooldown(ticker)

        except Exception as e:
            logging.error(f"Error scanning ticker {ticker}: {e}")


# ==========================================
# 5. TELEGRAM INTERACTIVE HANDLERS
# ==========================================

if bot:
    @bot.callback_query_handler(func=lambda call: call.data.startswith("calc_"))
    def handle_calc_risk(call):
        try:
            _, ticker, price_str, sl_str = call.data.split("_")
            price = float(price_str)
            sl = float(sl_str)

            msg = bot.send_message(
                call.message.chat.id,
                f"📊 **חישוב גודל פוזיציה עבור {ticker}**\n"
                f"מחיר כניסה: ${price} | סטופ לוס: ${sl}\n\n"
                f"אנא הקלד את סכום הסיכון הכספי שברצונך לסכן (לדוגמה: 100 או 500):"
            )
            bot.register_next_step_handler(msg, process_risk_amount, ticker, price, sl)
        except Exception as e:
            logging.error(f"Callback error: {e}")

    def process_risk_amount(message, ticker, price, sl):
        try:
            risk_amount = float(message.text.strip())
            risk_per_share = price - sl

            if risk_per_share <= 0:
                bot.reply_to(message, "שגיאה בחישוב: מחיר הסטופ לוס חייב להיות נמוך ממחיר הכניסה.")
                return

            shares = int(risk_amount / risk_per_share)
            total_position = shares * price
            potential_profit_6pct = shares * (price * 0.06)

            res_msg = (
                f"🧮 **תוצאות חישוב סיכון ל-{ticker}:**\n\n"
                f"• סכום סיכון מבוקש: ${risk_amount:.2f}\n"
                f"• כמות מניות לקנייה: **{shares}** מניות\n"
                f"• שווי פוזיציה כולל: ${total_position:,.2f}\n"
                f"• רווח פוטנציאלי ביעד 1 (6%): ${potential_profit_6pct:,.2f}"
            )
            bot.send_message(message.chat.id, res_msg, parse_mode="Markdown")
        except ValueError:
            bot.reply_to(message, "נא להזין מספר תקין בלבד.")
        except Exception as e:
            logging.error(f"Risk calculation error: {e}")

    @bot.message_handler(commands=['start'])
    def cmd_start(message):
        bot.reply_to(message, "🤖 PazPSTrading Bot פעיל ומוכן לפעולה!")

    @bot.message_handler(commands=['test_news'])
    def cmd_test_news(message):
        bot.reply_to(message, "🔍 מריץ סריקת חדשות ודוחות ידנית...")
        scan_breaking_news_events()

    @bot.message_handler(commands=['test_tech'])
    def cmd_test_tech(message):
        bot.reply_to(message, "📊 מריץ סריקה טכנית ידנית על ה-Watchlist...")
        scan_watchlist_technical()


# ==========================================
# 6. FLASK WEBHOOK & AP-SCHEDULER SETUP
# ==========================================

@app.route("/", methods=["GET"])
def index():
    return "PazPSTrading Bot is running smoothly!", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        if bot:
            bot.process_new_updates([update])
        return '', 200
    return 'Invalid', 400

def setup_scheduler():
    scheduler = BackgroundScheduler(timezone=pytz.utc)
    # Scan RSS news feeds every 15 minutes
    scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
    # Scan technical watchlist every 30 minutes
    scheduler.add_job(scan_watchlist_technical, 'interval', minutes=30)
    scheduler.start()

if __name__ == "__main__":
    setup_scheduler()
    
    if bot and RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        logging.info(f"Webhook set to: {webhook_url}")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
