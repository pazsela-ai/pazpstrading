import os
import re
import json
import time
import logging
from datetime import datetime, timedelta
import pytz
from flask import Flask, request
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import feedparser
from apscheduler.schedulers.background import BackgroundScheduler

# AI Providers
import google.generativeai as genai
from groq import Groq
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("PazPSTradingBot")

# ---------------------------------------------------------------------------
# Environment Variables & Clients Setup
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()

# Initialize AI Clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)

# Cache / Memory for State Management
seen_news_guids = set()
sent_ticker_cooldowns = {}  # ticker -> datetime when cooldown expires
user_trade_contexts = {}    # chat_id -> trade data dict

# ---------------------------------------------------------------------------
# Watchlist and Feeds Definition
# ---------------------------------------------------------------------------
WATCHLIST = [
    # US Stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC",
    "BA", "DIS", "JPM", "V", "MA", "PG", "UNH", "HD", "BAC", "XOM",
    "CVX", "LLY", "MRK", "ABBV", "PEP", "KO", "COST", "CSCO", "AVGO", "ORCL",
    # Israeli Stocks (TASE)
    "POLI.TA", "LUMI.TA", "NICE.TA", "ESLT.TA", "DLEKG.TA", "BEZQ.TA", "TEVA.TA", "ICL.TA",
    "HARL.TA", "AZRG.TA", "MZTF.TA", "FIBI.TA", "DIMON.TA", "MLSR.TA", "ENLT.TA"
]

BROAD_NEWS_FEEDS = [
    # Israeli Financial Feeds (.TA focus)
    "https://www.bizportal.co.il/rss/bizportalrss.xml",                             # Bizportal
    "https://www.globes.co.il/webservice/rss/rssfeeder.asmx/FeedsNode?nodeId=1725", # Globes Market
    "https://www.calcalist.co.il/Integration/StoryRss3656.xml",                     # Calcalist Market
    "https://www.themarker.com/coby/rss/market",                                     # TheMarker
    "https://news.google.com/rss/search?q=site:maya.tase.co.il&hl=he&gl=IL&ceid=IL:he",# Google News - MAYA reports
    "https://www.ynet.co.il/Integration/StoryRss538.xml",                           # Ynet Economy
    
    # Global / US Feeds
    "http://feeds.bbci.co.uk/news/business/rss.xml",                                # BBC Business
    "https://search.cnbc.com/rs/search/combinedrenderer.view?query=stocks&partnerId=2000&target=news", # CNBC
    "https://www.sec.gov/news/pressreleases.rss"                                     # SEC Press Releases
]

NEWS_SYSTEM_PROMPT = """
You are a top-tier financial analyst monitoring breaking news to identify stock trading opportunities.
Analyze the provided news headline and text.

CRITICAL INSTRUCTIONS FOR ISRAELI & US STOCKS:
1. Identify if the news is directly relevant to a specific publicly traded company.
2. If the news involves an Israeli company or is from an Israeli feed, translate/map the company name to its Yahoo Finance symbol with the '.TA' suffix.
   Examples:
   - "בנק הפועלים" / "הפועלים" -> "POLI.TA"
   - "בנק לאומי" / "לאומי" -> "LUMI.TA"
   - "אלביט מערכות" / "אלביט" -> "ESLT.TA"
   - "נייס" -> "NICE.TA"
   - "קבוצת דלק" / "דלק" -> "DLEKG.TA"
   - "בזק" -> "BEZQ.TA"
   - "טבע" -> "TEVA.TA"
   - "איסיאל" / "כימיקלים לישראל" -> "ICL.TA"
3. If US company: use standard ticker (e.g., AAPL, NVDA, TSLA).
4. If no specific stock ticker can be identified, set ticker to "UNKNOWN".

Return ONLY a valid JSON object with NO markdown formatting around it:
{
  "ticker": "SYMBOL.TA or SYMBOL",
  "company_name": "Company Name",
  "sentiment": "BULLISH / BEARISH / NEUTRAL",
  "urgency": "HIGH / MEDIUM / LOW",
  "summary": "Brief analysis of the event impact"
}
"""

# ---------------------------------------------------------------------------
# AI Multi-Engine Failover System
# ---------------------------------------------------------------------------
def call_ai_analysis(prompt_text):
    """
    Tries OpenAI -> Gemini -> Groq sequentially for text analysis / JSON extraction.
    """
    # 1. Try OpenAI
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            return parse_json_response(content)
        except Exception as e:
            logger.warning(f"OpenAI Primary Failed: {e}. Falling back to Gemini.")

    # 2. Try Gemini
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            full_prompt = f"{NEWS_SYSTEM_PROMPT}\n\nContent to analyze:\n{prompt_text}"
            response = model.generate_content(full_prompt)
            return parse_json_response(response.text.strip())
        except Exception as e:
            logger.warning(f"Gemini Failover Failed: {e}. Falling back to Groq.")

    # 3. Try Groq
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            return parse_json_response(content)
        except Exception as e:
            logger.error(f"Groq Failover Failed: {e}. All AI options exhausted.")

    return None

def parse_json_response(text):
    """Extracts valid JSON from raw AI text response."""
    try:
        clean_text = re.sub(r'```(?:json)?', '', text).strip('` \n\r')
        return json.loads(clean_text)
    except Exception as e:
        logger.error(f"Failed to parse JSON from AI: {text} | Error: {e}")
        return None

# ---------------------------------------------------------------------------
# Market Status and Liquidity Checkers
# ---------------------------------------------------------------------------
def check_market_status(ticker):
    """
    Determines if market is open or in valid trading/pre-market hours for ticker.
    Israeli stocks (.TA): Sun-Thu, 09:30 - 17:30 Israel Time.
    US stocks: Mon-Fri, Pre-market starts 11:00 IL time, regular market 16:30-23:00 IL time.
    """
    tz_il = pytz.timezone('Asia/Jerusalem')
    now_il = datetime.now(tz_il)
    weekday = now_il.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6

    if ticker.endswith('.TA'):
        # Israel Trading Days: Sunday (6) through Thursday (3)
        if weekday in [6, 0, 1, 2, 3]:
            start_time = now_il.replace(hour=9, minute=30, second=0, microsecond=0)
            end_time = now_il.replace(hour=17, minute=30, second=0, microsecond=0)
            if start_time <= now_il <= end_time:
                return True, "Israel Market Open"
        return False, "Israel Market Closed"
    else:
        # US Trading Days: Monday (0) through Friday (4)
        if weekday in [0, 1, 2, 3, 4]:
            pre_market_start = now_il.replace(hour=11, minute=0, second=0, microsecond=0)
            market_close = now_il.replace(hour=23, minute=0, second=0, microsecond=0)
            if pre_market_start <= now_il <= market_close:
                return True, "US Market / Pre-market Open"
        return False, "US Market Closed"

def check_liquidity_and_price(ticker):
    """
    Checks liquidity and fetches latest price.
    Israeli stocks (.TA) use ILS liquidity threshold (500,000 NIS).
    US stocks use USD liquidity threshold ($500,000).
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        
        if hist.empty or len(hist) < 5:
            return False, 0, "Insufficient market history"

        latest_price = hist['Close'].iloc[-1]
        avg_volume = hist['Volume'].tail(20).mean()
        avg_daily_turnover = latest_price * avg_volume

        if ticker.endswith('.TA'):
            MIN_TURNOVER_ILS = 500000  # ₪500,000 daily turnover
            if avg_daily_turnover < MIN_TURNOVER_ILS:
                return False, latest_price, f"Low ILS turnover: ₪{avg_daily_turnover:,.0f}"
        else:
            MIN_TURNOVER_USD = 500000  # $500,000 daily turnover
            if avg_daily_turnover < MIN_TURNOVER_USD:
                return False, latest_price, f"Low USD turnover: ${avg_daily_turnover:,.0f}"

        return True, latest_price, "OK"

    except Exception as e:
        return False, 0, f"Error checking liquidity: {str(e)}"

def is_in_cooldown(ticker):
    """Checks if ticker is currently in a 6-hour cooldown window."""
    if ticker in sent_ticker_cooldowns:
        if datetime.now() < sent_ticker_cooldowns[ticker]:
            return True
    return False

def set_cooldown(ticker, hours=6):
    """Sets a cooldown window for a ticker."""
    sent_ticker_cooldowns[ticker] = datetime.now() + timedelta(hours=hours)

# ---------------------------------------------------------------------------
# Technical Strategy Engine ("MA20 Pullback Strategy")
# ---------------------------------------------------------------------------
def analyze_ma20_pullback(ticker):
    """
    Evaluates MA20 Pullback Strategy with strictly calibrated conditions:
    1. Trend: SMA20 > SMA40/SMA50 with positive slope.
    2. Distance: High of last 3 days > SMA20, current price 0% to 1.5% above SMA20.
    3. Confirmation Candle: Hammer or Strong Green candle.
    4. Stop Loss 3% placed below SMA20.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo", interval="1d")

        if df.empty or len(df) < 50:
            return None

        # Calculate Indicators
        df['SMA20'] = ta.sma(df['Close'], length=20)
        df['SMA40'] = ta.sma(df['Close'], length=40)
        df['SMA50'] = ta.sma(df['Close'], length=50)

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        price = curr['Close']
        sma20 = curr['SMA20']
        sma40 = curr['SMA40'] if not pd.isna(curr['SMA40']) else curr['SMA20']
        sma50 = curr['SMA50'] if not pd.isna(curr['SMA50']) else curr['SMA20']

        # Condition 1: Uptrend confirmation
        if not (sma20 > sma40 and sma20 > sma50):
            return None

        sma20_slope = (sma20 - df['SMA20'].iloc[-10]) / df['SMA20'].iloc[-10]
        if sma20_slope <= 0:
            return None

        # Condition 2: Distance from MA20 (0% to 1.5%)
        pct_from_ma20 = ((price - sma20) / sma20) * 100
        recent_high = df['High'].tail(3).max()

        if not (recent_high > sma20 and 0.0 <= pct_from_ma20 <= 1.5):
            return None

        # Condition 3: Confirmation Candle Trigger (Hammer or Strong Green)
        body = abs(curr['Close'] - curr['Open'])
        lower_shadow = min(curr['Open'], curr['Close']) - curr['Low']
        
        is_hammer = lower_shadow >= (2 * body) and curr['Low'] <= sma20
        is_strong_green = (curr['Close'] > curr['Open']) and (curr['Close'] >= (curr['High'] - (curr['High'] - curr['Low']) * 0.5))

        if not (is_hammer or is_strong_green):
            return None

        candle_pattern = "Hammer Candle on MA20" if is_hammer else "Strong Green Reversal on MA20"

        # Condition 4: Stop Loss (3%) placing below SMA20
        entry_price = round(price, 2)
        stop_loss = round(entry_price * 0.97, 2)
        take_profit = round(entry_price * 1.06, 2)

        if stop_loss >= sma20:
            # Adjust SL slightly below SMA20 if standard 3% stays above it
            stop_loss = round(sma20 * 0.99, 2)

        currency_symbol = "₪" if ticker.endswith('.TA') else "$"

        return {
            "ticker": ticker,
            "price": entry_price,
            "ma20": round(sma20, 2),
            "dist_pct": round(pct_from_ma20, 2),
            "pattern": candle_pattern,
            "entry": entry_price,
            "sl": stop_loss,
            "tp1": take_profit,
            "currency": currency_symbol
        }

    except Exception as e:
        logger.error(f"Error in technical analysis for {ticker}: {e}")
        return None

# ---------------------------------------------------------------------------
# Telegram UI & Callback Handlers
# ---------------------------------------------------------------------------
def create_signal_keyboard(ticker, entry_price, sl_price, currency):
    """Generates interactive Telegram inline buttons."""
    keyboard = InlineKeyboardMarkup()
    btn_calc = InlineKeyboardButton(
        f"🎯 Calculate Position / Risk ({currency})",
        callback_data=f"calc_{ticker}_{entry_price}_{sl_price}_{currency}"
    )
    keyboard.add(btn_calc)
    return keyboard

def send_telegram_alert(message, keyboard=None):
    """Sends notification to Telegram target chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram Bot Token or Chat ID not configured.")
        return
    try:
        bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode="Markdown",
            reply_markup=keyboard
        )
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('calc_'))
def handle_calc_callback(call):
    try:
        _, ticker, entry_str, sl_str, curr = call.data.split('_')
        entry_price = float(entry_str)
        sl_price = float(sl_str)

        user_trade_contexts[call.message.chat.id] = {
            "ticker": ticker,
            "entry": entry_price,
            "sl": sl_price,
            "currency": curr
        }

        msg = (
            f"📊 *Position Sizing Calculator for {ticker}*\n"
            f"Entry Price: {curr}{entry_price}\n"
            f"Stop Loss: {curr}{sl_price}\n\n"
            f" Please reply with the maximum monetary risk amount you are willing to risk in {curr}:"
        )
        bot.send_message(call.message.chat.id, msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in callback handler: {e}")

@bot.message_handler(func=lambda message: message.chat.id in user_trade_contexts)
def handle_risk_input(message):
    try:
        chat_id = message.chat.id
        ctx = user_trade_contexts.pop(chat_id)

        risk_amount = float(message.text.strip())
        entry = ctx["entry"]
        sl = ctx["sl"]
        curr = ctx["currency"]
        ticker = ctx["ticker"]

        risk_per_share = entry - sl
        if risk_per_share <= 0:
            bot.reply_to(message, "Invalid Stop Loss level.")
            return

        shares_count = int(risk_amount / risk_per_share)
        total_position_value = shares_count * entry
        potential_profit = shares_count * (entry * 0.06)

        summary = (
            f"✅ *Risk & Position Sizing Summary ({ticker})*\n\n"
            f"• Max Monetary Risk: {curr}{risk_amount:,.2f}\n"
            f"• Recommended Position Size: *{shares_count:,} shares*\n"
            f"• Total Position Value: {curr}{total_position_value:,.2f}\n"
            f"• Potential Gain at TP1 (6%): {curr}{potential_profit:,.2f}\n"
        )
        bot.send_message(chat_id, summary, parse_mode="Markdown")

    except ValueError:
        bot.send_message(message.chat.id, "Please enter a valid numeric risk amount.")
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")

# ---------------------------------------------------------------------------
# Bot Commands Handler
# ---------------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def cmd_start(message):
    bot.reply_to(message, "🤖 PazPSTrading Bot is online and actively scanning markets!")

@bot.message_handler(commands=['test_news'])
def cmd_test_news(message):
    bot.reply_to(message, "🔍 Running manual Breaking News scan...")
    scan_breaking_news_events()

@bot.message_handler(commands=['test_tech'])
def cmd_test_tech(message):
    bot.reply_to(message, "📊 Running manual Technical Watchlist scan...")
    scan_watchlist_technical()

@bot.message_handler(commands=['tech'])
def cmd_tech_single(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "Usage: /tech <TICKER> (e.g. /tech POLI.TA or /tech AAPL)")
            return
        ticker = parts[1].upper()
        res = analyze_ma20_pullback(ticker)
        if res:
            curr = res["currency"]
            msg = (
                f"🎯 *Manual Technical Analysis: {res['ticker']}*\n"
                f"- Current Price: {curr}{res['price']}\n"
                f"- MA20 Level: {curr}{res['ma20']} (Dist: {res['dist_pct']}%)\n"
                f"- Pattern: {res['pattern']}\n"
                f"- Entry Price: {curr}{res['entry']}\n"
                f"- Stop Loss (3%): {curr}{res['sl']}\n"
                f"- Take Profit 1 (6%): {curr}{res['tp1']}\n"
            )
            kb = create_signal_keyboard(res['ticker'], res['entry'], res['sl'], curr)
            bot.send_message(message.chat.id, msg, parse_mode="Markdown", reply_markup=kb)
        else:
            bot.reply_to(message, f"No MA20 Pullback setup found for {ticker}.")
    except Exception as e:
        bot.reply_to(message, f"Error analyzing ticker: {e}")

# ---------------------------------------------------------------------------
# Scanner Engines
# ---------------------------------------------------------------------------
def scan_breaking_news_events():
    """
    Breaking News Scanner (Runs every 15 mins).
    Scans RSS feeds, utilizes AI to discover Israeli and US ticker opportunities,
    validates market liquidity/hours, and posts Telegram signals.
    """
    logger.info("Starting Breaking News scanner cycle...")
    for feed_url in BROAD_NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:  # Process latest 5 entries per feed
                guid = entry.get('id', entry.get('link', entry.get('title')))
                if guid in seen_news_guids:
                    continue
                seen_news_guids.add(guid)

                title = entry.get('title', '')
                summary = entry.get('summary', '')
                news_text = f"Title: {title}\nSummary: {summary}"

                # AI Extraction
                ai_res = call_ai_analysis(news_text)
                if not ai_res or ai_res.get("ticker") == "UNKNOWN":
                    continue

                ticker = ai_res.get("ticker").upper()
                sentiment = ai_res.get("sentiment", "NEUTRAL")

                if sentiment != "BULLISH":
                    continue

                if is_in_cooldown(ticker):
                    logger.info(f"Ticker {ticker} skipped due to active cooldown.")
                    continue

                # Market Status Check
                is_open, market_msg = check_market_status(ticker)
                if not is_open:
                    logger.info(f"Skipping {ticker}: Market is closed ({market_msg}).")
                    continue

                # Liquidity Check
                has_liquidity, price, liq_msg = check_liquidity_and_price(ticker)
                if not has_liquidity:
                    logger.info(f"Skipping {ticker}: Liquidity check failed ({liq_msg}).")
                    continue

                # Set Cooldown
                set_cooldown(ticker)

                # Format Telegram Alert
                currency = "₪" if ticker.endswith('.TA') else "$"
                sl = round(price * 0.97, 2)
                tp = round(price * 1.06, 2)

                alert_msg = (
                    f"📰 *BREAKING NEWS EVENT ALERT: {ticker}*\n\n"
                    f"• *Company:* {ai_res.get('company_name', ticker)}\n"
                    f"• *Current Price:* {currency}{price:.2f}\n"
                    f"• *Sentiment:* 🚀 {sentiment}\n"
                    f"• *Urgency:* {ai_res.get('urgency', 'MEDIUM')}\n\n"
                    f"• *Event Summary:* {ai_res.get('summary', '')}\n\n"
                    f"🎯 *Suggested Dynamic Parameters:*\n"
                    f"- Entry: {currency}{price:.2f}\n"
                    f"- Stop Loss (3%): {currency}{sl:.2f}\n"
                    f"- Take Profit (6%): {currency}{tp:.2f}\n"
                )

                kb = create_signal_keyboard(ticker, price, sl, currency)
                send_telegram_alert(alert_msg, kb)

        except Exception as e:
            logger.error(f"Error processing RSS feed {feed_url}: {e}")

def scan_watchlist_technical():
    """
    Technical Watchlist Scanner (Runs every 30 mins).
    Scans predefined WATCHLIST using strictly the MA20 Pullback Strategy.
    """
    logger.info("Starting Technical Watchlist scanner cycle...")
    for ticker in WATCHLIST:
        try:
            if is_in_cooldown(ticker):
                continue

            is_open, _ = check_market_status(ticker)
            if not is_open:
                continue

            has_liquidity, _, _ = check_liquidity_and_price(ticker)
            if not has_liquidity:
                continue

            res = analyze_ma20_pullback(ticker)
            if res:
                set_cooldown(ticker)
                curr = res["currency"]
                alert_msg = (
                    f"🎯 *TECHNICAL ENTRY ALERT: [{res['ticker']}]*\n\n"
                    f"- Current Price: {curr}{res['price']}\n"
                    f"- MA20 Level: {curr}{res['ma20']} (Dist: {res['dist_pct']}%)\n"
                    f"- Observed Pattern: {res['pattern']}\n"
                    f"- Recommended Entry Price: {curr}{res['entry']}\n"
                    f"- Stop Loss (3%): {curr}{res['sl']} (Placed below MA20)\n"
                    f"- First Take Profit (6%): {curr}{res['tp1']}\n"
                )
                kb = create_signal_keyboard(res['ticker'], res['entry'], res['sl'], curr)
                send_telegram_alert(alert_msg, kb)

        except Exception as e:
            logger.error(f"Error scanning technical ticker {ticker}: {e}")

# ---------------------------------------------------------------------------
# Flask & Webhook Setup
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return "PazPSTrading Bot Server is Running!", 200

@app.route(f'/{TELEGRAM_BOT_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    return 'Forbidden', 403

def setup_webhook():
    if RENDER_EXTERNAL_URL and TELEGRAM_BOT_TOKEN:
        webhook_url = f"{RENDER_EXTERNAL_URL}/{TELEGRAM_BOT_TOKEN}"
        bot.remove_webhook()
        time.sleep(1)
        bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook configured to: {webhook_url}")

# ---------------------------------------------------------------------------
# Application Initialization
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    setup_webhook()

    # Scheduler Setup
    scheduler = BackgroundScheduler(timezone="Asia/Jerusalem")
    scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
    scheduler.add_job(scan_watchlist_technical, 'interval', minutes=30)
    scheduler.start()

    # Run Flask App
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
