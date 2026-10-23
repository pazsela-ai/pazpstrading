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
from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import feedparser

# AI Engines
import openai
import google.generativeai as genai
from groq import Groq

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '').strip()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '').strip()

RENDER_EXTERNAL_URL = os.getenv('RENDER_EXTERNAL_URL', '').strip()

# Initialize TeleBot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)

# Initialize AI Clients
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS & MEMORY
# ---------------------------------------------------------------------------
# Watchlist for Technical Scanner (~50 top Israeli & US stocks)
WATCHLIST = [
    # Israel (TASE)
    "LUMI.TA", "POLI.TA", "NICE.TA", "ESLT.TA", "ICL.TA", "DLEKG.TA", "BEZQ.TA",
    "HARL.TA", "MZTF.TA", "FIBI.TA", "BIG.TA", "AZRG.TA", "TEVA.TA", "ALHE.TA",
    # US (Tech & Leaders)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC",
    "PLTR", "INDA", "SPY", "QQQ", "IWM", "DIS", "BA", "CAT", "JPM", "BAC", "V", "MA",
    "UNH", "JNJ", "PFE", "XOM", "CVX", "COST", "WMT", "HD", "PG", "CRM", "NOW"
]

# RSS Feeds for Breaking News Engine
BROAD_NEWS_FEEDS = [
    # Israeli Financial & Capital Market Feeds
    "https://www.bizportal.co.il/rss/bizportalrss.xml",
    "https://www.globes.co.il/webservice/rss/rssfeeder.asmx/FeedsNode?nodeId=1725",
    "https://www.calcalist.co.il/Integration/StoryRss3656.xml",
    "https://www.themarker.com/coby/rss/market",
    "https://news.google.com/rss/search?q=site:maya.tase.co.il&hl=he&gl=IL&ceid=IL:he",
    "https://www.ynet.co.il/Integration/StoryRss538.xml",
    # US / Global Feeds
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "https://search.cnbc.com/rs/search/combinedrenderer.view?query=stocks&partnerId=2000&target=news",
    "https://www.sec.gov/news/pressreleases.rss"
]

# Cooldown memory: {ticker: timestamp}
sent_ticker_cooldowns = {}
COOLDOWN_HOURS = 6

# Processed RSS entries cache to prevent duplicate scans
seen_rss_guids = set()

# ---------------------------------------------------------------------------
# AI FAILOVER ENGINE
# ---------------------------------------------------------------------------
def call_ai_with_failover(prompt, system_instruction=""):
    """
    Tries OpenAI -> Gemini -> Groq.
    Returns the text response.
    """
    # 1. OpenAI
    if OPENAI_API_KEY:
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}. Trying Gemini...")

    # 2. Gemini
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.warning(f"Gemini failed: {e}. Trying Groq...")

    # 3. Groq
    if groq_client:
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq failed: {e}.")

    return None

# ---------------------------------------------------------------------------
# MARKET HOURS & LIQUIDITY FILTERS
# ---------------------------------------------------------------------------
def check_market_status(ticker):
    """
    Checks if market is currently open or pre-market.
    - Israel (.TA): Sun-Thu, 09:30-17:30 Israel Time.
    - US: Mon-Fri, 11:00-23:00 Israel Time (including pre-market).
    """
    tz = pytz.timezone('Asia/Jerusalem')
    now = datetime.now(tz)
    weekday = now.weekday()  # Mon=0, Tue=1, ..., Sun=6

    if ticker.endswith('.TA'):
        # Sunday (6) to Thursday (3)
        if weekday in [6, 0, 1, 2, 3]:
            start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
            end_time = now.replace(hour=17, minute=30, second=0, microsecond=0)
            return start_time <= now <= end_time
        return False
    else:
        # Monday (0) to Friday (4)
        if weekday in [0, 1, 2, 3, 4]:
            start_time = now.replace(hour=11, minute=0, second=0, microsecond=0) # Pre-market start
            end_time = now.replace(hour=23, minute=0, second=0, microsecond=0)
            return start_time <= now <= end_time
        return False

def check_liquidity_and_price(ticker):
    """
    Validates minimum daily volume turnover:
    - Israel (.TA): 500,000 ILS
    - US: $500,000 USD
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            return False, 0, "No data available"

        latest_price = hist['Close'].iloc[-1]
        avg_volume = hist['Volume'].tail(20).mean()
        avg_turnover = latest_price * avg_volume

        if ticker.endswith('.TA'):
            min_turnover = 500000  # 500K ILS
            if avg_turnover < min_turnover:
                return False, latest_price, f"Low ILS turnover ({avg_turnover:,.0f} ILS)"
        else:
            min_turnover = 500000  # 500K USD
            if avg_turnover < min_turnover:
                return False, latest_price, f"Low USD turnover (${avg_turnover:,.0f})"

        return True, latest_price, "Liquidity OK"
    except Exception as e:
        return False, 0, f"Error checking liquidity: {str(e)}"

def is_in_cooldown(ticker):
    if ticker in sent_ticker_cooldowns:
        last_sent = sent_ticker_cooldowns[ticker]
        if datetime.now() - last_sent < timedelta(hours=COOLDOWN_HOURS):
            return True
    return False

def set_cooldown(ticker):
    sent_ticker_cooldowns[ticker] = datetime.now()

# ---------------------------------------------------------------------------
# TECHNICAL ANALYSIS ENGINE (MA20 Pullback Strategy)
# ---------------------------------------------------------------------------
def analyze_ma20_pullback(ticker):
    """
    Strict MA20 Pullback Strategy:
    1. Trend: SMA20 > SMA40 or SMA50, positive slope.
    2. Distance: High in last 3 days was above SMA20. Current/Low distance to SMA20 is 0% to 1.5% (not stretched > 2%).
    3. Confirmation Candle: Hammer or strong bullish close in top half after testing MA20.
    4. SL Alignment: 3% SL must sit below SMA20.
    """
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            return None

        # Flatten columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['SMA20'] = ta.sma(df['Close'], length=20)
        df['SMA40'] = ta.sma(df['Close'], length=40)
        df['SMA50'] = ta.sma(df['Close'], length=50)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        price = float(latest['Close'])
        high = float(latest['High'])
        low = float(latest['Low'])
        open_p = float(latest['Open'])
        sma20 = float(latest['SMA20'])
        sma40 = float(latest['SMA40']) if pd.notna(latest['SMA40']) else sma20
        sma50 = float(latest['SMA50']) if pd.notna(latest['SMA50']) else sma20

        # Condition 1: Trend
        if not (sma20 > sma40 or sma20 > sma50):
            return None
        if sma20 <= float(df['SMA20'].iloc[-5]):  # Upward slope check
            return None

        # Condition 2: Distance from MA20
        high_3d = df['High'].tail(3).max()
        if high_3d < sma20:
            return None

        dist_pct = ((price - sma20) / sma20) * 100
        low_dist_pct = ((low - sma20) / sma20) * 100

        # Must be near MA20 (0% - 1.5%), not stretched > 2%
        if dist_pct > 2.0 or dist_pct < -0.5:
            return None

        # Condition 3: Confirmation Candle
        body = abs(price - open_p)
        candle_range = high - low
        lower_shadow = min(open_p, price) - low

        is_hammer = (lower_shadow >= 2 * body) and (low <= sma20 * 1.005)
        is_strong_green = (price > open_p) and (price >= high - (candle_range * 0.3)) and (low <= sma20 * 1.01)

        if not (is_hammer or is_strong_green):
            return None

        candle_pattern = "נר פטיש על ממוצע 20" if is_hammer else "נר ירוק עוצמתי לאחר בדיקת MA20"

        # Condition 4: Risk Parameters & SL Placement
        sl_price = price * 0.97  # 3% SL
        tp_price = price * 1.06  # 6% TP

        if sl_price >= sma20:
            return None  # SL must sit below SMA20 support

        return {
            "ticker": ticker,
            "price": price,
            "sma20": sma20,
            "dist_pct": dist_pct,
            "pattern": candle_pattern,
            "sl": sl_price,
            "tp1": tp_price
        }

    except Exception as e:
        logger.error(f"Error in MA20 analysis for {ticker}: {e}")
        return None

# ---------------------------------------------------------------------------
# BREAKING NEWS SCANNER
# ---------------------------------------------------------------------------
NEWS_SYSTEM_PROMPT = """
You are an expert financial analyst monitoring news feeds for immediate stock trading catalysts.
Analyze the provided news headline and content.

CRITICAL INSTRUCTION FOR ISRAELI STOCKS:
- If the news mentions an Israeli company (in Hebrew or English), output its official Yahoo Finance ticker with '.TA' suffix.
Examples:
- "בנק הפועלים" / "הפועלים" -> "POLI.TA"
- "בנק לאומי" / "לאומי" -> "LUMI.TA"
- "אלביט מערכות" / "אלביט" -> "ESLT.TA"
- "נייס" -> "NICE.TA"
- "קבוצת דלק" / "דלק" -> "DLEKG.TA"
- "טבע" -> "TEVA.TA"

Return ONLY a strict JSON object with this exact structure:
{
  "ticker": "SYMBOL or SYMBOL.TA or UNKNOWN",
  "company_name": "Company Name",
  "sentiment": "BULLISH / BEARISH / NEUTRAL",
  "urgency": "HIGH / MEDIUM / LOW",
  "summary": "1-2 sentences summarizing the breaking event in Hebrew"
}
"""

def scan_breaking_news_events():
    logger.info("Starting Breaking News scan...")
    for feed_url in BROAD_NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:  # Process top 5 recent entries per feed
                guid = entry.get('id', entry.get('link', entry.get('title')))
                if guid in seen_rss_guids:
                    continue
                seen_rss_guids.add(guid)

                title = entry.get('title', '')
                summary = entry.get('summary', '')
                content = f"Title: {title}\nSummary: {summary}"

                ai_res = call_ai_with_failover(content, NEWS_SYSTEM_PROMPT)
                if not ai_res:
                    continue

                # Extract JSON
                json_match = re.search(r'\{.*\}', ai_res, re.DOTALL)
                if not json_match:
                    continue

                data = json.loads(json_match.group(0))
                ticker = data.get('ticker', 'UNKNOWN').upper()

                if ticker == 'UNKNOWN' or is_in_cooldown(ticker):
                    continue

                if data.get('sentiment') != 'BULLISH' or data.get('urgency') != 'HIGH':
                    continue

                # Check liquidity
                is_liquid, current_price, msg = check_liquidity_and_price(ticker)
                if not is_liquid:
                    logger.info(f"News trigger for {ticker} skipped: {msg}")
                    continue

                set_cooldown(ticker)
                send_news_alert(ticker, data, current_price)

        except Exception as e:
            logger.error(f"Error scanning feed {feed_url}: {e}")

def send_news_alert(ticker, data, price):
    currency_symbol = "₪" if ticker.endswith('.TA') else "$"
    sl = price * 0.96  # 4% ATR approximation
    tp = price * 1.08  # 8% TP target

    text = (
        f"🚨 **אירוע חדשותי מתפרץ: {ticker}**\n"
        f"🏢 חברה: {data.get('company_name')}\n"
        f"💵 מחיר נוכחי: {currency_symbol}{price:.2f}\n"
        f"📢 תקציר: {data.get('summary')}\n\n"
        f"🎯 מחיר כניסה מוצע: {currency_symbol}{price:.2f}\n"
        f"🛑 סטופ לוס: {currency_symbol}{sl:.2f}\n"
        f"🏁 יעד רווח: {currency_symbol}{tp:.2f}\n"
    )

    markup = InlineKeyboardMarkup()
    btn = InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"risk_{ticker}_{price:.2f}_{sl:.2f}")
    markup.add(btn)

    bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode="Markdown", reply_markup=markup)

# ---------------------------------------------------------------------------
# TECHNICAL SCANNER TASK
# ---------------------------------------------------------------------------
def scan_watchlist_technical():
    logger.info("Starting Technical Watchlist Scan...")
    for ticker in WATCHLIST:
        try:
            if is_in_cooldown(ticker):
                continue

            if not check_market_status(ticker):
                continue

            is_liquid, current_price, _ = check_liquidity_and_price(ticker)
            if not is_liquid:
                continue

            setup = analyze_ma20_pullback(ticker)
            if setup:
                set_cooldown(ticker)
                send_technical_alert(setup)

        except Exception as e:
            logger.error(f"Error scanning technical ticker {ticker}: {e}")

def send_technical_alert(setup):
    ticker = setup['ticker']
    curr = "₪" if ticker.endswith('.TA') else "$"

    text = (
        f"🎯 **התראת כניסה טכנית: {ticker}**\n"
        f"• מחיר נוכחי: {curr}{setup['price']:.2f}\n"
        f"• מיקום ממוצע 20: {curr}{setup['sma20']:.2f} (מרחק: {setup['dist_pct']:.2f}%)\n"
        f"• תבנית שנצפתה: {setup['pattern']}\n"
        f"• מחיר כניסה מומלץ: {curr}{setup['price']:.2f}\n"
        f"• סטופ לוס (3%): {curr}{setup['sl']:.2f} (מתחת לתמיכה)\n"
        f"• יעד רווח ראשון (6%): {curr}{setup['tp1']:.2f}\n"
    )

    markup = InlineKeyboardMarkup()
    btn = InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"risk_{ticker}_{setup['price']:.2f}_{setup['sl']:.2f}")
    markup.add(btn)

    bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode="Markdown", reply_markup=markup)

# ---------------------------------------------------------------------------
# TELEGRAM HANDLERS & RISK CALCULATOR
# ---------------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def cmd_start(message):
    bot.reply_to(message, "🤖 PazPSTrading Bot מחובר ופעיל בזמן אמת!")

@bot.message_handler(commands=['test_news'])
def cmd_test_news(message):
    bot.reply_to(message, "🔍 מריץ סריקת חדשות יזוקה...")
    scan_breaking_news_events()

@bot.message_handler(commands=['test_tech'])
def cmd_test_tech(message):
    bot.reply_to(message, "📊 מריץ סריקה טכנית יזוקה על רשימת המעקב...")
    scan_watchlist_technical()

@bot.message_handler(commands=['tech'])
def cmd_tech_single(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "אנא ציין טיקר, לדוגמה: `/tech TEVA.TA` או `/tech AAPL`", parse_mode="Markdown")
        return
    
    ticker = parts[1].upper()
    bot.reply_to(message, f"🔍 מנתח את {ticker} לפי אסטרטגיית 'שייטת ממוצע 20'...")
    setup = analyze_ma20_pullback(ticker)
    if setup:
        send_technical_alert(setup)
    else:
        bot.send_message(message.chat.id, f"❌ המניה {ticker} אינה עונה כעת על תנאי הסף של האסטרטגיה.")

@bot.callback_query_handler(func=lambda call: call.data.startswith('risk_'))
def handle_risk_calc(call):
    _, ticker, entry_str, sl_str = call.data.split('_')
    entry = float(entry_str)
    sl = float(sl_str)
    curr = "₪" if ticker.endswith('.TA') else "$"

    msg = bot.send_message(
        call.message.chat.id,
        f"🔢 **מחשבון סיכון עבור {ticker}**\n"
        f"מחיר כניסה: {curr}{entry}\n"
        f"סטופ לוס: {curr}{sl}\n\n"
        f"אנא הקלד את **סכום הסיכון הכספי** שברצונך לסכן ברמת העסקה (לדוגמה: 500):"
    )
    bot.register_next_step_handler(msg, process_risk_amount, ticker, entry, sl, curr)

def process_risk_amount(message, ticker, entry, sl, curr):
    try:
        risk_amount = float(message.text)
        risk_per_share = entry - sl
        if risk_per_share <= 0:
            bot.reply_to(message, "חישוב שגוי: מחיר הסטופ לוס חייב להיות נמוך ממחיר הכניסה.")
            return

        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry
        target_profit = shares * (entry * 1.06 - entry)

        bot.reply_to(
            message,
            f"📊 **תכנית ניהול סיכונים עבור {ticker}:**\n"
            f"• כמות מניות לקנייה: **{shares:,}**\n"
            f"• שווי פוזיציה כולל: **{curr}{position_value:,.2f}**\n"
            f"• סיכון מקסימלי בעסקה: **{curr}{risk_amount:,.2f}**\n"
            f"• רווח פוטנציאלי ביעד (6%): **{curr}{target_profit:,.2f}**",
            parse_mode="Markdown"
        )
    except ValueError:
        bot.reply_to(message, "ערך לא תקין. אנא הזן מספר בלבד.")

# ---------------------------------------------------------------------------
# WEBHOOK & FLASK SERVER Setup
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return "PazPSTrading Bot is running!", 200

# Fix 404 Webhook Routing (Supports both /webhook and /{TELEGRAM_BOT_TOKEN})
@app.route('/webhook', methods=['POST'])
@app.route(f'/{TELEGRAM_BOT_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    else:
        return 'Invalid content-type', 403

def setup_webhook():
    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        logger.info(f"Setting Webhook to: {webhook_url}")
        bot.remove_webhook()
        time.sleep(1)
        bot.set_webhook(url=webhook_url)

# ---------------------------------------------------------------------------
# SCHEDULER INITIALIZATION
# ---------------------------------------------------------------------------
scheduler = BackgroundScheduler(timezone="Asia/Jerusalem")
scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
scheduler.add_job(scan_watchlist_technical, 'interval', minutes=30)
scheduler.start()

# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    setup_webhook()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
