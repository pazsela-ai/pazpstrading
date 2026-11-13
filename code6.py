import os
import re
import json
import time
import sqlite3
import logging
import warnings
import calendar
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
import openai

# Suppress Pandas/yfinance deprecation warnings from flooding Render logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Pandas4Warning.*")
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except ImportError:
    pass

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

gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.warning(f"Gemini (google.genai) initialization failed: {e}")

groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logging.warning(f"Groq initialization failed: {e}")

# Initialize Telegram Bot & Flask App
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, threaded=False) if TELEGRAM_BOT_TOKEN else None
app = Flask(__name__)

# Broad News Feeds (RSS)
BROAD_NEWS_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://search.cnbc.com/rs/search/combinedrender?source=yahoo&partnerId=2001&collection=finance&limit=25",
    "https://www.google.com/finance/news/rss",
    "https://feeds.feedburner.com/realmoney",
    "https://www.sec.gov/news/pressreleases.rss"
]

# Watchlist for Technical Scanner (~46 Tickers)
WATCHLIST = [
    # US Stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO",
    "BAC", "COST", "MCD", "WMT", "CSCO", "AMD", "CRM", "INTC", "NFLX", "QCOM",
    # Israeli Stocks (.TA)
    "TEVA.TA", "NICE.TA", "ICL.TA", "MZR.TA", "LUMI.TA", "POLI.TA", "DSCT.TA", "FIBI.TA",
    "HARL.TA", "PHOE.TA", "ENLT.TA", "ENOG.TA", "BEZQ.TA", "AZRG.TA", "BIG.TA", "SAEN.TA"
]

TZ_IL = pytz.timezone("Asia/Jerusalem")
TZ_US = pytz.timezone("America/New_York")
DB_PATH = "trading_bot_state.db"


# ==========================================
# 0. PERSISTENT SQLITE STATE MANAGEMENT
# ==========================================

def init_db():
    """Initializes SQLite tables so state survives Render server restarts."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS ticker_cooldowns (
                ticker TEXT PRIMARY KEY,
                last_sent_ts REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS processed_links (
                link TEXT PRIMARY KEY,
                processed_ts REAL
            )
        ''')
        # Cleanup records older than 7 days
        cutoff = time.time() - (7 * 86400)
        c.execute('DELETE FROM processed_links WHERE processed_ts < ?', (cutoff,))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"SQLite init error: {e}")

init_db()

def is_in_cooldown(ticker: str, hours: int = 48) -> bool:
    """Checks if ticker was alerted within the last 48 hours."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT last_sent_ts FROM ticker_cooldowns WHERE ticker = ?', (ticker,))
        row = c.fetchone()
        conn.close()
        if row:
            if (time.time() - row[0]) < (hours * 3600):
                return True
        return False
    except Exception:
        return False

def set_cooldown(ticker: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('REPLACE INTO ticker_cooldowns (ticker, last_sent_ts) VALUES (?, ?)', (ticker, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"SQLite set_cooldown error: {e}")

def is_link_processed(link: str) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT 1 FROM processed_links WHERE link = ?', (link,))
        exists = c.fetchone() is not None
        conn.close()
        return exists
    except Exception:
        return False

def mark_link_processed(link: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('REPLACE INTO processed_links (link, processed_ts) VALUES (?, ?)', (link, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"SQLite mark_link error: {e}")


# ==========================================
# 1. FAILOVER AI ENGINE
# ==========================================

def analyze_with_ai(prompt: str) -> str:
    """Executes AI requests with direct failover: OpenAI -> Gemini -> Groq."""
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

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {e}. Falling back to Groq...")

    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API error: {e}.")

    logging.error("ALL AI providers failed to return a response!")
    return ""


# ==========================================
# 2. MARKET HOURS, LIVE PRICE & SAFETY CHECKS
# ==========================================

def get_tradingview_url(ticker: str) -> str:
    """Converts .TA tickers to TLV format for TradingView URLs."""
    if ticker.endswith(".TA"):
        clean_sym = ticker.replace(".TA", "")
        return f"https://www.tradingview.com/symbols/TLV-{clean_sym}/"
    return f"https://www.tradingview.com/symbols/{ticker}/"

def check_market_status(ticker: str) -> str:
    now_il = datetime.datetime.now(TZ_IL)
    weekday = now_il.weekday()

    if ticker.endswith(".TA"):
        if weekday in [6, 0, 1, 2, 3]:
            current_time = now_il.time()
            if datetime.time(10, 0) <= current_time <= datetime.time(17, 25):
                return "Israel (TASE) (OPEN)"
        return "CLOSED"
    else:
        now_us = datetime.datetime.now(TZ_US)
        us_weekday = now_us.weekday()
        if us_weekday < 5:
            us_time = now_us.time()
            if datetime.time(4, 0) <= us_time < datetime.time(9, 30):
                return "US Markets (PRE_MARKET)"
            elif datetime.time(9, 30) <= us_time <= datetime.time(16, 0):
                return "US Markets (OPEN)"
            elif datetime.time(16, 0) < us_time <= datetime.time(20, 0):
                return "US Markets (AFTER_HOURS)"
        return "CLOSED"

def get_live_quote_and_liquidity(ticker: str) -> dict:
    """
    Fetches live price (including extended hours), previous close, gap %,
    and 20-day average liquidity.
    """
    try:
        t_obj = yf.Ticker(ticker)
        hist = t_obj.history(period="25d")
        if hist.empty or len(hist) < 5:
            return {"valid": False}

        avg_volume_20d = float(hist['Volume'].tail(20).mean())
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else float(hist['Close'].iloc[-1])
        live_price = float(hist['Close'].iloc[-1])

        # Try fetching real-time / extended hours price via fast_info
        try:
            fast = t_obj.fast_info
            if hasattr(fast, 'last_price') and fast.last_price and fast.last_price > 0:
                live_price = float(fast.last_price)
            if hasattr(fast, 'previous_close') and fast.previous_close and fast.previous_close > 0:
                prev_close = float(fast.previous_close)
        except Exception:
            pass

        daily_liquidity_usd = live_price * avg_volume_20d
        if daily_liquidity_usd < 500000:
            return {"valid": False}

        gap_percent = ((live_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

        return {
            "valid": True,
            "live_price": live_price,
            "prev_close": prev_close,
            "gap_percent": gap_percent,
            "avg_volume_20d": avg_volume_20d,
            "liquidity": daily_liquidity_usd
        }
    except Exception as e:
        logging.error(f"Quote/liquidity error for {ticker}: {e}")
        return {"valid": False}

def is_earnings_within_days(ticker: str, days: int = 3) -> bool:
    """Checks if the company has an upcoming earnings report within N days."""
    try:
        t_obj = yf.Ticker(ticker)
        cal = t_obj.calendar
        if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
            earnings_dates = cal['Earnings Date']
            if earnings_dates:
                next_date = earnings_dates[0]
                if isinstance(next_date, datetime.datetime):
                    next_date = next_date.date()
                today = datetime.date.today()
                delta = (next_date - today).days
                if 0 <= delta <= days:
                    return True
    except Exception:
        pass
    return False

def is_article_fresh(entry, max_age_hours: float = 3.0) -> bool:
    """Rejects RSS articles older than max_age_hours to prevent stale alerts."""
    parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed_time:
        return True  # Allow if feed doesn't provide timestamp
    try:
        pub_ts = calendar.timegm(parsed_time)
        age_hours = (time.time() - pub_ts) / 3600.0
        return age_hours <= max_age_hours
    except Exception:
        return True

def get_intraday_volume_pace(ticker: str) -> float:
    """Estimates normalized relative volume pace based on elapsed market session."""
    if ticker.endswith(".TA"):
        now_il = datetime.datetime.now(TZ_IL)
        start = now_il.replace(hour=10, minute=0, second=0, microsecond=0)
        total_minutes = 445.0  # 10:00 to 17:25
        elapsed = (now_il - start).total_seconds() / 60.0
    else:
        now_us = datetime.datetime.now(TZ_US)
        start = now_us.replace(hour=9, minute=30, second=0, microsecond=0)
        total_minutes = 390.0  # 09:30 to 16:00
        elapsed = (now_us - start).total_seconds() / 60.0

    if elapsed <= 15:
        return 1.0  # Too early in session to penalize volume
    return min(elapsed / total_minutes, 1.0)


# ==========================================
# 3. TWO-TIER BREAKING NEWS & EARNINGS FILTER
# ==========================================

def parse_ai_json(text: str) -> dict:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return {}

def scan_breaking_news_events():
    """
    Two-Tier News Scanner with Stale-News Filter & Anti-Chasing Protection:
    - Tier 1 (Top Priority): Contracts, FDA/Medical trials, Defense/Security tensions, Geopolitics, M&A (Score >= 7.0).
    - Tier 2 (Strict Secondary Priority): Exceptional Actual Earnings Reports (Score >= 8.5, no mixed signals, strong guidance).
    """
    logging.info("Starting Two-Tier Breaking News & Earnings Scan...")
    checked_count = 0

    for feed_url in BROAD_NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:12]:
                link = entry.get("link", "")
                if not link or is_link_processed(link):
                    continue

                if not is_article_fresh(entry, max_age_hours=3.0):
                    mark_link_processed(link)
                    continue

                checked_count += 1
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                content_text = f"{title}\n{summary}"

                prompt = f"""
                You are a senior hedge fund analyst categorizing and filtering market-moving news for high-probability BUY setups.
                Analyze the following news item:

                Headline/Content: "{content_text}"

                CLASSIFICATION & RULES:
                1. EVENT TYPE ("event_type"):
                   - "breaking_catalyst": Major contracts, medical/clinical trial results, FDA approvals, defense/security tensions, geopolitical developments, M&A, strategic partnerships, or major non-earnings catalysts.
                   - "earnings_report": Quarterly/annual financial earnings releases.
                   - "other": Routine commentary, analyst opinions, earnings previews ("what to expect"), or irrelevant news.
                2. PREVIEW FILTER: If this is an earnings preview or prediction before a report, classify as "other" and set "is_relevant_trade" to false.
                3. MIXED SIGNALS ("mixed_or_negative"): True if there are notable caveats (e.g., beat EPS but missed revenue, weak volume, lowered guidance, or trial setbacks).
                4. GUIDANCE / CATALYST STRENGTH ("strong_catalyst_or_guidance"): True if guidance was raised OR the non-earnings catalyst is major and concrete.
                5. QUALITY SCORE ("quality_score", 1 to 10):
                   - For "breaking_catalyst": 7.0-10.0 = concrete positive catalyst affecting a specific stock (contract win, successful trial, defense demand).
                   - For "earnings_report": 8.5-10.0 = ONLY exceptional clean beats WITH raised guidance or massive outperformance (like Bloom Energy). Anything routine/minor/mixed must score <= 8.0.

                Return ONLY valid JSON:
                {{
                  "is_relevant_trade": true/false,
                  "event_type": "breaking_catalyst" or "earnings_report" or "other",
                  "ticker": "SYMBOL" or null,
                  "mixed_or_negative": true/false,
                  "strong_catalyst_or_guidance": true/false,
                  "quality_score": number_1_to_10,
                  "summary_hebrew": "ניתוח והסקה קצרים בעברית",
                  "recommendation_hebrew": "המלצה מפורטת בעברית"
                }}
                """

                ai_response = analyze_with_ai(prompt)
                mark_link_processed(link)
                if not ai_response:
                    continue

                parsed = parse_ai_json(ai_response)
                if not parsed.get("is_relevant_trade") or not parsed.get("ticker"):
                    continue

                ticker = parsed.get("ticker", "").upper().strip()
                event_type = parsed.get("event_type", "other")
                score = float(parsed.get("quality_score", 0))
                mixed = parsed.get("mixed_or_negative", True)
                strong_catalyst = parsed.get("strong_catalyst_or_guidance", False)

                if mixed or event_type == "other":
                    continue

                if event_type == "breaking_catalyst":
                    if score < 7.0:
                        logging.info(f"Skipped Tier-1 Catalyst {ticker}: Score={score} (< 7.0)")
                        continue
                    header_type = "🚨 **איתות חדשותי מתפרץ - עדיפות עליונה (קנייה)**"
                elif event_type == "earnings_report":
                    if score < 8.5 or not strong_catalyst:
                        logging.info(f"Skipped Tier-2 Earnings {ticker}: Score={score}, StrongGuidance={strong_catalyst}")
                        continue
                    header_type = "📄 **איתות דוח כספי מובחר (קנייה)**"
                else:
                    continue

                if is_in_cooldown(ticker, hours=48):
                    continue

                market_status = check_market_status(ticker)
                if market_status == "CLOSED":
                    logging.info(f"Skipped {ticker}: Market is CLOSED")
                    continue

                quote = get_live_quote_and_liquidity(ticker)
                if not quote.get("valid"):
                    continue

                price = quote["live_price"]
                gap_pct = quote["gap_percent"]

                # Anti-Chasing Filter: Block entry if stock already surged > 4.5%
                if gap_pct > 4.5:
                    logging.info(f"Anti-Chasing blocked {ticker}: Already up +{gap_pct:.2f}% from previous close.")
                    continue

                currency_sym = "₪" if ticker.endswith(".TA") else "$"
                sl = round(price * 0.965, 2)  # 3.5% Stop Loss for news volatility
                tp1 = round(price * 1.07, 2)
                tp2 = round(price * 1.14, 2)
                tv_url = get_tradingview_url(ticker)

                gap_note = f" (שינוי יומי נוכחי: {gap_pct:+.2f}%)" if abs(gap_pct) >= 0.1 else ""

                msg = (
                    f"{header_type}\n"
                    f"**מניה:** {ticker} | **עוצמה:** HIGH 🔥\n"
                    f"**מצב שוק:** {market_status}{gap_note}\n"
                    f"**כותרת/דיווח:** {title}\n\n"
                    f"💡 **ניתוח והסקה:**\n{parsed.get('summary_hebrew', '')}\n\n"
                    f"📌 **המלצת קנייה:** {parsed.get('recommendation_hebrew', '')}\n\n"
                    f"🎯 **פרמטרי פוזיציה (BUY):**\n"
                    f"• מחיר כניסה חי משוער: {currency_sym}{price:.2f}\n"
                    f"• Stop Loss (3.5%): {currency_sym}{sl:.2f}\n"
                    f"• Target 1 (7%): {currency_sym}{tp1:.2f} *(בהגעה ליעד: מימוש 50% והעלאת סטופ למחיר כניסה)*\n"
                    f"• Target 2 (14%): {currency_sym}{tp2:.2f}"
                )

                markup = InlineKeyboardMarkup()
                markup.add(
                    InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"calc_{ticker}_{price}_{sl}"),
                    InlineKeyboardButton("📈 TradingView", url=tv_url)
                )

                if bot and TELEGRAM_CHAT_ID:
                    bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown", reply_markup=markup)
                    set_cooldown(ticker)
                    logging.info(f"Sent {event_type} alert for {ticker} (Score: {score}, Gap: {gap_pct:.2f}%)")

        except Exception as e:
            logging.error(f"Error processing feed {feed_url}: {e}")

    logging.info(f"Finished News Scan. Evaluated {checked_count} fresh articles.")


# ==========================================
# 4. ENHANCED TECHNICAL WATCHLIST SCANNER
# ==========================================

def scan_watchlist_technical():
    """
    Enhanced MA20 Pullback Scanner:
    1. Rising SMA20 slope + SMA20 > SMA40/SMA50
    2. Pullback zone: 0.0% to 1.5% above SMA20
    3. RSI(14) between 45 and 65 (healthy pullback momentum)
    4. Confirmation candle (Hammer or Strong Green) + Normalized Volume Pace >= 1.1x
    5. Earnings Blackout check (skips if earnings within 3 days)
    6. ATR-adjusted Stop Loss below SMA20 (capped between 3.0% and 5.5%)
    """
    logging.info("Starting Enhanced Technical Watchlist Scan...")

    for ticker in WATCHLIST:
        try:
            if is_in_cooldown(ticker, hours=48):
                continue

            market_status = check_market_status(ticker)
            if market_status == "CLOSED":
                continue

            # Skip if earnings report is imminent (within 3 days)
            if is_earnings_within_days(ticker, days=3):
                logging.info(f"Skipped {ticker}: Upcoming earnings within 3 days.")
                continue

            quote = get_live_quote_and_liquidity(ticker)
            if not quote.get("valid"):
                continue

            price = quote["live_price"]
            avg_vol_20d = quote["avg_volume_20d"]

            df = yf.download(ticker, period="75d", interval="1d", progress=False)
            if df.empty or len(df) < 35:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['SMA20'] = ta.sma(df['Close'], length=20)
            df['SMA40'] = ta.sma(df['Close'], length=40)
            df['SMA50'] = ta.sma(df['Close'], length=50)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            curr_sma20 = float(df['SMA20'].iloc[-1])
            prev_sma20_5d = float(df['SMA20'].iloc[-6])
            curr_sma40 = float(df['SMA40'].iloc[-1]) if 'SMA40' in df and not pd.isna(df['SMA40'].iloc[-1]) else curr_sma20
            curr_sma50 = float(df['SMA50'].iloc[-1]) if 'SMA50' in df and not pd.isna(df['SMA50'].iloc[-1]) else curr_sma20
            curr_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 55.0
            curr_atr = float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else (price * 0.025)

            # 1. Trend & Rising SMA20 Slope Check
            trend_ok = (curr_sma20 > curr_sma40 or curr_sma20 > curr_sma50) and (curr_sma20 > prev_sma20_5d)
            if not trend_ok:
                continue

            # 2. Pullback Zone Check (0% to 1.5% above SMA20)
            dist_percent = ((price - curr_sma20) / curr_sma20) * 100
            if not (0.0 <= dist_percent <= 1.5):
                continue

            # 3. RSI Pullback Sweet Spot (45 to 65)
            if not (45.0 <= curr_rsi <= 65.0):
                continue

            # 4. Candle Pattern & Normalized Volume Check
            open_price = float(df['Open'].iloc[-1])
            high_price = float(df['High'].iloc[-1])
            low_price = float(df['Low'].iloc[-1])
            close_price = float(df['Close'].iloc[-1])
            curr_volume = float(df['Volume'].iloc[-1])

            body = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price

            is_hammer = (lower_shadow >= 2 * body) and (low_price <= curr_sma20 * 1.005)
            is_strong_green = (close_price > open_price) and (close_price >= (high_price + open_price) / 2)

            if not (is_hammer or is_strong_green):
                continue

            session_pace = get_intraday_volume_pace(ticker)
            expected_vol_so_far = avg_vol_20d * session_pace
            rvol = curr_volume / expected_vol_so_far if expected_vol_so_far > 0 else 1.0

            if rvol < 1.1:
                continue

            # 5. Dynamic ATR Stop Loss (Below SMA20, bounded between 3.0% and 5.5%)
            atr_stop = price - (1.5 * curr_atr)
            sma_stop = curr_sma20 * 0.992  # Just below SMA20 support
            raw_sl = min(atr_stop, sma_stop)

            min_sl_price = price * 0.945  # Max 5.5% risk cap
            max_sl_price = price * 0.970  # Min 3.0% risk floor

            sl_price = round(max(min_sl_price, min(max_sl_price, raw_sl)), 2)
            if sl_price >= curr_sma20:
                continue

            risk_pct = ((price - sl_price) / price) * 100
            tp1_pct = risk_pct * 2.0
            tp2_pct = risk_pct * 3.0

            tp1_price = round(price * (1 + tp1_pct / 100), 2)
            tp2_price = round(price * (1 + tp2_pct / 100), 2)

            currency_sym = "₪" if ticker.endswith(".TA") else "$"
            tv_url = get_tradingview_url(ticker)
            pattern_desc = "נר פטיש" if is_hammer else "נר ירוק חזק עקב נגיעה ב-MA20"

            msg = (
                f"🎯 **התראת כניסה: אסטרטגיית שייטת ממוצע 20**\n"
                f"**מניה:** {ticker} | **עוצמה:** HIGH 🔥\n"
                f"**מצב שוק:** {market_status}\n\n"
                f"💡 **ניתוח והסקה:**\n"
                f"• מיקום ממוצע 20: {currency_sym}{curr_sma20:.2f} (מרחק: {dist_percent:.2f}% | שיפוע עולה)\n"
                f"• תבנית שנצפתה: {pattern_desc} (נפח יחסי: {rvol:.1f}x | RSI: {curr_rsi:.0f})\n"
                f"• בדיקת סטופ לוס ({risk_pct:.1f}% מותאם ATR): {currency_sym}{sl_price:.2f} (מתחת לתמיכת MA20)\n\n"
                f"📌 **המלצת קנייה:** כניסה מומלצת על בסיס נגיעה ואישור תמיכה ב-MA20.\n\n"
                f"🎯 **פרמטרי פוזיציה (BUY):**\n"
                f"• מחיר כניסה משוער: {currency_sym}{price:.2f}\n"
                f"• Stop Loss ({risk_pct:.1f}%): {currency_sym}{sl_price:.2f}\n"
                f"• Target 1 ({tp1_pct:.1f}%): {currency_sym}{tp1_price:.2f} *(מימוש 50% והעלאת סטופ לכניסה)*\n"
                f"• Target 2 ({tp2_pct:.1f}%): {currency_sym}{tp2_price:.2f}"
            )

            markup = InlineKeyboardMarkup()
            markup.add(
                InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"calc_{ticker}_{price}_{sl_price}"),
                InlineKeyboardButton("📈 TradingView", url=tv_url)
            )

            if bot and TELEGRAM_CHAT_ID:
                bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown", reply_markup=markup)
                set_cooldown(ticker)
                logging.info(f"Sent technical alert for {ticker} (RVOL: {rvol:.2f}, RSI: {curr_rsi:.1f}, Risk: {risk_pct:.1f}%)")

        except Exception as e:
            logging.error(f"Error scanning ticker {ticker}: {e}")

    logging.info("Finished Enhanced Technical Watchlist Scan.")


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
            currency_sym = "₪" if ticker.endswith(".TA") else "$"

            msg = bot.send_message(
                call.message.chat.id,
                f"📊 **חישוב גודל פוזיציה עבור {ticker}**\n"
                f"מחיר כניסה: {currency_sym}{price} | סטופ לוס: {currency_sym}{sl}\n\n"
                f"אנא הקלד את סכום הסיכון הכספי שברצונך לסכן (לדוגמה: 100 או 500):"
            )
            bot.register_next_step_handler(msg, process_risk_amount, ticker, price, sl)
        except Exception as e:
            logging.error(f"Callback error: {e}")

    def process_risk_amount(message, ticker, price, sl):
        try:
            risk_amount = float(message.text.strip())
            risk_per_share = price - sl
            currency_sym = "₪" if ticker.endswith(".TA") else "$"

            if risk_per_share <= 0:
                bot.reply_to(message, "שגיאה בחישוב: מחיר הסטופ לוס חייב להיות נמוך ממחיר הכניסה.")
                return

            shares = int(risk_amount / risk_per_share)
            total_position = shares * price
            potential_profit_2r = shares * (risk_per_share * 2)

            res_msg = (
                f"🧮 **תוצאות חישוב סיכון ל-{ticker}:**\n\n"
                f"• סכום סיכון מבוקש: {currency_sym}{risk_amount:.2f}\n"
                f"• כמות מניות לקנייה: **{shares}** מניות\n"
                f"• שווי פוזיציה כולל: {currency_sym}{total_position:,.2f}\n"
                f"• רווח פוטנציאלי ביעד 1 (יחס 1:2): {currency_sym}{potential_profit_2r:,.2f}"
            )
            bot.send_message(message.chat.id, res_msg, parse_mode="Markdown")
        except ValueError:
            bot.reply_to(message, "נא להזין מספר תקין בלבד.")
        except Exception as e:
            logging.error(f"Risk calculation error: {e}")

    @bot.message_handler(commands=['start', 'status'])
    def cmd_start_or_status(message):
        bot.reply_to(message, "🤖 PazPSTrading Bot v3.0 פעיל, מחובר לשרת ומוכן לפעולה!")

    @bot.message_handler(commands=['test_news'])
    def cmd_test_news(message):
        bot.reply_to(message, "🔍 מריץ סריקת חדשות ודוחות דו-שכבתית ידנית...")
        scan_breaking_news_events()

    @bot.message_handler(commands=['test_tech'])
    def cmd_test_tech(message):
        bot.reply_to(message, "📊 מריץ סריקה טכנית משודרגת על ה-Watchlist...")
        scan_watchlist_technical()


# ==========================================
# 6. FLASK WEBHOOK & SCHEDULER SETUP
# ==========================================

@app.route("/", methods=["GET"])
def index():
    return "PazPSTrading Bot v3.0 is running smoothly!", 200

@app.route("/set_webhook", methods=["GET"])
def set_webhook_route():
    if bot and RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        return f"Webhook successfully set to {webhook_url}", 200
    return "Failed to set Webhook", 400

@app.route("/webhook", methods=["POST"])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        if bot:
            bot.process_new_updates([update])
        return '', 200
    return 'Invalid', 400


scheduler = BackgroundScheduler(timezone=pytz.utc)
if not scheduler.running:
    scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
    scheduler.add_job(scan_watchlist_technical, 'interval', minutes=30)
    scheduler.start()

if bot and RENDER_EXTERNAL_URL:
    try:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
    except Exception as e:
        logging.error(f"Error registering Webhook: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
