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
    import openai
except ImportError:
    openai = None

# ---------------------------------------------------------------------------
# 1. Basic Setup & Environment Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()

# Initialize TeleBot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None

# Initialize Flask
app = Flask(__name__)

# Initialize AI Clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if (GROQ_API_KEY and Groq) else None

if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

# ---------------------------------------------------------------------------
# Global State & Constants
# ---------------------------------------------------------------------------
seen_articles = set()
sent_ticker_cooldowns = {}
user_calc_state = {}

WATCHLIST = [
    # US Stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "XOM", "HD", "MA", "BAC", "ABBV", "PFE", "AVGO",
    "COST", "DIS", "KO", "CSCO", "PEP", "WMT", "AMD", "NFLX", "INTC", "CRM",
    
    # Israeli Stocks (TASE)
    "TEVA.TA", "NICE.TA", "LUMI.TA", "POLI.TA", "FIBI.TA", "DSCT.TA", "ICL.TA",
    "HARL.TA", "ESLT.TA", "BEZQ.TA", "AZRG.TA", "PZL.TA", "ENLT.TA", "MTRX.TA",
    "DLEKG.TA", "CHKP", "WIX", "MNDY", "SEDG", "TSEM"
]

BROAD_NEWS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.ynet.co.il/Integration/StoryRss1854.xml", # Ynet Main
    "https://www.ynet.co.il/Integration/StoryRss6.xml",    # Ynet Economy
    "https://news.google.com/rss/search?q=aviation+airline+incident+flight&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=war+military+strike+tensions+Middle+East&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=pharma+FDA+approval+clinical+trial+phase&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=oil+gas+strait+hormuz+pipeline+energy&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=acquisition+merger+deal+contract+partnership&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=sanctions+tariffs+semiconductor+export+policy&hl=en-US&gl=US&ceid=US:en",
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom",
    "https://news.google.com/rss/search?q=earnings+report+quarterly+results+revenue+EPS+beat&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=site:maya.tase.co.il+דוח+מיידי+OR+דוח+כספי+OR+תוצאות&hl=he&gl=IL&ceid=IL:he",
    "https://news.google.com/rss/search?q=גלובס+מניות+OR+בורסה+OR+דוחות&hl=he&gl=IL&ceid=IL:he"
]

# ---------------------------------------------------------------------------
# 3. AI Failover Engine
# ---------------------------------------------------------------------------
def call_ai_failover(prompt: str) -> str:
    """
    Attempts to call AI APIs in order: OpenAI -> Gemini -> Groq.
    Returns the raw string output from the first successful call.
    """
    # 1. Try OpenAI
    if OPENAI_API_KEY and openai:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}. Falling back to Gemini...")

    # 2. Try Gemini
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.warning(f"Gemini call failed: {e}. Falling back to Groq...")

    # 3. Try Groq
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq call failed: {e}. All AI engines exhausted.")

    return ""

# ---------------------------------------------------------------------------
# 4. Market & Financial Utilities
# ---------------------------------------------------------------------------
def check_market_status(ticker: str) -> dict:
    is_israel = ticker.endswith(".TA")
    tz_il = pytz.timezone("Asia/Jerusalem")
    now_il = datetime.now(tz_il)
    weekday = now_il.weekday()
    time_str = now_il.strftime("%H:%M")

    if is_israel:
        if weekday in [6, 0, 1, 2, 3]: # Sunday to Thursday
            if "09:59" <= time_str <= "17:30":
                return {"status": "OPEN", "market": "Israel (TASE)"}
        return {"status": "CLOSED", "market": "Israel (TASE)"}
    else:
        if weekday in [0, 1, 2, 3, 4]: # Monday to Friday
            if "11:00" <= time_str < "16:30":
                return {"status": "PRE_MARKET", "market": "US Markets"}
            elif "16:30" <= time_str <= "23:00":
                return {"status": "OPEN", "market": "US Markets"}
        return {"status": "CLOSED", "market": "US Markets"}

def check_liquidity_and_price(ticker: str, custom_sl=None, custom_tp1=None) -> dict:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="100d", interval="1d")
        if df.empty or len(df) < 20:
            return {"valid": False, "reason": "Insufficient market data"}

        current_price = df["Close"].iloc[-1]
        avg_vol_20 = df["Volume"].tail(20).mean()
        
        # Liquidity Check Adaptation for Israeli Stocks (.TA)
        is_israel = ticker.endswith(".TA")
        dollar_volume = current_price * avg_vol_20
        if is_israel:
            # Convert ILS volume equivalent to USD (~3.7 USD/ILS) for threshold validation
            usd_equivalent_volume = dollar_volume / 3.7
            if usd_equivalent_volume < 200000: # Lower liquidity threshold adapted for TASE
                return {"valid": False, "reason": "Low TASE liquidity"}
        else:
            if dollar_volume < 500000:
                return {"valid": False, "reason": "Low US liquidity"}

        # Custom SL/TP or ATR calculation
        if custom_sl is not None and custom_tp1 is not None:
            sl = custom_sl
            tp1 = custom_tp1
            tp2 = current_price * 1.12
        else:
            df.ta.atr(length=14, append=True)
            atr_col = [c for c in df.columns if c.startswith("ATRr_")]
            atr_val = df[atr_col[0]].iloc[-1] if atr_col else (df["High"].iloc[-1] - df["Low"].iloc[-1])
            sl = max(0.01, current_price - (1.2 * atr_val))
            tp1 = current_price + (2.0 * atr_val)
            tp2 = current_price + (4.0 * atr_val)

        return {
            "valid": True,
            "current_price": float(current_price),
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
# 5. Telegram UI & Notifications
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
    currency_symbol = "₪" if ticker.endswith(".TA") else "$"

    if "EARNINGS" in alert_type or "REPORT" in alert_type:
        badge = "📑 איתות דיווח חברה / דוח כספי (קנייה)"
    elif "NEWS" in alert_type:
        badge = "🚨 איתות חדשותי מתפרץ (קנייה)"
    else:
        badge = "🎯 התראת כניסה: אסטרטגיית שייטת ממוצע 20"

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
        f"<b>📌 המלצת קנייה:</b> {recommendation}\n\n"
        f"<b>🎯 פרמטרי פוזיציה (BUY):</b>\n"
        f"• מחיר כניסה משוער: <b>{currency_symbol}{price:.2f}</b>\n"
        f"• Stop Loss (3%): <b>{currency_symbol}{sl:.2f}</b>\n"
        f"• Target 1 (6%): <b>{currency_symbol}{tp1:.2f}</b>\n"
    )

    if intensity.upper() == "HIGH" and tp2 > 0:
        msg += f"• Target 2: <b>{currency_symbol}{tp2:.2f}</b>\n"

    tv_ticker = ticker.replace(".TA", "")
    tv_url = f"https://www.tradingview.com/symbols/TASE-{tv_ticker}/" if ticker.endswith(".TA") else f"https://www.tradingview.com/symbols/{tv_ticker}/"

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
            for entry in feed.entries:
                art_id = entry.get("id", entry.get("link", entry.get("title", "")))
                
                if art_id in seen_articles:
                    continue
                seen_articles.add(art_id)

                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))

                pub_parsed = entry.get("published_parsed")
                if pub_parsed:
                    pub_dt = datetime(*pub_parsed[:6])
                    if datetime.utcnow() - pub_dt > timedelta(hours=24):
                        continue

                prompt = f"""
אתה מנוע AI אנליטי מבריק למסחר פיננסי. תפקידך לנתח את הידיעה הבאה ולדרג את פוטנציאל ההשקעה:

כותרת הידיעה: {title}
תקציר הידיעה: {summary}

הנחיות לדירוג וניתוח:
1. **המלצות קנייה בלבד (BUY / LONG):** נתח האם לידיעה יש אימפקט חיובי (עליית מחיר / הזדמנות מסחרית) על מניה ספציפית בארה"ב או בישראל.
   - חשוב מאוד: עבור מניות ישראליות הנסחרות בבורסת תל אביב (TASE), החזר את הטיקר עם סיומת .TA (למשל TEVA.TA, LUMI.TA, POLI.TA, ICL.TA, ESLT.TA, HARL.TA).
   - עבור מניות בארה"ב, החזר טיקר רגיל (למשל AAPL, NVDA, TSLA).
   - אם הידיעה שלילית (SELL / SHORT) או לא משפיעה על מניה דירקטיבית - החזר "NONE".

2. **דירוג פוטנציאל ההשקעה (score 1-100):**
   - דרג 70 ומעלה רק אם לידיעה יש משמעות כלכלית/פנדמנטלית ממשית (עסקה, ניסוי קליני, אישור FDA, דוח כספי חזק, חוזה ענק, או אירוע מאקרו/גאו-פוליטי משפיע).
   - אם מדובר ברעשי רקע, פוליטיקה, פלילים או אירוע ללא אימפקט מסחרי - תן ציון נמוך מ-70 והחזר "NONE".

3. **ללא הזיות:** חלץ Ticker רק אם יש קשר סיבתי-כלכלי הגיוני.

החזר JSON בלבד:
{{
  "ticker": "NONE",
  "score": 0,
  "category": "NEWS_EVENT",
  "intensity": "HIGH",
  "analysis": "הסבר מפורט בעברית על ההסקה וההזדמנות החיובית",
  "recommendation": "המלצת קנייה קצרה בעברית"
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

                score = int(data.get("score", 0))
                ticker = str(data.get("ticker", "")).strip().upper()

                if score < 70 or not ticker or ticker in ["NONE", "NULL", "N/A"]:
                    continue

                market_info = check_market_status(ticker)
                if market_info["status"] == "CLOSED":
                    logger.info(f"Market for {ticker} is CLOSED ({market_info['market']}). Skipping alert.")
                    continue

                if is_in_cooldown(ticker):
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
    logger.info("Starting scan_watchlist_technical (MA20 Pullback Strategy)...")
    
    for ticker in WATCHLIST:
        try:
            market_info = check_market_status(ticker)
            if market_info["status"] == "CLOSED":
                continue

            if is_in_cooldown(ticker):
                continue

            t = yf.Ticker(ticker)
            df = t.history(period="100d", interval="1d")
            if df.empty or len(df) < 50:
                continue

            df.ta.sma(length=20, append=True)
            df.ta.sma(length=40, append=True)
            df.ta.sma(length=50, append=True)

            sma20 = df["SMA_20"].iloc[-1]
            sma40 = df["SMA_40"].iloc[-1] if "SMA_40" in df.columns else None
            sma50 = df["SMA_50"].iloc[-1] if "SMA_50" in df.columns else None

            if pd.isna(sma20):
                continue

            current_price = df["Close"].iloc[-1]
            high_3d = df["High"].tail(3).max()
            low_curr = df["Low"].iloc[-1]
            open_curr = df["Open"].iloc[-1]

            # 1. General Uptrend Check
            trend_ok = False
            if sma40 and sma20 > sma40:
                trend_ok = True
            elif sma50 and sma20 > sma50:
                trend_ok = True

            sma20_slope_ok = df["SMA_20"].iloc[-1] > df["SMA_20"].iloc[-5]
            if not (trend_ok and sma20_slope_ok):
                continue

            # 2. Distance from MA20 (0% to 1.5% pullback)
            if high_3d < sma20:
                continue

            dist_from_ma20 = (current_price - sma20) / sma20
            if dist_from_ma20 < 0.0 or dist_from_ma20 > 0.015:
                continue

            # 3. Confirmation Candle Pattern (Hammer or Strong Green Rebound)
            body = abs(current_price - open_curr)
            lower_shadow = min(open_curr, current_price) - low_curr
            
            is_hammer = (lower_shadow >= 2 * body) and (low_curr <= sma20 * 1.005)
            is_strong_green = (current_price > open_curr) and (current_price >= df["High"].iloc[-1] - (0.3 * (df["High"].iloc[-1] - low_curr)))

            if not (is_hammer or is_strong_green):
                continue

            # 4. Strict Risk Parameters
            stop_loss = current_price * 0.97
            take_profit = current_price * 1.06

            if stop_loss >= sma20:
                continue

            p_data = check_liquidity_and_price(ticker, custom_sl=stop_loss, custom_tp1=take_profit)
            if not p_data.get("valid"):
                continue

            pattern_desc = "נר פטיש על ממוצע 20" if is_hammer else "נר ירוק חזק מגיב לממוצע 20"
            analysis_text = f"המניה בכיקוב חיובי מעל ממוצע 20. נצפתה תבנית {pattern_desc} באזור התמיכה של MA20."
            recom_text = f"כניסה מומלצת באזור {current_price:.2f} עם SL ב-3% מתחת לממוצע 20."

            send_telegram_alert(
                ticker=ticker,
                alert_type="TECHNICAL_MA20_PULLBACK",
                analysis=analysis_text,
                recommendation=recom_text,
                intensity="HIGH",
                price_data=p_data,
                market_info=market_info
            )
        except Exception as e:
            logger.error(f"Error scanning technical {ticker}: {e}")

# ---------------------------------------------------------------------------
# 7. Telegram Interactive Callbacks & Commands
# ---------------------------------------------------------------------------
if bot:
    @bot.message_handler(commands=['start'])
    def cmd_start(message):
        bot.reply_to(message, "🤖 PazPSTrading Bot פעיל ומוכן למתן איתותי מסחר (ארה\"ב ותל אביב)!")

    @bot.message_handler(commands=['test_news'])
    def cmd_test_news(message):
        bot.reply_to(message, "🔍 מריץ סריקת חדשות ידנית...")
        scan_breaking_news_events(is_test=True)
        bot.reply_to(message, "✅ סריקת חדשות הושלמה.")

    @bot.message_handler(commands=['test_tech'])
    def cmd_test_tech(message):
        bot.reply_to(message, "🔍 מריץ סריקת Watchlist טכנית...")
        scan_watchlist_technical(is_test=True)
        bot.reply_to(message, "✅ סריקת Watchlist הושלמה.")

    @bot.message_handler(commands=['tech'])
    def cmd_tech(message):
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "אנא שלח טיקר, לדוגמה: `/tech NVDA` או `/tech TEVA.TA`", parse_mode="Markdown")
            return
        ticker = parts[1].upper()
        bot.reply_to(message, f"🔍 מריץ ניתוח טכני יזום על {ticker}...")
        m_info = check_market_status(ticker)
        p_data = check_liquidity_and_price(ticker)
        if not p_data.get("valid"):
            bot.reply_to(message, f"❌ שגיאה בשליפת {ticker}: {p_data.get('reason')}")
            return
        send_telegram_alert(ticker=ticker, alert_type="TECHNICAL_MANUAL", analysis=f"ניתוח יזום עבור {ticker}.", recommendation="מעקב / ניהול סיכונים", intensity="MEDIUM", price_data=p_data, market_info=m_info)

    @bot.callback_query_handler(func=lambda call: call.data.startswith("calc_"))
    def callback_calc(call):
        try:
            _, ticker, price, sl, tp1 = call.data.split("_")
            user_calc_state[call.from_user.id] = {"ticker": ticker, "price": float(price), "sl": float(sl), "tp1": float(tp1)}
            currency_symbol = "₪" if ticker.endswith(".TA") else "$"
            msg = (
                f"📊 <b>חישוב גודל פוזיציה עבור {ticker}</b>\n\n"
                f"מחיר כניסה: {currency_symbol}{float(price):.2f}\n"
                f"סטופ לוס: {currency_symbol}{float(sl):.2f}\n\n"
                f"אנא שלח/י בהודעה את סכום הסיכון המקסימלי שברצונך לסכן ברמת העסקה (ב-{currency_symbol}):"
            )
            bot.send_message(call.message.chat.id, msg, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Callback error: {e}")

    @bot.message_handler(func=lambda m: m.from_user.id in user_calc_state)
    def process_calc_input(message):
        uid = message.from_user.id
        state = user_calc_state.get(uid)
        if not state:
            return

        try:
            risk_amount = float(message.text.strip())
            price = state["price"]
            sl = state["sl"]
            ticker = state["ticker"]
            currency_symbol = "₪" if ticker.endswith(".TA") else "$"

            risk_per_share = price - sl
            if risk_per_share <= 0:
                bot.reply_to(message, "❌ שגיאה בחישוב: מחיר הסטופ גבוה ממחיר הכניסה.")
                del user_calc_state[uid]
                return

            shares = int(risk_amount / risk_per_share)
            total_position = shares * price
            expected_profit = shares * (state["tp1"] - price)

            res_msg = (
                f"📐 <b>תוצאת חישוב סיכון עבור {ticker}:</b>\n\n"
                f"• סכום סיכון מבוקש: <b>{currency_symbol}{risk_amount:,.2f}</b>\n"
                f"• כמות מניות לרכישה: <b>{shares:,}</b>\n"
                f"• שווי פוזיציה כולל: <b>{currency_symbol}{total_position:,.2f}</b>\n"
                f"• רווח צפוי ביעד (TP1): <b>{currency_symbol}{expected_profit:,.2f}</b>\n"
            )
            bot.reply_to(message, res_msg, parse_mode="HTML")
        except ValueError:
            bot.reply_to(message, "❌ אנא שלח מספר תקין בלבד (לדוגמה: 500).")
        finally:
            if uid in user_calc_state:
                del user_calc_state[uid]

# ---------------------------------------------------------------------------
# 8. Webhook & Flask Server Setup
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return "PazPSTrading Bot is running alive!", 200

@app.route(f"/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def webhook():
    if not bot:
        return "Bot not configured", 500
    json_str = request.get_data().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "OK", 200

def setup_webhook():
    if bot and RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_BOT_TOKEN}"
        try:
            bot.remove_webhook()
            time.sleep(1)
            bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook set successfully to {webhook_url}")
        except Exception as e:
            logger.error(f"Failed to set Webhook: {e}")

# Scheduler setup
scheduler = BackgroundScheduler(timezone="Asia/Jerusalem")
scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
scheduler.add_job(scan_watchlist_technical, 'interval', minutes=30)
scheduler.start()

if __name__ == "__main__":
    setup_webhook()
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
