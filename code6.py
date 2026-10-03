import os
import time
import json
import logging
import datetime
import pytz
import feedparser
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler

# --- Google Gemini SDK ---
import google.generativeai as genai

# --- Groq & OpenAI SDKs ---
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ==========================================
# 1. הגדרות לוגים ומשתני סביבה
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '').strip()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '').strip()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()

RENDER_EXTERNAL_URL = os.getenv('RENDER_EXTERNAL_URL', '').strip()

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)

# ==========================================
# 2. שמירת מצב בזיכרון קבוע (Persistent Storage)
# ==========================================
DATA_FILE = "bot_state.json"

def load_state():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"שגיאה בטעינת קובץ מצב: {e}")
    return {"processed_news": [], "cooldowns": {}}

def save_state(state):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"שגיאה בשמירת קובץ מצב: {e}")

state = load_state()
processed_news_ids = set(state.get("processed_news", []))
sent_ticker_cooldowns = state.get("cooldowns", {})

def update_cooldown(ticker):
    sent_ticker_cooldowns[ticker] = time.time()
    state["cooldowns"] = sent_ticker_cooldowns
    save_state(state)

def is_in_cooldown(ticker, hours=12):
    last_time = sent_ticker_cooldowns.get(ticker, 0)
    return (time.time() - last_time) < (hours * 3600)

def mark_news_processed(news_id):
    processed_news_ids.add(news_id)
    # שמירת 500 ידיעות אחרונות בזיכרון הדיסק למניעת כפילויות
    state["processed_news"] = list(processed_news_ids)[-500:]
    save_state(state)

# ==========================================
# 3. הגדרות רשימות ומקורות חדשות רוחביים
# ==========================================

# רשימת מעקב בלעדית לניתוח הטכני הסדיר
WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "PLTR",
    "MARA", "RIOT", "COIN", "BA", "LMT", "NOC", "RTX", "PFE", "MRNA", "LLY",
    "TEVA.TA", "NICE.TA", "LUMI.TA", "DSSL.TA", "ICL.TA"
]

# פידים רוחביים וכלל-נושאיים (כיסוי מלא: עסקים, מבזקים, גיאופוליטיקה ואירועים גלובליים)
BROAD_NEWS_FEEDS = [
    # --- Google News: פידים כלליים ורוחביים (ראשיים) ---
    "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",  # Google News - כלכלה ועסקים עולמיים (ראשי)
    "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",     # Google News - חדשות עולם וגיאופוליטיקה
    "https://news.google.com/rss?hl=he&gl=IL&ceid=IL:he",                                      # Google News - כותרות ראשיות בישראל (עברית)
    "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=he&gl=IL&ceid=IL:he",    # Google News - כלכלה ועסקים בישראל

    # --- ערוצי כלכלה ומבזקים מובילים בישראל ---
    "https://www.calcalist.co.il/Integration/StoryRss1854.xml",  # כלכליסט - שוק ההון ומבזקים
    "https://www.globes.co.il/news/rss/rssfeed.aspx?folderid=585", # גלובס - שוק ההון
    "https://www.bizportal.co.il/rss/flash",                       # ביזפורטל - מבזקים בזמן אמת

    # --- הודעות לעיתונות רשמיות בזמן אמת (כלל החברות הציבוריות) ---
    "https://www.prnewswire.com/rss/news-releases-list.rss",     # PR Newswire - הודעות כלליות
    "https://feed.businesswire.com/rss/home/?rss=G1QFDFWBXkZeGV1XWA==", # BusinessWire - הודעות כלליות

    # --- סוכנויות ידיעות וכלכלה בינלאומיות ---
    "https://feeds.bbci.co.uk/news/world/rss.xml",                # BBC World News
    "https://feeds.content.dowjones.io/public/rss/mw_topstories"  # MarketWatch Top Stories
]

# ==========================================
# 4. בדיקת שעות מסחר מותאמת
# ==========================================
def check_market_status(ticker):
    is_israel = ticker.endswith('.TA')
    tz = pytz.timezone('Asia/Jerusalem')
    now = datetime.datetime.now(tz)
    weekday = now.weekday() # 0=Monday ... 6=Sunday
    current_time = now.time()

    if is_israel:
        # ישראל: ראשון (6) עד חמישי (3)
        if weekday in [4, 5]: # שישי, שבת
            return False, "הבורסה בת\"א סגורה (סוף שבוע)"
        open_time = datetime.time(9, 50)
        close_time = datetime.time(17, 15) if weekday == 6 else datetime.time(17, 25)
        if open_time <= current_time <= close_time:
            return True, "המסחר בת\"א פעיל כעת"
        return False, "הבורסה בת\"א סגורה כעת"
    else:
        # ארה"ב: שני (0) עד שישי (4)
        if weekday in [5, 6]: # שבת, ראשון
            return False, "הבורסה בארה\"ב סגורה (סוף שבוע)"
        
        reg_open = datetime.time(16, 30)
        reg_close = datetime.time(23, 0)
        
        if reg_open <= current_time <= reg_close:
            return True, "המסחר בארה\"ב פעיל (שעות רגילות)"
        else:
            return False, "הבורסה בארה\"ב סגורה כעת"

# ==========================================
# 5. שליפת מחירי Real-Time ונתונים טכניים
# ==========================================
def fetch_realtime_data(ticker):
    try:
        t = yf.Ticker(ticker)
        # שליפת מחירי 1m לקבלת Real-Time מדויק
        df_min = t.history(period="1d", interval="1m")
        if not df_min.empty:
            current_price = float(df_min['Close'].iloc[-1])
        else:
            df_day = t.history(period="5d")
            if df_day.empty: return None
            current_price = float(df_day['Close'].iloc[-1])

        df = t.history(period="100d")
        if len(df) < 50:
            return None

        # חישוב אינדיקטורים טכניים
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Vol_SMA20'] = ta.sma(df['Volume'], length=20)

        latest = df.iloc[-1]
        avg_vol = latest['Vol_SMA20']
        dollar_volume = current_price * avg_vol if avg_vol else 0

        return {
            "price": current_price,
            "ema20": float(latest['EMA20']),
            "ema50": float(latest['EMA50']),
            "rsi": float(latest['RSI']),
            "atr": float(latest['ATR']),
            "volume": float(latest['Volume']),
            "avg_volume": float(avg_vol),
            "dollar_volume": dollar_volume
        }
    except Exception as e:
        logging.error(f"שגיאה בשליפת נתונים עבור {ticker}: {e}")
        return None

# ==========================================
# 6. מנוע AI כפול עם מנגנון גיבוי (Failover)
# ==========================================
def query_ai_engine(prompt):
    # 1. Google Gemini
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
            res = model.generate_content(prompt)
            if res.text: return res.text
        except Exception as e:
            logging.warning(f"Gemini נכשל, עובר לגיבוי: {e}")

    # 2. Groq
    if GROQ_API_KEY and Groq:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content
        except Exception as e:
            logging.warning(f"Groq נכשל, עובר לגיבוי: {e}")

    # 3. OpenAI
    if OPENAI_API_KEY and OpenAI:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI נכשל: {e}")

    return None

# ==========================================
# 7. סורק חדשות ואירועים (Event-Driven & Catalyst Engine)
# ==========================================
def scan_breaking_news_events():
    logging.info("מתחיל סריקת חדשות ואירועים מתפרצים...")
    for feed_url in BROAD_NEWS_FEEDS:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries[:5]:
            news_id = entry.get('id', entry.get('link'))
            
            if news_id in processed_news_ids:
                continue

            title = entry.title
            summary = entry.get('summary', '')

            # Prompt מתקדם להסקה רוחבית, קטליזטורים עתידיים והיפותטיים (תומך עברית ואנגלית)
            prompt = f"""
אתה אנליסט פיננסי בכיר ואסטרטג מסחר בורסאי המתמחה בזיהוי השפעות רוחביות והשלכות מסדר שני (Second-Order Effects) וקטליזטורים עתידיים (Catalyst Events).

נתח את הידיעה החדשותית הבאה (העשויה להיות בעברית או באנגלית):
כותרת: {title}
תקציר: {summary}

מטרת העל: לזהות אילו מניות (בארה"ב או בישראל בבורסת תל אביב) עשויות להרוויח דרמטית מהאירוע, כולל תרחישים היפותטיים, צופי פני עתיד, ניסויים קליניים, הסכמים, או תקלות/אירועים המשפיעים לטובה על מתחרים.

דוגמאות לחשיבה אנליטית:
1. אירוע/תקרית בחברת תעופה (למשל תקלה או כמעט התרסקות) -> הסקה על מעבר נוסעים ועלייה בביקוש למניות תעופה מתחרות או מקומיות (למשל: ESRX.TA, DAL, AAL).
2. ניסוי קליני, תכנית לניסוי או פיתוח חדשני בתרופה -> פוטנציאל זינוק חזק לחברה המפתחת (למשל: MRNA, PFE, TEVA.TA).
3. הסלמה ביטחונית / איומים -> עלייה בביקוש למערכות הגנה אווירת, רחפנים וסייבר (למשל: LMT, NOC, DSSL.TA).

הוראות ביצוע:
- בצע הסקה הגיונית וחופשית (אינך מוגבל לרשימה מסוימת).
- אם המניה נסחרת בתל אביב, הצג את הטיקר בפורמט yfinance תקין (למשל: ESRX.TA, LUMI.TA, TEVA.TA).
- אם מצאת מניה ספציפית בעלת פוטנציאל רווח בעקבות הידיעה, החזר JSON בלבד:
{{
  "relevant": true,
  "ticker": "הטיקר המדויק שחולץ (לדוגמה: MRNA, LMT, DAL, ESRX.TA וכו')",
  "impact": "HIGH/MEDIUM",
  "analysis": "הסבר אנליטי קצר וחד בעברית על השרשרת הסיבתית: מה קרה באירוע ואיך הוא מוביל לפוטנציאל הרווח במניה המומלצת",
  "reason": "נימוק קצר בעברית לתכנית העבודה והזרז (Catalyst) המוביל"
}}

אם הידיעה כללית לחלוטין ואין ממנה שום שרשרת סיבתית למניה ספציפית, החזר:
{{"relevant": false}}
"""
            res_text = query_ai_engine(prompt)
            if not res_text:
                continue

            try:
                clean_text = res_text.replace("```json", "").replace("```", "").strip()
                data_json = json.loads(clean_text)

                if data_json.get("relevant"):
                    ticker = data_json.get("ticker").upper()
                    
                    if is_in_cooldown(ticker, hours=8):
                        mark_news_processed(news_id)
                        continue

                    # שליפת נתוני אמת לכל מניה שחולצה
                    market_data = fetch_realtime_data(ticker)
                    if not market_data:
                        mark_news_processed(news_id)
                        continue

                    price = market_data['price']
                    atr = market_data['atr']
                    is_active, market_desc = check_market_status(ticker)

                    sl = round(price - (1.2 * atr), 2)
                    tp1 = round(price + (2.0 * atr), 2)
                    tp2 = round(price + (4.0 * atr), 2)

                    msg = (
                        f"📰 *ניתוח אירוע חדשותי ותכנית מסחר: {ticker}*\n"
                        f"⏰ מצב שוק: {('🟢' if is_active else '🔴')} {market_desc}\n"
                        f"🔥 עוצמת אירוע: *{data_json.get('impact')}*\n\n"
                        f"🔗 *אסמכתאות וידיעות רלוונטיות:*\n• [{title}]({entry.link})\n\n"
                        f"💡 *ניתוח המשמעות למניה:*\n{data_json.get('analysis')}\n\n"
                        f"📋 *תכנית עבודה יעודית:*\n"
                        f"• המלצה: כניסה\n"
                        f"• נימוק: {data_json.get('reason')}\n\n"
                        f"🎯 *פרמטרי פוזיציה מוצעים (R:R דינמי):*\n"
                        f"• מחיר נוכחי: ${price:.2f}\n"
                        f"• כניסה מומלצת (Limit): ${price:.2f}\n"
                        f"• יעד רווח 1 (TP1): ${tp1:.2f}\n"
                        f"• יעד רווח 2 מורחב (TP2): ${tp2:.2f}\n"
                        f"• סטופ לוס (SL): ${sl:.2f}\n"
                    )

                    markup = InlineKeyboardMarkup()
                    markup.add(
                        InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"calc_{ticker}_{price}_{sl}_{tp1}"),
                        InlineKeyboardButton("📈 TradingView", url=f"https://www.tradingview.com/symbols/{ticker.replace('.TA', '')}")
                    )

                    bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown", reply_markup=markup)
                    update_cooldown(ticker)
                    mark_news_processed(news_id)

            except Exception as e:
                logging.error(f"שגיאה בפענוח JSON מ-AI: {e}")
                mark_news_processed(news_id)

# ==========================================
# 8. סורק ניתוח טכני סדיר (Watchlist בלבד)
# ==========================================
def scan_watchlist_technical():
    logging.info("מתחיל סריקה טכנית יזומה...")
    for ticker in WATCHLIST:
        is_active, market_desc = check_market_status(ticker)
        
        # לא מפיקים איתותים טכניים כשהשוק סגור
        if not is_active:
            continue

        if is_in_cooldown(ticker, hours=12):
            continue

        data = fetch_realtime_data(ticker)
        if not data or data['dollar_volume'] < 500000:
            continue

        price = data['price']
        ema20 = data['ema20']
        ema50 = data['ema50']
        rsi = data['rsi']
        vol = data['volume']
        avg_vol = data['avg_volume']
        atr = data['atr']

        # בדיקת קרבה לממוצע (0% עד 2.5% מעל EMA20)
        pct_from_ema20 = (price - ema20) / ema20
        is_near_ema20 = 0 <= pct_from_ema20 <= 0.025
        
        is_uptrend = price > ema20 and ema20 > ema50
        is_rsi_healthy = 50 <= rsi <= 68
        is_volume_high = vol >= (avg_vol * 1.2)

        score = 0
        reasons = []

        if is_uptrend:
            score += 30
            reasons.append("מגמה עולה (מחיר מעל EMA20 ו-EMA50)")
        if is_near_ema20:
            score += 35
            reasons.append("מחיר קרוב ונתמך על EMA20 (נקודת כניסה אידיאלית)")
        if is_rsi_healthy:
            score += 20
            reasons.append(f"מומנטום בריא (RSI: {rsi:.1f})")
        if is_volume_high:
            score += 15
            reasons.append("נפח מסחר מוגבר מעל הממוצע")

        # ציון מחמיר (75+) בלבד
        if score >= 75:
            entry_price = price
            sl = round(entry_price - (1.2 * atr), 2)
            tp1 = round(entry_price + (2.0 * atr), 2)

            msg = (
                f"📊 *איתות טכני אוטומטי: {ticker}*\n"
                f"⏰ מצב שוק: 🟢 {market_desc}\n"
                f"📈 ציון טכני: *{score}/100*\n\n"
                f"📣 *המלצה:* איתות חיובי לכניסה\n\n"
                f"💡 *אינדיקטורים שנלכדו:*\n" +
                "\n".join([f"• {r}" for r in reasons]) +
                f"\n\n🎯 *תכנית עבודה מוצעת:*\n"
                f"• מחיר נוכחי: ${price:.2f}\n"
                f"• מחיר כניסה (Limit): ${entry_price:.2f}\n"
                f"• יעד רווח (TP1): ${tp1:.2f}\n"
                f"• סטופ לוס (SL): ${sl:.2f}\n"
            )

            markup = InlineKeyboardMarkup()
            markup.add(
                InlineKeyboardButton("🎯 בצע עסקה / חישוב סיכון", callback_data=f"calc_{ticker}_{entry_price}_{sl}_{tp1}"),
                InlineKeyboardButton("📈 TradingView", url=f"https://www.tradingview.com/symbols/{ticker.replace('.TA', '')}")
            )

            bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown", reply_markup=markup)
            update_cooldown(ticker)

# ==========================================
# 9. מחשבון סיכון אינטראקטיבי בטלגרם
# ==========================================
user_calc_state = {}

@bot.callback_query_handler(func=lambda call: call.data.startswith('calc_'))
def handle_calc_callback(call):
    _, ticker, entry, sl, tp1 = call.data.split('_')
    user_calc_state[call.from_user.id] = {
        'ticker': ticker,
        'entry': float(entry),
        'sl': float(sl),
        'tp1': float(tp1)
    }
    
    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton("💵 דולר ($)", callback_data="curr_USD"),
        InlineKeyboardButton("₪ שקל (ILS)", callback_data="curr_ILS")
    )
    bot.send_message(call.message.chat.id, "בחר את מטבע התיק שלך לחישוב הסיכון:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith('curr_'))
def handle_currency_choice(call):
    currency = call.data.split('_')[1]
    user_calc_state[call.from_user.id]['currency'] = currency
    msg = bot.send_message(call.message.chat.id, f"הקלד את סכום הסיכון הכספי שברצונך לסכן בעסקה זו ({'$' if currency=='USD' else '₪'}):")
    bot.register_next_step_handler(msg, process_risk_amount)

def process_risk_amount(message):
    try:
        risk_amount = float(message.text.strip())
        state = user_calc_state.get(message.from_user.id)
        if not state:
            bot.send_message(message.chat.id, "פג תוקף החישוב, נסה ללחוץ שוב על המקש.")
            return

        entry = state['entry']
        sl = state['sl']
        tp1 = state['tp1']
        risk_per_share = abs(entry - sl)

        if risk_per_share == 0:
            bot.send_message(message.chat.id, "שגיאה בחישוב המרחק לסטופ לוס.")
            return

        shares = int(risk_amount / risk_per_share)
        total_position = shares * entry
        potential_profit = shares * abs(tp1 - entry)
        curr_symbol = "$" if state.get('currency') == 'USD' else "₪"

        reply = (
            f"🧮 *תוצאות חישוב גודל פוזיציה עבור {state['ticker']}*\n\n"
            f"• כמות מניות לקנייה: *{shares} מניות*\n"
            f"• שווי פוזיציה כולל: *{curr_symbol}{total_position:,.2f}*\n"
            f"• סיכון מקסימלי בעסקה: *{curr_symbol}{risk_amount:,.2f}*\n"
            f"• רווח פוטנציאלי ביעד (TP1): *{curr_symbol}{potential_profit:,.2f}*\n"
        )
        bot.send_message(message.chat.id, reply, parse_mode="Markdown")
    except ValueError:
        bot.send_message(message.chat.id, "נא להזין מספר תקין בלבד.")

# ==========================================
# 10. תזמון משימות (APScheduler) ו-Flask Webhook
# ==========================================
scheduler = BackgroundScheduler(timezone="Asia/Jerusalem")

# סורק טכני - כל 60 דקות, סורק חדשות - כל 15 דקות
scheduler.add_job(scan_watchlist_technical, 'interval', minutes=60)
scheduler.add_job(scan_breaking_news_events, 'interval', minutes=15)
scheduler.start()

@app.route('/', methods=['GET'])
def index():
    return "PazPSTrading Bot is running!", 200

@app.route(f'/{TELEGRAM_BOT_TOKEN}', methods=['POST'])
def webhook():
    json_string = request.get_data().decode('utf-8')
    update = telebot.types.Update.de_json(json_string)
    bot.process_new_updates([update])
    return "OK", 200

if __name__ == '__main__':
    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/{TELEGRAM_BOT_TOKEN}"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        logging.info(f"Webhook set to {webhook_url}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
