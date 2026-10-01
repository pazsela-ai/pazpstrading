import os
import logging
import time
import json
import re
import requests
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser
from google import genai
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request
from telebot import TeleBot, types
from apscheduler.schedulers.background import BackgroundScheduler

# הגדרת לוגים
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# הגדרות משתני סביבה
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

bot = TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# אתחול מנוע ה-AI
ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini Client: {e}")

# מסד נתונים בזיכרון
simulated_trades = []
last_scans = {"news": "טרם בוצעה", "tech": "טרם בוצעה"}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

# ---------------------------------------------------------
# 1. WEBHOOK & HEALTH CHECK
# ---------------------------------------------------------

@app.route('/')
@app.route('/health')
def home():
    return "OK - Market Scanner Bot Active!", 200

@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def telegram_webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return 'OK', 200
    return 'Forbidden', 403

def setup_webhook():
    if RENDER_EXTERNAL_URL and TELEGRAM_TOKEN:
        webhook_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_TOKEN}"
        try:
            bot.remove_webhook()
            time.sleep(1)
            bot.set_webhook(url=webhook_url)
            logging.info(f"Webhook configured to: {webhook_url}")
        except Exception as e:
            logging.error(f"Webhook setup failed: {e}")

# ---------------------------------------------------------
# 2. TICKER FETCHERS (S&P 500, NASDAQ 100, TA-125)
# ---------------------------------------------------------

def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        tables = pd.read_html(resp.text)
        return [str(t).replace('.', '-') for t in tables[0]['Symbol'].tolist()]
    except Exception as e:
        logging.error(f"Error fetching S&P 500: {e}")
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM"]

def get_nasdaq100_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        tables = pd.read_html(resp.text)
        for t in tables:
            for col in ['Ticker', 'Symbol']:
                if col in t.columns:
                    return [str(x).replace('.', '-').strip() for x in t[col].dropna().tolist() if len(str(x)) <= 5]
        raise ValueError("Nasdaq table format error")
    except Exception as e:
        logging.error(f"Error fetching Nasdaq: {e}")
        return ["QQQ", "AVGO", "COST", "QCOM", "ADBE", "PANW", "NFLX"]

def get_ta125_tickers():
    try:
        url = "https://he.wikipedia.org/wiki/%D0%A0%D7%A9%D7%99%D7%9E%D7%AA_%D7%97%D7%91%D7%A8%D7%95%D7%AA_%D7%91%D7%9E%D7%93%D7%93_%D7%AA%22%D7%90-125"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        tables = pd.read_html(resp.text)
        df = tables[0]
        for col in df.columns:
            if any(k in str(col) for k in ['סימול', 'Ticker', 'סמל']):
                return [f"{str(t).strip().upper()}.TA" for t in df[col].dropna().tolist() if str(t).strip()]
        raise ValueError("TA125 table format error")
    except Exception as e:
        logging.error(f"Error fetching TA-125: {e}")
        return ["ELAL.TA", "TEVA.TA", "ICL.TA", "NICE.TA", "LUMI.TA", "POLI.TA", "ESLT.TA", "DSRG.TA"]

def get_all_market_tickers():
    all_tickers = list(set(get_sp500_tickers() + get_nasdaq100_tickers() + get_ta125_tickers()))
    return all_tickers

# ---------------------------------------------------------
# 3. ENHANCED TECHNICAL ANALYSIS ENGINE
# ---------------------------------------------------------

def analyze_technical(ticker):
    try:
        # שימוש ב-yfinance עם מניעת חסימות
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo", interval="1d")
        
        if df.empty or len(df) < 20:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # חישוב אינדיקטורים
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Vol_SMA20'] = ta.sma(df['Volume'], length=20)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else current_price * 0.03
        
        # תנאים מורחבים לאיתור פריצה
        above_ema = current_price > latest['EMA20']
        rsi_bullish = 50 <= latest['RSI'] <= 70 and latest['RSI'] > prev['RSI']
        volume_support = latest['Volume'] >= (prev['Volume'] * 0.9)

        is_breakout = above_ema and rsi_bullish and volume_support

        tp = round(current_price + (atr_val * 2.0), 2)
        sl = round(current_price - (atr_val * 1.2), 2)

        return {
            "ticker": ticker,
            "is_breakout": is_breakout,
            "price": round(current_price, 2),
            "tp": tp,
            "sl": sl,
            "rsi": round(latest['RSI'], 1)
        }
    except Exception as e:
        return None

def scan_single_ticker_task(ticker):
    tech_res = analyze_technical(ticker)
    if tech_res and tech_res["is_breakout"]:
        send_alert(ticker=ticker, trigger_type="TECHNICAL", tech_data=tech_res)

def scan_technical_market():
    logging.info("Running technical scan across full universe...")
    last_scans["tech"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tickers = get_all_market_tickers()
    
    # הרצה מקבילית מבוקרת (12 Workers כדי למנוע חסימת Rate Limit מה-API)
    with ThreadPoolExecutor(max_workers=12) as executor:
        executor.map(scan_single_ticker_task, tickers)

# ---------------------------------------------------------
# 4. ENHANCED NEWS & AI ENGINE
# ---------------------------------------------------------

def analyze_broad_news_with_ai(news_batch_text):
    """ניתוח מקבץ כתבות בבת אחת על ידי Gemini AI להפחתת עומס API"""
    if not ai_client:
        return None

    prompt = f"""
    אתה אנליסט פיננסי בכיר. להלן מקבץ כותרות ותקצירים מחדשות הכלכלה והבורסה:
    {news_batch_text}

    עבור כל ידיעה בעלת השפעה מסחרית חיובית ברורה על מניה ספציפית שנמצאת במדדים (S&P500, NASDAQ, ת"א 125):
    החזר תשובת JSON בלבד, כרשימה של אובייקטים:
    [
      {{
        "ticker": "סימול המניה בלבד (למשל NVDA, TEVA.TA, ELAL.TA)",
        "reason": "נימוק קצר וחד בעברית במשפט אחד"
      }}
    ]
    אם אין ידיעות חזקות ורלוונטיות, החזר רשימה ריקה [].
    """

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        if response and response.text:
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
    except Exception as e:
        logging.error(f"Error in Gemini Batch News scan: {e}")
    return None

def scan_news_feed():
    logging.info("Running automated news scan...")
    last_scans["news"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    rss_urls = [
        "https://news.google.com/rss/search?q=בורסה+מניות+דוחות+חוזה+תעופה&hl=he&gl=IL&ceid=IL:he",
        "https://news.google.com/rss/search?q=stock+market+breakout+earnings+contract+acquisition&hl=en-US&gl=US&ceid=US:en"
    ]

    collected_entries = []
    for url in rss_urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:6]:
                clean_summary = re.sub('<[^<]+?>', '', entry.get("summary", ""))
                collected_entries.append(f"כותרת: {entry.title}\nתקציר: {clean_summary}\n---")
        except Exception as e:
            logging.error(f"Error reading RSS from {url}: {e}")

    if not collected_entries:
        return False

    batch_text = "\n".join(collected_entries)
    ai_results = analyze_broad_news_with_ai(batch_text)

    found = False
    if ai_results and isinstance(ai_results, list):
        for item in ai_results:
            ticker = item.get("ticker")
            reason = item.get("reason")
            if ticker and reason:
                send_alert(ticker=ticker, trigger_type="NEWS", news_reason=reason)
                found = True
    return found

def analyze_ticker_specific_news(ticker):
    if not ai_client:
        return "❌ מנוע ה-AI אינו מחובר (חסר GEMINI_API_KEY)."

    clean_ticker = ticker.replace(".TA", "")
    is_ta = ".TA" in ticker
    
    query = f"{clean_ticker}+בורסה" if is_ta else f"{clean_ticker}+stock+news"
    lang = "hl=he&gl=IL&ceid=IL:he" if is_ta else "hl=en-US&gl=US&ceid=US:en"
    rss_url = f"https://news.google.com/rss/search?q={query}&{lang}"
    
    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=10)
        feed = feedparser.parse(resp.content)
        
        items = []
        for entry in feed.entries[:4]:
            summary = re.sub('<[^<]+?>', '', entry.get('summary', ''))
            items.append(f"• {entry.title}: {summary}")
            
        if not items:
            return f"ℹ️ לא נמצאו כתבות חדשות עדכניות עבור `{ticker}`."

        news_content = "\n".join(items)
        prompt = f"""
        נתח את הידיעות החדשותיות עבור המניה {ticker}:
        {news_content}

        החזר תשובה בעברית:
        1. סנטימנט (חיובי / ניטרלי / שלילי) והסבר ב-2 משפטים.
        2. שורת מסקנה קצרה למסחר.
        """
        
        response = ai_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return response.text.strip() if response and response.text else "❌ לא התקבלה תשובה מ-AI."
    except Exception as e:
        return f"❌ שגיאה בניתוח חדשות עבור `{ticker}`: {e}"

# ---------------------------------------------------------
# 5. TELEGRAM ALERTS & COMMANDS
# ---------------------------------------------------------

def get_tradingview_link(ticker):
    clean_ticker = ticker.replace(".TA", "")
    exchange = "TASE" if ".TA" in ticker else "NASDAQ"
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{clean_ticker}"

def send_alert(ticker, trigger_type="TECHNICAL", news_reason="", tech_data=None, target_chat_id=None):
    dest_id = target_chat_id or CHAT_ID
    if not dest_id:
        return

    if not tech_data:
        tech_data = analyze_technical(ticker)

    price = tech_data["price"] if tech_data else "N/A"
    tp = tech_data["tp"] if tech_data else "N/A"
    sl = tech_data["sl"] if tech_data else "N/A"
    rsi = tech_data.get("rsi", "N/A") if tech_data else "N/A"
    tv_link = get_tradingview_link(ticker)

    if trigger_type == "NEWS":
        header = f"🚨 **התראת קטליזטור חדשותי - {ticker}**"
        body = f"💡 **נימוק AI:**\n{news_reason}\n"
    elif trigger_type == "TECHNICAL":
        header = f"📊 **התראת פריצה טכנית - {ticker}**"
        body = f"📈 **ניתוח טכני:** פריצת מומנטום מעל EMA20 (RSI: {rsi}).\n"
    else:
        header = f"🔥 **התראה משולבת - {ticker}**"
        body = f"💡 **נימוק:** {news_reason}\n📈 **ניתוח טכני:** פריצת מומנטום חיובית.\n"

    currency = "₪" if ".TA" in ticker else "$"
    msg = f"{header}\n\n{body}\n" \
          f"🎯 **תכנית עבודה מומלצת:**\n" \
          f"• מחיר כניסה: {currency}{price}\n" \
          f"• יעד רווח (TP): {currency}{tp}\n" \
          f"• סטופ לוס (SL): {currency}{sl}\n"

    keyboard = types.InlineKeyboardMarkup()
    sim_btn = types.InlineKeyboardButton("📈 בצע סימולציית קנייה", callback_data=f"sim_{ticker}_{price}_{tp}_{sl}")
    tv_btn = types.InlineKeyboardButton("📊 פתח גרף ב-TradingView", url=tv_link)
    keyboard.add(sim_btn, tv_btn)

    bot.send_message(dest_id, msg, parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "🟢 **סורק המדדים PazPSTrading מחובר ופעיל!**\n\n"
        "המערכת סורקת באופן אוטומטי את כל המניות ב-S&P 500, Nasdaq 100 ות\"א 125.\n\n"
        "🛠️ **פקודות זמינות:**\n"
        "• `/status` - סטטוס סריקות וכמות מניות\n"
        "• `/test_news` - הרצת סריקת חדשות AI בזמן אמת\n"
        "• `/test_tech` - הרצת סריקה טכנית מלאה\n"
        "• `/tech <TICKER>` - ניתוח טכני ממוקד למניה\n"
        "• `/news_scan <TICKER>` - ניתוח חדשות ממוקד למניה\n"
        "• `/portfolio` - ניהול תיק סימולציות"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def handle_status(message):
    total_tickers = len(get_all_market_tickers())
    status_msg = (
        "⚙ **סטטוס מערכת:**\n\n"
        f"• מנוע AI (Gemini): {'✅ פעיל' if ai_client else '❌ לא מחובר'}\n"
        f"• סריקת חדשות אחרונה: `{last_scans['news']}`\n"
        f"• סריקה טכנית אחרונה: `{last_scans['tech']}`\n"
        f"• סך מניות בסריקה: `{total_tickers}`\n"
        f"• עסקאות בסימולטור: `{len(simulated_trades)}`"
    )
    bot.reply_to(message, status_msg, parse_mode="Markdown")

@bot.message_handler(commands=['test_news'])
def handle_test_news(message):
    bot.reply_to(message, "🔎 מריץ סריקת חדשות AI על המדדים...")
    found = scan_news_feed()
    if not found:
        bot.send_message(message.chat.id, "ℹ️ לא אותרו אירועים חדשותיים חריגים ברגע זה.")

@bot.message_handler(commands=['test_tech'])
def handle_test_tech(message):
    bot.reply_to(message, "🔎 מריץ סריקה טכנית מקבילית על כלל המניות...")
    scan_technical_market()
    bot.send_message(message.chat.id, "✅ הסריקה הטכנית הושלמה.")

@bot.message_handler(commands=['tech'])
def handle_tech_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "יש לציין סימול מניה. לדוגמה: `/tech NVDA` או `/tech ELAL.TA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    bot.reply_to(message, f"📊 מנתח אינדיקטורים טכניים עבור `{ticker}`...", parse_mode="Markdown")
    
    tech_data = analyze_technical(ticker)
    if tech_data:
        send_alert(ticker=ticker, trigger_type="TECHNICAL", tech_data=tech_data, target_chat_id=message.chat.id)
    else:
        bot.send_message(message.chat.id, f"❌ לא ניתן היה לשלוף נתונים טכניים עבור `{ticker}`.")

@bot.message_handler(commands=['news_scan'])
def handle_news_scan_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "יש לציין סימול מניה. לדוגמה: `/news_scan NVDA`", parse_mode="Markdown")
        return
    
    ticker = parts[1].upper()
    bot.reply_to(message, f"🔎 מריץ סריקה וניתוח AI עבור `{ticker}`...", parse_mode="Markdown")
    
    analysis_result = analyze_ticker_specific_news(ticker)
    tech_data = analyze_technical(ticker)
    
    price = tech_data["price"] if tech_data else "N/A"
    tp = tech_data["tp"] if tech_data else "N/A"
    sl = tech_data["sl"] if tech_data else "N/A"
    currency = "₪" if ".TA" in ticker else "$"

    response_msg = (
        f"📰 **ניתוח חדשות וסנטימנט AI עבור {ticker}:**\n\n"
        f"{analysis_result}\n\n"
        f"🎯 **מחירי עבודה נוכחיים:**\n"
        f"• מחיר כניסה: {currency}{price}\n"
        f"• יעד (TP): {currency}{tp}\n"
        f"• סטופ (SL): {currency}{sl}"
    )
    
    tv_link = get_tradingview_link(ticker)
    keyboard = types.InlineKeyboardMarkup()
    sim_btn = types.InlineKeyboardButton("📈 בצע סימולציית קנייה", callback_data=f"sim_{ticker}_{price}_{tp}_{sl}")
    tv_btn = types.InlineKeyboardButton("📊 פתח גרף ב-TradingView", url=tv_link)
    keyboard.add(sim_btn, tv_btn)

    bot.send_message(message.chat.id, response_msg, parse_mode="Markdown", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data.startswith('sim_'))
def handle_simulation_callback(call):
    _, ticker, price, tp, sl = call.data.split('_')
    simulated_trades.append({
        "ticker": ticker,
        "entry_price": price,
        "tp": tp,
        "sl": sl,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    bot.answer_callback_query(call.id, text=f"✅ עסקה וירטואלית על {ticker} נרשמה!")
    bot.send_message(call.message.chat.id, f"📝 **עסקה נרשמה בסימולטור!**\nמניה: `{ticker}`\nכניסה: `{price}` | TP: `{tp}` | SL: `{sl}`", parse_mode="Markdown")

@bot.message_handler(commands=['portfolio'])
def handle_portfolio(message):
    if not simulated_trades:
        bot.reply_to(message, "אין עסקאות פעילות כרגע בסימולטור.")
        return
    text = "💼 **תיק סימולציות פעיל:**\n\n"
    for idx, trade in enumerate(simulated_trades, 1):
        text += f"{idx}. **{trade['ticker']}** | כניסה: {trade['entry_price']} | TP: {trade['tp']} | SL: {trade['sl']}\n"
    bot.reply_to(message, text, parse_mode="Markdown")

# ---------------------------------------------------------
# 6. SCHEDULER & STARTUP
# ---------------------------------------------------------

scheduler = BackgroundScheduler()
scheduler.add_job(scan_technical_market, 'interval', minutes=30)
scheduler.add_job(scan_news_feed, 'interval', minutes=15)
scheduler.start()

setup_webhook()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
