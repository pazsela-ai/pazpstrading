import os
import logging
import threading
import time
import re
import feedparser
import json
import requests
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from flask import Flask
from telebot import TeleBot, types
from apscheduler.schedulers.background import BackgroundScheduler
from google import genai

# הגדרת לוגים
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# הגדרות משתני סביבה
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# אתחול לקוח AI (Gemini)
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# מסד נתונים בזיכרון
simulated_trades = []
last_scans = {"news": "טרם בוצעה", "tech": "טרם בוצעה"}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}

# ---------------------------------------------------------
# 1. טעינה דינמית מוגנת: S&P 500 + NASDAQ 100 + ת"א 125
# ---------------------------------------------------------

def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = requests.get(url, headers=HEADERS)
        tables = pd.read_html(req.text)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        return [str(t).replace('.', '-') for t in tickers]
    except Exception as e:
        logging.error(f"Error fetching S&P 500 tickers: {e}")
        return ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]

def get_nasdaq100_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        req = requests.get(url, headers=HEADERS)
        tables = pd.read_html(req.text)
        df = None
        for t in tables:
            if 'Ticker' in t.columns or 'Symbol' in t.columns:
                df = t
                break
        if df is not None:
            col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
            tickers = df[col].dropna().tolist()
            return [str(t).replace('.', '-').strip() for t in tickers]
        raise ValueError("Nasdaq 100 table not found")
    except Exception as e:
        logging.error(f"Error fetching Nasdaq 100 tickers: {e}")
        return ["QQQ", "AMD", "AVGO", "COST", "NFLX", "INTC", "QCOM", "TXN", "ADBE", "PANW"]

def get_ta125_tickers():
    try:
        url = "https://he.wikipedia.org/wiki/%D0%A0%D7%A9%D7%99%D7%9E%D7%AA_%D7%97%D7%91%D7%A8%D7%95%D7%AA_%D7%91%D7%9E%D7%93%D7%93_%D7%AA%22%D7%90-125"
        req = requests.get(url, headers=HEADERS)
        tables = pd.read_html(req.text)
        df = tables[0]
        ticker_col = None
        for col in df.columns:
            if 'סימול' in str(col) or 'Ticker' in str(col) or 'סמל' in str(col):
                ticker_col = col
                break
        if ticker_col:
            raw_tickers = df[ticker_col].dropna().tolist()
            return [f"{str(t).strip().upper()}.TA" for t in raw_tickers if str(t).strip()]
        raise ValueError("TA125 table parsing failed")
    except Exception as e:
        logging.error(f"Error fetching TA-125 tickers: {e}")
        return ["ELAL.TA", "TEVA.TA", "ICL.TA", "NICE.TA", "LUMI.TA", "POLI.TA", "ESLT.TA"]

def get_all_market_tickers():
    sp500 = get_sp500_tickers()
    nasdaq100 = get_nasdaq100_tickers()
    ta125 = get_ta125_tickers()
    all_tickers = list(set(sp500 + nasdaq100 + ta125))
    logging.info(f"Total unique tickers in universe: {len(all_tickers)}")
    return all_tickers

# ---------------------------------------------------------
# 2. מנוע ניתוח טכני מעודכן ומוגן
# ---------------------------------------------------------

def analyze_technical(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="60d", interval="1d")
        
        if df.empty or len(df) < 20:
            logging.warning(f"No data returned for {ticker}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else current_price * 0.03

        is_breakout = (current_price > latest['EMA20']) and (latest['RSI'] > 50) and (latest['RSI'] > prev['RSI'])

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
        logging.error(f"Error analyzing technical for {ticker}: {e}")
        return None

def scan_single_ticker_task(ticker):
    tech_res = analyze_technical(ticker)
    if tech_res and tech_res["is_breakout"]:
        send_alert(ticker=ticker, trigger_type="TECHNICAL", tech_data=tech_res)

def scan_technical_market():
    logging.info("Starting automated technical scan across full universe...")
    last_scans["tech"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tickers = get_all_market_tickers()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(scan_single_ticker_task, tickers)

    logging.info("Automated technical scan completed.")

# ---------------------------------------------------------
# 3. מנוע ניתוח חדשותי (אוטומטי + ממוקד)
# ---------------------------------------------------------

def analyze_broad_news_with_ai(headline, summary):
    if not ai_client:
        return None

    prompt = f"""
    אתה אנליסט בכיר. נתח את הידיעה:
    כותרת: {headline}
    תקציר: {summary}

    אם יש השפעה חיובית מובהקת על מניה במדדים המובילים (ת"א 125, S&P 500, Nasdaq 100), החזר JSON בלבד:
    {{"is_relevant": true, "ticker": "ELAL.TA", "reason": "נימוק...", "conviction": "HIGH"}}
    אחרת החזר: {{"is_relevant": false}}
    """
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        res_text = response.text.strip()
        res_text = re.sub(r'^```json\s*', '', res_text)
        res_text = re.sub(r'^```\s*', '', res_text)
        res_text = re.sub(r'\s*```$', '', res_text)
        return json.loads(res_text)
    except Exception as e:
        logging.error(f"Error in AI news reasoning: {e}")
        return None

def scan_news_feed():
    logging.info("Starting automated news scan...")
    last_scans["news"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    rss_urls = [
        "[https://news.google.com/rss/search?q=ישראל+תעופה+ביטחון+כלכלה+בורסה&hl=he&gl=IL&ceid=IL:he](https://news.google.com/rss/search?q=ישראל+תעופה+ביטחון+כלכלה+בורסה&hl=he&gl=IL&ceid=IL:he)",
        "[https://news.google.com/rss/search?q=stock+market+earnings+acquisition+defense+contracts&hl=en-US&gl=US&ceid=US:en](https://news.google.com/rss/search?q=stock+market+earnings+acquisition+defense+contracts&hl=en-US&gl=US&ceid=US:en)"
    ]

    found_any = False
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:8]:
            ai_res = analyze_broad_news_with_ai(entry.title, entry.get("summary", ""))
            if ai_res and ai_res.get("is_relevant"):
                ticker = ai_res["ticker"]
                reason = ai_res["reason"]
                send_alert(ticker=ticker, trigger_type="NEWS", news_reason=reason)
                found_any = True
    return found_any

def fetch_ticker_news_rss(ticker):
    clean_ticker = ticker.replace(".TA", "")
    rss_url = f"[https://news.google.com/rss/search?q=](https://news.google.com/rss/search?q=){clean_ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries[:5]:
        news_items.append(f"- כותרת: {entry.title}\n  תקציר: {entry.get('summary', '')}")
    return "\n".join(news_items)

def analyze_ticker_specific_news(ticker):
    if not ai_client:
        return "❌ מנוע ה-AI אינו מחובר (חסר GEMINI_API_KEY)."

    news_text = fetch_ticker_news_rss(ticker)
    if not news_text:
        return f"ℹ️ לא נמצאו חדשות עדכניות עבור `{ticker}`."

    prompt = f"""
    אתה אנליסט בכיר. להלן חדשות עבור המניה {ticker}:
    {news_text}

    1. נתח את הסנטימנט (חיובי/שלילי/ניטרלי) וההשפעה על המניה ב-2-3 משפטים בעברית.
    2. תן שורת סיכום ברורה: [חיובי / ניטרלי / שלילי].
    """
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error analyzing news for {ticker}: {e}")
        return f"❌ אירעה שגיאה בניתוח ה-AI עבור `{ticker}`."

# ---------------------------------------------------------
# 4. התראות ופקודות טלגרם
# ---------------------------------------------------------

def get_tradingview_link(ticker):
    clean_ticker = ticker.replace(".TA", "")
    exchange = "TASE" if ".TA" in ticker else "NASDAQ"
    return f"[https://www.tradingview.com/chart/?symbol=](https://www.tradingview.com/chart/?symbol=){exchange}:{clean_ticker}"

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
        body = f"💡 **נימוק והסקה אנליטית:**\n{news_reason}\n"
    elif trigger_type == "TECHNICAL":
        header = f"📊 **התראת פריצה טכנית - {ticker}**"
        body = f"📈 **ניתוח טכני:** פריצת מומנטום בגרף יומי (RSI: {rsi}).\n"
    else:
        header = f"🔥 **התראה משולבת: חדשות + טכני - {ticker}**"
        body = f"💡 **נימוק אנליטי:** {news_reason}\n📈 **ניתוח טכני:** פריצת מומנטום מעל ממוצעים נעים.\n"

    currency = "₪" if ".TA" in ticker else "$"
    msg = f"{header}\n\n{body}\n" \
          f"🎯 **מחירי עבודה מומלצים:**\n" \
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
        "🟢 **אפליקציית PazPSTrading פעילה ועובדת!**\n\n"
        "הבוט מנטר ברקע ובאופן אוטומטי לחלוטין את כל המניות במדדי S&P 500, Nasdaq 100 ות\"א 125.\n\n"
        "🛠️ **פקודות לבקרה ידנית:**\n"
        "• `/status` - סטטוס וכמות המניות שבמעקב\n"
        "• `/test_news` - הרצת סורק חדשות AI\n"
        "• `/test_tech` - הרצת סריקה טכנית מלאה\n"
        "• `/tech <TICKER>` - ניתוח טכני ממוקד למניה\n"
        "• `/news_scan <TICKER>` - ניתוח חדשות ממוקד למניה\n"
        "• `/portfolio` - צפייה בתיק סימולציות"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def handle_status(message):
    total_tickers = len(get_all_market_tickers())
    status_msg = (
        "⚙ **סטטוס מערכת:**\n\n"
        f"• חיבור ל-AI: {'✅ תקין' if ai_client else '❌ לא מחובר'}\n"
        f"• סריקת חדשות אחרונה: `{last_scans['news']}`\n"
        f"• סריקה טכנית אחרונה: `{last_scans['tech']}`\n"
        f"• מניות במעקב דינמי: `{total_tickers}`\n"
        f"• עסקאות בסימולטור: `{len(simulated_trades)}`"
    )
    bot.reply_to(message, status_msg, parse_mode="Markdown")

@bot.message_handler(commands=['test_news'])
def handle_test_news(message):
    bot.reply_to(message, "🔎 מריץ סורק חדשות מבוסס AI בלייב...")
    found = scan_news_feed()
    if not found:
        bot.send_message(message.chat.id, "ℹ️ לא נמצאו כרגע אירועים חדשותיים בעלי השפעה חיובית מובהקת.")

@bot.message_handler(commands=['test_tech'])
def handle_test_tech(message):
    bot.reply_to(message, "🔎 מריץ סריקה טכנית במקביל על **כל המניות** במדדים...")
    scan_technical_market()
    bot.send_message(message.chat.id, "✅ הסריקה הטכנית הושלמה.")

@bot.message_handler(commands=['tech'])
def handle_tech_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "יש לציין סימול מניה. לדוגמה: `/tech NVDA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    bot.reply_to(message, f"📊 מנתח אינדיקטורים טכניים עבור `{ticker}`...", parse_mode="Markdown")
    
    tech_data = analyze_technical(ticker)
    if tech_data:
        send_alert(ticker=ticker, trigger_type="TECHNICAL", tech_data=tech_data, target_chat_id=message.chat.id)
    else:
        bot.send_message(message.chat.id, f"❌ לא ניתן היה לשלוף נתונים טכניים עבור `{ticker}`. ודא שהסימול תקין (למשל NVDA או TEVA.TA).")

@bot.message_handler(commands=['news_scan'])
def handle_news_scan_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "יש לציין סימול מניה. לדוגמה: `/news_scan NVDA`", parse_mode="Markdown")
        return
    
    ticker = parts[1].upper()
    bot.reply_to(message, f"🔎 סורק חדשות ומריץ ניתוח AI עבור `{ticker}`...", parse_mode="Markdown")
    
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
# 5. Flask & Background Scheduler
# ---------------------------------------------------------

@app.route('/')
def home():
    return "PazPSTrading Bot - Broad Multi-Index Scanner Active!"

is_tasks_started = False

def start_background_tasks():
    global is_tasks_started
    if is_tasks_started:
        return
    is_tasks_started = True

    scheduler = BackgroundScheduler()
    # סריקות אוטומטיות ברקע
    scheduler.add_job(scan_technical_market, 'interval', minutes=30)
    scheduler.add_job(scan_news_feed, 'interval', minutes=15)
    scheduler.start()

    def run_bot():
        try:
            # מחיקת Webhook מוחלטת ואיפוס עדכונים ישנים שהצטברו
            bot.remove_webhook(drop_pending_updates=True)
            time.sleep(2)
            logging.info("Starting Telegram Bot Polling...")
            bot.infinity_polling(timeout=20, long_polling_timeout=10, skip_pending=True)
        except Exception as e:
            logging.error(f"Error running bot polling: {e}")

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

if __name__ == '__main__':
    start_background_tasks()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
else:
    start_background_tasks()
