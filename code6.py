import os
import logging
import time
import requests
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser
import threading
from google import genai
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request
from telebot import TeleBot, types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
RENDER_EXTERNAL_URL = (os.getenv("RENDER_EXTERNAL_URL") or "").strip()

bot = TeleBot(TELEGRAM_TOKEN, threaded=False)
app = Flask(__name__)

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini Client: {e}")

simulated_trades = []
last_scans = {"news": "טרם בוצעה", "tech": "טרם בוצעה"}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

# ---------------------------------------------------------
# 1. WEBHOOK & BOT COMMANDS SETUP
# ---------------------------------------------------------

def setup_bot_commands():
    """עדכון תפריט הפקודות בטלגרם"""
    try:
        commands = [
            types.BotCommand("start", "הפעלת הבוט ותפריט ראשי"),
            types.BotCommand("status", "בדיקת סטטוס מערכת וחיבורים"),
            types.BotCommand("test_tech", "הרצת סריקה טכנית בלייב"),
            types.BotCommand("test_news", "בדיקת תקינות מנוע חדשות AI"),
            types.BotCommand("tech", "ניתוח טכני למניה/קרן (למשל /tech QQQ)"),
            types.BotCommand("news_scan", "ניתוח חדשות למניה (למשל /news_scan NVDA)")
        ]
        bot.set_my_commands(commands)
    except Exception as e:
        logging.error(f"Failed to update bot commands: {e}")

@app.route('/')
@app.route('/health')
def home():
    return "OK - Market Scanner Bot Active!", 200

@app.route('/init_webhook', methods=['GET', 'POST'])
def init_webhook():
    if not TELEGRAM_TOKEN or not RENDER_EXTERNAL_URL:
        return "Missing variables", 400
    url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_TOKEN}"
    try:
        bot.remove_webhook()
        time.sleep(1)
        res = bot.set_webhook(url=url)
        setup_bot_commands()
        return f"Success! Webhook set to {url}" if res else "Telegram rejected webhook", 200
    except Exception as e:
        return f"Error: {e}", 500

@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def telegram_webhook():
    if request.headers.get('content-type') == 'application/json':
        try:
            json_string = request.get_data().decode('utf-8')
            update = types.Update.de_json(json_string)
            bot.process_new_updates([update])
            return 'OK', 200
        except Exception as e:
            logging.error(f"Webhook error: {e}")
            return 'OK', 200
    return 'Forbidden', 403

# ---------------------------------------------------------
# 2. EXPANDED TICKER LIST (STOCKS + ETFS + TASE)
# ---------------------------------------------------------

STATIC_TICKERS = [
    # --- Major ETFs (קרנות סל מובילות) ---
    "QQQ", "SPY", "IWM", "TQQQ", "SQQQ", "SOXX", "SMH", "XLK", "XLF", "XLE",
    "XLV", "XLY", "XLP", "XLI", "XLU", "ARKK", "ARKG", "BITO", "GLD", "SLV",

    # --- US Tech & Mega Cap ---
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "NFLX",
    "MS", "JPM", "BAC", "V", "MA", "UNH", "PG", "HD", "DIS", "PYPL", "COST", "CSCO",
    "ORCL", "CRM", "PEP", "KO", "XOM", "CVX", "NKE", "LLY", "AVGO", "QCOM", "TXN",
    "AMAT", "MU", "LRCX", "PANW", "SNOW", "PLTR", "UBER", "ABNB", "COIN", "MARA",
    "SQ", "SHOP", "ROKU", "SNAP", "PINS", "SE", "MELI", "BKNG", "SBUX", "MCD",
    "BABA", "BIDU", "PDD", "NIO", "XPEV", "RIVN", "LCID", "SOFI", "HOOD", "UPST",

    # --- TASE / Israel (תל אביב) ---
    "ELAL.TA", "TEVA.TA", "ICL.TA", "NICE.TA", "LUMI.TA", "POLI.TA", "ESLT.TA",
    "DSCT.TA", "FIBI.TA", "AZRG.TA", "MVRN.TA", "DELTG.TA", "ENLT.TA", "ORA.TA",
    "HARL.TA", "CLIS.TA", "PHOE.TA", "SAEN.TA", "SPEN.TA", "ARGO.TA", "BEZQ.TA",
    "AMOT.TA", "BIG.TA", "ALRO.TA", "FORTY.TA", "CAMT.TA", "TSEM.TA", "MLSR.TA"
]

def get_all_market_tickers():
    tickers = set(STATIC_TICKERS)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=HEADERS, timeout=4)
        if resp.status_code == 200:
            tables = pd.read_html(resp.text)
            wiki_sp = [str(t).replace('.', '-') for t in tables[0]['Symbol'].tolist()]
            tickers.update(wiki_sp)
    except Exception as e:
        logging.warning(f"Wikipedia fetch skipped: {e}")
    return list(tickers)

# ---------------------------------------------------------
# 3. TECHNICAL ANALYSIS ENGINE (WITH BUY RECOMMENDATION)
# ---------------------------------------------------------

def analyze_technical(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="60d", interval="1d")
        if df.empty or len(df) < 20:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else current_price * 0.03
        rsi_val = float(latest['RSI'])
        ema20 = float(latest['EMA20'])
        ema50 = float(latest['EMA50'])

        reasons = []
        if current_price > ema20:
            reasons.append(f"מחיר (${current_price:.2f}) מעל ממוצע נע EMA20 (${ema20:.2f})")
        if ema20 > ema50:
            reasons.append("מגמה עולה: EMA20 נמצא מעל EMA50")
        if 50 <= rsi_val <= 70:
            reasons.append(f"מומנטום חיובי בריא: RSI ברמה של {rsi_val:.1f}")

        is_breakout = (current_price > ema20) and (rsi_val >= 50)
        
        # המלצה חד משמעית
        recommendation = "🟢 **מומלץ לכניסה (איתות פריצה)**" if is_breakout else "🔴 **לא מומלץ לכניסה כעת**"

        return {
            "ticker": ticker,
            "is_breakout": is_breakout,
            "recommendation": recommendation,
            "price": round(current_price, 2),
            "tp": round(current_price + (atr_val * 2.0), 2),
            "sl": round(current_price - (atr_val * 1.2), 2),
            "rsi": round(rsi_val, 1),
            "reasons": reasons
        }
    except Exception as e:
        logging.error(f"Error analyzing {ticker}: {e}")
        return None

# ---------------------------------------------------------
# 4. AI NEWS ANALYSIS ENGINE (GEMINI API)
# ---------------------------------------------------------

def analyze_ticker_specific_news(ticker):
    if not ai_client:
        return "❌ מנוע ה-AI אינו מחובר (חסר GEMINI_API_KEY במשתני הסביבה)."

    clean_ticker = ticker.replace(".TA", "")
    query = f"{clean_ticker}+בורסה" if ".TA" in ticker else f"{clean_ticker}+stock"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=he&gl=IL&ceid=IL:he"
    
    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=8)
        feed = feedparser.parse(resp.content)
        items = [f"• {e.title}" for e in feed.entries[:5]]
        
        if not items:
            return f"ℹ️ לא נמצאו כתבות חדשות עדכניות ברשת עבור `{ticker}`."

        prompt = f"נתח בקצרה בעברית את הידיעות החדשותיות הבאות עבור מניית/קרן {ticker} ותן סיכום קצר של הסנטימנט (חיובי/שלילי/ניטרלי):\n" + "\n".join(items)
        
        # מודלי Gemini עדכניים
        candidate_models = ['gemini-2.5-flash', 'gemini-1.5-flash']
        
        for model_name in candidate_models:
            try:
                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    return f"📰 **סיכום חדשות AI עבור {ticker}:**\n\n{response.text.strip()}"
            except Exception as model_err:
                logging.warning(f"Model {model_name} failed: {model_err}")
                continue

        return "❌ לא התקבל מענה משרתי גוגל AI (בדוק את תקינות ה-API Key)."
    except Exception as e:
        return f"❌ שגיאה בשליפת החדשות: {e}"

# ---------------------------------------------------------
# 5. SCANNER TASKS
# ---------------------------------------------------------

def scan_single_ticker_task(ticker):
    tech_res = analyze_technical(ticker)
    if tech_res and tech_res["is_breakout"]:
        send_alert(ticker=ticker, tech_data=tech_res)

def _async_technical_scan(chat_id=None):
    last_scans["tech"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tickers = get_all_market_tickers()
    if chat_id:
        bot.send_message(chat_id, f"🔎 מתחיל סריקה טכנית בלייב על `{len(tickers)}` מניות וקרנות סל...", parse_mode="Markdown")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(scan_single_ticker_task, tickers)
        
    if chat_id:
        bot.send_message(chat_id, "✅ הסריקה הטכנית הושלמה!")

# ---------------------------------------------------------
# 6. TELEGRAM BOT HANDLERS
# ---------------------------------------------------------

def get_tradingview_link(ticker):
    clean_ticker = ticker.replace(".TA", "")
    exchange = "TASE" if ".TA" in ticker else "NASDAQ"
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{clean_ticker}"

def send_alert(ticker, tech_data=None, target_chat_id=None):
    dest_id = target_chat_id or CHAT_ID
    if not dest_id:
        return

    if not tech_data:
        tech_data = analyze_technical(ticker)

    if not tech_data:
        bot.send_message(dest_id, f"❌ לא ניתן לשלוף נתונים עבור `{ticker}`.")
        return

    price = tech_data["price"]
    tp = tech_data["tp"]
    sl = tech_data["sl"]
    rec = tech_data["recommendation"]
    currency = "₪" if ".TA" in ticker else "$"

    reasons_text = "\n".join([f"  • {r}" for r in tech_data["reasons"]]) if tech_data["reasons"] else "  • אין כרגע איתות פריצה משמעותי"

    msg = (
        f"📊 **ניתוח טכני - {ticker}**\n\n"
        f"📣 **המלצה:** {rec}\n\n"
        f"💡 **נימוקי הניתוח:**\n"
        f"{reasons_text}\n\n"
        f"🎯 **תכנית עבודה מוצעת:**\n"
        f"• מחיר נוכחי: {currency}{price}\n"
        f"• יעד רווח (TP): {currency}{tp}\n"
        f"• סטופ לוס (SL): {currency}{sl}\n"
    )

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(
        types.InlineKeyboardButton("📈 סימולציית קנייה", callback_data=f"sim_{ticker}_{price}_{tp}_{sl}"),
        types.InlineKeyboardButton("📊 TradingView", url=get_tradingview_link(ticker))
    )

    bot.send_message(dest_id, msg, parse_mode="Markdown", reply_markup=keyboard)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "🟢 **הבוט PazPSTrading מחובר ופעיל!**\n\n"
        "פקודות זמינות בתפריט:\n"
        "• `/status` - בדיקת סטטוס מערכת\n"
        "• `/test_tech` - הרצת סריקה טכנית בלייב\n"
        "• `/test_news` - בדיקת תקינות מנוע AI\n"
        "• `/tech <TICKER>` - ניתוח טכני למניה/קרן\n"
        "• `/news_scan <TICKER>` - ניתוח חדשות למניה"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def handle_status(message):
    total_tickers = len(get_all_market_tickers())
    status_msg = (
        "⚙ **סטטוס מערכת:**\n\n"
        f"• AI Engine: {'✅ פעיל' if ai_client else '❌ חסר מפתח'}\n"
        f"• מעקב דינמי (מניות + ETFs): `{total_tickers}`\n"
        f"• סריקה טכנית אחרונה: `{last_scans['tech']}`\n"
        f"• עסקאות בסימולטור: `{len(simulated_trades)}`"
    )
    bot.reply_to(message, status_msg, parse_mode="Markdown")

@bot.message_handler(commands=['test_tech'])
def handle_test_tech(message):
    threading.Thread(target=_async_technical_scan, args=(message.chat.id,), daemon=True).start()

@bot.message_handler(commands=['test_news'])
def handle_test_news(message):
    bot.reply_to(message, "📰 מריץ בדיקת חדשות AI עבור NVDA...")
    res = analyze_ticker_specific_news("NVDA")
    bot.send_message(message.chat.id, res, parse_mode="Markdown")

@bot.message_handler(commands=['tech'])
def handle_tech_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "ציין סימול מניה/קרן. לדוגמה: `/tech QQQ`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    tech_data = analyze_technical(ticker)
    send_alert(ticker=ticker, tech_data=tech_data, target_chat_id=message.chat.id)

@bot.message_handler(commands=['news_scan'])
def handle_news_scan_manual(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "ציין סימול מניה. לדוגמה: `/news_scan NVDA`", parse_mode="Markdown")
        return
    ticker = parts[1].upper()
    res = analyze_ticker_specific_news(ticker)
    bot.reply_to(message, res, parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('sim_'))
def handle_simulation_callback(call):
    _, ticker, price, tp, sl = call.data.split('_')
    simulated_trades.append({"ticker": ticker, "price": price, "tp": tp, "sl": sl})
    bot.answer_callback_query(call.id, text=f"✅ עסקה על {ticker} נרשמה בסימולטור!")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
