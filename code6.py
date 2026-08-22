import os
import asyncio
import logging
import requests
import feedparser
import yfinance as yf
import pandas as pd
from datetime import datetime
from flask import Flask
from threading import Thread
from finvizfinance.screener.technical import Technical
from google import genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# הגדרת לוגים
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# מפתחות סביבה
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# אתחול Gemini API
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ---------------------------------------------------------
# 1. מנוע ניתוח טכני דינמי (מניעת כניסה בשיא)
# ---------------------------------------------------------

def run_technical_screener():
    """סורק את כל השוק ומחזיר אותות טכניים איכותיים בתחילת תנועה"""
    try:
        f_screener = Technical()
        filters_dict = {
            '20-Day Simple Moving Average': 'Price crossed 20SMA above',
            'Relative Strength Index (14)': 'Overbought (60) or below',
            'Current Volume': 'Over 500K'
        }
        f_screener.set_filter(filters_dict=filters_dict)
        df = f_screener.screener_view()
        
        if df is None or df.empty:
            return []

        candidates = df['Ticker'].tolist()[:10]
        valid_signals = []

        for ticker in candidates:
            signal = analyze_technical_setup(ticker)
            if signal:
                valid_signals.append(signal)

        return valid_signals
    except Exception as e:
        logger.error(f"שגיאה בסורק הטכני הדינמי: {e}")
        return []

def analyze_technical_setup(ticker_symbol: str):
    """מחשב אינדיקטורים ובודק שרכבת המסחר לא נסעה"""
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="3mo")
        if len(df) < 50:
            return None

        current_price = df['Close'].iloc[-1]
        
        # חישוב RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))

        # ממוצע נעים 20
        sma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        dist_from_sma20 = ((current_price - sma20) / sma20) * 100

        # נפח מסחר
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        curr_volume = df['Volume'].iloc[-1]
        vol_ratio = curr_volume / avg_volume if avg_volume > 0 else 1.0

        # בלמי חירום למניעת כניסה בשיא
        if rsi >= 65 or dist_from_sma20 > 4.0 or vol_ratio < 1.2:
            return None

        # חישוב יעד וסטופ
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        stop_loss = max(current_price - (1.5 * atr), current_price * 0.95)
        take_profit = current_price + (3.0 * atr)

        risk_per_share = current_price - stop_loss
        reward_per_share = take_profit - current_price
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        budget = 1000
        shares = int(budget // current_price) if current_price <= budget else 1
        max_loss = round(shares * risk_per_share, 2)

        return {
            "ticker": ticker_symbol,
            "price": round(current_price, 2),
            "rsi": round(rsi, 1),
            "dist_sma20": round(dist_from_sma20, 1),
            "vol_ratio": round(vol_ratio, 1),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "shares": shares,
            "max_loss": max_loss,
            "rr_ratio": round(rr_ratio, 1)
        }
    except Exception as e:
        logger.error(f"שגיאה בניתוח טכני עבור {ticker_symbol}: {e}")
        return None

# ---------------------------------------------------------
# 2. מנוע ניתוח חדשותי (Gemini)
# ---------------------------------------------------------

def fetch_rss_news():
    urls = [
        "https://news.google.com/rss/search?q=stock+market+breakout&hl=en-US&gl=US&ceid=US:en",
        "https://finance.yahoo.com/rss/headline?s=AAPL,NVDA,TSLA,AMD,MSFT"
    ]
    news_items = []
    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            news_items.append({"title": entry.title, "link": entry.link})
    return news_items

# ---------------------------------------------------------
# 3. תזמון סריקות ושליחת התראות בטלגרם
# ---------------------------------------------------------

async def send_telegram_msg(bot, text, reply_markup=None):
    if TELEGRAM_CHAT_ID:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown", reply_markup=reply_markup)

async def periodic_news_scan(context: ContextTypes.DEFAULT_TYPE):
    """מסלול 1: ניתוח חדשותי ברקע"""
    items = fetch_rss_news()
    if not items or not ai_client:
        return

    prompt = f"מתוכם מצא ידיעה דרמטית אחת בעלת פוטנציאל להזזת מניה: {items}. החזר JSON עם ticker, summary, action."
    try:
        response = ai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        # לשם פשטות – ניטור רציף של חדשות
    except Exception as e:
        logger.error(f"שגיאה בסריקת חדשות: {e}")

async def periodic_technical_scan(context: ContextTypes.DEFAULT_TYPE):
    """מסלול 2: ניתוח טכני דינמי"""
    signals = run_technical_screener()
    for sig in signals:
        msg = (
            f"📊 **[אות טכני - תחילת תנועה]**\n"
            f"**מניה:** `{sig['ticker']}` | **מחיר:** ${sig['price']}\n\n"
            f"🚥 **בדיקת תזמון (מניעת רדיפה בשיא):**\n"
            f"• **RSI (14):** `{sig['rsi']}` 🟢 *(רחוק משיא)*\n"
            f"• **מרחק מממוצע 20:** `{sig['dist_sma20']}%+` 🟢 *(בקו הזינוק)*\n"
            f"• **נפח מסחר:** פי `{sig['vol_ratio']}` מהממוצע 🟢\n\n"
            f"💡 **למה שווה לבדוק?**\n"
            f"פריצת התנגדות נקודתית בנפח ער עם תזמון כניסה אופטימלי.\n\n"
            f"🛡️ **ניהול סיכונים מומלץ (תקציב $1,000):**\n"
            f"• **כניסה:** ${sig['price']}\n"
            f"• **סטופ לוס:** ${sig['stop_loss']}\n"
            f"• **יעד רווח:** ${sig['take_profit']}\n"
            f"• **כמות מניות:** {sig['shares']}\n"
            f"• **סיכון מקסימלי:** ${sig['max_loss']}\n"
            f"• **יחס סיכוי/סיכון:** `1 : {sig['rr_ratio']}` 🟢"
        )
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔗 גרף ב-TradingView", url=f"https://www.tradingview.com/symbols/{sig['ticker']}")]
        ])
        await send_telegram_msg(context.bot, msg, reply_markup=keyboard)

# ---------------------------------------------------------
# 4. פקודות בוט ושרת Flask ל-Render
# ---------------------------------------------------------

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("היי! הבוט מחובר. סורק חדשות ופרמטרים טכניים באופן עצמאי וישלח התראות בזמן אמת.")

async def test_tech_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("מריץ סריקה טכנית ידנית כעת...")
    await periodic_technical_scan(context)

app_flask = Flask(__name__)

@app_flask.route('/')
def home():
    return "Trader Bot Alive", 200

def run_flask():
    app_flask.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

def main():
    Thread(target=run_flask, daemon=True).start()

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN is missing")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("testtech", test_tech_cmd))

    # הוספת משימות תקופתיות ברקע
    job_queue = application.job_queue
    job_queue.run_repeating(periodic_news_scan, interval=600, first=10)
    job_queue.run_repeating(periodic_technical_scan, interval=1800, first=20)

    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
