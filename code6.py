import os
import logging
import asyncio
import requests
import feedparser
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from finvizfinance.quote import finvizfinance
import finnhub
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import google.generativeai as genai

# הגדרת הלוגים
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# טעינת מפתחות סביבה
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
CHAT_ID = os.getenv("CHAT_ID")

# הגדרת Gemini 2.5
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# הגדרת Finnhub
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY) if FINNHUB_API_KEY else None

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return "Autonomous Trading Alert System (US & IL) is Live!"

# ----------------------------------------------------
# מנועי איסוף נתונים (תמיכה במניות ישראליות וגלובליות)
# ----------------------------------------------------

def fetch_technical_signals(symbol: str) -> str:
    """ניתוח טכני מעמיק מותאם למניות ישראליות ואמריקאיות"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")
        if df.empty:
            return f"אין נתונים עבור {symbol}"
        
        # חישוב אינדיקטורים באמצעות pandas-ta
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        
        latest = df.iloc[-1]
        close_price = latest.get('Close', 0)
        sma20 = latest.get('SMA_20', 0)
        sma50 = latest.get('SMA_50', 0)
        rsi = latest.get('RSI_14', 0)
        
        # Finviz מעודכן (מבוסס מחלקה) עבור מניות הרשומות בארה"ב בלבד
        finviz_info = ""
        if not symbol.endswith(".TA"):
            try:
                stock = finvizfinance(symbol)
                fv_data = stock.ticker_fundament()
                volume = fv_data.get('Volume', 'N/A')
                rel_vol = fv_data.get('Rel Volume', 'N/A')
                finviz_info = f"Finviz Volume/Rel Vol: {volume} | {rel_vol}"
            except Exception as e:
                logger.warning(f"Finviz snapshot skipped for {symbol}: {e}")

        currency = "אג' / ש\"ח" if symbol.endswith(".TA") else "$"

        return (
            f"מחיר: {close_price:.2f} {currency}\n"
            f"SMA20: {sma20:.2f} | SMA50: {sma50:.2f}\n"
            f"RSI (14): {rsi:.2f}\n"
            f"{finviz_info}"
        )
    except Exception as e:
        logger.error(f"Technical error for {symbol}: {e}")
        return f"שגיאה בחישוב טכני: {e}"

def fetch_live_news(symbol: str) -> str:
    """איסוף חדשות מ-Finnhub ו-RSS (מנקה את סיומת .TA במידת הצורך)"""
    news_list = []
    clean_symbol = symbol.replace(".TA", "")
    
    # 1. Finnhub
    if finnhub_client:
        try:
            res = finnhub_client.company_news(clean_symbol, _from="2026-08-01", to="2026-08-22")
            for item in res[:3]:
                news_list.append(f"- Finnhub: {item.get('headline')} ({item.get('summary')})")
        except Exception as e:
            logger.error(f"Finnhub error: {e}")

    # 2. RSS Feeds
    try:
        rss_url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:2]:
            news_list.append(f"- RSS: {entry.title}")
    except Exception as e:
        logger.error(f"RSS error: {e}")

    return "\n".join(news_list) if news_list else "אין חדשות מתפרצות כעת."

# ----------------------------------------------------
# המנוע הראשי: סריקה אוטונומית ברקע
# ----------------------------------------------------

async def autonomous_market_scan(context: ContextTypes.DEFAULT_TYPE):
    logger.info("⚡ מריץ סריקה אוטונומית (מניות אמריקאיות וישראליות)...")
    
    watchlist_symbols = [
        "POLI.TA", "DSCT.TA", "ICL.TA",
        "TEVA", "NICE", "WIX", "CHKP", "MNDY",
        "NVDA", "AAPL", "TSLA"
    ] 
    
    for symbol in watchlist_symbols:
        tech_data = fetch_technical_signals(symbol)
        news_data = fetch_live_news(symbol)
        
        prompt = (
            f"אתה מנוע התראות מסחר אוטונומי בזמן אמת.\n"
            f"נתח את הנתונים הבאים עבור המנייה {symbol}:\n"
            f"נתונים טכניים (pandas-ta/yfinance):\n{tech_data}\n"
            f"חדשות בלייב (Finnhub/RSS):\n{news_data}\n\n"
            f"קבע אם יש אירוע חריג (כמו RSI קיצוני, פריצה, או חדשה בעלת אימפקט גבוה). "
            f"אם יש אירוע חריג, נסח התראה קצרה, חדה ומעוצבת בעברית כולל המלצת פעולה וסיכון. "
            f"אם אין שום דבר חריג, ענה במילה אחת בלבד: NONE."
        )
        
        try:
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            if "NONE" not in result_text and CHAT_ID:
                alert_message = f"🚨 **התראת מסחר אוטונומית בזמן אמת [{symbol}]**\n\n{result_text}"
                await context.bot.send_message(chat_id=CHAT_ID, text=alert_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in scan for {symbol}: {e}")

# ----------------------------------------------------
# מצב טסט (Test Mode)
# ----------------------------------------------------

async def handle_test_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧪 מריץ סימולציה מלאה של התראה עבור מנייה ישראלית...")
    
    test_symbol = "TEVA"
    tech_data = fetch_technical_signals(test_symbol)
    news_data = fetch_live_news(test_symbol)
    
    prompt = (
        f"אתה מנוע התראות מסחר. צור התראת טסט מעוצבת למנייה הישראלית {test_symbol} המבוססת על הנתונים הבאים:\n"
        f"טכני: {tech_data}\n"
        f"חדשות: {news_data}\n\n"
        f"השתמש באימוג'ים, הדגש את השורה התחתונה, רמות תמיכה/תנגדות ורמת סיכון."
    )
    
    try:
        response = model.generate_content(prompt)
        alert_msg = f"🧪 **[סימולציית התראה בזמן אמת - מנייה ישראלית]**\n\n{response.text}"
        await update.message.reply_text(alert_msg, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"שגיאה בהרצת הטסט: {e}")

# ----------------------------------------------------
# פקודות אופציונליות בלבד
# ----------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🤖 **מערכת התראות המסחר האוטונומית (ארה\"ב וישראל) פעילה!**\n\n"
        "📌 **פקודות בדיקה:**\n"
        "• `/test_alert` - הפעלת סימולציית התראה בלייב (מצב טסט)\n"
        "• `/technical <SYMBOL>` - ניתוח טכני יזום\n"
        "• `/news <SYMBOL>` - ניתוח חדשות יזום"
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def handle_technical(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("ציין סימול. דוגמה: `/technical TEVA`", parse_mode='Markdown')
        return
    symbol = context.args[0].upper()
    data = fetch_technical_signals(symbol)
    response = model.generate_content(f"סיכום טכני קצר בעברית עבור {symbol}:\n{data}")
    await update.message.reply_text(f"📊 **{symbol} ניתוח טכני:**\n\n{response.text}", parse_mode='Markdown')

async def handle_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("ציין סימול. דוגמה: `/news TEVA`", parse_mode='Markdown')
        return
    symbol = context.args[0].upper()
    data = fetch_live_news(symbol)
    response = model.generate_content(f"סיכום חדשות וסנטימנט בעברית עבור {symbol}:\n{data}")
    await update.message.reply_text(f"📰 **{symbol} חדשות:**\n\n{response.text}", parse_mode='Markdown')

# ----------------------------------------------------
# הרצת השרת
# ----------------------------------------------------

def main():
    tg_app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(CommandHandler("test_alert", handle_test_alert))
    tg_app.add_handler(CommandHandler("technical", handle_technical))
    tg_app.add_handler(CommandHandler("news", handle_news))

    job_queue = tg_app.job_queue
    if job_queue:
        job_queue.run_repeating(autonomous_market_scan, interval=300, first=10)

    logger.info("המערכת האוטונומית אותחלה בהצלחה...")
    tg_app.run_polling()

if __name__ == "__main__":
    main()
