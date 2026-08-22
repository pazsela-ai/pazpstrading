import os
import logging
import threading
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import finnhub

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from google import genai
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ==========================================
# 1. הגדרת Logging
# ==========================================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. טעינת משתני סביבה ולקוחות API
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")

# לקוח Gemini 2.5
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# לקוח Finnhub
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# ==========================================
# 3. שרת Flask ל-Keep Alive (מניעת הרדמה ב-Render)
# ==========================================
app = Flask(__name__)

@app.route("/")
def home():
    return "Trading Bot is Live & Active!", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

# ==========================================
# 4. פונקציות עזר: נתוני שוק ואינדיקטורים
# ==========================================
def get_live_technical_data(symbol: str):
    """מושך נתוני מחיר היסטוריים ומחשב אינדיקטורים בעזרת pandas-ta"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="3m")
    
    if df.empty or len(df) < 20:
        return None

    # חישוב אינדיקטורים בעזרת pandas-ta
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    latest = df.iloc[-1]
    
    current_price = round(latest["Close"], 2)
    rsi_val = round(latest["RSI_14"], 2) if "RSI_14" in latest and not pd.isna(latest["RSI_14"]) else 50.0
    sma20_val = round(latest["SMA_20"], 2) if "SMA_20" in latest and not pd.isna(latest["SMA_20"]) else current_price
    volume = int(latest["Volume"])
    avg_volume = int(df["Volume"].tail(20).mean())
    vol_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0

    # גזירת תוכנית מסחר דינמית לניהול סיכונים
    stop_loss = round(current_price * 0.96, 2)   # סיכון של 4%
    take_profit = round(current_price * 1.09, 2) # יעד רווח של 9%
    risk = round(current_price - stop_loss, 2)
    reward = round(take_profit - current_price, 2)
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    return {
        "symbol": symbol.upper(),
        "price": current_price,
        "rsi": rsi_val,
        "sma20": sma20_val,
        "vol_ratio": vol_ratio,
        "entry": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rr_ratio": rr_ratio,
    }

def get_latest_company_news(symbol: str):
    """מושך את הכתבה העדכנית ביותר מ-Finnhub"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    from_str = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    
    try:
        news_list = finnhub_client.company_news(symbol.upper(), _from=from_str, to=today_str)
        if news_list and len(news_list) > 0:
            top_news = news_list[0]
            return {
                "headline": top_news.get("headline", "אין כותרת"),
                "summary": top_news.get("summary", "אין תקציר זמין"),
                "source": top_news.get("source", "Finnhub News"),
                "url": top_news.get("url", ""),
            }
    except Exception as e:
        logger.error(f"Error fetching news from Finnhub: {e}")
    return None

# ==========================================
# 5. פקודות הבוט בטלגרם
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🤖 **ברוכים הבאים לבוט המסחר והסייען הפיננסי!**\n\n"
        "הבוט מחובר כעת לנתוני אמת של השוק ומשלב בינה מלאכותית (Gemini 2.5).\n\n"
        "📌 **פקודות זמינות:**\n"
        "• `/technical <SYMBOL>` - ניתוח טכני בלייב (דוגמה: `/technical NVDA`)\n"
        "• `/news <SYMBOL>` - ניתוח חדשות וסנטימנט בלייב (דוגמה: `/news AAPL`)\n"
        "• **טקסט חופשי** - שאל כל שאלה בנושאי שוק ההון ומסחר!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def technical_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """טיפול בפקודת ניתוח טכני בלייב"""
    if not context.args:
        await update.message.reply_text("❌ אנא ציין סימול מניה. דוגמה: `/technical TSLA`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"📊 מושך נתוני אמת ומחשב אינדיקטורים עבור **{symbol}**...", parse_mode="Markdown")

    data = get_live_technical_data(symbol)
    if not data:
        await update.message.reply_text(f"❌ לא ניתן היה למשוך נתונים עבור הסימול `{symbol}`. ודא שהסימול תקין.", parse_mode="Markdown")
        return

    report = (
        f"📊 **דוח ניתוח טכני בזמן אמת — ${data['symbol']}**\n"
        f"───────\n"
        f"💰 **מחיר נוכחי:** ${data['price']}\n"
        f"📈 **אינדיקטורים (pandas-ta):**\n"
        f"• **RSI (14):** {data['rsi']} " + ("(קניות יתר ⚠️)" if data['rsi'] > 70 else "(מכירות יתר 🟢)" if data['rsi'] < 30 else "(ניטרלי)") + "\n"
        f"• **SMA (20):** ${data['sma20']} " + ("(מגמה עולה 🟢)" if data['price'] > data['sma20'] else "(מגמה יורדת 🔴)") + "\n"
        f"• **יחס נפח מסחר:** x{data['vol_ratio']} מהממוצע\n\n"
        f"🎯 **תוכנית עבודה וניהול סיכונים:**\n"
        f"• **מחיר כניסה (Entry):** ${data['entry']}\n"
        f"• **יעד רווח (Take Profit):** ${data['take_profit']}\n"
        f"• **סטופ לוס (Stop Loss):** ${data['stop_loss']}\n"
        f"• **יחס סיכון/סיכוי (R:R):** 1:{data['rr_ratio']}\n"
    )
    await update.message.reply_text(report, parse_mode="Markdown")

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """טיפול בפקודת ניתוח חדשות בלייב באמצעות Finnhub + Gemini"""
    if not context.args:
        await update.message.reply_text("❌ אנא ציין סימול מניה. דוגמה: `/news NVDA`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"📰 מנטר חדשות אחרונות עבור **{symbol}** ומנתח בעזרת Gemini...", parse_mode="Markdown")

    news_item = get_latest_company_news(symbol)
    if not news_item:
        await update.message.reply_text(f"ℹ️ לא נמצאו דיווחים חדשותיים ב-5 הימים האחרונים עבור `{symbol}`.", parse_mode="Markdown")
        return

    # ניתוח הידיעה בעזרת Gemini 2.5
    prompt = (
        f"אתה אנליסט מומחה בשוק ההון. נתח את הידיעה החדשותית הבאה עבור מניית {symbol}:\n\n"
        f"כותרת: {news_item['headline']}\n"
        f"תקציר: {news_item['summary']}\n"
        f"מקור: {news_item['source']}\n\n"
        f"הנפק תמצית קצרה בעברית הכוללת:\n"
        f"1. ניתוח סנטימנט (חיובי/שלילי/ניטרלי והסבר קצר).\n"
        f"2. הערכת עוצמת ההשפעה על המחיר בטווח הקצר.\n"
        f"3. אזהרת סיכון FOMO במידה ורלוונטי."
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        ai_analysis = response.text
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        ai_analysis = "לא ניתן היה להשלים את הניתוח האוטומטי."

    report = (
        f"📰 **ניתוח חדשות ואירועי מפתח — ${symbol}**\n"
        f"───────\n"
        f"📌 **כותרת:** {news_item['headline']}\n"
        f"🏢 **מקור:** {news_item['source']}\n\n"
        f"🤖 **ניתוח AI (Gemini 2.5):**\n"
        f"{ai_analysis}\n\n"
        f"🔗 [לקריאת הכתבה המלאה]({news_item['url']})"
    )
    await update.message.reply_text(report, parse_mode="Markdown", disable_web_page_preview=True)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """מענה לשאילתות טקסט חופשיות דרך Gemini"""
    user_text = update.message.text
    prompt = f"אתה סייען מסחר ושוק ההון מקצועי. ענה בקצרה, בדיוק ובעברית ברורה: {user_text}"

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        await update.message.reply_text(response.text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error handling text message: {e}")
        await update.message.reply_text("מצטער, נתקלתי בשגיאה בעיבוד הבקשה.")

# ==========================================
# 6. הפעלת השרת והבוט (Main Loop)
# ==========================================
def main():
    # 1. הפעלת שרת ה-Flask בשרשור נפרד למניעת נתק ב-Render
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    logger.info("Flask Keep-Alive Server started successfully.")

    # 2. אתחול מנוע APScheduler (מוכן למשימות מתתוזמנות)
    scheduler = BackgroundScheduler()
    scheduler.start()

    # 3. אתחול והפעלת Telegram Application
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", start_command))
    application.add_handler(CommandHandler("technical", technical_command))
    application.add_handler(CommandHandler("news", news_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Telegram Bot started listening for messages...")
    application.run_polling()

if __name__ == "__main__":
    main()
