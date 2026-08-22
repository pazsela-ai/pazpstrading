import os
import logging
import asyncio
import threading
import random
from flask import Flask
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import google.generativeai as genai
from apscheduler.schedulers.background import BackgroundScheduler

# ---------------------------------------------------------
# 1. הגדרת לוגים (Logging)
# ---------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 2. טעינת משתני סביבה (Environment Variables)
# ---------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN is missing!")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is missing!")

# ---------------------------------------------------------
# 3. הגדרת Google Gemini API
# ---------------------------------------------------------
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

# ---------------------------------------------------------
# 4. הגדרת שרת Flask (Keep-Alive עבור Render)
# ---------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive and running!"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------------------------------------------------
# 5. הגדרת APScheduler
# ---------------------------------------------------------
scheduler = BackgroundScheduler()

# ---------------------------------------------------------
# 6. פונקציות הטיפול בהודעות טלגרם (Handlers)
# ---------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """פקודת /start"""
    user_first_name = update.effective_user.first_name if update.effective_user else "משתמש/ת"
    welcome_text = (
        f"שלום {user_first_name}! 👋\n"
        "אני הבוט האישי שלך לניתוח מסחר ושוק ההון.\n\n"
        "**פקודות זמינות לבדיקה:**\n"
        "• `/test_technical` - הרצת טסט טכני סימולטיבי למניה\n"
        "• `/test_news` - הרצת טסט חדשותי סימולטיבי למניה\n"
        "• `/help` - עזרה והנחיות"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """פקודת /help"""
    help_text = (
        "📌 **מפתח פקודות:**\n\n"
        "• `/test_technical` - מציג ניתוח טכני מלא לדוגמה (RSI, ממוצעים, נפח, יעד וסטופ).\n"
        "• `/test_news` - מציג ניתוח חדשותי לדוגמה (אירוע קטליזטור, סנטימנט וסינון FOMO).\n"
        "• שלח/י לי טקסט חופשי או סימול מניה ואענה באמצעות Gemini."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def test_technical(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """פקודת טסט לניתוח טכני סימולטיבי"""
    tech_stocks = [
        {"ticker": "NVDA", "name": "Nvidia Corp", "price": 128.50, "rsi": 54.2, "volume_ratio": "x2.4 מהממוצע", "sma20": "מעל SMA20 (122.10$)", "entry": 128.50, "target": 142.00, "stop": 122.00},
        {"ticker": "AAPL", "name": "Apple Inc", "price": 224.30, "rsi": 61.8, "volume_ratio": "x1.8 מהממוצע", "sma20": "מעל SMA20 (218.00$)", "entry": 224.30, "target": 245.00, "stop": 215.00},
        {"ticker": "AMD", "name": "Advanced Micro Devices", "price": 156.80, "rsi": 48.5, "volume_ratio": "x3.1 מהממוצע", "sma20": "חיתוך מעל SMA20 (150.20$)", "entry": 156.80, "target": 178.00, "stop": 148.50},
        {"ticker": "TSLA", "name": "Tesla Inc", "price": 210.40, "rsi": 58.1, "volume_ratio": "x2.0 מהממוצע", "sma20": "מעל SMA20 (198.50$)", "entry": 210.40, "target": 235.00, "stop": 197.00}
    ]
    
    stock = random.choice(tech_stocks)
    potential_gain = round(((stock['target'] - stock['entry']) / stock['entry']) * 100, 1)
    max_loss = round(((stock['entry'] - stock['stop']) / stock['entry']) * 100, 1)
    risk_reward = round((stock['target'] - stock['entry']) / (stock['entry'] - stock['stop']), 2)

    msg = (
        "⚠️ **טסט בלבד**\n\n"
        f"📊 **דו\"ח ניתוח טכני סימולטיבי - {stock['ticker']} ({stock['name']})**\n"
        "───────────────────────\n"
        f"🔹 **מחיר נוכחי:** ${stock['price']}\n"
        f"🔹 **נפח מסחר:** {stock['volume_ratio']} (זיהוי תנועה ראשונית)\n"
        f"🔹 **מדד RSI (14):** {stock['rsi']} (טווחי מומנטום בריאים)\n"
        f"🔹 **ממוצע נע SMA20:** {stock['sma20']}\n\n"
        "🎯 **תוכנית עבודה מומלצת (ניהול סיכונים):**\n"
        f"• **נקודת כניסה (Entry):** ${stock['entry']}\n"
        f"• **מחיר יעד (Take Profit):** ${stock['target']} (+{potential_gain}%)\n"
        f"• **קטיעת הפסד (Stop Loss):** ${stock['stop']} (-{max_loss}%)\n"
        f"• **יחס סיכון/סיכוי (R:R):** 1:{risk_reward}\n\n"
        "⚙️ *הערה: הנתונים המופקים לעיל הם לצורכי בדיקת המערכת בלבד.*"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def test_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """פקודת טסט לניתוח חדשותי סימולטיבי"""
    news_items = [
        {
            "ticker": "AMZN",
            "headline": "אמזון מודיעה על שיתוף פעולה אסטרטגי בתחום תשתיות AI ועננים",
            "source": "Bloomberg",
            "impact": "חיובי חזק (High Impact)",
            "fomo_risk": "נמוך - המניה בתחילת תנועה, טרם הגיעה לשיא",
            "summary": "החוזה כולל התחייבות לרכישת שירותי ענן בהיקף של 5 מיליארד דולר על פני 3 שנים, צפוי להעלות את התחזית לרבעון הבא."
        },
        {
            "ticker": "MSFT",
            "headline": "מיקרוסופט מציגה גידול של 28% בהכנסות מחטיבת הענן",
            "source": "Reuters",
            "impact": "חיובי בינוני (Medium Impact)",
            "fomo_risk": "בינוני - מומלץ להמתין למעקב נפח במסחר הסדיר",
            "summary": "התוצאות עברו את תחזיות האנליסטים בוול סטריט. התגובה במסחר המוקדם מתונה."
        },
        {
            "ticker": "GOOGL",
            "headline": "אישור רגולטורי באירופה להרחבת שירותי ה-Autonomous Driving",
            "source": "CNBC",
            "impact": "חיובי לטווח קצר/בינוני",
            "fomo_risk": "נמוך - זיהוי מוקדם של הודעת המפתח",
            "summary": "האישור מאפשר תחילת ניסויים מסחריים ב-3 ערים מרכזיות באירופה החל מהרבעון הרביעי."
        }
    ]
    
    item = random.choice(news_items)

    msg = (
        "⚠️ **טסט בלבד**\n\n"
        f"📰 **דו\"ח ניתוח חדשות ואירועי מפתח - {item['ticker']}**\n"
        "───────────────────────\n"
        f"📣 **כותרת:** {item['headline']}\n"
        f"🌐 **מקור:** {item['source']}\n"
        f"💥 **השפעה משוערת:** {item['impact']}\n"
        f"🛡️ **סינון סיכון FOMO:** {item['fomo_risk']}\n\n"
        f"📝 **תמצית הדיווח:**\n{item['summary']}\n\n"
        "⚙️ *הערה: הנתונים המופקים לעיל הם לצורכי בדיקת המערכת בלבד.*"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """טיפול בהודעות טקסט רגילות בעזרת Gemini"""
    user_text = update.message.text

    if not model:
        await update.message.reply_text("מפתח Gemini API אינו מוגדר. אנא בדוק/י את משתני הסביבה.")
        return

    try:
        response = model.generate_content(user_text)
        bot_response = response.text if response and response.text else "לא התקבלה תשובה מ-Gemini."
        await update.message.reply_text(bot_response)
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        await update.message.reply_text("התרחשה שגיאה בעת פנייה ל-Gemini. אנא נסה/י שוב מאוחר יותר.")

# ---------------------------------------------------------
# 7. פונקציית ההפעלה הראשית (Main)
# ---------------------------------------------------------
def main():
    if not TELEGRAM_TOKEN:
        logger.critical("Cannot start bot without TELEGRAM_TOKEN. Exiting.")
        return

    # א. הפעלת שרת ה-Flask בשרשור נפרד ברקע
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    logger.info("Flask server thread started.")

    # ב. יצירת והגדרת Event Loop מפורש עבור asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # ג. בניית אפליקציית הבוט של טלגרם
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # ד. רישום פקודות ואירועים
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("test_technical", test_technical))
    application.add_handler(CommandHandler("test_news", test_news))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # ה. הפעלת תזמונים במידה והוגדרו
    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler started.")

    # ו. הפעלת הבוט במצב Polling
    logger.info("Starting Telegram Bot polling...")
    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
