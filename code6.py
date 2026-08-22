import os
import logging
import asyncio
import threading
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

# בדיקת תקינות משתני הסביבה
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN is missing!")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is missing!")

# ---------------------------------------------------------
# 3. הגדרת Google Gemini API
# ---------------------------------------------------------
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # שימוש בדגם Gemini העדכני
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
    """הרצת שרת ה-Flask בשרשור נפרד כדי שלא יחסום את הבוט"""
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------------------------------------------------
# 5. הגדרת APScheduler (לתזמון משימות ברקע)
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
        "אני הבוט האישי שלך. תוכל/י לשאול אותי שאלות ואיעזר ב-Gemini כדי לענות."
    )
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """פקודת /help"""
    help_text = (
        "הנה מה שאני יודע לעשות:\n"
        "• שלח/י לי הודעת טקסט חופשית ואענה בעזרת בינה מלאכותית.\n"
        "• השתמש/י ב-/start כדי להתחיל מחדש."
    )
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """טיפול בהודעות טקסט רגילות בעזרת Gemini"""
    user_text = update.message.text

    if not model:
        await update.message.reply_text("מפתח Gemini API אינו מוגדר. אנא בדוק/י את משתני הסביבה.")
        return

    try:
        # שליחת הודעה מיועדת לעיבוד
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

    # ב. יצירת והגדרת Event Loop מפורש עבור asyncio (פתרון ל-Python 3.14/Render)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # ג. בניית אפליקציית הבוט של טלגרם
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # ד. רישום פקודות ואירועים
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
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
