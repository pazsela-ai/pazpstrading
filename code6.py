import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# טעינת משתני סביבה מקובץ .env
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# הגדרת לוגים להדפסת שגיאות ומידע לטרמינל
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def is_authorized(update: Update) -> bool:
    """פונקציית עזר לבדיקה האם השולח מורשה"""
    user_chat_id = str(update.effective_chat.id)
    if TELEGRAM_CHAT_ID and user_chat_id != str(TELEGRAM_CHAT_ID):
        logger.warning(f"פנייה לא מורשית מ-Chat ID: {user_chat_id}")
        return False
    return True

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("⛔ אין לך הרשאה להשתמש בבוט זה.")
        return
    await update.message.reply_text("👋 שלום! הבוט פעיל ומחובר בהצלחה.")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("⛔ אין לך הרשאה להשתמש בבוט זה.")
        return
    await update.message.reply_text("🟢 הבוט פעיל והמערכת פועלת כסדרה.")

async def test_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("⛔ אין לך הרשאה להשתמש בבוט זה.")
        return
    await update.message.reply_text("📰 בדיקת חדשות: המערכת מוכנה לקבלת עדכוני חדשות.")

async def test_tech_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("⛔ אין לך הרשאה להשתמש בבוט זה.")
        return
    await update.message.reply_text("📊 בדיקה טכנית: ניתוח אינדיקטורים פועל כשורה.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """הדפסת שגיאות לטרמינל במקרה של תקלה"""
    logger.error(msg="אירעה שגיאה בעת טיפול בהודעה:", exc_info=context.error)

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("❌ שגיאה: לא נמצא TELEGRAM_BOT_TOKEN בקובץ .env")
        return

    print("🚀 הבוט מופעל כעת ומאזין לפקודות...")
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # רישום פקודות
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("test_news", test_news_command))
    app.add_handler(CommandHandler("test_tech", test_tech_command))

    # רישום מנגנון טיפול בשגיאות
    app.add_error_handler(error_handler)

    # הפעלת הבוט
    app.run_polling()

if __name__ == "__main__":
    main()
