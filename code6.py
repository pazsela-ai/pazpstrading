import os
from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "Bot is alive!"

def run():
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

keep_alive()
import asyncio
import logging
import html
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from google import genai

# הגדרת הלוגים
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- מפתחות ה-API המעודכנים ---
TELEGRAM_BOT_TOKEN = "8633108999:AAE4rpqerPZFF4rBNCc0Yiaj0ACJKj8UV40"
GEMINI_API_KEY = "AQ.Ab8RN6IDZWWqNEQs6gF7jEgKO-1RTbxi0nenZqPtRzMdP5arGA"

# אתחול הלקוח של Gemini
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# --- פונקציית עזר לתקשורת מול GEMINI ---
async def call_gemini_with_retry(prompt: str, retries: int = 3) -> str:
    """שולח פנייה ל-Gemini עם מנגנון ניסיון חוזר למקרה של עומס (503) או שגיאת רשת"""
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.warning(f"ניסיון {attempt + 1} נכשל מול Gemini: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # השהייה אקספוננציאלית (1, 2, 4 שניות)
            else:
                return "מצטער, השרת חווה עומס כרגע. אנא נסה שוב בעוד מספר רגעים."


# --- הנדלרים של טלגרם ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """פקודת התחלה"""
    await update.message.reply_text("שלום! הבוט פעיל ומוכן לעבודה.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """טיפול בהודעות טקסט רגילות"""
    user_text = update.message.text
    
    # הודעה זמנית למשתמש
    status_msg = await update.message.reply_text("מעבד את הבקשה...")
    
    # פנייה ל-Gemini
    ai_response = await call_gemini_with_retry(user_text)
    
    # מניעת שגיאות Parse Mode בטלגרם על ידי ניקוי התווים והגנה על HTML
    safe_text = html.escape(ai_response)
    
    try:
        await status_msg.edit_text(safe_text, parse_mode=ParseMode.HTML)
    except Exception:
        # אם יש עדיין בעיית עיצוב, נשלח כטקסט נקי ללא Parse Mode
        await status_msg.edit_text(ai_response)


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """טיפול בלחיצות על כפתורים (Inline Buttons)"""
    query = update.callback_query
    
    # מענה מיידי לטלגרם למניעת שגיאת Query is too old
    await query.answer()
    
    # המשך הלוגיקה של הכפתור
    await query.edit_message_text(text=f"בחרת באפשרות: {query.data}")


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """מטפל שגיאות מרכזי - מונע קריסה מלאה של הבוט בבעיות אינטרנט/DNS"""
    logger.error("התרחשה שגיאה בלתי צפויה:", exc_info=context.error)
    
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "התרחשה שגיאת תקשורת זמנית. הבוט ימשיך לפעול מיד."
            )
        except Exception:
            pass


# --- הפעלת הבוט ---

def main():
    # בניית האפליקציה
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # רישום הנדלרים
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(handle_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # רישום מטפל השגיאות המרכזי
    application.add_error_handler(global_error_handler)

    # הרצת הבוט
    logger.info("הבוט מופעל...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
