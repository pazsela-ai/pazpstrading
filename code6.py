import os
from flask import Flask
from threading import Thread
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from google import genai

# ==========================================
# 1. שרת רשת פנימי לשמירה על השרת פעיל ב-Render
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is active and running!"

def run():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# הפעלת שרת הרשת
keep_alive()

# ==========================================
# 2. הגדרת יומנים (Logging)
# ==========================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==========================================
# 3. מפתחות API והגדרת לקוח Gemini
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 4. פונקציות הבוט בטלגרם
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("היי! הבוט פעיל ומוכן לקבל הודעות.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_text,
        )
        await update.message.reply_text(response.text)
    except Exception as e:
        logging.error(f"Error generating content: {e}")
        await update.message.reply_text("תרחשה שגיאה בעת פנייה ל-Gemini API.")

# ==========================================
# 5. הרצת הבוט
# ==========================================
if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    
    application.add_handler(start_handler)
    application.add_handler(message_handler)
    
    application.run_polling()
