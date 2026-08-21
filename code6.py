import os
import asyncio
import logging
from flask import Flask
from threading import Thread
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from google import genai

# ==========================================
# 1. הגדרת יומנים (Logging)
# ==========================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==========================================
# 2. מפתחות API
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 3. פונקציות הבוט בטלגרם
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

def run_telegram_bot():
    """הרצת הלולאה של טלגרם ב-Thread נפרד"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    
    application.add_handler(start_handler)
    application.add_handler(message_handler)
    
    logging.info("Starting Telegram Bot Polling...")
    application.run_polling(stop_signals=None)

# הפעלת הבוט בשרשור נפרד ברקע
bot_thread = Thread(target=run_telegram_bot, daemon=True)
bot_thread.start()

# ==========================================
# 4. שרת רשת ראשי (Flask) שרוכב על הפורט של Render
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive and running!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
