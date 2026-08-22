import os
import sys
import logging
import threading
from flask import Flask

import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ---------------------------------------------------------------------------
# 1. הגדרות לוגים
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2. משתני סביבה (Environment Variables)
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN_HERE")
CHAT_ID = os.environ.get("CHAT_ID", "YOUR_CHAT_ID_HERE")

# ---------------------------------------------------------------------------
# 3. הגדרת שרת Flask עבור Render (Web Service)
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot Server is Alive and Running!"

def run_flask():
    # Render מזין באופן אוטומטי את משתנה הסביבה PORT
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------------------------------------------------------------------
# 4. פונקציות ופקודות טלגרם
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("היי! הבוט פעיל ועובד בהצלחה.")

async def handle_test_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("בדיקת התראה: הכל תקין!")

async def handle_technical(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("מבצע ניתוח טכני...")

async def handle_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("טוען עדכוני חדשות...")

async def autonomous_market_scan(context: ContextTypes.DEFAULT_TYPE) -> None:
    # כאן נכנסת לוגיקת הסריקה המחזורית שלך
    logger.info("מבצע סריקה אוטונומית...")

# ---------------------------------------------------------------------------
# 5. נקודת הכניסה הראשית (Main)
# ---------------------------------------------------------------------------
def main():
    # הפעלת שרת ה-Flask ברקע (Thread נפרד)
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    logger.info("שרת ה-Flask הופעל ברקע.")

    # אתחול אפליקציית הטלגרם
    tg_app = Application.builder().token(TELEGRAM_TOKEN).build()

    # רישום Handlers
    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(CommandHandler("test_alert", handle_test_alert))
    tg_app.add_handler(CommandHandler("technical", handle_technical))
    tg_app.add_handler(CommandHandler("news", handle_news))

    # תזמון משימה מחזורית (במידה וקיים JobQueue)
    job_queue = tg_app.job_queue
    if job_queue:
        job_queue.run_repeating(autonomous_market_scan, interval=300, first=10)

    logger.info("מתחיל הרצת הבוט במצב Polling...")
    tg_app.run_polling()

if __name__ == "__main__":
    main()
