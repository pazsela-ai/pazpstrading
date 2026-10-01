import os
import logging
import threading
import time
import feedparser
import json
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from flask import Flask
from telebot import TeleBot, types
from apscheduler.schedulers.background import BackgroundScheduler
from google import genai

# הגדרת לוגים
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# הגדרות משתני סביבה
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# אתחול לקוח AI (Gemini)
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# מסד נתונים פשוט בזיכרון לסימולציות
simulated_trades = []
last_scans = {"news": "טרם בוצעה", "tech": "טרם בוצעה"}

# רשימת מעקב לדוגמה למניות במדדים מובילים (ת"א 125, S&P 500, Nasdaq)
WATCHLIST = [
    {"symbol": "ELAL.TA", "name": "אל על", "market": "TASE"},
    {"symbol": "TEVA.TA", "name": "טבע", "market": "TASE"},
    {"symbol": "NVDA", "name": "NVIDIA", "market": "NASDAQ"},
    {"symbol": "AAPL", "name": "Apple", "market": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft", "market": "NASDAQ"},
]

# ---------------------------------------------------------
# 1. מנוע ניתוח חדשות ואירועים (AI Catalyst Engine)
# ---------------------------------------------------------

def analyze_news_with_ai(headline, summary):
    """שולח את הידיעה ל-AI לניתוח סנטימנט, זיהוי מניה ונימוק"""
    if not ai_client:
        return None

    prompt = f"""
    נתח את הידיעה הכלכלית/חדשותית הבאה:
    כותרת: {headline}
    תקציר: {summary}

    אם הידיעה מצביעה על קטליזטור חיובי משמעותי עבור מניה מסוימת במדדים מובילים (ת"א 125, S&P500, Nasdaq):
    1. החזר את סימול המניה (למשל ELAL.TA, TEVA, NVDA).
    2. נימוק קצר וברור (משפט 1-2) מדוע הידיעה צפויה להשפיע לחיוב על מניית החברה או מתחרותיה.
    3. עוצמת סנטימנט (HIGH / MEDIUM).

    החזר תשובה בפורמט JSON בלבד:
    {{"is_relevant": true, "ticker": "ELAL.TA", "reason": "נימוק...", "conviction": "HIGH"}}
    אם אינה רלוונטית:
    {{"is_relevant": false}}
    """
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
res_text = response.text.strip().replace("```json", "").replace("```", "")
