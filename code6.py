import os
import re
import json
import logging
import time
from datetime import datetime, timedelta
import pytz
import feedparser
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from flask import Flask, request
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler

# AI SDKs with safe imports
import google.generativeai as genai

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------------------------------------------------------------------------
# 0. Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("PazPSTradingBot")

# ---------------------------------------------------------------------------
# 1. Environment & API Initialization
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, threaded=False) if TELEGRAM_BOT_TOKEN else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if Groq and GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI and OPENAI_API_KEY else None

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 2. Global State & Configuration
# ---------------------------------------------------------------------------
sent_ticker_cooldowns = {}
seen_articles = set()

# Watchlist לניתוח טכני סדיר בלבד
WATCHLIST = [
    # --- ארה"ב: טכנולוגיה, AI וסמיקונדקטורס ---
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "INTC",
    "QCOM", "ARM", "MU", "SMCI", "AMAT", "LRCX", "KLAC", "PANW", "CRWD", "PLTR",
    "SNOW", "ORCL", "IBM", "NOW", "DDOG", "NET", "ZS",

    # --- ארה"ב: פינטק, קריפטו ופיננסים ---
    "COIN", "MSTR", "MARA", "RIOT", "PYPL", "SQ", "HOOD", "V", "MA", "JPM", "BAC", "GS",

    # --- ארה"ב: תעופה, ביטחון, אנרגיה וקמעונאות ---
    "BA", "LMT", "NOC", "RTX", "DAL", "UAL", "AAL", "XOM", "CVX", "F", "GM", "RIVN", "LCID",

    # --- ארה"ב: פארמה, בריאות וביוטק ---
    "PFE", "MRNA", "LLY", "NVO", "JNJ", "ABBV", "WMT", "COST", "TGT", "DIS", "NFLX",

    # --- ישראל (בורסת תל אביב - .TA) ---
    "TEVA.TA", "NICE.TA", "ICL.TA", "TSEM.TA", "ELBT.TA", "CAMT.TA", "PRIO.TA",
    "POLI.TA", "LUMI.TA", "DISB.TA", "MZTF.TA", "FIBI.TA", "ENLT.TA", "ENOG.TA",
    "AZRG.TA", "NMVG.TA", "DELT.TA", "DNYA.TA", "PHRE.TA"
]

# RSS Feeds מקיפים לכיסוי חדשות, מאקרו, דוחות ואירועים גלובליים
BROAD
