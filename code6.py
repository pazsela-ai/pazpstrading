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

# AI SDKs
import google.generativeai as genai
from groq import Groq
from openai import OpenAI

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

# Initialize Telegram Bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None

# Initialize AI Clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Flask App
app = Flask(__name__)

# ---------------------------------------------------------------------------
# 2. Global State & Configuration
# ---------------------------------------------------------------------------
# Cooldown tracking: ticker -> datetime of last alert sent
sent_ticker_cooldowns = {}

# RSS seen articles tracking
seen_articles = set()

# WATCHLIST (לניתוח טכני סדיר בלבד - מורחב ומקיף)
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

# RSS Feeds משולב: חדשות מתפרצות + דיווחי דוחות כספיים ואירועים מהותיים בחברות
BROAD_NEWS_FEEDS = [
    # ------------------- חלק א': חדשות עולמיות וישראליות כלליות -------------------
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.ynet.co.il/Integration/StoryRss1854.xml", # מבזקי Ynet

    # ------------------- חלק ב': חדשות נושאיות (תעופה, ביטחון, פארמה, נפט) -------------------
    "https://news.google.com/rss/search?q=aviation+airline+incident+flight&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=war+military+strike+tensions+Middle+East&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=pharma+FDA+approval+clinical+trial+phase&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=oil+gas+strait+hormuz+pipeline+energy&hl=en-US&gl=US&ceid=US:en",

    # ------------------- חלק ג': דוחות כספיים, תוצאות רבעוניות ודיווחי חברות רשמיים -------------------
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom", # SEC 8-K (דיווחים מיידיים בארה"ב)
    "https://news.google.com/rss/search?q=earnings+report+quarterly+results+revenue+EPS+beat+miss&hl=en-US&gl=US&ceid=US:en", # דוחות רבעוניים ארה"ב
    "https://news.google.com/rss/search?q=site:maya.tase.co.il+דוח+מיידי+OR+דוח+כספי+OR+תוצאות&hl=he&gl=IL&ceid=IL:he", # דיווחי הבורסה בתל אביב (מאיה)
    "https://news.google.com/rss/search?q=PR+Newswire+earnings+release+quarterly&hl=en-US&gl=US&ceid=US:en" # הודעות לתקשורת מחברות
]

# ---------------------------------------------------------------------------
# 3. AI Failover Engine (Gemini -> Groq -> OpenAI)
# ---------------------------------------------------------------------------
def call_ai_failover(prompt: str) -> str:
    """
    Tries Gemini (2.0 Flash / 1.5 Flash), then Groq (llama-3.3-70b), then OpenAI (gpt-4o-mini).
    """
    # 1. Gemini Try
    if GEMINI_API_KEY:
        for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    logger.info(f"AI Success with Gemini ({model_name})")
                    return response.text
            except Exception as e:
                logger.warning(f"Gemini {model_name} failed: {e}")

    # 2. Groq Try
    if groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            res_text = completion.choices[0].message.content
            if res_text:
                logger.info("AI Success with Groq (llama-3.3-70b-versatile)")
                return res_text
        except Exception as e:
            logger.warning(f"Groq failed: {e}")

    # 3. OpenAI Try
    if openai_client:
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            res_text = completion.choices[0].message.content
            if res_text:
                logger.info("AI Success with OpenAI (gpt-4o-mini)")
                return res_text
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")

    logger.error("All AI providers failed.")
    return ""

# ---------------------------------------------------------------------------
# 4. Market Status & Helper Functions
# ---------------------------------------------------------------------------
def check_market_status(ticker: str) -> dict:
    """
    Checks if market is OPEN, PRE_MARKET, or CLOSED for given ticker.
    """
    is_israel = ticker.endswith(".TA")
    tz_il = pytz.timezone("Asia/Jerusalem")
    now_il = datetime.now(tz_il)

    weekday = now_il.weekday() # 0=Mon, 1=Tue, ..., 6=Sun
    time_str = now_il.strftime("%H:%M")

    if is_israel:
        # Israel Market: Sunday (6) to Thursday (3)
        if weekday in [6, 0, 1, 2, 3]:
            if "09:59" <= time_str <= "17:30":
                return {"status": "OPEN", "market": "Israel (TASE)"}
        return {"status": "CLOSED", "market": "Israel (TASE)"}
    else:
        # US Market: Monday (0) to Friday (4)
        if weekday in [0, 1, 2, 3, 4]:
            if "11:00" <= time_str < "16:30":
                return {"status": "PRE_MARKET", "market": "US Markets"}
            elif "16:30" <= time_str <= "23:00":
                return {"status": "OPEN", "market": "US Markets"}
        return {"status": "CLOSED", "market": "US Markets"}

def check_liquidity_and_price(ticker: str) -> dict:
