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

# Initialize Telegram Bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, threaded=False) if TELEGRAM_BOT_TOKEN else None

# Initialize AI Clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if Groq and GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI and OPENAI_API_KEY else None

# Flask App
app = Flask(__name__)

# ---------------------------------------------------------------------------
# 2. Global State & Configuration
# ---------------------------------------------------------------------------
sent_ticker_cooldowns = {}
seen_articles = set()

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

BROAD_NEWS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.ynet.co.il/Integration/StoryRss1854.xml",
    "https://news.google.com/rss/search?q=aviation+airline+incident+flight&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=war+military+strike+tensions+Middle+East&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=pharma+FDA+approval+clinical+trial+phase&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=oil+gas+strait+hormuz+pipeline+energy&hl=en-US&gl=US&ceid=US:en",
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom",
    "https://news.google.com/rss/search?q=earnings+report+quarterly+results+revenue+EPS+beat+miss&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=site:maya.tase.co.il+דוח+מיידי+OR+דוח+כספי+OR+תוצאות&hl=he&gl=IL&ceid=IL:he",
    "https://news.google.com/rss/search?q=PR+Newswire+earnings+release+quarterly&hl=en-US&gl=US&ceid=US:en"
]

# ---------------------------------------------------------------------------
# 3. AI Failover Engine
# ---------------------------------------------------------------------------
def call_ai_failover(prompt: str) -> str:
    if GEMINI_API_KEY:
        for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"Gemini {model_name} failed: {e}")

    if groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            res_text = completion.choices[0].message.content
            if res_text:
                return res_text
        except Exception as e:
            logger.warning(f"Groq failed: {e}")

    if openai_client:
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            res_text = completion.choices[0].message.content
            if res_text:
                return res_text
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")

    return ""

# ---------------------------------------------------------------------------
# 4. Market Status & Helper Functions
# ---------------------------------------------------------------------------
def check_market_status(ticker: str) -> dict:
    is_israel = ticker.endswith(".TA")
    tz_il = pytz.timezone("Asia/Jerusalem")
    now_il = datetime.now(tz_il)
    weekday = now_il.weekday()
    time_str = now_il.strftime("%H:%M")

    if is_israel:
        if weekday in [6, 0, 1, 2, 3]:
            if "09:59" <= time_str <= "17:30":
                return {"status": "OPEN", "market": "Israel (TASE)"}
        return {"status": "CLOSED", "market": "Israel (TASE)"}
    else:
        if weekday in [0, 1, 2, 3, 4]:
            if "11:00" <= time_str < "16:30":
                return {"status": "PRE_MARKET", "market": "US Markets"}
            elif "16:30" <= time_str <= "23:00":
                return {"status": "OPEN", "market": "US Markets"}
        return {"status": "CLOSED", "market": "US Markets"}

def check_liquidity_and_price(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="60d", interval="1d")
        if df.empty or len(df) < 20:
            return {"valid": False, "reason": "Insufficient market data"}

        df.ta.atr(length=14, append=True)
        atr_col = [c for c in df.columns if c.startswith("ATRr_")]
        atr_val = df[atr_col[0]].iloc[-1] if atr_col else (df["High"].iloc[-1] - df["Low"].iloc[-1])

        current_price = df["Close"].iloc[-1]
        avg_vol_20 = df["Volume"].tail(20).mean()
        dollar_volume = current_price * avg_vol_20

        sl = max(0.01, current_price - (1.2 * atr_val))
        tp1 = current_price + (2.0 * atr_val)
        tp2 = current_price + (4.0 * atr_val)

        return {
            "valid": True,
            "current_price": float(current_price),
            "atr": float(atr_val),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
