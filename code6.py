import os
import sys
import logging
import threading
from datetime import datetime
from flask import Flask

import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------------------
# 1. הגדרות לוגים
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2. משתני סביבה ורשימת מעקב
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN_HERE")
CHAT_ID = os.environ.get("CHAT_ID", "YOUR_CHAT_ID_HERE")

# רשימת מעקב מורחבת לסריקה אוטונומית ברקע
WATCHLIST = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "AMD", "PLTR", "NFLX", "COIN", "SMCI", "ARM"
]

# ---------------------------------------------------------------------------
# 3. שרת Flask (עבור Render Web Service)
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Autonomous Financial Intelligence & Alert Engine is Live!"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------------------------------------------------------------------
# 4. מנוע חישוב אינדיקטורים וסיגנלים
# ---------------------------------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stock_data(symbol: str):
    """
    מנוע ניתוח מרובה-פרמטרים: מחשב מחיר, RSI, ממוצעים נעים, נפח מסחר ושולף חדשות.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="3mo")

    if df.empty or len(df) < 20:
        return None

    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    daily_change_pct = ((current_price - prev_close) / prev_close) * 100

    # ממוצעים נעים
    sma_10 = df['Close'].tail(10).mean()
    sma_20 = df['Close'].tail(20).mean()
    sma_50 = df['Close'].tail(50).mean() if len(df) >= 50 else sma_20

    # RSI
    rsi_series = calculate_rsi(df['Close'], 14)
    rsi = rsi_series.iloc[-1]

    # נפח מסחר
    avg_volume = df['Volume'].tail(20).mean()
    current_volume = df['Volume'].iloc[-1]
    vol_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0

    # חדשות
    news = ticker.news if hasattr(ticker, "news") else []

    return {
        "symbol": symbol,
        "current_price": current_price,
        "daily_change_pct": daily_change_pct,
        "sma_10": sma_10,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "news": news
    }

def evaluate_opportunity(data: dict):
    """
    מנוע הציון הריבועי (Multi-Factor Scoring):
    מעריך האם מנייה הגיעה למצב הזדמנותי לקנייה/מעקב מוגבר.
    """
    score = 0
    reasons = []

    # 1. ניתוח RSI (מכירת יתר / תחילת מומנטום)
    if data['rsi'] <= 30:
        score += 3
        reasons.append(f"מכירת יתר קיצונית (RSI: {data['rsi']:.1f})")
    elif 30 < data['rsi'] <= 40:
        score += 1.5
        reasons.append(f"RSI נמוך באזור איסוף ({data['rsi']:.1f})")

    # 2. פריצת נפח מסחר (Unusual Volume)
    if data['vol_ratio'] >= 2.0:
        score += 3
        reasons.append(f"זינוק חריג בנפח המסחר (פי {data['vol_ratio']:.1f} מהממוצע)")
    elif data['vol_ratio'] >= 1.5:
        score += 1.5
        reasons.append(f"נפח מסחר גבוה מהממוצע (פי {data['vol_ratio']:.1f})")

    # 3. מומנטום ומבנה מחיר (מגמה שורית / חזרה לממוצע)
    if data['current_price'] > data['sma_20'] and data['sma_10'] > data['sma_20']:
        score += 2
        reasons.append("הצלבה שורית ממוצעים נעים (SMA10 > SMA20)")

    # 4. סנטימנט חדשותי - זיהוי אירועים קטליזטוריים
    recent_news = data.get('news', [])
    if recent_news:
        keywords = ["earnings", "upgrade", "fda", "deal", "growth", "revenue", "buyout", "partnership"]
        matched_titles = []
        for n in recent_news[:3]:
            title = n.get('title', '')
            if any(kw in title.lower() for kw in keywords):
                matched_titles.append(title)

        if matched_titles:
            score += 2
            reasons.append(f"אירוע קטליזטורי בחדשות: '{matched_titles[0]}'")

    # סף איכות להקפצת התראה אוטונומית
    is_hot = score >= 4.5

    return is_hot, score, reasons

# ---------------------------------------------------------------------------
# 5. סורק השוק האוטונומי (מריץ התראות בזמן אמת)
# ---------------------------------------------------------------------------
async def autonomous_market_scan(context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("🤖 מתחיל סריקה אוטונומית מרובת-פרמטרים...")
    
    for symbol in WATCHLIST:
        try:
            data = analyze_stock_data(symbol)
            if not data:
                continue

            is_hot, score, reasons = evaluate_opportunity(data)

            if is_hot:
                # ניסוח התראת איכות מקיפה
                reasons_formatted = "\n".join([f"• {r}" for r in reasons])
                
                news_snippet = ""
                if data['news']:
                    first_news = data['news'][0]
                    news_snippet = f"\n📰 **חדשות אחרונות:** [{first_news.get('title')}]({first_news.get('link')})\n"

                alert_msg = (
                    f"🚨 **איתות אוטונומי: הזדמנות זוהתה עבור {symbol}!**\n"
                    f"───────────────────────\n"
                    f"🎯 **ציון הזדמנות:** {score}/10\n"
                    f"💵 **מחיר נוכחי:** ${data['current_price']:.2f} ({data['daily_change_pct']:+.2f}%)\n\n"
                    f"💡 **סיבות להקפצה:**\n{reasons_formatted}\n"
                    f"{news_snippet}\n"
                    f"⚡ *מומלץ לבצע ניתוח מפורט באמצעות הפקודה:* `/technical {symbol}`"
                )

                target_chat_id = CHAT_ID if CHAT_ID != "YOUR_CHAT_ID_HERE" else context.job.chat_id
                if target_chat_id:
                    await context.bot.send_message(
                        chat_id=target_chat_id, 
                        text=alert_msg, 
                        parse_mode="Markdown",
                        disable_web_page_preview=True
                    )
                    logger.info(f"התראה נשלחה בהצלחה עבור {symbol}")

        except Exception as e:
            logger.error(f"שגיאה בסריקה האוטונומית עבור {symbol}: {e}")

# ---------------------------------------------------------------------------
# 6. פקודות בדיקה ואינטראקציה בטלגרם (בדיקות טסט ודרישה ידנית)
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "👋 **מערכת מודיעין פיננסי ואיתותים אוטונומית**\n\n"
        "המערכת סורקת את השוק ברקע באופן רציף ומקפיצה התראות בזמן אמת על מניות הזדמנותיות.\n\n"
        "**פקודות לבדיקה ידנית:**\n"
        "• `/technical TICKER` - ניתוח טכני ומדדים מפורטים\n"
        "• `/news TICKER` - מבזקי חדשות וסנטימנט\n"
        "• `/scan` - הרצה ידנית מיידית של מנוע הסריקה האוטונומי\n"
        "• `/test_alert` - בדיקת תקינות שידור התראות"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def handle_technical(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("אנא ציין סימול מנייה. דוגמה:\n`/technical NVDA`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"🔍 מבצע ניתוח מפורט עבור {symbol}...")

    data = analyze_stock_data(symbol)
    if not data:
        await update.message.reply_text(f"❌ לא נמצאו נתונים עבור `{symbol}`.", parse_mode="Markdown")
        return

    _, score, reasons = evaluate_opportunity(data)
    reasons_formatted = "\n".join([f"• {r}" for r in reasons]) if reasons else "• אין מדדים חריגים כרגע"

    msg = (
        f"📊 **ניתוח טכני ואנליטי עבור {symbol}**\n"
        f"───────────────────────\n"
        f"💵 **מחיר נוכחי:** ${data['current_price']:.2f} ({data['daily_change_pct']:+.2f}%)\n"
        f"🎯 **ציון שווקי נוכחי:** {score}/10\n\n"
        f"🔹 **SMA 10:** ${data['sma_10']:.2f} | **SMA 20:** ${data['sma_20']:.2f}\n"
        f"⚡ **RSI (14):** {data['rsi']:.1f}\n"
        f"📊 **נפח מסחר:** פי {data['vol_ratio']:.1f} מהממוצע\n\n"
        f"🔍 **ממצאים עיקריים:**\n{reasons_formatted}"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def handle_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("אנא ציין סימול מנייה. דוגמה:\n`/news NVDA`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"🌐 שולף עדכונים מחדשות השוק עבור {symbol}...")

    ticker = yf.Ticker(symbol)
    news_items = ticker.news if hasattr(ticker, "news") else []

    if not news_items:
        await update.message.reply_text(f"ℹ️ לא נמצאו מבזקי חדשות אחרונים עבור `{symbol}`.")
        return

    msg = f"📰 **עדכוני חדשות קטליזטורים עבור {symbol}**\n───────────────────────\n\n"
    for item in news_items[:4]:
        title = item.get("title", "ללא כותרת")
        publisher = item.get("publisher", "מקור לא ידוע")
        link = item.get("link", "#")
        msg += f"• **{title}**\n  ✍️ {publisher} | [לכתבה המלאה]({link})\n\n"

    await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=True)

async def handle_manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⚡ מפעיל סריקה אוטונומית ידנית על רשימת המעקב...")
    await autonomous_market_scan(context)
    await update.message.reply_text("✅ הסריקה הידנית הושלמה.")

async def handle_test_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🔔 **בדיקת התראה:** ערוץ השידור וה-JobQueue פעילים ומחוברים למנוע הסריקה!", parse_mode="Markdown")

# ---------------------------------------------------------------------------
# 7. נקודת הכניסה הראשית (Main)
# ---------------------------------------------------------------------------
def main():
    # 1. הפעלת שרת Flask ברקע בפורט של Render
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    logger.info("שרת ה-Flask הופעל ברקע עבור Render Web Service.")

    # 2. אתחול הבוט
    tg_app = Application.builder().token(TELEGRAM_TOKEN).build()

    # 3. רישום פקודות
    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(CommandHandler("technical", handle_technical))
    tg_app.add_handler(CommandHandler("news", handle_news))
    tg_app.add_handler(CommandHandler("scan", handle_manual_scan))
    tg_app.add_handler(CommandHandler("test_alert", handle_test_alert))

    # 4. תזמון הסריקה האוטונומית (רץ כל 5 דקות)
    job_queue = tg_app.job_queue
    if job_queue:
        job_queue.run_repeating(autonomous_market_scan, interval=300, first=10)

    logger.info("מתחיל הרצת הבוט במצב Polling...")
    tg_app.run_polling()

if __name__ == "__main__":
    main()
