import os
import logging
import urllib.parse
import requests
import asyncio
import yfinance as yf
from deep_translator import GoogleTranslator

# יבוא ספריות ה-Telegram ורכיבי התזמון (APScheduler)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# הגדרת לוגים
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# CHAT_ID יעד להתראות אוטומטיות
TARGET_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# רשימת מעקב לסריקה אוטומטית
WATCHLIST = ["AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", "GOOGL", "META", "MRNA", "PLTR"]

# ==========================================
# פונקציות עזר - תרגום, חישוב טכני ומקלדות
# ==========================================

def translate_to_hebrew(text: str) -> str:
    """מתרגם טקסט מאנגלית לעברית"""
    try:
        if not text or not text.strip():
            return text
        translated = GoogleTranslator(source='en', target='he').translate(text)
        return translated if translated else text
    except Exception as e:
        logger.error(f"שגיאת תרגום: {e}")
        return text

def build_action_keyboard(symbol: str) -> InlineKeyboardMarkup:
    """בונה מקלדת כפתורים מהירה מתחת להודעה"""
    keyboard = [
        [
            InlineKeyboardButton("📰 חדשות נוספות", callback_data=f"news_{symbol}"),
            InlineKeyboardButton("📊 ניתוח טכני", callback_data=f"tech_{symbol}")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def calculate_rsi(series, period=14):
    """חישוב RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# מנוע ניתוח מלא (טכני + חדשות + תוכנית מסחר)
# ==========================================

def generate_full_analysis_report(symbol: str) -> dict:
    """
    מנתח מנייה באופן מקיף ומפיק דוח טכני + דוח חדשות + תוכנית מסחר מלאה
    """
    symbol = symbol.upper()
    ticker = yf.Ticker(symbol)
    
    hist = ticker.history(period="1mo")
    if hist.empty or len(hist) < 20:
        return {"has_data": False, "symbol": symbol}

    current_price = float(hist['Close'].iloc[-1])
    volume_today = float(hist['Volume'].iloc[-1])
    avg_volume = float(hist['Volume'].mean())

    hist['RSI'] = calculate_rsi(hist['Close'])
    rsi_value = float(hist['RSI'].dropna().iloc[-1]) if not hist['RSI'].dropna().empty else 50.0
    sma20_value = float(hist['Close'].rolling(window=20).mean().iloc[-1])

    entry_price = round(current_price, 2)
    take_profit = round(current_price * 1.06, 2)
    stop_loss = round(current_price * 0.97, 2)
    risk_reward = "1:2.0"

    news_items = ticker.news if hasattr(ticker, "news") else []
    pos_keywords = ["breakthrough", "win", "surges", "soars", "beat", "growth", "deal", "approval", "fda", "buy", "upgrade"]
    neg_keywords = ["lawsuit", "investigation", "drop", "decline", "miss", "risk", "cut", "downgrade", "bankrupt", "loss"]

    pos_score, neg_score = 0, 0
    translated_titles = []

    for item in news_items[:3]:
        content = item.get("content", item)
        raw_title = content.get("title") or item.get("title", "")
        if raw_title:
            title_lower = raw_title.lower()
            for kw in pos_keywords:
                if kw in title_lower: pos_score += 1.5
            for kw in neg_keywords:
                if kw in title_lower: neg_score += 1.5
            translated_titles.append(translate_to_hebrew(raw_title))

    is_fomo_risk = rsi_value > 72
    fomo_text = "⚠️ **סיכון FOMO גבוה:** המנייה במתיחת יתר (RSI גבוה), מומלץ להמתין לממשו" if is_fomo_risk else "✅ **רמת סיכון FOMO:** תקינה לכניסה"

    news_summary_text = " • " + "\n • ".join(translated_titles) if translated_titles else "אין חדשות מהותיות כרגע"

    msg = (
        f"🚨 **התראת סורק אוטומטית - זיהוי מומנטום עבור {symbol}**\n"
        f"───────────────────────\n\n"
        f"📊 **1. דוח ניתוח טכני:**\n"
        f"• **מחיר נוכחי:** ${current_price:.2f}\n"
        f"• **נפח מסחר ביחס לממוצע:** {('חריג 🚀' if volume_today > avg_volume else 'רגיל')}\n"
        f"• **מדד מומנטום RSI (14):** {rsi_value:.1f}\n"
        f"• **ממוצע נע SMA20:** ${sma20_value:.2f}\n\n"
        f"🎯 **תוכנית מסחר וניהול סיכונים:**\n"
        f"• **כניסה (Entry):** ${entry_price}\n"
        f"• **יעד רווח (Take Profit):** ${take_profit} (+6%)\n"
        f"• **סטופ לוס (Stop Loss):** ${stop_loss} (-3%)\n"
        f"• **יחס סיכון/סיכוי (R:R):** {risk_reward}\n\n"
        f"───────────────────────\n"
        f"📰 **2. דוח חדשות וסנטימנט:**\n"
        f"{news_summary_text}\n\n"
        f"🛡️ {fomo_text}\n"
    )

    is_alert_triggered = (volume_today > avg_volume * 1.1) or (pos_score > neg_score) or (rsi_value > 60)

    return {
        "has_data": True,
        "symbol": symbol,
        "is_alert": is_alert_triggered,
        "formatted_message": msg
    }

# ==========================================
# סורק רקע אוטומטי (Background Job)
# ==========================================

async def auto_market_scanner_job(app: Application):
    """
    ריצה תקופתית ברקע שסורקת את מניות המעקב ומקפיצה התראות אוטומטיות לטלגרם
    """
    logger.info("🔍 מתחיל סריקת שוק אוטומטית ברקע...")
    
    global TARGET_CHAT_ID
    if not TARGET_CHAT_ID:
        logger.warning("לא הוגדר TELEGRAM_CHAT_ID. שלח /start לבוט בטלגרם כדי לקשר אותו אליך.")
        return

    for symbol in WATCHLIST:
        try:
            report = generate_full_analysis_report(symbol)
            if report.get("has_data") and report.get("is_alert"):
                logger.info(f"⚡ נמצאה קפיצה/טריגר במנייה {symbol}! שולח התראה...")
                
                reply_markup = build_action_keyboard(symbol)
                await app.bot.send_message(
                    chat_id=TARGET_CHAT_ID,
                    text=report["formatted_message"],
                    parse_mode="Markdown",
                    reply_markup=reply_markup,
                    disable_web_page_preview=True
                )
                await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"שגיאה בסריקת {symbol}: {e}")

# ==========================================
# פקודות ידניות בדיקה / טסטים
# ==========================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """הודעת פתיחה ושמירת ה-Chat ID של המשתמש לקבלת התראות"""
    global TARGET_CHAT_ID
    TARGET_CHAT_ID = str(update.effective_chat.id)
    
    welcome_text = (
        "👋 **ברוכים הבאים לסייען המסחר הפיננסי האוטומטי!**\n\n"
        "🤖 **הבוט מנטר כעת את השוק ברקע באופן אוטומטי.**\n"
        "ברגע שזוהו טריגרים (קפיצות נפח, מומנטום או חדשות חמות) - תקבל התראה קופצת בזמן אמת!\n\n"
        "💡 **פקודות סימולציה/טסטים ידניות:**\n"
        "• `/news MRNA` - ניתוח חדשותי נקודתי\n"
        "• `/scan` - הפעלת סריקה אקטיבית מידית על רשימת המעקב"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def handle_manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """הפעלה ידנית של הסורק לצורכי טסטים"""
    await update.message.reply_text("🔎 מפעיל סריקה ידנית יזומה על רשימת המעקב...")
    await auto_market_scanner_job(context.application)
    await update.message.reply_text("✅ הסריקה הסתיימה.")

async def handle_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """פקודה ידנית לטסט חדשותי"""
    if not context.args:
        await update.message.reply_text("אנא ציין סימול מנייה. דוגמה: `/news MRNA`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"🌐 מנתח ומזקק את הנתונים עבור {symbol}...")

    report = generate_full_analysis_report(symbol)
    if not report.get("has_data"):
        await update.message.reply_text("ℹ️ לא נמצאו נתונים זמינים עבור מנייה זו.")
        return

    reply_markup = build_action_keyboard(symbol)
    await update.message.reply_text(
        report["formatted_message"],
        parse_mode="Markdown",
        reply_markup=reply_markup,
        disable_web_page_preview=True
    )

# ==========================================
# הוספת ה-Scheduler בתוך ה-Event Loop של Telegram
# ==========================================

async def post_init(application: Application) -> None:
    """פונקציה זו רצה מיד לאחר שלולאת האירועים הופעלה בהצלחה"""
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        auto_market_scanner_job, 
        'interval', 
        minutes=15, 
        args=[application]
    )
    scheduler.start()
    logger.info("🤖 APScheduler הופעל בהצלחה בתוך לולאת האירועים הראשיות!")

def main():
    TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")

    application = Application.builder().token(TOKEN).post_init(post_init).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", handle_manual_scan))
    application.add_handler(CommandHandler("news", handle_news))

    logger.info("🤖 הבוט עולה לאוויר...")
    application.run_polling()

if __name__ == "__main__":
    main()
