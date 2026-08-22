import os
import sys
import logging
import threading
from datetime import datetime
from flask import Flask

import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    ContextTypes, 
    CallbackQueryHandler, 
    MessageHandler, 
    filters, 
    ConversationHandler
)

import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------------------
# 1. הגדרות לוגים ומצבי שיחה
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

WAITING_FOR_PRICE = 1

# ---------------------------------------------------------------------------
# 2. משתני סביבה ורשימת מעקב
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN_HERE")
CHAT_ID = os.environ.get("CHAT_ID", "YOUR_CHAT_ID_HERE")

WATCHLIST = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "AMD", "PLTR", "NFLX", "COIN", "SMCI", "ARM"
]

# ---------------------------------------------------------------------------
# 3. שרת Flask עבור Render
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Autonomous Trading Bot with Reasoning & TradingView Integration is Live!"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ---------------------------------------------------------------------------
# 4. מנוע חישוב ואינדיקטורים
# ---------------------------------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stock_data(symbol: str):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="3mo")

    if df.empty or len(df) < 20:
        return None

    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    daily_change_pct = ((current_price - prev_close) / prev_close) * 100

    sma_10 = df['Close'].tail(10).mean()
    sma_20 = df['Close'].tail(20).mean()
    sma_50 = df['Close'].tail(50).mean() if len(df) >= 50 else sma_20

    rsi_series = calculate_rsi(df['Close'], 14)
    rsi = rsi_series.iloc[-1]

    avg_volume = df['Volume'].tail(20).mean()
    current_volume = df['Volume'].iloc[-1]
    vol_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0

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
    score = 0
    reasons = []

    if data['rsi'] <= 30:
        score += 3
        reasons.append(f"מכירת יתר קיצונית ב-RSI ({data['rsi']:.1f}) – מתאים לאיסוף מחירים")
    elif 30 < data['rsi'] <= 40:
        score += 1.5
        reasons.append(f"RSI באזור נוח לקנייה ({data['rsi']:.1f})")

    if data['vol_ratio'] >= 2.0:
        score += 3
        reasons.append(f"זינוק חריג בנפח המסחר (פי {data['vol_ratio']:.1f} מהממוצע) – מעיד על כניסת מוסדיים")
    elif data['vol_ratio'] >= 1.5:
        score += 1.5
        reasons.append(f"נפח מסחר גבוה מהרגיל (פי {data['vol_ratio']:.1f})")

    if data['current_price'] > data['sma_20'] and data['sma_10'] > data['sma_20']:
        score += 2
        reasons.append("מבנה מחיר שורי: ממוצע 10 חוצה מעל ממוצע 20")

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
            reasons.append(f"קטליזטור חדשותי חיובי: '{matched_titles[0]}'")

    is_hot = score >= 4.5
    return is_hot, score, reasons

def build_action_keyboard(symbol: str):
    tradingview_url = f"https://www.tradingview.com/chart/?symbol={symbol}"
    keyboard = [
        [
            InlineKeyboardButton("📊 צפייה בגרף ב-TradingView", url=tradingview_url),
            InlineKeyboardButton("💼 בצע עסקה", callback_data=f"trade_{symbol}")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

# ---------------------------------------------------------------------------
# 5. פקודות טלגרם
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "👋 **מערכת מודיעין פיננסי ואיתותים אוטונומית**\n\n"
        "פקודות זמינות:\n"
        "• `/technical TICKER` - ניתוח טכני + המלצת השקעה מנומקת ומחיר כניסה\n"
        "• `/news TICKER` - חדשות + המלצה מנומקת לפי סנטימנט\n"
        "• `/scan` - הרצה ידנית מורחבת של הסורק\n"
        "• `/test_alert` - בדיקת תקינות התראות"
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

    # בניית המלצה + נימוק מפורט
    if score >= 4.5:
        recommendation = "✅ **שווה להשקיע!**"
        reasoning = (
            f"**נימוק להמלצה:** המנייה קיבלה ציון מגוון של {score}/10. "
            f"השילוב של האינדיקטורים מעיד על מומנטום חיובי ונפח מסחר תומך (" + 
            ", ".join(reasons) + ")."
        )
        entry_price = min(data['current_price'], data['sma_10'])
        entry_text = f"🎯 **מחיר כניסה מומלץ:** ${entry_price:.2f} (באזור ממוצע SMA10 / המחיר הנוכחי)"
    else:
        recommendation = "⏳ **לא מומלץ להשקיע כרגע.**"
        reasoning = (
            f"**נימוק להמלצה:** המנייה קיבלה ציון של {score}/10 בלבד. "
            f"כעת אין איתות קנייה חזק או מומנטום מספיק שמצדיק כניסה לעסקה בסיכון נמוך."
        )
        entry_text = "🎯 **מחיר כניסה מומלץ:** מומלץ להמתין לפריצה טכנית או לאיתות נוסף."

    msg = (
        f"📊 **ניתוח טכני עבור {symbol}**\n"
        f"───────────────────────\n"
        f"💵 **מחיר נוכחי:** ${data['current_price']:.2f} ({data['daily_change_pct']:+.2f}%)\n"
        f"🎯 **ציון שווקי:** {score}/10\n\n"
        f"🔹 **SMA 10:** ${data['sma_10']:.2f} | **SMA 20:** ${data['sma_20']:.2f}\n"
        f"⚡ **RSI (14):** {data['rsi']:.1f}\n"
        f"📊 **נפח מסחר:** פי {data['vol_ratio']:.1f} מהממוצע\n\n"
        f"💡 **המלצה:** {recommendation}\n"
        f"🧠 {reasoning}\n\n"
        f"{entry_text}"
    )

    reply_markup = build_action_keyboard(symbol)
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=reply_markup)

async def handle_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("אנא ציין סימול מנייה. דוגמה:\n`/news NVDA`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"🌐 שולף עדכונים עבור {symbol}...")

    ticker = yf.Ticker(symbol)
    news_items = ticker.news if hasattr(ticker, "news") else []

    if not news_items:
        await update.message.reply_text(
            f"ℹ️ **אין חדשות מעניינות או מבזקים חדשים כרגע עבור `{symbol}`.**", 
            parse_mode="Markdown"
        )
        return

    msg = f"📰 **עדכוני חדשות עבור {symbol}**\n───────────────────────\n\n"
    catalyst_titles = []

    for item in news_items[:4]:
        title = item.get("title", "ללא כותרת")
        publisher = item.get("publisher", "מקור לא ידוע")
        link = item.get("link", "#")
        msg += f"• **{title}**\n  ✍️ {publisher} | [לכתבה המלאה]({link})\n\n"
        
        if any(kw in title.lower() for kw in ["earnings", "upgrade", "growth", "buy", "deal", "fda"]):
            catalyst_titles.append(title)

    msg += "💡 **המלצה ונימוק סנטימנטלי:**\n"
    if catalyst_titles:
        msg += (
            f"✅ **שווה לשקול השקעה!**\n"
            f"**נימוק:** זוהו כתבות קטליזטוריות חיוביות (למשל: '{catalyst_titles[0]}') "
            f"עשויות להוות טריגר לעליית מחירים בטווח הקרוב."
        )
    else:
        msg += (
            f"ℹ️ **לא מומלץ להיכנס רק על בסיס החדשות.**\n"
            f"**נימוק:** הדיווחים האחרונים ניטרליים ואינם כוללים אירוע קטליזטורי מובהק."
        )

    reply_markup = build_action_keyboard(symbol)
    await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=True, reply_markup=reply_markup)

# ---------------------------------------------------------------------------
# 6. מנגנון תהליך "בצע עסקה" וחישוב Stop-Loss
# ---------------------------------------------------------------------------
async def start_trade_flow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    symbol = query.data.split("_")[1]
    context.user_data['trade_symbol'] = symbol

    await query.message.reply_text(
        f"💼 **ביצוע עסקה עבור {symbol}**\n"
        f"אנא הזיני את המחיר המבוקש שבו את מעוניינת להיכנס להשקעה (לדוגמה: `125.5`):",
        parse_mode="Markdown"
    )
    return WAITING_FOR_PRICE

async def process_trade_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    symbol = context.user_data.get('trade_symbol', 'UNKNOWN')

    try:
        entry_price = float(text)
    except ValueError:
        await update.message.reply_text("❌ מחיר לא תקין. אנא הזיני מספר בלבד (למשל: 120.5):")
        return WAITING_FOR_PRICE

    stop_loss = entry_price * 0.95
    take_profit = entry_price * 1.10
    risk_per_share = entry_price - stop_loss

    msg = (
        f"📐 **תוכנית מסחר מחושבת עבור {symbol}**\n"
        f"───────────────────────\n"
        f"💵 **מחיר כניסה מבוקש:** ${entry_price:.2f}\n"
        f"🛑 **סטופ-לוס מומלץ (Stop-Loss):** ${stop_loss:.2f} (-5%)\n"
        f"🎯 **יעד רווח מומלץ (Take-Profit):** ${take_profit:.2f} (+10%)\n"
        f"⚠️ **סיכון למנייה:** ${risk_per_share:.2f}\n\n"
        f"💡 *נימוק ניהול סיכונים: עצר-הפסד ב-5% שומר על יחס סיכון/סיכוי של 1:2 לפחות.*"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")
    return ConversationHandler.END

async def cancel_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("התהליך בוטל.")
    return ConversationHandler.END

# ---------------------------------------------------------------------------
# 7. סריקה אוטונומית ברקע
# ---------------------------------------------------------------------------
async def autonomous_market_scan(context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("🤖 מתחיל סריקה אוטונומית ברקע...")
    
    for symbol in WATCHLIST:
        try:
            data = analyze_stock_data(symbol)
            if not data:
                continue

            is_hot, score, reasons = evaluate_opportunity(data)

            if is_hot:
                reasons_formatted = "\n".join([f"• {r}" for r in reasons])
                
                news_snippet = ""
                if data['news']:
                    first_news = data['news'][0]
                    news_snippet = f"\n📰 **חדשות:** [{first_news.get('title')}]({first_news.get('link')})\n"

                entry_price = min(data['current_price'], data['sma_10'])

                alert_msg = (
                    f"🚨 **איתות אוטונומי: הזדמנות זוהתה עבור {symbol}!**\n"
                    f"───────────────────────\n"
                    f"🎯 **ציון הזדמנות:** {score}/10\n"
                    f"💵 **מחיר נוכחי:** ${data['current_price']:.2f} ({data['daily_change_pct']:+.2f}%)\n"
                    f"🎯 **מחיר כניסה מומלץ:** ${entry_price:.2f}\n\n"
                    f"💡 **נימוקים וסיבות להקפצה:**\n{reasons_formatted}\n"
                    f"{news_snippet}"
                )

                reply_markup = build_action_keyboard(symbol)
                target_chat_id = CHAT_ID if CHAT_ID != "YOUR_CHAT_ID_HERE" else context.job.chat_id

                if target_chat_id:
                    await context.bot.send_message(
                        chat_id=target_chat_id, 
                        text=alert_msg, 
                        parse_mode="Markdown",
                        disable_web_page_preview=True,
                        reply_markup=reply_markup
                    )
        except Exception as e:
            logger.error(f"שגיאה בסריקה האוטונומית עבור {symbol}: {e}")

async def handle_manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⚡ מפעיל סריקה אוטונומית ידנית...")
    await autonomous_market_scan(context)
    await update.message.reply_text("✅ הסריקה הידנית הושלמה.")

async def handle_test_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🔔 **בדיקת התראה:** המערכת פועלת כהלכה!", parse_mode="Markdown")

# ---------------------------------------------------------------------------
# 8. הרצה ראשית (Main)
# ---------------------------------------------------------------------------
def main():
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    tg_app = Application.builder().token(TELEGRAM_TOKEN).build()

    trade_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(start_trade_flow, pattern="^trade_")],
        states={
            WAITING_FOR_PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_trade_price)]
        },
        fallbacks=[CommandHandler("cancel", cancel_trade)]
    )

    tg_app.add_handler(trade_conv)
    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(CommandHandler("technical", handle_technical))
    tg_app.add_handler(CommandHandler("news", handle_news))
    tg_app.add_handler(CommandHandler("scan", handle_manual_scan))
    tg_app.add_handler(CommandHandler("test_alert", handle_test_alert))

    job_queue = tg_app.job_queue
    if job_queue:
        job_queue.run_repeating(autonomous_market_scan, interval=300, first=10)

    logger.info("הבוט פועל ברקע...")
    tg_app.run_polling()

if __name__ == "__main__":
    main()
