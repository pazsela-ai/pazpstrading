import os
import logging
import asyncio
import threading
import yfinance as yf
from flask import Flask
from deep_translator import GoogleTranslator

# יבוא ספריות Telegram וסדרן עבודות
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
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

# שער חליפין משוער (USD to ILS) לחישובי המחשבון
USD_TO_ILS = 3.65

# ==========================================
# שרת Web זעיר לשמירה על Render פעיל (Health Check)
# ==========================================
web_app = Flask(__name__)

@web_app.route('/')
def health_check():
    return "Bot is alive and running!", 200

def run_web_server():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)

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
    """בונה מקלדת כפתורים אינטראקטיבית אחידה לכל הודעה"""
    tradingview_url = f"https://www.tradingview.com/chart/?symbol={symbol}"
    keyboard = [
        [
            InlineKeyboardButton("📈 צפייה בגרף", url=tradingview_url),
            InlineKeyboardButton("💰 חישוב עסקה", callback_data=f"calc_{symbol}")
        ],
        [
            InlineKeyboardButton("📰 חדשות בלבד", callback_data=f"news_{symbol}"),
            InlineKeyboardButton("📊 ניתוח טכני בלבד", callback_data=f"tech_{symbol}")
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
# מנוע ניתוח מרכזי - מייצר את הפורמט האחיד
# ==========================================

def generate_full_analysis_report(symbol: str) -> dict:
    """
    מנתח מנייה באופן מקיף ומפיק את תבנית התוצר האחידה והקבועה לכל סוגי הסריקות.
    """
    symbol = symbol.upper()
    ticker = yf.Ticker(symbol)
    
    try:
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

        # ניתוח חדשות מקיף (כללי, כלכלי, ביטחוני, טכנולוגי, רגולטורי ועוד)
        news_items = ticker.news if hasattr(ticker, "news") else []
        
        pos_keywords = [
            "breakthrough", "win", "surges", "soars", "beat", "growth", "deal", "approval", 
            "fda", "defense", "contract", "buy", "record", "profit", "expansion", "bullish", 
            "partnership", "patent", "upgrade", "success", "launch"
        ]
        neg_keywords = [
            "lawsuit", "investigation", "drop", "decline", "miss", "risk", "cut", "downgrade", 
            "bankrupt", "loss", "conflict", "sanctions", "bearish", "recall", "fine", "penalty", 
            "delay", "crisis", "war", "tariff"
        ]

        pos_score, neg_score = 0, 0
        translated_titles = []

        for item in news_items[:4]:
            content = item.get("content", item)
            raw_title = content.get("title") or item.get("title", "")
            if raw_title:
                title_lower = raw_title.lower()
                for kw in pos_keywords:
                    if kw in title_lower: pos_score += 1.5
                for kw in neg_keywords:
                    if kw in title_lower: neg_score += 1.5
                translated_titles.append(translate_to_hebrew(raw_title))

        # ניתוח והמלצה טכנית
        if rsi_value > 72:
            tech_recommendation = "🟡 המתנה / זהירות"
            tech_reasoning = f"המנייה במתיחת יתר (RSI={rsi_value:.1f}). קיימת סכנת תיקון כלפי מטה (FOMO)."
        elif current_price > sma20_value and volume_today > avg_volume:
            tech_recommendation = "🟢 מומלץ להשקיע (כניסה)"
            tech_reasoning = f"מחיר המנייה מעל ממוצע SMA20 (${sma20_value:.2f}) בליווי נפח מסחר חריג, המעיד על מומנטום חזק."
        else:
            tech_recommendation = "🟡 מעקב בלבד"
            tech_reasoning = "המדדים הפיננסיים ניטרליים. לא זוהתה פריצת מומנטום מובהקת כרגע."

        # ניתוח והמלצה חדשותית
        if pos_score > neg_score:
            news_recommendation = "🟢 מומלץ להשקיע (חיובי)"
            news_reasoning = "סורק החדשות זיהה סנטימנט חיובי בולט בדיווחים האחרונים (חוזים, דוחות, צמיחה או התפתחויות בשוק)."
        elif neg_score > pos_score:
            news_recommendation = "🔴 לא מומלץ (סיכון)"
            news_reasoning = "סורק החדשות זיהה סנטימנט שלילי או חשיפה לסיכונים (משפטיים, רגולטוריים, מאקרו-כלכליים וכד')."
        else:
            news_recommendation = "🟡 ניטרלי"
            news_reasoning = "אין כרגע אירועים חדשותיים או דיווחים בעלי השפעה דרמטית על כיוון המנייה."

        # המלצה סופית משוקללת
        if "🟢" in tech_recommendation or "🟢" in news_recommendation:
            final_recommendation = "🟢 מומלץ להיכנס להשקעה"
            final_reason = f"שילוב נתונים חיובי: {tech_reasoning} {news_reasoning}"
        else:
            final_recommendation = "🟡 להמתין / לעקוב"
            final_reason = f"{tech_reasoning} {news_reasoning}"

        entry_reason = f"מחיר השוק הנוכחי (${entry_price}) מהווה נקודת ייחוס מיטבית לכניסה במומנטום הנוכחי."
        stop_reason = f"נקבע ב-${stop_loss} (כ-3% מתחת לכניסה) כדי להגן על ההון ולמנוע הפסדים עמוקים במקרה של תנודה שלילית."

        news_summary_text = " • " + "\n • ".join(translated_titles) if translated_titles else "אין חדשות מהותיות כרגע"

        # **תבנית הודעה אחידה וקבועה לכל הסריקות**
        msg = (
            f"🚨 **דוח ניתוח מקיף עבור {symbol}**\n"
            f"───────────────────────\n\n"
            f"💡 **המלצה סופית:** {final_recommendation}\n"
            f"📌 **נימוק:** {final_reason}\n\n"
            f"📊 **ניתוח טכני:**\n"
            f"• מחיר: ${current_price:.2f} | RSI: {rsi_value:.1f} | SMA20: ${sma20_value:.2f}\n"
            f"• סטטוס טכני: {tech_recommendation}\n\n"
            f"📰 **חדשות וסנטימנט:**\n"
            f"{news_summary_text}\n"
            f"• סטטוס חדשותי: {news_recommendation}\n\n"
            f"🎯 **ניהול סיכונים ותוכנית מסחר:**\n"
            f"• **מחיר כניסה מומלץ:** ${entry_price}\n"
            f"  👈 *מדוע:* {entry_reason}\n"
            f"• **מחיר סטופ לוס (Stop Loss):** ${stop_loss}\n"
            f"  👈 *מדוע:* {stop_reason}\n"
            f"• **יעד רווח (Take Profit):** ${take_profit} (+6%)\n"
        )

        is_alert_triggered = (volume_today > avg_volume * 1.1) or (pos_score > neg_score) or (rsi_value > 60)

        return {
            "has_data": True,
            "symbol": symbol,
            "is_alert": is_alert_triggered,
            "formatted_message": msg,
            "current_price": current_price,
            "entry_price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "tech_recommendation": tech_recommendation,
            "tech_reasoning": tech_reasoning,
            "news_recommendation": news_recommendation,
            "news_reasoning": news_reasoning,
            "news_titles": translated_titles,
            "entry_reason": entry_reason,
            "stop_reason": stop_reason
        }
    except Exception as e:
        logger.error(f"שגיאה בהפקת דוח ל-{symbol}: {e}")
        return {"has_data": False, "symbol": symbol}

# ==========================================
# סורק רקע אוטומטי (Background Job)
# ==========================================

async def auto_market_scanner_job(app: Application):
    """סריקה אוטומטית ברקע – משתמשת בתבנית ההודעה האחידה"""
    global TARGET_CHAT_ID
    if not TARGET_CHAT_ID:
        return

    for symbol in WATCHLIST:
        try:
            report = generate_full_analysis_report(symbol)
            if report.get("has_data") and report.get("is_alert"):
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
# פקודות ידניות - משתמשות באותה תבנית אחידה
# ==========================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """הודעת פתיחה ושמירת Chat ID"""
    global TARGET_CHAT_ID
    TARGET_CHAT_ID = str(update.effective_chat.id)
    
    welcome_text = (
        "👋 **ברוכים הבאים לסייען המסחר הפיננסי האוטומטי!**\n\n"
        "🤖 **הבוט מנטר את השוק ברקע באופן אוטומטי ושולח התראות בזמן אמת.**\n\n"
        "💡 **סריקות ידניות (מציגות פורמט זהה להתראות האוטומטיות):**\n"
        "• `/tech TSLA` - סריקה טכנית מבוססת גרפים ומדדים\n"
        "• `/news TSLA` - סריקה חדשותית מקיפה\n"
        "• `/scan` - הפעלת סריקה ידנית יזומה על כל רשימת המעקב"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def handle_tech(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """סריקה ידנית טכנית - פורמט זהה לחלוטין להתראה האוטומטית"""
    if not context.args:
        await update.message.reply_text("אנא ציין סימול מנייה. דוגמה: `/tech TSLA`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"📊 מבצע סריקה וניתוח עבור {symbol}...")

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

async def handle_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """סריקה ידנית חדשותית - פורמט זהה לחלוטין להתראה האוטומטית"""
    if not context.args:
        await update.message.reply_text("אנא ציין סימול מנייה. דוגמה: `/news TSLA`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"📰 סורק חדשות וסנטימנט עבור {symbol}...")

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

async def handle_manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """הפעלה ידנית של הסורק האוטומטי על כל הרשימה"""
    global TARGET_CHAT_ID
    TARGET_CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text("🔎 מפעיל סריקה יזומה מקיפה על כל רשימת המעקב...")
    await auto_market_scanner_job(context.application)
    await update.message.reply_text("✅ הסריקה הידנית הושלמה.")

# ==========================================
# מטפל בלחיצות כפתורים (Callback Queries)
# ==========================================

async def button_click_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("calc_"):
        symbol = data.replace("calc_", "")
        context.user_data["awaiting_investment_for"] = symbol
        
        keyboard = [
            [
                InlineKeyboardButton("💵 דולר ($)", callback_data=f"curr_USD_{symbol}"),
                InlineKeyboardButton("₪ שקל (ILS)", callback_data=f"curr_ILS_{symbol}")
            ]
        ]
        await query.message.reply_text(
            f"💰 **מחשבון עסקה עבור {symbol}:**\nבאיזה מטבע תרצה להזמין/לחשב את ההשקעה?",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    elif data.startswith("curr_"):
        parts = data.split("_")
        currency = parts[1]
        symbol = parts[2]
        context.user_data["currency"] = currency
        context.user_data["awaiting_amount_for"] = symbol

        symbol_sign = "$" if currency == "USD" else "₪"
        await query.message.reply_text(
            f"כמה {symbol_sign} תרצה להשקיע במניית **{symbol}**?\nהשב להודעה זו עם הסכום בלבד (לדוגמה: `5000`).",
            parse_mode="Markdown"
        )

    elif data.startswith("news_"):
        symbol = data.replace("news_", "")
        report = generate_full_analysis_report(symbol)
        news_list = report.get("news_titles", [])
        news_text = " • " + "\n • ".join(news_list) if news_list else "אין חדשות מהותיות כרגע"
        await query.message.reply_text(
            f"📰 **חדשות מעודכנות עבור {symbol}:**\n\n{news_text}\n\n💡 **המלצה:** {report['news_recommendation']}\n📌 **נימוק:** {report['news_reasoning']}",
            parse_mode="Markdown"
        )

    elif data.startswith("tech_"):
        symbol = data.replace("tech_", "")
        report = generate_full_analysis_report(symbol)
        if report.get("has_data"):
            msg = (
                f"📊 **ניתוח טכני עבור {symbol}:**\n\n"
                f"💡 **המלצה:** {report['tech_recommendation']}\n"
                f"📌 **נימוק:** {report['tech_reasoning']}\n\n"
                f"• **מחיר:** ${report['current_price']:.2f}\n"
                f"• **כניסה מומלצת:** ${report['entry_price']}\n"
                f"• **סטופ לוס:** ${report['stop_loss']}"
            )
            await query.message.reply_text(msg, parse_mode="Markdown")

# ==========================================
# מטפל בהזנת סכום ההשקעה (מחשבון רווח/סיכון)
# ==========================================

async def handle_user_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = context.user_data.get("awaiting_amount_for")
    if not symbol:
        return

    text = update.message.text.strip()
    try:
        amount = float(text)
    except ValueError:
        await update.message.reply_text("⚠️ אנא הזן מספר תקין בלבד (לדוגמה: 5000).")
        return

    currency = context.user_data.get("currency", "USD")
    context.user_data["awaiting_amount_for"] = None

    report = generate_full_analysis_report(symbol)
    if not report.get("has_data"):
        await update.message.reply_text("שגיאה בשליפת נתוני המנייה לצורך חישוב.")
        return

    amount_in_usd = amount if currency == "USD" else amount / USD_TO_ILS
    entry_p = report["entry_price"]
    tp_p = report["take_profit"]
    sl_p = report["stop_loss"]

    num_shares = amount_in_usd / entry_p
    expected_profit_usd = (tp_p - entry_p) * num_shares
    expected_risk_usd = (entry_p - sl_p) * num_shares

    if currency == "ILS":
        expected_profit_final = f"₪{expected_profit_usd * USD_TO_ILS:.2f} (${expected_profit_usd:.2f})"
        expected_risk_final = f"₪{expected_risk_usd * USD_TO_ILS:.2f} (${expected_risk_usd:.2f})"
        input_display = f"₪{amount:,.2f}"
    else:
        expected_profit_final = f"${expected_profit_usd:.2f}"
        expected_risk_final = f"${expected_risk_usd:.2f}"
        input_display = f"${amount:,.2f}"

    calc_msg = (
        f"📊 **ניתוח עסקה מותאם אישית עבור {symbol}**\n"
        f"───────────────────────\n"
        f"💵 **סכום השקעה מוזן:** {input_display}\n"
        f"📦 **כמות מניות מוערכת:** {num_shares:.2f} מניות\n\n"
        f"🎯 **תחזית תוצאות:**\n"
        f"📈 **רווח צפוי (Take Profit - 6%+):** {expected_profit_final}\n"
        f"🛡️ **סיכון מרבי (Stop Loss - 3%-):** {expected_risk_final}\n\n"
        f"📌 **המלצה משוקללת:** {report['tech_recommendation']}"
    )

    tradingview_url = f"https://www.tradingview.com/chart/?symbol={symbol}"
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("📈 צפייה בגרף ב-TradingView", url=tradingview_url)]])

    await update.message.reply_text(calc_msg, parse_mode="Markdown", reply_markup=keyboard)

# ==========================================
# הפעלת הבוט וה-Scheduler
# ==========================================

async def post_init(application: Application) -> None:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        auto_market_scanner_job, 
        'interval', 
        minutes=15, 
        args=[application]
    )
    scheduler.start()
    logger.info("🤖 APScheduler הופעל בהצלחה!")

def main():
    threading.Thread(target=run_web_server, daemon=True).start()

    TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")

    application = Application.builder().token(TOKEN).post_init(post_init).build()

    # רישום פקודות
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", handle_manual_scan))
    application.add_handler(CommandHandler("news", handle_news))
    application.add_handler(CommandHandler("tech", handle_tech))
    
    # רישום כפתורים ותשובות טקסט של המשתמש
    application.add_handler(CallbackQueryHandler(button_click_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_text_input))

    logger.info("🤖 הבוט עולה לאוויר...")
    application.run_polling()

if __name__ == "__main__":
    main()
