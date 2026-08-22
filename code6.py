import os
import asyncio
import logging
import requests
import feedparser
import yfinance as yf
import pandas as pd
from datetime import datetime
from flask import Flask
from threading import Thread
from finvizfinance.screener.technical import Technical
from google import genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# הגדרת לוגים
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# מפתחות סביבה
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# אתחול Gemini API
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# זיכרון זמני למעקב ומחשבון
user_trade_state = {}

# ---------------------------------------------------------
# 1. מנוע ניתוח טכני דינמי
# ---------------------------------------------------------

def run_technical_screener():
    try:
        f_screener = Technical()
        filters_dict = {
            '20-Day Simple Moving Average': 'Price crossed 20SMA above',
            'Relative Strength Index (14)': 'Overbought (60) or below',
            'Current Volume': 'Over 500K'
        }
        f_screener.set_filter(filters_dict=filters_dict)
        df = f_screener.screener_view()
        
        if df is None or df.empty:
            return []

        candidates = df['Ticker'].tolist()[:10]
        valid_signals = []

        for ticker in candidates:
            signal = analyze_technical_setup(ticker)
            if signal:
                valid_signals.append(signal)

        return valid_signals
    except Exception as e:
        logger.error(f"שגיאה בסורק הטכני: {e}")
        return []

def analyze_technical_setup(ticker_symbol: str):
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="3mo")
        if len(df) < 50:
            return None

        current_price = df['Close'].iloc[-1]
        
        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))

        # SMA 20
        sma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        dist_from_sma20 = ((current_price - sma20) / sma20) * 100

        # Volume
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        curr_volume = df['Volume'].iloc[-1]
        vol_ratio = curr_volume / avg_volume if avg_volume > 0 else 1.0

        # סינון לרכבת שנסעה
        if rsi >= 68 or dist_from_sma20 > 5.0 or vol_ratio < 1.1:
            return None

        is_warning = rsi >= 60

        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        stop_loss = max(current_price - (1.5 * atr), current_price * 0.95)
        take_profit = current_price + (3.0 * atr)

        risk_per_share = current_price - stop_loss
        reward_per_share = take_profit - current_price
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        return {
            "ticker": ticker_symbol,
            "price": round(current_price, 2),
            "rsi": round(rsi, 1),
            "dist_sma20": round(dist_from_sma20, 1),
            "vol_ratio": round(vol_ratio, 1),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "rr_ratio": round(rr_ratio, 1),
            "is_warning": is_warning
        }
    except Exception as e:
        logger.error(f"שגיאה בניתוח עבור {ticker_symbol}: {e}")
        return None

# ---------------------------------------------------------
# 2. מנוע ניתוח חדשותי
# ---------------------------------------------------------

def fetch_rss_news():
    urls = [
        "https://news.google.com/rss/search?q=stock+market+breakout&hl=en-US&gl=US&ceid=US:en",
        "https://finance.yahoo.com/rss/headline?s=AAPL,NVDA,TSLA,AMD,MSFT"
    ]
    news_items = []
    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            news_items.append({"title": entry.title, "link": entry.link})
    return news_items

# ---------------------------------------------------------
# 3. תזמון וטיפול בהודעות
# ---------------------------------------------------------

async def send_telegram_msg(bot, text, reply_markup=None):
    if TELEGRAM_CHAT_ID:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown", reply_markup=reply_markup)

async def periodic_news_scan(context: ContextTypes.DEFAULT_TYPE):
    items = fetch_rss_news()
    if not items or not ai_client:
        return
    try:
        prompt = f"מצא ידיעה דרמטית אחת בעלת פוטנציאל להזזת מניה: {items}. החזר JSON עם ticker, summary, action."
        response = ai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    except Exception as e:
        logger.error(f"שגיאה בסריקת חדשות: {e}")

async def periodic_technical_scan(context: ContextTypes.DEFAULT_TYPE):
    signals = run_technical_screener()
    for sig in signals:
        header = "⚠️ **[אזהרת מומנטום - רגע לפני שהרכבת נוסעת]**" if sig['is_warning'] else "📊 **[אות טכני - תחילת תנועה]**"
        status_note = "מתקרב לקצה טווח הכניסה! חלון הזדמנות אחרון." if sig['is_warning'] else "נקודת כניסה אופטימלית בקו הזינוק."

        msg = (
            f"{header}\n"
            f"**מניה:** `{sig['ticker']}` | **מחיר:** ${sig['price']}\n\n"
            f"🚥 **בדיקת תזמון:**\n"
            f"• **RSI (14):** `{sig['rsi']}` 🟢\n"
            f"• **מרחק מממוצע 20:** `{sig['dist_sma20']}%+` 🟢\n"
            f"• **סטטוס:** {status_note}\n\n"
            f"💡 **למה שווה לבדוק?**\n"
            f"פריצת התנגדות בנפח מסחר ער (פי `{sig['vol_ratio']}` מהממוצע).\n\n"
            f"🛡️ **פרמטרים לעסקה:**\n"
            f"• **כניסה:** ${sig['price']}\n"
            f"• **סטופ לוס:** ${sig['stop_loss']}\n"
            f"• **יעד רווח:** ${sig['take_profit']}\n"
            f"• **יחס סיכוי/סיכון:** `1 : {sig['rr_ratio']}` 🟢"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("💰 אני רוצה לבצע עסקה", callback_data=f"calc_{sig['ticker']}_{sig['price']}_{sig['stop_loss']}_{sig['take_profit']}")],
            [InlineKeyboardButton("🔗 גרף ב-TradingView", url=f"https://www.tradingview.com/symbols/{sig['ticker']}")]
        ])
        await send_telegram_msg(context.bot, msg, reply_markup=keyboard)

# ---------------------------------------------------------
# 4. מחשבון עסקאות אינטראקטיבי
# ---------------------------------------------------------

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data.split("_")
    if data[0] == "calc":
        ticker, price, stop, target = data[1], float(data[2]), float(data[3]), float(data[4])
        user_trade_state[query.from_user.id] = {
            "ticker": ticker, "price": price, "stop": stop, "target": target
        }
        await query.message.reply_text(f"רשמי כעת את הסכום הכולל ב-$ שתרצי להשקיע במניית {ticker}:")

async def handle_user_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in user_trade_state:
        try:
            budget = float(update.message.text.replace("$", "").strip())
            trade = user_trade_state.pop(user_id)
            
            price = trade['price']
            stop = trade['stop']
            target = trade['target']
            
            shares = int(budget // price) if price <= budget else 1
            total_cost = round(shares * price, 2)
            max_loss = round(shares * (price - stop), 2)
            max_gain = round(shares * (target - price), 2)
            rr = round(max_gain / max_loss, 1) if max_loss > 0 else 0
            
            msg = (
                f"🎯 **[תוכנית ביצוע מותאמת אישית]**\n"
                f"**מניה:** `{trade['ticker']}` | **תקציב שהזנת:** ${budget:,.0f}\n\n"
                f"📊 **תוכנית קנייה מדויקת:**\n"
                f"• **כמות מניות לקנייה:** `{shares}` מניות\n"
                f"• **סכום השקעה בפועל:** ${total_cost:,.2f}\n"
                f"• **מחיר סטופ לוס:** ${stop}\n"
                f"• **מחיר יעד רווח:** ${target}\n\n"
                f"🛡️ **ניהול סיכונים בשורה התחתונה:**\n"
                f"• **הפסד מקסימלי בעסקה:** **${max_loss}** 🛑\n"
                f"• **רווח פוטנציאלי:** **${max_gain}** 🎯\n"
                f"• **יחס סיכוי/סיכון:** `1 : {rr}` 🟢"
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
        except ValueError:
            await update.message.reply_text("אנא הזיני מספר תקין (למשל: 1500).")

# ---------------------------------------------------------
# 5. פקודות ושרת Flask
# ---------------------------------------------------------

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("היי! הבוט מחובר ומנטר ניתוח חדשותי וטכני בזמן אמת.")

async def test_tech_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("מריץ סריקה טכנית ידנית כעת...")
    await periodic_technical_scan(context)

app_flask = Flask(__name__)

@app_flask.route('/')
def home():
    return "Trader Bot Alive", 200

def run_flask():
    app_flask.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

def main():
    Thread(target=run_flask, daemon=True).start()

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN is missing")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("testtech", test_tech_cmd))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_input))

    job_queue = application.job_queue
    job_queue.run_repeating(periodic_news_scan, interval=600, first=10)
    job_queue.run_repeating(periodic_technical_scan, interval=1800, first=20)

    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
