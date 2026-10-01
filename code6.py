import os
import requests
import pandas as pd
import pandas_ta as ta
from textblob import TextBlob
from flask import Flask, request, jsonify
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio
import threading

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "YOUR_NEWS_API_KEY_HERE")
PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__)

telegram_app = None

# ------------------------------------------------------------------
# FUNCTIONS & LOGIC
# ------------------------------------------------------------------
def fetch_crypto_data(symbol="BTCUSDT", interval="1h", limit=100):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        res = requests.get(url, timeout=10)
        data = res.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['close'] = df['close'].astype(float)
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        latest = df.iloc[-1]
        rsi_val = round(latest['RSI'], 2) if not pd.isna(latest['RSI']) else 50
        
        tech_signal = "BUY" if rsi_val < 30 else ("SELL" if rsi_val > 70 else "NEUTRAL")
            
        return {"symbol": symbol, "close": latest['close'], "rsi": rsi_val, "signal": tech_signal}
    except Exception as e:
        print(f"Error fetching technical data: {e}")
        return None

def fetch_sentiment_analysis(query="crypto"):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language=en"
        res = requests.get(url, timeout=10)
        data = res.json()
        
        if data.get("status") != "ok" or not data.get("articles"):
            return {"score": 0, "label": "NEUTRAL", "articles_count": 0}
            
        articles = data["articles"][:10]
        total_polarity = sum(TextBlob((art.get("title") or "") + " " + (art.get("description") or "")).sentiment.polarity for art in articles)
        avg_polarity = total_polarity / len(articles)
        
        sentiment_label = "BULLISH" if avg_polarity > 0.05 else ("BEARISH" if avg_polarity < -0.05 else "NEUTRAL")
        return {"score": round(avg_polarity, 3), "label": sentiment_label, "articles_count": len(articles)}
    except Exception as e:
        print(f"Error fetching sentiment: {e}")
        return {"score": 0, "label": "NEUTRAL", "articles_count": 0}

def generate_combined_signal(symbol="BTCUSDT"):
    tech = fetch_crypto_data(symbol)
    sent = fetch_sentiment_analysis()
    
    if not tech:
        return "⚠️ שגיאה בשליפת נתונים טכניים."
        
    final_signal = "HOLD / NEUTRAL"
    if tech['signal'] == "BUY" and sent['label'] == "BULLISH":
        final_signal = "STRONG BUY 🚀"
    elif tech['signal'] == "BUY" or sent['label'] == "BULLISH":
        final_signal = "WEAK BUY 📈"
    elif tech['signal'] == "SELL" and sent['label'] == "BEARISH":
        final_signal = "STRONG SELL 📉"
    elif tech['signal'] == "SELL" or sent['label'] == "BEARISH":
        final_signal = "WEAK SELL ⚠️"
        
    return (
        f"📊 **ניתוח משולב עבור {symbol}**\n\n"
        f"💵 מחיר: `{tech['close']}`\n"
        f"📈 RSI: `{tech['rsi']}` ({tech['signal']})\n"
        f"📰 סנטימנט: `{sent['label']}` (ציון: {sent['score']})\n\n"
        f"🎯 **איתות סופי:** **{final_signal}**"
    )

# ------------------------------------------------------------------
# TELEGRAM HANDLERS
# ------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("שלום! הבוט מחובר ופעיל.\nפקודות: /status, /test_tech, /test_news")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ הבוט פועל כסדרו ומחובר לשרת!")

async def test_tech_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("מחשב ניתוח טכני וסנטימנט...")
    report = generate_combined_signal("BTCUSDT")
    await update.message.reply_text(report, parse_mode='Markdown')

async def test_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("בודק סנטימנט חדשות...")
    sent = fetch_sentiment_analysis()
    msg = f"📰 **סנטימנט חדשות:** **{sent['label']}** (ציון: `{sent['score']}`)"
    await update.message.reply_text(msg, parse_mode='Markdown')

# ------------------------------------------------------------------
# FLASK ROUTES
# ------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return "Trading Bot is Active!", 200

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json or {}
    msg = data.get("message", "התקבלה התראה מ-TradingView!")
    if telegram_app and TELEGRAM_CHAT_ID:
        asyncio.run(telegram_app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🔔 **Webhook:**\n{msg}"))
    return jsonify({"status": "success"}), 200

# ------------------------------------------------------------------
# STARTUP LOGIC FOR GUNICORN & RENDER
# ------------------------------------------------------------------
def start_bot():
    global telegram_app
    try:
        print("Starting Telegram Bot Polling...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        telegram_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        telegram_app.add_handler(CommandHandler("start", start_command))
        telegram_app.add_handler(CommandHandler("status", status_command))
        telegram_app.add_handler(CommandHandler("test_tech", test_tech_command))
        telegram_app.add_handler(CommandHandler("test_news", test_news_command))

        telegram_app.run_polling(drop_pending_updates=True, stop_signals=None, close_loop=False)
    except Exception as e:
        print(f"Error starting Telegram bot: {e}")

# הפעלת האזנת הבוט מיד בעת טעינת המודול ב-Gunicorn
bot_thread = threading.Thread(target=start_bot, daemon=True)
bot_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
