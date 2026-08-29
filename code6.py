import os
import threading
import time
from datetime import datetime, timedelta
from flask import Flask
import telebot
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from deep_translator import GoogleTranslator

# ------------------------------------------------------------------------------
# 1. הגדרות ומשתני סביבה
# ------------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
FINNHUB_KEY = os.environ.get('FINNHUB_KEY')
ALPHAVANTAGE_KEY = os.environ.get('ALPHAVANTAGE_KEY', '')

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# תיקון קוד השפה לעברית עבור גוגל תרגום ('iw')
translator = GoogleTranslator(source='auto', target='iw')
app = Flask(__name__)

# ------------------------------------------------------------------------------
# 2. פונקציות שליפת נתונים מתוקנות (מחיר + חדשות מגובות)
# ------------------------------------------------------------------------------
def get_stock_data(ticker_symbol):
    """
    שליפת נתוני מניה ומחיר עדכני מתוך היסטוריית המחירים ישירות,
    כדי למנוע תקלות NaN ב- regularMarketPrice.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="60d")
        
        if df.empty or len(df) < 14:
            return None, "לא נמצאו מספיק נתונים עבור הסימול המבוקש."
            
        current_price = float(df['Close'].iloc[-1])
        
        if len(df) > 1:
            prev_close = float(df['Close'].iloc[-2])
            price_change_pct = ((current_price - prev_close) / prev_close) * 100
        else:
            price_change_pct = 0.0
            
        # הגנה מפני NaN
        if pd.isna(current_price):
            current_price = 0.0
        if pd.isna(price_change_pct):
            price_change_pct = 0.0

        return {
            'price': current_price,
            'change_pct': price_change_pct,
            'df': df
        }, None
        
    except Exception as e:
        return None, f"שגיאה בשליפת נתוני מניה: {str(e)}"

def get_company_news(ticker_symbol):
    """
    שליפת חדשות עם גיבוי:
    1. ניסיון פנייה ל-Finnhub
    2. במקרה של שגיאה (Error 500) או חוסר בחדשות - גיבוי מ-Alpha Vantage
    """
    news_items = []
    symbol = ticker_symbol.upper()
    
    # ניסיון 1: Finnhub
    if FINNHUB_KEY:
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={week_ago}&to={today}&token={FINNHUB_KEY}"
            
            res = requests.get(url, timeout=7)
            if res.status_code == 200 and isinstance(res.json(), list):
                for item in res.json()[:3]:
                    headline = item.get('headline', '')
                    if headline:
                        news_items.append(headline)
        except Exception as e:
            print(f"Finnhub Error: {e}")

    # ניסיון 2: Alpha Vantage (גיבוי במידה ו-Finnhub החזיר שגיאה או רשימה ריקה)
    if not news_items and ALPHAVANTAGE_KEY:
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit=3&apikey={ALPHAVANTAGE_KEY}"
            res = requests.get(url, timeout=7)
            if res.status_code == 200:
                data = res.json()
                if "feed" in data:
                    for item in data["feed"][:3]:
                        title = item.get('title', '')
                        if title:
                            news_items.append(title)
        except Exception as e:
            print(f"Alpha Vantage Error: {e}")

    return news_items

# ------------------------------------------------------------------------------
# 3. מנוע ניתוח טכני וסנטימנט NLP
# ------------------------------------------------------------------------------
def analyze_stock(ticker_symbol):
    stock_data, err = get_stock_data(ticker_symbol)
    if err:
        return err

    df = stock_data['df']
    price = stock_data['price']
    change_pct = stock_data['change_pct']
    
    # חישוב אינדיקטורים טכניים
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    rsi_val = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
    ema_val = df['EMA_20'].iloc[-1] if not pd.isna(df['EMA_20'].iloc[-1]) else price
    atr_val = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else (price * 0.02)
    
    # ניתוח מגמה
    trend = "שורי (Bullish)" if price > ema_val else "דובי (Bearish)"
    
    # ניתוח חדשות וסנטימנט
    news_list = get_company_news(ticker_symbol)
    news_summary_hebrew = []
    
    keywords_bullish = ['profit', 'growth', 'surge', 'buy', 'upgrade', 'fda', 'approval', 'record', 'partner', 'revenue']
    keywords_bearish = ['loss', 'decline', 'drop', 'sell', 'downgrade', 'lawsuit', 'investigation', 'miss']
    
    sentiment_score = 0
    for headline in news_list:
        # תרגום הכתבה לעברית בבטיחות
        try:
            translated = translator.translate(headline)
        except Exception:
            translated = headline
            
        news_summary_hebrew.append(f"• {translated}")
        
        # ניתוח סנטימנט בסיסי לפי מילות מפתח
        h_lower = headline.lower()
        if any(w in h_lower for w in keywords_bullish):
            sentiment_score += 1
        if any(w in h_lower for w in keywords_bearish):
            sentiment_score -= 1

    if sentiment_score > 0:
        sentiment_str = "חיובי 🟢"
    elif sentiment_score < 0:
        sentiment_str = "שלילי 🔴"
    else:
        sentiment_str = "ניטרלי ⚪"

    # חישוב יעד וסטופ לוסט מבוססי ATR
    stop_loss = price - (1.5 * atr_val)
    take_profit = price + (3.0 * atr_val)
    
    # הרכבת הדיווח בעברית
    news_text = "\n".join(news_summary_hebrew) if news_summary_hebrew else "לא נמצאו חדשות עדכניות."
    
    analysis_msg = (
        f"📊 **ניתוח מניה: {ticker_symbol.upper()}**\n"
        f"-----------------------------------\n"
        f"💵 **מחיר נוכחי:** ${price:.2f} ({change_pct:+.2f}%)\n"
        f"📈 **מגמה טכנית:** {trend}\n"
        f"📉 **RSI (14):** {rsi_val:.1f}\n"
        f"🎯 **יעד רווח משוער (Take Profit):** ${take_profit:.2f}\n"
        f"🛡️ **רמת עצירת הפסד (Stop Loss):** ${stop_loss:.2f}\n\n"
        f"📰 **סנטימנט חדשות:** {sentiment_str}\n"
        f"**חדשות אחרונות:**\n{news_text}"
    )
    
    return analysis_msg

# ------------------------------------------------------------------------------
# 4. פקודות בוט בטלגרם
# ------------------------------------------------------------------------------
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "שלום! שלח לי סימול מניה (למשל: AAPL, NVDA, TSLA) ואחזיר לך ניתוח מקיף בזמן אמת.")

@bot.message_handler(func=lambda message: True)
def handle_stock_request(message):
    symbol = message.text.strip().upper()
    bot.reply_to(message, f"מנתח את {symbol}... מיד מחזיר תוצאות ⏳")
    result = analyze_stock(symbol)
    bot.send_message(message.chat.id, result, parse_mode='Markdown')

# ------------------------------------------------------------------------------
# 5. Flask & Keep-Alive Server עבור Render
# ------------------------------------------------------------------------------
@app.route('/')
def home():
    return "The Trading Bot is Live and Running!"

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# ------------------------------------------------------------------------------
# 6. הרצת היישום
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # הרצת השרת ברקע
    threading.Thread(target=run_flask).start()
    
    # הרצת הבוט בלולאה רציפה
    print("Bot is starting...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
