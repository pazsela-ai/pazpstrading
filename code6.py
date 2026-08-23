import os
import time
import threading
import math
import requests
import yfinance as yf
from flask import Flask
from telebot import TeleBot, types
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler
from deep_translator import GoogleTranslator

# ------------------------------------------------------------------------------
# 1. הגדרות סביבה ומשתנים גלובליים (Environment & Setup)
# ------------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
PORT = int(os.environ.get("PORT", 5000))
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")

bot = TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)
translator = GoogleTranslator(source='auto', target='iw')

# שמירת מזהי צ'אט של משתמשים שנרשמו לבוד (לשליחת התראות אוטומטיות)
REGISTERED_CHAT_IDS = set()

# מילון לשמירת מצבי שיחה של משתמשים בעת הזנת סכומים במחשבון
# מבנה: {chat_id: {"symbol": str, "entry": float, "stop": float, "pattern_depth": float, "is_usd": bool}}
USER_CALC_STATE = {}

# ------------------------------------------------------------------------------
# 2. שרת FLASK ומנגנון KEEP-ALIVE (Flask Server & Self-Ping)
# ------------------------------------------------------------------------------
@app.route('/')
def health_check():
    return "OK - Telegram Trading Bot is Running!", 200

def keep_alive_ping():
    """מנגנון Self-Ping המונע משרת הענן (Render) להיכנס למצב שינה"""
    while True:
        try:
            time.sleep(600)  # כל 10 דקות
            requests.get(SELF_URL, timeout=10)
        except Exception as e:
            print(f"[Keep-Alive Error]: {e}")

# ------------------------------------------------------------------------------
# 3. מנוע ניתוח טכני וזיהוי תבניות (Technical Analysis Engine)
# ------------------------------------------------------------------------------
def analyze_technical_patterns(symbol: str) -> dict:
    """
    מוריד נתוני מסחר מ-yfinance, מציג אינדיקטורים ומזהה תבניות פריצה ונרות יפניים.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")
        if df.empty or len(df) < 30:
            return None

        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100

        # חישוב RSI (14 ימים)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))

        # חישוב SMA20 ונפח מסחר
        sma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        avg_volume = df['Volume'].iloc[-21:-1].mean()
        current_volume = df['Volume'].iloc[-1]
        volume_accumulating = current_volume > (avg_volume * 1.2)

        # זיהוי תבניות מחיר ונרות
        patterns = []
        pattern_depth = 0.15  # עומק תבנית ברירת מחדל 15%

        # ספל וידית (Cup & Handle)
        high_30 = df['High'].iloc[-30:].max()
        low_30 = df['Low'].iloc[-30:].min()
        if (high_30 - current_price) / high_30 < 0.05 and current_price > sma20:
            patterns.append("ספל וידית (Cup & Handle)")
            pattern_depth = max(pattern_depth, (high_30 - low_30) / high_30)

        # דגל שורי (Bullish Flag)
        recent_gain = (df['High'].iloc[-5:].max() - df['Low'].iloc[-15:].min()) / df['Low'].iloc[-15:].min()
        if recent_gain > 0.12 and abs(current_price - df['Close'].iloc[-3:].mean()) / current_price < 0.02:
            patterns.append("דגל שורי (Bullish Flag)")
            pattern_depth = max(pattern_depth, recent_gain)

        # תחתית כפולה (Double Bottom)
        lows = df['Low'].iloc[-20:]
        if len(lows[lows <= low_30 * 1.02]) >= 2:
            patterns.append("תחתית כפולה (Double Bottom)")

        # משולש עולה (Ascending Triangle)
        if df['Low'].iloc[-10] < df['Low'].iloc[-5] < df['Low'].iloc[-1] and abs(df['High'].iloc[-10:].max() - current_price) / current_price < 0.03:
            patterns.append("משולש עולה (Ascending Triangle)")

        # נרות יפניים
        open_p, close_p, high_p, low_p = df['Open'].iloc[-1], df['Close'].iloc[-1], df['High'].iloc[-1], df['Low'].iloc[-1]
        body = abs(close_p - open_p)
        lower_shadow = min(open_p, close_p) - low_p
        
        if lower_shadow > 2 * body and body > 0:
            patterns.append("נר פטיש (Hammer)")
        
        prev_open, prev_close = df['Open'].iloc[-2], df['Close'].iloc[-2]
        if prev_close < prev_open and close_p > open_p and close_p > prev_open and open_p < prev_close:
            patterns.append("בליעה שורית (Bullish Engulfing)")

        # חישוב תוכנית מסחר (סטופ לוס מתחת ל-SMA20 או 3% מתחת למחיר)
        stop_loss = min(sma20 * 0.98, current_price * 0.97)

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi, 1),
            "sma20": round(sma20, 2),
            "volume_accumulating": volume_accumulating,
            "patterns": patterns,
            "pattern_depth": pattern_depth,
            "stop_loss": round(stop_loss, 2)
        }
    except Exception as e:
        print(f"[Tech Analysis Error] {symbol}: {e}")
        return None

# ------------------------------------------------------------------------------
# 4. מנוע חדשות, קטליזטורים וסנטימנט (News & Sentiment Engine)
# ------------------------------------------------------------------------------
def fetch_news_and_catalysts(symbol: str) -> dict:
    """
    שולף חדשות מ-Yahoo Finance, מתרגם לעברית ומזהה קטליזטורים מוקדמים.
    """
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news
        if not news_items:
            return {"headlines": [], "catalyst": "לא אותרו אירועים מיוחדים", "sentiment": "ניטרלי"}

        translated_headlines = []
        catalysts_found = []
        positive_keywords = ["approval", "fda", "merger", "acquisition", "contract", "partner", "guidance", "beat", "surge", "growth"]

        for item in news_items[:3]:
            title = item.get("title", "")
            if not title:
                continue
            
            # תרגום כותרת לעברית
            try:
                translated_title = translator.translate(title)
            except:
                translated_title = title
            
            translated_headlines.append(translated_title)

            # זיהוי קטליזטורים מילות מפתח
            title_lower = title.lower()
            if any(k in title_lower for k in ["merger", "acquisition"]):
                catalysts_found.append("עסקת מיזוג/רכישה")
            elif any(k in title_lower for k in ["fda", "approval"]):
                catalysts_found.append("אישור פארמה/FDA")
            elif any(k in title_lower for k in ["contract", "deal", "partner"]):
                catalysts_found.append("חתימת חוזה/שותפות אסטרטגית")
            elif any(k in title_lower for k in ["guidance", "earnings", "beat"]):
                catalysts_found.append("עדכון תחזית חיובי / דוחות")

        catalyst_str = ", ".join(set(catalysts_found)) if catalysts_found else "אין קטליזטור ישיר מעבר למומנטום"
        sentiment = "חיובי 🟢" if (catalysts_found or len(translated_headlines) > 0) else "ניטרלי 🟡"

        return {
            "headlines": translated_headlines,
            "catalyst": catalyst_str,
            "sentiment": sentiment
        }
    except Exception as e:
        print(f"[News Error] {symbol}: {e}")
        return {"headlines": [], "catalyst": "שגיאה בשליפת חדשות", "sentiment": "לא ידוע"}

# ------------------------------------------------------------------------------
# 5. אלגוריתם תוכנית המסחר והמחשבון המודולרי (TP1/TP2 Strategy Engine)
# ------------------------------------------------------------------------------
def calculate_trade_plan(entry_price: float, stop_loss: float, pattern_depth_pct: float = 0.15) -> dict:
    """
    מחשב תוכנית מסחר דינמית עם 2 יעדים ואיפוס סיכון:
    - TP1: יעד 1:2 מול הסטופ (למימוש 50% והעלאת סטופ ל-Break-even)
    - TP2: יעד מורחב לפי עומק התבנית (Measured Move)
    """
    risk_per_share = entry_price - stop_loss
    risk_pct = risk_per_share / entry_price
    
    tp1_price = entry_price + (risk_per_share * 2)
    tp1_pct = ((tp1_price - entry_price) / entry_price) * 100
    
    tp2_pct_calculated = max(pattern_depth_pct * 100, (risk_pct * 3.5) * 100)
    tp2_price = entry_price * (1 + (tp2_pct_calculated / 100))
    
    return {
        "entry": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "tp1": round(tp1_price, 2),
        "tp1_pct": round(tp1_pct, 1),
        "tp2": round(tp2_price, 2),
        "tp2_pct": round(tp2_pct_calculated, 1),
        "new_stop_after_tp1": round(entry_price, 2)
    }

def generate_calculator_response(amount: float, is_usd: bool, usd_rate: float, trade_plan: dict) -> str:
    """
    מחולל את תצוגת מחשבון העסקה המודולרי עם פירוק היעדים עבור הטלגרם
    """
    currency_symbol = "$" if is_usd else "₪"
    amount_in_usd = amount if is_usd else amount / usd_rate
    
    entry = trade_plan["entry"]
    stop = trade_plan["stop_loss"]
    tp1 = trade_plan["tp1"]
    tp2 = trade_plan["tp2"]
    
    total_shares = int(amount_in_usd / entry)
    if total_shares < 1:
        return "⚠️ סכום ההשקעה נמוך מדי לרכישת מנייה אחת לפחות במחיר הכניסה."
    
    half_shares = total_shares // 2
    rem_shares = total_shares - half_shares
    
    profit_tp1_usd = half_shares * (tp1 - entry)
    profit_tp2_usd = rem_shares * (tp2 - entry)
    total_profit_usd = profit_tp1_usd + profit_tp2_usd
    max_risk_usd = total_shares * (entry - stop)
    
    profit_tp1_ils = profit_tp1_usd * usd_rate
    profit_tp2_ils = profit_tp2_usd * usd_rate
    total_profit_ils = total_profit_usd * usd_rate
    max_risk_ils = max_risk_usd * usd_rate

    msg = f"""
<b>💰 תוכנית מודולרית לניהול העסקה ({amount:,.0f}{currency_symbol}):</b>

<b>📌 כניסה וכמות:</b>
• מחיר כניסה מומלץ: <code>${entry:.2f}</code>
• כמות מניות כוללת לרכישה: <b>{total_shares} מניות</b>

---
<b>🎯 שלב 1: מימוש ראשוני ואיפוס סיכון (TP1)</b>
• מחיר יעד 1: <code>${tp1:.2f}</code> (+{trade_plan['tp1_pct']}%)
• <b>פעולה לבצוע:</b> למכור <b>{half_shares} מניות</b> (50% מהכמות).
• <b>רווח ננעל במזומן:</b> ${profit_tp1_usd:.2f} ({profit_tp1_ils:.2f} ₪)
• 🛡️ <b>הוראת סטופ לוס חדשה:</b> להעלות מיד ל-<code>${trade_plan['new_stop_after_tp1']:.2f}</code> (מחיר הכניסה).
<i>(מאותו רגע העסקה בסיכון 0! גם אם המנייה תתרסק, לא נפסיד שקל).</i>

---
<b>🚀 שלב 2: הרצת השארית ליעד התבנית המלא (TP2)</b>
• מחיר יעד 2 (דינמי): <code>${tp2:.2f}</code> (+{trade_plan['tp2_pct']}%)
• <b>פעולה לבצוע:</b> למכור את <b>{rem_shares} המניות הנותרות</b> (50%).
• <b>רווח נוסף בשלב זה:</b> ${profit_tp2_usd:.2f} ({profit_tp2_ils:.2f} ₪)

---
<b>📊 סיכום כספי כולל לעסקה:</b>
🟢 <b>סך רווח צפוי בהגעת 2 היעדים:</b> ${total_profit_usd:.2f} ({total_profit_ils:.2f} ₪)
🔴 <b>סיכון מרבי (אם חלילה נתפס הסטופ המקורי ב-${stop:.2f}):</b> ${max_risk_usd:.2f} ({max_risk_ils:.2f} ₪)
"""
    return msg

# ------------------------------------------------------------------------------
# 6. מחולל הדוחות והמקלדת האינטראקטיבית (Report Builder & Telebot UI)
# ------------------------------------------------------------------------------
def create_report_message(symbol: str) -> tuple:
    """
    מפיק דוח ניתוח מקיף ומצרף מקלדת כפתורים אינטראקטיבית
    """
    tech = analyze_technical_patterns(symbol)
    if not tech:
        return f"❌ לא ניתן היה לשלוף נתונים עבור המנייה <b>{symbol}</b>.", None

    news = fetch_news_and_catalysts(symbol)
    plan = calculate_trade_plan(tech["current_price"], tech["stop_loss"], tech["pattern_depth"])

    # קביעת ההמלצה הסופית
    if tech["patterns"] and (tech["volume_accumulating"] or "חיובי" in news["sentiment"]):
        recommendation = "🟢 <b>מומלץ להיכנס להשקעה</b> (זיהוי פריצה טכנית/קטליזטור תומך)"
    elif tech["patterns"]:
        recommendation = "🟡 <b>להמתין לעקוב</b> (קיימת תבנית, מומלץ להמתין לאישור נפח/פריצה)"
    else:
        recommendation = "🔴 <b>לא מומלץ כעת</b> (לא זוהו תבניות פריצה ברורות)"

    patterns_str = ", ".join(tech["patterns"]) if tech["patterns"] else "לא זוהו תבניות מיוחדות"
    headlines_str = "\n".join([f"• {h}" for h in news["headlines"]]) if news["headlines"] else "• אין כותרות חדשות מהותיות"

    msg = f"""
<b>📊 דוח ניתוח מקיף עבור {symbol}</b>

<b>💡 המלצה סופית:</b>
{recommendation}

---
<b>📈 נתונים טכניים ותבניות:</b>
• מחיר נוכחי: <code>${tech['current_price']}</code> ({'+' if tech['change_pct']>0 else ''}{tech['change_pct']}%)
• מדד RSI: <code>{tech['rsi']}</code> | SMA20: <code>${tech['sma20']}</code>
• תבניות שנמצאו: <b>{patterns_str}</b>
• איסוף נפח מסחר: {'כן 🟢' if tech['volume_accumulating'] else 'רגיל ⚪'}

---
<b>📰 חדשות ואירועים מוקדמים:</b>
• קטליזטור שנמצא: <b>{news['catalyst']}</b>
• סנטימנט חדשותי: <b>{news['sentiment']}</b>
<b>כותרות אחרונות:</b>
{headlines_str}

---
<b>🎯 תוכנית מסחר מומלצת:</b>
• מחיר כניסה: <code>${plan['entry']}</code>
• 🛑 סטופ לוס מקורי: <code>${plan['stop_loss']}</code>
• 🎯 יעד 1 (TP1 - למימוש 50% ואיפוס סיכון): <code>${plan['tp1']}</code> (+{plan['tp1_pct']}%)
• 🚀 יעד 2 (TP2 - יעד תבנית מורחב): <code>${plan['tp2']}</code> (+{plan['tp2_pct']}%)
"""

    # מקלדת כפתורים מתחת להודעה
    markup = InlineKeyboardMarkup(row_width=2)
    btn_chart = InlineKeyboardButton("📈 צפייה בגרף", url=f"https://www.tradingview.com/chart/?symbol={symbol}")
    btn_calc = InlineKeyboardButton("💰 חישוב עסקה", callback_data=f"calc_{symbol}_{plan['entry']}_{plan['stop_loss']}_{tech['pattern_depth']}")
    btn_news = InlineKeyboardButton("📰 חדשות בלבד", callback_data=f"news_{symbol}")
    btn_tech = InlineKeyboardButton("📊 ניתוח טכני בלבד", callback_data=f"tech_{symbol}")
    
    markup.add(btn_chart, btn_calc)
    markup.add(btn_news, btn_tech)

    return msg, markup

# ------------------------------------------------------------------------------
# 7. ניהול פקודות ואירועים בטלגרם (Telegram Handlers)
# ------------------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def cmd_start(message):
    REGISTERED_CHAT_IDS.add(message.chat.id)
    welcome_text = """
<b>ברוכים הבאים לסורק השוק והסייען הפיננסי! 🚀</b>

הבוט מנטר באופן רציף את הבורסה האמריקאית, מזהה תבניות פריצה טכניות, מנתח חדשות ואירועים מוקדמים, ומספק תוכנית מסחר מודולרית עם איפוס סיכון.

<b>פקודות זמינות:</b>
/scan - סריקה ידנית מורחבת לאיתור הזדמנויות מיידיות
/tech &lt;SYMBOL&gt; - ניתוח טכני ממוקד למנייה (לדוגמה: <code>/tech TSLA</code>)
/news &lt;SYMBOL&gt; - ניתוח חדשות וסנטימנט למנייה (לדוגמה: <code>/news NVDA</code>)
"""
    bot.reply_to(message, welcome_text, parse_mode="HTML")

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    bot.reply_to(message, "🔍 מתחיל סריקה מורחבת בשוק, אנא המתן...")
    # סריקת מניות מובילות כדוגמה
    top_symbols = ["AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", "META"]
    found = False
    for sym in top_symbols:
        tech = analyze_technical_patterns(sym)
        if tech and tech["patterns"]:
            msg, markup = create_report_message(sym)
            bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
            found = True
    if not found:
        # אם לא זוהתה מנייה עם תבנית חריגה, מחזירים ניתוח על המנייה המובילה
        msg, markup = create_report_message("NVDA")
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)

@bot.message_handler(commands=['tech'])
def cmd_tech(message):
    try:
        symbol = message.text.split()[1].upper()
        msg, markup = create_report_message(symbol)
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
    except IndexError:
        bot.reply_to(message, "⚠️ נא לציין סימול מנייה. לדוגמה: <code>/tech AAPL</code>", parse_mode="HTML")

@bot.message_handler(commands=['news'])
def cmd_news(message):
    try:
        symbol = message.text.split()[1].upper()
        msg, markup = create_report_message(symbol)
        bot.send_message(message.chat.id, msg, parse_mode="HTML", reply_markup=markup)
    except IndexError:
        bot.reply_to(message, "⚠️ נא לציין סימול מנייה. לדוגמה: <code>/news NVDA</code>", parse_mode="HTML")

# ------------------------------------------------------------------------------
# 8. טיפול בלחיצות כפתורים (Callback Queries)
# ------------------------------------------------------------------------------
@bot.callback_query_handler(func=lambda call: True)
def handle_callbacks(call):
    data = call.data
    chat_id = call.message.chat.id

    if data.startswith("calc_"):
        # פורמט: calc_SYMBOL_ENTRY_STOP_DEPTH
        _, symbol, entry, stop, depth = data.split("_")
        USER_CALC_STATE[chat_id] = {
            "symbol": symbol,
            "entry": float(entry),
            "stop": float(stop),
            "pattern_depth": float(depth)
        }
        
        markup = InlineKeyboardMarkup()
        markup.add(
            InlineKeyboardButton("דולר ($)", callback_data="curr_USD"),
            InlineKeyboardButton("שקל (₪)", callback_data="curr_ILS")
        )
        bot.send_message(chat_id, f"💰 <b>מחשבון עסקה עבור {symbol}</b>\nבחר באיזה מטבע תרצה להזין את סכום ההשקעה:", parse_mode="HTML", reply_markup=markup)

    elif data.startswith("curr_"):
        is_usd = (data == "curr_USD")
        if chat_id in USER_CALC_STATE:
            USER_CALC_STATE[chat_id]["is_usd"] = is_usd
            curr_str = "דולר ($)" if is_usd else "שקלים (₪)"
            msg = bot.send_message(chat_id, f"רשום כעת את סכום ההשקעה המבוקש ב-{curr_str}: (למשל: 5000)")
            bot.register_next_step_handler(msg, process_calculator_amount)

    elif data.startswith("news_"):
        symbol = data.split("_")[1]
        news = fetch_news_and_catalysts(symbol)
        headlines_str = "\n".join([f"• {h}" for h in news['headlines']]) if news['headlines'] else "אין כותרות"
        text = f"<b>📰 חדשות וסנטימנט עבור {symbol}:</b>\n\n• קטליזטור: <b>{news['catalyst']}</b>\n• סנטימנט: <b>{news['sentiment']}</b>\n\n<b>כותרות:</b>\n{headlines_str}"
        bot.send_message(chat_id, text, parse_mode="HTML")

    elif data.startswith("tech_"):
        symbol = data.split("_")[1]
        tech = analyze_technical_patterns(symbol)
        if tech:
            patterns_str = ", ".join(tech["patterns"]) if tech["patterns"] else "אין"
            text = f"<b>📊 ניתוח טכני בלבד עבור {symbol}:</b>\n\n• מחיר: <code>${tech['current_price']}</code>\n• RSI: <code>{tech['rsi']}</code>\n• SMA20: <code>${tech['sma20']}</code>\n• תבניות: <b>{patterns_str}</b>\n• סטופ לוס מומלץ: <code>${tech['stop_loss']}</code>"
            bot.send_message(chat_id, text, parse_mode="HTML")

def process_calculator_amount(message):
    """
    קולט את סכום ההשקעה מהמשתמש ומפיק את תפוקת המחשבון המודולרי
    """
    chat_id = message.chat.id
    if chat_id not in USER_CALC_STATE:
        bot.reply_to(message, "❌ פג תוקף החישוב, אנא לחץ שוב על 'חישוב עסקה'.")
        return

    try:
        amount = float(message.text.replace(",", "").strip())
        state = USER_CALC_STATE[chat_id]
        
        # שער דולר/שקל מוערך (ניתן לעדכן דינמית)
        usd_rate = 3.60
        
        plan = calculate_trade_plan(state["entry"], state["stop"], state["pattern_depth"])
        response_text = generate_calculator_response(amount, state["is_usd"], usd_rate, plan)
        
        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton("📈 צפייה בגרף ב-TradingView", url=f"https://www.tradingview.com/chart/?symbol={state['symbol']}"))
        
        bot.send_message(chat_id, response_text, parse_mode="HTML", reply_markup=markup)
        del USER_CALC_STATE[chat_id]
    except ValueError:
        bot.reply_to(message, "⚠️ נא להזין מספר תקין בלבד (לדוגמה: 5000). נסה שוב:")
        bot.register_next_step_handler(message, process_calculator_amount)

# ------------------------------------------------------------------------------
# 9. מנוע סריקה אוטומטית ברקע (Background Scheduler)
# ------------------------------------------------------------------------------
def scheduled_market_scan():
    """
    סריקה אוטומטית הרצה מדי 15 דקות ושולחת התראות למשתמשים רשומים
    """
    if not REGISTERED_CHAT_IDS:
        return
    
    watch_list = ["NVDA", "TSLA", "AAPL", "AMD", "AMZN"]
    for sym in watch_list:
        tech = analyze_technical_patterns(sym)
        if tech and tech["patterns"] and tech["volume_accumulating"]:
            msg, markup = create_report_message(sym)
            for chat_id in REGISTERED_CHAT_IDS:
                try:
                    bot.send_message(chat_id, f"🚨 <b>התראת סריקה אוטומטית!</b>\n{msg}", parse_mode="HTML", reply_markup=markup)
                except Exception as e:
                    print(f"[Scheduled Notification Error] to {chat_id}: {e}")

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(scheduled_market_scan, 'interval', minutes=15)
scheduler.start()

# ------------------------------------------------------------------------------
# 10. הרצת השרת והבוט (Main Execution Loop)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # הפעלת מנגנון Self-Ping ברקע
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    
    # הפעלת שרת ה-Flask בשרשור נפרד
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False), daemon=True).start()
    
    print("🤖 Telegram Trading Bot is active and polling...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
