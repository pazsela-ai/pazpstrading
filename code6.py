import os
import logging
import asyncio
import threading
import requests
import yfinance as yf
from flask import Flask
from deep_translator import GoogleTranslator

# יבוא ספריות Finviz
try:
    from finvizfinance.screener.overview import Overview
    HAS_FINVIZ = True
except ImportError:
    HAS_FINVIZ = False

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

TARGET_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
USD_TO_ILS = 3.65

# ==========================================
# שרת Web זעיר + מנגנון Keep-Alive (Uptime 24/7)
# ==========================================
web_app = Flask(__name__)

@web_app.route('/')
def health_check():
    return "Bot is alive and running!", 200

def run_web_server():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)

async def self_ping_keep_alive_job():
    url = os.getenv("RENDER_EXTERNAL_URL")
    if url:
        try:
            res = requests.get(url, timeout=5)
            logger.info(f"🔄 Self-Ping Uptime Check status: {res.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Self-Ping נכשל: {e}")

# ==========================================
# פונקציות עזר וטכני (זיהוי תבניות מחיר מרובות)
# ==========================================

def translate_to_hebrew(text: str) -> str:
    try:
        if not text or not text.strip():
            return text
        translated = GoogleTranslator(source='en', target='he').translate(text)
        return translated if translated else text
    except Exception as e:
        logger.error(f"שגיאת תרגום: {e}")
        return text

def build_action_keyboard(symbol: str) -> InlineKeyboardMarkup:
    """ייצור המקדלת האינטראקטיבית הכוללת קישור ישיר לגרף ב-TradingView"""
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
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_candlestick_pattern(df) -> list:
    """זיהוי תבניות נרות יפניים (Hammer, Bullish Engulfing)"""
    patterns = []
    if len(df) < 2:
        return patterns

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    o, h, l, c = float(curr['Open']), float(curr['High']), float(curr['Low']), float(curr['Close'])
    body = abs(c - o)
    candle_range = h - l

    # 1. נר פטיש (Hammer)
    if candle_range > 0:
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        if lower_wick >= 2 * body and upper_wick <= body * 0.5:
            patterns.append("🔨 נר פטיש (Hammer - איסוף/היפוך קונים)")

    # 2. בליעה שורית (Bullish Engulfing)
    prev_o, prev_c = float(prev['Open']), float(prev['Close'])
    if prev_c < prev_o and c > o:
        if c >= prev_o and o <= prev_c:
            patterns.append("🔥 בליעה שורית (Bullish Engulfing - השתלטות קונים)")

    return patterns

def detect_chart_patterns(df) -> list:
    """זיהוי תבניות מחיר טכניות מוכרות (Cup & Handle, Bull Flag, Double Bottom, Ascending Triangle)"""
    patterns = []
    if len(df) < 30:
        return patterns

    closes = df['Close'].tail(30).tolist()
    highs = df['High'].tail(30).tolist()
    lows = df['Low'].tail(30).tolist()
    curr_price = closes[-1]

    # 1. תבנית ספל וידית (Cup & Handle)
    left_rim = max(closes[:10])
    bottom = min(closes[10:20])
    right_rim = max(closes[20:26])
    handle_min = min(closes[25:])

    cup_depth = (left_rim - bottom) / left_rim
    rim_diff = abs(left_rim - right_rim) / left_rim

    if 0.05 <= cup_depth <= 0.35 and rim_diff <= 0.04:
        if handle_min >= bottom and (right_rim - handle_min) / right_rim <= 0.08:
            if abs(curr_price - right_rim) / right_rim <= 0.03:
                patterns.append("☕ תבנית ספל וידית (Cup & Handle)")

    # 2. תבנית דגל שורי (Bullish Flag / Pennant)
    # עלייה חדה ב-10-20 ימים האחרונים ולאחריה התכנסות קלה ב-5 ימים האחרונים
    pole_gain = (max(closes[10:25]) - min(closes[:10])) / min(closes[:10])
    flag_range = (max(highs[25:]) - min(lows[25:])) / curr_price
    if pole_gain >= 0.08 and flag_range <= 0.035:
        patterns.append("🚩 תבנית דגל שורי (Bullish Flag - דחיסה לפני פריצה)")

    # 3. תבנית תחתית כפולה (Double Bottom)
    first_bottom = min(lows[:15])
    second_bottom = min(lows[15:])
    diff_bottoms = abs(first_bottom - second_bottom) / first_bottom
    if diff_bottoms <= 0.02 and curr_price > max(closes[10:20]):
        patterns.append("⚓ תבנית תחתית כפולה (Double Bottom - היפוך מגמה)")

    # 4. תבנית משולש עולה (Ascending Triangle)
    resistance = max(highs[10:])
    low1 = min(lows[:10])
    low2 = min(lows[10:20])
    low3 = min(lows[20:])
    if low3 > low2 > low1 and abs(curr_price - resistance) / resistance <= 0.025:
        patterns.append("📐 תבנית משולש עולה (Ascending Triangle - לחץ קונים על התנגדות)")

    return patterns

# ==========================================
# מנוע ניתוח מורחב (הגדרות טכניות מתקדמות + קטליזטורים)
# ==========================================

def generate_full_analysis_report(symbol: str) -> dict:
    symbol = symbol.upper()
    ticker = yf.Ticker(symbol)
    
    try:
        hist = ticker.history(period="2mo")
        if hist.empty or len(hist) < 20:
            return {"has_data": False, "symbol": symbol}

        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        price_change_pct = ((current_price - prev_close) / prev_close) * 100

        volume_today = float(hist['Volume'].iloc[-1])
        avg_volume = float(hist['Volume'].mean())
        volume_ratio = volume_today / avg_volume if avg_volume > 0 else 1.0

        hist['RSI'] = calculate_rsi(hist['Close'])
        rsi_value = float(hist['RSI'].dropna().iloc[-1]) if not hist['RSI'].dropna().empty else 50.0
        sma20_value = float(hist['Close'].rolling(window=20).mean().iloc[-1])

        # ----------------------------------------------------
        # ניתוח טכני: SMA20 + נרות + תבניות מחירי פריצה
        # ----------------------------------------------------
        dist_from_sma20_pct = ((current_price - sma20_value) / sma20_value) * 100
        near_sma20 = abs(dist_from_sma20_pct) <= 1.5

        is_accumulating_vol = (volume_ratio >= 1.2) and (-2.0 <= price_change_pct <= 4.0)
        
        candlestick_patterns = detect_candlestick_pattern(hist)
        chart_patterns = detect_chart_patterns(hist)
        all_detected_patterns = candlestick_patterns + chart_patterns

        tech_score = 0
        tech_signals = []

        if is_accumulating_vol:
            tech_score += 2
            tech_signals.append(f"צבירת נפח שקטה ({volume_ratio:.1f}x)")
        
        if near_sma20:
            tech_score += 2
            tech_signals.append(f"התכנסות/דחיסה על ממוצע 20 (${sma20_value:.2f})")
        elif current_price > sma20_value:
            tech_score += 1
            tech_signals.append("מגמה עולה מעל ממוצע 20")

        if all_detected_patterns:
            tech_score += 3 if chart_patterns else 2
            tech_signals.extend(all_detected_patterns)

        if 40 <= rsi_value <= 65:
            tech_score += 1
            tech_signals.append(f"RSI מאוזן לקראת פריצה ({rsi_value:.1f})")

        # קביעת המלצה טכנית לפי ניקוד
        if tech_score >= 4:
            tech_recommendation = "🟢 מומלץ להיכנס (מבנה טכני מושלם לפני פריצה)"
        elif tech_score >= 2:
            tech_recommendation = "🟢 מומלץ להיכנס (סימני מומנטום/איסוף ראשוניים)"
        elif rsi_value > 75:
            tech_recommendation = "🟡 המתנה (סכנת מתיחת יתר)"
        else:
            tech_recommendation = "🟡 מעקב בלבד"

        tech_reasoning = " | ".join(tech_signals) if tech_signals else "ללא איתותים טכניים מיוחדים"

        # ----------------------------------------------------
        # ניתוח קטליזטורים וחדשות
        # ----------------------------------------------------
        early_catalyst_keywords = {
            "עסקה / מיזוג": ["merger", "acquisition", "buyout", "takeover", "deal"],
            "שיתוף פעולה / חוזה": ["partnership", "collaboration", "contract", "joint venture", "award"],
            "ביטחון / גאו-פוליטיקה": ["defense", "military", "sanctions", "tariff", "supply chain"],
            "פארמה / אישור": ["fda", "phase", "trial", "patent", "approval", "cleared"],
            "תחזית / דוחות": ["guidance", "raised", "outlook", "pre-announcement", "beat"]
        }

        pos_keywords = ["win", "surges", "soars", "growth", "buy", "record", "profit", "bullish"]
        neg_keywords = ["lawsuit", "investigation", "drop", "decline", "miss", "risk", "downgrade"]

        pos_score, neg_score = 0, 0
        detected_catalysts = []
        translated_titles = []

        news_items = ticker.news if hasattr(ticker, "news") else []

        for item in news_items[:5]:
            content = item.get("content", item)
            raw_title = content.get("title") or item.get("title", "")
            if raw_title:
                title_lower = raw_title.lower()
                for cat_type, keywords in early_catalyst_keywords.items():
                    for kw in keywords:
                        if kw in title_lower:
                            pos_score += 2.0
                            if cat_type not in detected_catalysts:
                                detected_catalysts.append(cat_type)

                for kw in pos_keywords:
                    if kw in title_lower: pos_score += 1.5
                for kw in neg_keywords:
                    if kw in title_lower: neg_score += 1.5
                
                translated_titles.append(translate_to_hebrew(raw_title))

        has_early_catalyst = len(detected_catalysts) > 0

        if has_early_catalyst:
            catalyst_list_str = ", ".join(detected_catalysts)
            news_recommendation = f"🟢 מומלץ להיכנס (קטליזטור: {catalyst_list_str})"
            news_reasoning = f"זיהוי דיווחים בתחום {catalyst_list_str}."
        elif pos_score > neg_score:
            news_recommendation = "🟢 סנטימנט חיובי"
            news_reasoning = "זיהוי סנטימנט חיובי בדיווחים."
        elif neg_score > pos_score:
            news_recommendation = "🔴 לא מומלץ (סיכון)"
            news_reasoning = "זיהוי סנטימנט שלילי."
        else:
            news_recommendation = "🟡 ניטרלי"
            news_reasoning = "אין אירועים חדשותיים דרמטיים."

        # המלצה סופית וניהול סיכונים
        entry_price = round(current_price, 2)
        take_profit = round(current_price * 1.08, 2)
        stop_loss = round(current_price * 0.97, 2)

        if tech_score >= 4 or (has_early_catalyst and neg_score == 0):
            final_recommendation = "🟢 מומלץ להיכנס להשקעה (זיהוי מוקדם איכותי)"
            final_reason = f"שילוב אישור טכני ({tech_reasoning}) וניתוח פונדמנטלי ({news_reasoning})."
        elif "🟢" in tech_recommendation or "🟢" in news_recommendation:
            final_recommendation = "🟢 מומלץ להיכנס להשקעה"
            final_reason = f"{tech_reasoning} | {news_reasoning}"
        else:
            final_recommendation = "🟡 להמתין / לעקוב"
            final_reason = f"{tech_reasoning} | {news_reasoning}"

        entry_reason = f"מחיר השוק (${entry_price}) בנקודת היפוך/צבירה אופטימלית."
        stop_reason = f"נקבע ב-${stop_loss} (-3%) מתחת לתמיכה או לממוצע 20."

        news_summary_text = " • " + "\n • ".join(translated_titles) if translated_titles else "אין חדשות מהותיות כרגע"

        msg = (
            f"🚨 **דוח ניתוח מקיף עבור {symbol}**\n"
            f"───────────────────────\n\n"
            f"💡 **המלצה סופית:** {final_recommendation}\n"
            f"📌 **נימוק:** {final_reason}\n\n"
            f"📊 **ניתוח טכני מתקדם:**\n"
            f"• מחיר: ${current_price:.2f} ({price_change_pct:+.1f}%) | RSI: {rsi_value:.1f}\n"
            f"• ממוצע 20 (SMA20): ${sma20_value:.2f}\n"
            f"• איתותים ותבניות: {tech_reasoning}\n"
            f"• סטטוס טכני: {tech_recommendation}\n\n"
            f"📰 **חדשות ואירועים מוקדמים:**\n"
            f"{news_summary_text}\n"
            f"• סטטוס חדשותי: {news_recommendation}\n\n"
            f"🎯 **ניהול סיכונים ותוכנית מסחר:**\n"
            f"• **מחיר כניסה מומלץ:** ${entry_price}\n"
            f"  👈 *מדוע:* {entry_reason}\n"
            f"• **מחיר סטופ לוס (Stop Loss):** ${stop_loss}\n"
            f"  👈 *מדוע:* {stop_reason}\n"
            f"• **יעד רווח (Take Profit):** ${take_profit} (+8%)\n"
        )

        is_alert_triggered = tech_score >= 3 or has_early_catalyst

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
            "news_titles": translated_titles
        }
    except Exception as e:
        logger.error(f"שגיאה בהפקת דוח ל-{symbol}: {e}")
        return {"has_data": False, "symbol": symbol}

# ==========================================
# סורק שוק רחב (סריקה אוטומטית + סריקה ידנית)
# ==========================================

def discover_broad_market_opportunities() -> list:
    discovered_symbols = set()
    if HAS_FINVIZ:
        try:
            fviz = Overview()
            fviz.set_filter(signal='Unusual Volume')
            df = fviz.screener_view()
            if not df.empty and 'Ticker' in df.columns:
                for sym in df['Ticker'].head(6):
                    if sym and "." not in sym:
                        discovered_symbols.add(str(sym))
        except Exception as e:
            logger.error(f"שגיאה בסריקת Finviz: {e}")

    yahoo_keys = ["most_actives", "day_gainers"]
    headers = {'User-Agent': 'Mozilla/5.0'}

    for key in yahoo_keys:
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&key={key}"
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
                for item in results[:5]:
                    sym = item.get("symbol")
                    if sym and "." not in sym:
                        discovered_symbols.add(sym)
        except Exception as e:
            logger.error(f"שגיאה בסריקת Yahoo ({key}): {e}")

    return list(discovered_symbols)

async def auto_market_scanner_job(app: Application):
    """סריקת רקע אוטומטית - שולחת התראות רק כשיש הזדמנות איכותית"""
    global TARGET_CHAT_ID
    if not TARGET_CHAT_ID:
        return

    candidate_symbols = discover_broad_market_opportunities()

    for symbol in candidate_symbols:
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
# פקודות ותגובות משתמש
# ==========================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TARGET_CHAT_ID
    TARGET_CHAT_ID = str(update.effective_chat.id)
    
    welcome_text = (
        "👋 **ברוכים הבאים לסורק השוק והסייען הפיננסי האוטומטי!**\n\n"
        "🤖 **הבוט מנטר את השוק ברקע לאיתור מניות פוטנציאליות עם שילוב תבניות (ספל וידית, דגל שורי, תחתית כפולה, משולש עולה), נפח וחדשות מוקדמות.**\n\n"
        "💡 **פקודות זמינות:**\n"
        "• `/scan` - הפעלת סריקת שוק ידנית עכשיו (מציג תוצאות זהות להתראות)\n"
        "• `/tech TSLA` - סריקה טכנית ממוקדת למנייה\n"
        "• `/news TSLA` - סריקה חדשותית ממוקדת למנייה"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def handle_manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """סריקה ידנית מורחבת לבקשת המשתמש - התוצרים מוצגים בדיוק באותו פורמט מקיף"""
    global TARGET_CHAT_ID
    TARGET_CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text("🔎 מפעיל סורק שוק ידני לאיתור מניות חמות ותבניות פריצה...")

    candidate_symbols = discover_broad_market_opportunities()
    found_any = False

    for symbol in candidate_symbols:
        try:
            report = generate_full_analysis_report(symbol)
            if report.get("has_data"):
                reply_markup = build_action_keyboard(symbol)
                await update.message.reply_text(
                    report["formatted_message"],
                    parse_mode="Markdown",
                    reply_markup=reply_markup,
                    disable_web_page_preview=True
                )
                found_any = True
                await asyncio.sleep(1.5)
        except Exception as e:
            logger.error(f"שגיאה בסריקה ידנית ל-{symbol}: {e}")

    if not found_any:
        await update.message.reply_text("ℹ️ הסריקה הושלמה - לא נמצאו הזדמנויות חריגות ברגע זה.")

async def handle_tech(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
            f"💰 **מחשבון עסקה עבור {symbol}:**\nבאיזה מטבע תרצה לחשב את ההשקעה?",
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
        f"📈 **רווח צפוי (Take Profit):** {expected_profit_final}\n"
        f"🛡️ **סיכון מרבי (Stop Loss):** {expected_risk_final}\n\n"
        f"📌 **המלצה משוקללת:** {report['tech_recommendation']}"
    )

    tradingview_url = f"https://www.tradingview.com/chart/?symbol={symbol}"
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("📈 צפייה בגרף ב-TradingView", url=tradingview_url)]])

    await update.message.reply_text(calc_msg, parse_mode="Markdown", reply_markup=keyboard)

# ==========================================
# הפעלה
# ==========================================

async def post_init(application: Application) -> None:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(auto_market_scanner_job, 'interval', minutes=15, args=[application])
    scheduler.add_job(self_ping_keep_alive_job, 'interval', minutes=10)
    scheduler.start()
    logger.info("🤖 APScheduler ומנגנון Uptime הופעלו בהצלחה!")

def main():
    threading.Thread(target=run_web_server, daemon=True).start()

    TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")

    application = Application.builder().token(TOKEN).post_init(post_init).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", handle_manual_scan))
    application.add_handler(CommandHandler("news", handle_news))
    application.add_handler(CommandHandler("tech", handle_tech))
    
    application.add_handler(CallbackQueryHandler(button_click_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_text_input))

    logger.info("🤖 הבוט עולה לאוויר...")
    application.run_polling()

if __name__ == "__main__":
    main()
