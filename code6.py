import urllib.parse
import requests

def translate_to_hebrew(text: str) -> str:
    """מתרגם טקסט מאנגלית לעברית ללא צורך בספריות חיצוניות"""
    try:
        if not text or text.strip() == "":
            return text
        encoded_text = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=he&dt=t&q={encoded_text}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            result = response.json()
            translated = "".join([item[0] for item in result[0] if item[0]])
            return translated if translated else text
        return text
    except Exception as e:
        logger.error(f"שגיאת תרגום: {e}")
        return text

async def handle_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("אנא ציין סימול מנייה. דוגמה:\n`/news MRNA`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"🌐 מנתח ומזקק את התמונה החדשותית עבור {symbol}...")

    ticker = yf.Ticker(symbol)
    news_items = ticker.news if hasattr(ticker, "news") else []

    if not news_items:
        await update.message.reply_text(
            f"ℹ️ **אין עדכונים חדשותיים מהותיים כרגע עבור `{symbol}`.**", 
            parse_mode="Markdown"
        )
        return

    # מילון מונחים משוכלל המזהה הקשר חיובי ושלילי
    pos_keywords = [
        "breakthrough", "win", "surges", "soars", "beat", "growth", "deal", 
        "approval", "fda", "buy", "upgrade", "record", "profit", "rallies", "brings back"
    ]
    neg_keywords = [
        "lawsuit", "investigation", "drop", "decline", "miss", "risk", "cut", 
        "downgrade", "bankrupt", "loss", "crash", "plummet", "fail"
    ]

    pos_score = 0
    neg_score = 0
    translated_summaries = []

    for item in news_items[:4]:
        content = item.get("content", item)
        raw_title = content.get("title") or item.get("title", "")

        if raw_title:
            title_lower = raw_title.lower()

            # ספירת נקודות סנטימנט
            for kw in pos_keywords:
                if kw in title_lower:
                    pos_score += 1.5
            for kw in neg_keywords:
                if kw in title_lower:
                    neg_score += 1.5

            # תרגום הכותרת
            translated_title = translate_to_hebrew(raw_title)
            translated_summaries.append(translated_title)

    if not translated_summaries:
        await update.message.reply_text(f"ℹ️ **אין חדשות זמינות עבור `{symbol}`.**", parse_mode="Markdown")
        return

    # יצירת שורה תחתונה אחת מרוכזת
    bottom_line_summary = " • " + "\n • ".join(translated_summaries)

    # קביעת המגמה וההמלצה הסופית
    if pos_score > neg_score:
        expected_trend = "📈 **מגמה צפויה:** חיובית (עצימות שורית)"
        recommendation = "✅ **שווה לשקול השקעה!**"
        reasoning = (
            f"**תמצית הניתוח:** הדיווחים מציגים התפתחויות חיוביות וקטליזטורים מעודדים "
            f"(כגון פריצות דרך, גידול בפעילות או שיתופי פעולה). "
            f"הסנטימנט הכללי בחדשות תומך בעליית מחירים בטווח הקצר."
        )
    elif neg_score > pos_score:
        expected_trend = "📉 **מגמה צפויה:** שלילית (עצימות דובית)"
        recommendation = "🛑 **לא מומלץ להשקיע כרגע.**"
        reasoning = (
            f"**תמצית הניתוח:** הדיווחים מציגים חדשות שליליות, אזהרות רווח או סיכונים משפטיים/רגולטוריים. "
            f"הסנטימנט בחדשות מפעיל לחץ מוכרים על המנייה."
        )
    else:
        expected_trend = "➡️ **מגמה צפויה:** ניטרלית / ללא כיוון ברור"
        recommendation = "⏳ **להמתין ולא להיכנס רק על בסיס החדשות.**"
        reasoning = (
            f"**תמצית הניתוח:** הידיעות ניטרליות או מעורבות ולא מציגות טריגר חד-משמעי לפריצה."
        )

    msg = (
        f"📰 **סיכום חדשות פיננסי עבור {symbol}**\n"
        f"───────────────────────\n\n"
        f"📌 **השורה התחתונה - מה קרה בשוק:**\n"
        f"{bottom_line_summary}\n\n"
        f"───────────────────────\n"
        f"{expected_trend}\n"
        f"💡 **המלצה:** {recommendation}\n\n"
        f"🧠 {reasoning}"
    )

    reply_markup = build_action_keyboard(symbol)
    await update.message.reply_text(
        msg, 
        parse_mode="Markdown", 
        disable_web_page_preview=True, 
        reply_markup=reply_markup
    )
