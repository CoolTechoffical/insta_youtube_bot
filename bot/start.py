from telegram import Update
from telegram.ext import ContextTypes


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *InstaTube Helper Bot*\n\n"
        "📥 Send Instagram Reel or YouTube link\n"
        "⚡ Fast & Free\n\n"
        "🚀 More features coming soon!",
        parse_mode="Markdown"
    )
