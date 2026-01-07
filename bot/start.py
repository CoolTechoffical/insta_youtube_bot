from telegram import Update
from telegram.ext import ContextTypes

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome!\n\n"
        "🤖 I am your Instagram & YouTube Helper Bot.\n"
        "📥 Send me a link to get started."
    )
