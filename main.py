import threading
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from config import BOT_TOKEN
from app import run_web
from bot.start import start
from bot.handler import video_handler, callback_handler

def main():
    # 🌐 Start web service (Render requirement)
    threading.Thread(target=run_web, daemon=True).start()

    tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(
        MessageHandler(
            filters.TEXT | filters.VIDEO | filters.Document.VIDEO,
            video_handler
        )
    )
    tg_app.add_handler(CallbackQueryHandler(callback_handler))

    print("🤖 Video Resizer Bot Running...")
    tg_app.run_polling()

if __name__ == "__main__":
    main()
