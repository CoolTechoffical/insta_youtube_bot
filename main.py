import threading
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)
from config import BOT_TOKEN
from app import run_web
from bot.start import start
from bot.handler import video_handler

def main():
    # Web server (Render requirement)
    threading.Thread(target=run_web).start()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(
            filters.TEXT | filters.VIDEO | filters.Document.VIDEO,
            video_handler
        )
    )

    print("🤖 Video Resizer Bot Running")
    app.run_polling()

if __name__ == "__main__":
    main()
