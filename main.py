from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)
from config import BOT_TOKEN
from bot.start import start
from bot.downloader import download_handler


def main():
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, download_handler)
    )

    print("🤖 Insta & YouTube Helper Bot running (Background Worker)")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
