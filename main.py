import threading
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from config import BOT_TOKEN
from bot.start import start
from bot.reels import handle_link, quality_callback
import app  # web server

def run_web():
    app.app.run(host="0.0.0.0", port=10000)

def main():
    threading.Thread(target=run_web).start()

    app_tg = ApplicationBuilder().token(BOT_TOKEN).build()

    app_tg.add_handler(CommandHandler("start", start))
    app_tg.add_handler(CallbackQueryHandler(quality_callback))
    app_tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_link))

    app_tg.run_polling()

if __name__ == "__main__":
    main()
