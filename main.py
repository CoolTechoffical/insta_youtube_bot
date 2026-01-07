import os
import threading
from flask import Flask
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)
from config import BOT_TOKEN
from bot.start import start
from bot.downloader import download_handler

# --------------------
# Flask Web Server (PORT FIX)
# --------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 Telegram Bot is running!"

def run_bot():
    tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, download_handler)
    )

    print("🤖 Bot polling started")
    tg_app.run_polling(drop_pending_updates=True)

# Start bot in background
threading.Thread(target=run_bot, daemon=True).start()

# Start Flask server (THIS OPENS THE PORT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
