import os
import threading
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)
from flask import Flask
from config import BOT_TOKEN
from bot.start import start
from bot.downloader import download_handler

# 🌐 Dummy web server for Render
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

def main():
    tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, download_handler)
    )

    print("🤖 Insta & YouTube Helper Bot running...")
    tg_app.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_web).start()
    main()
