from flask import Flask, request
from pyrogram import idle
import asyncio
from bot import bot
from config import WEBHOOK_URL

app = Flask(__name__)

@app.route("/")
def home():
    return "Video Highlight Bot Running 🚀"

@app.route("/webhook", methods=["POST"])
def webhook():
    update = request.get_json()
    asyncio.run(bot.process_update(update))
    return "OK", 200

async def start_bot():
    await bot.start()
    await bot.set_webhook(WEBHOOK_URL)
    print("Webhook set")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(start_bot())
    app.run(host="0.0.0.0", port=10000)
