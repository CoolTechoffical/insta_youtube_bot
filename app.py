import asyncio
import threading
from flask import Flask
from bot import bot

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ AI Detector Bot is running!"

def run_flask():
    app.run(host="0.0.0.0", port=8080)

async def run_bot():
    await bot.start()
    print("🤖 Bot started successfully")
    await asyncio.Event().wait()  # keep alive forever

if __name__ == "__main__":
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Start Pyrogram bot in asyncio loop
    asyncio.run(run_bot())
