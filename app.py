import os
import threading
from flask import Flask
from bot import bot

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ AI Detector Bot is running!"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    print("🤖 Starting Flask + Bot...")

    # Flask in background thread
    threading.Thread(target=run_flask, daemon=True).start()

    # Pyrogram MUST run in main thread
    bot.run()
