import threading
import asyncio
from flask import Flask
from bot import bot, resume_pending_jobs

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot running with MongoDB auto-recovery"

def flask():
    app.run(host="0.0.0.0", port=10000)


if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()

    # Run pyrogram in MAIN thread
    bot.run()
