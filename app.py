import threading
import asyncio
from flask import Flask
from bot import bot, resume_pending_jobs

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot running with MongoDB auto-recovery"

def run_bot():
    asyncio.run(resume_pending_jobs())
    bot.run()

if __name__ == "__main__":
    threading.Thread(target=run_bot, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
