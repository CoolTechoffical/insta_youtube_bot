import asyncio
from flask import Flask
from bot import bot

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ AI Detector Bot is running!"

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(BOT.start())
    app.run(host="0.0.0.0", port=8080)
