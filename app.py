import threading
from flask import Flask
from bot import bot

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Video Highlight Bot is running"

def run_bot():
    bot.run()  # Pyrogram polling

if __name__ == "__main__":
    threading.Thread(target=run_bot).start()
    app.run(host="0.0.0.0", port=10000)
