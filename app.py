import threading
from flask import Flask
from bot import bot, resume_jobs

app = Flask(__name__)

@app.route("/")
def home():
    return "Auto-Recovery Highlight Bot Running"

def run_bot():
    bot.start()
    resume_jobs()
    bot.idle()

if __name__ == "__main__":
    threading.Thread(target=run_bot, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
