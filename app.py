import threading
from flask import Flask
from bot import bot, resume_tasks

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Highlight Bot Running"

def run_flask():
    app.run(host="0.0.0.0", port=10000)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    bot.run(resume_tasks())
