import threading
from flask import Flask, jsonify
from bot import bot, tasks

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Highlight Bot Running"

@app.route("/status")
def status():
    return jsonify(tasks)

def run_flask():
    app.run(host="0.0.0.0", port=10000)

if __name__ == "__main__":
    # Start webserver in background
    threading.Thread(target=run_flask, daemon=True).start()
    # Resume pending tasks
    bot.run(resume_pending_tasks())
