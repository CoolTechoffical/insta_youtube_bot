import threading
from flask import Flask, jsonify
from bot import bot, tasks, resume_tasks

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Auto-Resume Highlight Bot Running"

@app.route("/status")
def status():
    return jsonify(tasks)

def run_flask():
    app.run(host="0.0.0.0", port=10000)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    bot.run(resume_tasks())
