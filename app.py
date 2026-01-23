import asyncio
from flask import Flask
from bot import bot

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ AI Detector Bot is running!"

async def run_bot():
    await bot.start()
    print("🤖 Bot started successfully")
    await asyncio.Event().wait()  # keep running forever

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.create_task(run_bot())

    app.run(host="0.0.0.0", port=8080)
