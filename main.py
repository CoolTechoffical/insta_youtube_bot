import asyncio
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from pyrogram import Client, idle

from config import API_ID, API_HASH, BOT_TOKEN
from bot.handler import register_handlers
from bot.db import init_db
from bot.worker import worker


# -------------------------
# 🌐 Render Health Server
# -------------------------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")


def run_web():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


# -------------------------
# 🤖 Telegram Bot
# -------------------------
app = Client(
    "video_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)


async def main():
    # Start web server in background
    threading.Thread(target=run_web, daemon=True).start()

    # Init DB
    init_db()

    # Register handlers
    register_handlers(app)

    # Start bot
    await app.start()

    # Start worker
    asyncio.create_task(worker(app))

    print("🤖 Bot + Web Server running on Render")
    await idle()


asyncio.run(main())
