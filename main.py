import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from pyrogram import Client
from config import API_ID, API_HASH, BOT_TOKEN

import bot.handler  # IMPORTANT


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")


def run_web():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()


class Bot(Client):
    def __init__(self):
        super().__init__(
            "video-helper-bot",
            api_id=API_ID,
            api_hash=API_HASH,
            bot_token=BOT_TOKEN
        )


def main():
    threading.Thread(target=run_web, daemon=True).start()

    bot = Bot()
    print("🤖 Pyrogram Video Helper Bot running...")
    bot.run()


if __name__ == "__main__":
    main()
