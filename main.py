import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from pyrogram import Client
from config import API_ID, API_HASH, BOT_TOKEN

import bot.handler # IMPORTANT: registers handlers


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")


def run_web():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


def main():
    threading.Thread(target=run_web, daemon=True).start()

    app = Client(
        "video-helper-bot",
        api_id=API_ID,
        api_hash=API_HASH,
        bot_token=BOT_TOKEN
    )

    print("🤖 Pyrogram Video Helper Bot running...")
    app.run()


if __name__ == "__main__":
    main()
