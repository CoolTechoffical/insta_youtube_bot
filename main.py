import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from pyrogram import Client
from config import API_ID, API_HASH, BOT_TOKEN
from bot.handler import register_handlers


# 🔹 Render health check
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")


def run_web():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()


def main():
    threading.Thread(target=run_web, daemon=True).start()

    app = Client(
        "video-helper-bot",
        api_id=API_ID,
        api_hash=API_HASH,
        bot_token=BOT_TOKEN
    )

    # ✅ Register handlers HERE
    register_handlers(app)

    print("🤖 Bot started successfully")
    app.run()


if __name__ == "__main__":
    main()
