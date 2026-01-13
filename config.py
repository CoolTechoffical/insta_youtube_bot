import os

# 🔹 Telegram API (get from https://my.telegram.org)
API_ID = int(os.getenv("API_ID", "12345678"))
API_HASH = os.getenv("API_HASH", "YOUR_API_HASH")

# 🔹 Bot token (from @BotFather)
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN")
DOWNLOAD_DIR = "downloads"
OUTPUT_DIR = "output"

MAX_FRAMES_LIMIT = 200  # hard server safety limit
DEFAULT_EXTRACT_COUNT = 25
