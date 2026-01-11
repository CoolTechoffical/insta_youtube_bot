import os

# 🔹 Telegram API (get from https://my.telegram.org)
API_ID = int(os.getenv("API_ID", "12345678"))
API_HASH = os.getenv("API_HASH", "YOUR_API_HASH")

# 🔹 Bot token (from @BotFather)
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN")

MONGO_URI = os.getenv("MONGO_URI")

MAX_IMAGES = 200
MAX_SAFE_SECONDS = 18   # Render safe
MAX_WIDTH = 1280
