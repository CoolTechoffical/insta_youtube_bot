import os

# 🔹 Telegram API (get from https://my.telegram.org)
API_ID = int(os.getenv("API_ID", "12345678"))
API_HASH = os.getenv("API_HASH", "YOUR_API_HASH")

# 🔹 Bot token (from @BotFather)
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN")

# 🔹 Download folder
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "downloads")

# 🔹 Safety limits (VERY IMPORTANT for Render Free)
# Max video duration allowed for FFmpeg (seconds)
MAX_CONVERT_DURATION = int(os.getenv("MAX_CONVERT_DURATION", "60"))

# Max video size allowed for FFmpeg (MB)
MAX_CONVERT_SIZE_MB = int(os.getenv("MAX_CONVERT_SIZE_MB", "25"))

# 🔹 Telegram limits (optional future use)
MAX_TELEGRAM_UPLOAD_MB = 50

# 🔹 FFmpeg binary paths (Render usually auto-detects)
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.getenv("FFPROBE_PATH", "ffprobe")
