import os
from yt_dlp import YoutubeDL
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from config import DOWNLOAD_DIR, BOT_USERNAME


os.makedirs(DOWNLOAD_DIR, exist_ok=True)


async def download_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.message
    url = m.text.strip()

    if not ("instagram.com" in url or "youtube.com" in url or "youtu.be" in url):
        await m.reply_text("❌ Send a valid Instagram or YouTube link")
        return

    await context.bot.send_chat_action(m.chat.id, ChatAction.TYPING)

    try:
        ydl_opts = {
            "format": "mp4/best",
            "outtmpl": f"{DOWNLOAD_DIR}/%(title)s.%(ext)s",
            "quiet": True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)

        await m.reply_video(
            video=open(file_path, "rb"),
            caption=f"✅ Download completed\n\n🔗 Powered by {BOT_USERNAME}"
        )

        os.remove(file_path)

    except Exception:
        await m.reply_text("❌ Download failed. Try another link.")
