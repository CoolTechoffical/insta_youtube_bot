import os
import asyncio
from yt_dlp import YoutubeDL
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from config import DOWNLOAD_DIR, BOT_USERNAME

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def ytdlp_download(url: str):
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(title).80s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0"
        }
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


async def download_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.message
    url = m.text.strip()

    if not (
        "instagram.com" in url
        or "youtube.com" in url
        or "youtu.be" in url
    ):
        await m.reply_text("❌ Send a valid Instagram or YouTube link")
        return

    await context.bot.send_chat_action(
        chat_id=m.chat.id,
        action=ChatAction.TYPING
    )

    loop = asyncio.get_running_loop()

    try:
        file_path = await loop.run_in_executor(
            None, ytdlp_download, url
        )

        await m.reply_video(
            video=open(file_path, "rb"),
            caption=f"✅ Download completed\n\n🔗 Powered by {BOT_USERNAME}",
            supports_streaming=True
        )

        os.remove(file_path)

    except Exception as e:
        await m.reply_text(
            "❌ Download failed.\n"
            "Some YouTube videos require login.\n"
            "Try another link."
        )
