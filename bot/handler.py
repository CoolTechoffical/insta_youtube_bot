import os
import uuid
import asyncio

from pyrogram import filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from yt_dlp import YoutubeDL

from config import DOWNLOAD_DIR, MAX_CONVERT_DURATION, MAX_CONVERT_SIZE_MB
from bot.video import get_video_info, resize_video

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

USER_DATA = {}

QUALITY_MAP = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080
}


def register_handlers(app):

    @app.on_message(filters.command("start"))
    async def start(_, message):
        await message.reply_text(
            "👋 Welcome!\n\n"
            "🎬 Send Instagram Reel / YouTube link or upload a video.\n"
            "⚠️ Large videos will be sent without conversion."
        )

    @app.on_message(filters.video | filters.text)
    async def video_handler(_, message):
        status = await message.reply_text("📥 Receiving video...")
        uid = str(uuid.uuid4())
        input_path = f"{DOWNLOAD_DIR}/{uid}.mp4"

        # 🔹 Download
        if message.text and any(x in message.text for x in ["instagram.com", "youtube.com", "youtu.be"]):
            ydl_opts = {
                "format": "best[ext=mp4]/best",
                "outtmpl": input_path,
                "quiet": True,
                "noplaylist": True
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([message.text])

        elif message.video:
            await message.video.download(input_path)
        else:
            await status.edit_text("❌ Invalid input")
            return

        await status.edit_text("📊 Analyzing video...")

        w, h, d = get_video_info(input_path)
        size_mb = os.path.getsize(input_path) / (1024 * 1024)

        # 🔹 Safety check
        if d > MAX_CONVERT_DURATION or size_mb > MAX_CONVERT_SIZE_MB:
            await status.edit_text("🚀 Sending original video")
            await message.reply_document(input_path)
            os.remove(input_path)
            return

        USER_DATA[message.from_user.id] = input_path

        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("480p", callback_data="480p"),
                InlineKeyboardButton("720p", callback_data="720p"),
                InlineKeyboardButton("1080p", callback_data="1080p"),
            ]
        ])

        await status.edit_text(
            f"🎬 Video Ready\n"
            f"Resolution: {w}x{h}\n"
            f"Duration: {int(d)} sec\n\n"
            "Choose quality:",
            reply_markup=kb
        )

    @app.on_callback_query()
    async def callback_handler(_, callback):
        await callback.answer()
        user_id = callback.from_user.id

        if user_id not in USER_DATA:
            await callback.message.edit_text("❌ Session expired")
            return

        quality = callback.data
        input_path = USER_DATA[user_id]
        output_path = input_path.replace(".mp4", f"_{quality}.mp4")

        await callback.message.edit_text("⏳ Processing...")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            resize_video,
            input_path,
            output_path,
            QUALITY_MAP[quality]
        )

        await callback.message.reply_video(
            output_path,
            caption=f"✅ Converted to {quality}",
            supports_streaming=True
        )

        os.remove(input_path)
        os.remove(output_path)
        USER_DATA.pop(user_id)
