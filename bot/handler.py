# bot/handler.py
import os
from pyrogram import filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from bot.db import add_job
from bot.video import get_video_info
from config import DOWNLOAD_DIR, MAX_DB_SIZE_MB, MAX_DB_DURATION

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

QUALITY_MAP = {
    "480p": "480p",
    "720p": "720p",
    "1080p": "1080p"
}

def register_handlers(app):

    @app.on_message(filters.command("start"))
    async def start(_, message):
        await message.reply_text(
            "👋 Welcome\n\n"
            "🎬 Send a video\n"
            "📦 Up to 50MB / 15 min stored safely"
        )

    @app.on_message(filters.video)
    async def video_handler(_, message):
        v = message.video
        size_mb = v.file_size / (1024 * 1024)

        if size_mb > MAX_DB_SIZE_MB:
            await message.reply_text("❌ Video too large for free processing")
            return

        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("480p", callback_data="480p"),
                InlineKeyboardButton("720p", callback_data="720p"),
                InlineKeyboardButton("1080p", callback_data="1080p")
            ]
        ])

        await message.reply_text(
            f"📊 Video received\n"
            f"Duration: {v.duration}s\n"
            f"Size: {round(size_mb,2)} MB\n\n"
            "Choose quality:",
            reply_markup=kb
        )

    @app.on_callback_query()
    async def callback(_, cq):
        q = cq.data
        m = cq.message
        v = m.reply_to_message.video

        if v.duration > MAX_DB_DURATION:
            await cq.answer("Too long for conversion", show_alert=True)
            return

        add_job({
            "user_id": cq.from_user.id,
            "chat_id": m.chat.id,
            "file_id": v.file_id,
            "file_unique_id": v.file_unique_id,
            "file_size": v.file_size,
            "duration": v.duration,
            "requested_quality": q
        })

        await cq.message.edit_text(
            "✅ Added to queue\n"
            "⏳ Will process when server is free"
        )
