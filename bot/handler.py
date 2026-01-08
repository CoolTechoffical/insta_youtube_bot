import os
import uuid
import asyncio

from pyrogram import Client, filters
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


@Client.on_message(filters.command("start"))
async def start(client, message):
    await message.reply_text(
        "👋 Welcome!\n\n"
        "🎬 Send an Instagram Reel, YouTube Short, or upload a video.\n"
        "⚠️ Large videos will be sent without conversion."
    )


@Client.on_message(filters.video | filters.text)
async def video_handler(client, message):
    status = await message.reply_text("📥 Receiving video...")
    file_id = str(uuid.uuid4())
    input_path = f"{DOWNLOAD_DIR}/{file_id}.mp4"

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
        await status.edit_text("❌ Send a valid video or link")
        return

    await status.edit_text("📊 Analyzing video...")

    width, height, duration = get_video_info(input_path)
    size_mb = os.path.getsize(input_path) / (1024 * 1024)

    # 🔹 Too large / long → send original
    if duration > MAX_CONVERT_DURATION or size_mb > MAX_CONVERT_SIZE_MB:
        await status.edit_text(
            "⚠️ Large or long video detected.\n"
            "🚀 Sending original file."
        )
        await message.reply_document(input_path)
        os.remove(input_path)
        return

    USER_DATA[message.from_user.id] = input_path

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("480p", callback_data="480p"),
            InlineKeyboardButton("720p", callback_data="720p"),
            InlineKeyboardButton("1080p", callback_data="1080p")
        ]
    ])

    await status.edit_text(
        f"🎬 Video Ready\n"
        f"Resolution: {width}x{height}\n"
        f"Duration: {int(duration)} sec\n\n"
        "Choose output quality:",
        reply_markup=keyboard
    )


@Client.on_callback_query()
async def callback_handler(client, callback):
    await callback.answer()

    user_id = callback.from_user.id
    quality = callback.data

    if user_id not in USER_DATA:
        await callback.message.edit_text("❌ Session expired")
        return

    input_path = USER_DATA[user_id]
    output_path = input_path.replace(".mp4", f"_{quality}.mp4")

    await callback.message.edit_text("⏳ Processing video...")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        resize_video,
        input_path,
        output_path,
        QUALITY_MAP[quality]
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    if size_mb > 45:
        await callback.message.reply_document(
            output_path,
            caption=f"✅ Converted to {quality}"
        )
    else:
        await callback.message.reply_video(
            output_path,
            caption=f"✅ Converted to {quality}",
            supports_streaming=True
        )

    os.remove(input_path)
    os.remove(output_path)
    USER_DATA.pop(user_id, None)
