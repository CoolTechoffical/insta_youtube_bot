import os
import uuid
import asyncio
from yt_dlp import YoutubeDL
from pyrogram import filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from config import DOWNLOAD_DIR, MAX_CONVERT_DURATION, MAX_CONVERT_SIZE_MB
from bot.video import get_video_info, resize_video

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

QUALITY_MAP = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080
}


async def start(client, message):
    await message.reply_text(
        "🎬 Send an Instagram Reel, YouTube Short, or upload a video.\n\n"
        "⚠️ Large / long videos will be sent without conversion."
    )


async def video_handler(client, message):
    status = await message.reply_text("📥 Receiving video...")

    file_id = str(uuid.uuid4())
    input_path = f"{DOWNLOAD_DIR}/{file_id}.mp4"

    # 🔹 Download
    if message.text and any(x in message.text for x in ["instagram.com", "youtu.be", "youtube.com"]):
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
        await status.edit_text("❌ Send a valid link or upload a video")
        return

    await status.edit_text("📊 Analyzing video...")

    # 🔹 Info
    w, h, d = get_video_info(input_path)
    size_mb = os.path.getsize(input_path) / (1024 * 1024)

    # 🔹 Long / large → send original
    if d > MAX_CONVERT_DURATION or size_mb > MAX_CONVERT_SIZE_MB:
        await status.edit_text(
            "⚠️ Large or long video detected.\n"
            "🚀 Sending original file."
        )
        await message.reply_document(input_path)
        os.remove(input_path)
        return

    # 🔹 Store temp data
    message._client.storage[message.from_user.id] = {
        "video": input_path
    }

    keyboard = InlineKeyboardMarkup([
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
        "Choose output quality:",
        reply_markup=keyboard
    )


async def callback_handler(client, callback):
    await callback.answer()

    user_id = callback.from_user.id
    quality = callback.data
    height = QUALITY_MAP[quality]

    data = client.storage.get(user_id)
    if not data:
        await callback.message.edit_text("❌ Session expired")
        return

    input_path = data["video"]
    output_path = input_path.replace(".mp4", f"_{quality}.mp4")

    await callback.message.edit_text("⏳ Processing video...")

    # 🔹 Run FFmpeg async
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, resize_video, input_path, output_path, height
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
    client.storage.pop(user_id, None)
