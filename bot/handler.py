import os
import uuid
import asyncio
from yt_dlp import YoutubeDL

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from config import DOWNLOAD_DIR, MAX_CONVERT_DURATION, MAX_CONVERT_SIZE_MB
from bot.video import get_video_info, resize_video

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

QUALITY_MAP = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎬 Send an Instagram Reel, YouTube Short, or upload a video.\n\n"
        "⚠️ Large / long videos will be sent without conversion."
    )


async def video_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.message
    await context.bot.send_chat_action(m.chat.id, ChatAction.TYPING)

    file_id = str(uuid.uuid4())
    input_path = f"{DOWNLOAD_DIR}/{file_id}.mp4"

    status = await m.reply_text("📥 Receiving video...")

    # 🔹 DOWNLOAD
    if m.text and ("instagram.com" in m.text or "youtu.be" in m.text or "youtube.com" in m.text):
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": input_path,
            "quiet": True,
            "noplaylist": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([m.text])

    elif m.video:
        file = await m.video.get_file()
        await file.download_to_drive(input_path)

    else:
        await status.edit_text("❌ Send a valid link or upload a video")
        return

    await status.edit_text("📊 Analyzing video...")

    # 🔹 INFO
    w, h, d = get_video_info(input_path)
    size_mb = os.path.getsize(input_path) / (1024 * 1024)

    # 🔹 LONG / LARGE → DIRECT SEND
    if d > MAX_CONVERT_DURATION or size_mb > MAX_CONVERT_SIZE_MB:
        await status.edit_text(
            "⚠️ Large or long video detected.\n"
            "🚀 Sending original file (conversion disabled)."
        )
        await m.reply_document(open(input_path, "rb"))
        os.remove(input_path)
        return

    context.user_data["video"] = input_path
    context.user_data["duration"] = d

    kb = [
        [
            InlineKeyboardButton("480p", callback_data="480p"),
            InlineKeyboardButton("720p", callback_data="720p"),
            InlineKeyboardButton("1080p", callback_data="1080p"),
        ]
    ]

    await status.edit_text(
        f"🎬 Video Ready\n"
        f"Resolution: {w}x{h}\n"
        f"Duration: {int(d)} sec\n\n"
        "Choose output quality:",
        reply_markup=InlineKeyboardMarkup(kb)
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    quality = q.data
    height = QUALITY_MAP[quality]

    input_path = context.user_data["video"]
    output_path = input_path.replace(".mp4", f"_{quality}.mp4")

    await q.edit_message_text("⏳ Processing video...\n(This may take time)")

    # 🔹 FAKE PROGRESS BAR
    async def progress():
        for msg in ["⏳ Still working...", "⏳ Almost done..."]:
            await asyncio.sleep(8)
            try:
                await q.message.edit_text(msg)
            except:
                pass

    task = asyncio.create_task(progress())

    try:
        resize_video(input_path, output_path, height)
    finally:
        task.cancel()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    if size_mb > 45:
        await q.message.reply_document(
            open(output_path, "rb"),
            caption=f"✅ Converted to {quality}"
        )
    else:
        await q.message.reply_video(
            open(output_path, "rb"),
            caption=f"✅ Converted to {quality}"
        )

    os.remove(input_path)
    os.remove(output_path)
