import os
import uuid
import asyncio
from yt_dlp import YoutubeDL

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from config import DOWNLOAD_DIR
from bot.video import get_video_info, resize_video

# Ensure directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

QUALITY_MAP = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}

# -------- MAIN ENTRY --------
async def video_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.message

    # Instant response (prevents Telegram timeout)
    status = await m.reply_text("⏳ Processing your video...")

    # Run heavy work in background
    asyncio.create_task(
        _process_video(update, context, status)
    )


# -------- BACKGROUND PROCESS --------
async def _process_video(update: Update, context: ContextTypes.DEFAULT_TYPE, status_msg):
    m = update.message

    file_id = str(uuid.uuid4())
    input_path = f"{DOWNLOAD_DIR}/{file_id}.mp4"

    try:
        await context.bot.send_chat_action(m.chat.id, ChatAction.TYPING)

        # 📥 DOWNLOAD
        if m.text and any(x in m.text for x in ("instagram.com", "youtu.be", "youtube.com")):
            ydl_opts = {
                "format": "mp4/best",
                "outtmpl": input_path,
                "quiet": True,
                "noplaylist": True,
                "merge_output_format": "mp4",
            }

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([m.text])

        elif m.video:
            file = await m.video.get_file()
            await file.download_to_drive(input_path)

        else:
            await status_msg.edit_text("❌ Send a valid Reel / Short or upload a video")
            return

        # 📊 VIDEO INFO
        width, height, duration = get_video_info(input_path)
        size_mb = os.path.getsize(input_path) / (1024 * 1024)

        context.user_data["video_path"] = input_path
        context.user_data["video_size"] = size_mb
        context.user_data["video_height"] = height

        # ❌ LARGE VIDEO → NO RE-ENCODE
        if size_mb > 100 or duration > 900:
            await status_msg.edit_text(
                f"🎬 Video received\n\n"
                f"Resolution: {width}x{height}\n"
                f"Duration: {int(duration)} sec\n"
                f"Size: {int(size_mb)} MB\n\n"
                "⚠️ Large videos are sent in original quality only."
            )

            await context.bot.send_video(
                chat_id=m.chat.id,
                video=open(input_path, "rb"),
                caption="✅ Original quality"
            )

            os.remove(input_path)
            return

        # ✅ SHOW QUALITY OPTIONS (SAFE ONLY)
        keyboard = [
            [
                InlineKeyboardButton("480p", callback_data="q:480p"),
                InlineKeyboardButton("720p", callback_data="q:720p"),
                InlineKeyboardButton("1080p", callback_data="q:1080p"),
            ]
        ]

        await status_msg.edit_text(
            f"🎬 Video Info\n\n"
            f"Resolution: {width}x{height}\n"
            f"Duration: {int(duration)} sec\n"
            f"Size: {int(size_mb)} MB\n\n"
            "Choose output resolution:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    except Exception as e:
        await status_msg.edit_text(f"❌ Failed: {e}")
        if os.path.exists(input_path):
            os.remove(input_path)


# -------- CALLBACK HANDLER --------
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if not q.data.startswith("q:"):
        return

    quality = q.data.split(":")[1]
    height = QUALITY_MAP.get(quality)

    input_path = context.user_data.get("video_path")
    size_mb = context.user_data.get("video_size", 0)

    if not input_path or not os.path.exists(input_path):
        await q.edit_message_text("❌ Video expired. Please send again.")
        return

    # Safety check
    if size_mb > 100:
        await q.edit_message_text("⚠️ Quality conversion disabled for large videos.")
        return

    output_path = input_path.replace(".mp4", f"_{quality}.mp4")

    try:
        await q.edit_message_text("⏳ Converting video...")

        # Run blocking ffmpeg safely
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            resize_video,
            input_path,
            output_path,
            height,
        )

        await q.message.reply_video(
            video=open(output_path, "rb"),
            caption=f"✅ Converted to {quality}"
        )

    except Exception as e:
        await q.message.reply_text(f"❌ Conversion failed: {e}")

    finally:
        for f in (input_path, output_path):
            if f and os.path.exists(f):
                os.remove(f)
