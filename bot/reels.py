import os
import uuid
from yt_dlp import YoutubeDL
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from config import DOWNLOAD_DIR
from bot.video import get_video_info, resize_video

QUALITY_MAP = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}

async def video_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.message
    await context.bot.send_chat_action(m.chat.id, ChatAction.TYPING)

    file_id = str(uuid.uuid4())
    input_path = f"{DOWNLOAD_DIR}/{file_id}.mp4"

    # 📥 DOWNLOAD
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
        await m.reply_text("❌ Send a valid Reel / Short or upload a video")
        return

    # 📊 INFO
    w, h, d = get_video_info(input_path)

    context.user_data["video"] = input_path

    kb = [
        [InlineKeyboardButton("480p", callback_data="480p"),
         InlineKeyboardButton("720p", callback_data="720p"),
         InlineKeyboardButton("1080p", callback_data="1080p")]
    ]

    await m.reply_text(
        f"🎬 Video Info:\n"
        f"Resolution: {w}x{h}\n"
        f"Duration: {int(d)} sec\n\n"
        "Choose output resolution:",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    quality = q.data
    height = QUALITY_MAP[quality]

    input_path = context.user_data["video"]
    output_path = input_path.replace(".mp4", f"_{quality}.mp4")

    await q.edit_message_text("⏳ Processing video...")

    resize_video(input_path, output_path, height)

    await q.message.reply_video(
        video=open(output_path, "rb"),
        caption=f"✅ Converted to {quality}"
    )

    os.remove(input_path)
    os.remove(output_path)
