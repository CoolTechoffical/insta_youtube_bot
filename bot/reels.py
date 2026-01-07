import os
from yt_dlp import YoutubeDL
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from config import DOWNLOAD_DIR
from bot.keyboards import quality_keyboard
from bot.captions import generate_caption

user_links = {}

async def handle_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text.strip()

    if "instagram.com/reel" in url or "youtube.com/shorts" in url:
        user_links[update.effective_user.id] = url
        await update.message.reply_text(
            "🎞 Choose video quality:",
            reply_markup=quality_keyboard()
        )
    else:
        await update.message.reply_text("❌ Send Instagram Reel or YouTube Shorts link only")

async def quality_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    quality = query.data.split(":")[1]
    user_id = query.from_user.id
    url = user_links.get(user_id)

    if not url:
        await query.message.reply_text("⚠️ Send link again")
        return

    await context.bot.send_chat_action(query.message.chat.id, ChatAction.UPLOAD_VIDEO)

    ydl_opts = {
        "format": f"bestvideo[height<={quality}]+bestaudio/best",
        "outtmpl": f"{DOWNLOAD_DIR}/%(id)s.%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)

        caption = generate_caption(info.get("title", ""))

        await query.message.reply_video(
            video=open(file_path, "rb"),
            caption=caption
        )

        os.remove(file_path)

    except Exception:
        await query.message.reply_text("❌ Download failed for this quality")
