import os
import cv2
import numpy as np
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN

DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

bot = Client(
    "web_video_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    in_memory=True
)

@bot.on_message(filters.command("start"))
async def start(client, message):
    await message.reply_text(
        "👋 Send me a video\n"
        "I will extract highlight images 📸"
    )

@bot.on_message(filters.video)
async def video_handler(client, message):
    msg = await message.reply("⬇️ Downloading video...")

    video_path = await message.download(file_name=f"{DOWNLOAD_DIR}/")
    await msg.edit("🎞 Processing video...")

    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    frame_no = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % 30 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = np.sum(diff)

            if motion > 3_000_000:
                img = f"{FRAME_DIR}/highlight_{saved}.jpg"
                cv2.imwrite(img, frame)
                await message.reply_photo(img)
                saved += 1

        prev_gray = gray
        if saved >= 10:
            break

    cap.release()
    await msg.edit("✅ Highlights sent")
