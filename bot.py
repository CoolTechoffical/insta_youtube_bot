import os
import cv2
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from detector import detect_image_ai, detect_video_ai

bot = Client(
    "ai_detector",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🤖 **AI Image & Video Detector**\n\n"
        "📤 Send an image or video\n"
        "🧠 Detects AI vs Real\n"
        "🔍 Predicts AI tool\n\n"
        "_Prediction only, not guaranteed._"
    )

@bot.on_message(filters.photo)
async def image_handler(_, msg):
    status = await msg.reply("🔍 Analyzing image...")
    path = await msg.download(DOWNLOAD_DIR)

    result = detect_image_ai(path)

    await status.edit(
        f"🖼 **Image Scan Result**\n\n"
        f"🤖 AI Generated: **{result['is_ai']}**\n"
        f"🧠 Tool: **{result['tool']}**\n"
        f"📊 Confidence: **{result['confidence']}%**\n"
        f"🆔 Synthetic ID: `{result['synthetic_id']}`\n\n"
        f"🔍 Detection Signals:\n• " + "\n• ".join(result["signals"])
    )

    os.remove(path)

@bot.on_message(filters.video)
async def video_handler(_, msg):
    status = await msg.reply("🎥 Processing video...")
    video_path = await msg.download(DOWNLOAD_DIR)

    cap = cv2.VideoCapture(video_path)
    results = []

    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = f"{DOWNLOAD_DIR}/frame_{i}.jpg"
        cv2.imwrite(frame_path, frame)

        results.append(detect_image_ai(frame_path))
        os.remove(frame_path)

    cap.release()
    os.remove(video_path)

    final = detect_video_ai(results)

    await status.edit(
        f"🎥 **Video Result**\n\n"
        f"🤖 AI Generated: **{final['is_ai']}**\n"
        f"🧠 Tool: **{final['tool']}**\n"
        f"📊 Confidence: **{final['confidence']}%**"
    )
