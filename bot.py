import os
import cv2
import numpy as np
from PIL import Image
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import set_count, get_count

DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

bot = Client(
    "video_highlight_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "👋 Send a video\n\n"
        "📸 I will extract highlight images\n"
        "📄 Then send them as a PDF\n\n"
        "⚙️ Use /settings to choose image count"
    )

@bot.on_message(filters.command("settings"))
async def settings(_, msg):
    await msg.reply(
        "⚙️ Send image count like:\n\n"
        "`/settings 5`\n"
        "`/settings 10`\n"
        "`/settings 20`",
        quote=True
    )

@bot.on_message(filters.command("settings") & filters.regex(r"\d+"))
async def set_settings(_, msg):
    count = int(msg.text.split()[-1])

    if count < 1 or count > 1000:
        await msg.reply("❌ Choose between 1 – 1000 images")
        return

    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Highlight image count set to {count}")

@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    max_images = get_count(user_id)

    status = await msg.reply("⬇️ Downloading video...")
    video_path = await msg.download(file_name=DOWNLOAD_DIR + "/")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dynamic step so we can reach 1000 images
    step = max(1, total_frames // (max_images * 3))

    await status.edit(
        "🎞 Processing video...\n"
        f"📸 Target images: {max_images}\n"
        "⏳ Progress: 0%"
    )

    prev_gray = None
    frame_no = 0
    saved = 0
    image_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = np.sum(diff)

            if motion > 2_000_000:  # softer threshold for more images
                img_path = f"{FRAME_DIR}/{user_id}_{saved}.jpg"

                # 🔥 HIGH QUALITY SAVE
                cv2.imwrite(
                    img_path,
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

                image_paths.append(img_path)
                saved += 1

                progress = int((frame_no / total_frames) * 100)
                await status.edit(
                    "🎞 Processing video...\n"
                    f"📸 Extracted: {saved}/{max_images}\n"
                    f"⏳ Progress: {progress}%"
                )

        prev_gray = gray
        if saved >= max_images:
            break

    cap.release()

    if not image_paths:
        await status.edit("❌ No highlights detected")
        return

    await status.edit("📄 Creating HIGH-QUALITY PDF...")

    images = []
    for img in image_paths:
        images.append(Image.open(img).convert("RGB"))

    pdf_path = f"{OUTPUT_DIR}/{user_id}_highlights.pdf"

    images[0].save(
        pdf_path,
        save_all=True,
        append_images=images[1:],
        resolution=300.0
    )

    await status.edit("📤 Uploading PDF...")
    await msg.reply_document(
        pdf_path,
        caption=(
            "✅ Highlight PDF Generated\n"
            f"📸 Images: {len(images)}\n"
            "🖼 Quality: High (Original)"
        )
    )

    await status.edit("✅ Completed successfully 🎉")
