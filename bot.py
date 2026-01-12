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

# Haar face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

bot = Client(
    "face_body_highlight_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ---------- START ----------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "👋 Send a video\n\n"
        "✨ Face-priority highlights\n"
        "🧍 Full frame (face + body)\n"
        "📄 Output: PDF\n\n"
        "⚙️ /settings <count> (max 200)"
    )

# ---------- SETTINGS ----------
@bot.on_message(filters.command("settings") & filters.regex(r"\d+"))
async def settings(_, msg):
    count = int(msg.text.split()[-1])

    if count < 1 or count > 200:
        await msg.reply("❌ Render limit: 1 – 200 images")
        return

    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Highlight count set to {count}")

# ---------- VIDEO HANDLER ----------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    max_images = get_count(user_id)

    status = await msg.reply("⬇️ Downloading video...")
    video_path = await msg.download(file_name=DOWNLOAD_DIR + "/")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, total_frames // (max_images * 3))

    await status.edit(
        "🎞 Processing video...\n"
        f"🎯 Target: {max_images}\n"
        "⏳ Progress: 0%"
    )

    prev_gray = None
    frame_no = 0
    saved = 0
    images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # -------- FACE SCORE --------
        face_score = 0
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, fw, fh) in faces:
            face_area = fw * fh
            frame_area = w * h
            ratio = face_area / frame_area

            # Medium face = face + body visible
            if 0.02 < ratio < 0.2:
                face_score += 150
            else:
                face_score += 80

            # Face near top = body visible
            if y < h * 0.4:
                face_score += 50

        # -------- MOTION SCORE --------
        motion_score = 0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = int(np.sum(diff) / 1_000_000)

        prev_gray = gray

        # -------- SHARPNESS --------
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = 20 if sharpness > 120 else 0

        total_score = face_score + motion_score + sharp_score

        # -------- SAVE FULL FRAME --------
        if total_score > 120:
            img_path = f"{FRAME_DIR}/{user_id}_{saved}.jpg"
            cv2.imwrite(
                img_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            images.append(img_path)
            saved += 1

            progress = int((frame_no / total_frames) * 100)
            await status.edit(
                "🎞 Processing...\n"
                f"📸 Saved: {saved}/{max_images}\n"
                f"⏳ Progress: {progress}%"
            )

        if saved >= max_images:
            break

    cap.release()

    if not images:
        await status.edit("❌ No highlights detected")
        return

    # -------- CREATE PDF --------
    await status.edit("📄 Creating PDF...")
    pil_images = [Image.open(p).convert("RGB") for p in images]
    pdf_path = f"{OUTPUT_DIR}/{user_id}_highlights.pdf"

    pil_images[0].save(
        pdf_path,
        save_all=True,
        append_images=pil_images[1:],
        resolution=200
    )

    await status.edit("📤 Uploading PDF...")
    await msg.reply_document(
        pdf_path,
        caption=(
            "✅ Face + Body Highlights\n"
            f"📸 Images: {len(pil_images)}\n"
            "🖼 Full frame | High quality"
        )
    )

    await status.edit("✅ Done 🎉")
