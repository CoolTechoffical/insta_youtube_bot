import os
import cv2
import numpy as np
import mediapipe as mp
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

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.6
)

bot = Client(
    "face_body_highlight_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ---------------- START ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "👋 Send a video\n\n"
        "✨ Face-priority highlights\n"
        "🧍 Face + Body visible\n"
        "📄 Output as PDF\n\n"
        "⚙️ /settings <count> (1–1000)"
    )

# ---------------- SETTINGS ----------------
@bot.on_message(filters.command("settings") & filters.regex(r"\d+"))
async def settings(_, msg):
    count = int(msg.text.split()[-1])
    if count < 1 or count > 1000:
        await msg.reply("❌ Choose between 1 – 1000")
        return
    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Highlight count set to {count}")

# ---------------- VIDEO HANDLER ----------------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    max_images = get_count(user_id)

    status = await msg.reply("⬇️ Downloading video...")
    video_path = await msg.download(file_name=DOWNLOAD_DIR + "/")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // (max_images * 2))

    await status.edit(
        "🎞 Processing video...\n"
        f"🎯 Target: {max_images}\n"
        "⏳ Progress: 0%"
    )

    prev_gray = None
    frame_no = 0
    saved = 0
    collected = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------- FACE DETECTION ----------
        face_score = 0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mp_face.process(rgb)

        if faces.detections:
            for d in faces.detections:
                box = d.location_data.relative_bounding_box
                face_area = (box.width * w) * (box.height * h)
                frame_area = w * h
                ratio = face_area / frame_area

                # Medium face = best (face + body)
                if 0.02 < ratio < 0.2:
                    face_score += 150
                else:
                    face_score += 80

                # Face position (top = body visible)
                if box.ymin < 0.4:
                    face_score += 50

        # ---------- MOTION ----------
        motion_score = 0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = int(np.sum(diff) / 1_000_000)

        prev_gray = gray

        # ---------- SHARPNESS ----------
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = 20 if sharpness > 150 else 0

        total_score = face_score + motion_score + sharp_score

        # ---------- SAVE FULL FRAME ----------
        if total_score > 120:
            img_path = f"{FRAME_DIR}/{user_id}_{saved}.jpg"
            cv2.imwrite(
                img_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            collected.append(img_path)
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

    if not collected:
        await status.edit("❌ No highlights detected")
        return

    # ---------- CREATE PDF ----------
    await status.edit("📄 Creating PDF...")
    images = [Image.open(p).convert("RGB") for p in collected]
    pdf_path = f"{OUTPUT_DIR}/{user_id}_highlights.pdf"

    images[0].save(
        pdf_path,
        save_all=True,
        append_images=images[1:],
        resolution=300
    )

    await status.edit("📤 Uploading PDF...")
    await msg.reply_document(
        pdf_path,
        caption=(
            "✅ Face + Body Highlight PDF\n"
            f"📸 Images: {len(images)}\n"
            "🖼 Full Frame | High Quality"
        )
    )

    await status.edit("✅ Completed 🎉")
