import os
import cv2
import shutil
import numpy as np
from threading import Thread
from flask import Flask

from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import set_count, get_count
from nsfw import nsfw_scene_score
from pose_detector import detect_body_pose
from pdf_exporter import images_to_pdf


Thread(target=run_flask, daemon=True).start()

# ---------------- PATHS ----------------
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"
MAX_LIMIT = 200

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- DETECTORS ----------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- BOT ----------------
bot = Client(
    "vision_highlight_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ---------------- HELPERS ----------------
def scene_change_score(prev, cur):
    if prev is None:
        return 0
    h1 = cv2.calcHist([prev], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([cur], [0], None, [64], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return int(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) * 150)

def fluid_motion(prev_gray, gray):
    if prev_gray is None:
        return 0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return int(np.mean(mag) * 120)

# ---------------- COMMANDS ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 Send a video\n\n"
        "🧠 Vision-only analysis\n"
        "👁 Face + Pose + Body + Motion\n"
        "💧 Fluid motion detection\n"
        "📄 Output: PDF (images only)\n\n"
        "⚙ /settings <1-200>"
    )

@bot.on_message(filters.command("settings"))
async def settings(_, msg):
    if len(msg.command) < 2:
        return await msg.reply("Usage: /settings <1-200>")

    count = int(msg.command[1])
    if count < 1 or count > MAX_LIMIT:
        return await msg.reply(f"Limit 1–{MAX_LIMIT}")

    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Extract count set to {count}")

# ---------------- VIDEO HANDLER ----------------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    target = min(get_count(user_id), MAX_LIMIT)

    status = await msg.reply("⬇️ Downloading…")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // (target * 3))

    scores = []
    frame_no = 0
    prev_gray = None
    prev_color = None

    await status.edit("🎞 Analysing frames…")

    # ---------- PASS 1: SCORING ----------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        nsfw_val, gray_out = nsfw_scene_score(frame, faces, prev_gray)
        motion_val = fluid_motion(prev_gray, gray)
        pose_tags, pose_score = detect_body_pose(frame)

        total_score = (
            nsfw_val * 1.6 +
            motion_val * 1.4 +
            pose_score * 1.3 +
            len(faces) * 150 +
            scene_change_score(prev_color, frame) * 1.2
        )

        scores.append((total_score, frame_no))
        prev_gray = gray_out
        prev_color = frame

    cap.release()

    scores.sort(reverse=True)
    selected = sorted([f for _, f in scores[:target]])

    if not selected:
        return await status.edit("❌ No highlights")

    # ---------- PASS 2: EXTRACT ----------
    cap = cv2.VideoCapture(video_path)
    current = saved = 0
    prev_gray = None
    pdf_images = []

    await status.edit("📸 Extracting…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current += 1
        if current not in selected:
            continue

        frame = cv2.resize(
            frame, (960, int(frame.shape[0] * 960 / frame.shape[1]))
        )

        img_path = f"{FRAME_DIR}/{user_id}_{saved}.jpg"
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        pdf_images.append(img_path)
        saved += 1

        if saved >= target:
            break

    cap.release()

    # ---------- PDF ----------
    pdf_path = f"{OUTPUT_DIR}/{user_id}_highlights.pdf"
    images_to_pdf(pdf_images, pdf_path)

    await msg.reply_document(
        pdf_path,
        caption=f"📄 Vision highlights\n📸 Images: {saved}"
    )

    await status.edit("✅ Done")

    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
