import os
import cv2
import shutil
import numpy as np
from flask import Flask
from threading import Thread

from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import set_count, get_count
from nsfw import nsfw_scene_score
from caption_engine import get_caption
from pdf_exporter import images_to_pdf


# ---------------- PATHS ----------------
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"
MAX_LIMIT = 200

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- DETECTORS ----------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ---------------- BOT ----------------
bot = Client(
    "highlight_bot",
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

def body_score(frame):
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8),
                                    padding=(16, 16), scale=1.05)
    score = 0
    h, w, _ = frame.shape
    for (_, _, bw, bh) in boxes:
        ratio = (bw * bh) / (w * h)
        score += 200 if ratio > 0.15 else 100
    return score

def fluid_motion(prev_gray, gray):
    if prev_gray is None:
        return 0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return int(np.mean(mag) * 120)

# ---------------- COMMANDS ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 Send a video\n\n"
        "🔥 NSFW + Pose + Motion detection\n"
        "🧠 Smart captions\n"
        "📄 Output: PDF with captions\n\n"
        "⚙ /settings <1-200>"
    )

@bot.on_message(filters.command("settings"))
async def settings(_, msg):
    if len(msg.command) < 2:
        return await msg.reply("Usage: /settings <1-200>")
    count = int(msg.command[1])
    if count < 1 or count > MAX_LIMIT:
        return await msg.reply(f"❌ Max {MAX_LIMIT}")
    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Image count set to {count}")

# ---------------- VIDEO HANDLER ----------------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    target = min(get_count(user_id), MAX_LIMIT)

    status = await msg.reply("⬇️ Downloading video…")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // (target * 3))

    scores = []
    frame_no = 0
    prev_gray = None
    prev_color = None

    await status.edit("🎞 Analysing video…")

    # -------- PASS 1: SCORE --------
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
        body_val = body_score(frame)

        total = (
            nsfw_val * 1.8 +
            motion_val * 1.4 +
            body_val * 1.2 +
            len(faces) * 150 +
            scene_change_score(prev_color, frame) * 1.2
        )

        scores.append((total, frame_no))
        prev_gray = gray_out
        prev_color = frame

    cap.release()
    scores.sort(reverse=True)
    selected = sorted([f for _, f in scores[:target]])

    if not selected:
        return await status.edit("❌ No highlights detected")

    # -------- PASS 2: EXTRACT + CAPTION --------
    cap = cv2.VideoCapture(video_path)
    current = saved = 0
    prev_gray = None
    pdf_data = []

    await status.edit("📸 Extracting highlights…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current += 1
        if current not in selected:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        nsfw_val, gray_out = nsfw_scene_score(frame, faces, prev_gray)
        motion_val = fluid_motion(prev_gray, gray)
        prev_gray = gray_out

        tags = []
        if nsfw_val > 700:
            tags.append("adult_scene")
        elif nsfw_val > 450:
            tags.append("intimate_pose")
        elif nsfw_val > 300:
            tags.append("suggestive")
        if motion_val > 200:
            tags.append("high_motion")
        if body_score(frame) > 250:
            tags.append("pose_lower")

        caption = get_caption(tags)

        img_path = f"{FRAME_DIR}/{user_id}_{saved}.jpg"
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        pdf_data.append((img_path, caption))

        saved += 1
        if saved >= target:
            break

    cap.release()

    # -------- PDF OUTPUT --------
    pdf_path = f"{OUTPUT_DIR}/{user_id}_highlights.pdf"
    images_to_pdf(pdf_data, pdf_path)

    await msg.reply_document(
        pdf_path,
        caption=f"📄 Highlights PDF\n📸 Images: {saved}"
    )

    await status.edit("✅ Done")

    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
