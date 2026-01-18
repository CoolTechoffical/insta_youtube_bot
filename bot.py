import os
import cv2
import zipfile
import shutil
import numpy as np
import subprocess

from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import get_count, set_count
from nsfw import nsfw_score

# ---------------- PATHS ----------------
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"
CLIP_DIR = "clips"

MAX_LIMIT = 200

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR, CLIP_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- GLOBAL CANCEL ----------------
CANCEL_TASKS = set()

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
def progress_bar(done, total, size=10):
    if total == 0:
        return "░" * size
    filled = int(size * done / total)
    return "█" * filled + "░" * (size - filled)

def scene_change_score(prev_frame, frame):
    if prev_frame is None:
        return 0
    h1 = cv2.calcHist([prev_frame], [0], None, [64], [0,256])
    h2 = cv2.calcHist([frame], [0], None, [64], [0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return int(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) * 150)

def motion_score(prev_gray, gray):
    if prev_gray is None:
        return 0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return int(np.mean(mag) * 30)

def body_score(frame):
    boxes, _ = hog.detectMultiScale(
        frame, winStride=(8,8), padding=(16,16), scale=1.05
    )
    score = 0
    h, w, _ = frame.shape
    for (x,y,bw,bh) in boxes:
        ratio = (bw * bh) / (w * h)
        score += 150 if ratio > 0.15 else 80
    return score

# ---------------- COMMANDS ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 **AI Highlight Bot**\n\n"
        "Reply to a video with:\n"
        "📸 `/extract` – Highlight images\n"
        "✂ `/edit` – AI auto-cut video clips\n\n"
        "⚙ `/settings <1-200>`"
    )

@bot.on_message(filters.command("settings"))
async def settings(_, msg):
    if len(msg.command) < 2:
        return await msg.reply("Usage: /settings <1-200>")
    count = int(msg.command[1])
    if count < 1 or count > MAX_LIMIT:
        return await msg.reply("❌ Max 200")
    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Image count set to {count}")

# ---------------- CANCEL ----------------
@bot.on_callback_query(filters.regex("^cancel_"))
async def cancel_handler(_, cq):
    user_id = int(cq.data.split("_")[1])
    CANCEL_TASKS.add(user_id)
    await cq.answer("❌ Cancelled", show_alert=True)

# ======================================================
# =============== SHARED ANALYSIS LOGIC ================
# ======================================================
async def analyse_video(msg, status):
    user_id = msg.from_user.id
    target = min(get_count(user_id), MAX_LIMIT)

    video_path = await msg.reply_to_message.download(file_name=DOWNLOAD_DIR)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // (target * 3))

    frame_no = 0
    prev_gray = None
    prev_color = None
    scores = []

    while cap.isOpened():
        if user_id in CANCEL_TASKS:
            cap.release()
            return None, video_path

        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        people, _ = hog.detectMultiScale(frame)

        score = (
            len(faces) * 120 +
            body_score(frame) +
            motion_score(prev_gray, gray) +
            scene_change_score(prev_color, frame) +
            nsfw_score(frame, faces, people, prev_color, prev_gray)
        )

        scores.append((score, frame_no))
        prev_gray = gray
        prev_color = frame

        if frame_no % (step * 5) == 0:
            bar = progress_bar(frame_no, total_frames)
            percent = int((frame_no / total_frames) * 100)
            await status.edit(f"🎞 Analysing…\n[{bar}] {percent}%")

    cap.release()
    scores.sort(reverse=True)

    frames = sorted([f for _, f in scores[:target]])
    return frames, video_path

# ======================================================
# ================= /extract ===========================
# ======================================================
@bot.on_message(filters.command("extract") & filters.reply)
async def extract(_, msg):
    user_id = msg.from_user.id
    CANCEL_TASKS.discard(user_id)

    status = await msg.reply(
        "⬇ Downloading…",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{user_id}")]
        ])
    )

    frames, video_path = await analyse_video(msg, status)
    if not frames:
        return await status.edit("❌ Cancelled")

    cap = cv2.VideoCapture(video_path)
    saved = 0

    for i, fno in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imwrite(f"{FRAME_DIR}/{user_id}_{i}.jpg", frame)
        saved += 1

        bar = progress_bar(saved, len(frames))
        await status.edit(f"📸 Extracting…\n[{bar}]")

    cap.release()

    zip_path = f"{OUTPUT_DIR}/{user_id}_highlights.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for i in range(saved):
            z.write(f"{FRAME_DIR}/{user_id}_{i}.jpg", f"{i+1}.jpg")

    await msg.reply_document(zip_path, caption=f"✅ {saved} images extracted")
    await status.edit("✅ Done")

# ======================================================
# ================= /edit (AI AUTOCUT) =================
# ======================================================
@bot.on_message(filters.command("edit") & filters.reply)
async def edit(_, msg):
    user_id = msg.from_user.id
    CANCEL_TASKS.discard(user_id)

    status = await msg.reply(
        "⬇ Downloading video…",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{user_id}")]
        ])
    )

    frames, video_path = await analyse_video(msg, status)
    if not frames:
        return await status.edit("❌ Cancelled")

    await status.edit("✂ Creating AI video clips…")

    clips_sent = 0
    fps = 30

    for i, fno in enumerate(frames[:10]):  # max 10 clips
        start = max(0, fno - fps * 2)
        duration = 4

        out = f"{CLIP_DIR}/{user_id}_clip_{i}.mp4"

        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start / fps),
            "-t", str(duration),
            "-vf", "scale=720:-2",
            "-preset", "veryfast",
            out
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        await msg.reply_video(out, caption=f"🎬 AI Clip {i+1}")
        clips_sent += 1

    await status.edit(f"✅ {clips_sent} AI clips created")
