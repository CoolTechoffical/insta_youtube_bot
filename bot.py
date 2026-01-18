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


# ================= CONFIG =================
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"

MAX_LIMIT = 200

SCENE_BEFORE = 2     # seconds before highlight
SCENE_AFTER = 2      # seconds after highlight
MAX_SCENES = 20
OUTPUT_RES = "720:-2"

# =========================================

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

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


# ================= SCENE BUILDER =================
def build_scenes(frames, fps):
    scenes = []
    for f in frames[:MAX_SCENES]:
        start = max(0, (f / fps) - SCENE_BEFORE)
        end = (f / fps) + SCENE_AFTER
        scenes.append((start, end))

    scenes.sort()
    merged = []
    for s, e in scenes:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged


# ================= VIDEO EDITOR =================
def create_edited_video(video_path, scenes, user_id):
    parts = []
    list_file = f"{OUTPUT_DIR}/{user_id}_list.txt"

    with open(list_file, "w") as f:
        for i, (start, end) in enumerate(scenes):
            out = f"{OUTPUT_DIR}/{user_id}_part_{i}.mp4"
            parts.append(out)

            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(start),
                "-to", str(end),
                "-vf", f"scale={OUTPUT_RES}",
                "-preset", "veryfast",
                out
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            f.write(f"file '{os.path.abspath(out)}'\n")

    final = f"{OUTPUT_DIR}/{user_id}_edited.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        final
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for p in parts:
        os.remove(p)
    os.remove(list_file)

    return final


# ================= COMMANDS =================
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 **AI Video Highlight Bot**\n\n"
        "📸 /extract – Images\n"
        "✂ /edit – Full edited video\n"
        "⚙ /settings <1-200>\n\n"
        "Reply command to a video"
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

@bot.on_callback_query(filters.regex("^cancel_"))
async def cancel_handler(_, cq):
    user_id = int(cq.data.split("_")[1])
    CANCEL_TASKS.add(user_id)
    await cq.answer("❌ Cancelled", show_alert=True)


# ================= ANALYSIS CORE =================
async def analyse_video(msg, status):
    user_id = msg.from_user.id
    target = min(get_count(user_id), MAX_LIMIT)

    video_path = await msg.reply_to_message.download(
        file_name=f"{DOWNLOAD_DIR}/"
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // (target * 3))

    prev_gray = None
    prev_color = None
    frame_no = 0
    scores = []

    while cap.isOpened():
        if user_id in CANCEL_TASKS:
            cap.release()
            return None, None

        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        total = (
            len(faces) * 120 * 1.2 +
            body_score(frame) * 1.3 +
            motion_score(prev_gray, gray) +
            scene_change_score(prev_color, frame) * 1.5 +
            nsfw_score(frame, faces) * 1.6
        )

        scores.append((total, frame_no))
        prev_gray = gray
        prev_color = frame

        if frame_no % (step * 5) == 0:
            bar = progress_bar(frame_no, total_frames)
            await status.edit(f"🎞 Analysing\n[{bar}]")

    cap.release()

    scores.sort(reverse=True)
    return [f for _, f in scores[:target]], video_path


# ================= EXTRACT =================
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

    frames, video = await analyse_video(msg, status)
    if not frames:
        return await status.edit("❌ Cancelled")

    cap = cv2.VideoCapture(video)
    saved = 0
    frames = set(frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if idx in frames:
            cv2.imwrite(f"{FRAME_DIR}/{saved}.jpg", frame)
            saved += 1
            await status.edit(
                f"📸 Extracting\n[{progress_bar(saved, len(frames))}]"
            )

    cap.release()

    zip_path = f"{OUTPUT_DIR}/{user_id}_images.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for i in range(saved):
            z.write(f"{FRAME_DIR}/{i}.jpg", f"{i+1}.jpg")

    await msg.reply_document(zip_path, caption=f"📸 {saved} images")
    await status.edit("✅ Done")

    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)


# ================= EDIT =================
@bot.on_message(filters.command("edit") & filters.reply)
async def edit(_, msg):
    user_id = msg.from_user.id
    CANCEL_TASKS.discard(user_id)

    status = await msg.reply(
        "⬇ Downloading…",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{user_id}")]
        ])
    )

    frames, video = await analyse_video(msg, status)
    if not frames:
        return await status.edit("❌ Cancelled")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    scenes = build_scenes(frames, fps)

    await status.edit("✂ Editing video…")
    final = create_edited_video(video, scenes, user_id)

    await msg.reply_document(
        final,
        caption=(
            "✅ **AI Edited Video**\n"
            f"🎞 Scenes kept: {len(scenes)}\n"
            f"⏱ Avg duration: {SCENE_BEFORE + SCENE_AFTER}s"
        )
    )

    await status.edit("✅ Editing completed")
