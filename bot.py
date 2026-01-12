import os
import cv2
import json
import shutil
import numpy as np
from PIL import Image
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import set_count, get_count

# ---------------- PATHS ----------------
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"
TASKS_FILE = "tasks.json"

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- SAFE JSON LOAD ----------------
def load_tasks():
    if not os.path.exists(TASKS_FILE):
        return {}
    try:
        with open(TASKS_FILE, "r") as f:
            data = f.read().strip()
            if not data:
                return {}
            return json.loads(data)
    except Exception:
        print("⚠️ tasks.json corrupted, resetting")
        return {}

tasks = load_tasks()

# ---------------- SAFE JSON SAVE ----------------
def save_tasks():
    tmp = TASKS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(tasks, f)
    os.replace(tmp, TASKS_FILE)

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- BOT ----------------
bot = Client(
    "auto_resume_highlight_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ---------------- START ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 Send a video\n"
        "✨ Face-priority highlights\n"
        "🧍 Full frame (face + body)\n"
        "📄 PDF output\n\n"
        "⚙ /settings <1-200>"
    )

# ---------------- SETTINGS ----------------
@bot.on_message(filters.command("settings"))
async def settings(_, msg):
    if len(msg.command) < 2:
        await msg.reply("Usage: /settings <1-200>")
        return

    count = int(msg.command[1])
    if count < 1 or count > 200:
        await msg.reply("❌ Max 200 images (Render limit)")
        return

    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Image count set to {count}")

# ---------------- VIDEO ----------------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = str(msg.from_user.id)
    max_images = get_count(msg.from_user.id)

    status = await msg.reply("⬇️ Downloading video...")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    tasks[user_id] = {
        "video": video_path,
        "frame": 0,
        "images": [],
        "max": max_images,
        "status": "processing"
    }
    save_tasks()

    await process_video(user_id, msg, status)

# ---------------- PROCESS VIDEO ----------------
async def process_video(user_id, msg, status):
    task = tasks[user_id]
    cap = cv2.VideoCapture(task["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, task["frame"])

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // (task["max"] * 3))
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        task["frame"] += 1
        if task["frame"] % step != 0:
            continue

        # Resize for Render
        h, w, _ = frame.shape
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(
                frame,
                (1280, int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face score
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60,60))
        face_score = sum(150 if 0.02 < (fw*fh)/(h*w) < 0.2 else 80 for (_,_,fw,fh) in faces)

        # Motion score
        motion = 0
        if prev_gray is not None:
            motion = int(np.sum(cv2.absdiff(prev_gray, gray)) / 1_000_000)
        prev_gray = gray

        # Sharpness
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = 20 if sharp > 120 else 0

        if face_score + motion + sharp_score > 120:
            img_path = f"{FRAME_DIR}/{user_id}_{len(task['images'])}.jpg"
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            task["images"].append(img_path)
            save_tasks()

            progress = int((task["frame"] / total) * 100)
            await status.edit(
                f"📸 {len(task['images'])}/{task['max']} images\n⏳ {progress}%"
            )

        if len(task["images"]) >= task["max"]:
            break

        # Auto pause before Render kills process
        if len(task["images"]) >= 180:
            await status.edit("⚠️ Server busy. Auto-resume enabled.")
            task["status"] = "paused"
            save_tasks()
            cap.release()
            return

    cap.release()

    if not task["images"]:
        task["status"] = "done"
        save_tasks()
        await status.edit("❌ No highlights found")
        return

    # ---------------- PDF ----------------
    await status.edit("📄 Creating PDF...")
    imgs = [Image.open(p).convert("RGB") for p in task["images"]]
    pdf_path = f"{OUTPUT_DIR}/{user_id}_highlights.pdf"
    imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:], resolution=150)

    await msg.reply_document(
        pdf_path,
        caption=f"✅ {len(imgs)} highlights\n🧍 Face + body"
    )

    task["status"] = "done"
    save_tasks()

    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)

# ---------------- AUTO RESUME ----------------
async def resume_tasks():
    for uid, task in tasks.items():
        if task["status"] in ("processing", "paused"):
            try:
                await bot.send_message(int(uid), "🔄 Resuming your previous task...")
                dummy = await bot.get_messages(int(uid), 1)
                await process_video(uid, dummy, await bot.send_message(int(uid), "▶️ Resumed"))
            except Exception as e:
                print("Resume failed:", uid, e)
