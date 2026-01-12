import os
import cv2
import json
import shutil
import time
import asyncio
import numpy as np
from PIL import Image
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import set_count, get_count

# ---------------- Directories ----------------
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"
TASKS_FILE = "tasks.json"

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- Load or Create Task DB ----------------
if os.path.exists(TASKS_FILE):
    with open(TASKS_FILE, "r") as f:
        tasks = json.load(f)
else:
    tasks = {}

def save_tasks():
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f)

# ---------------- Face Detector ----------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- Bot ----------------
bot = Client(
    "highlight_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ---------------- Start Command ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 Send a video\n"
        "✨ Face-priority highlights\n"
        "🧍 Full frame (face + body)\n"
        "📄 Output: PDF\n\n"
        "⚙ /settings <count> (max 200)"
    )

# ---------------- Settings Command ----------------
@bot.on_message(filters.command("settings"))
async def settings_cmd(_, msg):
    if len(msg.command) < 2:
        await msg.reply("Usage: /settings <1-200>")
        return
    count = int(msg.command[1])
    if count < 1 or count > 200:
        await msg.reply("❌ Render limit: 1–200 images only")
        return
    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Highlight count set to {count}")

# ---------------- Video Handler ----------------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = str(msg.from_user.id)
    max_images = get_count(msg.from_user.id)

    await msg.reply("⬇️ Downloading video...")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    # Create task
    tasks[user_id] = {
        "video_path": video_path,
        "last_frame": 0,
        "saved_images": [],
        "max_images": max_images,
        "status": "processing"
    }
    save_tasks()

    await process_video(user_id, msg)

# ---------------- Process Video Function ----------------
async def process_video(user_id, msg):
    task = tasks[user_id]
    video_path = task["video_path"]
    last_frame = task.get("last_frame", 0)
    saved_images = task.get("saved_images", [])
    max_images = task.get("max_images", 100)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // (max_images * 3))
    frame_no = 0
    prev_gray = None
    saved_count = len(saved_images)

    status_msg = await msg.reply(f"🎞 Processing video… Saved: {saved_count}/{max_images}")

    # Resume from last frame
    if last_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        current_frame = last_frame + frame_no

        if current_frame % step != 0:
            continue

        # Render-safe resize
        h, w, _ = frame.shape
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h*scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face score
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60,60))
        face_score = 0
        for (x,y,fw,fh) in faces:
            ratio = (fw*fh) / (gray.shape[0]*gray.shape[1])
            face_score += 150 if 0.02<ratio<0.2 else 80
            if y < gray.shape[0]*0.4: face_score +=50

        # Motion score
        motion = 0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = int(np.sum(diff)/1_000_000)
        prev_gray = gray

        # Sharpness
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = 20 if sharp>120 else 0

        total_score = face_score + motion + sharp_score

        if total_score > 120:
            img_path = f"{FRAME_DIR}/{user_id}_{saved_count}.jpg"
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_images.append(img_path)
            saved_count += 1
            tasks[user_id]["saved_images"] = saved_images
            save_tasks()

            progress = int((current_frame/total_frames)*100)
            await status_msg.edit(f"📸 Saved: {saved_count}/{max_images}\n⏳ {progress}%")

        tasks[user_id]["last_frame"] = current_frame
        save_tasks()

        if saved_count >= max_images:
            break

        # Server protection: manual safe check
        if saved_count >= 180:
            await status_msg.edit("⚠️ Server limit approaching, pausing automatically…")
            tasks[user_id]["status"] = "paused"
            save_tasks()
            cap.release()
            return

    cap.release()

    if not saved_images:
        await status_msg.edit("❌ No highlights detected")
        tasks[user_id]["status"] = "done"
        save_tasks()
        return

    # PDF creation
    await status_msg.edit("📄 Creating PDF…")
    first = Image.open(saved_images[0]).convert("RGB")
    rest = [Image.open(p).convert("RGB") for p in saved_images[1:]]
    pdf_path = f"{OUTPUT_DIR}/{user_id}_highlights.pdf"
    first.save(pdf_path, save_all=True, append_images=rest, resolution=150)

    await msg.reply_document(pdf_path, caption=f"✅ {len(saved_images)} highlights | Face + Body | PDF")

    tasks[user_id]["status"] = "done"
    save_tasks()

    # Cleanup frames
    for f in saved_images:
        if os.path.exists(f): os.remove(f)
    tasks[user_id]["saved_images"] = []
    save_tasks()

# ---------------- Resume Pending Tasks on Startup ----------------
async def resume_pending_tasks():
    for user_id, task in tasks.items():
        if task.get("status") == "processing" or task.get("status")=="paused":
            try:
                chat_id = int(user_id)
                # Send notice to user
                await bot.send_message(chat_id, "⚙️ Resuming your previous video processing…")
                await process_video(user_id, await bot.get_messages(chat_id, 1))
            except Exception as e:
                print("Failed to resume task for user:", user_id, e)
