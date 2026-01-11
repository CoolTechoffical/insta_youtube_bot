import os
import cv2
import uuid
import time
import shutil
import numpy as np
from PIL import Image
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN, MAX_IMAGES, MAX_SAFE_SECONDS
from user_settings import set_count, get_count
from mongo_db import create_job, update_job, get_active_jobs

DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

bot = Client(
    "highlight_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ---------------- START ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 Send a video\n"
        "⚙ /settings <count>\n"
        "🔁 Auto-recovery enabled"
    )

# ---------------- SETTINGS ----------------
@bot.on_message(filters.command("settings"))
async def settings(_, msg):
    if len(msg.command) < 2:
        return await msg.reply("Usage: /settings <1-200>")

    count = int(msg.command[1])
    if count < 1 or count > MAX_IMAGES:
        return await msg.reply("❌ Max 200 images (Render limit)")

    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Set to {count}")

# ---------------- VIDEO ----------------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    max_images = get_count(user_id)

    status = await msg.reply("⬇️ Downloading video...")
    video_path = await msg.download(file_name=DOWNLOAD_DIR + "/")

    job_id = str(uuid.uuid4())

    create_job({
        "job_id": job_id,
        "user_id": user_id,
        "video_path": video_path,
        "last_frame": 0,
        "saved": 0,
        "target": max_images,
        "status": "processing",
        "stage": "extract",
        "updated": time.time()
    })

    await status.edit("🎞 Processing (auto-recovery enabled)")
    await process_job(job_id, status)

# ---------------- CORE PROCESS ----------------
async def process_job(job_id, status_msg):
    from mongo_db import get_job

    job = get_job(job_id)
    start_time = time.time()

    cap = cv2.VideoCapture(job["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, job["last_frame"])

    prev_gray = None

    while cap.isOpened():
        if time.time() - start_time > MAX_SAFE_SECONDS:
            update_job(job_id, {"status": "waiting"})
            await status_msg.edit("⏳ Server busy, auto-resume soon")
            return

        ret, frame = cap.read()
        if not ret:
            break

        job["last_frame"] += 1

        if frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(
                frame,
                (1280, int(frame.shape[0] * scale))
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        motion = 0
        if prev_gray is not None:
            motion = int(np.sum(cv2.absdiff(prev_gray, gray)) / 1_000_000)
        prev_gray = gray

        score = motion + (150 if len(faces) else 0)

        if score > 120:
            img_path = f"{FRAME_DIR}/{job_id}_{job['saved']}.jpg"
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            job["saved"] += 1

        update_job(job_id, {
            "last_frame": job["last_frame"],
            "saved": job["saved"],
            "updated": time.time()
        })

        if job["saved"] >= job["target"]:
            break

    cap.release()

    await create_pdf_and_send(job_id, status_msg)

# ---------------- PDF ----------------
async def create_pdf_and_send(job_id, status_msg):
    from mongo_db import get_job

    job = get_job(job_id)
    images = sorted(
        [f for f in os.listdir(FRAME_DIR) if f.startswith(job_id)]
    )

    if not images:
        return await status_msg.edit("❌ No highlights found")

    first = Image.open(os.path.join(FRAME_DIR, images[0])).convert("RGB")
    rest = [
        Image.open(os.path.join(FRAME_DIR, i)).convert("RGB")
        for i in images[1:]
    ]

    pdf_path = f"{OUTPUT_DIR}/{job_id}.pdf"
    first.save(pdf_path, save_all=True, append_images=rest, resolution=150)

    await bot.send_document(
        job["user_id"],
        pdf_path,
        caption=f"✅ {len(images)} highlights"
    )

    update_job(job_id, {"status": "completed"})
    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)

# ---------------- AUTO RESUME ON START ----------------
async def resume_pending_jobs():
    jobs = get_active_jobs()
    for job in jobs:
        update_job(job["job_id"], {"status": "processing"})
        await process_job(job["job_id"], None)
