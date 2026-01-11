import os
import cv2
import uuid
import time
import shutil
import asyncio
import numpy as np
from PIL import Image

from aiohttp import web
from pyrogram import Client, filters

from config import API_ID, API_HASH, BOT_TOKEN, MAX_IMAGES, MAX_SAFE_SECONDS
from user_settings import set_count, get_count
from mongo_db import (
    create_job,
    update_job,
    get_job,
    get_active_jobs
)

# ---------------- DIRECTORIES ----------------
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- FACE CASCADE ----------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- BOT ----------------
bot = Client(
    "highlight_bot",
    api_id=int(API_ID),
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

    try:
        count = int(msg.command[1])
    except ValueError:
        return await msg.reply("❌ Enter a number")

    if count < 1 or count > MAX_IMAGES:
        return await msg.reply("❌ Max 200 images (server limit)")

    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Set to {count}")

# ---------------- VIDEO ----------------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    max_images = get_count(user_id)

    status = await msg.reply("⬇️ Downloading video...")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    job_id = str(uuid.uuid4())

    create_job({
        "job_id": job_id,
        "user_id": user_id,
        "video_path": video_path,
        "last_frame": 0,
        "saved": 0,
        "target": max_images,
        "status": "processing",
        "updated": time.time()
    })

    await status.edit("🎞 Processing (auto-recovery ON)")
    await process_job(job_id, status)

# ---------------- CORE PROCESS ----------------
async def process_job(job_id, status_msg=None):
    job = get_job(job_id)
    if not job:
        return

    start_time = time.time()
    cap = cv2.VideoCapture(job["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, job["last_frame"])

    prev_gray = None

    while cap.isOpened():
        if time.time() - start_time > MAX_SAFE_SECONDS:
            update_job(job_id, {"status": "waiting"})
            if status_msg:
                await status_msg.edit("⏳ Server busy, auto-resume soon")
            cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            break

        job["last_frame"] += 1

        if frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(
                frame,
                (1280, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA
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

        if status_msg:
            await status_msg.edit(
                f"📸 {job['saved']}/{job['target']} images"
            )

        if job["saved"] >= job["target"]:
            break

    cap.release()
    await create_pdf_and_send(job_id, status_msg)

# ---------------- PDF ----------------
async def create_pdf_and_send(job_id, status_msg=None):
    job = get_job(job_id)
    if not job:
        return

    images = sorted(f for f in os.listdir(FRAME_DIR) if f.startswith(job_id))

    if not images:
        if status_msg:
            await status_msg.edit("❌ No highlights found")
        return

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

# ---------------- AUTO RESUME ----------------
async def resume_pending_jobs():
    loop = asyncio.get_running_loop()
    jobs = await loop.run_in_executor(None, get_active_jobs)

    for job in jobs:
        update_job(job["job_id"], {"status": "processing"})
        await process_job(job["job_id"], None)

# ---------------- WEB SERVER ----------------
async def health(request):
    return web.Response(text="Bot running")

async def main():
    # resume jobs
    await resume_pending_jobs()

    # start bot
    await bot.start()

    # start web server (Render requirement)
    app = web.Application()
    app.router.add_get("/", health)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.environ.get("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    print(f"🌐 Web service running on port {port}")
    print("🤖 Bot started")

    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
