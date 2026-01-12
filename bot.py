import os
import cv2
import time
import json
import sqlite3
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
DB_PATH = "tasks.db"

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- DATABASE ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS tasks (
    user_id TEXT PRIMARY KEY,
    video_path TEXT,
    last_frame INTEGER,
    saved_images TEXT,
    max_images INTEGER,
    status TEXT
)
""")
conn.commit()

def save_task(user_id, video_path, last_frame, images, max_images, status):
    cur.execute("""
    INSERT OR REPLACE INTO tasks
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        video_path,
        last_frame,
        json.dumps(images),
        max_images,
        status
    ))
    conn.commit()

def get_task(user_id):
    cur.execute("SELECT * FROM tasks WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "video_path": row[1],
        "last_frame": row[2],
        "saved_images": json.loads(row[3]),
        "max_images": row[4],
        "status": row[5]
    }

def get_pending_tasks():
    cur.execute("SELECT * FROM tasks WHERE status IN ('processing','paused')")
    rows = cur.fetchall()
    tasks = []
    for r in rows:
        tasks.append({
            "user_id": r[0],
            "video_path": r[1],
            "last_frame": r[2],
            "saved_images": json.loads(r[3]),
            "max_images": r[4],
            "status": r[5]
        })
    return tasks

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- BOT ----------------
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
        "✨ Face-priority highlights\n"
        "🧍 Full frame (face + body)\n"
        "📄 PDF output\n\n"
        "⚙ /settings <1-200>"
    )

# ---------------- SETTINGS ----------------
@bot.on_message(filters.command("settings"))
async def settings_cmd(_, msg):
    if len(msg.command) < 2:
        await msg.reply("Usage: /settings <1-200>")
        return

    count = int(msg.command[1])
    if count < 1 or count > 200:
        await msg.reply("❌ Render limit: 1–200 only")
        return

    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Highlight count set to {count}")

# ---------------- VIDEO ----------------
@bot.on_message(filters.video)
async def handle_video(_, msg):
    user_id = str(msg.from_user.id)
    max_images = get_count(msg.from_user.id)

    await msg.reply("⬇️ Downloading video…")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    save_task(
        user_id,
        video_path,
        0,
        [],
        max_images,
        "processing"
    )

    await process_video(user_id, msg.chat.id)

# ---------------- PROCESS VIDEO ----------------
async def process_video(user_id, chat_id):
    task = get_task(user_id)
    if not task:
        return

    cap = cv2.VideoCapture(task["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, task["last_frame"])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // (task["max_images"] * 3))

    saved_images = task["saved_images"]
    saved_count = len(saved_images)
    prev_gray = None
    frame_no = task["last_frame"]

    await bot.send_message(
        chat_id,
        f"🎞 Processing resumed\n📸 {saved_count}/{task['max_images']}"
    )

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        # Render-safe resize
        h, w, _ = frame.shape
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face score
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        score = 0
        for (x, y, fw, fh) in faces:
            ratio = (fw * fh) / (gray.shape[0] * gray.shape[1])
            score += 150 if 0.02 < ratio < 0.2 else 80
            if y < gray.shape[0] * 0.4:
                score += 50

        # Motion
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score += int(np.sum(diff) / 1_000_000)
        prev_gray = gray

        # Sharpness
        if cv2.Laplacian(gray, cv2.CV_64F).var() > 120:
            score += 20

        if score > 120:
            img_path = f"{FRAME_DIR}/{user_id}_{saved_count}.jpg"
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_images.append(img_path)
            saved_count += 1

            save_task(
                user_id,
                task["video_path"],
                frame_no,
                saved_images,
                task["max_images"],
                "processing"
            )

        # Server protection (auto-pause)
        if saved_count >= 180 or time.time() - start_time > 20:
            save_task(
                user_id,
                task["video_path"],
                frame_no,
                saved_images,
                task["max_images"],
                "paused"
            )
            cap.release()
            return

        if saved_count >= task["max_images"]:
            break

    cap.release()

    # Create PDF
    first = Image.open(saved_images[0]).convert("RGB")
    rest = [Image.open(p).convert("RGB") for p in saved_images[1:]]

    pdf_path = f"{OUTPUT_DIR}/{user_id}_highlights.pdf"
    first.save(pdf_path, save_all=True, append_images=rest, resolution=150)

    await bot.send_document(
        chat_id,
        pdf_path,
        caption=f"✅ {saved_count} highlights\n📄 PDF ready"
    )

    save_task(
        user_id,
        task["video_path"],
        frame_no,
        [],
        task["max_images"],
        "done"
    )

    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)

# ---------------- AUTO RESUME ----------------
async def resume_tasks():
    for task in get_pending_tasks():
        await bot.send_message(
            int(task["user_id"]),
            "⚙️ Server restarted. Resuming your project automatically…"
        )
        await process_video(task["user_id"], int(task["user_id"]))
