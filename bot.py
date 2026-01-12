import os
import cv2
import shutil
import zipfile
import numpy as np
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import set_count, get_count

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
        "🎬 Send a video\n\n"
        "✨ Face-priority highlights\n"
        "🧍 Full frame (face + body)\n"
        "📦 Output: ZIP\n\n"
        "⚙ /settings <count> (max 200)"
    )

# ---------------- SETTINGS ----------------
@bot.on_message(filters.command("settings"))
async def settings(_, msg):
    if len(msg.command) < 2:
        await msg.reply("Usage: /settings <1-200>")
        return

    count = int(msg.command[1])

    if count < 1 or count > 200:
        await msg.reply("❌ Render limit: 1–200 images only")
        return

    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Highlight count set to {count}")

# ---------------- VIDEO ----------------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    target_count = get_count(user_id)

    status = await msg.reply("⬇️ Downloading video...")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, total_frames // (target_count * 4))
    frame_no = 0
    prev_gray = None
    scored_frames = []

    await status.edit("🎞 Scanning video…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        # ---- Resize ----
        h, w, _ = frame.shape
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(
                frame,
                (1280, int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Face score ----
        faces = face_cascade.detectMultiScale(
            gray, 1.2, 5, minSize=(60, 60)
        )

        face_score = 0
        for (x, y, fw, fh) in faces:
            ratio = (fw * fh) / (gray.shape[0] * gray.shape[1])
            face_score += 120 if 0.02 < ratio < 0.2 else 60
            if y < gray.shape[0] * 0.4:
                face_score += 40

        # ---- Motion ----
        motion = 0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = int(np.sum(diff) / 1_000_000)
        prev_gray = gray

        # ---- Sharpness ----
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = 20 if sharp > 120 else 0

        total_score = face_score + motion + sharp_score

        scored_frames.append((total_score, frame.copy()))

    cap.release()

    if not scored_frames:
        await status.edit("❌ No frames detected")
        return

    # -------- SELECT EXACT COUNT --------
    scored_frames.sort(key=lambda x: x[0], reverse=True)
    selected = scored_frames[:target_count]

    await status.edit("📸 Saving images…")

    image_paths = []
    for i, (_, frame) in enumerate(selected):
        path = f"{FRAME_DIR}/{user_id}_{i}.jpg"
        cv2.imwrite(
            path,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        image_paths.append(path)

    # -------- CREATE ZIP --------
    zip_path = f"{OUTPUT_DIR}/{user_id}_highlights.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for img in image_paths:
            zipf.write(img, arcname=os.path.basename(img))

    await msg.reply_document(
        zip_path,
        caption=(
            f"✅ Highlights extracted\n"
            f"📸 Images: {len(image_paths)}\n"
            "🧍 Face + body\n"
            "📦 ZIP format"
        )
    )

    await status.edit("✅ Done")

    # -------- CLEANUP --------
    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
