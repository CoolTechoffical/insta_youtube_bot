import os
import cv2
import shutil
import numpy as np
from PIL import Image
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
        "📄 Output: PDF\n\n"
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
    max_images = get_count(user_id)

    status = await msg.reply("⬇️ Downloading video...")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, total_frames // (max_images * 3))

    saved = 0
    frame_no = 0
    prev_gray = None
    images = []

    await status.edit("🎞 Processing video…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        # -------- Resize (Render Safe) --------
        h, w, _ = frame.shape
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(
                frame,
                (1280, int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------- Face Score --------
        faces = face_cascade.detectMultiScale(
            gray, 1.2, 5, minSize=(60, 60)
        )

        face_score = 0
        for (x, y, fw, fh) in faces:
            ratio = (fw * fh) / (gray.shape[0] * gray.shape[1])
            if 0.02 < ratio < 0.2:
                face_score += 150
            else:
                face_score += 80
            if y < gray.shape[0] * 0.4:
                face_score += 50

        # -------- Motion --------
        motion = 0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = int(np.sum(diff) / 1_000_000)
        prev_gray = gray

        # -------- Sharpness --------
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = 20 if sharp > 120 else 0

        total_score = face_score + motion + sharp_score

        if total_score > 120:
            img_path = f"{FRAME_DIR}/{user_id}_{saved}.jpg"
            cv2.imwrite(
                img_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            images.append(img_path)
            saved += 1

            progress = int((frame_no / total_frames) * 100)
            await status.edit(
                f"📸 Saved: {saved}/{max_images}\n⏳ {progress}%"
            )

        if saved >= max_images:
            break

    cap.release()

    if not images:
        await status.edit("❌ No highlights detected")
        return

    # -------- PDF (Low Memory) --------
    await status.edit("📄 Creating PDF…")

    first = Image.open(images[0]).convert("RGB")
    rest = [Image.open(i).convert("RGB") for i in images[1:]]

    pdf_path = f"{OUTPUT_DIR}/{user_id}_highlights.pdf"
    first.save(
        pdf_path,
        save_all=True,
        append_images=rest,
        resolution=150
    )

    await msg.reply_document(
        pdf_path,
        caption=f"✅ {len(images)} highlights\n🧍 Face + body\n📄 PDF"
    )

    await status.edit("✅ Done")

    # -------- Cleanup --------
    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
