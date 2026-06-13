import os
import cv2
import zipfile
import shutil
import numpy as np
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import get_count, set_count
from nsfw import nsfw_scene_score
from down import download_video_from_url

# ---------------- PATHS ----------------
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"

MAX_LIMIT = 200
URL_REGEX = r"^https?://"

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- DETECTORS ----------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

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
def scene_change_score(prev_frame, frame):
    if prev_frame is None:
        return 0
    h1 = cv2.calcHist([prev_frame], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([frame], [0], None, [64], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return int(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) * 150)

def motion_score(prev_gray, gray):
    if prev_gray is None:
        return 0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return int(np.mean(mag) * 30)

def body_score(frame):
    boxes, _ = hog.detectMultiScale(
        frame, winStride=(8, 8),
        padding=(16, 16), scale=1.05
    )
    score = 0
    h, w, _ = frame.shape
    for (x, y, bw, bh) in boxes:
        ratio = (bw * bh) / (w * h)
        score += 150 if ratio > 0.15 else 80
    return score, boxes

async def process_video(video_path, user_id, target, status_msg):
    """Process video and return ZIP path or None"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, total_frames // (target * 3))

    frame_index = 0
    prev_gray = None
    prev_frame = None
    scored_frames = []

    await status_msg.edit("🎞 Analysing video…")

    # PASS 1 (SCORING)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % step != 0:
            continue

        if frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        body_val, people_boxes = body_score(frame)

        nsfw = nsfw_scene_score(
            frame,
            faces,
            people_boxes,
            prev_frame,
            prev_gray
        )

        total = (
            len(faces) * 120 +
            body_val * 1.3 +
            motion_score(prev_gray, gray) * 1.2 +
            scene_change_score(prev_frame, frame) * 1.4 +
            nsfw * 1.6
        )

        scored_frames.append((total, frame_index))
        prev_gray = gray
        prev_frame = frame

    cap.release()

    if not scored_frames:
        return None

    scored_frames.sort(reverse=True)
    selected_frames = sorted([f for _, f in scored_frames[:target]])
    selected_set = set(selected_frames)

    # PASS 2 (EXTRACTION)
    cap = cv2.VideoCapture(video_path)
    current = 0
    saved = 0

    await status_msg.edit("📸 Extracting frames…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current += 1
        if current not in selected_set:
            continue

        if frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)))

        cv2.imwrite(
            f"{FRAME_DIR}/{user_id}_{saved}.jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        saved += 1
        if saved >= target:
            break

    cap.release()

    if saved == 0:
        return None

    # CREATE ZIP
    zip_path = f"{OUTPUT_DIR}/{user_id}_highlights.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for i in range(saved):
            z.write(
                f"{FRAME_DIR}/{user_id}_{i}.jpg",
                arcname=f"{i+1}.jpg"
            )

    # Cleanup frames
    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)

    return zip_path

# ---------------- COMMANDS ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 **Video Highlight Bot**\n\n"
        "Send me a video or use /url to get the best highlights!\n\n"
        "✨ **Features:**\n"
        "• Scene detection\n"
        "• Motion analysis\n"
        "• Body detection\n"
        "• NSFW filtering\n\n"
        "⚙ **Commands:**\n"
        "/settings <1-200> - Set number of images\n"
        "/url <video_url> - Process video from URL\n\n"
        "📦 **Output:** ZIP file with highlights"
    )

@bot.on_message(filters.command("settings"))
async def settings(_, msg):
    if len(msg.command) < 2:
        return await msg.reply("**Usage:** `/settings <1-200>`")
    
    try:
        count = int(msg.command[1])
    except ValueError:
        return await msg.reply("❌ Please enter a valid number")
    
    if count < 1 or count > MAX_LIMIT:
        return await msg.reply(f"❌ Please enter a number between 1 and {MAX_LIMIT}")
    
    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Image count set to **{count}**")

@bot.on_message(filters.command("url"))
async def url_handler(_, msg):
    user_id = msg.from_user.id
    target = min(get_count(user_id), MAX_LIMIT)

    if len(msg.command) < 2:
        return await msg.reply(
            "**Usage:** `/url <video_url>`\n\n"
            "**Example:** `/url https://example.com/video.mp4`"
        )

    url = msg.command[1].strip()

    if not url.startswith(("http://", "https://")):
        return await msg.reply("❌ Invalid URL. Please include http:// or https://")

    status = await msg.reply("⬇️ **Downloading video from URL...**")

    try:
        video_path = download_video_from_url(url)
        
        if not os.path.exists(video_path):
            return await status.edit("❌ Video file not found after download")

    except Exception as e:
        return await status.edit(f"❌ **Download failed**\n```{str(e)}```")

    await status.edit("🎞 **Processing video...**")
    
    zip_path = await process_video(video_path, user_id, target, status)
    
    if not zip_path:
        return await status.edit("❌ **Processing failed**\nNo frames could be extracted")

    await msg.reply_document(
        zip_path,
        caption=f"✅ **Highlights extracted from URL**\n\n"
                f"📸 **Images:** {target}\n"
                f"🎬 **Source:** {url[:50]}..."
    )

    await status.edit("✅ **Done!** Check the ZIP file above.")
    
    # Cleanup downloaded video
    try:
        os.remove(video_path)
    except:
        pass

@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    target = min(get_count(user_id), MAX_LIMIT)

    status = await msg.reply("⬇️ **Downloading video...**")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    await status.edit("🎞 **Processing video...**")
    
    zip_path = await process_video(video_path, user_id, target, status)
    
    if not zip_path:
        return await status.edit("❌ **Processing failed**\nNo frames could be extracted")

    await msg.reply_document(
        zip_path,
        caption=f"✅ **Highlights extracted**\n\n"
                f"📸 **Images:** {target}\n"
                f"📹 **Source:** Telegram video"
    )

    await status.edit("✅ **Done!** Check the ZIP file above.")
    
    # Cleanup downloaded video
    try:
        os.remove(video_path)
    except:
        pass

# ---------------- ERROR HANDLER ----------------
@bot.on_message(filters.command("help"))
async def help_command(_, msg):
    await msg.reply(
        "**📖 Help Guide**\n\n"
        "**How to use:**\n"
        "1️⃣ Send me any video file\n"
        "2️⃣ Or use `/url <video_link>`\n"
        "3️⃣ I'll analyze and extract highlights\n"
        "4️⃣ Get ZIP file with best frames\n\n"
        "**Settings:**\n"
        "• `/settings 50` - Extract 50 images\n"
        "• Maximum: 200 images\n\n"
        "**Features:**\n"
        "• Smart scene detection\n"
        "• Motion analysis\n"
        "• Face and body detection\n"
        "• NSFW content filtering\n"
        "• High-quality JPEG output\n\n"
        "**Support:**\n"
        "• Send /start to begin\n"
        "• Send /help for this menu"
    )
