import os
import cv2
import zipfile
import shutil
import numpy as np
import asyncio
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import get_count, set_count
from nsfw import nsfw_scene_score
from down import download_with_status, get_current_limits
from upscale import (
    process_zip,
    cleanup_user
)


# ---------------- PATHS ----------------
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"

MAX_LIMIT = 200

# Cancel tracking
cancel_users = {}

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
        if ratio > 0.20:
            score += 300
        elif ratio > 0.10:
            score += 150
        else:
            score += 50
            
    return score, boxes

async def process_video(video_path, user_id, target, status_msg):
    """Process video and return ZIP path or None"""
    # Check if video exists
    if not os.path.exists(video_path):
        await status_msg.edit("❌ Video file not found")
        return None
    
    # Check file size
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        await status_msg.edit("❌ Video file is empty")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    # Validate video opened successfully
    if not cap.isOpened():
        await status_msg.edit("❌ Failed to open video file")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        cap.release()
        await status_msg.edit("❌ Video has no frames")
        return None

    if fps <= 0:
        cap.release()
        await status_msg.edit("❌ Invalid FPS")
        return None

    duration = total_frames / fps
    
    # Check duration limit (2 hours)
    if duration > 7200:
        await status_msg.edit(
            "❌ Video too long (max 120 minutes)"
        )
        cap.release()
        return None

    # Calculate step size
    step = max(15, total_frames // max(target, 1))
    
    # If step is too large, reduce it
    if step > 100:
        step = 50  # Ensure we get enough frames

    frame_index = 0
    prev_gray = None
    prev_frame = None
    scored_frames = []

    # Create cancel button markup
    cancel_markup = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "❌ Cancel",
                    callback_data="cancel"
                )
            ]
        ]
    )

    await status_msg.edit(
        f"🎞 Analysing video…\n"
        f"📊 Total frames: {total_frames}\n"
        f"📸 Target images: {target}",
        reply_markup=cancel_markup
    )

    # PASS 1 (SCORING)
    while cap.isOpened():
        # Check for cancel
        if cancel_users.get(user_id):
            cap.release()
            await status_msg.edit("🛑 Cancelled by user.")
            cancel_users[user_id] = False
            return None
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        
        # Prevent bot from dying on long videos
        if frame_index % 100 == 0:
            await asyncio.sleep(0)
        
        # Progress bar - Analysis phase
        if frame_index % 500 == 0:
            progress = int((frame_index / total_frames) * 100)
            try:
                await status_msg.edit(
                    f"🎞 Analysing video…\n"
                    f"📊 Progress: {progress}%\n"
                    f"📸 Frames processed: {frame_index}/{total_frames}",
                    reply_markup=cancel_markup
                )
            except:
                pass
        
        if frame_index % step != 0:
            continue

        # Resize for performance
        if frame.shape[1] > 720:
            scale = 720 / frame.shape[1]
            frame = cv2.resize(frame, (720, int(frame.shape[0] * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        body_val, people_boxes = body_score(frame)

        try:
            nsfw = nsfw_scene_score(
                frame,
                faces,
                people_boxes,
                prev_frame,
                prev_gray
            )
        except:
            nsfw = 0

        # Improved scoring
        total = (
            len(faces) * 60 +
            body_val * 2.0 +
            motion_score(prev_gray, gray) +
            scene_change_score(prev_frame, frame) +
            nsfw * 3.5
        )

        scored_frames.append((total, frame_index))
        prev_gray = gray
        prev_frame = frame

    cap.release()

    if not scored_frames:
        await status_msg.edit("❌ No frames analyzed")
        return None

    # Sort and select best frames
    scored_frames.sort(reverse=True)
    selected_frames = sorted([f for _, f in scored_frames[:target]])
    
    if not selected_frames:
        await status_msg.edit("❌ No frames selected")
        return None
    
    selected_set = set(selected_frames)

    # PASS 2 (EXTRACTION)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await status_msg.edit("❌ Failed to reopen video for extraction")
        return None
    
    current = 0
    saved = 0

    await status_msg.edit(
        "📸 Extracting frames…",
        reply_markup=cancel_markup
    )

    while cap.isOpened():
        # Check for cancel
        if cancel_users.get(user_id):
            cap.release()
            await status_msg.edit("🛑 Cancelled by user.")
            cancel_users[user_id] = False
            return None
        
        ret, frame = cap.read()
        if not ret:
            break

        current += 1
        if current not in selected_set:
            continue

        # Resize for output
        if frame.shape[1] > 720:
            scale = 720 / frame.shape[1]
            frame = cv2.resize(frame, (720, int(frame.shape[0] * scale)))

        # Save frame
        frame_path = f"{FRAME_DIR}/{user_id}_{saved}.jpg"
        success = cv2.imwrite(
            frame_path,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        
        if not success:
            await status_msg.edit(f"❌ Failed to save frame {saved}")
            cap.release()
            return None

        saved += 1
        
        # Update progress
        progress = int((saved / target) * 100)
        await status_msg.edit(
            f"📸 Extracting frames…\n"
            f"📊 Progress: {progress}%\n"
            f"✅ Saved: {saved}/{target}",
            reply_markup=cancel_markup
        )
        
        if saved >= target:
            break

    cap.release()

    if saved == 0:
        await status_msg.edit("❌ No frames extracted")
        return None

    # CREATE ZIP
    zip_path = f"{OUTPUT_DIR}/{user_id}_highlights.zip"
    try:
        with zipfile.ZipFile(zip_path, "w") as z:
            for i in range(saved):
                frame_path = f"{FRAME_DIR}/{user_id}_{i}.jpg"
                if os.path.exists(frame_path):
                    z.write(frame_path, arcname=f"{i+1}.jpg")
                else:
                    await status_msg.edit(f"❌ Missing frame: {i}")
                    return None
    except Exception as e:
        await status_msg.edit(f"❌ ZIP creation failed: {str(e)}")
        return None

    # Cleanup frames
    try:
        shutil.rmtree(FRAME_DIR)
        os.makedirs(FRAME_DIR, exist_ok=True)
    except:
        pass

    return zip_path

# ---------------- CALLBACK HANDLER ----------------
@bot.on_callback_query()
async def callback(_, query):
    if query.data == "cancel":
        user_id = query.from_user.id
        cancel_users[user_id] = True
        await query.answer("🛑 Cancelled processing!")
        await query.message.edit("🛑 Processing cancelled.")

# ---------------- COMMANDS ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    limits = get_current_limits()
    
    await msg.reply(
        f"🎬 **Video Highlight Bot**\n\n"
        f"Send me a video or use /url to get the best highlights!\n\n"
        f"✨ **Features:**\n"
        f"• Scene detection\n"
        f"• Motion analysis\n"
        f"• Body detection (Enhanced)\n"
        f"• NSFW filtering (High Priority)\n"
        f"• Progress bars\n"
        f"• Cancel button\n\n"
        f"📦 **Limits:**\n"
        f"• Max Size: {limits['max_size_gb']} GB\n"
        f"• Max Duration: {limits['max_duration_minutes']} minutes\n\n"
        f"⚙ **Commands:**\n"
        f"/settings <1-200> - Set number of images\n"
        f"/url <video_url> - Process video from URL\n"
        f"/limits - Show current limits\n\n"
        f"📦 **Output:** ZIP file with highlights"
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

@bot.on_message(filters.command("limits"))
async def limits_command(_, msg):
    limits = get_current_limits()
    await msg.reply(
        f"📊 **Current Limits:**\n\n"
        f"📦 **Max File Size:** {limits['max_size_gb']} GB ({limits['max_size_mb']} MB)\n"
        f"⏱️ **Max Duration:** {limits['max_duration_minutes']} minutes ({limits['max_duration_hours']} hours)\n\n"
        f"⚠️ Videos exceeding these limits will be rejected."
    )

@bot.on_message(filters.command("url"))
async def url_handler(_, msg):
    user_id = msg.from_user.id
    target = min(get_count(user_id), MAX_LIMIT)
    
    cancel_users[user_id] = False

    if len(msg.command) < 2:
        return await msg.reply(
            "**Usage:** `/url <video_url>`\n\n"
            "**Example:** `/url https://example.com/video.mp4`"
        )

    url = msg.command[1].strip()

    if not url.startswith(("http://", "https://")):
        return await msg.reply("❌ Invalid URL. Please include http:// or https://")

    status = await msg.reply("🔍 **Checking video...**")

    try:
        video_path = await download_with_status(url, status)
        
        if not video_path:
            return

    except Exception as e:
        return await status.edit(f"❌ **Download failed**\n```{str(e)}```")

    # Process the downloaded video
    await status.edit("🎞 **Processing video...**")
    zip_path = await process_video(video_path, user_id, target, status)
    
    if not zip_path:
        await status.edit("❌ **Processing failed**")
        return

    await msg.reply_document(
        zip_path,
        caption=f"✅ **Highlights extracted from URL**\n\n"
                f"📸 **Images:** {target}\n"
                f"🎬 **Source:** {url[:50]}..."
    )
    
    # Clean ZIP after sending
    try:
        os.remove(zip_path)
    except:
        pass

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
    
    cancel_users[user_id] = False

    status = await msg.reply("⬇️ **Downloading video...**")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    # Process the downloaded video
    await status.edit("🎞 **Processing video...**")
    zip_path = await process_video(video_path, user_id, target, status)
    
    if not zip_path:
        await status.edit("❌ **Processing failed**")
        return

    await msg.reply_document(
        zip_path,
        caption=f"✅ **Highlights extracted**\n\n"
                f"📸 **Images:** {target}\n"
                f"📹 **Source:** Telegram video"
    )
    
    # Clean ZIP after sending
    try:
        os.remove(zip_path)
    except:
        pass

    await status.edit("✅ **Done!** Check the ZIP file above.")
    
    # Cleanup downloaded video
    try:
        os.remove(video_path)
    except:
        pass


# ---------------- HELP COMMAND ----------------
@bot.on_message(filters.command("help"))
async def help_command(_, msg):
    limits = get_current_limits()
    
    await msg.reply(
        f"**📖 Help Guide**\n\n"
        f"**How to use:**\n"
        f"1️⃣ Send me any video file\n"
        f"2️⃣ Or use `/url <video_link>`\n"
        f"3️⃣ I'll analyze and extract highlights\n"
        f"4️⃣ Get ZIP file with best frames\n\n"
        f"**Settings:**\n"
        f"• `/settings 50` - Extract 50 images\n"
        f"• Maximum: {MAX_LIMIT} images\n"
        f"• `/limits` - Show current limits\n\n"
        f"**Limits:**\n"
        f"• Max Size: {limits['max_size_gb']} GB\n"
        f"• Max Duration: {limits['max_duration_minutes']} minutes\n\n"
        f"**Features:**\n"
        f"• Smart scene detection\n"
        f"• Motion analysis\n"
        f"• Enhanced face and body detection\n"
        f"• NSFW content filtering (High Priority)\n"
        f"• High-quality JPEG output\n"
        f"• 720p processing for better performance\n"
        f"• Support for long videos (up to {limits['max_duration_minutes']} min)\n"
        f"• Real-time progress bars\n"
        f"• Cancel button\n\n"
        f"**Support:**\n"
        f"• Send /start to begin\n"
        f"• Send /help for this menu"
    )
