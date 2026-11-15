import os
import cv2
import zipfile
import shutil
import numpy as np
import asyncio
import signal
import threading
import logging
import time
import requests

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
from image_video import register_image_video_handlers


log = logging.getLogger("bot")

# ---------------- SHUTDOWN FLAG ----------------

SHUTDOWN_FLAG = os.environ.get(
    "SHUTDOWN_FLAG",
    "/tmp/shutdown.flag"
)

_shutdown_event = threading.Event()
_BOT_START_TIME = time.time()


def request_shutdown():
    log.warning("Shutdown requested")
    _shutdown_event.set()

    try:
        with open(SHUTDOWN_FLAG, "w") as f:
            f.write(str(os.getpid()))
    except Exception as error:
        log.warning("Could not write shutdown flag: %s", error)


def is_shutting_down():
    if _shutdown_event.is_set():
        return True

    if not os.path.exists(SHUTDOWN_FLAG):
        return False

    try:
        with open(SHUTDOWN_FLAG) as f:
            writer = f.read().strip()

        if writer and writer != str(os.getpid()):
            return False

    except Exception:
        pass

    return True


# ---------------- STATE ----------------

upscale_waiting = set()
upscale_cancelled = set()
upscale_caption_waiting = set()
upscale_captions = {}

# ---------------- PATHS ----------------

DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"

MAX_LIMIT = 200

cancel_users = {}

for directory in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(directory, exist_ok=True)


# ---------------- DETECTORS ----------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load OpenCV Haar cascade")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)


# ---------------- SINGLE PYROGRAM EVENT LOOP ----------------
#
# IMPORTANT:
# Pyrogram's Client captures an event loop during construction.
# Create and install the loop BEFORE constructing Client.
# Do not use asyncio.run() for this Client.

_BOT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(_BOT_LOOP)


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

    h1 = cv2.calcHist(
        [prev_frame], [0], None, [64], [0, 256]
    )
    h2 = cv2.calcHist(
        [frame], [0], None, [64], [0, 256]
    )

    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)

    try:
        method = cv2.HISTCMP_BHATTACHARYYA
    except AttributeError:
        method = cv2.CMP_BHATTACHARYYA

    return int(cv2.compareHist(h1, h2, method) * 150)


def motion_score(prev_gray, gray):
    if prev_gray is None:
        return 0

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    mag, _ = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    return int(np.mean(mag) * 30)


def body_score(frame):
    boxes, _ = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
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


# ---------------- VIDEO PROCESSING ----------------

async def process_video(video_path, user_id, target, status_msg):
    """Process video and return ZIP path or None."""

    if is_shutting_down():
        await status_msg.edit("Server restarting, please retry.")
        return None

    if not os.path.exists(video_path):
        await status_msg.edit("Video file not found")
        return None

    file_size = os.path.getsize(video_path)

    if file_size == 0:
        await status_msg.edit("Video file is empty")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        await status_msg.edit("Failed to open video file")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames == 0:
        cap.release()
        await status_msg.edit("Video has no frames")
        return None

    if fps <= 0:
        cap.release()
        await status_msg.edit("Invalid FPS")
        return None

    duration = total_frames / fps

    if duration > 7200:
        cap.release()
        await status_msg.edit(
            "Video too long (maximum 120 minutes)"
        )
        return None

    step = max(15, total_frames // max(target, 1))

    if step > 100:
        step = 50

    frame_index = 0
    prev_gray = None
    prev_frame = None
    scored_frames = []

    cancel_markup = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Cancel",
                callback_data="cancel"
            )
        ]
    ])

    await status_msg.edit(
        f"Analysing video...\n"
        f"Total frames: {total_frames}\n"
        f"Target images: {target}",
        reply_markup=cancel_markup
    )

    # PASS 1: SCORING

    while cap.isOpened():
        if is_shutting_down():
            cap.release()
            await status_msg.edit(
                "Server restarting, please retry."
            )
            return None

        if cancel_users.get(user_id):
            cap.release()
            await status_msg.edit("Processing cancelled.")
            cancel_users[user_id] = False
            return None

        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        if frame_index % 100 == 0:
            await asyncio.sleep(0)

        if frame_index % 500 == 0:
            progress = int((frame_index / total_frames) * 100)

            try:
                await status_msg.edit(
                    f"Analysing video...\n"
                    f"Progress: {progress}%\n"
                    f"Frames processed: {frame_index}/{total_frames}",
                    reply_markup=cancel_markup
                )
            except Exception:
                pass

        if frame_index % step != 0:
            continue

        if frame.shape[1] > 720:
            scale = 720 / frame.shape[1]
            frame = cv2.resize(
                frame,
                (720, int(frame.shape[0] * scale))
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            1.2,
            5
        )

        body_val, people_boxes = body_score(frame)

        try:
            nsfw = nsfw_scene_score(
                frame,
                faces,
                people_boxes,
                prev_frame,
                prev_gray
            )
        except Exception:
            nsfw = 0

        total = (
            len(faces) * 60
            + body_val * 2.0
            + motion_score(prev_gray, gray)
            + scene_change_score(prev_frame, frame)
            + nsfw * 3.5
        )

        scored_frames.append((total, frame_index))
        prev_gray = gray
        prev_frame = frame

    cap.release()

    if not scored_frames:
        await status_msg.edit("No frames analysed")
        return None

    scored_frames.sort(reverse=True)

    selected_frames = sorted([
        frame_number
        for _, frame_number in scored_frames[:target]
    ])

    if not selected_frames:
        await status_msg.edit("No frames selected")
        return None

    selected_set = set(selected_frames)

    # PASS 2: EXTRACTION

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        await status_msg.edit(
            "Failed to reopen video for extraction"
        )
        return None

    current = 0
    saved = 0

    user_frame_dir = os.path.join(
        FRAME_DIR,
        str(user_id)
    )

    os.makedirs(user_frame_dir, exist_ok=True)

    # Remove leftover frames from an earlier interrupted job.
    for old_name in os.listdir(user_frame_dir):
        old_path = os.path.join(user_frame_dir, old_name)
        try:
            if os.path.isfile(old_path):
                os.remove(old_path)
        except OSError:
            pass

    await status_msg.edit(
        "Extracting frames...",
        reply_markup=cancel_markup
    )

    try:
        while cap.isOpened():
            if is_shutting_down():
                await status_msg.edit(
                    "Server restarting, please retry."
                )
                return None

            if cancel_users.get(user_id):
                await status_msg.edit("Processing cancelled.")
                cancel_users[user_id] = False
                return None

            ret, frame = cap.read()

            if not ret:
                break

            current += 1

            if current not in selected_set:
                continue

            if frame.shape[1] > 720:
                scale = 720 / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (720, int(frame.shape[0] * scale))
                )

            frame_path = os.path.join(
                user_frame_dir,
                f"{saved}.jpg"
            )

            success = cv2.imwrite(
                frame_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if not success:
                await status_msg.edit(
                    f"Failed to save frame {saved}"
                )
                return None

            saved += 1

            progress = int(
                (saved / len(selected_frames)) * 100
            )

            try:
                await status_msg.edit(
                    f"Extracting frames...\n"
                    f"Progress: {progress}%\n"
                    f"Saved: {saved}/{len(selected_frames)}",
                    reply_markup=cancel_markup
                )
            except Exception:
                pass

            if saved >= len(selected_frames):
                break

    finally:
        cap.release()

    if saved == 0:
        await status_msg.edit("No frames extracted")
        return None

    # CREATE ZIP

    zip_path = os.path.join(
        OUTPUT_DIR,
        f"{user_id}_highlights.zip"
    )

    try:
        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED
        ) as archive:
            for i in range(saved):
                frame_path = os.path.join(
                    user_frame_dir,
                    f"{i}.jpg"
                )

                if not os.path.exists(frame_path):
                    await status_msg.edit(
                        f"Missing frame: {i}"
                    )
                    return None

                archive.write(
                    frame_path,
                    arcname=f"{i + 1}.jpg"
                )

    except Exception as error:
        await status_msg.edit(
            f"ZIP creation failed: {error}"
        )
        return None

    finally:
        shutil.rmtree(
            user_frame_dir,
            ignore_errors=True
        )

    return zip_path


# ---------------- CALLBACK ----------------

@bot.on_callback_query()
async def callback(_, query):
    if query.data == "cancel":
        user_id = query.from_user.id
        cancel_users[user_id] = True

        await query.answer(
            "Processing cancellation requested!"
        )

        try:
            await query.message.edit(
                "Processing cancellation requested."
            )
        except Exception:
            pass


# ---------------- COMMANDS ----------------

@bot.on_message(filters.command("start"))
async def start(_, msg):
    limits = get_current_limits()

    await msg.reply(
        f"🎬 **Video Highlight Bot**\n\n"
        f"Send me a video or use /url to get "
        f"the best highlights!\n\n"
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
        return await msg.reply(
            "**Usage:** `/settings <1-200>`"
        )

    try:
        count = int(msg.command[1])
    except ValueError:
        return await msg.reply(
            "Please enter a valid number"
        )

    if count < 1 or count > MAX_LIMIT:
        return await msg.reply(
            f"Please enter a number between 1 and {MAX_LIMIT}"
        )

    set_count(msg.from_user.id, count)

    await msg.reply(
        f"Image count set to **{count}**"
    )


@bot.on_message(filters.command("limits"))
async def limits_command(_, msg):
    limits = get_current_limits()

    await msg.reply(
        f"📊 **Current Limits:**\n\n"
        f"📦 **Max File Size:** "
        f"{limits['max_size_gb']} GB ({limits['max_size_mb']} MB)\n"
        f"⏱️ **Max Duration:** "
        f"{limits['max_duration_minutes']} minutes "
        f"({limits['max_duration_hours']} hours)\n\n"
        f"Videos exceeding these limits will be rejected."
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
        return await msg.reply(
            "Invalid URL. Please include http:// or https://"
        )

    status = await msg.reply("Checking video...")
    video_path = None
    zip_path = None

    try:
        video_path = await download_with_status(url, status)

        if not video_path:
            return

        await status.edit("Processing video...")

        zip_path = await process_video(
            video_path,
            user_id,
            target,
            status
        )

        if not zip_path:
            return await status.edit("Processing failed")

        await msg.reply_document(
            zip_path,
            caption=(
                f"✅ **Highlights extracted from URL**\n\n"
                f"📸 **Images:** {target}\n"
                f"🎬 **Source:** {url[:50]}..."
            )
        )

        await status.edit("Done! Check the ZIP file above.")

    except Exception as error:
        log.exception("URL handler failed for user %s", user_id)
        try:
            await status.edit(
                f"Download or processing failed\n{str(error)[:500]}"
            )
        except Exception:
            pass

    finally:
        for path in (video_path, zip_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                log.exception("Could not remove URL temporary file")


@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    target = min(get_count(user_id), MAX_LIMIT)
    cancel_users[user_id] = False

    status = await msg.reply("Downloading video...")
    video_path = None
    zip_path = None

    try:
        video_path = await msg.download(
            file_name=f"{DOWNLOAD_DIR}/"
        )

        if not video_path:
            return await status.edit("Video download failed")

        await status.edit("Processing video...")

        zip_path = await process_video(
            video_path,
            user_id,
            target,
            status
        )

        if not zip_path:
            return await status.edit("Processing failed")

        await msg.reply_document(
            zip_path,
            caption=(
                f"✅ **Highlights extracted**\n\n"
                f"📸 **Images:** {target}\n"
                f"📹 **Source:** Telegram video"
            )
        )

        await status.edit("Done! Check the ZIP file above.")

    except Exception as error:
        log.exception(
            "Video handler failed for user %s",
            user_id
        )

        try:
            await status.edit(
                f"Video processing error:\n{str(error)[:500]}"
            )
        except Exception:
            pass

    finally:
        for path in (video_path, zip_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                log.exception("Could not remove temporary file")


@bot.on_message(filters.command("help"))
async def help_command(_, msg):
    limits = get_current_limits()

    await msg.reply(
        f"**📖 Help Guide**\n\n"
        f"**How to use:**\n"
        f"1️⃣ Send me any video file\n"
        f"2️⃣ Or use `/url <video_link>`\n"
        f"3️⃣ I'll analyse and extract highlights\n"
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
        f"• NSFW content filtering\n"
        f"• High-quality JPEG output\n"
        f"• 720p processing\n"
        f"• Real-time progress bars\n"
        f"• Cancel button\n\n"
        f"**Support:**\n"
        f"• Send /start to begin\n"
        f"• Send /help for this menu"
    )


# ---------------- UPSCALER ----------------

@bot.on_message(filters.command("up"))
async def up_command(_, msg):
    user_id = msg.from_user.id

    upscale_waiting.add(user_id)
    upscale_cancelled.discard(user_id)
    upscale_caption_waiting.discard(user_id)

    upscale_captions[user_id] = (
        "✅ 4× Image Upscaling Complete"
    )

    await msg.reply(
        "🖼️ **4× Image Upscaler**\n\n"
        "📦 Send your ZIP file containing images.\n\n"
        "📝 To use a custom caption:\n"
        "`/caption Your custom caption`\n\n"
        "Example:\n"
        "`/caption ✨ My 4K Images`\n\n"
        "❌ `/cancel` to stop."
    )


@bot.on_message(filters.command("caption"))
async def set_upscale_caption(_, msg):
    user_id = msg.from_user.id

    if user_id not in upscale_waiting:
        return await msg.reply(
            "Start the upscaler first with `/up`."
        )

    if len(msg.command) < 2:
        return await msg.reply(
            "Usage: `/caption Your custom caption`"
        )

    caption = msg.text.split(" ", 1)[1].strip()

    if not caption:
        return await msg.reply("Caption cannot be empty.")

    if len(caption) > 1024:
        return await msg.reply(
            "Caption is too long. Maximum: 1024 characters."
        )

    upscale_captions[user_id] = caption

    await msg.reply(
        "✅ **Custom caption saved.**\n\n"
        f"📝 {caption}\n\n"
        "Now send your ZIP file."
    )


@bot.on_message(filters.document)
async def upscale_zip_handler(_, msg):
    user_id = msg.from_user.id

    if user_id not in upscale_waiting:
        return

    filename = (msg.document.file_name or "").lower()

    if not filename.endswith(".zip"):
        return await msg.reply("Please send a ZIP file.")

    upscale_waiting.discard(user_id)
    upscale_cancelled.discard(user_id)

    caption = upscale_captions.get(
        user_id,
        "✅ 4× Image Upscaling Complete"
    )

    status = await msg.reply(
        "📦 **ZIP received**\n⬇️ Downloading..."
    )

    user_dir = os.path.join(
        "upscale_work",
        str(user_id)
    )

    zip_path = None

    try:
        os.makedirs(user_dir, exist_ok=True)

        zip_path = await msg.download(
            file_name=os.path.join(
                user_dir,
                "input.zip"
            )
        )

        if not zip_path:
            raise RuntimeError("ZIP download returned no file path")

    except Exception as error:
        cleanup_user(user_id)
        upscale_captions.pop(user_id, None)

        return await status.edit(
            f"Download failed\n\n{error}"
        )

    await status.edit(
        "ZIP received successfully\n\nFinding images..."
    )

    loop = asyncio.get_running_loop()
    last_update = {"percent": -1}

    async def update_status(percent, current, total):
        if (
            percent == last_update["percent"]
            and percent != 100
        ):
            return

        last_update["percent"] = percent
        bar_length = 12
        filled = int(bar_length * percent / 100)

        bar = (
            "█" * filled
            + "░" * (bar_length - filled)
        )

        try:
            await status.edit(
                "🖼️ **Upscaling Images**\n\n"
                f"`{bar}` **{percent}%**\n\n"
                f"📸 Image: **{current}/{total}**\n"
                "🔍 Scale: **4×**\n"
                "📦 Output: ZIP\n\n"
                "🛑 Use `/cancel` to stop."
            )
        except Exception:
            pass

    def progress_callback(percent, current, total):
        try:
            asyncio.run_coroutine_threadsafe(
                update_status(percent, current, total),
                loop
            )
        except Exception:
            pass

    def cancel_callback():
        return (
            user_id in upscale_cancelled
            or is_shutting_down()
        )

    output_zip = None
    processed = 0
    failed = 0
    cancelled = False

    try:
        (
            output_zip,
            processed,
            failed,
            cancelled
        ) = await asyncio.to_thread(
            process_zip,
            zip_path,
            user_id,
            4,
            progress_callback,
            cancel_callback
        )

    except Exception as error:
        log.exception(
            "Upscaling failed for user %s",
            user_id
        )

        cleanup_user(user_id)
        upscale_cancelled.discard(user_id)
        upscale_captions.pop(user_id, None)

        return await status.edit(
            f"Upscaling failed\n\n{error}"
        )

    if cancelled:
        cleanup_user(user_id)
        upscale_cancelled.discard(user_id)
        upscale_captions.pop(user_id, None)

        return await status.edit(
            "🛑 **Upscaling cancelled.**\n\n"
            f"📸 Processed before cancellation: **{processed}**"
        )

    if not output_zip or not os.path.exists(output_zip):
        cleanup_user(user_id)
        upscale_captions.pop(user_id, None)

        return await status.edit("Output ZIP was not created.")

    await status.edit(
        "📦 **Upscaling complete!**\n\n"
        "⬆️ Uploading final ZIP..."
    )

    try:
        await msg.reply_document(
            output_zip,
            caption=caption
        )

        await status.edit(
            "✅ **Done!**\n\n"
            f"🖼️ Processed: **{processed}**\n"
            f"⚠️ Failed: **{failed}**\n"
            "🔍 Scale: **4×**\n"
            "📦 Output: ZIP"
        )

    except Exception as error:
        await status.edit(
            f"Failed to send ZIP\n\n{error}"
        )

    finally:
        cleanup_user(user_id)
        upscale_cancelled.discard(user_id)
        upscale_caption_waiting.discard(user_id)
        upscale_captions.pop(user_id, None)


@bot.on_message(filters.command("cancel"))
async def cancel_upscale(_, msg):
    user_id = msg.from_user.id

    if user_id in upscale_waiting:
        upscale_waiting.discard(user_id)
        upscale_caption_waiting.discard(user_id)
        upscale_cancelled.discard(user_id)
        upscale_captions.pop(user_id, None)

        cleanup_user(user_id)

        return await msg.reply(
            "🛑 **Upscaler cancelled.**"
        )

    upscale_cancelled.add(user_id)

    await msg.reply(
        "🛑 **Cancel requested.**\n\n"
        "The current image will finish, then processing will stop."
    )


# ============================================================
# /HEAL AND /HEALTH
# ============================================================

def _human_uptime(seconds):
    seconds = int(seconds)

    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    parts = []

    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")

    parts.append(f"{secs}s")

    return " ".join(parts)


def _dir_size_mb(path):
    total = 0

    try:
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(
                        os.path.join(root, name)
                    )
                except OSError:
                    pass
    except Exception:
        pass

    return total / (1024 * 1024)


async def _health_report():
    import shutil as _shutil

    uptime_seconds = None

    try:
        with open("/proc/self/stat") as f:
            stat = f.read().split()

        starttime_ticks = int(stat[21])
        clock_ticks_per_sec = os.sysconf("SC_CLK_TCK")
        boot_seconds = time.clock_gettime(time.CLOCK_BOOTTIME)

        uptime_seconds = (
            boot_seconds
            - (starttime_ticks / clock_ticks_per_sec)
        )

    except Exception:
        uptime_seconds = None

    if uptime_seconds is None or uptime_seconds <= 0:
        uptime_seconds = time.time() - _BOT_START_TIME

    mem_mb = None

    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_mb = int(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass

    try:
        disk = _shutil.disk_usage(".")
        disk_used_pct = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 ** 3)
    except Exception:
        disk_used_pct = None
        disk_free_gb = None

    iv_work_mb = _dir_size_mb("image_video_work")
    downloads_mb = _dir_size_mb(DOWNLOAD_DIR)
    output_mb = _dir_size_mb(OUTPUT_DIR)

    try:
        from image_video import sessions as iv_sessions
        iv_count = len(iv_sessions)
    except Exception:
        iv_count = 0

    upscale_count = len(upscale_waiting)

    video_busy = sum(
        1 for value in cancel_users.values()
        if value
    )

    try:
        connected = bot.is_connected
    except Exception:
        connected = False

    flask_ok = False

    try:
        port = os.environ.get("PORT", "10000")
        response = requests.get(
            f"http://127.0.0.1:{port}/health",
            timeout=3
        )
        flask_ok = response.status_code == 200
    except Exception:
        flask_ok = False

    shutting = is_shutting_down()

    lines = [
        "🏥 **Server Health Report**",
        "",
        "**Process**",
        f"⏱️ Uptime: `{_human_uptime(uptime_seconds)}`",
    ]

    if mem_mb is not None:
        lines.append(f"💾 Memory (RSS): `{mem_mb:.1f} MB`")

    lines.append(
        f"🔁 Shutting down: `{'yes' if shutting else 'no'}`"
    )

    lines.extend([
        "",
        "**Services**",
        f"🤖 Pyrogram: `{'connected' if connected else 'disconnected'}`",
        f"🌐 Flask: `{'ok' if flask_ok else 'unreachable'}`",
        "",
        "**Disk**",
    ])

    if disk_used_pct is not None:
        lines.append(
            f"💽 Disk used: `{disk_used_pct:.1f}%` "
            f"(free `{disk_free_gb:.2f} GB`)"
        )

    lines.extend([
        f"📁 image_video_work: `{iv_work_mb:.1f} MB`",
        f"📁 downloads: `{downloads_mb:.1f} MB`",
        f"📁 output: `{output_mb:.1f} MB`",
        "",
        "**Workload**",
        f"🎬 Image→video sessions: `{iv_count}`",
        f"🖼️ Upscaler waiting: `{upscale_count}`",
        f"🔄 Video jobs in progress: `{video_busy}`",
    ])

    return "\n".join(lines)


@bot.on_message(filters.command(["heal", "health"]))
async def health_command(_, msg):
    try:
        report = await _health_report()
        await msg.reply(report)

    except Exception as error:
        log.exception("Health report failed: %s", error)

        await msg.reply(
            f"Health check failed: `{str(error)[:300]}`"
        )


# ============================================================
# IMAGE TO VIDEO HANDLERS
# ============================================================

register_image_video_handlers(bot)


# ============================================================
# GRACEFUL SHUTDOWN AND PYROGRAM EVENT LOOP
# ============================================================

def run_bot_forever():
    """
    Run Pyrogram on the exact event loop captured by Client.

    The loop is created and installed before Client construction.
    It is reused for bot.start(), dispatcher workers, and bot.stop().
    """

    # Ensure this process owns the shutdown flag.
    try:
        if os.path.exists(SHUTDOWN_FLAG):
            os.remove(SHUTDOWN_FLAG)
            log.warning(
                "Removed stale shutdown flag: %s",
                SHUTDOWN_FLAG
            )
    except Exception:
        log.exception("Could not clear stale shutdown flag")

    # Install signal handlers.
    def _on_signal(signum, _frame):
        log.warning(
            "Signal %s received in bot process",
            signum
        )
        request_shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            log.exception("Could not install signal handler")

    log.info("Bot starting")
    log.info(
        "Using dedicated Pyrogram event loop: %s",
        id(_BOT_LOOP)
    )

    async def _run():
        started = False

        try:
            await bot.start()
            started = True

            log.info(
                "Bot started (connected=%s)",
                bot.is_connected
            )

            try:
                me = await bot.get_me()
                log.info(
                    "Bot identity: @%s (id=%s)",
                    me.username,
                    me.id
                )
            except Exception:
                log.exception("Telegram get_me failed")

            last_health = 0.0

            while not is_shutting_down():
                now = time.time()

                if now - last_health >= 60:
                    try:
                        connected = bot.is_connected
                    except Exception:
                        connected = False

                    if not connected:
                        log.warning("Pyrogram connection dropped")

                        try:
                            if started:
                                await bot.stop()
                                started = False

                            # Restart on the same dedicated event loop.
                            await bot.start()
                            started = True

                            log.info(
                                "Pyrogram restarted successfully"
                            )

                            me = await bot.get_me()
                            log.info(
                                "Bot identity after restart: @%s",
                                me.username
                            )

                        except Exception:
                            log.exception("Pyrogram restart failed")

                    last_health = now

                await asyncio.sleep(1)

        finally:
            if started:
                log.info(
                    "Shutdown requested - stopping bot"
                )

                try:
                    await bot.stop()
                    log.info("Bot stopped cleanly")
                except Exception:
                    log.exception("Bot shutdown failed")

    try:
        # Do not use asyncio.run(): it creates a different event loop.
        asyncio.set_event_loop(_BOT_LOOP)
        _BOT_LOOP.run_until_complete(_run())

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt - exiting bot")

    except Exception:
        log.exception("Bot loop error")

    finally:
        # Finish any remaining tasks on this same loop before closing it.
        try:
            pending = asyncio.all_tasks(_BOT_LOOP)

            for task in pending:
                task.cancel()

            if pending:
                _BOT_LOOP.run_until_complete(
                    asyncio.gather(
                        *pending,
                        return_exceptions=True
                    )
                )

            _BOT_LOOP.run_until_complete(
                _BOT_LOOP.shutdown_asyncgens()
            )

        except Exception:
            log.exception("Error while cleaning up event loop")

        try:
            _BOT_LOOP.close()
        except Exception:
            log.exception("Could not close bot event loop")

        try:
            asyncio.set_event_loop(None)
        except Exception:
            pass

        log.info("Bot process exiting")

        try:
            if os.path.exists(SHUTDOWN_FLAG):
                os.remove(SHUTDOWN_FLAG)
        except Exception:
            pass
