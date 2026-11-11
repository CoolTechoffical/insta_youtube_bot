import os
import re
import shutil
import zipfile
import tempfile
import asyncio
import queue
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pyrogram import filters
from pyrogram.handlers import MessageHandler


# ---------------- CONFIGURATION ----------------

MAX_IMAGES = 300
MAX_ZIP_SIZE = 100 * 1024 * 1024
MAX_EXTRACTED_SIZE = 500 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

# 720p / 15 FPS keeps CPU and memory use much lower than the old 1080p/30 FPS
WIDTH = 1280
HEIGHT = 720
FPS = 15

MIN_IMAGE_DURATION = 0.65
DEFAULT_IMAGE_DURATION = 1.2
MAX_VIDEO_DURATION = 600

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac"}

WORK_DIR = "image_video_work"
OUTPUT_DIR = "output"

sessions = {}


# ---------------- HELPERS ----------------

def natural_sort_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def prepare_image(source, destination):
    """Normalize one image to 720p and save it as JPEG."""
    with Image.open(source) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image resolution is too large.")

        image = ImageOps.exif_transpose(image).convert("RGB")

        target_ratio = WIDTH / HEIGHT
        current_ratio = image.width / image.height

        if current_ratio > target_ratio:
            new_width = max(1, int(image.height * target_ratio))
            left = (image.width - new_width) // 2
            image = image.crop((left, 0, left + new_width, image.height))
        else:
            new_height = max(1, int(image.width / target_ratio))
            top = (image.height - new_height) // 2
            image = image.crop((0, top, image.width, top + new_height))

        image = image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
        image.save(destination, "JPEG", quality=88, optimize=False)


# ---------------- ZIP EXTRACTION ----------------

def extract_zip(zip_path, work):
    if os.path.getsize(zip_path) > MAX_ZIP_SIZE:
        raise ValueError("ZIP file exceeds the 100 MB limit.")

    extracted = []
    total_size = 0

    with zipfile.ZipFile(zip_path) as archive:
        members = []

        for info in archive.infolist():
            if info.is_dir():
                continue

            filename = info.filename.replace("\\", "/")
            parts = Path(filename).parts

            # Reject absolute paths and path traversal.
            if filename.startswith("/") or ".." in parts:
                continue

            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            members.append(info)

        members.sort(key=lambda item: natural_sort_key(item.filename))

        if len(members) > MAX_IMAGES:
            raise ValueError(f"Maximum {MAX_IMAGES} images allowed.")

        for index, info in enumerate(members):
            total_size += info.file_size
            if total_size > MAX_EXTRACTED_SIZE:
                raise ValueError("Extracted images exceed the 500 MB limit.")

            extension = Path(info.filename).suffix.lower()
            destination = os.path.join(work, f"source_{index:04d}{extension}")

            # Stream each file to disk; do not load the ZIP into RAM.
            written = 0
            with archive.open(info) as source, open(destination, "wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.file_size or written > MAX_EXTRACTED_SIZE:
                        raise ValueError("Invalid or oversized ZIP entry.")
                    output.write(chunk)

            extracted.append((info.filename, destination))

    if not extracted:
        raise ValueError("The ZIP file contains no supported images.")

    return extracted


# ---------------- IMAGE COLLECTION ----------------

def prepare_images(session):
    work = session["work"]
    sources = []

    if session.get("zip_path"):
        sources.extend(extract_zip(session["zip_path"], work))

    sources.extend(session["images"])
    sources.sort(key=lambda item: natural_sort_key(item[0]))

    if not sources:
        raise ValueError("No images uploaded.")

    if len(sources) > MAX_IMAGES:
        raise ValueError(f"Maximum {MAX_IMAGES} images allowed.")

    prepared = []

    for index, (filename, source) in enumerate(sources):
        destination = os.path.join(work, f"image_{index:04d}.jpg")
        try:
            prepare_image(source, destination)
            prepared.append(destination)
        except Exception as error:
            print(f"Skipping invalid image {filename}: {error}")

    if not prepared:
        raise ValueError("No valid images found.")

    return prepared


# ---------------- LOW-RESOURCE VIDEO RENDERING ----------------

def _make_frame(base, effect, frame_index, frame_count):
    """
    Create one output frame at a time.
    Only a single 720p frame is held in memory; no video-sized frame list.
    """
    progress = frame_index / max(1, frame_count - 1)
    zoom = 1.0 + (0.07 * progress if effect == "zoom_in" else 0.07 * (1.0 - progress)
                 if effect == "zoom_out" else 0.055)

    crop_w = max(2, int(WIDTH / zoom))
    crop_h = max(2, int(HEIGHT / zoom))

    max_x = max(0, WIDTH - crop_w)
    max_y = max(0, HEIGHT - crop_h)

    if effect == "pan_left":
        x = int(max_x * progress)
        y = (HEIGHT - crop_h) // 2
    elif effect == "pan_right":
        x = int(max_x * (1.0 - progress))
        y = (HEIGHT - crop_h) // 2
    elif effect == "pan_up":
        x = (WIDTH - crop_w) // 2
        y = int(max_y * progress)
    elif effect == "pan_down":
        x = (WIDTH - crop_w) // 2
        y = int(max_y * (1.0 - progress))
    else:
        x = (WIDTH - crop_w) // 2
        y = (HEIGHT - crop_h) // 2

    # The crop dimensions are never larger than the source image.
    x = max(0, min(x, WIDTH - crop_w))
    y = max(0, min(y, HEIGHT - crop_h))
    cropped = base[y:y + crop_h, x:x + crop_w]

    if cropped.size == 0:
        cropped = base

    frame = cv2.resize(cropped, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    # Lightweight cinematic fade at the start and end of each image.
    fade_frames = max(1, int(FPS * 0.16))
    fade_in = min(1.0, (frame_index + 1) / fade_frames)
    fade_out = min(1.0, (frame_count - frame_index) / fade_frames)
    alpha = min(fade_in, fade_out)

    if alpha < 1.0:
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

    return frame


def render_video(images, output_path, total_seconds, progress_callback=None):
    """
    Render with OpenCV VideoWriter. No subprocess and no external ffmpeg command.
    Note: OpenCV may use codec libraries internally, depending on its build.
    """
    count = len(images)
    if count == 0:
        raise ValueError("No images available to render.")

    duration = max(MIN_IMAGE_DURATION, float(total_seconds) / count)
    frames_per_image = max(2, round(duration * FPS))
    actual_duration = (frames_per_image * count) / FPS

    if actual_duration > MAX_VIDEO_DURATION:
        raise ValueError("Video exceeds the 10-minute limit.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # mp4v is commonly available in opencv-python-headless Linux wheels.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "OpenCV could not open an MP4 video writer. "
            "Check that opencv-python-headless is installed with MP4 codec support."
        )

    effects = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]
    total_frames = frames_per_image * count
    completed_frames = 0
    last_reported = -1

    try:
        for image_index, image_path in enumerate(images):
            base = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if base is None:
                print(f"Skipping unreadable prepared image: {image_path}")
                continue

            if base.shape[1] != WIDTH or base.shape[0] != HEIGHT:
                base = cv2.resize(base, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

            effect = effects[image_index % len(effects)]

            for frame_index in range(frames_per_image):
                frame = _make_frame(base, effect, frame_index, frames_per_image)
                writer.write(frame)
                completed_frames += 1

                percent = int((completed_frames / max(1, total_frames)) * 100)
                # Send progress only at 5% intervals to avoid Telegram API spam.
                if progress_callback and (percent >= last_reported + 5 or percent == 100):
                    last_reported = percent
                    progress_callback(percent, image_index + 1, count)

            del base

    finally:
        writer.release()

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
        raise RuntimeError("Video rendering failed: output file is empty.")

    return actual_duration


# ---------------- FINISH VIDEO ----------------

async def finish_video(client, message):
    if not message.from_user:
        return

    user_id = message.from_user.id
    session = sessions.get(user_id)

    if not session:
        return await message.reply("Start with /imagevideo first.")

    if session.get("processing"):
        return await message.reply("Your video is already being created.")

    session["processing"] = True
    status = await message.reply("🖼️ Checking uploaded images...")
    output = None

    try:
        images = await asyncio.to_thread(prepare_images, session)
        count = len(images)

        requested_duration = session.get("duration")
        if requested_duration is None:
            requested_duration = max(5, count * DEFAULT_IMAGE_DURATION)

        requested_duration = min(float(requested_duration), MAX_VIDEO_DURATION)

        await status.edit(
            "🎬 Preparing low-resource video render...\n\n"
            f"📸 Images: {count}\n"
            f"⏱️ Target duration: {int(requested_duration)} seconds\n"
            f"🎞️ Resolution: {WIDTH}×{HEIGHT} at {FPS} FPS\n"
            "⚙️ Renderer: OpenCV (no external FFmpeg command)\n"
            "Progress: 0%"
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output = os.path.join(OUTPUT_DIR, f"image_video_{user_id}.mp4")

        loop = asyncio.get_running_loop()
        progress_queue = queue.Queue()

        def report_progress(percent, current_image, total_images):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                (percent, current_image, total_images)
            )

        render_task = asyncio.create_task(
            asyncio.to_thread(
                render_video,
                images,
                output,
                requested_duration,
                report_progress
            )
        )

        last_percent = -1
        last_edit_time = 0

        while not render_task.done():
            try:
                percent, current_image, total_images = await asyncio.wait_for(
                    asyncio.to_thread(progress_queue.get, True, 2.0),
                    timeout=2.5
                )
                now = time.monotonic()
                if percent != last_percent and (now - last_edit_time >= 1.5 or percent == 100):
                    bar_filled = percent // 5
                    bar = "█" * bar_filled + "░" * (20 - bar_filled)
                    try:
                        await status.edit(
                            "🎬 Creating your video...\n\n"
                            f"[{bar}] {percent}%\n"
                            f"📸 Image: {current_image}/{total_images}\n"
                            f"🎞️ {WIDTH}×{HEIGHT} · {FPS} FPS"
                        )
                        last_percent = percent
                        last_edit_time = now
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                continue
            except queue.Empty:
                continue

        actual_duration = await render_task

        await message.reply_video(
            output,
            caption=(
                "🎬 **Your video is ready!**\n\n"
                f"📸 Images: {count}\n"
                f"⏱️ Duration: {int(actual_duration)} seconds\n"
                f"🎞️ Resolution: {WIDTH}×{HEIGHT} · {FPS} FPS\n"
                "✨ Automatic pan, zoom and fade effects"
            ),
            supports_streaming=True
        )

        await status.edit("✅ Video created successfully!")

    except Exception as error:
        try:
            await status.edit(f"❌ Video creation failed:\n{str(error)[:1500]}")
        except Exception:
            pass

    finally:
        sessions.pop(user_id, None)
        shutil.rmtree(session["work"], ignore_errors=True)

        if output and os.path.exists(output):
            try:
                os.remove(output)
            except OSError:
                pass


# ---------------- PYROGRAM HANDLERS ----------------

def register_image_video_handlers(bot):

    async def start(_, message):
        if not message.from_user:
            return

        user_id = message.from_user.id
        old = sessions.pop(user_id, None)

        if old:
            shutil.rmtree(old["work"], ignore_errors=True)

        duration = None
        if len(message.command) > 1:
            try:
                duration = float(message.command[1])
                if not 5 <= duration <= MAX_VIDEO_DURATION:
                    raise ValueError
            except ValueError:
                return await message.reply(
                    "Usage: /imagevideo 90\n"
                    "Duration must be 5–600 seconds."
                )

        os.makedirs(WORK_DIR, exist_ok=True)
        work = tempfile.mkdtemp(prefix=f"{user_id}_", dir=WORK_DIR)

        sessions[user_id] = {
            "work": work,
            "images": [],
            "zip_path": None,
            "duration": duration,
            "processing": False
        }

        await message.reply(
            "🎬 **Image to Video Creator**\n\n"
            "Send one ZIP file OR individual images.\n\n"
            "📸 Maximum: 300 images\n"
            "🎞️ Output: 720p · 15 FPS\n"
            "✨ Automatic pan, zoom and fade effects\n"
            "⏱️ Duration: Automatic, or set it with /imagevideo 90\n\n"
            "Commands:\n"
            "/imagevideo 90 - Set 90-second duration\n"
            "/ivmusic - Check audio support\n"
            "/finishvideo - Create video\n"
            "/cancelvideo - Cancel session"
        )

    async def receive_photo(_, message):
        if not message.from_user:
            return

        session = sessions.get(message.from_user.id)
        if not session or session.get("processing"):
            return

        if session["zip_path"]:
            return await message.reply("You already sent a ZIP file.")

        if len(session["images"]) >= MAX_IMAGES:
            return await message.reply(f"Maximum {MAX_IMAGES} images reached.")

        index = len(session["images"]) + 1
        filename = f"photo_{index:04d}.jpg"
        path = os.path.join(session["work"], filename)

        await message.download(file_name=path)
        session["images"].append((filename, path))
        await message.reply(f"✅ Image added: {len(session['images'])}/{MAX_IMAGES}")

    async def receive_document(_, message):
        if not message.from_user or not message.document:
            return

        session = sessions.get(message.from_user.id)
        if not session or session.get("processing"):
            return

        filename = message.document.file_name or "upload"
        extension = Path(filename).suffix.lower()
        file_size = message.document.file_size or 0

        if extension in AUDIO_EXTENSIONS:
            return await message.reply(
                "🎵 Background music is not available in this lightweight "
                "OpenCV-only version. Video creation will continue without audio."
            )

        if extension == ".zip":
            if session["images"] or session["zip_path"]:
                return await message.reply("Use one ZIP OR individual images.")

            if file_size > MAX_ZIP_SIZE:
                return await message.reply("ZIP limit is 100 MB.")

            path = os.path.join(session["work"], "images.zip")
            await message.download(file_name=path)
            session["zip_path"] = path
            return await message.reply("📦 ZIP received. Send /finishvideo when ready.")

        if extension not in IMAGE_EXTENSIONS:
            return

        if session["zip_path"]:
            return await message.reply("You already sent a ZIP file.")

        if len(session["images"]) >= MAX_IMAGES:
            return await message.reply(f"Maximum {MAX_IMAGES} images reached.")

        path = os.path.join(
            session["work"],
            f"document_{len(session['images']) + 1:04d}{extension}"
        )
        await message.download(file_name=path)
        session["images"].append((filename, path))
        await message.reply(f"✅ Image added: {len(session['images'])}/{MAX_IMAGES}")

    async def finish(_, message):
        await finish_video(bot, message)

    async def cancel(_, message):
        if not message.from_user:
            return

        session = sessions.get(message.from_user.id)
        if not session:
            return await message.reply("No active session.")

        if session.get("processing"):
            return await message.reply(
                "Video rendering is already running. Please wait for it to finish."
            )

        sessions.pop(message.from_user.id, None)
        shutil.rmtree(session["work"], ignore_errors=True)
        await message.reply("🛑 Image-to-video session cancelled.")

    async def add_music(_, message):
        if not message.from_user:
            return

        if message.from_user.id not in sessions:
            return await message.reply("Start with /imagevideo first.")

        await message.reply(
            "🎵 Music muxing is disabled in this lightweight version so the "
            "server does not need an external FFmpeg command. The video will "
            "be created without background audio."
        )

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bot.add_handler(MessageHandler(start, filters.command("imagevideo")), group=1)
    bot.add_handler(MessageHandler(finish, filters.command("finishvideo")), group=1)
    bot.add_handler(MessageHandler(cancel, filters.command("cancelvideo")), group=1)
    bot.add_handler(MessageHandler(add_music, filters.command("ivmusic")), group=1)
    bot.add_handler(MessageHandler(receive_photo, filters.photo), group=1)
    bot.add_handler(MessageHandler(receive_document, filters.document), group=1)
