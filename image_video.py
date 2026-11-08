
import os
import re
import shutil
import random
import zipfile
import tempfile
import subprocess
import asyncio

from pathlib import Path
from PIL import Image, ImageOps
from pyrogram import filters
from pyrogram.handlers import MessageHandler

# ---------------- CONFIGURATION ----------------

MAX_IMAGES = 300
MAX_ZIP_SIZE = 100 * 1024 * 1024
MAX_EXTRACTED_SIZE = 500 * 1024 * 1024

WIDTH = 1280
HEIGHT = 720
FPS = 30

MIN_IMAGE_DURATION = 0.65
DEFAULT_IMAGE_DURATION = 1.2
MAX_VIDEO_DURATION = 600

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac"
}

WORK_DIR = "image_video_work"
OUTPUT_DIR = "output"

sessions = {}


# ---------------- HELPERS ----------------

def natural_sort_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is not installed on the server."
        )


def run_ffmpeg(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-1500:]
        )


def prepare_image(source, destination):

    with Image.open(source) as image:

        if image.width * image.height > 40_000_000:
            raise ValueError("Image resolution is too large.")

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        target_ratio = WIDTH / HEIGHT
        current_ratio = image.width / image.height

        if current_ratio > target_ratio:

            new_width = int(image.height * target_ratio)
            left = (image.width - new_width) // 2

            image = image.crop((
                left,
                0,
                left + new_width,
                image.height
            ))

        else:

            new_height = int(image.width / target_ratio)
            top = (image.height - new_height) // 2

            image = image.crop((
                0,
                top,
                image.width,
                top + new_height
            ))

        image = image.resize(
            (WIDTH, HEIGHT),
            Image.Resampling.LANCZOS
        )

        image.save(
            destination,
            "JPEG",
            quality=94
        )


# ---------------- ZIP EXTRACTION ----------------

def extract_zip(zip_path, work):

    if os.path.getsize(zip_path) > MAX_ZIP_SIZE:
        raise ValueError("ZIP file exceeds 100 MB.")

    extracted = []
    total_size = 0

    with zipfile.ZipFile(zip_path) as archive:

        members = []

        for info in archive.infolist():

            if info.is_dir():
                continue

            filename = info.filename.replace("\\", "/")

            if filename.startswith("/") or ".." in Path(filename).parts:
                continue

            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            members.append(info)

        members.sort(
            key=lambda item: natural_sort_key(item.filename)
        )

        if len(members) > MAX_IMAGES:
            raise ValueError(
                f"Maximum {MAX_IMAGES} images allowed."
            )

        for index, info in enumerate(members):

            total_size += info.file_size

            if total_size > MAX_EXTRACTED_SIZE:
                raise ValueError(
                    "Extracted images exceed 500 MB."
                )

            extension = Path(info.filename).suffix.lower()

            destination = os.path.join(
                work,
                f"source_{index:04d}{extension}"
            )

            with archive.open(info) as source:
                with open(destination, "wb") as output:
                    shutil.copyfileobj(source, output)

            extracted.append((
                info.filename,
                destination
            ))

    return extracted


# ---------------- IMAGE COLLECTION ----------------

def prepare_images(session):

    work = session["work"]
    sources = []

    if session.get("zip_path"):

        sources.extend(
            extract_zip(
                session["zip_path"],
                work
            )
        )

    sources.extend(session["images"])

    sources.sort(
        key=lambda item: natural_sort_key(item[0])
    )

    if not sources:
        raise ValueError("No images uploaded.")

    if len(sources) > MAX_IMAGES:
        raise ValueError("Maximum 300 images allowed.")

    prepared = []

    for index, (filename, source) in enumerate(sources):

        destination = os.path.join(
            work,
            f"image_{index:04d}.jpg"
        )

        try:
            prepare_image(source, destination)
            prepared.append(destination)

        except Exception as error:
            print(
                f"Skipping invalid image {filename}: {error}"
            )

    if not prepared:
        raise ValueError("No valid images found.")

    return prepared


# ---------------- VIDEO RENDERING ----------------

def render_video(images, output_path, total_seconds, music=None):

    check_ffmpeg()

    count = len(images)

    duration = max(
        MIN_IMAGE_DURATION,
        float(total_seconds) / count
    )

    actual_duration = duration * count

    if actual_duration > MAX_VIDEO_DURATION:
        raise ValueError("Video exceeds the 10-minute limit.")

    work = os.path.dirname(images[0])

    effects = [
        "zoom_in",
        "zoom_out",
        "pan_left",
        "pan_right",
        "pan_up",
        "pan_down"
    ]

    random.shuffle(effects)

    clips = []

    for index, image in enumerate(images):

        clip = os.path.join(
            work,
            f"clip_{index:04d}.mp4"
        )

        effect = effects[index % len(effects)]

        frames = max(
            2,
            round(duration * FPS)
        )

        zoom = {
            "zoom_in": "min(zoom+0.001,1.12)",
            "zoom_out": "max(zoom-0.001,1.0)",
            "pan_left": "1.08",
            "pan_right": "1.08",
            "pan_up": "1.08",
            "pan_down": "1.08"
        }[effect]

        x = {
            "pan_left": "iw/2-(iw/zoom/2)+on*0.35",
            "pan_right": "iw/2-(iw/zoom/2)-on*0.35"
        }.get(effect, "iw/2-(iw/zoom/2)")

        y = {
            "pan_up": "ih/2-(ih/zoom/2)+on*0.25",
            "pan_down": "ih/2-(ih/zoom/2)-on*0.25"
        }.get(effect, "ih/2-(ih/zoom/2)")

        fade_start = max(0, duration - 0.18)

        video_filter = (
            f"zoompan=z='{zoom}':"
            f"x='{x}':y='{y}':"
            f"d={frames}:"
            f"s={WIDTH}x{HEIGHT}:"
            f"fps={FPS},"
            f"fade=t=in:st=0:d=0.18,"
            f"fade=t=out:st={fade_start}:d=0.18,"
            "eq=contrast=1.04:saturation=1.08,"
            "vignette=PI/5,"
            "format=yuv420p"
        )

        run_ffmpeg([
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", image,
            "-vf", video_filter,
            "-frames:v", str(frames),
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            clip
        ])

        clips.append(clip)

    # Join all rendered image clips.

    concat_file = os.path.join(
        work,
        "concat.txt"
    )

    with open(concat_file, "w") as file:

        for clip in clips:
            safe_path = os.path.abspath(clip).replace("'", "'\\''")
            file.write(f"file '{safe_path}'\n")

    if music and os.path.exists(music):

        run_ffmpeg([
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-stream_loop", "-1",
            "-i", music,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "160k",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ])

    else:

        run_ffmpeg([
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path
        ])

    return actual_duration


# ---------------- FINISH VIDEO ----------------

async def finish_video(client, message):

    user_id = message.from_user.id
    session = sessions.get(user_id)

    if not session:
        return await message.reply(
            "Start with /imagevideo first."
        )

    status = await message.reply(
        "🖼️ Checking uploaded images..."
    )

    output = None

    try:

        images = await asyncio.to_thread(
            prepare_images,
            session
        )

        count = len(images)

        requested_duration = session.get("duration")

        if requested_duration is None:

            requested_duration = max(
                30,
                count * DEFAULT_IMAGE_DURATION
            )

        requested_duration = min(
            requested_duration,
            MAX_VIDEO_DURATION
        )

        await status.edit(
            "🎬 Rendering cinematic video...\n\n"
            f"📸 Images: {count}\n"
            f"⏱️ Target duration: {int(requested_duration)} seconds\n"
            "✨ Mixed automatic effects\n"
            "🎞️ FFmpeg rendering..."
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output = os.path.join(
            OUTPUT_DIR,
            f"image_video_{user_id}.mp4"
        )

        actual_duration = await asyncio.to_thread(
            render_video,
            images,
            output,
            requested_duration,
            session.get("music_path")
        )

        await message.reply_video(
            output,
            caption=(
                "🎬 **Your cinematic video is ready!**\n\n"
                f"📸 Images: {count}\n"
                f"⏱️ Duration: {int(actual_duration)} seconds\n"
                "✨ Mixed automatic motion effects"
            ),
            supports_streaming=True
        )

        await status.edit("✅ Video created successfully!")

    except Exception as error:

        await status.edit(
            f"❌ Video creation failed:\n{str(error)[:2000]}"
        )

    finally:

        sessions.pop(user_id, None)

        shutil.rmtree(
            session["work"],
            ignore_errors=True
        )

        if output and os.path.exists(output):
            os.remove(output)


# ---------------- PYROGRAM HANDLERS ----------------

def register_image_video_handlers(bot):

    async def start(_, message):

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

        work = tempfile.mkdtemp(
            prefix=f"{user_id}_",
            dir=WORK_DIR
        )

        sessions[user_id] = {
            "work": work,
            "images": [],
            "zip_path": None,
            "music_path": None,
            "duration": duration
        }

        await message.reply(
            "🎬 **Image to Video Creator**\n\n"
            "Send a ZIP file OR individual images.\n\n"
            "📸 Maximum: 300 images\n"
            "✨ Style: Mixed automatic\n"
            "⏱️ Duration: Automatic\n\n"
            "Optional commands:\n"
            "/imagevideo 90 - Set 90-second duration\n"
            "/ivmusic - Add background music\n"
            "/finishvideo - Create video\n"
            "/cancelvideo - Cancel session"
        )

    async def receive_photo(_, message):

        user_id = message.from_user.id
        session = sessions.get(user_id)

        if not session:
            return

        if session["zip_path"]:
            return await message.reply(
                "You already sent a ZIP file."
            )

        if len(session["images"]) >= MAX_IMAGES:
            return await message.reply(
                "Maximum 300 images reached."
            )

        index = len(session["images"]) + 1

        filename = f"photo_{index:04d}.jpg"

        path = os.path.join(
            session["work"],
            filename
        )

        await message.download(file_name=path)

        session["images"].append((filename, path))

        await message.reply(
            f"✅ Image added: {len(session['images'])}/{MAX_IMAGES}"
        )

    async def receive_document(_, message):

        user_id = message.from_user.id
        session = sessions.get(user_id)

        if not session:
            return

        filename = message.document.file_name or "upload"
        extension = Path(filename).suffix.lower()

        if extension == ".zip":

            if session["images"] or session["zip_path"]:
                return await message.reply(
                    "Use one ZIP OR individual images."
                )

            if message.document.file_size > MAX_ZIP_SIZE:
                return await message.reply(
                    "ZIP limit is 100 MB."
                )

            path = os.path.join(
                session["work"],
                "images.zip"
            )

            await message.download(file_name=path)

            session["zip_path"] = path

            return await message.reply(
                "📦 ZIP received. Send /finishvideo when ready."
            )

        if extension in AUDIO_EXTENSIONS:

            if not session.get("want_music"):
                return

            path = os.path.join(
                session["work"],
                "background_music" + extension
            )

            await message.download(file_name=path)

            session["music_path"] = path
            session["want_music"] = False

            return await message.reply(
                "🎵 Background music added."
            )

        if extension not in IMAGE_EXTENSIONS:
            return

        if session["zip_path"]:
            return await message.reply(
                "You already sent a ZIP file."
            )

        if len(session["images"]) >= MAX_IMAGES:
            return await message.reply(
                "Maximum 300 images reached."
            )

        path = os.path.join(
            session["work"],
            f"document_{len(session['images']) + 1:04d}{extension}"
        )

        await message.download(file_name=path)

        session["images"].append((filename, path))

        await message.reply(
            f"✅ Image added: {len(session['images'])}/{MAX_IMAGES}"
        )

    async def finish(_, message):
        await finish_video(bot, message)

    async def cancel(_, message):

        session = sessions.pop(
            message.from_user.id,
            None
        )

        if session:
            shutil.rmtree(
                session["work"],
                ignore_errors=True
            )

            return await message.reply(
                "🛑 Image-to-video session cancelled."
            )

        await message.reply("No active session.")

    async def add_music(_, message):

        session = sessions.get(message.from_user.id)

        if not session:
            return await message.reply(
                "Start with /imagevideo first."
            )

        session["want_music"] = True

        await message.reply(
            "🎵 Send your music file now."
        )

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bot.add_handler(
        MessageHandler(start, filters.command("imagevideo")),
        group=1
    )

    bot.add_handler(
        MessageHandler(finish, filters.command("finishvideo")),
        group=1
    )

    bot.add_handler(
        MessageHandler(cancel, filters.command("cancelvideo")),
        group=1
    )

    bot.add_handler(
        MessageHandler(add_music, filters.command("ivmusic")),
        group=1
    )

    bot.add_handler(
        MessageHandler(receive_photo, filters.photo),
        group=1
    )

    bot.add_handler(
        MessageHandler(receive_document, filters.document),
        group=1
    )
