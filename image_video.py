import os
import re
import shutil
import zipfile
import tempfile
import asyncio
import queue
import time
import logging
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pyrogram import filters
from pyrogram.handlers import MessageHandler


# ---------------- LOGGING ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("image_video")


# ---------------- CONFIGURATION ----------------

MAX_IMAGES = 300
MAX_ZIP_SIZE = 100 * 1024 * 1024
MAX_EXTRACTED_SIZE = 500 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

WIDTH = 1280
HEIGHT = 720
FPS = 15

MIN_IMAGE_DURATION = 0.65
DEFAULT_IMAGE_DURATION = 1.2
MAX_VIDEO_DURATION = 600

AUTO_FINISH_SILENCE = 10.0
QUESTION_TIMEOUT = 30.0
LITE_THRESHOLD = 30

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


def _has_transparency(path):
    try:
        if Path(path).suffix.lower() != ".png":
            return False
        with Image.open(path) as im:
            if im.mode not in ("RGBA", "LA"):
                return False
            alpha = np.array(im.convert("RGBA").split()[-1])
            return float((alpha < 32).mean()) > 0.05
    except Exception as error:
        log.warning("logo detect failed %s: %s", path, error)
        return False


def prepare_image(source, destination):
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


def _analyze_image(path):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return {"brightness": 0.5, "saturation": 0.3, "warmth": 0.0,
                "edge": 0.1, "portrait": False}

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = float(arr.mean())
    max_c = arr.max(axis=-1)
    min_c = arr.min(axis=-1)
    sat = float(np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0).mean())
    warmth = float((r - b).mean())
    gray = arr.mean(axis=-1)
    edge = float(np.abs(np.diff(gray, axis=1)).mean()
                 + np.abs(np.diff(gray, axis=0)).mean())
    return {"brightness": brightness, "saturation": sat,
            "warmth": warmth, "edge": edge, "portrait": im.height > im.width}


def _choose_mood(stats_list):
    if not stats_list:
        return "cinematic"
    avg_bright = sum(s["brightness"] for s in stats_list) / len(stats_list)
    avg_sat = sum(s["saturation"] for s in stats_list) / len(stats_list)
    avg_warm = sum(s["warmth"] for s in stats_list) / len(stats_list)
    avg_edge = sum(s["edge"] for s in stats_list) / len(stats_list)
    portrait_share = sum(1 for s in stats_list if s["portrait"]) / len(stats_list)
    if avg_sat < 0.15:
        return "mono"
    if avg_warm > 0.03 and avg_sat > 0.25 and portrait_share > 0.5:
        return "romantic"
    if avg_edge > 0.09:
        return "clean"
    return "cinematic"


def _effect_for(stats, index, forced=None):
    if forced and forced != "auto":
        return forced
    rotation = ["zoom_in", "zoom_out", "pan_left", "pan_right", "ken_burns"]
    if stats["portrait"]:
        candidates = ["ken_burns", "pulse", "zoom_in", "zoom_out"]
    else:
        candidates = rotation
    if stats["brightness"] < 0.25:
        candidates = ["zoom_in"] + [c for c in candidates if c != "zoom_in"]
    elif stats["brightness"] > 0.8:
        candidates = ["zoom_out"] + [c for c in candidates if c != "zoom_out"]
    return candidates[index % len(candidates)]


def _transition_for(index, prev_stats, next_stats):
    base = ["crossfade", "slide", "circle_reveal", "zoom_blur"]
    if prev_stats and next_stats and prev_stats["edge"] > 0.09 and next_stats["edge"] > 0.09:
        return "crossfade"
    return base[index % len(base)]


# ---------------- DOWNLOAD ----------------

async def safe_download(message, path, retries=2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    last_error = None
    for attempt in range(retries + 1):
        try:
            log.info("download start msg=%s target=%s attempt=%s",
                     getattr(message, "id", "?"), path, attempt + 1)
            result = await message.download(file_name=path)
            if result and os.path.exists(result):
                log.info("download ok msg=%s path=%s size=%s",
                         getattr(message, "id", "?"), result,
                         os.path.getsize(result))
                return result
            last_error = "no file"
        except FileNotFoundError as error:
            last_error = error
            log.warning("download temp vanished msg=%s attempt=%s err=%s",
                        getattr(message, "id", "?"), attempt + 1, error)
        except Exception as error:  # noqa: BLE001
            last_error = error
            log.warning("download error msg=%s attempt=%s err=%s",
                        getattr(message, "id", "?"), attempt + 1, error)
        await asyncio.sleep(0.5 * (attempt + 1))
    log.error("download failed msg=%s path=%s err=%s",
              getattr(message, "id", "?"), path, last_error)
    return None


# ---------------- ZIP ----------------

def extract_zip(zip_path, work):
    log.info("zip extract start path=%s", zip_path)
    if os.path.getsize(zip_path) > MAX_ZIP_SIZE:
        raise ValueError("ZIP file exceeds the 100 MB limit.")
    extracted, total_size = [], 0
    with zipfile.ZipFile(zip_path) as archive:
        members = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            filename = info.filename.replace("\\", "/")
            parts = Path(filename).parts
            if filename.startswith("/") or ".." in parts:
                log.warning("zip skip unsafe path: %s", filename)
                continue
            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            members.append(info)
        members.sort(key=lambda item: natural_sort_key(item.filename))
        log.info("zip members=%s", len(members))
        if len(members) > MAX_IMAGES:
            raise ValueError(f"Maximum {MAX_IMAGES} images allowed.")
        for index, info in enumerate(members):
            total_size += info.file_size
            if total_size > MAX_EXTRACTED_SIZE:
                raise ValueError("Extracted images exceed the 500 MB limit.")
            extension = Path(info.filename).suffix.lower()
            destination = os.path.join(work, f"source_{index:04d}{extension}")
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
            log.info("zip extracted [%s/%s] %s (%s bytes)",
                     index + 1, len(members), info.filename, written)
    if not extracted:
        raise ValueError("The ZIP file contains no supported images.")
    return extracted


# ---------------- PREPARE ----------------

def prepare_images(session):
    work = session["work"]
    sources = []
    if session.get("zip_path"):
        sources.extend(extract_zip(session["zip_path"], work))
    sources.extend(session["images"])
    sources.sort(key=lambda item: natural_sort_key(item[0]))
    log.info("prepare_images sources=%s", len(sources))
    if not sources:
        raise ValueError("No images uploaded.")
    if len(sources) > MAX_IMAGES:
        raise ValueError(f"Maximum {MAX_IMAGES} images allowed.")
    prepared, stats_list, skipped = [], [], 0
    for index, (filename, source) in enumerate(sources):
        destination = os.path.join(work, f"image_{index:04d}.jpg")
        try:
            prepare_image(source, destination)
            prepared.append(destination)
            stats_list.append(_analyze_image(destination))
            log.info("prepared [%s/%s] %s -> %s",
                     index + 1, len(sources), filename, destination)
        except Exception as error:
            skipped += 1
            log.warning("skip invalid image [%s/%s] %s: %s",
                        index + 1, len(sources), filename, error)
    log.info("prepare_images done prepared=%s skipped=%s", len(prepared), skipped)
    if not prepared:
        raise ValueError("No valid images found.")
    return prepared, stats_list


# ---------------- EFFECTS ----------------

def _apply_mood_tint(frame, mood):
    if mood == "romantic":
        b, g, r = cv2.split(frame.astype(np.float32))
        r = np.clip(r * 1.10 + 8, 0, 255)
        b = np.clip(b * 0.95, 0, 255)
        frame = cv2.merge([b, g, r]).astype(np.uint8)
        blurred = cv2.GaussianBlur(frame, (0, 0), 6)
        frame = cv2.addWeighted(frame, 0.85, blurred, 0.15, 0)
    elif mood == "cinematic":
        b, g, r = cv2.split(frame.astype(np.float32))
        r = np.clip(r * 1.05 + 4, 0, 255)
        b = np.clip(b * 1.08 + 6, 0, 255)
        frame = cv2.merge([b, g, r]).astype(np.uint8)
    elif mood == "mono":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


def _apply_vignette(frame, strength=0.35):
    h, w = frame.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = 1.0 - np.clip(r / (0.75 * max(cx, cy)), 0, 1) * strength
    mask = np.dstack([mask] * 3)
    return (frame.astype(np.float32) * mask).astype(np.uint8)


def _apply_particles(frame, mood, frame_index, frame_count, lite):
    if lite or mood not in ("romantic", "cinematic"):
        return frame
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)
    rng = random.Random(1234 + frame_index // 4)
    count = 25 if mood == "romantic" else 18
    for _ in range(count):
        px = rng.randint(0, w - 1)
        py = rng.randint(0, h - 1)
        radius = rng.choice([2, 3, 4])
        color = (180, 190, 255) if mood == "romantic" else (200, 220, 255)
        cv2.circle(overlay, (px, py), radius, color, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), 2)
    return cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)


def _apply_light_leak(frame, frame_index, frame_count, lite):
    if lite:
        return frame
    h, w = frame.shape[:2]
    progress = frame_index / max(1, frame_count - 1)
    band_x = int((progress * 1.4 - 0.2) * w)
    band_w = int(w * 0.25)
    x0 = max(0, band_x - band_w // 2)
    x1 = min(w, band_x + band_w // 2)
    if x1 <= x0:
        return frame
    overlay = frame.copy()
    gradient = np.zeros((h, x1 - x0, 3), dtype=np.uint8)
    gradient[..., 2] = 60
    gradient[..., 1] = 30
    overlay[:, x0:x1] = cv2.add(overlay[:, x0:x1], gradient)
    return cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)


def _crop_zoom(base, zoom, x_off, y_off):
    crop_w = max(2, int(WIDTH / zoom))
    crop_h = max(2, int(HEIGHT / zoom))
    max_x = max(0, WIDTH - crop_w)
    max_y = max(0, HEIGHT - crop_h)
    x = max(0, min(int(max_x * x_off), max_x))
    y = max(0, min(int(max_y * y_off), max_y))
    return base[y:y + crop_h, x:x + crop_w]


def _make_frame(base, effect, frame_index, frame_count, mood, lite,
                logo=None, logo_phase=None):
    progress = frame_index / max(1, frame_count - 1)
    if effect == "zoom_in":
        zoom, x_off, y_off = 1.0 + 0.07 * progress, 0.5, 0.5
    elif effect == "zoom_out":
        zoom, x_off, y_off = 1.07 - 0.07 * progress, 0.5, 0.5
    elif effect == "ken_burns":
        zoom = 1.0 + 0.06 * progress
        x_off = 0.35 + 0.3 * progress
        y_off = 0.4 + 0.2 * progress
    elif effect == "pulse":
        zoom = 1.0 + 0.04 * math.sin(progress * math.pi * 2)
        x_off, y_off = 0.5, 0.5
    elif effect == "pan_left":
        zoom, x_off, y_off = 1.06, 1.0 - progress, 0.5
    elif effect == "pan_right":
        zoom, x_off, y_off = 1.06, progress, 0.5
    elif effect == "pan_up":
        zoom, x_off, y_off = 1.06, 0.5, progress
    elif effect == "pan_down":
        zoom, x_off, y_off = 1.06, 0.5, 1.0 - progress
    else:
        zoom, x_off, y_off = 1.0, 0.5, 0.5
    zoom = max(1.0, zoom)
    cropped = _crop_zoom(base, zoom, x_off, y_off)
    if cropped.size == 0:
        cropped = base
    frame = cv2.resize(cropped, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    fade_frames = max(1, int(FPS * 0.16))
    alpha = min(1.0, (frame_index + 1) / fade_frames,
                1.0, (frame_count - frame_index) / fade_frames)
    if alpha < 1.0:
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

    frame = _apply_mood_tint(frame, mood)
    frame = _apply_vignette(frame, 0.35 if mood != "clean" else 0.15)
    frame = _apply_particles(frame, mood, frame_index, frame_count, lite)
    frame = _apply_light_leak(frame, frame_index, frame_count, lite)

    if logo is not None and logo_phase == "intro":
        frame = _blend_logo(frame, logo, frame_index, frame_count, mode="intro")
    elif logo is not None and logo_phase == "corner":
        frame = _blend_logo(frame, logo, frame_index, frame_count, mode="corner")
    return frame


def _blend_logo(frame, logo, frame_index, frame_count, mode):
    if logo is None:
        return frame
    lh, lw = logo.shape[:2]
    fh, fw = frame.shape[:2]
    if mode == "intro":
        progress = frame_index / max(1, frame_count - 1)
        if progress < 0.15:
            alpha = progress / 0.15
        elif progress < 0.75:
            alpha = 1.0
        else:
            alpha = max(0.0, 1.0 - (progress - 0.75) / 0.25)
        scale = 0.35
    else:
        alpha = 0.55
        scale = 0.14
    target_w = int(fw * scale)
    target_h = max(1, int(lh * (target_w / lw)))
    lw2, lh2 = target_w, target_h
    if lw2 <= 0 or lh2 <= 0 or lw2 >= fw or lh2 >= fh:
        return frame
    resized = cv2.resize(logo, (lw2, lh2), interpolation=cv2.INTER_AREA)
    if mode == "intro":
        x, y = (fw - lw2) // 2, (fh - lh2) // 2
    else:
        margin = int(fw * 0.02)
        x, y = fw - lw2 - margin, fh - lh2 - margin
    roi = frame[y:y + lh2, x:x + lw2]
    if roi.shape[:2] != resized.shape[:2]:
        return frame
    bgr = resized[..., :3].astype(np.float32)
    a = (resized[..., 3:4].astype(np.float32) / 255.0) * alpha
    frame[y:y + lh2, x:x + lw2] = (
        bgr * a + roi.astype(np.float32) * (1 - a)
    ).astype(np.uint8)
    return frame


def _slide_transition(a, b, t, direction=1):
    w = a.shape[1]
    shift = int(w * t) * direction
    out = a.copy()
    if direction > 0:
        out[:, :w - shift] = a[:, shift:]
        out[:, w - shift:] = b[:, :shift]
    else:
        shift = abs(shift)
        out[:, shift:] = a[:, :w - shift]
        out[:, :shift] = b[:, w - shift:]
    return out


def _circle_reveal(a, b, t):
    h, w = a.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = int(math.sqrt((w / 2) ** 2 + (h / 2) ** 2) * t)
    cv2.circle(mask, (w // 2, h // 2), max(1, radius), 255, -1)
    mask3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    return (b.astype(np.float32) * mask3
            + a.astype(np.float32) * (1 - mask3)).astype(np.uint8)


def _zoom_blur_transition(a, b, t):
    strength = 1.0 + 0.2 * math.sin(t * math.pi)
    zoomed = cv2.resize(b, None, fx=strength, fy=strength,
                        interpolation=cv2.INTER_LINEAR)
    y0 = (zoomed.shape[0] - HEIGHT) // 2
    x0 = (zoomed.shape[1] - WIDTH) // 2
    zoomed = zoomed[y0:y0 + HEIGHT, x0:x0 + WIDTH]
    if zoomed.shape[:2] != a.shape[:2]:
        zoomed = cv2.resize(zoomed, (WIDTH, HEIGHT))
    return cv2.addWeighted(a, 1 - t, zoomed, t, 0)


def _apply_transition(a, b, name, t):
    if name == "slide":
        return _slide_transition(a, b, t)
    if name == "circle_reveal":
        return _circle_reveal(a, b, t)
    if name == "zoom_blur":
        return _zoom_blur_transition(a, b, t)
    return cv2.addWeighted(a, 1 - t, b, t, 0)


# ---------------- RENDER ----------------

def render_video(images, stats_list, output_path, total_seconds,
                 mood="cinematic", lite=False, logo_path=None,
                 forced_effect=None, progress_callback=None):
    count = len(images)
    log.info("render start images=%s target=%.1fs mood=%s lite=%s effect=%s logo=%s",
             count, total_seconds, mood, lite, forced_effect, bool(logo_path))
    if count == 0:
        raise ValueError("No images available to render.")
    per_image = max(MIN_IMAGE_DURATION, float(total_seconds) / count)
    frames_per_image = max(2, round(per_image * FPS))
    actual_duration = (frames_per_image * count) / FPS
    log.info("render plan per=%.2fs frames=%s actual=%.1fs",
             per_image, frames_per_image, actual_duration)
    if actual_duration > MAX_VIDEO_DURATION:
        raise ValueError("Video exceeds the 10-minute limit.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("OpenCV could not open MP4 writer.")

    logo = None
    if logo_path and os.path.exists(logo_path):
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is not None and (logo.ndim < 3 or logo.shape[2] != 4):
            log.warning("logo missing alpha; ignoring logo")
            logo = None

    transition_frames = max(2, int(FPS * 0.25))
    total_frames = frames_per_image * count
    completed = 0
    last_pct = -1

    try:
        prev_last_frame = None
        prev_stats = None
        for i, image_path in enumerate(images):
            base = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if base is None:
                log.warning("render skip unreadable %s", image_path)
                continue
            if base.shape[1] != WIDTH or base.shape[0] != HEIGHT:
                base = cv2.resize(base, (WIDTH, HEIGHT),
                                  interpolation=cv2.INTER_AREA)
            stats = stats_list[i] if i < len(stats_list) else {
                "brightness": 0.5, "saturation": 0.3, "warmth": 0.0,
                "edge": 0.1, "portrait": False}
            effect = _effect_for(stats, i, forced_effect)
            transition = _transition_for(i, prev_stats, stats)
            log.info("render image [%s/%s] effect=%s transition=%s",
                     i + 1, count, effect, transition)

            frames_buffer = []
            for fi in range(frames_per_image):
                if logo is not None and i == 0:
                    logo_phase = "intro"
                elif logo is not None:
                    logo_phase = "corner"
                else:
                    logo_phase = None
                frames_buffer.append(_make_frame(
                    base, effect, fi, frames_per_image, mood, lite,
                    logo, logo_phase))

            if prev_last_frame is not None and len(frames_buffer) >= transition_frames:
                for ti in range(transition_frames):
                    t = (ti + 1) / (transition_frames + 1)
                    blended = _apply_transition(
                        prev_last_frame, frames_buffer[ti], transition, t)
                    writer.write(blended)
                    completed += 1
                start_idx = transition_frames
            else:
                start_idx = 0
            for frame in frames_buffer[start_idx:]:
                writer.write(frame)
                completed += 1
            prev_last_frame = frames_buffer[-1] if frames_buffer else None
            prev_stats = stats

            if progress_callback:
                pct = int((completed / max(1, total_frames)) * 100)
                if pct >= last_pct + 5 or pct == 100:
                    last_pct = pct
                    log.info("render progress %s%% (%s/%s)",
                             pct, completed, total_frames)
                    progress_callback(pct, i + 1, count)
            del base
    finally:
        writer.release()

    size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    log.info("render done path=%s size=%s duration=%.1fs",
             output_path, size, actual_duration)
    if size < 1024:
        raise RuntimeError("Video rendering failed: output file is empty.")
    return actual_duration


# ---------------- SESSION HELPERS ----------------

def _new_session(user_id, chat_id):
    os.makedirs(WORK_DIR, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"{user_id}_", dir=WORK_DIR)
    return {
        "work": work,
        "images": [],
        "zip_path": None,
        "logo_path": None,
        "processing": False,
        "lock": asyncio.Lock(),
        "last_activity": time.monotonic(),
        "chat_id": chat_id,
        "state": "uploading",  # uploading -> q_duration -> q_mood -> q_effect -> rendering
        "chosen_duration": None,
        "chosen_mood": None,
        "chosen_effect": None,
        "ask_started": 0.0,
        "question_msg": None,
        "invalid_count": 0,
        "watcher": None,
        "flow": None,
    }


def _bump(session):
    session["last_activity"] = time.monotonic()


async def _ensure_session(bot, message):
    user_id = message.from_user.id
    session = sessions.get(user_id)
    if session:
        session["chat_id"] = message.chat.id
        return session
    session = _new_session(user_id, message.chat.id)
    sessions[user_id] = session
    log.info("session created user=%s work=%s", user_id, session["work"])
    session["watcher"] = asyncio.create_task(_auto_finish_watcher(bot, user_id))
    return session


def _default_duration(count):
    if count >= 40:
        return 1.0
    if count <= 5:
        return 1.6
    return DEFAULT_IMAGE_DURATION


# ---------------- AUTO FINISH / QUESTION FLOW ----------------

async def _auto_finish_watcher(bot, user_id):
    """Wait for silence, then start the interactive question flow."""
    try:
        while True:
            session = sessions.get(user_id)
            if not session:
                return
            if session["state"] != "uploading":
                return
            if session.get("processing"):
                return
            lock = session.get("lock")
            if lock and lock.locked():
                _bump(session)
                await asyncio.sleep(1.0)
                continue
            idle = time.monotonic() - session["last_activity"]
            if idle >= AUTO_FINISH_SILENCE and (session["images"] or session.get("zip_path")):
                log.info("auto-finish trigger user=%s idle=%.1fs images=%s",
                         user_id, idle, len(session["images"]))
                await _start_question_flow(bot, user_id)
                return
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        return
    except Exception as error:
        log.exception("watcher error user=%s: %s", user_id, error)


async def _start_question_flow(bot, user_id):
    session = sessions.get(user_id)
    if not session or session["state"] != "uploading":
        return
    if not (session["images"] or session.get("zip_path")):
        return

    # Basic count (ZIP contents resolved later; use uploaded count for prompt).
    count = len(session["images"]) if session["images"] else 1
    session["state"] = "q_duration"
    session["ask_started"] = time.monotonic()
    session["invalid_count"] = 0

    default_dur = _default_duration(count)
    prompt = (
        f"✅ {count} image(s) saved.\n\n"
        "❓ **Q1 / 3 — Image duration**\n\n"
        "How long should each image show?\n"
        "Reply a number in seconds (e.g. `1.5`),\n"
        "or `auto` for default,\n"
        "or `cancel` to abort.\n\n"
        f"⏱️ Auto-skip in {int(QUESTION_TIMEOUT)}s → {default_dur:.2f}s"
    )
    try:
        session["question_msg"] = await bot.send_message(session["chat_id"], prompt)
    except Exception as error:
        log.warning("send q1 failed: %s", error)

    session["flow"] = asyncio.create_task(_question_timeout(bot, user_id))


async def _question_timeout(bot, user_id):
    """Auto-skip to the next question when the user doesn't answer."""
    try:
        while True:
            session = sessions.get(user_id)
            if not session or session["state"] in ("rendering", "uploading", "done"):
                return
            idle = time.monotonic() - session["ask_started"]
            if idle >= QUESTION_TIMEOUT:
                log.info("question timeout user=%s state=%s",
                         user_id, session["state"])
                await _advance_question(bot, user_id, use_default=True)
                return
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        return


async def _advance_question(bot, user_id, use_default=False):
    session = sessions.get(user_id)
    if not session:
        return
    state = session["state"]

    if state == "q_duration":
        if use_default or session["chosen_duration"] is None:
            count = len(session["images"]) if session["images"] else 1
            session["chosen_duration"] = _default_duration(count)
            log.info("duration default used user=%s val=%.2f",
                     user_id, session["chosen_duration"])
        session["state"] = "q_mood"
        session["ask_started"] = time.monotonic()
        session["invalid_count"] = 0
        try:
            session["question_msg"] = await bot.send_message(
                session["chat_id"],
                "❓ **Q2 / 3 — Mood**\n\n"
                "Reply one of:\n"
                "`romantic` · `cinematic` · `mono` · `clean` · `auto`\n\n"
                f"⏱️ Auto-skip in {int(QUESTION_TIMEOUT)}s → auto-detect",
            )
        except Exception as error:
            log.warning("send q2 failed: %s", error)
        session["flow"] = asyncio.create_task(_question_timeout(bot, user_id))
        return

    if state == "q_mood":
        if session["chosen_mood"] is None:
            session["chosen_mood"] = "auto"
            log.info("mood default=auto user=%s", user_id)
        session["state"] = "q_effect"
        session["ask_started"] = time.monotonic()
        session["invalid_count"] = 0
        try:
            session["question_msg"] = await bot.send_message(
                session["chat_id"],
                "❓ **Q3 / 3 — Effect**\n\n"
                "Reply one of:\n"
                "`zoom_in` · `zoom_out` · `ken_burns` · `pulse` ·\n"
                "`pan_left` · `pan_right` · `auto`\n\n"
                f"⏱️ Auto-skip in {int(QUESTION_TIMEOUT)}s → auto",
            )
        except Exception as error:
            log.warning("send q3 failed: %s", error)
        session["flow"] = asyncio.create_task(_question_timeout(bot, user_id))
        return

    if state == "q_effect":
        if session["chosen_effect"] is None:
            session["chosen_effect"] = "auto"
            log.info("effect default=auto user=%s", user_id)
        await _start_render(bot, user_id)


# ---------------- RENDER FLOW ----------------

async def _start_render(bot, user_id):
    session = sessions.get(user_id)
    if not session or session.get("processing"):
        return
    if not (session["images"] or session.get("zip_path")):
        return
    session["state"] = "rendering"
    lock = session["lock"]
    async with lock:
        session["processing"] = True

    chat_id = session["chat_id"]
    status = None
    output = None
    try:
        status = await bot.send_message(chat_id, "🖼️ Preparing images...")
        images, stats_list = await asyncio.to_thread(prepare_images, session)
        count = len(images)

        chosen_duration = float(session["chosen_duration"] or _default_duration(count))
        total_requested = chosen_duration * count
        capped = False
        if total_requested > MAX_VIDEO_DURATION:
            chosen_duration = max(MIN_IMAGE_DURATION, MAX_VIDEO_DURATION / count)
            total_requested = chosen_duration * count
            capped = True
            log.info("duration capped user=%s new=%.2fs", user_id, chosen_duration)

        mood = session["chosen_mood"]
        if not mood or mood == "auto":
            mood = _choose_mood(stats_list)
        effect = session["chosen_effect"] or "auto"

        lite = count >= LITE_THRESHOLD

        summary = (
            "✅ **Settings locked**\n\n"
            f"📸 Images: {count}\n"
            f"⏱️ Image duration: {chosen_duration:.2f}s "
            f"(total ~{int(total_requested)}s)\n"
            f"🎨 Mood: {mood}\n"
            f"✨ Effect: {effect}\n"
            f"⚡ Lite mode: {'on' if lite else 'off'}"
        )
        if capped:
            summary += "\n\n⚠️ Total capped to 600s (Telegram limit)."
        summary += "\n\n🎬 Rendering..."
        try:
            await status.edit_text(summary)
        except Exception:
            pass

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output = os.path.join(OUTPUT_DIR, f"image_video_{user_id}.mp4")

        loop = asyncio.get_running_loop()
        progress_q = queue.Queue()

        def report(pct, cur, total):
            loop.call_soon_threadsafe(progress_q.put_nowait, (pct, cur, total))

        render_task = asyncio.create_task(asyncio.to_thread(
            render_video, images, stats_list, output, total_requested,
            mood, lite, session.get("logo_path"),
            None if effect == "auto" else effect, report))

        last_edit = 0
        last_pct = -1
        while not render_task.done():
            try:
                pct, cur, total = await asyncio.wait_for(
                    asyncio.to_thread(progress_q.get, True, 2.0), timeout=2.5)
                now = time.monotonic()
                if pct != last_pct and (now - last_edit >= 1.5 or pct == 100):
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    try:
                        await status.edit_text(
                            "🎬 Rendering...\n\n"
                            f"[{bar}] {pct}%\n"
                            f"📸 Image: {cur}/{total}\n"
                            f"🎨 Mood: {mood} · ✨ {effect}")
                        last_pct, last_edit = pct, now
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                continue
            except queue.Empty:
                continue

        duration = await render_task

        await bot.send_video(
            chat_id, output,
            caption=(
                "🎬 **Your video is ready!**\n\n"
                f"📸 Images: {count}\n"
                f"⏱️ Duration: {int(duration)}s\n"
                f"🎨 Mood: {mood}\n"
                f"✨ Effect: {effect}\n"
                f"🎞️ {WIDTH}×{HEIGHT} · {FPS} FPS"),
            supports_streaming=True,
        )
        try:
            await status.edit_text("✅ Video created successfully!")
        except Exception:
            pass
        log.info("render success user=%s images=%s duration=%.1fs",
                 user_id, count, duration)

    except Exception as error:
        log.exception("render failed user=%s: %s", user_id, error)
        try:
            if status:
                await status.edit_text(f"❌ Render failed: {str(error)[:800]}")
        except Exception:
            pass
    finally:
        session = sessions.pop(user_id, None)
        if session:
            for task_key in ("watcher", "flow"):
                t = session.get(task_key)
                if t and not t.done():
                    t.cancel()
            work = session.get("work")
            if work and os.path.isdir(work):
                shutil.rmtree(work, ignore_errors=True)
        if output and os.path.exists(output):
            try:
                os.remove(output)
            except OSError:
                pass


# ---------------- MESSAGE HANDLERS ----------------

def register_image_video_handlers(bot):

    async def on_photo(_, message):
        if not message.from_user:
            return
        session = await _ensure_session(bot, message)
        if session["state"] != "uploading":
            # Extra images during Q&A: keep answers, add to list.
            if session["state"] in ("q_duration", "q_mood", "q_effect"):
                pass
            else:
                return
        _bump(session)
        async with session["lock"]:
            if len(session["images"]) >= MAX_IMAGES:
                return await message.reply(f"Maximum {MAX_IMAGES} images reached.")
            filename = f"photo_{message.id}.jpg"
            path = os.path.join(session["work"], filename)
            result = await safe_download(message, path)
            if not result:
                return await message.reply("⚠️ Could not download, please resend.")
            session["images"].append((filename, result))
            _bump(session)
            log.info("photo stored user=%s total=%s",
                     message.from_user.id, len(session["images"]))
            await message.reply(
                f"📥 {len(session['images'])} image(s) received "
                f"(questions start after {int(AUTO_FINISH_SILENCE)}s silence)")

    async def on_document(_, message):
        if not message.from_user or not message.document:
            return
        session = await _ensure_session(bot, message)
        if session["state"] not in ("uploading", "q_duration", "q_mood", "q_effect"):
            return
        _bump(session)
        filename = message.document.file_name or "upload"
        extension = Path(filename).suffix.lower()
        file_size = message.document.file_size or 0
        async with session["lock"]:
            if extension in AUDIO_EXTENSIONS:
                return await message.reply(
                    "🎵 Background music is disabled in this build.")
            if extension == ".zip":
                if session["images"] or session.get("zip_path"):
                    return await message.reply("Use one ZIP OR individual images.")
                if file_size > MAX_ZIP_SIZE:
                    return await message.reply("ZIP limit is 100 MB.")
                path = os.path.join(session["work"], f"images_{message.id}.zip")
                result = await safe_download(message, path)
                if not result:
                    return await message.reply("⚠️ ZIP download failed.")
                session["zip_path"] = result
                _bump(session)
                return await message.reply(
                    f"📦 ZIP received (questions start after "
                    f"{int(AUTO_FINISH_SILENCE)}s silence)")
            if extension not in IMAGE_EXTENSIONS:
                return
            path = os.path.join(session["work"], f"document_{message.id}{extension}")
            result = await safe_download(message, path)
            if not result:
                return await message.reply("⚠️ Download failed, please resend.")
            if session.get("logo_path") is None and _has_transparency(result):
                session["logo_path"] = result
                _bump(session)
                return await message.reply("🖼️ Logo detected and saved.")
            if len(session["images"]) >= MAX_IMAGES:
                return await message.reply(f"Maximum {MAX_IMAGES} images reached.")
            session["images"].append((filename, result))
            _bump(session)
            await message.reply(
                f"📥 {len(session['images'])} image(s) received "
                f"(questions start after {int(AUTO_FINISH_SILENCE)}s silence)")

    async def on_text(_, message):
        if not message.from_user or not message.text:
            return
        user_id = message.from_user.id
        session = sessions.get(user_id)
        if not session or session.get("processing"):
            return

        text = message.text.strip().lower()
        _bump(session)
        state = session["state"]

        # Global cancel
        if text in {"cancel", "stop", "x", "abort"}:
            for task_key in ("watcher", "flow"):
                t = session.get(task_key)
                if t and not t.done():
                    t.cancel()
            sessions.pop(user_id, None)
            work = session.get("work")
            if work and os.path.isdir(work):
                shutil.rmtree(work, ignore_errors=True)
            log.info("user cancel user=%s", user_id)
            return await message.reply("🛑 Cancelled.")

        if state == "uploading":
            if text in {"finish", "done", "go", "render"}:
                return await _start_question_flow(bot, user_id)
            return  # ignore other text while uploading

        if state == "q_duration":
            if text == "auto":
                session["chosen_duration"] = None
                await _answer_received(message, "⏱️ Duration: auto")
                return await _advance_question(bot, user_id, use_default=True)
            try:
                value = float(text)
                if value < MIN_IMAGE_DURATION or value > 30:
                    raise ValueError
                session["chosen_duration"] = value
                await _answer_received(
                    message, f"⏱️ Duration: {value:.2f}s per image")
                return await _advance_question(bot, user_id)
            except ValueError:
                session["invalid_count"] += 1
                if session["invalid_count"] >= 2:
                    session["chosen_duration"] = None
                    await message.reply(
                        "⚠️ Invalid twice → using auto duration.")
                    return await _advance_question(bot, user_id, use_default=True)
                return await message.reply(
                    f"⚠️ Invalid. Reply a number "
                    f"({MIN_IMAGE_DURATION}–30 seconds) or `auto`.")

        if state == "q_mood":
            moods = {"romantic", "cinematic", "mono", "bw", "clean", "auto"}
            if text in moods:
                session["chosen_mood"] = "mono" if text == "bw" else text
                await _answer_received(
                    message, f"🎨 Mood: {session['chosen_mood']}")
                return await _advance_question(bot, user_id)
            session["invalid_count"] += 1
            if session["invalid_count"] >= 2:
                session["chosen_mood"] = "auto"
                await message.reply("⚠️ Invalid twice → using auto mood.")
                return await _advance_question(bot, user_id)
            return await message.reply(
                "⚠️ Invalid. Reply one of:\n"
                "`romantic` · `cinematic` · `mono` · `clean` · `auto`")

        if state == "q_effect":
            effects = {"zoom_in", "zoom_out", "ken_burns", "pulse",
                       "pan_left", "pan_right", "auto"}
            if text in effects:
                session["chosen_effect"] = text
                await _answer_received(message, f"✨ Effect: {text}")
                return await _advance_question(bot, user_id)
            session["invalid_count"] += 1
            if session["invalid_count"] >= 2:
                session["chosen_effect"] = "auto"
                await message.reply("⚠️ Invalid twice → using auto effect.")
                return await _advance_question(bot, user_id)
            return await message.reply(
                "⚠️ Invalid. Reply one of:\n"
                "`zoom_in` · `zoom_out` · `ken_burns` · `pulse` · "
                "`pan_left` · `pan_right` · `auto`")

    async def _answer_received(message, text):
        try:
            await message.reply(text)
        except Exception:
            pass

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bot.add_handler(MessageHandler(on_photo, filters.photo), group=1)
    bot.add_handler(MessageHandler(on_document, filters.document), group=1)
    bot.add_handler(MessageHandler(on_text, filters.text), group=1)
    log.info("image_video handlers registered")
