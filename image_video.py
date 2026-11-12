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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("image_video")


# ---------------- CONFIG ----------------

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
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", str(value))]


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
            raise ValueError("Image resolution too large.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        target = WIDTH / HEIGHT
        current = image.width / image.height
        if current > target:
            nw = max(1, int(image.height * target))
            left = (image.width - nw) // 2
            image = image.crop((left, 0, left + nw, image.height))
        else:
            nh = max(1, int(image.width / target))
            top = (image.height - nh) // 2
            image = image.crop((0, top, image.width, top + nh))
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
    max_c, min_c = arr.max(axis=-1), arr.min(axis=-1)
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
    n = len(stats_list)
    avg_sat = sum(s["saturation"] for s in stats_list) / n
    avg_warm = sum(s["warmth"] for s in stats_list) / n
    avg_edge = sum(s["edge"] for s in stats_list) / n
    portrait = sum(1 for s in stats_list if s["portrait"]) / n
    if avg_sat < 0.15:
        return "mono"
    if avg_warm > 0.03 and avg_sat > 0.25 and portrait > 0.5:
        return "romantic"
    if avg_edge > 0.09:
        return "clean"
    return "cinematic"


def _effect_for(stats, index, forced=None):
    if forced and forced != "auto":
        return forced
    base = ["zoom_in", "zoom_out", "pan_left", "pan_right", "ken_burns"]
    if stats["portrait"]:
        candidates = ["ken_burns", "pulse", "zoom_in", "zoom_out"]
    else:
        candidates = base
    if stats["brightness"] < 0.25:
        candidates = ["zoom_in"] + [c for c in candidates if c != "zoom_in"]
    elif stats["brightness"] > 0.8:
        candidates = ["zoom_out"] + [c for c in candidates if c != "zoom_out"]
    return candidates[index % len(candidates)]


def _transition_for(index, prev, nxt):
    base = ["crossfade", "slide", "circle_reveal", "zoom_blur"]
    if prev and nxt and prev["edge"] > 0.09 and nxt["edge"] > 0.09:
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

def count_zip_images(zip_path):
    """Count valid image entries inside a ZIP without extracting."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            count = 0
            for info in archive.infolist():
                if info.is_dir():
                    continue
                filename = info.filename.replace("\\", "/")
                parts = Path(filename).parts
                if filename.startswith("/") or ".." in parts:
                    continue
                if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                count += 1
            return count
    except Exception as error:
        log.warning("zip count failed %s: %s", zip_path, error)
        return 0


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
        members.sort(key=lambda i: natural_sort_key(i.filename))
        log.info("zip members=%s", len(members))
        if len(members) > MAX_IMAGES:
            raise ValueError(f"Maximum {MAX_IMAGES} images allowed.")
        for index, info in enumerate(members):
            total_size += info.file_size
            if total_size > MAX_EXTRACTED_SIZE:
                raise ValueError("Extracted images exceed the 500 MB limit.")
            extension = Path(info.filename).suffix.lower()
            dest = os.path.join(work, f"source_{index:04d}{extension}")
            written = 0
            with archive.open(info) as source, open(dest, "wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.file_size or written > MAX_EXTRACTED_SIZE:
                        raise ValueError("Invalid or oversized ZIP entry.")
                    output.write(chunk)
            extracted.append((info.filename, dest))
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
    sources.sort(key=lambda i: natural_sort_key(i[0]))
    log.info("prepare_images sources=%s", len(sources))
    if not sources:
        raise ValueError("No images uploaded.")
    if len(sources) > MAX_IMAGES:
        raise ValueError(f"Maximum {MAX_IMAGES} images allowed.")
    prepared, stats_list, skipped = [], [], 0
    for index, (filename, source) in enumerate(sources):
        dest = os.path.join(work, f"image_{index:04d}.jpg")
        try:
            prepare_image(source, dest)
            prepared.append(dest)
            stats_list.append(_analyze_image(dest))
            log.info("prepared [%s/%s] %s -> %s",
                     index + 1, len(sources), filename, dest)
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
    r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    mask = 1.0 - np.clip(r / (0.75 * max(w / 2, h / 2)), 0, 1) * strength
    mask = np.dstack([mask] * 3)
    return (frame.astype(np.float32) * mask).astype(np.uint8)


def _apply_particles(frame, mood, idx, count, lite):
    if lite or mood not in ("romantic", "cinematic"):
        return frame
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)
    rng = random.Random(1234 + idx // 4)
    n = 25 if mood == "romantic" else 18
    for _ in range(n):
        px, py = rng.randint(0, w - 1), rng.randint(0, h - 1)
        radius = rng.choice([2, 3, 4])
        color = (180, 190, 255) if mood == "romantic" else (200, 220, 255)
        cv2.circle(overlay, (px, py), radius, color, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), 2)
    return cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)


def _apply_light_leak(frame, idx, count, lite):
    if lite:
        return frame
    h, w = frame.shape[:2]
    progress = idx / max(1, count - 1)
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
    cw = max(2, int(WIDTH / zoom))
    ch = max(2, int(HEIGHT / zoom))
    mx, my = max(0, WIDTH - cw), max(0, HEIGHT - ch)
    x = max(0, min(int(mx * x_off), mx))
    y = max(0, min(int(my * y_off), my))
    return base[y:y + ch, x:x + cw]


def _make_frame(base, effect, fi, fc, mood, lite, logo=None, phase=None):
    progress = fi / max(1, fc - 1)
    if effect == "zoom_in":
        zoom, xo, yo = 1.0 + 0.07 * progress, 0.5, 0.5
    elif effect == "zoom_out":
        zoom, xo, yo = 1.07 - 0.07 * progress, 0.5, 0.5
    elif effect == "ken_burns":
        zoom, xo, yo = 1.0 + 0.06 * progress, 0.35 + 0.3 * progress, 0.4 + 0.2 * progress
    elif effect == "pulse":
        zoom = 1.0 + 0.04 * math.sin(progress * math.pi * 2)
        xo, yo = 0.5, 0.5
    elif effect == "pan_left":
        zoom, xo, yo = 1.06, 1.0 - progress, 0.5
    elif effect == "pan_right":
        zoom, xo, yo = 1.06, progress, 0.5
    elif effect == "pan_up":
        zoom, xo, yo = 1.06, 0.5, progress
    elif effect == "pan_down":
        zoom, xo, yo = 1.06, 0.5, 1.0 - progress
    else:
        zoom, xo, yo = 1.0, 0.5, 0.5
    zoom = max(1.0, zoom)
    cropped = _crop_zoom(base, zoom, xo, yo)
    if cropped.size == 0:
        cropped = base
    frame = cv2.resize(cropped, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    fade_n = max(1, int(FPS * 0.16))
    alpha = min(1.0, (fi + 1) / fade_n, 1.0, (fc - fi) / fade_n)
    if alpha < 1.0:
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

    frame = _apply_mood_tint(frame, mood)
    frame = _apply_vignette(frame, 0.35 if mood != "clean" else 0.15)
    frame = _apply_particles(frame, mood, fi, fc, lite)
    frame = _apply_light_leak(frame, fi, fc, lite)

    if logo is not None and phase == "intro":
        frame = _blend_logo(frame, logo, fi, fc, "intro")
    elif logo is not None and phase == "corner":
        frame = _blend_logo(frame, logo, fi, fc, "corner")
    return frame


def _blend_logo(frame, logo, fi, fc, mode):
    lh, lw = logo.shape[:2]
    fh, fw = frame.shape[:2]
    if mode == "intro":
        progress = fi / max(1, fc - 1)
        if progress < 0.15:
            alpha = progress / 0.15
        elif progress < 0.75:
            alpha = 1.0
        else:
            alpha = max(0.0, 1.0 - (progress - 0.75) / 0.25)
        scale = 0.35
    else:
        alpha, scale = 0.55, 0.14
    tw = int(fw * scale)
    th = max(1, int(lh * (tw / lw)))
    if tw <= 0 or th <= 0 or tw >= fw or th >= fh:
        return frame
    resized = cv2.resize(logo, (tw, th), interpolation=cv2.INTER_AREA)
    if mode == "intro":
        x, y = (fw - tw) // 2, (fh - th) // 2
    else:
        m = int(fw * 0.02)
        x, y = fw - tw - m, fh - th - m
    roi = frame[y:y + th, x:x + tw]
    if roi.shape[:2] != resized.shape[:2]:
        return frame
    bgr = resized[..., :3].astype(np.float32)
    a = (resized[..., 3:4].astype(np.float32) / 255.0) * alpha
    frame[y:y + th, x:x + tw] = (bgr * a + roi.astype(np.float32) * (1 - a)).astype(np.uint8)
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
    m3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    return (b.astype(np.float32) * m3 + a.astype(np.float32) * (1 - m3)).astype(np.uint8)


def _zoom_blur_transition(a, b, t):
    strength = 1.0 + 0.2 * math.sin(t * math.pi)
    zoomed = cv2.resize(b, None, fx=strength, fy=strength, interpolation=cv2.INTER_LINEAR)
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
    per = max(MIN_IMAGE_DURATION, float(total_seconds) / count)
    fpi = max(2, round(per * FPS))
    actual = (fpi * count) / FPS
    log.info("render plan per=%.2fs fpi=%s actual=%.1fs", per, fpi, actual)
    if actual > MAX_VIDEO_DURATION:
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
            log.warning("logo missing alpha; ignoring")
            logo = None
        elif logo is not None:
            log.info("logo loaded shape=%s", logo.shape)

    trans_frames = max(2, int(FPS * 0.25))
    total_frames = fpi * count
    completed = 0
    last_pct = -1

    try:
        prev_last, prev_stats = None, None
        for i, path in enumerate(images):
            base = cv2.imread(path, cv2.IMREAD_COLOR)
            if base is None:
                log.warning("render skip unreadable %s", path)
                continue
            if base.shape[1] != WIDTH or base.shape[0] != HEIGHT:
                base = cv2.resize(base, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            stats = stats_list[i] if i < len(stats_list) else {
                "brightness": 0.5, "saturation": 0.3, "warmth": 0.0,
                "edge": 0.1, "portrait": False}
            effect = _effect_for(stats, i, forced_effect)
            transition = _transition_for(i, prev_stats, stats)
            log.info("render image [%s/%s] effect=%s transition=%s",
                     i + 1, count, effect, transition)

            buf = []
            for fi in range(fpi):
                if logo is not None:
                    phase = "intro" if i == 0 else "corner"
                else:
                    phase = None
                buf.append(_make_frame(base, effect, fi, fpi, mood, lite, logo, phase))

            if prev_last is not None and len(buf) >= trans_frames:
                for ti in range(trans_frames):
                    t = (ti + 1) / (trans_frames + 1)
                    writer.write(_apply_transition(prev_last, buf[ti], transition, t))
                    completed += 1
                start = trans_frames
            else:
                start = 0
            for frame in buf[start:]:
                writer.write(frame)
                completed += 1
            prev_last = buf[-1] if buf else None
            prev_stats = stats

            if progress_callback:
                pct = int((completed / max(1, total_frames)) * 100)
                if pct >= last_pct + 5 or pct == 100:
                    last_pct = pct
                    log.info("render progress %s%% (%s/%s)", pct, completed, total_frames)
                    progress_callback(pct, i + 1, count)
            del base
    finally:
        writer.release()

    size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    log.info("render done path=%s size=%s duration=%.1fs", output_path, size, actual)
    if size < 1024:
        raise RuntimeError("Video rendering failed: output file is empty.")
    return actual


# ---------------- SESSION ----------------

def _new_session(user_id, chat_id):
    os.makedirs(WORK_DIR, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"{user_id}_", dir=WORK_DIR)
    return {
        "work": work, "images": [], "zip_path": None, "logo_path": None,
        "processing": False, "lock": asyncio.Lock(),
        "last_activity": time.monotonic(), "chat_id": chat_id,
        "state": "uploading", "chosen_duration": None,
        "chosen_mood": None, "chosen_effect": None,
        "ask_started": 0.0, "question_msg": None, "invalid_count": 0,
        "watcher": None, "flow": None, "zip_image_count": None,
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


def _effective_image_count(session):
    """Count of images to be rendered: ZIP entries + direct uploads."""
    total = len(session["images"])
    if session.get("zip_path"):
        if session.get("zip_image_count") is None:
            session["zip_image_count"] = count_zip_images(session["zip_path"])
        total += session["zip_image_count"]
    return total


# ---------------- AUTO FINISH / QUESTION FLOW ----------------

async def _auto_finish_watcher(bot, user_id):
    log.info("watcher started user=%s", user_id)
    try:
        while True:
            session = sessions.get(user_id)
            if not session:
                log.info("watcher exit user=%s (no session)", user_id)
                return
            if session["state"] != "uploading":
                log.info("watcher exit user=%s (state=%s)", user_id, session["state"])
                return
            if session.get("processing"):
                log.info("watcher exit user=%s (processing)", user_id)
                return
            lock = session.get("lock")
            if lock and lock.locked():
                _bump(session)
                await asyncio.sleep(1.0)
                continue
            idle = time.monotonic() - session["last_activity"]
            has_any = bool(session["images"]) or bool(session.get("zip_path"))
            if idle >= AUTO_FINISH_SILENCE and has_any:
                log.info("auto-finish trigger user=%s idle=%.1fs images=%s zip=%s",
                         user_id, idle, len(session["images"]),
                         bool(session.get("zip_path")))
                await _start_question_flow(bot, user_id)
                return
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        log.info("watcher cancelled user=%s", user_id)
        return
    except Exception as error:
        log.exception("watcher error user=%s: %s", user_id, error)


async def _start_question_flow(bot, user_id):
    session = sessions.get(user_id)
    if not session:
        log.warning("_start_question_flow: no session user=%s", user_id)
        return
    if session["state"] != "uploading":
        log.warning("_start_question_flow: bad state user=%s state=%s",
                    user_id, session["state"])
        return
    if not (session["images"] or session.get("zip_path")):
        log.warning("_start_question_flow: nothing to render user=%s", user_id)
        return

    count = _effective_image_count(session)
    log.info("_start_question_flow user=%s count=%s images=%s zip_count=%s",
             user_id, count, len(session["images"]), session.get("zip_image_count"))

    if count == 0:
        log.error("_start_question_flow: effective count is 0 user=%s", user_id)
        try:
            await bot.send_message(
                session["chat_id"],
                "⚠️ The ZIP file contains no supported images. "
                "Please upload JPG/PNG/WEBP/BMP and try again."
            )
        except Exception:
            pass
        sessions.pop(user_id, None)
        return

    session["state"] = "q_duration"
    session["ask_started"] = time.monotonic()
    session["invalid_count"] = 0

    default_dur = _default_duration(count)
    prompt = (
        f"✅ {count} image(s) detected.\n\n"
        "❓ **Q1 / 3 — Image duration**\n\n"
        "How long should each image show?\n"
        "Reply a number in seconds (e.g. `1.5`),\n"
        "or `auto` for default,\n"
        "or `cancel` to abort.\n\n"
        f"⏱️ Auto-skip in {int(QUESTION_TIMEOUT)}s → {default_dur:.2f}s"
    )
    try:
        session["question_msg"] = await bot.send_message(session["chat_id"], prompt)
        log.info("Q1 sent user=%s msg_id=%s",
                 user_id, getattr(session["question_msg"], "id", "?"))
    except Exception as error:
        log.exception("send Q1 failed user=%s: %s", user_id, error)
        sessions.pop(user_id, None)
        return

    session["flow"] = asyncio.create_task(_question_timeout(bot, user_id))


async def _question_timeout(bot, user_id):
    log.info("question timer started user=%s", user_id)
    try:
        while True:
            session = sessions.get(user_id)
            if not session or session["state"] in ("rendering", "uploading", "done"):
                log.info("question timer exit user=%s", user_id)
                return
            idle = time.monotonic() - session["ask_started"]
            if idle >= QUESTION_TIMEOUT:
                log.info("question timeout user=%s state=%s idle=%.1fs",
                         user_id, session["state"], idle)
                await _advance_question(bot, user_id, use_default=True)
                return
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        log.info("question timer cancelled user=%s", user_id)
        return
    except Exception as error:
        log.exception("question timer error user=%s: %s", user_id, error)


async def _advance_question(bot, user_id, use_default=False):
    session = sessions.get(user_id)
    if not session:
        log.warning("_advance_question: no session user=%s", user_id)
        return
    state = session["state"]
    log.info("_advance_question user=%s from=%s use_default=%s",
             user_id, state, use_default)

    if state == "q_duration":
        if use_default or session["chosen_duration"] is None:
            count = _effective_image_count(session)
            session["chosen_duration"] = _default_duration(max(1, count))
            log.info("duration default user=%s val=%.2f",
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
                f"⏱️ Auto-skip in {int(QUESTION_TIMEOUT)}s → auto",
            )
            log.info("Q2 sent user=%s msg_id=%s",
                     user_id, getattr(session["question_msg"], "id", "?"))
        except Exception as error:
            log.exception("send Q2 failed user=%s: %s", user_id, error)
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
            log.info("Q3 sent user=%s msg_id=%s",
                     user_id, getattr(session["question_msg"], "id", "?"))
        except Exception as error:
            log.exception("send Q3 failed user=%s: %s", user_id, error)
        session["flow"] = asyncio.create_task(_question_timeout(bot, user_id))
        return

    if state == "q_effect":
        if session["chosen_effect"] is None:
            session["chosen_effect"] = "auto"
            log.info("effect default=auto user=%s", user_id)
        await _start_render(bot, user_id)


# ---------------- RENDER FLOW ----------------

async def _start_render(bot, user_id):
    log.info("_start_render enter user=%s", user_id)
    session = sessions.get(user_id)
    if not session:
        log.warning("_start_render: no session user=%s", user_id)
        return
    if session.get("processing"):
        log.warning("_start_render: already processing user=%s", user_id)
        return
    if not (session["images"] or session.get("zip_path")):
        log.warning("_start_render: no images user=%s", user_id)
        return
    session["state"] = "rendering"
    async with session["lock"]:
        session["processing"] = True

    chat_id = session["chat_id"]
    status = None
    output = None
    try:
        log.info("_start_render prep user=%s", user_id)
        status = await bot.send_message(chat_id, "🖼️ Preparing images...")
        images, stats_list = await asyncio.to_thread(prepare_images, session)
        count = len(images)
        log.info("_start_render prepared user=%s count=%s", user_id, count)

        dur = float(session["chosen_duration"] or _default_duration(count))
        total = dur * count
        capped = False
        if total > MAX_VIDEO_DURATION:
            dur = max(MIN_IMAGE_DURATION, MAX_VIDEO_DURATION / count)
            total = dur * count
            capped = True
            log.info("duration capped user=%s new=%.2fs", user_id, dur)

        mood = session["chosen_mood"]
        if not mood or mood == "auto":
            mood = _choose_mood(stats_list)
        effect = session["chosen_effect"] or "auto"
        lite = count >= LITE_THRESHOLD

        summary = (
            "✅ **Settings locked**\n\n"
            f"📸 Images: {count}\n"
            f"⏱️ Image duration: {dur:.2f}s (total ~{int(total)}s)\n"
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
        pq = queue.Queue()

        def report(pct, cur, tot):
            loop.call_soon_threadsafe(pq.put_nowait, (pct, cur, tot))

        rt = asyncio.create_task(asyncio.to_thread(
            render_video, images, stats_list, output, total,
            mood, lite, session.get("logo_path"),
            None if effect == "auto" else effect, report))

        last_edit = 0
        last_pct = -1
        while not rt.done():
            try:
                pct, cur, tot = await asyncio.wait_for(
                    asyncio.to_thread(pq.get, True, 2.0), timeout=2.5)
                now = time.monotonic()
                if pct != last_pct and (now - last_edit >= 1.5 or pct == 100):
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    try:
                        await status.edit_text(
                            "🎬 Rendering...\n\n"
                            f"[{bar}] {pct}%\n"
                            f"📸 Image: {cur}/{tot}\n"
                            f"🎨 Mood: {mood} · ✨ {effect}")
                        last_pct, last_edit = pct, now
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                continue
            except queue.Empty:
                continue

        duration = await rt
        log.info("_start_render video ready user=%s duration=%.1fs",
                 user_id, duration)

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
        log.info("_start_render success user=%s", user_id)

    except Exception as error:
        log.exception("_start_render failed user=%s: %s", user_id, error)
        try:
            if status:
                await status.edit_text(f"❌ Render failed: {str(error)[:800]}")
        except Exception:
            pass
    finally:
        log.info("_start_render finally user=%s", user_id)
        session = sessions.pop(user_id, None)
        if session:
            for key in ("watcher", "flow"):
                t = session.get(key)
                if t and not t.done():
                    t.cancel()
            work = session.get("work")
            if work and os.path.isdir(work):
                shutil.rmtree(work, ignore_errors=True)
                log.info("cleanup work=%s", work)
        if output and os.path.exists(output):
            try:
                os.remove(output)
                log.info("cleanup output=%s", output)
            except OSError as error:
                log.warning("cleanup output failed: %s", error)


# ---------------- HANDLERS ----------------

def register_image_video_handlers(bot):

    async def on_photo(_, message):
        if not message.from_user:
            return
        session = await _ensure_session(bot, message)
        if session["state"] not in ("uploading", "q_duration", "q_mood", "q_effect"):
            return
        if session.get("processing"):
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
            count = _effective_image_count(session)
            log.info("photo stored user=%s direct=%s effective=%s",
                     message.from_user.id, len(session["images"]), count)
            await message.reply(
                f"📥 {count} image(s) received "
                f"(questions start after {int(AUTO_FINISH_SILENCE)}s silence)")

    async def on_document(_, message):
        if not message.from_user or not message.document:
            return
        session = await _ensure_session(bot, message)
        if session["state"] not in ("uploading", "q_duration", "q_mood", "q_effect"):
            return
        if session.get("processing"):
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
                session["zip_image_count"] = count_zip_images(result)
                _bump(session)
                log.info("zip stored user=%s path=%s images=%s",
                         message.from_user.id, result, session["zip_image_count"])
                if session["zip_image_count"] == 0:
                    return await message.reply(
                        "⚠️ ZIP contains no supported images.")
                return await message.reply(
                    f"📦 ZIP received ({session['zip_image_count']} images). "
                    f"Questions start after {int(AUTO_FINISH_SILENCE)}s silence.")
            if extension not in IMAGE_EXTENSIONS:
                return
            path = os.path.join(session["work"], f"document_{message.id}{extension}")
            result = await safe_download(message, path)
            if not result:
                return await message.reply("⚠️ Download failed, please resend.")
            if session.get("logo_path") is None and _has_transparency(result):
                session["logo_path"] = result
                _bump(session)
                log.info("logo saved user=%s", message.from_user.id)
                return await message.reply("🖼️ Logo detected and saved.")
            if len(session["images"]) >= MAX_IMAGES:
                return await message.reply(f"Maximum {MAX_IMAGES} images reached.")
            session["images"].append((filename, result))
            _bump(session)
            count = _effective_image_count(session)
            log.info("document stored user=%s direct=%s effective=%s",
                     message.from_user.id, len(session["images"]), count)
            await message.reply(
                f"📥 {count} image(s) received "
                f"(questions start after {int(AUTO_FINISH_SILENCE)}s silence)")

    async def on_text(_, message):
        if not message.from_user or not message.text:
            return
        user_id = message.from_user.id
        session = sessions.get(user_id)
        text = message.text.strip().lower()

        if not session:
            log.info("text ignored user=%s (no session) text=%r", user_id, text[:30])
            return

        log.info("text received user=%s state=%s processing=%s text=%r",
                 user_id, session["state"], session.get("processing"), text[:40])

        if session.get("processing"):
            return

        _bump(session)
        state = session["state"]

        if text in {"cancel", "stop", "x", "abort"}:
            for key in ("watcher", "flow"):
                t = session.get(key)
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
                log.info("manual finish trigger user=%s", user_id)
                await _start_question_flow(bot, user_id)
            return

        if state == "q_duration":
            if text == "auto":
                session["chosen_duration"] = None
                await message.reply("⏱️ Duration: auto")
                return await _advance_question(bot, user_id, use_default=True)
            try:
                value = float(text)
                if value < MIN_IMAGE_DURATION or value > 30:
                    raise ValueError
                session["chosen_duration"] = value
                log.info("duration chosen user=%s val=%.2f", user_id, value)
                await message.reply(f"⏱️ Duration: {value:.2f}s per image")
                return await _advance_question(bot, user_id)
            except ValueError:
                session["invalid_count"] += 1
                if session["invalid_count"] >= 2:
                    session["chosen_duration"] = None
                    await message.reply("⚠️ Invalid twice → using auto.")
                    return await _advance_question(bot, user_id, use_default=True)
                return await message.reply(
                    f"⚠️ Invalid. Reply a number "
                    f"({MIN_IMAGE_DURATION}–30s) or `auto`.")

        if state == "q_mood":
            moods = {"romantic", "cinematic", "mono", "bw", "clean", "auto"}
            if text in moods:
                session["chosen_mood"] = "mono" if text == "bw" else text
                log.info("mood chosen user=%s val=%s", user_id, session["chosen_mood"])
                await message.reply(f"🎨 Mood: {session['chosen_mood']}")
                return await _advance_question(bot, user_id)
            session["invalid_count"] += 1
            if session["invalid_count"] >= 2:
                session["chosen_mood"] = "auto"
                await message.reply("⚠️ Invalid twice → using auto.")
                return await _advance_question(bot, user_id)
            return await message.reply(
                "⚠️ Invalid. Reply:\n"
                "`romantic` · `cinematic` · `mono` · `clean` · `auto`")

        if state == "q_effect":
            effects = {"zoom_in", "zoom_out", "ken_burns", "pulse",
                       "pan_left", "pan_right", "auto"}
            if text in effects:
                session["chosen_effect"] = text
                log.info("effect chosen user=%s val=%s", user_id, text)
                await message.reply(f"✨ Effect: {text}")
                return await _advance_question(bot, user_id)
            session["invalid_count"] += 1
            if session["invalid_count"] >= 2:
                session["chosen_effect"] = "auto"
                await message.reply("⚠️ Invalid twice → using auto.")
                return await _advance_question(bot, user_id)
            return await message.reply(
                "⚠️ Invalid. Reply:\n"
                "`zoom_in` · `zoom_out` · `ken_burns` · `pulse` · "
                "`pan_left` · `pan_right` · `auto`")

        log.info("text unhandled user=%s state=%s text=%r",
                 user_id, state, text[:40])

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bot.add_handler(MessageHandler(on_photo, filters.photo), group=1)
    bot.add_handler(MessageHandler(on_document, filters.document), group=1)
    bot.add_handler(MessageHandler(on_text, filters.text), group=1)
    log.info("image_video handlers registered")
