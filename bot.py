import os
import cv2
import zipfile
import shutil
import numpy as np
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from user_settings import set_count, get_count

# ---------------- PATHS ----------------
DOWNLOAD_DIR = "downloads"
FRAME_DIR = "frames"
OUTPUT_DIR = "output"

MAX_LIMIT = 120   # 🔥 SAFE FOR RENDER

for d in (DOWNLOAD_DIR, FRAME_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- DETECTORS ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
def motion_score(prev_gray, gray):
    if prev_gray is None:
        return 0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 2, 10, 2, 3, 1.1, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return int(np.mean(mag) * 20)

def scene_change_score(prev, cur):
    if prev is None:
        return 0
    h1 = cv2.calcHist([prev], [0], None, [64], [0,256])
    h2 = cv2.calcHist([cur], [0], None, [64], [0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return int(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) * 120)

def pose_body_score(frame):
    h, w, _ = frame.shape
    boxes, _ = hog.detectMultiScale(frame, (8,8), (16,16), 1.05)
    score = 0

    for (x,y,bw,bh) in boxes:
        ratio = bh / max(bw,1)
        area = (bw*bh)/(w*h)

        if area > 0.2:
            score += 120
        if ratio < 1.3:
            score += 90   # lying / intimate
        if ratio > 2.3:
            score += 60   # standing

    if len(boxes) >= 2:
        score += 180     # intimate proximity

    return min(score, 350)

def nsfw_score(frame, faces, prev_gray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, (0,30,50), (25,255,255))
    skin_ratio = cv2.countNonZero(skin)/(frame.shape[0]*frame.shape[1])

    proximity = 0
    if len(faces) >= 2:
        for i in range(len(faces)):
            for j in range(i+1, len(faces)):
                x1,y1,w1,_ = faces[i]
                x2,y2,w2,_ = faces[j]
                if abs(x1-x2) < (w1+w2)*0.6:
                    proximity += 200

    motion = motion_score(prev_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    score = (
        skin_ratio * 300 +
        proximity * 1.1 +
        motion * 0.7
    )

    return int(min(score, 900))

# ---------------- COMMANDS ----------------
@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 Send video\n"
        "🔥 Smart Pose + Body + Motion + Face + NSFW detection\n"
        "📦 Output: ZIP\n"
        "⚙ /settings <1-120>"
    )

@bot.on_message(filters.command("settings"))
async def settings(_, msg):
    if len(msg.command) < 2:
        return await msg.reply("Usage: /settings <1-120>")
    count = int(msg.command[1])
    if count < 1 or count > MAX_LIMIT:
        return await msg.reply("❌ Max 120 (Render safe)")
    set_count(msg.from_user.id, count)
    await msg.reply(f"✅ Image count set to {count}")

# ---------------- VIDEO HANDLER ----------------
@bot.on_message(filters.video)
async def video_handler(_, msg):
    user_id = msg.from_user.id
    target = min(get_count(user_id), MAX_LIMIT)

    status = await msg.reply("⬇️ Downloading…")
    video_path = await msg.download(file_name=f"{DOWNLOAD_DIR}/")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // (target * 3))

    scores = []
    prev_gray = None
    prev_frame = None
    frame_no = 0

    await status.edit("🎞 Analysing video…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % step != 0:
            continue

        if frame.shape[1] > 960:
            scale = 960/frame.shape[1]
            frame = cv2.resize(frame,(960,int(frame.shape[0]*scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.2,5)

        total_score = (
            pose_body_score(frame) * 1.3 +
            motion_score(prev_gray, gray) * 1.0 +
            scene_change_score(prev_frame, frame) * 1.4 +
            nsfw_score(frame, faces, prev_gray) * 1.5
        )

        scores.append((total_score, frame_no))
        prev_gray = gray
        prev_frame = frame

    cap.release()

    if not scores:
        return await status.edit("❌ No analyzable frames")

    scores.sort(reverse=True)
    selected = sorted([f for _,f in scores[:target]])

    cap = cv2.VideoCapture(video_path)
    saved = 0
    cur = 0

    await status.edit("📸 Extracting…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cur += 1
        if cur not in selected:
            continue

        cv2.imwrite(f"{FRAME_DIR}/{user_id}_{saved}.jpg", frame)
        saved += 1
        if saved >= target:
            break

    cap.release()

    zip_path = f"{OUTPUT_DIR}/{user_id}_highlights.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for i in range(saved):
            z.write(f"{FRAME_DIR}/{user_id}_{i}.jpg", arcname=f"{i+1}.jpg")

    await msg.reply_document(
        zip_path,
        caption=f"✅ Extracted {saved} highlights"
    )

    await status.edit("✅ Done")

    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
