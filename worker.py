import cv2
import time
import os
import numpy as np
from PIL import Image
from db import update_job
from config import MAX_IMAGES, MAX_SAFE_SECONDS, MAX_WIDTH

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

def process_job(job, bot):
    start_time = time.time()

    cap = cv2.VideoCapture(job["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, job["last_frame"])

    saved = job["saved_images"]
    last_frame = job["last_frame"]
    prev_gray = None

    while cap.isOpened():
        if time.time() - start_time > MAX_SAFE_SECONDS:
            update_job(job["job_id"], {"status": "waiting"})
            return

        ret, frame = cap.read()
        if not ret:
            break

        last_frame += 1

        h, w, _ = frame.shape
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / w
            frame = cv2.resize(
                frame, (MAX_WIDTH, int(h * scale))
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        motion = 0
        if prev_gray is not None:
            motion = int(np.sum(
                cv2.absdiff(prev_gray, gray)
            ) / 1_000_000)
        prev_gray = gray

        score = motion + (150 if len(faces) else 0)

        if score > 120 and saved < MAX_IMAGES:
            path = f"frames/{job['job_id']}_{saved}.jpg"
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1

        update_job(job["job_id"], {
            "last_frame": last_frame,
            "saved_images": saved,
            "status": "processing"
        })

        if saved >= MAX_IMAGES:
            break

    cap.release()

    # ---------- PDF ----------
    images = [
        f"frames/{job['job_id']}_{i}.jpg"
        for i in range(saved)
    ]

    if images:
        first = Image.open(images[0]).convert("RGB")
        rest = [Image.open(i).convert("RGB") for i in images[1:]]

        pdf_path = f"output/{job['job_id']}.pdf"
        first.save(pdf_path, save_all=True, append_images=rest)

        update_job(job["job_id"], {
            "status": "completed",
            "pdf_path": pdf_path
        })

        bot.send_document(
            job["user_id"],
            pdf_path,
            caption=f"✅ {saved} highlights (auto-recovered)"
        )
