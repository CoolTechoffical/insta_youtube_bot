# bot/worker.py
import os
import asyncio
from pyrogram import Client
from bot.db import get_next_job, update_status
from bot.video import resize_video
from config import DOWNLOAD_DIR

QUALITY_HEIGHT = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080
}

async def worker(app: Client):
    while True:
        job = get_next_job()
        if not job:
            await asyncio.sleep(5)
            continue

        job_id, _, chat_id, file_id, _, _, _, quality, _, _ = job
        update_status(job_id, "processing")

        input_path = f"{DOWNLOAD_DIR}/{job_id}.mp4"
        output_path = f"{DOWNLOAD_DIR}/{job_id}_{quality}.mp4"

        await app.download_media(file_id, input_path)
        resize_video(input_path, output_path, QUALITY_HEIGHT[quality])

        await app.send_video(chat_id, output_path, caption=f"✅ {quality} ready")

        os.remove(input_path)
        os.remove(output_path)
        update_status(job_id, "done")
