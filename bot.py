import uuid
import asyncio
from pyrogram import Client, filters
from config import API_ID, API_HASH, BOT_TOKEN
from db import create_job, get_pending_jobs
from worker import process_job

bot = Client("auto_recovery_bot", API_ID, API_HASH, BOT_TOKEN)

@bot.on_message(filters.command("start"))
async def start(_, msg):
    await msg.reply(
        "🎬 Send a video\n"
        "⚙ Auto-recovery enabled\n"
        "🔁 Server-safe processing"
    )

@bot.on_message(filters.video)
async def video_handler(_, msg):
    video_path = await msg.download()
    job_id = str(uuid.uuid4())

    job = {
        "job_id": job_id,
        "user_id": msg.from_user.id,
        "video_path": video_path,
        "status": "processing",
        "last_frame": 0,
        "saved_images": 0
    }

    create_job(job)

    await msg.reply(
        "⏳ Processing started\n"
        "🛡 Auto-resume enabled if server busy"
    )

def resume_jobs():
    pending = get_pending_jobs()
    for job in pending:
        process_job(job, bot)
