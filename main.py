from pyrogram import Client
import asyncio

from config import API_ID, API_HASH, BOT_TOKEN
from bot.handler import register_handlers
from bot.db import init_db
from bot.worker import worker

app = Client(
    "video_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

async def main():
    init_db()
    register_handlers(app)
    await app.start()
    asyncio.create_task(worker(app))
    print("🤖 Bot running...")
    await idle()

from pyrogram.idle import idle
asyncio.run(main())
