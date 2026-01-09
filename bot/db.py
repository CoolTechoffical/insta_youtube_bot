# bot/db.py
import sqlite3
from datetime import datetime

DB_PATH = "videos.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS video_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        chat_id INTEGER,
        file_id TEXT,
        file_unique_id TEXT,
        file_size INTEGER,
        duration INTEGER,
        requested_quality TEXT,
        status TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()


def add_job(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO video_jobs 
    (user_id, chat_id, file_id, file_unique_id, file_size, duration, requested_quality, status, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["user_id"],
        data["chat_id"],
        data["file_id"],
        data["file_unique_id"],
        data["file_size"],
        data["duration"],
        data["requested_quality"],
        "queued",
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()


def get_next_job():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM video_jobs WHERE status='queued' ORDER BY id LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row


def update_status(job_id, status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE video_jobs SET status=? WHERE id=?", (status, job_id))
    conn.commit()
    conn.close()
