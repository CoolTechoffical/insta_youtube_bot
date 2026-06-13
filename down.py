import os
import yt_dlp

DOWNLOAD_DIR = "downloads"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_video_from_url(url):

ydl_opts = {
    "outtmpl": f"{DOWNLOAD_DIR}/%(id)s.%(ext)s",
    "quiet": True,
    "noplaylist": True,
    "merge_output_format": "mp4",
    "restrictfilenames": True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:

    info = ydl.extract_info(
        url,
        download=True
    )

    video_path = ydl.prepare_filename(info)

    if os.path.exists(video_path):
        return video_path

    base = os.path.splitext(video_path)[0]

    for ext in (
        ".mp4",
        ".mkv",
        ".webm",
        ".mov",
        ".avi"
    ):

        candidate = base + ext

        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Downloaded file not found"
    )
