import os
import yt_dlp

DOWNLOAD_DIR = "downloads"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_video_from_url(url):

    ydl_opts = {
        "outtmpl": f"{DOWNLOAD_DIR}/url_%(id)s.%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "merge_output_format": "mp4"
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(
            url,
            download=True
        )

        return ydl.prepare_filename(info)
