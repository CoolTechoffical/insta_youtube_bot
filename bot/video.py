# bot/video.py
import subprocess
import json

def get_video_info(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=width,height",
        "-of", "json", path
    ]
    data = json.loads(subprocess.check_output(cmd))
    duration = int(float(data["format"]["duration"]))
    stream = next(s for s in data["streams"] if s.get("width"))
    return stream["width"], stream["height"], duration


def resize_video(input_path, output_path, height):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale=-2:{height}",
        "-preset", "ultrafast",
        "-c:v", "libx264",
        "-crf", "28",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
