import subprocess
import json
import os


def get_video_info(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=width,height",
        "-of", "json",
        path
    ]
    result = subprocess.check_output(cmd)
    data = json.loads(result)

    duration = float(data["format"]["duration"])
    stream = next(s for s in data["streams"] if s.get("width"))
    return int(stream["width"]), int(stream["height"]), duration


def resize_video(input_path, output_path, height):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
