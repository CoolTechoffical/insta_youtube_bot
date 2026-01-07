import subprocess
import json

def get_video_info(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-of", "json",
        path
    ]
    result = subprocess.check_output(cmd)
    data = json.loads(result)
    s = data["streams"][0]
    return int(s["width"]), int(s["height"]), float(s["duration"])

def resize_video(input_path, output_path, height):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
