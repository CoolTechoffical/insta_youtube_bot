from PIL import Image
from PIL.ExifTags import TAGS

AI_SOFTWARE_KEYWORDS = [
    "stable diffusion",
    "midjourney",
    "dall-e",
    "openai",
    "firefly",
    "leonardo",
    "gemini",
    "nano banana",
    "synthetic"
]

AI_TOOLS_MAP = {
    "stable diffusion": "Stable Diffusion",
    "midjourney": "Midjourney",
    "dall-e": "DALL·E",
    "openai": "DALL·E",
    "firefly": "Adobe Firefly",
    "leonardo": "Leonardo AI",
    "gemini": "Gemini Image",
    "nano banana": "Nano Banana",
}

def read_exif(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif() or {}
        data = {}

        for tag, value in exif.items():
            name = TAGS.get(tag, tag)
            data[name] = str(value).lower()

        return data
    except Exception:
        return {}

def detect_image_ai(image_path):
    exif = read_exif(image_path)

    signals = []
    detected_tool = "Unknown"

    # Camera evidence
    if "Make" in exif or "Model" in exif:
        signals.append("Camera metadata found")

    # AI software evidence
    for value in exif.values():
        for key in AI_SOFTWARE_KEYWORDS:
            if key in value:
                signals.append(f"AI software tag: {key}")
                detected_tool = AI_TOOLS_MAP.get(key, "Unknown AI Tool")

    is_ai = len(signals) >= 2
    confidence = min(95, 30 + len(signals) * 25) if is_ai else 15 + len(signals) * 10

    return {
        "is_ai": is_ai,
        "confidence": confidence,
        "tool": detected_tool if is_ai else "Real Camera",
        "signals": signals or ["No AI indicators found"]
    }

def detect_video_ai(frame_results):
    ai_frames = [f for f in frame_results if f["is_ai"]]
    ratio = len(ai_frames) / len(frame_results) if frame_results else 0

    tool_votes = {}
    for f in ai_frames:
        tool_votes[f["tool"]] = tool_votes.get(f["tool"], 0) + 1

    tool = max(tool_votes, key=tool_votes.get) if tool_votes else "Real Camera"

    return {
        "is_ai": ratio >= 0.6,
        "confidence": round(ratio * 100, 2),
        "tool": tool
    }
