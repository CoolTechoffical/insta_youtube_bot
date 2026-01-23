import cv2
import numpy as np
from PIL import Image, ExifTags

AI_TOOLS = {
    "midjourney": "Midjourney",
    "stable diffusion": "Stable Diffusion",
    "dall-e": "DALL·E",
    "openai": "DALL·E",
    "firefly": "Adobe Firefly",
    "leonardo": "Leonardo AI",
    "gemini": "Gemini Image",
    "nano banana": "Nano Banana",
}

def exif_check(path):
    score = 0
    tool = None
    try:
        img = Image.open(path)
        exif = img.getexif()
        for tag, val in exif.items():
            text = str(val).lower()
            for k in AI_TOOLS:
                if k in text:
                    score += 3
                    tool = AI_TOOLS[k]
    except:
        pass
    return score, tool

def resolution_check(img):
    h, w = img.shape[:2]
    if (w * h) > 12_000_000 or (w % 64 == 0 and h % 64 == 0):
        return 1
    return 0

def noise_check(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = np.std(gray)
    if noise < 20:
        return 1
    return 0

def color_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    if entropy < 3.5:
        return 1
    return 0

def detect_image_ai(path):
    img = cv2.imread(path)
    if img is None:
        return {"is_ai": False, "confidence": 0, "tool": "Unknown", "signals": ["Unreadable image"]}

    score = 0
    signals = []
    tool = None

    s, t = exif_check(path)
    score += s
    if s:
        signals.append("AI software metadata detected")
        tool = t

    r = resolution_check(img)
    if r:
        score += 1
        signals.append("AI-style resolution pattern")

    n = noise_check(img)
    if n:
        score += 1
        signals.append("Low sensor noise (synthetic)")

    e = color_entropy(img)
    if e:
        score += 1
        signals.append("Low color entropy")

    if score >= 4:
        status = "AI Generated"
        confidence = min(95, 60 + score * 8)
    elif score >= 2:
        status = "Possibly AI"
        confidence = 45 + score * 5
    else:
        status = "Real"
        confidence = 15 + score * 10

    return {
        "is_ai": status != "Real",
        "confidence": confidence,
        "tool": tool or ("Unknown AI Tool" if status != "Real" else "Camera"),
        "signals": signals or ["No synthetic indicators"]
    }

def detect_video_ai(frames):
    ai = sum(1 for f in frames if f["is_ai"])
    ratio = ai / len(frames) if frames else 0

    tool_votes = {}
    for f in frames:
        if f["tool"] != "Camera":
            tool_votes[f["tool"]] = tool_votes.get(f["tool"], 0) + 1

    tool = max(tool_votes, key=tool_votes.get) if tool_votes else "Camera"

    return {
        "is_ai": ratio >= 0.6,
        "confidence": round(ratio * 100, 2),
        "tool": tool
    }
