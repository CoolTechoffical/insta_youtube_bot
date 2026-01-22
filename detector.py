import random

AI_TOOLS = [
    "Stable Diffusion",
    "Midjourney",
    "DALL·E",
    "Adobe Firefly",
    "Leonardo AI"
]

def detect_image_ai(image_path):
    """
    Placeholder logic
    Replace this with PyTorch / ONNX model later
    """

    # Fake probability logic (for now)
    ai_probability = random.uniform(0.4, 0.95)

    is_ai = ai_probability > 0.6

    result = {
        "is_ai": is_ai,
        "confidence": round(ai_probability * 100, 2),
        "tool": random.choice(AI_TOOLS) if is_ai else "Real Camera"
    }

    return result


def detect_video_ai(frame_results):
    """
    frame_results = list of detect_image_ai outputs
    """

    ai_frames = sum(1 for r in frame_results if r["is_ai"])
    total = len(frame_results)

    ai_ratio = ai_frames / total if total else 0

    return {
        "is_ai": ai_ratio >= 0.6,
        "confidence": round(ai_ratio * 100, 2),
        "tool": max(
            [r["tool"] for r in frame_results if r["is_ai"]],
            default="Unknown",
            key=lambda x: frame_results.count(x),
        ),
    }
