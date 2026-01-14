import cv2
import numpy as np

# ---------------- SKIN MASK ----------------
def skin_mask_ratio(frame):
    if frame is None or frame.size == 0:
        return 0.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 50], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = frame.shape[0] * frame.shape[1]

    return skin_pixels / max(total_pixels, 1)


# ---------------- FACE PROXIMITY ----------------
def face_proximity_score(faces):
    if len(faces) < 2:
        return 0

    score = 0
    centers = [(x + w // 2, y + h // 2, w) for (x, y, w, h) in faces]

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            x1, y1, w1 = centers[i]
            x2, y2, w2 = centers[j]

            dist = np.hypot(x1 - x2, y1 - y2)
            avg_width = (w1 + w2) / 2

            if dist < avg_width * 0.55:
                score += 220
            elif dist < avg_width * 0.75:
                score += 140

    return min(score, 400)


# ---------------- BODY EXPOSURE ----------------
def body_exposure_score(frame):
    h = frame.shape[0]

    upper = frame[int(h * 0.18):int(h * 0.48)]
    mid   = frame[int(h * 0.48):int(h * 0.68)]
    lower = frame[int(h * 0.68):int(h * 0.90)]

    score = 0

    if skin_mask_ratio(upper) > 0.32:
        score += 120

    if skin_mask_ratio(mid) > 0.28:
        score += 180

    if skin_mask_ratio(lower) > 0.26:
        score += 220

    return min(score, 450)


# ---------------- MOTION INTENSITY ----------------
def motion_intensity(prev_gray, gray):
    if prev_gray is None:
        return 0

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_motion = np.mean(mag)

    if mean_motion > 2.2:
        return 260
    elif mean_motion > 1.4:
        return 160
    elif mean_motion > 0.8:
        return 80

    return 30


# ---------------- MASTER NSFW SCORE ----------------
def nsfw_scene_score(frame, faces, prev_gray=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    score = 0

    score += int(skin_mask_ratio(frame) * 320)
    score += face_proximity_score(faces)
    score += body_exposure_score(frame)
    score += motion_intensity(prev_gray, gray)

    # Hard clamp (prevents runaway scores)
    score = min(score, 1200)

    return score, gray
