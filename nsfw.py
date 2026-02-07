import cv2
import numpy as np

# =====================================================
# SKIN DETECTION
# =====================================================

def skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], np.uint8)
    upper = np.array([25, 150, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def skin_ratio(frame):
    mask = skin_mask(frame)
    return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])


# =====================================================
# FACE PROXIMITY (INTIMACY / KISSING-LIKE)
# =====================================================

def face_proximity_score(faces):
    if len(faces) < 2:
        return 0

    score = 0
    centers = [(x + w // 2, y + h // 2, w) for (x, y, w, h) in faces]

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            x1, y1, w1 = centers[i]
            x2, y2, w2 = centers[j]
            dist = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
            avg = (w1 + w2) / 2

            if dist < avg * 0.6:
                score += 180
            elif dist < avg * 0.8:
                score += 100

    return score


# =====================================================
# BODY EXPOSURE (UPPER / LOWER REGIONS)
# =====================================================

def body_exposure_score(frame):
    h = frame.shape[0]

    upper = frame[int(h * 0.20):int(h * 0.50)]
    lower = frame[int(h * 0.50):int(h * 0.85)]

    score = 0
    if skin_ratio(upper) > 0.30:
        score += 150
    if skin_ratio(lower) > 0.25:
        score += 200

    return score


# =====================================================
# PERSON REGION ANALYSIS (HOG BOXES)
# =====================================================

def private_region_score(frame, people_boxes):
    score = 0

    for (x, y, w, h) in people_boxes:
        person = frame[y:y + h, x:x + w]
        if person.size == 0:
            continue

        ph = person.shape[0]
        mid = person[int(ph * 0.45):int(ph * 0.65)]
        low = person[int(ph * 0.65):int(ph * 0.90)]

        if skin_ratio(mid) > 0.35:
            score += 160
        if skin_ratio(low) > 0.30:
            score += 220

    return score


# =====================================================
# MOTION / FLOW (RHYTHM / FLUID / WATER)
# =====================================================

def motion_flow(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag, ang

def rhythmic_motion_score(prev_gray, gray):
    mag, _ = motion_flow(prev_gray, gray)
    active = (mag > 1.5) & (mag < 4.5)
    ratio = active.sum() / mag.size
    if ratio > 0.12:
        return min(260, int(ratio * 1600))
    return 0

def fluid_motion_score(prev_gray, gray):
    mag, ang = motion_flow(prev_gray, gray)
    strong = mag > 2.5
    if strong.sum() < 50:
        return 0
    if np.std(ang[strong]) > 1.4:
        return min(250, int(np.mean(mag[strong]) * 45))
    return 0

def wet_surface_score(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    wet = (v > 200) & (s < 70)
    ratio = wet.sum() / (frame.shape[0] * frame.shape[1])
    return min(200, int(ratio * 900))


# =====================================================
# CONTEXT INTENSITY (COMBINED SIGNALS)
# =====================================================

def intimacy_context_score(faces, skin_val, motion_val):
    score = 0

    if len(faces) >= 2 and skin_val > 0.25:
        score += 120

    if motion_val > 120 and skin_val > 0.28:
        score += 180

    if len(faces) >= 2 and motion_val > 150:
        score += 220

    return score


# =====================================================
# FINAL NSFW SCORE (MAIN FUNCTION)
# =====================================================

def nsfw_scene_score(
    frame,
    faces,
    people_boxes=None,
    prev_frame=None,
    prev_gray=None
):
    score = 0

    # --- Skin ---
    skin_val = skin_ratio(frame)
    score += int(skin_val * 280)

    # --- Faces ---
    score += face_proximity_score(faces)

    # --- Body ---
    score += body_exposure_score(frame)

    if people_boxes is not None:
        score += private_region_score(frame, people_boxes)

    # --- Motion / Fluid / Water ---
    motion_val = 0
    if prev_gray is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_val += rhythmic_motion_score(prev_gray, gray)
        motion_val += fluid_motion_score(prev_gray, gray)
        score += motion_val

    if prev_frame is not None:
        score += wet_surface_score(frame)

    # --- Context Boost ---
    score += intimacy_context_score(faces, skin_val, motion_val)

    return score
