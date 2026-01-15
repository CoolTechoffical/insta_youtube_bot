import cv2
import numpy as np

def skin_mask_ratio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 50])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

def face_proximity_score(faces):
    if len(faces) < 2:
        return 0
    score = 0
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            x1, y1, w1, _ = faces[i]
            x2, y2, w2, _ = faces[j]
            dist = np.hypot(x1 - x2, y1 - y2)
            if dist < (w1 + w2) * 0.5:
                score += 200
    return min(score, 300)

def body_exposure_score(frame):
    h = frame.shape[0]
    upper = frame[int(h*0.2):int(h*0.45)]
    mid = frame[int(h*0.45):int(h*0.7)]
    score = 0
    if skin_mask_ratio(upper) > 0.3:
        score += 150
    if skin_mask_ratio(mid) > 0.25:
        score += 250
    return score

def motion_intensity(prev_gray, gray):
    if prev_gray is None:
        return 0
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return int(np.mean(mag) * 100)

def nsfw_scene_score(frame, faces, prev_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    score = (
        skin_mask_ratio(frame) * 300 +
        face_proximity_score(faces) * 1.2 +
        body_exposure_score(frame) * 1.3 +
        motion_intensity(prev_gray, gray) * 0.8
    )

    score = int(min(score, 1000))

    if score > 850:
        level = "high"
    elif score > 550:
        level = "medium"
    else:
        level = "low"

    return score, gray, level
