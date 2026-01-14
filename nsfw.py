import cv2
import numpy as np

def skin_mask_ratio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 35, 60], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

def face_proximity_score(faces):
    if len(faces) < 2:
        return 0
    score = 0
    centers = [(x+w//2, y+h//2, w) for (x,y,w,h) in faces]

    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            x1,y1,w1 = centers[i]
            x2,y2,w2 = centers[j]
            dist = ((x1-x2)**2 + (y1-y2)**2) ** 0.5
            avg = (w1+w2)/2
            if dist < avg*0.6:
                score += 180
            elif dist < avg*0.8:
                score += 100
    return score

def body_exposure_score(frame):
    h = frame.shape[0]
    upper = frame[int(h*0.20):int(h*0.50)]
    lower = frame[int(h*0.50):int(h*0.85)]
    score = 0
    if skin_mask_ratio(upper) > 0.30:
        score += 150
    if skin_mask_ratio(lower) > 0.25:
        score += 200
    return score

def motion_intensity(prev_gray, gray):
    if prev_gray is None:
        return 0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return int(np.mean(mag) * 80)

def nsfw_scene_score(frame, faces, prev_gray=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = int(skin_mask_ratio(frame) * 300)
    score += face_proximity_score(faces)
    score += body_exposure_score(frame)
    score += motion_intensity(prev_gray, gray)
    return score, gray
