import cv2
import numpy as np

def skin_ratio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 80], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

def kissing_score(faces):
    if len(faces) < 2:
        return 0
    score = 0
    for i in range(len(faces)):
        for j in range(i+1, len(faces)):
            x1,y1,w1,h1 = faces[i]
            x2,y2,w2,h2 = faces[j]
            c1 = (x1+w1//2, y1+h1//2)
            c2 = (x2+w2//2, y2+h2//2)
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5
            if dist < (w1+w2)/2 * 0.7:
                score += 150
    return score

def body_exposure_score(frame):
    h = frame.shape[0]
    upper = frame[int(h*0.25):int(h*0.55), :]
    lower = frame[int(h*0.55):int(h*0.85), :]
    score = 0
    if skin_ratio(upper) > 0.35:
        score += 120
    if skin_ratio(lower) > 0.30:
        score += 150
    return score

def nsfw_score(frame, faces):
    return int(
        skin_ratio(frame) * 300 +
        kissing_score(faces) +
        body_exposure_score(frame)
    )
