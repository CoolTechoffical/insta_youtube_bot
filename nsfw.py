import cv2
import numpy as np

# ---------------- SKIN MASK ----------------
def skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], np.uint8)
    upper = np.array([20, 150, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def skin_ratio(frame):
    mask = skin_mask(frame)
    return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

# ---------------- KISSING HEURISTIC ----------------
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
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
            if dist < (w1+w2)/2 * 0.7:
                score += 150
    return score

# ---------------- BODY EXPOSURE ----------------
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

# ---------------- PRIVATE ZONE ----------------
def private_zone_score(frame, people_boxes):
    score = 0
    for (x,y,w,h) in people_boxes:
        person = frame[y:y+h, x:x+w]
        if person.size == 0:
            continue
        ph = person.shape[0]
        mid = person[int(ph*0.45):int(ph*0.65), :]
        low = person[int(ph*0.65):int(ph*0.9), :]
        if skin_ratio(mid) > 0.38:
            score += 160
        if skin_ratio(low) > 0.32:
            score += 220
    return score

# ---------------- FLUID-LIKE MOTION ----------------
def fluid_like_score(frame, prev_frame):
    if prev_frame is None:
        return 0

    diff = cv2.absdiff(frame, prev_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, motion = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1]
    val = hsv[:,:,2]

    bright = cv2.inRange(val, 200, 255)
    low_sat = cv2.inRange(sat, 0, 60)

    combined = motion & bright & low_sat
    area = cv2.countNonZero(combined)

    return min(250, int(area / 120))

# ---------------- MAIN NSFW SCORE ----------------
def nsfw_score(frame, faces, people_boxes=None, prev_frame=None):
    score = 0
    score += int(skin_ratio(frame) * 280)
    score += kissing_score(faces)
    score += body_exposure_score(frame)

    if people_boxes is not None:
        score += private_zone_score(frame, people_boxes)

    if prev_frame is not None:
        score += fluid_like_score(frame, prev_frame)

    return score
