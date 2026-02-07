import cv2
import numpy as np

# ================= SKIN =================

def skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], np.uint8)
    upper = np.array([20, 150, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def skin_ratio(frame):
    if frame.size == 0:
        return 0
    mask = skin_mask(frame)
    return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

# ================= LIPS / FACE PROXIMITY =================

def lips_focus_score(faces, frame):
    score = 0
    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue
        # lower face = lips / mouth zone
        lips = face[int(h*0.55):int(h*0.75), int(w*0.2):int(w*0.8)]
        if skin_ratio(lips) > 0.42:
            score += 120
    return score

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
                score += 180
    return score

# ================= BODY =================

def body_exposure_score(frame):
    h = frame.shape[0]
    upper = frame[int(h*0.20):int(h*0.45)]
    lower = frame[int(h*0.45):int(h*0.80)]
    score = 0
    if skin_ratio(upper) > 0.33:
        score += 140
    if skin_ratio(lower) > 0.30:
        score += 180
    return score

# ================= CHEST / HIP =================

def chest_hip_score(frame, people_boxes):
    score = 0
    for (x,y,w,h) in people_boxes:
        person = frame[y:y+h, x:x+w]
        if person.size == 0:
            continue
        ph = person.shape[0]

        chest = person[int(ph*0.25):int(ph*0.45)]
        hips  = person[int(ph*0.60):int(ph*0.80)]

        if skin_ratio(chest) > 0.36:
            score += 160
        if skin_ratio(hips) > 0.34:
            score += 220
    return score

# ================= PRIVATE ZONE =================

def private_zone_score(frame, people_boxes):
    score = 0
    for (x,y,w,h) in people_boxes:
        person = frame[y:y+h, x:x+w]
        if person.size == 0:
            continue
        ph = person.shape[0]
        mid = person[int(ph*0.45):int(ph*0.65)]
        low = person[int(ph*0.65):int(ph*0.90)]

        if skin_ratio(mid) > 0.38:
            score += 180
        if skin_ratio(low) > 0.32:
            score += 240
    return score

# ================= FLUID / MOTION =================

def fluid_flow_score(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    strong = mag > 2.0
    if strong.sum() == 0:
        return 0
    if np.std(ang[strong]) < 0.9:
        return int(np.mean(mag[strong]) * 35)
    return 0

def fluid_spray_score(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    spray = mag > 3.5
    return min(300, int(spray.sum() / 200))

def droplet_cluster_score(frame, prev_frame):
    diff = cv2.absdiff(frame, prev_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    droplets = sum(1 for c in cnts if 8 < cv2.contourArea(c) < 120)
    return min(250, droplets * 6)

def fluid_trail_score(prev_gray, gray):
    diff = cv2.absdiff(gray, prev_gray)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return min(220, int(cv2.countNonZero(th) / 250))

# ================= FINAL SCORE =================

def nsfw_score(frame, faces, people_boxes=None, prev_frame=None, prev_gray=None):
    score = 0

    score += int(skin_ratio(frame) * 280)
    score += kissing_score(faces)
    score += lips_focus_score(faces, frame)
    score += body_exposure_score(frame)

    if people_boxes is not None:
        score += chest_hip_score(frame, people_boxes)
        score += private_zone_score(frame, people_boxes)

    if prev_frame is not None and prev_gray is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score += fluid_flow_score(prev_gray, gray)
        score += fluid_spray_score(prev_gray, gray)
        score += droplet_cluster_score(frame, prev_frame)
        score += fluid_trail_score(prev_gray, gray)

    return score
