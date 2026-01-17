import cv2
import numpy as np

# ---------------- SKIN ----------------
def skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0,30,60], np.uint8)
    upper = np.array([20,150,255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def skin_ratio(frame):
    mask = skin_mask(frame)
    return cv2.countNonZero(mask)/(frame.shape[0]*frame.shape[1])

# ---------------- HEURISTICS ----------------
def kissing_score(faces):
    if len(faces) < 2:
        return 0
    score = 0
    for i in range(len(faces)):
        for j in range(i+1,len(faces)):
            x1,y1,w1,h1 = faces[i]
            x2,y2,w2,h2 = faces[j]
            d = ((x1-x2)**2 + (y1-y2)**2)**0.5
            if d < (w1+w2)/2:
                score += 150
    return score

def body_exposure_score(frame):
    h = frame.shape[0]
    up = frame[int(h*0.25):int(h*0.55)]
    low = frame[int(h*0.55):int(h*0.85)]
    score = 0
    if skin_ratio(up) > 0.35:
        score += 120
    if skin_ratio(low) > 0.30:
        score += 150
    return score

def private_zone_score(frame, people):
    score = 0
    for (x,y,w,h) in people:
        person = frame[y:y+h, x:x+w]
        if person.size == 0:
            continue
        ph = person.shape[0]
        mid = person[int(ph*0.45):int(ph*0.65)]
        low = person[int(ph*0.65):int(ph*0.9)]
        if skin_ratio(mid) > 0.38:
            score += 160
        if skin_ratio(low) > 0.32:
            score += 220
    return score

# ---------------- ADVANCED FLUID ----------------
def fluid_flow_score(prev_gray, gray):
    if prev_gray is None:
        return 0
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,0.5,3,15,3,5,1.2,0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    strong = mag > 2.5
    if strong.sum() == 0:
        return 0
    if np.std(ang[strong]) < 0.9:
        return int(np.mean(mag[strong]) * 35)
    return 0

def fluid_spray_score(prev_gray, gray):
    if prev_gray is None:
        return 0
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,0.5,3,15,3,5,1.2,0)
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    return min(300, int((mag > 3.5).sum()/180))

def droplet_cluster_score(frame, prev_frame):
    if prev_frame is None:
        return 0
    diff = cv2.absdiff(frame, prev_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    droplets = sum(1 for c in cnts if 8 < cv2.contourArea(c) < 120)
    return min(250, droplets * 6)

def fluid_trail_score(prev_gray, gray):
    if prev_gray is None:
        return 0
    diff = cv2.absdiff(gray, prev_gray)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return min(220, int(cv2.countNonZero(th)/250))

# ---------------- FINAL SCORE ----------------
def nsfw_score(frame, faces, people=None, prev_frame=None, prev_gray=None):
    score = 0
    score += int(skin_ratio(frame)*280)
    score += kissing_score(faces)
    score += body_exposure_score(frame)

    if people is not None:
        score += private_zone_score(frame, people)

    if prev_gray is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score += fluid_flow_score(prev_gray, gray)
        score += fluid_spray_score(prev_gray, gray)
        score += droplet_cluster_score(frame, prev_frame)
        score += fluid_trail_score(prev_gray, gray)

    return score
