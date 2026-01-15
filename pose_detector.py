import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_body_pose(frame):
    h, w, _ = frame.shape
    boxes, _ = hog.detectMultiScale(frame, (8,8), (16,16), 1.05)

    tags = []
    score = 0

    for (x,y,bw,bh) in boxes:
        ratio = bh / max(bw,1)

        if bh > h * 0.65:
            tags.append("full_body")
            score += 120
        if y < h * 0.35:
            tags.append("upper_body")
            score += 80
        if ratio < 1.2:
            tags.append("lying_pose")
            score += 90
        if ratio > 2.4:
            tags.append("standing_pose")
            score += 60

    if len(boxes) >= 2:
        tags.append("intimate_pose")
        score += 150

    return list(set(tags)), min(score, 300)
