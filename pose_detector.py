# pose_detector.py
import cv2
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_body_pose(frame):
    """
    Returns:
    pose_tags: list[str]
    pose_score: int
    """
    pose_tags = []
    pose_score = 0

    h, w, _ = frame.shape
    boxes, _ = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
    )

    for (x, y, bw, bh) in boxes:
        ratio = bh / bw if bw else 0

        if bh > h * 0.6:
            pose_tags.append("full_body")
            pose_score += 120

        if y < h * 0.3 and bh < h * 0.5:
            pose_tags.append("upper_body")
            pose_score += 80

        if ratio < 1.1:
            pose_tags.append("lying_pose")
            pose_score += 100

        if ratio > 2.5:
            pose_tags.append("standing_pose")
            pose_score += 50

    if len(boxes) >= 2:
        pose_tags.append("intimate_pose")
        pose_score += 150

    return list(set(pose_tags)), pose_score
