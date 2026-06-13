import cv2
import numpy as np

# =====================================================
# SKIN DETECTION
# =====================================================

def skin_mask(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 25, 60], np.uint8)
    upper = np.array([25, 180, 255], np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask


def skin_ratio(frame):

    if frame.size == 0:
        return 0

    mask = skin_mask(frame)

    return cv2.countNonZero(mask) / (
        frame.shape[0] * frame.shape[1]
    )

# =====================================================
# FACE CLOSE INTERACTION
# =====================================================

def face_interaction_score(faces):

    if len(faces) < 2:
        return 0

    score = 0

    centers = []

    for (x, y, w, h) in faces:

        cx = x + w // 2
        cy = y + h // 2

        centers.append((cx, cy, w))

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):

            x1, y1, w1 = centers[i]
            x2, y2, w2 = centers[j]

            dist = np.sqrt(
                (x1 - x2) ** 2 +
                (y1 - y2) ** 2
            )

            avg = (w1 + w2) / 2

            ratio = dist / avg

            if ratio < 0.45:
                score += 260

            elif ratio < 0.7:
                score += 160

            elif ratio < 1.0:
                score += 80

    return score

# =====================================================
# FULL FRAME EXPOSURE
# =====================================================

def exposure_score(frame):

    h = frame.shape[0]

    upper = frame[0:int(h * 0.35)]
    middle = frame[int(h * 0.35):int(h * 0.7)]
    lower = frame[int(h * 0.7):]

    score = 0

    upper_skin = skin_ratio(upper)
    middle_skin = skin_ratio(middle)
    lower_skin = skin_ratio(lower)

    score += int(upper_skin * 180)
    score += int(middle_skin * 260)
    score += int(lower_skin * 140)

    return score

# =====================================================
# PERSON REGION ANALYSIS
# =====================================================

def body_focus_score(frame, people_boxes):

    if people_boxes is None:
        return 0

    score = 0

    for (x, y, w, h) in people_boxes:

        person = frame[y:y+h, x:x+w]

        if person.size == 0:
            continue

        ph = person.shape[0]

        upper = person[
            int(ph * 0.18):int(ph * 0.50),
            :
        ]

        middle = person[
            int(ph * 0.50):int(ph * 0.78),
            :
        ]

        upper_skin = skin_ratio(upper)
        middle_skin = skin_ratio(middle)

        if upper_skin > 0.25:
            score += int(upper_skin * 400)

        if middle_skin > 0.22:
            score += int(middle_skin * 500)

    return score

# =====================================================
# BODY CONTACT / HUG
# =====================================================

def contact_score(people_boxes):

    if people_boxes is None:
        return 0

    score = 0

    for i in range(len(people_boxes)):
        for j in range(i + 1, len(people_boxes)):

            x1, y1, w1, h1 = people_boxes[i]
            x2, y2, w2, h2 = people_boxes[j]

            xa = max(x1, x2)
            ya = max(y1, y2)

            xb = min(x1 + w1, x2 + w2)
            yb = min(y1 + h1, y2 + h2)

            overlap = max(0, xb - xa) * max(0, yb - ya)

            if overlap > 0:
                score += 240

    return score

# =====================================================
# MOTION ANALYSIS
# =====================================================

def optical_flow(prev_gray, gray):

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    mag, ang = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    return mag, ang


def rhythmic_motion_score(prev_gray, gray):

    mag, _ = optical_flow(prev_gray, gray)

    active = (mag > 1.5) & (mag < 5.5)

    ratio = active.sum() / mag.size

    if ratio > 0.10:
        return min(
            300,
            int(ratio * 1700)
        )

    return 0


def directional_motion_score(prev_gray, gray):

    mag, ang = optical_flow(prev_gray, gray)

    strong = mag > 2.0

    if strong.sum() < 100:
        return 0

    std = np.std(ang[strong])

    if std > 1.0:
        return min(
            220,
            int(np.mean(mag[strong]) * 40)
        )

    return 0

# =====================================================
# CONTEXT BOOST
# =====================================================

def context_score(
    faces,
    people_boxes,
    skin_val,
    motion_val
):

    score = 0

    if len(faces) >= 2 and skin_val > 0.22:
        score += 120

    if motion_val > 140 and skin_val > 0.24:
        score += 180

    if people_boxes is not None and len(people_boxes) >= 2:
        score += 100

    return score
    
def multi_person_score(people_boxes):

    if people_boxes is None:
        return 0

    count = len(people_boxes)

    if count >= 4:
        return 250

    if count >= 3:
        return 180

    if count >= 2:
        return 100

    return 0

def full_frame_exposure_bonus(frame):

    ratio = skin_ratio(frame)

    if ratio > 0.60:
        return 600

    if ratio > 0.45:
        return 350

    if ratio > 0.30:
        return 180

    return 0

def person_coverage_score(frame, people_boxes):

    if people_boxes is None:
        return 0

    score = 0

    for (x, y, w, h) in people_boxes:

        person = frame[y:y+h, x:x+w]

        if person.size == 0:
            continue

        ratio = skin_ratio(person)

        if ratio > 0.50:
            score += 450

        elif ratio > 0.35:
            score += 250

        elif ratio > 0.20:
            score += 120

    return score

def body_size_score(frame, people_boxes):

    if people_boxes is None:
        return 0

    h, w = frame.shape[:2]

    frame_area = h * w

    score = 0

    for (_, _, bw, bh) in people_boxes:

        area_ratio = (bw * bh) / frame_area

        if area_ratio > 0.35:
            score += 220

        elif area_ratio > 0.20:
            score += 120

    return score
# =====================================================
# FINAL NSFW SCORE
# =====================================================

def nsfw_scene_score(
    frame,
    faces,
    people_boxes=None,
    prev_frame=None,
    prev_gray=None
):

    score = 0

    # -----------------------------------
    # GLOBAL SKIN
    # -----------------------------------

    skin_val = skin_ratio(frame)

    score += int(skin_val * 320)

    # -----------------------------------
    # FACE INTERACTION
    # -----------------------------------

    score += face_interaction_score(faces)

    # -----------------------------------
    # EXPOSURE
    # -----------------------------------

    score += exposure_score(frame)

    # -----------------------------------
    # BODY REGIONS
    # -----------------------------------

    score += body_focus_score(
        frame,
        people_boxes
    )

    # -----------------------------------
    # BODY CONTACT
    # -----------------------------------

    score += multi_person_score(people_boxes)

    score += full_frame_exposure_bonus(frame)

    score += person_coverage_score(
        frame,
        people_boxes
)

    score += body_size_score(
        frame,
        people_boxes
)


    # -----------------------------------
    # MOTION
    # -----------------------------------

    motion_val = 0

    if prev_gray is not None:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        motion_val += rhythmic_motion_score(
            prev_gray,
            gray
        )

        motion_val += directional_motion_score(
            prev_gray,
            gray
        )

        score += motion_val

    # -----------------------------------
    # CONTEXT BOOST
    # -----------------------------------

    score += context_score(
        faces,
        people_boxes,
        skin_val,
        motion_val
    )

    # -----------------------------------
    # NORMALIZE
    # -----------------------------------

    score = int(
        max(0, min(score, 3000))
    )

    return score
