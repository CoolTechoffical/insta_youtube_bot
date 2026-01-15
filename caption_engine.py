import json
import random

with open("captions.json", "r", encoding="utf-8") as f:
    CAPTIONS = json.load(f)

PRIORITY = [
    "adult_scene",
    "intercourse",
    "oral_play",
    "kissing",
    "intimate_pose",
    "suggestive",
    "nudity_detected"
]

def get_caption(tags):
    if not tags:
        return random.choice(CAPTIONS["default"])

    for key in PRIORITY:
        if key in tags and key in CAPTIONS:
            return random.choice(CAPTIONS[key])

    for tag in tags:
        if tag in CAPTIONS:
            return random.choice(CAPTIONS[tag])

    return random.choice(CAPTIONS["default"])
