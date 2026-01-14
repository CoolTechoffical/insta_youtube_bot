import json
import os
import random

CAPTIONS_FILE = "captions.json"

if os.path.exists(CAPTIONS_FILE):
    with open(CAPTIONS_FILE, "r") as f:
        captions_dict = json.load(f)
else:
    captions_dict = {
        "adult_scene": ["Explicit adult scene detected"],
        "intimate_pose": ["Intimate pose detected"],
        "suggestive": ["Suggestive content detected"],
        "default": ["Highlight frame"]
    }

def get_caption(tags):
    for tag in tags:
        if tag in captions_dict:
            return random.choice(captions_dict[tag])
    return random.choice(captions_dict["default"])
