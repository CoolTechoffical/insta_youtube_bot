import json
import random

with open("captions.json", "r") as f:
    CAPTIONS = json.load(f)

def generate_caption(tags):
    for tag in tags:
        if tag in CAPTIONS:
            return random.choice(CAPTIONS[tag])
    return random.choice(CAPTIONS["default"])
