# caption_engine.py
import json
import random
import os

CAPTION_FILE = "captions.json"

if not os.path.exists(CAPTION_FILE):
    raise FileNotFoundError("captions.json not found")

with open(CAPTION_FILE, "r", encoding="utf-8") as f:
    CAPTIONS = json.load(f)


def get_caption(tags: list[str]) -> str:
    """
    Returns caption ONLY if tag matches detected content
    """
    matched = []

    for tag in tags:
        if tag in CAPTIONS:
            matched.extend(CAPTIONS[tag])

    if not matched:
        return random.choice(
            CAPTIONS.get("safe", ["Scene detected"])
        )

    return random.choice(matched)
