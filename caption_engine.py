import json
import random

with open("captions.json", "r", encoding="utf-8") as f:
    CAPTIONS = json.load(f)

PRIORITY = [
    # 🔴 Highest severity / explicit
    "adult_scene",
    "intercourse",
    "vaginal_sex",
    "anal",
    "threesome",

    # 🔥 Oral & active sexual acts
    "oral_play",
    "blowjob",
    "pussy_eating",
    "handjob",
    "erotic_touch",

    # 💦 Fluids / climax
    "hot_cum",
    "cums",
    "squirting",

    # 🍒 Breast-focused
    "breast_sex",
    "drinking_breastmilk",
    "hot_drinking_breastmilk",
    "breast_kissing",
    "big_boobs",
    "tits",
    "breasts",

    # 💋 Romance / intimacy
    "kissing",
    "lips",
    "intimate_pose",

    # 👀 Exposure / body
    "upper_body_exposed",
    "lower_body_exposed",
    "butts",
    "nudity_detected",

    # 🎞 Motion / action
    "intense_motion",
    "slow_motion",

    # 🌶 Mild / suggestive
    "suggestive",

    # ✅ Safe fallback
    "safe"
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
