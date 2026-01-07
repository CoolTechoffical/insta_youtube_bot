import random

def generate_caption(title: str):
    tags = [
        "#reels", "#shorts", "#viral", "#trending",
        "#instagram", "#youtube", "#explore"
    ]
    random.shuffle(tags)

    caption = (
        f"{title}\n\n"
        "🔥 Trending reel\n"
        "💬 Follow for more\n\n"
        + " ".join(tags[:6])
    )
    return caption
