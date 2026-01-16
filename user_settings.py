# user_settings.py

USER_LIMITS = {}

def set_count(user_id: int, count: int):
    USER_LIMITS[user_id] = count

def get_count(user_id: int):
    return USER_LIMITS.get(user_id, 30)  # default 30 images
