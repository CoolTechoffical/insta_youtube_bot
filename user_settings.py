# user_settings.py
user_settings = {}

DEFAULT_COUNT = 25
MAX_LIMIT = 200

def get_count(user_id: int) -> int:
    if user_id not in _user_settings:
        _user_settings[user_id] = DEFAULT_COUNT
    return _user_settings[user_id]

def set_count(user_id: int, count: int):
    if count < 1:
        count = 1
    if count > MAX_LIMIT:
        count = MAX_LIMIT
    _user_settings[user_id] = count
