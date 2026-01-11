user_settings = {}

def set_count(user_id, count):
    user_settings[user_id] = count

def get_count(user_id):
    return user_settings.get(user_id, 100)  # default 100
