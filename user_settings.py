user_settings = {}

def get_user(user_id):
    if user_id not in user_settings:
        user_settings[user_id] = {
            "count": 100,
            "nsfw": True
        }
    return user_settings[user_id]
