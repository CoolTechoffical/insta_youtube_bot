user_limits = {}

def set_count(uid, count):
    user_limits[uid] = count

def get_count(uid):
    return user_limits.get(uid, 30)
