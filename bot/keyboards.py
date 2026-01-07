from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def quality_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("480p", callback_data="q:480"),
            InlineKeyboardButton("720p", callback_data="q:720"),
        ],
        [
            InlineKeyboardButton("1080p", callback_data="q:1080"),
            InlineKeyboardButton("2K", callback_data="q:1440"),
        ]
    ])
