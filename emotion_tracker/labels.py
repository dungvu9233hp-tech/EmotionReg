"""Nhãn, tên hiển thị và màu của 4 lớp cảm xúc.

Thứ tự phải trùng với lúc train (train.py --scheme 4): happy, surprise, neutral, uncomfortable.
"""

CLASSES = ("happy", "surprise", "neutral", "uncomfortable")
LABELS_VI = ("Vui", "Ngạc nhiên", "Bình thường", "Không thoải mái")
COLORS = ("#f1c40f", "#e67e22", "#95a5a6", "#8e44ad")
NUM_CLASSES = len(CLASSES)
NEUTRAL = CLASSES.index("neutral")
UNCOMFORTABLE = CLASSES.index("uncomfortable")