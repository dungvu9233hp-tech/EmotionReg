"""Vẽ chữ tiếng Việt có dấu lên ảnh OpenCV (cv2.putText không hỗ trợ dấu) bằng Pillow."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REGULAR = ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
_BOLD = ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")


@dataclass(frozen=True)
class TextItem:
    x: int
    y: int
    text: str
    size: int = 16
    color: str = "#ffffff"
    bold: bool = False
    anchor: str = "lm"     # l/m/r (ngang) + m/s (dọc), theo quy ước của Pillow


def bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


@lru_cache(maxsize=64)
def _font(size: int, bold: bool):
    for path in (_BOLD if bold else _REGULAR):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def draw_texts(img_bgr: np.ndarray, items: List[TextItem]) -> np.ndarray:
    """Vẽ toàn bộ chữ trong một lần chuyển đổi ảnh (nhanh hơn vẽ từng chữ)."""
    if not items:
        return img_bgr
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for it in items:
        font = _font(it.size, it.bold)
        try:
            draw.text((it.x, it.y), it.text, font=font, fill=it.color, anchor=it.anchor)
        except ValueError:                      # font bitmap dự phòng không có anchor
            draw.text((it.x, it.y), it.text, font=font, fill=it.color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
