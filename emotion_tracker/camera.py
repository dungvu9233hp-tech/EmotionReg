"""Đọc webcam, cắt về tỉ lệ 4:3 và lật gương."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int, size: Tuple[int, int] = (640, 480)):
        self.index = index
        self.size = size
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY   # DSHOW mở nhanh hơn trên Windows
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self._cap = cap
        return True

    def read(self) -> Optional[np.ndarray]:
        """Trả về khung hình BGR đã chuẩn hoá kích thước và lật gương, hoặc None nếu đọc lỗi."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return cv2.flip(self.fit(frame, self.size), 1)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @staticmethod
    def fit(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Cắt giữa về đúng tỉ lệ đích (không kéo giãn) rồi resize."""
        tw, th = size
        h, w = frame.shape[:2]
        want_w = int(h * tw / th)
        if want_w < w:                      # ảnh quá rộng (vd 16:9) -> cắt hai bên
            x0 = (w - want_w) // 2
            frame = frame[:, x0:x0 + want_w]
        elif want_w > w:                    # ảnh quá cao -> cắt trên dưới
            want_h = int(w * th / tw)
            y0 = (h - want_h) // 2
            frame = frame[y0:y0 + want_h, :]
        if frame.shape[1] != tw or frame.shape[0] != th:
            frame = cv2.resize(frame, (tw, th))
        return frame
