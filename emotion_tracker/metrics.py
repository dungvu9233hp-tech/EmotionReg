"""Đo FPS và tốc độ suy luận để hiển thị."""
from __future__ import annotations

import time


class FpsMeter:
    """FPS làm mượt bằng EMA."""

    def __init__(self, smoothing: float = 0.9):
        self._s = smoothing
        self._last = time.perf_counter()
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        inst = 1.0 / max(1e-6, now - self._last)
        self._last = now
        self.fps = inst if self.fps == 0 else self._s * self.fps + (1 - self._s) * inst
        return self.fps


class RateCounter:
    """Đếm số sự kiện/giây trong cửa sổ trượt (cập nhật mỗi `window` giây)."""

    def __init__(self, window: float = 0.5):
        self._window = window
        self._count = 0
        self._t0 = time.time()
        self.rate = 0.0

    def add(self, n: int = 1) -> bool:
        """Cộng sự kiện; trả về True nếu `rate` vừa được cập nhật."""
        self._count += n
        now = time.time()
        if now - self._t0 >= self._window:
            self.rate = self._count / (now - self._t0)
            self._count, self._t0 = 0, now
            return True
        return False
