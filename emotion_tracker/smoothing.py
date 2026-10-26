"""Làm mượt xác suất theo thời gian và chọn nhãn hiển thị ổn định (thuần logic, dễ test)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class EmotionSmoother:
    """EMA trên vector xác suất + trễ (hysteresis) khi đổi nhãn để nhãn không nhảy liên tục.

    `extra_margin` cho phép đòi biên lớn hơn khi chuyển SANG một lớp cụ thể
    (vd. {UNCOMFORTABLE: 0.10} -> lớp "không thoải mái" khó được chọn hơn).
    """

    def __init__(self, alpha: float, switch_margin: float, extra_margin: Optional[Dict[int, float]] = None):
        if not 0 < alpha <= 1:
            raise ValueError("alpha phải trong (0, 1]")
        self.alpha = alpha
        self.switch_margin = switch_margin
        self.extra_margin = dict(extra_margin or {})
        self.probs: Optional[np.ndarray] = None
        self.label: Optional[int] = None

    def update(self, probs: np.ndarray) -> None:
        probs = np.asarray(probs, dtype=np.float32)
        self.probs = probs if self.probs is None else self.alpha * probs + (1 - self.alpha) * self.probs
        top = int(self.probs.argmax())
        if self.label is None:
            self.label = top
        elif top != self.label:
            margin = self.switch_margin + self.extra_margin.get(top, 0.0)
            if self.probs[top] > self.probs[self.label] + margin:
                self.label = top

    def reset(self) -> None:
        self.probs = None
        self.label = None

    @property
    def has_state(self) -> bool:
        return self.probs is not None
