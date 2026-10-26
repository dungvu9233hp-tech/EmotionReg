"""Chạy model ở luồng riêng để giao diện không bao giờ bị đứng."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    def predict(self, face_gray: np.ndarray) -> np.ndarray: ...


class InferenceWorker(threading.Thread):
    """Nhận một ảnh mặt mỗi lần; nếu đang bận thì bỏ qua ảnh mới (không xếp hàng)."""

    def __init__(self, predictor: Predictor):
        super().__init__(daemon=True, name="inference-worker")
        self._predictor = predictor
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job: Optional[np.ndarray] = None
        self._result: Optional[np.ndarray] = None
        self._busy = False
        self._alive = True
        self.latency_ms = 0.0

    def submit(self, face_gray: np.ndarray) -> bool:
        """Gửi ảnh để dự đoán. Trả về False nếu worker đang bận hoặc ảnh rỗng."""
        with self._lock:
            if self._busy or face_gray.size == 0:
                return False
            self._job, self._busy = face_gray, True
        self._wake.set()
        return True

    def take(self) -> Optional[np.ndarray]:
        """Lấy kết quả mới nhất (mỗi kết quả chỉ lấy được một lần)."""
        with self._lock:
            result, self._result = self._result, None
            return result

    def stop(self) -> None:
        self._alive = False
        self._wake.set()

    def run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if not self._alive:
                return
            started = time.perf_counter()
            try:
                probs = self._predictor.predict(self._job)  # type: ignore[arg-type]
            except Exception:
                logger.exception("Lỗi suy luận")
                probs = None
            with self._lock:
                self._result = probs
                self.latency_ms = (time.perf_counter() - started) * 1000
                self._busy = False
