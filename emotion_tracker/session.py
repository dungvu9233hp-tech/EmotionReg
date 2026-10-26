"""Ghi log phiên theo dõi (CSV) và tính tỉ lệ thời gian theo từng biểu cảm."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .labels import CLASSES, NUM_CLASSES

_MIN_TRACKED_SEC = 1.0


@dataclass(frozen=True)
class SessionSummary:
    shares: Optional[np.ndarray]   # tỉ lệ thời gian mỗi lớp (tổng = 1); None nếu phiên quá ngắn
    csv_path: Optional[Path]


class SessionRecorder:
    def __init__(self, logs_dir: Path, interval_sec: float = 1.0):
        self._logs_dir = Path(logs_dir)
        self._interval = interval_sec
        self._active = False
        self._dwell = np.zeros(NUM_CLASSES)
        self._last: Optional[float] = None
        self._file = None
        self._writer = None
        self._path: Optional[Path] = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self, write_csv: bool = True) -> None:
        self._dwell[:] = 0
        self._last = None
        self._path = None
        if write_csv:
            self._logs_dir.mkdir(parents=True, exist_ok=True)
            self._path = self._logs_dir / datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
            self._file = open(self._path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(["time", "face", "label", *CLASSES])
        self._active = True

    def update(self, now: float, label: Optional[int], probs: Optional[np.ndarray]) -> None:
        """Gọi mỗi khung hình; chỉ thực sự ghi mỗi `interval_sec` giây."""
        if not self._active:
            return
        if self._last is not None and now - self._last < self._interval:
            return
        elapsed = 0.0 if self._last is None else now - self._last
        self._last = now
        has_face = label is not None and probs is not None
        if has_face:
            self._dwell[label] += elapsed
        if self._writer:
            ts = datetime.now().isoformat(timespec="seconds")
            if has_face:
                self._writer.writerow([ts, 1, CLASSES[label], *(f"{v:.4f}" for v in probs)])
            else:
                self._writer.writerow([ts, 0, "", *([""] * NUM_CLASSES)])
            self._file.flush()

    def stop(self) -> SessionSummary:
        was_active, self._active = self._active, False
        if self._file:
            self._file.close()
            self._file = self._writer = None
        total = float(self._dwell.sum())
        shares = self._dwell / total if (was_active and total >= _MIN_TRACKED_SEC) else None
        return SessionSummary(shares, self._path)
