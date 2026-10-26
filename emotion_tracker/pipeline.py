"""Luồng xử lý một khung hình: dò mặt -> cắt -> (cứ N khung) gửi model -> làm mượt.

Không phụ thuộc giao diện nên test được mà không cần camera hay Tkinter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

from .config import Config
from .detector import Box, crop_face
from .labels import UNCOMFORTABLE
from .smoothing import EmotionSmoother
from .worker import InferenceWorker, Predictor


class Detector(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray) -> Optional[Box]: ...


@dataclass(frozen=True)
class FrameResult:
    box: Optional[Tuple[float, float, float, float]]  # khung mặt đã làm mượt, None nếu không có mặt
    probs: Optional[np.ndarray]                        # xác suất đã làm mượt
    label: Optional[int]                               # nhãn hiển thị (đã chống nhảy)
    new_inference: bool                                # khung này vừa có kết quả model mới
    reset: bool                                        # khung này vừa xoá trạng thái vì mất mặt

    @property
    def changed(self) -> bool:
        return self.new_inference or self.reset


class EmotionPipeline:
    def __init__(self, cfg: Config, detector: Detector, predictor: Predictor):
        self.cfg = cfg
        self.detector = detector
        self.infer_every = cfg.infer_every            # có thể đổi khi đang chạy (từ giao diện)
        self.predictor = predictor
        self.smoother = EmotionSmoother(cfg.ema_alpha, cfg.switch_margin, {UNCOMFORTABLE: cfg.unc_extra_margin})
        self._worker = InferenceWorker(predictor)
        self._box: Optional[np.ndarray] = None
        self._last_face_t = float("-inf")
        self._frame_idx = 0

    # --- vòng đời ---
    def start(self) -> None:
        self._worker.start()

    def close(self) -> None:
        self._worker.stop()

    def reset(self) -> None:
        self._box = None
        self.smoother.reset()

    @property
    def latency_ms(self) -> float:
        return self._worker.latency_ms

    # --- xử lý ---
    def process(self, frame_bgr: np.ndarray, now: float) -> FrameResult:
        self._frame_idx += 1
        did_reset = False
        box = self.detector.detect(frame_bgr)

        if box is not None:
            self._last_face_t = now
            new_box = np.asarray(box, dtype=float)
            w = self.cfg.box_smoothing
            self._box = new_box if self._box is None else w * new_box + (1 - w) * self._box
            if self._frame_idx % max(1, self.infer_every) == 0:
                # gửi ảnh MÀU (BGR); mỗi chuyên gia tự đổi sang xám/RGB theo nhu cầu
                self._worker.submit(crop_face(frame_bgr, tuple(self._box), self.cfg.face_margin))
        elif now - self._last_face_t > self.cfg.no_face_timeout and (self._box is not None or self.smoother.has_state):
            self.reset()
            did_reset = True

        new_inference = False
        probs = self._worker.take()
        if probs is not None and self._box is not None:
            self.smoother.update(probs)
            new_inference = True

        box_out = None if self._box is None else tuple(float(v) for v in self._box)
        return FrameResult(box_out, self.smoother.probs, self.smoother.label, new_inference, did_reset)
