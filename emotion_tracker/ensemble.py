"""Kết hợp model của bạn + HSEmotion + MediaPipe thành một Predictor duy nhất.

Cách làm:
1. Trung bình có trọng số các vector xác suất (nguồn nào lỗi/không thấy mặt thì bị loại, trọng số chia lại).
2. Cổng đồng thuận cho "không thoải mái": nếu chỉ 1 nguồn cho lớp này >= ngưỡng, giảm xác suất của nó
   và trả phần bị trừ về "neutral". Đây là chỗ chặn lỗi "nhíu mày một tí là khó chịu".
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .labels import NEUTRAL, UNCOMFORTABLE

logger = logging.getLogger(__name__)


class _GrayAdapter:
    """Model của bạn nhận ảnh xám; pipeline gửi ảnh màu."""

    def __init__(self, clf):
        self._clf = clf

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        gray = face_bgr if face_bgr.ndim == 2 else cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return self._clf.predict(gray)


class EnsemblePredictor:
    def __init__(self, cfg: Config, own, hse=None, mediapipe=None):
        self._cfg = cfg
        self.mediapipe = mediapipe
        self._experts = [("model", _GrayAdapter(own), cfg.w_model)]
        if hse is not None:
            self._experts.append(("hse", hse, cfg.w_hse))
        if mediapipe is not None:
            self._experts.append(("mp", mediapipe, cfg.w_mediapipe))
        self.debug = ""        # chuỗi hiển thị trên giao diện để dễ chỉnh trọng số

    def calibrate(self) -> None:
        if self.mediapipe is not None:
            self.mediapipe.calibrate()

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        parts: Dict[str, Tuple[np.ndarray, float]] = {}
        for name, expert, weight in self._experts:
            try:
                p = expert.predict(face_bgr)
            except Exception:
                logger.exception("Chuyên gia '%s' lỗi, bỏ qua khung này", name)
                p = None
            if p is not None:
                parts[name] = (np.asarray(p, dtype=np.float64), weight)
        if not parts:
            raise RuntimeError("Không có chuyên gia nào trả kết quả")
        fused = self.fuse(parts)
        self._update_debug(parts)
        return fused

    def fuse(self, parts: Dict[str, Tuple[np.ndarray, float]]) -> np.ndarray:
        total_w = sum(w for _, w in parts.values())
        fused = sum(p * (w / total_w) for p, w in parts.values())

        if len(parts) >= 2:
            votes = sum(p[UNCOMFORTABLE] >= self._cfg.unc_vote_threshold for p, _ in parts.values())
            if votes < 2:                                   # chỉ một nguồn "đơn độc" cho là khó chịu
                moved = fused[UNCOMFORTABLE] * (1.0 - self._cfg.unc_lone_damp)
                fused[UNCOMFORTABLE] -= moved
                fused[NEUTRAL] += moved
        return (fused / fused.sum()).astype(np.float32)

    def _update_debug(self, parts) -> None:
        unc = "/".join(f"{parts[n][0][UNCOMFORTABLE] * 100:02.0f}" if n in parts else "--"
                       for n in ("model", "hse", "mp"))
        text = f"K.chịu % M/H/P : {unc}"
        if self.mediapipe is not None and self.mediapipe.calibrating:
            text += "\nMediaPipe đang hiệu chuẩn..."
        self.debug = text


def build_ensemble(cfg: Config) -> EnsemblePredictor:
    """Nạp model của bạn (bắt buộc) và hai chuyên gia phụ (lỗi/thiếu thư viện thì bỏ qua, chỉ cảnh báo)."""
    from .classifier import EmotionClassifier, find_model_file

    own = EmotionClassifier(find_model_file(cfg.models_dir, cfg.model_path))
    hse = mp = None
    if cfg.use_hse:
        try:
            from .experts import HseEmotion
            hse = HseEmotion(cfg.hse_model_name)
        except Exception as exc:
            logger.warning("Không dùng được HSEmotion: %s", exc)
    if cfg.use_mediapipe:
        try:
            from .experts import MediaPipeExpression
            mp = MediaPipeExpression(cfg.models_dir, cfg.calibration_sec)
        except Exception as exc:
            logger.warning("Không dùng được MediaPipe: %s", exc)
    logger.info("Ensemble: model%s%s", " + HSE" if hse else "", " + MediaPipe" if mp else "")
    return EnsemblePredictor(cfg, own, hse, mp)
