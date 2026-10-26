"""Phát hiện khuôn mặt và cắt vùng mặt đưa vào model."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # (x, y, w, h)

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/" + YUNET_FILE)
_MIN_VALID_BYTES = 100_000   # file nhỏ hơn mức này là lỗi tải / con trỏ Git LFS
_DETECT_WIDTH = 320          # chạy bộ dò trên ảnh thu nhỏ cho nhanh


class FaceDetector:
    """YuNet (nhanh, chính xác). Tự tải lần đầu; nếu thất bại thì dùng Haar có sẵn của OpenCV."""

    def __init__(self, models_dir: Path, score_threshold: float = 0.7):
        self.name = "YuNet"
        self._yunet = None
        self._haar = None
        try:
            self._yunet = self._load_yunet(Path(models_dir) / YUNET_FILE, score_threshold)
        except Exception as exc:  # mạng lỗi, file hỏng, OpenCV quá cũ...
            logger.warning("Không dùng được YuNet (%s), chuyển sang Haar", exc)
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.name = "Haar (dự phòng)"

    @staticmethod
    def _load_yunet(path: Path, score_threshold: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            logger.info("Đang tải bộ dò khuôn mặt YuNet (~230 KB)...")
            part = path.with_suffix(".part")
            urllib.request.urlretrieve(YUNET_URL, part)
            part.replace(path)
        if path.stat().st_size < _MIN_VALID_BYTES:
            path.unlink()
            raise RuntimeError("file YuNet tải về không hợp lệ")
        return cv2.FaceDetectorYN.create(str(path), "", (_DETECT_WIDTH, 240), score_threshold, 0.3, 5000)

    def detect(self, frame_bgr: np.ndarray) -> Optional[Box]:
        """Trả về khuôn mặt lớn nhất dạng (x, y, w, h) trên ảnh gốc, hoặc None."""
        if self._yunet is not None:
            return self._detect_yunet(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _detect_yunet(self, frame: np.ndarray) -> Optional[Box]:
        h, w = frame.shape[:2]
        scale = _DETECT_WIDTH / w
        small = cv2.resize(frame, (_DETECT_WIDTH, int(round(h * scale))))
        self._yunet.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._yunet.detect(small)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda r: r[2] * r[3])
        x, y, bw, bh = best[:4] / scale
        return int(x), int(y), int(bw), int(bh)

    def _detect_haar(self, frame: np.ndarray) -> Optional[Box]:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self._haar.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        return tuple(int(v) for v in max(faces, key=lambda r: r[2] * r[3]))  # type: ignore[return-value]


def crop_face(gray: np.ndarray, box, margin: float = 0.10) -> np.ndarray:
    """Cắt ô vuông quanh khuôn mặt (nới thêm `margin`, đệm biên nếu tràn ra ngoài ảnh)."""
    x, y, w, h = box
    cx, cy, side = x + w / 2, y + h / 2, max(w, h) * (1 + 2 * margin)
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    x1, y1 = int(cx + side / 2), int(cy + side / 2)
    height, width = gray.shape[:2]   # nhận cả ảnh xám lẫn ảnh màu
    pad = max(0, -x0, -y0, x1 - width, y1 - height)
    if pad:
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return gray[y0:y1, x0:x1]
