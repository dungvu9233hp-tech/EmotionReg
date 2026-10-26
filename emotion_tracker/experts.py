"""Hai chuyên gia bổ sung, đều trả về vector xác suất 4 lớp theo thứ tự labels.CLASSES
(happy, surprise, neutral, uncomfortable):

- HseEmotion          : CNN HSEmotion (8 cảm xúc AffectNet/AFEW) gộp về 4 lớp.
- MediaPipeExpression : 52 blendshape của MediaPipe Face Landmarker -> điểm 4 lớp theo luật.

Cả hai nhận ảnh khuôn mặt BGR (np.uint8, HxWx3).
"""
from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .labels import NUM_CLASSES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- HSEmotion
# Mỗi hàng: (happy, surprise, neutral, uncomfortable), tổng = 1.
# Anger/Contempt/Sadness là những nhãn hay bị kích hoạt bởi "nhíu mày khi tập trung",
# nên chỉ chuyển MỘT PHẦN sang "uncomfortable", phần còn lại về "neutral".
HSE_TO_4 = {
    "happiness": (1.0, 0.0, 0.0, 0.0),
    "surprise":  (0.0, 1.0, 0.0, 0.0),
    "neutral":   (0.0, 0.0, 1.0, 0.0),
    "anger":     (0.0, 0.0, 0.4, 0.6),
    "disgust":   (0.0, 0.0, 0.1, 0.9),
    "sadness":   (0.0, 0.0, 0.3, 0.7),
    "fear":      (0.0, 0.4, 0.0, 0.6),
    "contempt":  (0.0, 0.0, 0.5, 0.5),
}


class HseEmotion:
    def __init__(self, model_name: str = "enet_b0_8_best_afew"):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer  # pip install hsemotion-onnx
        self._fer = HSEmotionRecognizer(model_name=model_name)
        idx_to_class = self._fer.idx_to_class
        self._map = np.zeros((len(idx_to_class), NUM_CLASSES), dtype=np.float64)
        for i, name in idx_to_class.items():
            row = HSE_TO_4.get(str(name).lower())
            if row is None:
                raise RuntimeError(f"HSEmotion có nhãn lạ '{name}', hãy thêm vào HSE_TO_4")
            self._map[int(i)] = row
        self.predict(np.zeros((224, 224, 3), np.uint8))   # khởi động

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)    # HSEmotion cần ảnh RGB
        _, logits = self._fer.predict_emotions(rgb, logits=True)
        z = np.asarray(logits, dtype=np.float64).ravel()
        e = np.exp(z - z.max())
        p4 = (e / e.sum()) @ self._map
        return (p4 / p4.sum()).astype(np.float32)


# ---------------------------------------------------------------- MediaPipe
LANDMARKER_FILE = "face_landmarker.task"
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/" + LANDMARKER_FILE)
_MIN_VALID_BYTES = 1_000_000


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


class MediaPipeExpression:
    """Blendshape -> xác suất 4 lớp. Có hiệu chuẩn: trừ nét mặt thả lỏng của chính người dùng,
    vì mỗi người có lông mày/miệng "mặc định" khác nhau."""

    def __init__(self, models_dir: Path, calibration_sec: float = 3.0):
        import mediapipe as mp                                   # pip install mediapipe
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = Path(models_dir) / LANDMARKER_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            logger.info("Đang tải MediaPipe Face Landmarker (~4 MB)...")
            part = path.with_suffix(".part")
            urllib.request.urlretrieve(LANDMARKER_URL, part)
            part.replace(path)
        if path.stat().st_size < _MIN_VALID_BYTES:
            path.unlink()
            raise RuntimeError("file face_landmarker.task tải về không hợp lệ")

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._calib_sec = calibration_sec
        self._baseline: Dict[str, float] = {}
        self._samples: list = []
        self._t0: Optional[float] = None
        self.calibrating = False

    # --- hiệu chuẩn ---
    def calibrate(self) -> None:
        """Bắt đầu đo nét mặt thả lỏng. Trong lúc đo, chuyên gia này tạm không bỏ phiếu."""
        self._samples, self._t0, self.calibrating = [], None, True

    # --- dự đoán ---
    def predict(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        bs = self._blendshapes(face_bgr)
        if bs is None:
            return None
        if self.calibrating:
            now = time.time()
            self._t0 = self._t0 or now
            self._samples.append(bs)
            if now - self._t0 >= self._calib_sec and len(self._samples) >= 8:
                self._baseline = {k: float(np.median([s.get(k, 0.0) for s in self._samples]))
                                  for k in self._samples[0]}
                self.calibrating = False
            return None
        return self._scores(bs)

    def _blendshapes(self, face_bgr: np.ndarray) -> Optional[Dict[str, float]]:
        rgb = cv2.cvtColor(cv2.resize(face_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(image)
        if not result.face_blendshapes:
            return None
        return {c.category_name: float(c.score) for c in result.face_blendshapes[0]}

    def _scores(self, bs: Dict[str, float]) -> np.ndarray:
        def g(*names: str) -> float:   # trung bình các blendshape sau khi trừ baseline
            return float(np.mean([max(bs.get(n, 0.0) - self._baseline.get(n, 0.0), 0.0) for n in names]))

        smile = g("mouthSmileLeft", "mouthSmileRight")
        brow_up = max(g("browInnerUp"), g("browOuterUpLeft", "browOuterUpRight"))
        opened = max(g("jawOpen"), g("eyeWideLeft", "eyeWideRight"))
        brow_down = g("browDownLeft", "browDownRight")
        nose = g("noseSneerLeft", "noseSneerRight")
        mouth = g("mouthPressLeft", "mouthPressRight", "mouthFrownLeft", "mouthFrownRight")
        squint = g("eyeSquintLeft", "eyeSquintRight")

        s_happy = _clip01(1.5 * smile)
        s_surprise = _clip01(2.0 * np.sqrt(brow_up * opened))     # cần CẢ nhướng mày LẪN mở mắt/miệng
        # nhíu mày chỉ chiếm 25%: nhíu mày đơn lẻ không bao giờ vượt ~0.4, cần thêm mũi/miệng/mắt
        s_unc = _clip01(2.5 * (0.25 * brow_down + 0.25 * nose + 0.30 * mouth + 0.20 * squint))
        s_unc *= 1.0 - 0.7 * s_happy                              # đang cười thì không phải khó chịu
        s_neutral = 1.0 - max(s_happy, s_surprise, s_unc)

        p = np.array([s_happy, s_surprise, s_neutral, s_unc], dtype=np.float64) + 1e-6
        return (p / p.sum()).astype(np.float32)
