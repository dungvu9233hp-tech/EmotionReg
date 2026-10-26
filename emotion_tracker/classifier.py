"""Nạp model ONNX và dự đoán cảm xúc từ ảnh khuôn mặt xám."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .labels import NUM_CLASSES

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DEFAULT_SIZE = 112


class ModelLoadError(RuntimeError):
    """Không tìm thấy hoặc không nạp được model."""


def find_model_file(models_dir: Path, explicit: Optional[Path] = None) -> Path:
    """Chọn file model: dùng đường dẫn chỉ định, hoặc tìm trong models_dir (ưu tiên tên chứa '_fp')."""
    if explicit is not None:
        if not Path(explicit).is_file():
            raise ModelLoadError(f"Không thấy file model: {explicit}")
        return Path(explicit)
    candidates = sorted(p for p in Path(models_dir).glob("*.onnx") if "yunet" not in p.name.lower())
    if not candidates:
        raise ModelLoadError(f"Không thấy file .onnx cảm xúc trong {models_dir}. Hãy copy file model vào đó.")
    preferred = [p for p in candidates if "_fp" in p.name]
    return (preferred or candidates)[0]


class EmotionClassifier:
    """Bọc một InferenceSession. Tự nhận biết model đã gộp sẵn tiền xử lý (1 kênh) hay chưa (3 kênh)."""

    def __init__(self, path: Path, threads: Optional[int] = None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or min(4, os.cpu_count() or 2)
        try:
            self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ModelLoadError(
                f"Không nạp được model {Path(path).name}: {str(exc)[:200]}\n"
                "Nếu lỗi nhắc tới file '.onnx.data': model được xuất kèm một file trọng số riêng. "
                "Hãy để '<tên model>.onnx.data' cạnh file .onnx (giữ nguyên tên) hoặc gộp thành một file."
            ) from exc
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self.channels = inp.shape[1] if isinstance(inp.shape[1], int) else 1
        self.size = inp.shape[2] if isinstance(inp.shape[2], int) else _DEFAULT_SIZE
        self.predict(np.zeros((self.size, self.size), np.uint8))  # chạy thử để khởi động
        logger.info("Model %s: input %dx%dx%d", Path(path).name, self.channels, self.size, self.size)

    def predict(self, face_gray: np.ndarray) -> np.ndarray:
        """Trả về vector xác suất độ dài NUM_CLASSES (float32)."""
        s = self.size
        interp = cv2.INTER_AREA if face_gray.shape[0] > s else cv2.INTER_CUBIC
        img = cv2.resize(face_gray, (s, s), interpolation=interp).astype(np.float32)
        if self.channels == 1:  # model đã gộp sẵn chuẩn hoá + softmax
            x = img[None, None]
        else:
            x = np.stack([(img / 255.0 - m) / sd for m, sd in zip(_IMAGENET_MEAN, _IMAGENET_STD)])[None]
            x = x.astype(np.float32)
        out = self._session.run(None, {self._input_name: x})[0][0].astype(np.float64)
        if out.shape[0] != NUM_CLASSES:
            raise ValueError(f"Model trả về {out.shape[0]} lớp, mong đợi {NUM_CLASSES}")
        if out.min() < 0 or abs(out.sum() - 1) > 1e-3:  # là logits -> softmax
            e = np.exp(out - out.max())
            out = e / e.sum()
        return out.astype(np.float32)
