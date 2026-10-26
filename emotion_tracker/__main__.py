"""Điểm vào của chương trình.

Chạy từ thư mục chứa package (thư mục cha của package này):
    python -m <tên_package>
    python -m <tên_package> --model models/xxx.onnx --cam 0
"""
from __future__ import annotations

import logging
import sys

from .classifier import EmotionClassifier, ModelLoadError, find_model_file
from .config import Config
from .detector import FaceDetector
from .pipeline import EmotionPipeline
from .ui.app import run


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.from_args(argv)
    try:
        classifier = EmotionClassifier(find_model_file(cfg.models_dir, cfg.model_path))
    except ModelLoadError as exc:
        print(exc, file=sys.stderr)
        return 1
    detector = FaceDetector(cfg.models_dir, cfg.detector_score)
    pipeline = EmotionPipeline(cfg, detector, classifier)
    return run(cfg, pipeline)


if __name__ == "__main__":
    raise SystemExit(main())