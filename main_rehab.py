"""Chạy game rehab: python main_rehab.py  (đặt cạnh main.py, bên ngoài thư mục emotion_tracker/)."""
import logging

from emotion_tracker.config import Config
from emotion_tracker.detector import FaceDetector
from emotion_tracker.ensemble import build_ensemble
from emotion_tracker.pipeline import EmotionPipeline
from emotion_tracker.ui.rehab_app import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = Config.from_args()
    detector = FaceDetector(cfg.models_dir, cfg.detector_score)
    predictor = build_ensemble(cfg)          # model của bạn + HSEmotion + MediaPipe
    pipeline = EmotionPipeline(cfg, detector, predictor)
    run(cfg, pipeline)
