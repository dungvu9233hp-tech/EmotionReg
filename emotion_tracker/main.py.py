from emotion_tracker.config import Config
from emotion_tracker.pipeline import EmotionPipeline
from emotion_tracker.ui.app import run

if __name__ == "__main__":
    cfg = Config()
    pipeline = EmotionPipeline(cfg)
    run(cfg, pipeline)