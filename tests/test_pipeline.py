import time

import numpy as np

from emotion_tracker.config import Config
from emotion_tracker.pipeline import EmotionPipeline


class FakeDetector:
    name = "fake"

    def __init__(self):
        self.box = (200, 150, 120, 140)

    def detect(self, frame):
        return self.box


class FakePredictor:
    def predict(self, face_gray):
        p = np.full(7, 0.02, np.float32)
        p[3] = 0.88
        return p


def make(cfg=None, **kw):
    cfg = cfg or Config(infer_every=2, ema_alpha=1.0, **kw)
    pipe = EmotionPipeline(cfg, FakeDetector(), FakePredictor())
    pipe.start()
    return pipe


def frame():
    return np.zeros((480, 640, 3), np.uint8)


def test_inference_runs_every_n_frames_and_produces_label():
    pipe = make()
    try:
        results, t = [], 0.0
        for _ in range(60):
            t += 0.03
            results.append(pipe.process(frame(), t))
            time.sleep(0.005)
        assert any(r.new_inference for r in results)
        assert results[-1].label == 3
        assert results[-1].box is not None
    finally:
        pipe.close()


def test_no_inference_before_first_n_frames():
    pipe = make(Config(infer_every=100, ema_alpha=1.0))
    try:
        r = pipe.process(frame(), 0.1)
        assert not r.new_inference and r.label is None and r.box is not None
    finally:
        pipe.close()


def test_state_resets_after_face_lost_timeout():
    pipe = make()
    try:
        t = 0.0
        for _ in range(60):
            t += 0.03
            pipe.process(frame(), t)
            time.sleep(0.005)
        assert pipe.smoother.has_state
        pipe.detector.box = None
        short = pipe.process(frame(), t + 0.5)           # chưa quá timeout 1s -> giữ trạng thái
        assert not short.reset and short.label == 3
        late = pipe.process(frame(), t + 2.0)            # quá timeout -> xoá
        assert late.reset and late.label is None and late.box is None and late.changed
        again = pipe.process(frame(), t + 2.1)           # chỉ báo reset đúng một lần
        assert not again.reset
    finally:
        pipe.close()
