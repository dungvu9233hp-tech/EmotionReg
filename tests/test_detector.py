import urllib.request

import numpy as np

from emotion_tracker.detector import FaceDetector, crop_face


def test_crop_is_square_and_includes_margin():
    gray = np.zeros((480, 640), np.uint8)
    crop = crop_face(gray, (300, 200, 100, 100), margin=0.1)
    assert crop.shape == (120, 120)


def test_crop_near_borders_keeps_square_shape():
    gray = np.zeros((480, 640), np.uint8)
    for box in [(0, 0, 100, 100), (580, 430, 100, 100), (-20, 10, 80, 80)]:
        h, w = crop_face(gray, box, margin=0.1).shape
        assert h == w and h > 0


def test_detector_falls_back_to_haar_when_download_fails(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("không có mạng")
    monkeypatch.setattr(urllib.request, "urlretrieve", boom)
    det = FaceDetector(tmp_path)
    assert "Haar" in det.name
    assert det.detect(np.zeros((480, 640, 3), np.uint8)) is None
