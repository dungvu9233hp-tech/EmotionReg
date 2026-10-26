import numpy as np
import pytest

from emotion_tracker.classifier import EmotionClassifier, ModelLoadError, find_model_file


def test_predict_returns_probability_vector(fake_model):
    clf = EmotionClassifier(fake_model)
    assert clf.size == 112 and clf.channels == 1
    p = clf.predict((np.random.rand(200, 180) * 255).astype(np.uint8))
    assert p.shape == (7,) and p.dtype == np.float32
    assert p.sum() == pytest.approx(1.0, abs=1e-4)


def test_three_channel_model_is_supported(fake_model_3ch):
    clf = EmotionClassifier(fake_model_3ch)
    assert clf.channels == 3
    assert clf.predict(np.full((64, 64), 128, np.uint8)).sum() == pytest.approx(1.0, abs=1e-4)


def test_missing_external_data_gives_helpful_error(model_missing_data):
    with pytest.raises(ModelLoadError, match=r"\.onnx\.data"):
        EmotionClassifier(model_missing_data)


def test_find_model_prefers_fp_and_skips_yunet(tmp_path):
    for name in ("a.onnx", "b_fp_best.onnx", "face_detection_yunet_2023mar.onnx"):
        (tmp_path / name).write_bytes(b"x")
    assert find_model_file(tmp_path).name == "b_fp_best.onnx"


def test_find_model_errors(tmp_path):
    with pytest.raises(ModelLoadError):
        find_model_file(tmp_path)
    with pytest.raises(ModelLoadError):
        find_model_file(tmp_path, tmp_path / "khong_co.onnx")
