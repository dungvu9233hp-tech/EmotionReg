import numpy as np
import pytest

from emotion_tracker.camera import Camera
from emotion_tracker.config import Config


def test_fit_crops_widescreen_without_stretching():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, :160] = 255                             # 720*4/3 = 960 -> cắt bỏ 160 cột mỗi bên
    frame[:, -160:] = 255
    out = Camera.fit(frame, (640, 480))
    assert out.shape == (480, 640, 3)
    assert out.max() == 0                            # hai dải bị cắt không còn trong kết quả


def test_fit_keeps_target_size_for_tall_and_exact_frames():
    assert Camera.fit(np.zeros((800, 400, 3), np.uint8), (640, 480)).shape == (480, 640, 3)
    assert Camera.fit(np.zeros((480, 640, 3), np.uint8), (640, 480)).shape == (480, 640, 3)


def test_config_defaults_and_cli():
    cfg = Config.from_args(["--every", "5", "--cam", "1", "--alpha", "0.5"])
    assert (cfg.infer_every, cfg.camera_index, cfg.ema_alpha) == (5, 1, 0.5)


@pytest.mark.parametrize("args", [["--every", "0"], ["--alpha", "0"], ["--alpha", "2"]])
def test_config_rejects_invalid_values(args):
    with pytest.raises(SystemExit):
        Config.from_args(args)
