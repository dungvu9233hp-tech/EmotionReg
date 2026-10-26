import numpy as np
import pytest

from emotion_tracker.smoothing import EmotionSmoother


def onehot(i, n=7):
    v = np.zeros(n, np.float32)
    v[i] = 1
    return v


def test_first_update_sets_state():
    s = EmotionSmoother(0.5, 0.1)
    assert not s.has_state
    s.update(onehot(3))
    assert s.label == 3 and s.has_state


def test_ema_blends_values():
    s = EmotionSmoother(0.5, 0.1)
    s.update(onehot(0))
    s.update(onehot(1))
    assert s.probs[0] == pytest.approx(0.5) and s.probs[1] == pytest.approx(0.5)


def test_hysteresis_blocks_small_changes_and_allows_big_ones():
    s = EmotionSmoother(1.0, 0.2)       # alpha=1: không làm mượt, chỉ xét hysteresis
    s.update(np.array([0.5, 0.45, 0, 0, 0, 0, 0.05], np.float32))
    assert s.label == 0
    s.update(np.array([0.45, 0.50, 0, 0, 0, 0, 0.05], np.float32))   # chênh 0.05 < 0.2
    assert s.label == 0
    s.update(np.array([0.1, 0.85, 0, 0, 0, 0, 0.05], np.float32))    # chênh 0.75 > 0.2
    assert s.label == 1


def test_reset_clears_state():
    s = EmotionSmoother(0.5, 0.1)
    s.update(onehot(2))
    s.reset()
    assert s.probs is None and s.label is None and not s.has_state


@pytest.mark.parametrize("alpha", [0, -0.1, 1.5])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        EmotionSmoother(alpha, 0.1)
