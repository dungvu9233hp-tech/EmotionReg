import csv

import numpy as np
import pytest

from emotion_tracker.session import SessionRecorder


def probs(i):
    p = np.zeros(7, np.float32)
    p[i] = 1
    return p


def test_writes_csv_and_computes_shares(tmp_path):
    rec = SessionRecorder(tmp_path, interval_sec=1.0)
    rec.start(write_csv=True)
    for t in (0.0, 1.0, 2.0, 3.0):                  # 3 giây "vui"
        rec.update(t, 3, probs(3))
    rec.update(4.0, 4, probs(4))
    rec.update(5.0, 4, probs(4))                    # 2 giây "buồn" (tính theo khoảng tới lần ghi trước)
    rec.update(6.0, None, None)                     # không thấy mặt
    summary = rec.stop()

    assert summary.shares is not None
    assert summary.shares[3] == pytest.approx(3 / 5)
    assert summary.shares[4] == pytest.approx(2 / 5)
    rows = list(csv.reader(open(summary.csv_path, encoding="utf-8")))
    assert rows[0][:3] == ["time", "face", "label"] and len(rows) == 8
    assert rows[1][1:3] == ["1", "happy"] and rows[-1][1] == "0"


def test_respects_interval(tmp_path):
    rec = SessionRecorder(tmp_path, interval_sec=1.0)
    rec.start(write_csv=True)
    for t in np.arange(0, 3.0, 0.1):
        rec.update(float(t), 0, probs(0))
    summary = rec.stop()
    assert len(list(csv.reader(open(summary.csv_path, encoding="utf-8")))) == 1 + 3


def test_short_session_has_no_summary_and_no_csv_when_disabled(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.start(write_csv=False)
    rec.update(0.0, 1, probs(1))
    summary = rec.stop()
    assert summary.shares is None and summary.csv_path is None
    assert not list(tmp_path.glob("*.csv"))
