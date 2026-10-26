"""Ghép một khung hình hoàn chỉnh: game ở bên trái, camera + biểu đồ cảm xúc thu nhỏ ở bên phải.

Cùng một ảnh này vừa hiển thị trên màn hình vừa được ghi vào video -> bác sĩ thấy đúng những gì người chơi thấy.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..labels import COLORS, LABELS_VI, NUM_CLASSES, UNCOMFORTABLE
from .game import GAME_H, GAME_W
from .textdraw import TextItem, bgr, draw_texts

SIDE_W = 320
CAM_H = 240
TOTAL_W = GAME_W + SIDE_W
_SIDE_BG = "#1e1e1e"


class Compositor:
    def __init__(self, history_sec: int = 40):
        self._span = history_sec
        self._history: deque = deque()

    @property
    def size(self) -> Tuple[int, int]:
        return TOTAL_W, GAME_H

    # ---------- lịch sử cho biểu đồ ----------
    def add(self, t: float, probs: np.ndarray) -> None:
        self._history.append((t, np.asarray(probs, dtype=np.float32).copy()))
        while self._history and t - self._history[0][0] > self._span:
            self._history.popleft()

    def clear(self) -> None:
        self._history.clear()

    # ---------- ghép ảnh ----------
    def compose(self, game_img: np.ndarray, game_texts: List[TextItem], cam_frame: Optional[np.ndarray],
                result, now: float, rec_sec: Optional[float]) -> np.ndarray:
        canvas = np.empty((GAME_H, TOTAL_W, 3), np.uint8)
        canvas[:, :GAME_W] = game_img
        canvas[:, GAME_W:] = bgr(_SIDE_BG)
        texts = list(game_texts)
        label = None if result is None else result.label
        probs = None if result is None else result.probs

        # camera thu nhỏ + khung mặt
        if cam_frame is not None:
            canvas[0:CAM_H, GAME_W:] = cv2.resize(cam_frame, (SIDE_W, CAM_H), interpolation=cv2.INTER_AREA)
            if result is not None and result.box is not None:
                k = SIDE_W / cam_frame.shape[1]
                x, y, w, h = (v * k for v in result.box)
                color = COLORS[label] if label is not None else "#00ff88"
                cv2.rectangle(canvas, (GAME_W + int(x), int(y)), (GAME_W + int(x + w), int(y + h)), bgr(color), 2)
        else:
            texts.append(TextItem(GAME_W + SIDE_W // 2, CAM_H // 2, "Chưa bật camera", 14, "#777777", False, "mm"))

        if rec_sec is not None:
            cv2.circle(canvas, (GAME_W + 16, 16), 6, bgr("#e74c3c"), -1, cv2.LINE_AA)
            texts.append(TextItem(GAME_W + 28, 16, f"REC {int(rec_sec // 60):02d}:{int(rec_sec % 60):02d}",
                                  12, "#ffffff", True, "lm"))

        # nhãn lớn
        if label is not None:
            texts.append(TextItem(GAME_W + SIDE_W // 2, 266, LABELS_VI[label], 22, COLORS[label], True, "mm"))
        else:
            msg = "…" if (result is not None and result.box is not None) else "Không thấy mặt"
            texts.append(TextItem(GAME_W + SIDE_W // 2, 266, msg, 18, "#888888", False, "mm"))

        self._draw_bars(canvas, texts, probs, label)
        self._draw_timeline(canvas, texts, now)

        # viền đỏ quanh game khi đang "không thoải mái" (dễ thấy khi tua video)
        if label == UNCOMFORTABLE:
            cv2.rectangle(canvas, (3, 3), (GAME_W - 4, GAME_H - 4), bgr("#e74c3c"), 6)
            texts.append(TextItem(GAME_W // 2, GAME_H - 22, "Phát hiện: Không thoải mái", 18, "#e74c3c", True, "mm"))
        return draw_texts(canvas, texts)

    def _draw_bars(self, canvas, texts, probs, highlight) -> None:
        x_name, x_bar, bar_w = GAME_W + 8, GAME_W + 112, 140
        for i in range(NUM_CLASSES):
            y = 296 + i * 26
            p = 0.0 if probs is None else float(probs[i])
            cv2.rectangle(canvas, (x_bar, y - 7), (x_bar + bar_w, y + 7), bgr("#3a3a3a"), -1)
            cv2.rectangle(canvas, (x_bar, y - 7), (x_bar + int(bar_w * p), y + 7), bgr(COLORS[i]), -1)
            texts.append(TextItem(x_name, y, LABELS_VI[i], 11, "#ffffff", i == highlight, "lm"))
            texts.append(TextItem(GAME_W + SIDE_W - 6, y, f"{p * 100:.0f}%", 11, "#cccccc", False, "rm"))

    def _draw_timeline(self, canvas, texts, now: float) -> None:
        x0, x1, y0, y1 = GAME_W + 30, TOTAL_W - 8, 440, 585
        for v in (0, 0.5, 1):
            y = int(y1 - (y1 - y0) * v)
            cv2.line(canvas, (x0, y), (x1, y), bgr("#444444"), 1)
            texts.append(TextItem(x0 - 4, y, f"{int(v * 100)}%", 9, "#aaaaaa", False, "rm"))
        for k, off in zip(range(NUM_CLASSES), (0, 44, 118, 190)):      # chú thích
            texts.append(TextItem(GAME_W + 8 + off, 424, "● " + LABELS_VI[k], 10, COLORS[k], False, "lm"))
        for k in range(NUM_CLASSES):
            for seg in self._segments(k, now, x0, x1, y0, y1):
                if len(seg) >= 2:
                    cv2.polylines(canvas, [np.array(seg, np.int32).reshape(-1, 1, 2)], False,
                                  bgr(COLORS[k]), 2, cv2.LINE_AA)

    def _segments(self, k, now, x0, x1, y0, y1):
        segs, pts, prev = [], [], None
        for t, p in self._history:
            x = x0 + (x1 - x0) * (1 - (now - t) / self._span)
            y = y1 - (y1 - y0) * float(p[k])
            if prev is not None and t - prev > 1.0:      # mất mặt >1 giây -> ngắt đường
                segs.append(pts)
                pts = []
            pts.append((int(x), int(y)))
            prev = t
        segs.append(pts)
        return segs
