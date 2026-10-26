"""Game rehab "Chạm mục tiêu": mục tiêu xuất hiện ở 8 vị trí cố định, mỗi vị trí = một "động tác".

Người chơi dùng chuột / màn hình cảm ứng để chạm. Mỗi lượt được ghi lại (tên động tác, thời điểm,
trúng/trượt, thời gian phản ứng) để đối chiếu với biểu cảm khó chịu trong báo cáo.

Muốn thêm game khác: viết lớp có cùng giao diện (begin, update, click, render, trials, done).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .textdraw import TextItem, bgr

GAME_W, GAME_H = 800, 600
FIELD = (110, 175, 690, 520)          # vùng đặt mục tiêu (x0, y0, x1, y1)
HUD_H = 70
COUNTDOWN_SEC = 4.0                  # đủ thời gian cho MediaPipe hiệu chuẩn (3 giây)
GAP_SEC = 0.8


@dataclass(frozen=True)
class Zone:
    name: str
    fx: float
    fy: float


ZONES = (
    Zone("Với cao - trái", 0.00, 0.00),
    Zone("Với cao - giữa", 0.50, 0.00),
    Zone("Với cao - phải", 1.00, 0.00),
    Zone("Ngang vai - phải", 1.00, 0.50),
    Zone("Với thấp - phải", 1.00, 1.00),
    Zone("Với thấp - trái", 0.00, 1.00),
    Zone("Ngang vai - trái", 0.00, 0.50),
    Zone("Giữa - trước ngực", 0.50, 0.50),
)

# mức độ: (bán kính mục tiêu px, thời gian tối đa mỗi lượt giây)
LEVELS: Dict[str, Tuple[int, float]] = {"Dễ": (62, 6.0), "Vừa": (48, 4.5), "Khó": (38, 3.5)}


class ReachGame:
    def __init__(self, level: str = "Vừa", rounds: int = 3):
        self.level = level if level in LEVELS else "Vừa"
        self.radius, self.timeout = LEVELS[self.level]
        self.sequence: List[Zone] = [z for _ in range(max(1, rounds)) for z in ZONES]
        self.state = "idle"                 # idle -> countdown -> target <-> gap -> done
        self.trials: List[dict] = []
        self.score = 0
        self._index = -1
        self._t_begin = 0.0
        self._t_trial = 0.0
        self._t_gap = 0.0
        self._pos = (0, 0)
        self._last_feedback: Optional[Tuple[str, Tuple[int, int], int]] = None

    # ---------- trạng thái ----------
    @property
    def done(self) -> bool:
        return self.state == "done"

    @property
    def total(self) -> int:
        return len(self.sequence)

    @property
    def movement(self) -> Optional[str]:
        if self.state in ("target", "gap") and 0 <= self._index < self.total:
            return self.sequence[self._index].name
        return None

    def begin(self, now: float) -> None:
        self.state, self._t_begin = "countdown", now

    @staticmethod
    def _zone_pos(z: Zone) -> Tuple[int, int]:
        x0, y0, x1, y1 = FIELD
        return int(x0 + z.fx * (x1 - x0)), int(y0 + z.fy * (y1 - y0))

    # ---------- logic ----------
    def update(self, now: float) -> None:
        if self.state == "countdown" and now - self._t_begin >= COUNTDOWN_SEC:
            self._next_trial(now)
        elif self.state == "target" and now - self._t_trial >= self.timeout:
            self._finish(now, hit=False)
        elif self.state == "gap" and now - self._t_gap >= GAP_SEC:
            self._next_trial(now)

    def click(self, x: int, y: int, now: float) -> None:
        if self.state != "target":
            return
        if math.hypot(x - self._pos[0], y - self._pos[1]) <= self.radius * 1.15:
            self._finish(now, hit=True)

    def _next_trial(self, now: float) -> None:
        self._index += 1
        if self._index >= self.total:
            self.state = "done"
            return
        self._pos = self._zone_pos(self.sequence[self._index])
        self._t_trial = now
        self.state = "target"

    def _finish(self, now: float, hit: bool) -> None:
        reaction = now - self._t_trial
        gained = int(100 + 100 * max(0.0, 1 - reaction / self.timeout)) if hit else 0
        self.score += gained
        self.trials.append({"idx": self._index + 1, "zone": self.sequence[self._index].name,
                            "t_start": self._t_trial, "t_end": now, "hit": hit, "reaction": reaction})
        self._last_feedback = ("hit" if hit else "miss", self._pos, gained)
        self._t_gap = now
        self.state = "gap"

    # ---------- vẽ ----------
    def render(self, now: float, mouse: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, List[TextItem]]:
        img = np.empty((GAME_H, GAME_W, 3), np.uint8)
        img[:] = bgr("#17212f")
        texts: List[TextItem] = []
        x0, y0, x1, y1 = FIELD
        cv2.rectangle(img, (x0 - 45, y0 - 45), (x1 + 45, y1 + 45), bgr("#1d2b3d"), -1)
        for z in ZONES:                                            # chấm mờ báo trước các vị trí
            cv2.circle(img, self._zone_pos(z), self.radius, bgr("#2f435c"), 2, cv2.LINE_AA)

        # HUD
        cv2.rectangle(img, (0, 0), (GAME_W, HUD_H), bgr("#0f1722"), -1)
        progress = max(0, self._index + (1 if self.state == "gap" else 0))
        cv2.rectangle(img, (0, HUD_H), (int(GAME_W * min(1.0, progress / self.total)), HUD_H + 5),
                      bgr("#1abc9c"), -1)
        texts.append(TextItem(20, 35, f"Điểm: {self.score}", 22, "#ffffff", True, "lm"))
        title = self.movement or {"idle": "Sẵn sàng", "countdown": "Chuẩn bị", "done": "Hoàn thành"}.get(self.state, "")
        texts.append(TextItem(GAME_W // 2, 35, title, 24, "#f1c40f", True, "mm"))
        shown = min(self.total, max(0, self._index + 1))
        texts.append(TextItem(GAME_W - 20, 35, f"Lượt {shown}/{self.total}", 20, "#ffffff", False, "rm"))

        cx, cy = GAME_W // 2, GAME_H // 2 + 20
        if self.state == "idle":
            texts.append(TextItem(cx, cy - 50, "Game phục hồi: Chạm mục tiêu", 30, "#ffffff", True, "mm"))
            texts.append(TextItem(cx, cy, "Nhấn \"Bắt đầu phiên\" để chơi", 20, "#1abc9c", False, "mm"))
            texts.append(TextItem(cx, cy + 40, "Dùng chuột hoặc màn hình cảm ứng để chạm vào mục tiêu", 15, "#9fb3c8", False, "mm"))
        elif self.state == "countdown":
            left = max(0.0, COUNTDOWN_SEC - (now - self._t_begin))
            texts.append(TextItem(cx, cy - 20, str(int(math.ceil(left))), 96, "#f1c40f", True, "mm"))
            texts.append(TextItem(cx, cy + 60, "Giữ mặt thả lỏng, nhìn thẳng camera", 20, "#ffffff", False, "mm"))
        elif self.state == "target":
            ratio = max(0.0, 1 - (now - self._t_trial) / self.timeout)
            cv2.circle(img, self._pos, self.radius, bgr("#1abc9c"), -1, cv2.LINE_AA)
            cv2.circle(img, self._pos, 6, bgr("#ffffff"), -1, cv2.LINE_AA)
            cv2.circle(img, self._pos, self.radius + 6 + int(40 * ratio), bgr("#f1c40f"), 3, cv2.LINE_AA)
        elif self.state == "gap" and self._last_feedback:
            kind, pos, gained = self._last_feedback
            color = "#2ecc71" if kind == "hit" else "#e74c3c"
            cv2.circle(img, pos, self.radius + 8, bgr(color), 4, cv2.LINE_AA)
            label = f"Tốt! +{gained}" if kind == "hit" else "Hết giờ"
            texts.append(TextItem(pos[0], max(HUD_H + 20, pos[1] - self.radius - 22), label, 20, color, True, "mm"))
        elif self.state == "done":
            hits = sum(1 for t in self.trials if t["hit"])
            texts.append(TextItem(cx, cy - 30, "Hoàn thành!", 44, "#2ecc71", True, "mm"))
            texts.append(TextItem(cx, cy + 30, f"Chạm đúng {hits}/{self.total} lượt - {self.score} điểm", 22, "#ffffff", False, "mm"))

        if mouse is not None and 0 <= mouse[0] < GAME_W and 0 <= mouse[1] < GAME_H:
            cv2.circle(img, mouse, 9, bgr("#ffffff"), 2, cv2.LINE_AA)
        return img, texts
