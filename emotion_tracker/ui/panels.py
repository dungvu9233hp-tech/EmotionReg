"""Các thành phần giao diện tái sử dụng. Mỗi panel chỉ lo vẽ, không chứa logic nghiệp vụ."""
from __future__ import annotations

import tkinter as tk
from collections import deque
from tkinter import ttk
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from ..config import Config
from ..labels import COLORS, LABELS_VI, NUM_CLASSES

_HILITE = "#00ff88"


class VideoPanel(tk.Canvas):
    """Khung video kèm khung mặt và nhãn."""

    def __init__(self, master, size: Tuple[int, int]):
        super().__init__(master, width=size[0], height=size[1], bg="black", highlightthickness=0)
        self._img = self.create_image(0, 0, anchor="nw")
        self._box = self.create_rectangle(0, 0, 0, 0, width=3, state="hidden")
        self._txt = self.create_text(0, 0, anchor="sw", font=("Segoe UI", 16, "bold"), state="hidden")
        self._photo: Optional[ImageTk.PhotoImage] = None   # giữ tham chiếu, nếu không ảnh bị xoá

    def show(self, frame_bgr: np.ndarray, box=None, caption: Optional[str] = None, color: str = _HILITE) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.itemconfig(self._img, image=self._photo)
        if box is None:
            self.itemconfig(self._box, state="hidden")
            self.itemconfig(self._txt, state="hidden")
            return
        x, y, w, h = box
        self.coords(self._box, x, y, x + w, y + h)
        self.itemconfig(self._box, state="normal", outline=color)
        if caption:
            self.coords(self._txt, x, max(24, y - 4))
            self.itemconfig(self._txt, state="normal", fill=color, text=caption)
        else:
            self.itemconfig(self._txt, state="hidden")


class BarsPanel(tk.Canvas):
    """Thanh xác suất cho từng lớp."""

    ROW = 30
    BAR_X0, BAR_W = 125, 160   # tên lớp dài ("Không thoải mái") nên thanh bắt đầu lệch phải hơn

    def __init__(self, master, width: int = 330):
        super().__init__(master, width=width, height=NUM_CLASSES * self.ROW + 4, highlightthickness=0)
        self._width = width
        self.render(None, None)

    def render(self, probs: Optional[np.ndarray], highlight: Optional[int]) -> None:
        self.delete("all")
        p = np.zeros(NUM_CLASSES) if probs is None else probs
        for i in range(NUM_CLASSES):
            y = i * self.ROW + 2
            weight = "bold" if i == highlight else "normal"
            self.create_text(0, y + 12, anchor="w", text=LABELS_VI[i], font=("Segoe UI", 10, weight))
            self.create_rectangle(self.BAR_X0, y + 3, self.BAR_X0 + self.BAR_W, y + 21, fill="#eeeeee", outline="")
            self.create_rectangle(self.BAR_X0, y + 3, self.BAR_X0 + self.BAR_W * float(p[i]), y + 21, fill=COLORS[i], outline="")
            self.create_text(self._width - 5, y + 12, anchor="e", text=f"{p[i] * 100:4.0f}%", font=("Consolas", 10))


class TimelinePanel(tk.Canvas):
    """Đồ thị xác suất theo thời gian (N giây gần nhất)."""

    PAD_L, PAD_B, PAD_T, PAD_R = 36, 20, 12, 8

    def __init__(self, master, width: int, height: int, history_sec: int):
        super().__init__(master, width=width, height=height, bg="#1e1e1e", highlightthickness=0)
        self._canvas_w, self._canvas_h, self._span = width, height, history_sec
        self._history: deque = deque()

    def add(self, t: float, probs: np.ndarray) -> None:
        self._history.append((t, np.asarray(probs).copy()))
        while self._history and t - self._history[0][0] > self._span:
            self._history.popleft()

    def clear(self) -> None:
        self._history.clear()

    def render(self, now: float) -> None:
        self.delete("all")
        W, H = self._canvas_w, self._canvas_h
        plot_h = H - self.PAD_B - self.PAD_T
        for v in (0, 0.5, 1):
            y = H - self.PAD_B - plot_h * v
            self.create_line(self.PAD_L, y, W - self.PAD_R, y, fill="#444")
            self.create_text(self.PAD_L - 4, y, anchor="e", text=f"{int(v * 100)}%", fill="#aaa", font=("Consolas", 8))
        for k in range(NUM_CLASSES):
            for seg in self._segments(k, now, W, H, plot_h):
                if len(seg) >= 4:
                    self.create_line(*seg, fill=COLORS[k], width=2)
            self.create_text(self.PAD_L + 8 + k * 150, H - 8, anchor="w", text="● " + LABELS_VI[k],
                             fill=COLORS[k], font=("Segoe UI", 9))
        self.create_text(W - self.PAD_R, 4, anchor="ne", text=f"{self._span} giây gần nhất",
                         fill="#aaa", font=("Segoe UI", 8))

    def _segments(self, k: int, now: float, W: int, H: int, plot_h: float):
        """Tách đường thành các đoạn liền; chỗ mất mặt (>1 giây) để hở."""
        segs, pts, prev = [], [], None
        for t, p in self._history:
            x = self.PAD_L + (W - self.PAD_L - self.PAD_R) * (1 - (now - t) / self._span)
            y = H - self.PAD_B - plot_h * float(p[k])
            if prev is not None and t - prev > 1.0:
                segs.append(pts)
                pts = []
            pts += [x, y]
            prev = t
        segs.append(pts)
        return segs


class ControlPanel(tk.LabelFrame):
    """Chọn camera, tần suất đánh giá, bật ghi CSV, nút Bắt đầu/Dừng."""

    def __init__(self, master, cfg: Config, on_toggle: Callable[[], None],
                 on_calibrate: Optional[Callable[[], None]] = None):
        super().__init__(master, text="Điều khiển")
        self._camera = tk.IntVar(value=cfg.camera_index)
        self._every = tk.IntVar(value=cfg.infer_every)
        self._csv = tk.BooleanVar(value=True)
        self._last_every = cfg.infer_every
        tk.Label(self, text="Camera số:").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Spinbox(self, from_=0, to=5, width=4, textvariable=self._camera).grid(row=0, column=1, pady=2)
        tk.Label(self, text="Đánh giá mỗi N khung:").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Spinbox(self, from_=1, to=15, width=4, textvariable=self._every).grid(row=1, column=1, pady=2)
        tk.Checkbutton(self, text="Ghi CSV (logs/)", variable=self._csv).grid(row=2, column=0, columnspan=2, sticky="w")
        self._button = ttk.Button(self, text="Bắt đầu", command=on_toggle)
        self._button.grid(row=3, column=0, columnspan=2, pady=4)
        if on_calibrate:
            ttk.Button(self, text="Hiệu chuẩn (mặt thả lỏng 3s)", command=on_calibrate).grid(
                row=4, column=0, columnspan=2, pady=(0, 4))

    @property
    def camera_index(self) -> int:
        try:
            return int(self._camera.get())
        except (tk.TclError, ValueError):
            return 0

    @property
    def infer_every(self) -> int:
        try:                                  # ô đang để trống/gõ dở thì giữ giá trị hợp lệ trước đó
            self._last_every = max(1, int(self._every.get()))
        except (tk.TclError, ValueError):
            pass
        return self._last_every

    @property
    def write_csv(self) -> bool:
        return bool(self._csv.get())

    def set_running(self, running: bool) -> None:
        self._button.config(text="Dừng" if running else "Bắt đầu")