"""Cửa sổ chính: nối camera, pipeline, ghi log và các panel."""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import numpy as np

from ..camera import Camera
from ..config import Config
from ..labels import COLORS, LABELS_VI
from ..metrics import FpsMeter, RateCounter
from ..pipeline import EmotionPipeline, FrameResult
from ..session import SessionRecorder, SessionSummary
from .panels import BarsPanel, ControlPanel, TimelinePanel, VideoPanel

_SIDE_W = 350
_TIMELINE_REFRESH_MS = 250


class App:
    def __init__(self, root: tk.Tk, cfg: Config, pipeline: EmotionPipeline):
        self.root, self.cfg, self.pipeline = root, cfg, pipeline
        self.recorder = SessionRecorder(cfg.logs_dir, cfg.log_every_sec)
        self.camera: Optional[Camera] = None
        self._running = False
        self._fps = FpsMeter()
        self._rate = RateCounter()
        self._build_ui()
        pipeline.start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(300, self.start)
        root.after(_TIMELINE_REFRESH_MS, self._refresh_timeline)

    # ---------- dựng giao diện ----------
    def _build_ui(self) -> None:
        w, h = self.cfg.display_size
        self.root.title("Theo dõi biểu cảm thời gian thực")
        self.video = VideoPanel(self.root, (w, h))
        self.video.grid(row=0, column=0, padx=8, pady=8)
        side = tk.Frame(self.root)
        side.grid(row=0, column=1, sticky="n", padx=8, pady=8)
        self.big = tk.Label(side, text="—", font=("Segoe UI", 26, "bold"), width=16)
        self.big.pack(pady=(0, 6))
        self.bars = BarsPanel(side)
        self.bars.pack()
        self.controls = ControlPanel(side, self.cfg, self.toggle, self.calibrate)
        self.controls.pack(fill="x", pady=8)
        self.stats = tk.Label(side, justify="left", font=("Consolas", 9), anchor="w")
        self.stats.pack(fill="x")
        self.timeline = TimelinePanel(self.root, w + _SIDE_W, 180, self.cfg.history_sec)
        self.timeline.grid(row=1, column=0, columnspan=2, padx=8, pady=(0, 8))

    # ---------- điều khiển ----------
    def calibrate(self) -> None:
        """Đo nét mặt bình thường của người dùng cho MediaPipe (giữ mặt thả lỏng ~3 giây)."""
        fn = getattr(self.pipeline.predictor, "calibrate", None)
        if fn:
            fn()

    def toggle(self) -> None:
        self.stop() if self._running else self.start()

    def start(self) -> None:
        if self._running:
            return
        camera = Camera(self.controls.camera_index, self.cfg.display_size)
        if not camera.open():
            messagebox.showerror(
                "Camera",
                f"Không mở được camera số {camera.index}.\n"
                "Thử đổi số camera hoặc đóng ứng dụng khác đang dùng camera.")
            return
        self.camera = camera
        self.pipeline.reset()
        self.calibrate()
        self.timeline.clear()
        self.bars.render(None, None)
        self.recorder.start(self.controls.write_csv)
        self._running = True
        self.controls.set_running(True)
        self._tick()

    def stop(self) -> None:
        was_running, self._running = self._running, False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        summary = self.recorder.stop()
        self.controls.set_running(False)
        if was_running:
            self._show_summary(summary)

    def close(self) -> None:
        self.stop()
        self.pipeline.close()
        self.root.destroy()

    # ---------- vòng lặp chính ----------
    def _tick(self) -> None:
        if not self._running:
            return
        frame = self.camera.read() if self.camera else None
        if frame is None:
            self.root.after(30, self._tick)
            return
        now = time.time()
        self.pipeline.infer_every = self.controls.infer_every
        result = self.pipeline.process(frame, now)

        if result.changed:
            self.bars.render(result.probs, result.label)
        if result.new_inference:
            self.timeline.add(now, result.probs)
        self.recorder.update(now, result.label, result.probs)
        self._render_frame(frame, result)
        self._update_stats(result)
        self.root.after(1, self._tick)

    def _render_frame(self, frame: np.ndarray, r: FrameResult) -> None:
        if r.box is not None and r.label is not None:
            caption = f"{LABELS_VI[r.label]} {r.probs[r.label] * 100:.0f}%"
            self.video.show(frame, r.box, caption, COLORS[r.label])
        else:
            self.video.show(frame, r.box)
        if r.label is not None:
            self.big.config(text=LABELS_VI[r.label], fg=COLORS[r.label])
        else:
            self.big.config(text="Không thấy mặt" if r.box is None else "…", fg="#888")

    def _update_stats(self, r: FrameResult) -> None:
        fps = self._fps.tick()
        if self._rate.add(1 if r.new_inference else 0):
            self.stats.config(text=(
                f"Camera       : {fps:4.1f} FPS\n"
                f"Đánh giá     : mỗi {self.pipeline.infer_every} khung (~{self._rate.rate:.1f} lần/giây)\n"
                f"Độ trễ model : {self.pipeline.latency_ms:.0f} ms\n"
                f"Bộ dò mặt    : {self.pipeline.detector.name}\n"
                f"{getattr(self.pipeline.predictor, 'debug', '')}"))

    def _refresh_timeline(self) -> None:
        self.timeline.render(time.time())
        self.root.after(_TIMELINE_REFRESH_MS, self._refresh_timeline)

    def _show_summary(self, summary: SessionSummary) -> None:
        if summary.shares is None:
            return
        lines = [f"  {LABELS_VI[i]}: {summary.shares[i] * 100:.0f}%"
                 for i in np.argsort(-summary.shares) if summary.shares[i] >= 0.01]
        msg = "Tỉ lệ thời gian theo biểu cảm:\n" + "\n".join(lines)
        if summary.csv_path:
            msg += f"\n\nĐã lưu: {summary.csv_path}"
        messagebox.showinfo("Tổng kết phiên", msg)


def run(cfg: Config, pipeline: EmotionPipeline) -> int:
    root = tk.Tk()
    App(root, cfg, pipeline)
    root.mainloop()
    return 0