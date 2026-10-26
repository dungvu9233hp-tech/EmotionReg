"""Cửa sổ game rehab: game ở giữa/trái, camera + biểu đồ cảm xúc thu nhỏ bên phải, ghi video mp4."""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from ..camera import Camera
from ..config import Config
from ..metrics import FpsMeter, RateCounter
from ..pipeline import EmotionPipeline
from ..rehab.compositor import Compositor
from ..rehab.game import GAME_W, LEVELS, ReachGame
from ..rehab.recorder import RehabRecorder
from ..rehab.report import mmss
from ..rehab.settings import RehabSettings

_AUTO_STOP_AFTER_DONE = 3.0


class RehabApp:
    def __init__(self, root: tk.Tk, cfg: Config, pipeline: EmotionPipeline, settings: Optional[RehabSettings] = None):
        self.root, self.cfg, self.pipeline = root, cfg, pipeline
        self.settings = settings or RehabSettings()
        self.comp = Compositor(min(cfg.history_sec, 40))
        self.game = ReachGame()
        self.camera: Optional[Camera] = None
        self.recorder: Optional[RehabRecorder] = None
        self._running = False
        self._finalizing = False
        self._mouse: Optional[Tuple[int, int]] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._done_at: Optional[float] = None
        self._fps = FpsMeter()
        self._rate = RateCounter()
        self._build_ui()
        pipeline.start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._show_idle()

    # ---------- giao diện ----------
    def _build_ui(self) -> None:
        self.root.title("Game phục hồi chức năng + theo dõi biểu cảm")
        self.root.resizable(False, False)
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=6)
        self._cam = tk.IntVar(value=self.cfg.camera_index)
        self._level = tk.StringVar(value="Vừa")
        self._rounds = tk.IntVar(value=3)
        tk.Label(bar, text="Camera:").pack(side="left")
        ttk.Spinbox(bar, from_=0, to=5, width=3, textvariable=self._cam).pack(side="left", padx=(2, 10))
        tk.Label(bar, text="Mức độ:").pack(side="left")
        ttk.Combobox(bar, values=list(LEVELS), textvariable=self._level, width=6, state="readonly").pack(side="left", padx=(2, 10))
        tk.Label(bar, text="Số vòng (8 động tác/vòng):").pack(side="left")
        ttk.Spinbox(bar, from_=1, to=5, width=3, textvariable=self._rounds).pack(side="left", padx=(2, 10))
        self._every = tk.IntVar(value=self.cfg.infer_every)
        tk.Label(bar, text="Đánh giá mỗi N khung:").pack(side="left")
        ttk.Spinbox(bar, from_=1, to=15, width=3, textvariable=self._every).pack(side="left", padx=(2, 10))
        self._button = ttk.Button(bar, text="Bắt đầu phiên", command=self.toggle)
        self._button.pack(side="left", padx=4)
        ttk.Button(bar, text="Hiệu chuẩn (mặt thả lỏng 3s)", command=self.calibrate).pack(side="left", padx=4)
        self.status = tk.Label(bar, text="", anchor="w")
        self.status.pack(side="left", padx=10)
        self.display = tk.Label(self.root, bd=0, highlightthickness=0, cursor="crosshair")
        self.display.pack(anchor="nw")
        self.stats = tk.Label(self.root, justify="left", font=("Consolas", 9), anchor="w")
        self.stats.pack(fill="x", padx=8, pady=(2, 6))
        self.display.bind("<Button-1>", self._on_click)
        self.display.bind("<Motion>", self._on_move)
        self.display.bind("<Leave>", lambda e: setattr(self, "_mouse", None))

    def _show(self, canvas: np.ndarray) -> None:
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        self.display.config(image=self._photo)

    def _show_idle(self) -> None:
        self.game = ReachGame()
        self.comp.clear()
        img, texts = self.game.render(time.time(), None)
        self._show(self.comp.compose(img, texts, None, None, time.time(), None))

    # ---------- chuột ----------
    def _on_move(self, e) -> None:
        self._mouse = (e.x, e.y)

    def _on_click(self, e) -> None:
        if self._running and e.x < GAME_W:
            self.game.click(e.x, e.y, time.time())

    # ---------- điều khiển ----------
    def calibrate(self) -> None:
        fn = getattr(self.pipeline.predictor, "calibrate", None)
        if fn:
            fn()

    def toggle(self) -> None:
        self.stop() if self._running else self.start()

    def _int(self, var: tk.IntVar, default: int) -> int:
        try:
            return int(var.get())
        except (tk.TclError, ValueError):
            return default

    def start(self) -> None:
        if self._running or self._finalizing:
            return
        camera = Camera(self._int(self._cam, 0), self.cfg.display_size)
        if not camera.open():
            messagebox.showerror("Camera", f"Không mở được camera số {camera.index}.\n"
                                           "Thử đổi số camera hoặc đóng ứng dụng khác đang dùng camera.")
            return
        try:
            self.recorder = RehabRecorder(self.settings.sessions_dir, self.comp.size, self.settings)
        except Exception as exc:
            camera.release()
            messagebox.showerror("Video", f"Không tạo được file video: {exc}")
            return
        self.camera = camera
        self.pipeline.reset()
        self.calibrate()
        self.comp.clear()
        self.game = ReachGame(self._level.get(), max(1, min(5, self._int(self._rounds, 3))))
        now = time.time()
        self.recorder.start(now)
        self.game.begin(now)
        self._done_at = None
        self._running = True
        self._button.config(text="Dừng & xuất video")
        self.status.config(text="● Đang ghi video")
        self._tick()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        meta = {"date": time.strftime("%d/%m/%Y %H:%M"), "level": self.game.level,
                "duration": time.time() - self.recorder.t0}
        self.recorder.stop_async(list(self.game.trials), meta)
        self._finalizing = True
        self._button.config(state="disabled", text="Đang xuất...")
        self.status.config(text="Đang xuất video và báo cáo, vui lòng đợi...")
        self._poll_finalize()

    def _poll_finalize(self) -> None:
        if not self.recorder.finished:
            self.root.after(300, self._poll_finalize)
            return
        self._finalizing = False
        out = self.recorder.result
        self._button.config(state="normal", text="Bắt đầu phiên")
        self.status.config(text="")
        self._show_idle()
        if out is not None:
            messagebox.showinfo("Đã lưu phiên", f"Thư mục: {out.folder}\n"
                                                f"Video: session.mp4\nBáo cáo: report.html\n\n{out.summary}")

    def close(self) -> None:
        if self._running:
            self.stop()
        if self._finalizing and self.recorder is not None:
            self.root.title("Đang xuất video, vui lòng đợi...")
            self.recorder.join()
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
        self.pipeline.infer_every = max(1, self._int(self._every, self.pipeline.infer_every))
        result = self.pipeline.process(frame, now)
        if result.new_inference and result.probs is not None:
            self.comp.add(now, result.probs)
        self.game.update(now)
        g_img, g_texts = self.game.render(now, self._mouse)
        canvas = self.comp.compose(g_img, g_texts, frame, result, now, now - self.recorder.t0)
        self._show(canvas)
        self.recorder.add(now, canvas, result.label, result.probs)
        self._update_stats(result)
        if self.game.done:
            if self._done_at is None:
                self._done_at = now
            elif now - self._done_at > _AUTO_STOP_AFTER_DONE:
                self.stop()
                return
        self.root.after(1, self._tick)


    def _update_stats(self, r) -> None:
        fps = self._fps.tick()
        if self._rate.add(1 if r.new_inference else 0):
            debug = getattr(self.pipeline.predictor, "debug", "").replace("\n", "  |  ")
            self.stats.config(text=(
                f"Camera {fps:4.1f} FPS  |  Đánh giá mỗi {self.pipeline.infer_every} khung (~{self._rate.rate:.1f} lần/giây)"
                f"  |  Độ trễ model {self.pipeline.latency_ms:.0f} ms  |  Bộ dò mặt: {self.pipeline.detector.name}\n{debug}"))


def run(cfg: Config, pipeline: EmotionPipeline) -> int:
    root = tk.Tk()
    RehabApp(root, cfg, pipeline)
    root.mainloop()
    return 0
