"""Ghi video mp4 phiên chơi (luồng riêng, không làm giật giao diện) + nhật ký khung hình + xuất báo cáo."""
from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import report
from .settings import RehabSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionOutput:
    folder: Path
    video: Optional[Path]
    report: Optional[Path]
    summary: str


class _Writer(threading.Thread):
    def __init__(self, path: Path, size: Tuple[int, int], fps: int):
        super().__init__(daemon=True, name="video-writer")
        self._vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._vw.isOpened():
            raise RuntimeError("Không tạo được file video (VideoWriter)")
        self._q: queue.Queue = queue.Queue(maxsize=90)
        self.start()

    def put(self, frame: np.ndarray, repeat: int) -> bool:
        try:
            self._q.put_nowait((frame, repeat))
            return True
        except queue.Full:
            return False

    def run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            frame, repeat = item
            for _ in range(repeat):
                self._vw.write(frame)
        self._vw.release()

    def close(self) -> None:
        self._q.put(None)
        self.join()


def _find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class RehabRecorder:
    def __init__(self, sessions_dir: Path, size: Tuple[int, int], settings: RehabSettings):
        self.settings = settings
        self.fps = settings.record_fps
        self.folder = Path(sessions_dir) / datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.folder.mkdir(parents=True, exist_ok=True)
        self._raw = self.folder / "raw.mp4"
        self._writer = _Writer(self._raw, size, self.fps)
        self.t0 = 0.0
        self.rows: List[list] = []
        self._written = 0
        self._thread: Optional[threading.Thread] = None
        self.result: Optional[SessionOutput] = None

    def start(self, now: float) -> None:
        self.t0 = now

    def add(self, now: float, frame_bgr: np.ndarray, label: Optional[int], probs) -> None:
        t = now - self.t0
        p = [0.0] * 4 if probs is None else [float(v) for v in probs]
        self.rows.append([t, -1 if label is None else int(label), *p])
        n = min(int(t * self.fps) + 1 - self._written, self.fps * 2)   # lặp khung để video đúng thời gian thật
        if n > 0 and self._writer.put(frame_bgr, n):
            self._written += n

    # ---------- kết thúc ----------
    def stop_async(self, trials: List[dict], meta: dict) -> None:
        self._thread = threading.Thread(target=self._job, args=(trials, meta), name="finalize")
        self._thread.start()

    @property
    def finished(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _job(self, trials: List[dict], meta: dict) -> None:
        try:
            self._writer.close()
            video = self.folder / "session.mp4"
            h264 = self._convert(self._raw, video)
            results = report.analyze(self.rows, trials, self.t0, self.settings)
            moves = report.by_movement(results)
            report.write_csvs(self.folder, self.rows, results)
            meta = {**meta, "flag_frac": self.settings.flag_frac, "flag_mean": self.settings.flag_mean}
            html_path = report.write_html(self.folder, results, moves, meta, h264)
            self.result = SessionOutput(self.folder, video, html_path, report.summary_text(moves))
        except Exception as exc:
            logger.exception("Xuất phiên lỗi")
            self.result = SessionOutput(self.folder, None, None, f"Lỗi khi xuất video/báo cáo: {exc}")

    @staticmethod
    def _convert(raw: Path, out: Path) -> bool:
        """Chuyển mp4v -> H.264 (trình duyệt/điện thoại phát được). Không có ffmpeg thì giữ nguyên mp4v."""
        exe = _find_ffmpeg()
        if exe:
            try:
                subprocess.run([exe, "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-crf", "23", "-preset", "veryfast", str(out)],
                               capture_output=True, check=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                raw.unlink(missing_ok=True)
                return True
            except Exception:
                logger.exception("ffmpeg lỗi, giữ video mp4v")
        raw.replace(out)
        return False
