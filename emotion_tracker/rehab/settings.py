"""Cấu hình riêng cho chế độ game rehab (không đụng tới config.py cũ)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import PROJECT_ROOT


@dataclass(frozen=True)
class RehabSettings:
    sessions_dir: Path = PROJECT_ROOT / "sessions"   # mỗi phiên một thư mục con
    record_fps: int = 20                              # FPS của video xuất ra
    window_lag_sec: float = 0.7                       # cộng thêm sau mỗi động tác (biểu cảm trễ do EMA)
    flag_frac: float = 0.30                           # >= 30% thời gian nhãn "không thoải mái" -> gắn cờ
    flag_mean: float = 0.40                           # hoặc xác suất "không thoải mái" trung bình >= 0.40
