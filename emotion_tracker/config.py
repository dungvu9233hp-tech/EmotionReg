"""Cấu hình tập trung: mọi tham số chỉnh được nằm ở đây, không rải trong code."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # --- đường dẫn ---
    model_path: Optional[Path] = None             # None: tự tìm *.onnx trong models_dir
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # --- camera / hiển thị ---
    camera_index: int = 0
    display_size: Tuple[int, int] = (640, 480)

    # --- tần suất đánh giá ---
    infer_every: int = 3            # cứ N khung hình đánh giá 1 lần (3 khung ≈ 10 lần/giây ở 30 FPS)

    # --- làm mượt kết quả ---
    ema_alpha: float = 0.35         # trọng số kết quả mới: nhỏ = mượt hơn nhưng trễ hơn
    switch_margin: float = 0.08     # nhãn chỉ đổi khi lớp mới hơn lớp cũ ít nhất ngần này
    box_smoothing: float = 0.4      # trọng số khung mặt mới (chống rung khung)

    # --- khuôn mặt ---
    detector_score: float = 0.7     # ngưỡng tin cậy của YuNet
    face_margin: float = 0.25       # nới khung mặt thêm 10% mỗi bên trước khi cắt
    no_face_timeout: float = 1.0    # mất mặt quá ngần này giây thì xoá trạng thái

    # --- kết hợp nhiều model (ensemble) ---
    use_hse: bool = True            # HSEmotion (EfficientNet, AffectNet)
    use_mediapipe: bool = True      # MediaPipe Face Landmarker (blendshape)
    hse_model_name: str = "enet_b0_8_best_afew"
    w_model: float = 0.40           # trọng số model của bạn (đã train đúng 4 lớp, định nghĩa "không thoải mái" của bạn)
    w_hse: float = 0.30             # HSEmotion (tổng quát, nhưng phải gộp 8 -> 4 lớp nên mất chính xác)
    w_mediapipe: float = 0.30       # MediaPipe (hình học khuôn mặt, không bị "lối tắt nhíu mày" của CNN)
    unc_vote_threshold: float = 0.40  # nguồn nào cho "không thoải mái" >= ngưỡng này được tính là 1 phiếu
    unc_lone_damp: float = 0.50     # chỉ 1 nguồn bỏ phiếu -> nhân "không thoải mái" với hệ số này
    unc_extra_margin: float = 0.10  # biên bổ sung để NHÃN chuyển sang "không thoải mái" (khó vào hơn các nhãn khác)
    calibration_sec: float = 3.0    # thời gian đo nét mặt bình thường cho MediaPipe

    # --- đồ thị / ghi log ---
    history_sec: int = 60
    log_every_sec: float = 1.0

    @classmethod
    def from_args(cls, argv: Optional[Sequence[str]] = None) -> "Config":
        d = cls()
        ap = argparse.ArgumentParser(description="Theo dõi biểu cảm thời gian thực qua webcam")
        ap.add_argument("--model", type=Path, default=None, help="đường dẫn file .onnx (mặc định: tự tìm trong models/)")
        ap.add_argument("--cam", type=int, default=d.camera_index, help="số thứ tự camera")
        ap.add_argument("--every", type=int, default=d.infer_every, help="cứ N khung hình đánh giá 1 lần")
        ap.add_argument("--alpha", type=float, default=d.ema_alpha, help="hệ số làm mượt EMA (0..1)")
        ap.add_argument("--no-hse", action="store_true", help="tắt HSEmotion")
        ap.add_argument("--no-mp", action="store_true", help="tắt MediaPipe")
        a = ap.parse_args(argv)
        if a.every < 1:
            ap.error("--every phải >= 1")
        if not 0 < a.alpha <= 1:
            ap.error("--alpha phải trong (0, 1]")
        return cls(model_path=a.model, camera_index=a.cam, infer_every=a.every, ema_alpha=a.alpha,
                   use_hse=not a.no_hse, use_mediapipe=not a.no_mp)
