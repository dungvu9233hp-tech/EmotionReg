"""Báo cáo cho bác sĩ: động tác nào làm người chơi có biểu hiện khó chịu.

Với mỗi lượt chơi (= một động tác) lấy cửa sổ [bắt đầu, kết thúc + lag] rồi tính:
- frac: tỉ lệ khung có nhãn "không thoải mái"
- mean/peak: xác suất "không thoải mái" trung bình / cao nhất
Lượt vượt ngưỡng được gắn cờ. Kết quả gộp theo tên động tác để thấy động tác nào hay bị gắn cờ.
Đây là công cụ sàng lọc hỗ trợ, không phải chẩn đoán đau; bác sĩ xem lại video để kết luận.
"""
from __future__ import annotations

import csv
import html
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..labels import CLASSES, UNCOMFORTABLE
from .settings import RehabSettings


def mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def analyze(rows: List[list], trials: List[dict], t0: float, st: RehabSettings) -> List[dict]:
    """rows: [t_rel, label(-1 nếu không thấy mặt), p0..p3]."""
    arr = np.asarray(rows, dtype=float).reshape(-1, 2 + len(CLASSES))
    out = []
    for tr in trials:
        s, e = tr["t_start"] - t0, tr["t_end"] - t0
        m = (arr[:, 0] >= s) & (arr[:, 0] <= e + st.window_lag_sec) & (arr[:, 1] >= 0)
        n = int(m.sum())
        if n:
            frac = float((arr[m, 1] == UNCOMFORTABLE).mean())
            pu = arr[m, 2 + UNCOMFORTABLE]
            mean, peak = float(pu.mean()), float(pu.max())
        else:
            frac = mean = peak = None
        flagged = n > 0 and (frac >= st.flag_frac or mean >= st.flag_mean)
        out.append({**tr, "start_s": s, "end_s": e, "n": n, "frac": frac, "mean": mean,
                    "peak": peak, "flagged": flagged})
    return out


def by_movement(results: List[dict]) -> List[dict]:
    groups: Dict[str, List[dict]] = {}
    for r in results:
        groups.setdefault(r["zone"], []).append(r)
    rows = []
    for zone, items in groups.items():
        seen = [i for i in items if i["mean"] is not None]
        rows.append({
            "zone": zone, "trials": len(items), "flagged": sum(i["flagged"] for i in items),
            "mean": float(np.mean([i["mean"] for i in seen])) if seen else None,
            "hit_rate": float(np.mean([i["hit"] for i in items])),
            "reaction": float(np.mean([i["reaction"] for i in items])),
        })
    rows.sort(key=lambda r: (-r["flagged"] / max(1, r["trials"]), -(r["mean"] or 0)))
    return rows


def summary_text(moves: List[dict]) -> str:
    bad = [m for m in moves if m["flagged"] > 0]
    if not bad:
        return "Không có động tác nào vượt ngưỡng khó chịu."
    lines = [f"- {m['zone']}: {m['flagged']}/{m['trials']} lượt bị gắn cờ" for m in bad[:4]]
    return "Động tác cần xem lại:\n" + "\n".join(lines)


def write_csvs(folder: Path, rows: List[list], results: List[dict]) -> None:
    with open(folder / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "label", *CLASSES])
        for r in rows:
            w.writerow([f"{r[0]:.2f}", CLASSES[int(r[1])] if r[1] >= 0 else "", *(f"{v:.4f}" for v in r[2:])])
    with open(folder / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lượt", "động tác", "bắt đầu (s)", "kết thúc (s)", "chạm trúng", "phản ứng (s)",
                    "tỉ lệ khó chịu", "p khó chịu TB", "p khó chịu max", "gắn cờ"])
        fmt = lambda v: "" if v is None else f"{v:.3f}"
        for r in results:
            w.writerow([r["idx"], r["zone"], f"{r['start_s']:.2f}", f"{r['end_s']:.2f}", int(r["hit"]),
                        f"{r['reaction']:.2f}", fmt(r["frac"]), fmt(r["mean"]), fmt(r["peak"]), int(r["flagged"])])


def write_html(folder: Path, results: List[dict], moves: List[dict], meta: dict, video_ok_in_browser: bool) -> Path:
    esc = html.escape
    pct = lambda v: "—" if v is None else f"{v * 100:.0f}%"
    move_rows = "".join(
        f"<tr class='{'flag' if m['flagged'] else ''}'><td>{esc(m['zone'])}</td><td>{m['trials']}</td>"
        f"<td>{m['flagged']}</td><td>{pct(m['mean'])}</td><td>{pct(m['hit_rate'])}</td><td>{m['reaction']:.1f}s</td></tr>"
        for m in moves)
    trial_rows = "".join(
        f"<tr class='{'flag' if r['flagged'] else ''}'><td>{r['idx']}</td><td>{esc(r['zone'])}</td>"
        f"<td><a href='#' onclick='seek({max(0, r['start_s'] - 1):.1f});return false'>{mmss(r['start_s'])}</a></td>"
        f"<td>{'có' if r['hit'] else 'không'}</td><td>{pct(r['frac'])}</td><td>{pct(r['mean'])}</td>"
        f"<td>{pct(r['peak'])}</td><td>{'⚑' if r['flagged'] else ''}</td></tr>"
        for r in results)
    note = "" if video_ok_in_browser else (
        "<p class='warn'>Video chưa chuyển sang H.264 nên trình duyệt có thể không phát được. "
        "Mở <b>session.mp4</b> bằng VLC, hoặc cài <code>imageio-ffmpeg</code> rồi chơi lại.</p>")
    page = f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>Báo cáo phiên phục hồi</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 12px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
th{{background:#f3f3f3}}tr.flag{{background:#fdeaea}}.warn{{color:#b9770e}}video{{width:100%;background:#000}}
.muted{{color:#666}}</style>
<h1>Báo cáo phiên phục hồi</h1>
<p class="muted">{esc(meta.get('date', ''))} · mức {esc(str(meta.get('level', '')))} · thời lượng {mmss(meta.get('duration', 0))}</p>
<video id="v" src="session.mp4" controls></video>{note}
<script>function seek(t){{var v=document.getElementById('v');v.currentTime=t;v.play();}}</script>
<h2>Tổng hợp theo động tác</h2>
<table><tr><th>Động tác</th><th>Số lượt</th><th>Bị gắn cờ</th><th>Khó chịu TB</th><th>Chạm đúng</th><th>Phản ứng TB</th></tr>{move_rows}</table>
<h2>Chi tiết từng lượt <span class="muted">(bấm thời điểm để tua video)</span></h2>
<table><tr><th>#</th><th>Động tác</th><th>Thời điểm</th><th>Chạm đúng</th><th>Tỉ lệ khó chịu</th><th>Khó chịu TB</th><th>Cao nhất</th><th>Cờ</th></tr>{trial_rows}</table>
<p class="muted">Gắn cờ = ≥{meta.get('flag_frac', 0.3) * 100:.0f}% thời gian bị nhận là "Không thoải mái" hoặc xác suất trung bình ≥{meta.get('flag_mean', 0.4) * 100:.0f}%.
Đây là công cụ sàng lọc, không thay thế đánh giá lâm sàng.</p></html>"""
    path = folder / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
