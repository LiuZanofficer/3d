#!/usr/bin/env python3
"""Print one snapshot of E4 training progress.

Reads the live train.log + per_class_iou_last.json / dead_classes_last.json
under ``--run-dir`` and prints a short multi-line summary covering:

* current epoch / step / it/s / wall-time
* train/loss_step, train/loss_epoch (last completed epoch)
* val/miou (raw), val/miou_0 (per-dataset 0)
* f-mIoU computed from per_class_iou_last.json (exclude wall/floor/ceiling)
* dead-class count from dead_classes_last.json
* "best so far" line driven by per_class_iou_best.json

Usage:
    python print_status.py --run-dir /root/autodl-tmp/lz_outputs/E4_v2

Designed to be cheap (~tens of ms) so a 30-min monitor loop can call it
without disturbing training.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Background classes excluded from foreground mIoU. Names match the
# class_name field in per_class_iou_*.json on this repo.
_BG_NAMES = {"wall", "floor", "ceiling"}


def _tail_bytes(path: Path, n: int = 100_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", errors="ignore")


def _last_progress_line(log_text: str) -> Optional[str]:
    # tqdm uses \r as separator; we also split on newlines to be safe.
    chunks = re.split(r"[\r\n]+", log_text)
    for line in reversed(chunks):
        if "Epoch " in line and ("train/loss" in line or "it/s" in line):
            return line
    return None


def _parse_progress(line: str) -> Dict[str, Any]:
    def _num(rx: str) -> Optional[float]:
        m = re.search(rx, line)
        return float(m.group(1)) if m else None

    m_ep = re.search(r"Epoch (\d+):\s*(\d+)%", line)
    m_step = re.search(r"\|\s*(\d+)/(\d+)\s*\[", line)
    m_it = re.search(r"([0-9.]+)\s*it/s", line)
    return {
        "epoch": int(m_ep.group(1)) if m_ep else None,
        "epoch_pct": int(m_ep.group(2)) if m_ep else None,
        "step": int(m_step.group(1)) if m_step else None,
        "step_total": int(m_step.group(2)) if m_step else None,
        "it_per_s": float(m_it.group(1)) if m_it else None,
        "loss_step": _num(r"train/loss_step=([0-9.eE+-]+)"),
        "loss_epoch": _num(r"train/loss_epoch=([0-9.eE+-]+)"),
        "val_miou": _num(r"val/miou=([0-9.eE+-]+)"),
        "val_miou_0": _num(r"val/miou_0=([0-9.eE+-]+)"),
    }


def _load_perclass(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:  # pragma: no cover - corrupted mid-write
        return None
    by_dl = data.get("by_dataloader") or {}
    rows = by_dl.get("0") or by_dl.get(0) or []
    if not rows and isinstance(data, list):
        rows = data
    return rows


def _compute_fmiou(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    fg_rows = [r for r in rows if str(r.get("class_name", "")).lower() not in _BG_NAMES]
    if not fg_rows:
        return None
    ious = [float(r.get("iou", 0.0) or 0.0) for r in fg_rows]
    miou = sum(ious) / len(ious)
    dead = sum(1 for v in ious if v <= 1e-6)
    return {
        "f_miou": miou,
        "fg_count": len(fg_rows),
        "dead": dead,
    }


def _epoch_from_metafile(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return int(json.load(f).get("epoch", -1))
    except Exception:  # pragma: no cover
        return None


def _proc_alive(pid_file: Path) -> Optional[bool]:
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return None
    return Path(f"/proc/{pid}").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    log_path = run_dir / "train.log"
    pid_path = run_dir / "train.pid"
    perclass_last = run_dir / "per_class_iou_last.json"
    perclass_best = run_dir / "per_class_iou_best.json"
    dead_last = run_dir / "dead_classes_last.json"

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"========== E4 status @ {now} ==========")
    print(f"  run-dir:    {run_dir}")

    alive = _proc_alive(pid_path)
    if alive is True:
        print(f"  process:    ALIVE (pid {pid_path.read_text().strip()})")
    elif alive is False:
        print(f"  process:    EXITED (pid {pid_path.read_text().strip()} no longer in /proc)")
    else:
        print("  process:    pid file missing — check pgrep manually")

    log_text = _tail_bytes(log_path, 100_000)
    last_line = _last_progress_line(log_text)
    if last_line is None:
        print("  progress:   (no Epoch lines yet)")
    else:
        prog = _parse_progress(last_line)
        ep = prog.get("epoch")
        ep_pct = prog.get("epoch_pct")
        step = prog.get("step")
        total = prog.get("step_total")
        itr = prog.get("it_per_s")
        print(
            f"  epoch:      {ep} / {args.max_epochs}"
            + (f"   ({ep_pct}% of epoch)" if ep_pct is not None else "")
        )
        if step is not None:
            print(f"  step:       {step} / {total}")
        if itr is not None:
            print(f"  speed:      {itr:.2f} it/s")
        if prog.get("loss_step") is not None:
            print(f"  loss_step:  {prog['loss_step']:.3f}")
        if prog.get("loss_epoch") is not None:
            print(f"  loss_epoch: {prog['loss_epoch']:.3f}  (last finished epoch)")
        if prog.get("val_miou") is not None:
            print(f"  val/miou:   {prog['val_miou']:.6e}")
        if prog.get("val_miou_0") is not None:
            print(f"  val/miou_0: {prog['val_miou_0']:.6e}")

    fmiou_last = _compute_fmiou(_load_perclass(perclass_last) or [])
    fmiou_best = _compute_fmiou(_load_perclass(perclass_best) or [])
    ep_last = _epoch_from_metafile(perclass_last)
    ep_best = _epoch_from_metafile(perclass_best)
    if fmiou_last:
        print(
            f"  f-mIoU last: {fmiou_last['f_miou']*100:.3f} %"
            f"   (epoch={ep_last}, fg={fmiou_last['fg_count']}, dead={fmiou_last['dead']})"
        )
    if fmiou_best:
        print(
            f"  f-mIoU best: {fmiou_best['f_miou']*100:.3f} %"
            f"   (epoch={ep_best}, fg={fmiou_best['fg_count']}, dead={fmiou_best['dead']})"
        )
    print()


if __name__ == "__main__":
    main()
