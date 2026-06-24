#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    import resource
except Exception:
    resource = None

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:
    event_accumulator = None


PYTHON_BIN = "/root/autodl-tmp/conda-envs/llz/bin/python"
REPO_DIR = Path("/root/lz")
DATA_DIR = "/root/autodl-tmp/datasets/mosaic3d/data"
OUTPUT_ROOT = Path("/root/lz_outputs")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def auto_num_workers() -> int:
    cpu = os.cpu_count() or 8
    workers = 2 if cpu >= 4 else 1
    return workers


def ensure_nofile_limit(min_soft: int = 8192) -> Optional[int]:
    if resource is None:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return None
    target = soft
    if hard == resource.RLIM_INFINITY:
        target = max(soft, min_soft)
    else:
        target = min(hard, max(soft, min_soft))
    if target > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        except Exception:
            pass
    return soft


def run_cmd(
    cmd: List[str], log_path: Path, cwd: Path, timeout_sec: Optional[int] = None
) -> Tuple[int, float, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write("\n" + "=" * 100 + "\n")
        logf.write(f"[{datetime.now().isoformat()}] CMD: {' '.join(cmd)}\n")
        logf.flush()
        env = os.environ.copy()
        env.setdefault("PYTORCH_SHARING_STRATEGY", "file_system")
        env.setdefault("OMP_NUM_THREADS", "8")
        env.setdefault("MKL_NUM_THREADS", "8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        timed_out = False
        gpu_peak = 0.0
        deadline = (start + timeout_sec) if timeout_sec else None

        while True:
            rc = proc.poll()
            gpu = query_gpu()
            mem = gpu.get("mem_used_mb")
            if mem is not None and mem > gpu_peak:
                gpu_peak = float(mem)
            if rc is not None:
                break
            if deadline is not None and time.time() > deadline:
                timed_out = True
                break
            time.sleep(1.0)

        if timed_out:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                rc = proc.wait()
        logf.write(f"[{datetime.now().isoformat()}] EXIT_CODE: {rc}\n")
        if timed_out:
            logf.write(f"[{datetime.now().isoformat()}] TIMEOUT_SEC: {timeout_sec}\n")
        logf.write(f"[{datetime.now().isoformat()}] GPU_PEAK_MB: {gpu_peak}\n")
        logf.flush()
    return rc, time.time() - start, timed_out, gpu_peak


def query_gpu() -> Dict[str, Optional[float]]:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        row = res.stdout.strip().splitlines()[0]
        util, mem_used, mem_total = [x.strip() for x in row.split(",")]
        util = util.replace("%", "").strip()
        mem_used = mem_used.replace("MiB", "").strip()
        mem_total = mem_total.replace("MiB", "").strip()
        return {
            "gpu_util": float(util),
            "mem_used_mb": float(mem_used),
            "mem_total_mb": float(mem_total),
        }
    except Exception:
        return {
            "gpu_util": None,
            "mem_used_mb": None,
            "mem_total_mb": None,
        }


def read_log_tail(path: Path, max_lines: int = 400) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        data = f.read()
    text = data.decode("utf-8", errors="ignore").replace("\r", "\n")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def parse_train_progress(log_tail: str) -> Dict[str, Optional[float]]:
    out = {
        "epoch": None,
        "step_in_epoch": None,
        "steps_per_epoch": None,
        "train_loss_step": None,
    }
    if not log_tail:
        return out

    epoch_matches = re.findall(r"Epoch\s+(\d+):", log_tail)
    if epoch_matches:
        out["epoch"] = float(epoch_matches[-1])

    step_matches = re.findall(r"\|\s*(\d+)/(\d+)\s*\[", log_tail)
    if step_matches:
        step, total = step_matches[-1]
        out["step_in_epoch"] = float(step)
        out["steps_per_epoch"] = float(total)

    loss_matches = re.findall(r"train/loss_step=([0-9.eE+\-]+)", log_tail)
    if loss_matches:
        try:
            out["train_loss_step"] = float(loss_matches[-1])
        except Exception:
            out["train_loss_step"] = None

    return out


def gather_scalar_series(input_dir: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    series: Dict[str, List[Tuple[int, float, float]]] = {}
    if event_accumulator is None:
        return series
    event_files = glob.glob(str(input_dir / "**" / "events.out.tfevents*"), recursive=True)
    for ef in event_files:
        try:
            ea = event_accumulator.EventAccumulator(ef, size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            for tag in tags:
                for ev in ea.Scalars(tag):
                    series.setdefault(tag, []).append((int(ev.step), float(ev.wall_time), float(ev.value)))
        except Exception:
            continue
    for tag, vals in series.items():
        vals.sort(key=lambda x: (x[0], x[1]))
    return series


def write_metrics_json(input_dir: Path, output_json: Path) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    series = gather_scalar_series(input_dir)
    for tag, vals in series.items():
        if not vals:
            continue
        only_vals = [x[2] for x in vals]
        metrics[tag] = {
            "best": max(only_vals),
            "last": only_vals[-1],
            "min": min(only_vals),
            "count": float(len(only_vals)),
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def latest_metric(metrics: Dict[str, Dict[str, float]], key: str) -> Optional[float]:
    if key not in metrics:
        return None
    return metrics[key].get("last")


def best_metric(metrics: Dict[str, Dict[str, float]], key: str) -> Optional[float]:
    if key not in metrics:
        return None
    return metrics[key].get("best")


def has_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class MonitorThread(threading.Thread):
    def __init__(self, out_dir: Path, train_log: Path, latest_eval_file: Path, interval_sec: int = 120):
        super().__init__(daemon=True)
        self.out_dir = out_dir
        self.train_log = train_log
        self.latest_eval_file = latest_eval_file
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.jsonl = out_dir / "monitor_live.jsonl"
        self.csv_path = out_dir / "gpu_samples.csv"
        self._csv_initialized = False

    def _read_latest_eval(self) -> Optional[float]:
        if not self.latest_eval_file.exists():
            return None
        try:
            return float(self.latest_eval_file.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def _append_csv(self, row: Dict[str, Optional[float]]) -> None:
        headers = ["timestamp", "gpu_util", "mem_used_mb", "mem_total_mb"]
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def run(self):
        while not self.stop_event.is_set():
            ts = datetime.now().isoformat()
            gpu = query_gpu()
            log_tail = read_log_tail(self.train_log)
            progress = parse_train_progress(log_tail)
            latest_eval = self._read_latest_eval()
            obj = {
                "timestamp": ts,
                "epoch": progress["epoch"],
                "step_in_epoch": progress["step_in_epoch"],
                "steps_per_epoch": progress["steps_per_epoch"],
                "train_loss_step": progress["train_loss_step"],
                "latest_eval_miou_present_gt": latest_eval,
                "gpu_util": gpu["gpu_util"],
                "mem_used_mb": gpu["mem_used_mb"],
                "mem_total_mb": gpu["mem_total_mb"],
            }
            append_jsonl(self.jsonl, obj)
            self._append_csv(
                {
                    "timestamp": ts,
                    "gpu_util": gpu["gpu_util"],
                    "mem_used_mb": gpu["mem_used_mb"],
                    "mem_total_mb": gpu["mem_total_mb"],
                }
            )
            self.stop_event.wait(self.interval_sec)


def build_common_overrides(batch_size: int, output_dir: Path, num_workers: int) -> List[str]:
    return [
        "model=spunet34c_lz",
        "data=sc_memory_check",
        f"paths.root_dir={REPO_DIR}",
        f"paths.data_dir={DATA_DIR}",
        f"paths.output_dir={output_dir}",
        "data.train_dataset.split=train",
        "data.val_datasets.0.split=train",
        "model.loss.weights.seg_loss=1.0",
        "model.loss.weights.instance_loss=0.0",
        "model.loss.weights.hpza_loss=0.0",
        "model.loss.seg_loss.lovasz_weight=0.0",
        "model.text_encoder_cfg.use_clip=false",
        "+model.eval_cfg.eval_bn_batch_stats=true",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "+trainer.precision=16-mixed",
        f"data.num_workers={num_workers}",
        "data.pin_memory=false",
        f"data.batch_size={batch_size}",
    ]


def run_batch_probe(out_dir: Path, num_workers: int) -> Dict:
    probe_root = out_dir / "batch_probe"
    probe_root.mkdir(parents=True, exist_ok=True)
    results = []
    selected = None

    for batch in [4, 2, 1]:
        run_dir = probe_root / f"batch{batch}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        cmd = [PYTHON_BIN, "src/train.py"] + build_common_overrides(batch, run_dir, num_workers) + [
            "trainer.max_epochs=1",
            "+trainer.max_steps=100",
            "+trainer.limit_train_batches=100",
            "+trainer.limit_val_batches=0",
            "+trainer.num_sanity_val_steps=0",
            "callbacks.model_checkpoint.save_last=true",
            "callbacks.model_checkpoint.save_top_k=0",
        ]
        rc, dur, timed_out, gpu_peak_mb = run_cmd(
            cmd, log_path=log_path, cwd=REPO_DIR, timeout_sec=600
        )
        log_tail = read_log_tail(log_path, max_lines=2000)
        step_hits = re.findall(r"Epoch\s+\d+:\s+\s*\d+%\|.*?\|\s*(\d+)/100", log_tail)
        steps_completed = int(step_hits[-1]) if step_hits else 0
        denom = max(1, steps_completed)
        mean_step = dur / float(denom)
        oom = has_oom(log_path)
        stable_timeout_ok = timed_out and (steps_completed >= 10) and (mean_step < 30.0) and (not oom)
        passed = ((rc == 0) and (not oom) and (not timed_out) and (mean_step < 30.0)) or stable_timeout_ok
        if passed:
            reason = "ok"
        elif oom:
            reason = "oom"
        elif timed_out:
            reason = "timeout_or_too_slow"
        elif mean_step >= 30.0:
            reason = "mean_step_too_slow"
        else:
            reason = f"exit_{rc}"
        if passed and selected is None:
            selected = batch
        results.append(
            {
                "batch_size": batch,
                "return_code": rc,
                "duration_sec": dur,
                "steps_completed": steps_completed,
                "mean_step_sec": mean_step,
                "oom_detected": oom,
                "timed_out": timed_out,
                "gpu_peak_memory_mb": gpu_peak_mb,
                "passed": passed,
                "reason": reason,
                "log_path": str(log_path),
            }
        )
        if selected is not None:
            break

    out = {
        "probe_steps": 100,
        "selection_rule": "largest stable passing batch among [4,2,1]",
        "num_workers": num_workers,
        "results": results,
        "selected_batch_size": selected,
    }
    with (out_dir / "batch_probe.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def run_eval(
    checkpoint_path: Path, epoch: int, batch_size: int, out_dir: Path, num_workers: int
) -> Dict:
    eval_dir = out_dir / f"eval_ep{epoch:03d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_dir / "eval.log"
    cmd = [PYTHON_BIN, "src/eval.py"] + build_common_overrides(batch_size, eval_dir, num_workers) + [
        f"ckpt_path={checkpoint_path}",
    ]
    rc, dur, _, _ = run_cmd(cmd, log_path=log_path, cwd=REPO_DIR)
    metrics = write_metrics_json(eval_dir, eval_dir / "metrics.json")
    return {
        "epoch": epoch,
        "return_code": rc,
        "duration_sec": dur,
        "eval_dir": str(eval_dir),
        "miou_present_gt": latest_metric(metrics, "val/miou_present_gt"),
        "miou": latest_metric(metrics, "val/miou"),
        "miou_fg_mosaic": latest_metric(metrics, "val/miou_fg_mosaic"),
        "use_text_guidance_max": best_metric(metrics, "debug/use_text_guidance"),
    }


def choose_last_ckpt(train_dir: Path) -> Optional[Path]:
    ckpt_dir = train_dir / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return None


def run_pipeline(out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_log = out_dir / "train.log"
    latest_eval_file = out_dir / "latest_eval_miou_present_gt.txt"

    nofile_soft = ensure_nofile_limit(min_soft=8192)
    num_workers = auto_num_workers()
    fixed_protocol = {
        "model": "spunet34c_lz",
        "data": "sc_memory_check",
        "train_split": "train",
        "val_split": "train",
        "ce_only": {
            "seg_loss": 1.0,
            "instance_loss": 0.0,
            "hpza_loss": 0.0,
            "lovasz_weight": 0.0,
        },
        "eval_bn_batch_stats": True,
        "eval_interval_epochs": 5,
        "num_workers": num_workers,
        "nofile_soft_limit": nofile_soft,
    }
    with (out_dir / "fixed_protocol.json").open("w", encoding="utf-8") as f:
        json.dump(fixed_protocol, f, indent=2, ensure_ascii=False)

    probe = run_batch_probe(out_dir, num_workers=num_workers)
    selected_batch = probe.get("selected_batch_size")
    if selected_batch is None:
        result = {
            "status": "FAILED_BEFORE_TRAIN",
            "reason": "no batch size passed probe",
            "selected_batch_size": None,
            "pass_fail": "FAIL",
        }
        with (out_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    monitor = MonitorThread(out_dir=out_dir, train_log=train_log, latest_eval_file=latest_eval_file, interval_sec=120)
    monitor.start()

    eval_results = []
    train_failed = False
    train_fail_reason = ""
    ckpt_path: Optional[Path] = None

    try:
        for epoch_end in range(5, 101, 5):
            cmd = [PYTHON_BIN, "src/train.py"] + build_common_overrides(
                selected_batch, train_dir, num_workers
            ) + [
                f"trainer.max_epochs={epoch_end}",
                "+trainer.check_val_every_n_epoch=5",
                "callbacks.model_checkpoint.save_last=true",
                "callbacks.model_checkpoint.save_top_k=3",
                "+callbacks.model_checkpoint.every_n_train_steps=1000",
            ]
            if ckpt_path is not None:
                cmd.append(f"ckpt_path={ckpt_path}")
            rc, _, _, _ = run_cmd(cmd, log_path=train_log, cwd=REPO_DIR)
            if rc != 0:
                train_failed = True
                train_fail_reason = f"train_exit_{rc}_epoch_{epoch_end}"
                break

            ckpt_path = choose_last_ckpt(train_dir)
            if ckpt_path is None:
                train_failed = True
                train_fail_reason = f"missing_ckpt_after_epoch_{epoch_end}"
                break

            eval_info = run_eval(
                checkpoint_path=ckpt_path,
                epoch=epoch_end,
                batch_size=selected_batch,
                out_dir=out_dir,
                num_workers=num_workers,
            )
            eval_results.append(eval_info)
            if eval_info.get("miou_present_gt") is not None:
                latest_eval_file.write_text(f"{eval_info['miou_present_gt']}", encoding="utf-8")

            train_metrics = write_metrics_json(train_dir, train_dir / "metrics.json")
            train_best = best_metric(train_metrics, "val/miou_present_gt")
            eval_best = max([x["miou_present_gt"] for x in eval_results if x.get("miou_present_gt") is not None], default=None)
            if train_best is not None and eval_best is not None and train_best >= 0.8 and eval_best >= 0.8:
                break

    finally:
        monitor.stop_event.set()
        monitor.join(timeout=10)

    train_metrics = write_metrics_json(train_dir, train_dir / "metrics.json")
    eval_best = max([x["miou_present_gt"] for x in eval_results if x.get("miou_present_gt") is not None], default=None)
    eval_last = eval_results[-1]["miou_present_gt"] if eval_results else None
    train_best = best_metric(train_metrics, "val/miou_present_gt")
    train_last = latest_metric(train_metrics, "val/miou_present_gt")

    use_text_train = best_metric(train_metrics, "debug/use_text_guidance")
    use_text_eval_vals = [x.get("use_text_guidance_max") for x in eval_results if x.get("use_text_guidance_max") is not None]
    use_text_eval_max = max(use_text_eval_vals) if use_text_eval_vals else 0.0
    use_text_all_zero = (use_text_train is None or use_text_train == 0.0) and (use_text_eval_max == 0.0)

    passed = (
        (not train_failed)
        and (train_best is not None and train_best >= 0.8)
        and (eval_best is not None and eval_best >= 0.8)
    )

    result = {
        "status": "DONE" if not train_failed else "FAILED_DURING_TRAIN",
        "train_fail_reason": train_fail_reason,
        "selected_batch_size": selected_batch,
        "selected_num_workers": num_workers,
        "train_best_val_miou_present_gt": train_best,
        "train_last_val_miou_present_gt": train_last,
        "eval_best_val_miou_present_gt": eval_best,
        "eval_last_val_miou_present_gt": eval_last,
        "debug_use_text_guidance_all_zero": use_text_all_zero,
        "eval_nodes": eval_results,
        "pass_fail": "PASS" if passed else "FAIL",
        "artifacts": {
            "train_log": str(train_log),
            "batch_probe_json": str(out_dir / "batch_probe.json"),
            "monitor_live_jsonl": str(out_dir / "monitor_live.jsonl"),
            "gpu_samples_csv": str(out_dir / "gpu_samples.csv"),
            "train_metrics_json": str(train_dir / "metrics.json"),
        },
    }

    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "# CAPACITY_CHECK_SUMMARY",
        f"- final_decision: {result['pass_fail']}",
        f"- selected_batch_size: {selected_batch}",
        f"- train_best_val/miou_present_gt: {train_best}",
        f"- eval_best_val/miou_present_gt: {eval_best}",
        f"- debug/use_text_guidance_all_zero: {use_text_all_zero}",
        f"- train_failed: {train_failed}",
        f"- fail_reason: {train_fail_reason}",
        "",
        "## Artifacts",
        f"- {out_dir / 'train.log'}",
        f"- {out_dir / 'batch_probe.json'}",
        f"- {out_dir / 'monitor_live.jsonl'}",
        f"- {out_dir / 'gpu_samples.csv'}",
        f"- {out_dir / 'result.json'}",
    ]
    with (out_dir / "CAPACITY_CHECK_SUMMARY.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_ROOT / f"sc_full_capacity_check_{now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RUNNING").write_text(datetime.now().isoformat(), encoding="utf-8")
    try:
        result = run_pipeline(out_dir=out_dir)
    except Exception as e:
        fail = {
            "status": "CRASHED",
            "error": str(e),
            "pass_fail": "FAIL",
        }
        with (out_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(fail, f, indent=2, ensure_ascii=False)
        with (out_dir / "CAPACITY_CHECK_SUMMARY.md").open("w", encoding="utf-8") as f:
            f.write(f"# CAPACITY_CHECK_SUMMARY\n- final_decision: FAIL\n- error: {e}\n")
        raise
    finally:
        running = out_dir / "RUNNING"
        if running.exists():
            running.unlink()

    print(json.dumps({"out_dir": str(out_dir), "final_decision": result.get("pass_fail")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
