#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
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


def set_nofile_limit(target_soft: int = 8192) -> Optional[Tuple[int, int]]:
    if resource is None:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, max(soft, target_soft))
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        return resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return None


def query_gpu() -> Dict[str, Optional[float]]:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        row = res.stdout.strip().splitlines()[0]
        util, mem_used, mem_total, power = [x.strip() for x in row.split(",")]
        return {
            "gpu_util": float(util.replace("%", "").strip()),
            "mem_used_mb": float(mem_used.replace("MiB", "").strip()),
            "mem_total_mb": float(mem_total.replace("MiB", "").strip()),
            "power_w": float(power.replace("W", "").strip()),
        }
    except Exception:
        return {"gpu_util": None, "mem_used_mb": None, "mem_total_mb": None, "power_w": None}


def run_cmd_with_gpu_sampling(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    timeout_sec: int,
    sample_sec: float = 1.0,
) -> Tuple[int, float, bool, List[Dict[str, float]]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    samples: List[Dict[str, float]] = []
    with log_path.open("w", encoding="utf-8") as logf:
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
        while True:
            rc = proc.poll()
            g = query_gpu()
            g["t"] = time.time() - start
            samples.append(g)  # type: ignore[arg-type]
            if rc is not None:
                break
            if (time.time() - start) > timeout_sec:
                timed_out = True
                break
            time.sleep(sample_sec)
        if timed_out:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                proc.wait(timeout=20)
        end_rc = proc.poll()
        if end_rc is None:
            end_rc = -9
        logf.write(f"[{datetime.now().isoformat()}] EXIT_CODE: {end_rc}\n")
        logf.write(f"[{datetime.now().isoformat()}] TIMEOUT: {timed_out}\n")
        logf.flush()
    return end_rc, time.time() - start, timed_out, samples


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def detect_failures(log_text: str) -> Dict[str, bool]:
    low = log_text.lower()
    return {
        "oom": ("out of memory" in low) or ("cuda error: out of memory" in low),
        "worker_killed": (
            ("dataloader worker" in low and "exited unexpectedly" in low)
            or ("pin memory thread exited unexpectedly" in low)
            or ("worker (pid" in low and "killed" in low)
        ),
    }


def parse_steps(log_text: str, expected: int = 200) -> Tuple[int, int]:
    matches = re.findall(r"(\d+)/(\d+)\s*\[", log_text)
    if not matches:
        return 0, expected
    step, total = matches[-1]
    return int(step), int(total)


def get_event_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("**/events.out.tfevents*"))


def mean_step_from_tensorboard(run_dir: Path, warmup_steps: int = 20) -> Optional[float]:
    if event_accumulator is None:
        return None
    event_files = get_event_files(run_dir)
    if not event_files:
        return None
    all_points: List[Tuple[int, float]] = []
    for ef in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(ef), size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            if "train/loss_step" not in tags:
                continue
            for ev in ea.Scalars("train/loss_step"):
                all_points.append((int(ev.step), float(ev.wall_time)))
        except Exception:
            continue
    if not all_points:
        return None
    all_points.sort(key=lambda x: (x[0], x[1]))
    dedup: Dict[int, float] = {}
    for step, wall in all_points:
        dedup[step] = wall
    steps = sorted(dedup.keys())
    if len(steps) < 5:
        return None
    start_candidates = [s for s in steps if s >= warmup_steps]
    if not start_candidates:
        return None
    start_step = start_candidates[0]
    end_step = steps[-1]
    if end_step <= start_step:
        return None
    dt = dedup[end_step] - dedup[start_step]
    ds = end_step - start_step
    if dt <= 0 or ds <= 0:
        return None
    return dt / float(ds)


def gpu_stats(samples: List[Dict[str, float]]) -> Tuple[Optional[float], Optional[float]]:
    utils = [s["gpu_util"] for s in samples if s.get("gpu_util") is not None]
    mems = [s["mem_used_mb"] for s in samples if s.get("mem_used_mb") is not None]
    util_median = float(statistics.median(utils)) if utils else None
    mem_peak = float(max(mems)) if mems else None
    return util_median, mem_peak


@dataclass
class ProbeResult:
    precision: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    return_code: int
    timeout: bool
    duration_sec: float
    steps_completed: int
    steps_total: int
    mean_step_sec: Optional[float]
    samples_per_sec: Optional[float]
    gpu_util_median: Optional[float]
    gpu_mem_peak_mb: Optional[float]
    oom: bool
    worker_killed: bool
    stable_200_steps: bool
    train_log: str
    run_dir: str


def build_base_overrides(
    run_dir: Path,
    batch_size: int,
    num_workers: int,
    precision: str,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> List[str]:
    return [
        "model=spunet34c_lz",
        "data=sc_memory_check",
        f"paths.root_dir={REPO_DIR}",
        f"paths.data_dir={DATA_DIR}",
        f"paths.output_dir={run_dir}",
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
        f"+trainer.precision={precision}",
        f"data.batch_size={batch_size}",
        f"data.num_workers={num_workers}",
        f"data.pin_memory={'true' if pin_memory else 'false'}",
        f"++data.persistent_workers={'true' if persistent_workers else 'false'}",
        f"++data.prefetch_factor={prefetch_factor}",
    ]


def pick_precision(out_dir: Path) -> str:
    test_dir = out_dir / "precision_probe"
    test_log = test_dir / "train.log"
    cmd = [PYTHON_BIN, "src/train.py"] + build_base_overrides(
        run_dir=test_dir,
        batch_size=2,
        num_workers=4,
        precision="bf16-mixed",
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    ) + [
        "trainer.max_epochs=1",
        "+trainer.max_steps=20",
        "+trainer.limit_train_batches=20",
        "+trainer.limit_val_batches=0",
        "+trainer.num_sanity_val_steps=0",
        "callbacks.model_checkpoint.save_last=false",
        "callbacks.model_checkpoint.save_top_k=0",
    ]
    rc, _, timeout, _ = run_cmd_with_gpu_sampling(cmd, cwd=REPO_DIR, log_path=test_log, timeout_sec=900)
    text = read_text(test_log).lower()
    if rc == 0 and not timeout:
        return "bf16-mixed"
    if "bf16" in text or "bfloat16" in text:
        return "16-mixed"
    return "bf16-mixed"


def run_probe_combo(
    out_dir: Path,
    precision: str,
    batch_size: int,
    num_workers: int,
    timeout_sec: int,
) -> ProbeResult:
    run_dir = out_dir / "probe_runs" / f"p_{precision.replace('-', '')}_b{batch_size}_w{num_workers}"
    log_path = run_dir / "train.log"
    cmd = [PYTHON_BIN, "src/train.py"] + build_base_overrides(
        run_dir=run_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    ) + [
        "trainer.max_epochs=1",
        "+trainer.max_steps=200",
        "+trainer.limit_train_batches=200",
        "+trainer.limit_val_batches=0",
        "+trainer.num_sanity_val_steps=0",
        "callbacks.model_checkpoint.save_last=false",
        "callbacks.model_checkpoint.save_top_k=0",
    ]

    rc, duration, timed_out, samples = run_cmd_with_gpu_sampling(
        cmd, cwd=REPO_DIR, log_path=log_path, timeout_sec=timeout_sec
    )
    text = read_text(log_path)
    fails = detect_failures(text)
    steps_completed, steps_total = parse_steps(text, expected=200)
    mean_step = mean_step_from_tensorboard(run_dir, warmup_steps=20)
    if mean_step is None and steps_completed > 0:
        effective_steps = max(1, steps_completed - 20)
        effective_time = duration * (effective_steps / max(1, steps_completed))
        mean_step = effective_time / float(effective_steps)
    sps = (batch_size / mean_step) if mean_step and mean_step > 0 else None
    util_median, mem_peak = gpu_stats(samples)
    stable = (not timed_out) and (rc == 0) and (not fails["oom"]) and (not fails["worker_killed"]) and (steps_completed >= 200)

    return ProbeResult(
        precision=precision,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        return_code=rc,
        timeout=timed_out,
        duration_sec=duration,
        steps_completed=steps_completed,
        steps_total=steps_total,
        mean_step_sec=mean_step,
        samples_per_sec=sps,
        gpu_util_median=util_median,
        gpu_mem_peak_mb=mem_peak,
        oom=fails["oom"],
        worker_killed=fails["worker_killed"],
        stable_200_steps=stable,
        train_log=str(log_path),
        run_dir=str(run_dir),
    )


def choose_best(results: List[ProbeResult]) -> Optional[ProbeResult]:
    candidates = [r for r in results if r.stable_200_steps and r.samples_per_sec is not None]
    if not candidates:
        return None

    def key(r: ProbeResult):
        return (
            round(float(r.samples_per_sec), 8),
            round(float(r.gpu_util_median or 0.0), 8),
            round(float((81920.0 - (r.gpu_mem_peak_mb or 81920.0))), 8),
        )

    candidates.sort(key=key, reverse=True)
    best = candidates[0]
    near = [c for c in candidates if (best.samples_per_sec - c.samples_per_sec) / best.samples_per_sec < 0.05]
    if len(near) == 1:
        return best
    near.sort(
        key=lambda r: (
            round(float(r.gpu_util_median or 0.0), 8),
            round(float(81920.0 - (r.gpu_mem_peak_mb or 81920.0)), 8),
        ),
        reverse=True,
    )
    return near[0]


def run_one_epoch_verify(out_dir: Path, best: ProbeResult) -> Dict:
    verify_dir = out_dir / "one_epoch_verify"
    log_path = verify_dir / "train.log"
    cmd = [PYTHON_BIN, "src/train.py"] + build_base_overrides(
        run_dir=verify_dir,
        batch_size=best.batch_size,
        num_workers=best.num_workers,
        precision=best.precision,
        pin_memory=best.pin_memory,
        persistent_workers=best.persistent_workers,
        prefetch_factor=best.prefetch_factor,
    ) + [
        "trainer.max_epochs=1",
        "+trainer.check_val_every_n_epoch=5",
        "+trainer.num_sanity_val_steps=0",
        "callbacks.model_checkpoint.save_last=true",
        "callbacks.model_checkpoint.save_top_k=1",
        "+callbacks.model_checkpoint.every_n_train_steps=1000",
    ]
    rc, duration, timed_out, samples = run_cmd_with_gpu_sampling(
        cmd, cwd=REPO_DIR, log_path=log_path, timeout_sec=7200
    )
    text = read_text(log_path)
    steps_completed, steps_total = parse_steps(text, expected=0)
    mean_step = mean_step_from_tensorboard(verify_dir, warmup_steps=20)
    if mean_step is None and steps_completed > 0:
        mean_step = duration / float(steps_completed)
    sps = (best.batch_size / mean_step) if mean_step and mean_step > 0 else None
    util_median, mem_peak = gpu_stats(samples)
    return {
        "return_code": rc,
        "timeout": timed_out,
        "duration_sec": duration,
        "one_epoch_time_min": duration / 60.0,
        "steps_completed": steps_completed,
        "steps_total": steps_total,
        "batch_size": best.batch_size,
        "num_workers": best.num_workers,
        "precision": best.precision,
        "pin_memory": best.pin_memory,
        "persistent_workers": best.persistent_workers,
        "prefetch_factor": best.prefetch_factor,
        "mean_step_sec": mean_step,
        "samples_per_sec": sps,
        "gpu_util_median": util_median,
        "gpu_mem_peak_mb": mem_peak,
        "train_log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--timeout-sec", type=int, default=1500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_ROOT / f"a800_throughput_tune_{now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    limit_info = set_nofile_limit(8192)
    precision = pick_precision(out_dir)
    if precision not in {"bf16-mixed", "16-mixed"}:
        precision = "16-mixed"

    batches = [2, 4, 6, 8]
    workers = [4, 8, 12, 16]
    results: List[ProbeResult] = []

    for batch_size in batches:
        for num_workers in workers:
            result = run_probe_combo(
                out_dir=out_dir,
                precision=precision,
                batch_size=batch_size,
                num_workers=num_workers,
                timeout_sec=args.timeout_sec,
            )
            results.append(result)

    result_dicts = [asdict(r) for r in results]
    with (out_dir / "probe_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "nofile_limit": limit_info,
                "selected_precision": precision,
                "results": result_dicts,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (out_dir / "probe_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_dicts[0].keys()) if result_dicts else [])
        if result_dicts:
            writer.writeheader()
            writer.writerows(result_dicts)

    best = choose_best(results)
    best_json_path = out_dir / "best_config.json"
    one_epoch_path = out_dir / "one_epoch_verify.json"
    rec_path = out_dir / "recommended_overrides.txt"
    summary_path = out_dir / "THROUGHPUT_TUNING_SUMMARY.md"

    if best is None:
        fail = {
            "status": "BLOCKED",
            "reason": "No stable 200-step combo passed.",
            "selected_precision": precision,
        }
        best_json_path.write_text(json.dumps(fail, indent=2, ensure_ascii=False), encoding="utf-8")
        one_epoch_path.write_text(json.dumps({"status": "SKIPPED"}, indent=2), encoding="utf-8")
        rec_path.write_text("No stable config found.\n", encoding="utf-8")
        summary_path.write_text(
            "\n".join(
                [
                    "# THROUGHPUT_TUNING_SUMMARY",
                    "- FINAL_DECISION: BLOCKED",
                    f"- selected_precision: {precision}",
                    "- reason: no stable 200-step combo passed",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"final_decision": "BLOCKED", "out_dir": str(out_dir)}, ensure_ascii=False))
        return 0

    best_payload = asdict(best)
    best_json_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rec_lines = [
        f"+trainer.precision={best.precision}",
        f"data.batch_size={best.batch_size}",
        f"data.num_workers={best.num_workers}",
        f"data.pin_memory={'true' if best.pin_memory else 'false'}",
        f"++data.persistent_workers={'true' if best.persistent_workers else 'false'}",
        f"++data.prefetch_factor={best.prefetch_factor}",
        "model=spunet34c_lz",
        "data=sc_memory_check",
        "data.train_dataset.split=train",
        "data.val_datasets.0.split=train",
        "model.loss.weights.seg_loss=1.0",
        "model.loss.weights.instance_loss=0.0",
        "model.loss.weights.hpza_loss=0.0",
        "model.loss.seg_loss.lovasz_weight=0.0",
        "model.text_encoder_cfg.use_clip=false",
        "+model.eval_cfg.eval_bn_batch_stats=true",
    ]
    rec_path.write_text("\n".join(rec_lines) + "\n", encoding="utf-8")

    one_epoch = run_one_epoch_verify(out_dir, best)
    one_epoch_path.write_text(json.dumps(one_epoch, indent=2, ensure_ascii=False), encoding="utf-8")

    probe_sps = best.samples_per_sec or 0.0
    verify_sps = one_epoch.get("samples_per_sec") or 0.0
    rel_err = abs(verify_sps - probe_sps) / probe_sps if probe_sps > 0 else None
    ready = (
        one_epoch.get("return_code") == 0
        and (not one_epoch.get("timeout"))
        and (rel_err is None or rel_err <= 0.15)
    )

    summary_lines = [
        "# THROUGHPUT_TUNING_SUMMARY",
        f"- FINAL_DECISION: {'READY' if ready else 'BLOCKED'}",
        f"- selected_precision: {best.precision}",
        f"- selected_batch_size: {best.batch_size}",
        f"- selected_num_workers: {best.num_workers}",
        f"- selected_pin_memory/persistent_workers/prefetch_factor: {best.pin_memory}/{best.persistent_workers}/{best.prefetch_factor}",
        f"- best_samples_per_sec: {best.samples_per_sec}",
        f"- one_epoch_time_min: {one_epoch.get('one_epoch_time_min')}",
        f"- one_epoch_samples_per_sec: {one_epoch.get('samples_per_sec')}",
        f"- probe_vs_verify_relative_error: {rel_err}",
        "",
        "## Artifacts",
        f"- {out_dir / 'probe_results.csv'}",
        f"- {out_dir / 'probe_results.json'}",
        f"- {best_json_path}",
        f"- {rec_path}",
        f"- {one_epoch_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "final_decision": "READY" if ready else "BLOCKED",
                "out_dir": str(out_dir),
                "selected_precision": best.precision,
                "selected_batch_size": best.batch_size,
                "selected_num_workers": best.num_workers,
                "best_samples_per_sec": best.samples_per_sec,
                "one_epoch_time_min": one_epoch.get("one_epoch_time_min"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
