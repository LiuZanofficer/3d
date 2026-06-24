#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
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
    sample_path = log_path.parent / "gpu_samples.csv"
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

    with sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "gpu_util", "mem_used_mb", "mem_total_mb", "power_w"])
        writer.writeheader()
        for s in samples:
            writer.writerow(s)
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
        "bf16_error": ("bf16" in low or "bfloat16" in low) and ("not supported" in low or "unsupported" in low),
    }


def parse_steps(log_text: str, expected: int = 200) -> Tuple[int, int]:
    matches = re.findall(r"(\d+)/(\d+)\s*\[", log_text)
    if not matches:
        return 0, expected
    step, total = matches[-1]
    return int(step), int(total)


def mean_step_from_tensorboard(run_dir: Path, warmup_steps: int = 20) -> Optional[float]:
    if event_accumulator is None:
        return None
    event_files = sorted(run_dir.glob("**/events.out.tfevents*"))
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


def parse_log_timestamps(log_text: str) -> Optional[float]:
    ts = re.findall(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)\]", log_text)
    if len(ts) < 2:
        return None
    try:
        t0 = datetime.fromisoformat(ts[0]).timestamp()
        t1 = datetime.fromisoformat(ts[-1]).timestamp()
        if t1 > t0:
            return t1 - t0
    except Exception:
        return None
    return None


@dataclass
class ProbeResult:
    source: str
    precision: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    return_code: int
    timeout: bool
    duration_sec: Optional[float]
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
        source="new",
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


def load_existing_combo(base_runs: List[Path], batch_size: int, num_workers: int) -> Optional[ProbeResult]:
    for base in base_runs:
        pattern = str(base / "probe_runs" / f"p_*_b{batch_size}_w{num_workers}")
        candidates = sorted(glob.glob(pattern), reverse=True)
        for c in candidates:
            run_dir = Path(c)
            log_path = run_dir / "train.log"
            if not log_path.exists():
                continue
            text = read_text(log_path)
            if "EXIT_CODE:" not in text:
                continue
            rc_match = re.findall(r"EXIT_CODE:\s*(-?\d+)", text)
            to_match = re.findall(r"TIMEOUT:\s*(True|False)", text)
            if not rc_match:
                continue
            rc = int(rc_match[-1])
            timed_out = (to_match[-1] == "True") if to_match else False
            fails = detect_failures(text)
            steps_completed, steps_total = parse_steps(text, expected=200)
            duration = parse_log_timestamps(text)
            mean_step = mean_step_from_tensorboard(run_dir, warmup_steps=20)
            if mean_step is None and duration and steps_completed > 0:
                effective_steps = max(1, steps_completed - 20)
                effective_time = duration * (effective_steps / max(1, steps_completed))
                mean_step = effective_time / float(effective_steps)
            sps = (batch_size / mean_step) if mean_step and mean_step > 0 else None
            stable = (not timed_out) and (rc == 0) and (not fails["oom"]) and (not fails["worker_killed"]) and (steps_completed >= 200)
            prec = "bf16-mixed" if "p_bf16mixed_" in run_dir.name else "16-mixed"
            return ProbeResult(
                source=f"reuse:{base.name}",
                precision=prec,
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
                gpu_util_median=None,
                gpu_mem_peak_mb=None,
                oom=fails["oom"],
                worker_killed=fails["worker_killed"],
                stable_200_steps=stable,
                train_log=str(log_path),
                run_dir=str(run_dir),
            )
    return None


def choose_best_batch(step1: List[ProbeResult]) -> Optional[int]:
    candidates = [r for r in step1 if r.stable_200_steps and r.samples_per_sec is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda r: float(r.samples_per_sec), reverse=True)
    return int(candidates[0].batch_size)


def choose_best_combo(candidates: List[ProbeResult]) -> Optional[ProbeResult]:
    good = [r for r in candidates if r.stable_200_steps and r.samples_per_sec is not None]
    if not good:
        return None
    good.sort(
        key=lambda r: (
            float(r.samples_per_sec),
            float(r.gpu_util_median or 0.0),
            float(81920.0 - (r.gpu_mem_peak_mb or 81920.0)),
        ),
        reverse=True,
    )
    best = good[0]
    near = [r for r in good if (best.samples_per_sec - r.samples_per_sec) / best.samples_per_sec < 0.05]
    if len(near) == 1:
        return best
    near.sort(
        key=lambda r: (
            float(r.gpu_util_median or 0.0),
            float(81920.0 - (r.gpu_mem_peak_mb or 81920.0)),
        ),
        reverse=True,
    )
    return near[0]


def run_one_epoch_verify(out_dir: Path, best: ProbeResult, timeout_sec: int = 7200) -> Dict:
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
        cmd, cwd=REPO_DIR, log_path=log_path, timeout_sec=timeout_sec
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
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--base-runs", nargs="*", default=[])
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_ROOT / f"a800_throughput_tune_{now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_nofile_limit(8192)

    base_runs = [Path(p) for p in args.base_runs if Path(p).exists()]
    precision = "bf16-mixed"

    step1_targets = [(2, 8), (4, 8), (6, 8), (8, 8)]
    step2_targets: List[Tuple[int, int]] = []

    all_results: List[ProbeResult] = []

    # Step 1
    for b, w in step1_targets:
        reused = load_existing_combo(base_runs, b, w)
        if reused is not None and reused.stable_200_steps:
            all_results.append(reused)
        else:
            if reused is not None:
                all_results.append(reused)
        result = run_probe_combo(out_dir, precision, b, w, timeout_sec=args.timeout_sec)
        # precision fallback
        if result.return_code != 0 and result.steps_completed == 0:
            text = read_text(Path(result.train_log))
            if detect_failures(text).get("bf16_error"):
                precision = "16-mixed"
                result = run_probe_combo(out_dir, precision, b, w, timeout_sec=args.timeout_sec)
        all_results.append(result)

    step1_results = [r for r in all_results if (r.batch_size, r.num_workers) in step1_targets]
    best_batch = choose_best_batch(step1_results)

    if best_batch is not None:
        step2_targets = [(best_batch, 4), (best_batch, 12)]
        for b, w in step2_targets:
            reused = load_existing_combo(base_runs, b, w)
            if reused is not None and reused.stable_200_steps:
                all_results.append(reused)
            else:
                if reused is not None:
                    all_results.append(reused)
            result = run_probe_combo(out_dir, precision, b, w, timeout_sec=args.timeout_sec)
            all_results.append(result)

    # best over selected batch workers [4,8,12]
    final_candidates = []
    if best_batch is not None:
        for w in [4, 8, 12]:
            for r in all_results:
                if r.batch_size == best_batch and r.num_workers == w:
                    final_candidates.append(r)
                    break
    best = choose_best_combo(final_candidates) if final_candidates else None

    result_rows = [asdict(r) for r in all_results]
    with (out_dir / "probe_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "scheme": "economy_4_2_1",
                "step1_targets": step1_targets,
                "step2_targets": step2_targets,
                "results": result_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with (out_dir / "probe_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()) if result_rows else [])
        if result_rows:
            writer.writeheader()
            writer.writerows(result_rows)

    best_path = out_dir / "best_config.json"
    rec_path = out_dir / "recommended_overrides.txt"
    verify_path = out_dir / "one_epoch_verify.json"
    summary_path = out_dir / "THROUGHPUT_TUNING_SUMMARY.md"

    if best is None:
        best_path.write_text(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": "no stable combo from economy 4+2 sweep",
                    "best_batch_from_step1": best_batch,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        verify_path.write_text(json.dumps({"status": "SKIPPED"}, indent=2), encoding="utf-8")
        rec_path.write_text("No stable config found.\n", encoding="utf-8")
        summary_path.write_text(
            "\n".join(
                [
                    "# THROUGHPUT_TUNING_SUMMARY",
                    "- FINAL_DECISION: BLOCKED",
                    f"- best_batch_from_step1: {best_batch}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"final_decision": "BLOCKED", "out_dir": str(out_dir)}, ensure_ascii=False))
        return 0

    best_path.write_text(json.dumps(asdict(best), indent=2, ensure_ascii=False), encoding="utf-8")
    rec_lines = [
        f"+trainer.precision={best.precision}",
        f"data.batch_size={best.batch_size}",
        f"data.num_workers={best.num_workers}",
        "data.pin_memory=true",
        "++data.persistent_workers=true",
        "++data.prefetch_factor=4",
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

    verify = run_one_epoch_verify(out_dir, best)
    verify_path.write_text(json.dumps(verify, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "# THROUGHPUT_TUNING_SUMMARY",
        "- FINAL_DECISION: READY" if verify.get("return_code") == 0 and not verify.get("timeout") else "- FINAL_DECISION: BLOCKED",
        f"- selected_precision: {best.precision}",
        f"- selected_batch_size: {best.batch_size}",
        f"- selected_num_workers: {best.num_workers}",
        f"- selected_pin_memory/persistent_workers/prefetch_factor: {best.pin_memory}/{best.persistent_workers}/{best.prefetch_factor}",
        f"- best_samples_per_sec: {best.samples_per_sec}",
        f"- one_epoch_time_min: {verify.get('one_epoch_time_min')}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "final_decision": "READY" if verify.get("return_code") == 0 and not verify.get("timeout") else "BLOCKED",
                "out_dir": str(out_dir),
                "selected_precision": best.precision,
                "selected_batch_size": best.batch_size,
                "selected_num_workers": best.num_workers,
                "best_samples_per_sec": best.samples_per_sec,
                "one_epoch_time_min": verify.get("one_epoch_time_min"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
