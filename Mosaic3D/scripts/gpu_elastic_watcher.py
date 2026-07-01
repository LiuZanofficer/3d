#!/usr/bin/env python3
"""Elastic GPU watcher for Method B (hard-negative caption loss) training.

Runs on the HOST (where nvidia-smi works). It launches / manages the training
process INSIDE the docker container `mosaic3d-coeus-torch` via `docker exec`.

Guarantees requested by the user:
1. Epoch continuity: every (re)launch resumes from RUN_DIR/checkpoints/last.ckpt
   once it exists, so current_epoch / optimizer / scheduler continue (e.g. a run
   that reached ep32 on 2 cards continues at ep33 on more cards, never from 0).
   Cold start (no last.ckpt yet) loads pretrained weights via init_ckpt_path.
2. Adaptive crash handling (never give up): if a multi-GPU launch crashes, the
   watcher retries and, if needed, backs off to fewer cards (N -> ... -> 1) until
   it finds a config that makes progress. It always resumes from last.ckpt.
3. Scale up: when more cards become stably free, it gracefully stops (SIGUSR1 ->
   checkpoint) and relaunches with more cards.

Done condition: the inner python exits with return code 0 while we were NOT
expecting a graceful stop -> training finished (reached max_epochs).
"""

import argparse
import os
import random
import subprocess
import sys
import time
from datetime import datetime

CONTAINER = "mosaic3d-coeus-torch"
PROJECT_HOST = "/mnt/sdc/lz/3d/Mosaic3D"
PROJECT_CTR = "/workspace/Mosaic3D"
INNER = os.path.join(PROJECT_CTR, "scripts/launch_hardneg_inner.sh")

# Filled in by main() once --run-name is known.
RUN_DIR_HOST = ""
RUN_DIR_CTR = ""
STATUS_PATH = ""
PID_PATH = ""
CKPT_PATH = ""
ATTEMPTS_DIR = ""
WATCHER_LOG = ""
CKPT_EVERY = 200


def _set_run_paths(run_name):
    global RUN_DIR_HOST, RUN_DIR_CTR, STATUS_PATH, PID_PATH, CKPT_PATH
    global ATTEMPTS_DIR, WATCHER_LOG
    rel = os.path.join("logs/train/runs", run_name)
    RUN_DIR_HOST = os.path.join(PROJECT_HOST, rel)
    RUN_DIR_CTR = os.path.join(PROJECT_CTR, rel)
    STATUS_PATH = os.path.join(RUN_DIR_HOST, "status.txt")
    PID_PATH = os.path.join(RUN_DIR_HOST, "train.pid")
    CKPT_PATH = os.path.join(RUN_DIR_HOST, "checkpoints", "last.ckpt")
    ATTEMPTS_DIR = os.path.join(RUN_DIR_HOST, "attempts")
    WATCHER_LOG = os.path.join(RUN_DIR_HOST, "elastic_watcher.log")

_logf = None


def log(msg):
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    if _logf:
        _logf.write(line + "\n")
        _logf.flush()


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def free_gpus(min_free_mib, max_util):
    """Return dict {index: (free_mib, util)} for all GPUs from host nvidia-smi."""
    r = sh("nvidia-smi --query-gpu=index,memory.free,utilization.gpu "
           "--format=csv,noheader,nounits")
    out = {}
    if r.returncode != 0:
        log(f"WARN nvidia-smi failed: {r.stderr.strip()}")
        return out
    for ln in r.stdout.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 3:
            continue
        idx, free, util = int(parts[0]), int(parts[1]), int(parts[2])
        out[idx] = (free, util)
    return out


def container_pid_alive(pid):
    r = sh(f"sudo -n docker exec {CONTAINER} kill -0 {pid} 2>/dev/null")
    return r.returncode == 0


def read_pid():
    try:
        with open(PID_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return None


def send_sigusr1():
    pid = read_pid()
    if pid is None:
        log("WARN no pidfile, cannot send SIGUSR1")
        return False
    r = sh(f"sudo -n docker exec {CONTAINER} kill -USR1 {pid}")
    log(f"sent SIGUSR1 to container pid {pid} (rc={r.returncode})")
    return r.returncode == 0


def hard_kill():
    pid = read_pid()
    if pid is not None:
        sh(f"sudo -n docker exec {CONTAINER} bash -lc 'pkill -9 -P {pid}; kill -9 {pid}' 2>/dev/null")
    # belt and suspenders: kill any stray train.py in the container
    sh(f"sudo -n docker exec {CONTAINER} bash -lc \"pkill -9 -f 'src/train.py'\" 2>/dev/null")


def launch(devices, resume, max_steps=None):
    """Start the inner training script via docker exec. Returns (Popen, logpath)."""
    os.makedirs(ATTEMPTS_DIR, exist_ok=True)
    cvd = ",".join(str(d) for d in devices)
    ndev = len(devices)
    port = random.randint(20000, 40000)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = os.path.join(ATTEMPTS_DIR, f"attempt_{ts}_n{ndev}_{cvd.replace(',', '-')}.log")
    env = (f"-e CUDA_VISIBLE_DEVICES={cvd} -e NDEV={ndev} -e RESUME={resume} "
           f"-e MASTER_PORT={port} -e RUN_DIR={RUN_DIR_CTR} -e CKPT_EVERY_STEPS={CKPT_EVERY}")
    if max_steps is not None:
        env += f" -e MAX_STEPS={max_steps}"
    cmd = f"sudo -n docker exec {env} {CONTAINER} bash {INNER}"
    log(f"LAUNCH devices={cvd} ndev={ndev} resume={resume} max_steps={max_steps} "
        f"port={port} -> {os.path.basename(logpath)}")
    lf = open(logpath, "w")
    p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)
    p._logfile = lf  # type: ignore[attr-defined]
    return p, logpath


def status_text():
    try:
        with open(STATUS_PATH) as f:
            return f.read().strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="hardneg_elastic",
                    help="run dir name under logs/train/runs/")
    ap.add_argument("--min-free-mib", type=int, default=30000,
                    help="a GPU counts as free above this free memory")
    ap.add_argument("--max-util", type=int, default=20,
                    help="a GPU counts as free below this utilization %%")
    ap.add_argument("--max-dev", type=int, default=8)
    ap.add_argument("--poll-sec", type=int, default=30)
    ap.add_argument("--stable-sec", type=int, default=90,
                    help="external card must stay free this long before we grab it")
    ap.add_argument("--min-healthy-sec", type=int, default=420,
                    help="below this lifetime a crash is treated as a startup crash")
    ap.add_argument("--max-retry-same", type=int, default=2,
                    help="mid-run crash retries at same card count before backing off")
    ap.add_argument("--upscale-cooldown-sec", type=int, default=1800,
                    help="cooldown for a card count that crashed right after upscale")
    ap.add_argument("--smoke-max-steps", type=int, default=None,
                    help="if set, pass trainer.max_steps for a quick smoke run")
    ap.add_argument("--ckpt-every-steps", type=int, default=200,
                    help="step-level checkpoint interval (controls resume granularity)")
    args = ap.parse_args()

    global _logf, CKPT_EVERY
    CKPT_EVERY = args.ckpt_every_steps
    _set_run_paths(args.run_name)
    os.makedirs(RUN_DIR_HOST, exist_ok=True)
    _logf = open(WATCHER_LOG, "a")
    log("=" * 70)
    log(f"watcher start cfg={vars(args)}")

    proc = None
    cur_devices = []
    attempt_start = 0.0
    expecting_graceful = False
    known_good_n = 0
    same_retries = 0
    healthy_marked = False          # whether current attempt was counted healthy
    pending_n = args.max_dev        # target device count for the next launch
    cooldown_until = {}             # n -> timestamp until which count n is on cooldown
    free_since = {}                 # gpu idx -> first-seen-free timestamp (external cards)

    def pick_devices(target_max):
        """Choose a device set honoring cooldown and current freeness."""
        fg = free_gpus(args.min_free_mib, args.max_util)
        avail = sorted(i for i, (f, u) in fg.items()
                       if f >= args.min_free_mib and u <= args.max_util)
        # union with cards we already hold (they read busy but are ours)
        hold = [d for d in cur_devices if d in fg]
        cand = sorted(set(avail) | set(hold))
        n = min(len(cand), target_max, args.max_dev)
        now = time.time()
        # respect cooldown: if n is cooling down, step down to a non-cooling count
        while n >= 1 and cooldown_until.get(n, 0) > now:
            n -= 1
        n = max(n, 1)
        return cand[:n]

    while True:
        # ---- no process running: single (re)launch site, driven by pending_n ----
        if proc is None:
            devices = pick_devices(pending_n)
            if not devices:
                log("no free GPU available yet; waiting")
                time.sleep(args.poll_sec)
                continue
            resume = 1 if os.path.exists(CKPT_PATH) else 0
            proc, _ = launch(devices, resume, max_steps=args.smoke_max_steps)
            cur_devices = devices
            attempt_start = time.time()
            expecting_graceful = False
            healthy_marked = False
            free_since.clear()
            time.sleep(args.poll_sec)
            continue

        rc = proc.poll()

        # ---- process still alive ----
        if rc is None:
            lifetime = time.time() - attempt_start
            if (not healthy_marked) and lifetime >= args.min_healthy_sec:
                healthy_marked = True
                same_retries = 0
                if len(cur_devices) > known_good_n:
                    known_good_n = len(cur_devices)
                log(f"config n={len(cur_devices)} proven healthy "
                    f"(lifetime={int(lifetime)}s, known_good_n={known_good_n})")

            # scale-up check: only once a checkpoint exists (so resume works)
            if os.path.exists(CKPT_PATH) and len(cur_devices) < args.max_dev:
                fg = free_gpus(args.min_free_mib, args.max_util)
                ext_free = [i for i, (f, u) in fg.items()
                            if f >= args.min_free_mib and u <= args.max_util
                            and i not in cur_devices]
                now = time.time()
                for i in list(free_since):
                    if i not in ext_free:
                        free_since.pop(i, None)
                for i in ext_free:
                    free_since.setdefault(i, now)
                stable_ext = [i for i in ext_free
                              if now - free_since.get(i, now) >= args.stable_sec]
                target_n = min(len(cur_devices) + len(stable_ext), args.max_dev)
                if target_n > len(cur_devices) and cooldown_until.get(target_n, 0) <= now:
                    log(f"scale-up: {len(cur_devices)} -> {target_n} "
                        f"(stable extra cards {stable_ext}); graceful stopping")
                    expecting_graceful = True
                    pending_n = target_n
                    send_sigusr1()
                    t0 = time.time()
                    while proc.poll() is None and time.time() - t0 < 900:
                        time.sleep(10)
                    if proc.poll() is None:
                        log("graceful stop timed out; hard killing")
                        hard_kill()
                        time.sleep(15)
                    proc._logfile.close()
                    proc = None
                    continue
            time.sleep(args.poll_sec)
            continue

        # ---- process exited ----
        proc._logfile.close()
        proc = None
        st = status_text()
        lifetime = time.time() - attempt_start
        log(f"process exited rc={rc} status='{st}' lifetime={int(lifetime)}s "
            f"devices={cur_devices} expecting_graceful={expecting_graceful}")

        if expecting_graceful:
            # planned stop for scale-up; pending_n already set, relaunch next loop
            continue

        if rc == 0:
            log("training FINISHED (clean exit rc=0). watcher done.")
            break

        # crash: classify and adapt (never stop)
        hard_kill()
        time.sleep(10)
        n_now = len(cur_devices)
        if lifetime >= args.min_healthy_sec:
            same_retries += 1
            if same_retries <= args.max_retry_same:
                log(f"mid-run crash; retry same n={n_now} "
                    f"(retry {same_retries}/{args.max_retry_same})")
                pending_n = n_now
                continue
            log("mid-run crash retries exhausted; backing off card count")
        else:
            log("startup-period crash; backing off card count")

        # backoff: cool down the failing count and drop to fewer cards
        cooldown_until[n_now] = time.time() + args.upscale_cooldown_sec
        same_retries = 0
        pending_n = max(n_now - 1, 1)
        log(f"backoff: next target n<={pending_n} (cooldown n={n_now} "
            f"for {args.upscale_cooldown_sec}s)")

    log("watcher exiting")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("watcher interrupted by KeyboardInterrupt")
        sys.exit(130)
