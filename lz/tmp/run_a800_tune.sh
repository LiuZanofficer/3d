#!/usr/bin/env bash
set -euo pipefail
for p in $(pgrep -f "[s]c_capacity_pipeline.py" || true); do kill -TERM "$p"; done
for p in $(pgrep -f "[s]rc/train.py .*sc_memory_check" || true); do kill -TERM "$p"; done
sleep 1
for p in $(pgrep -f "[s]c_capacity_pipeline.py" || true); do kill -KILL "$p"; done
for p in $(pgrep -f "[s]rc/train.py .*sc_memory_check" || true); do kill -KILL "$p"; done
cd /root/lz
RUN_ID="a800_throughput_tune_$(date +%Y%m%d_%H%M%S)"
OUT="/root/lz_outputs/${RUN_ID}"
mkdir -p "$OUT"
echo "RUN_ID=${RUN_ID}"
echo "OUT=${OUT}"
/root/autodl-tmp/conda-envs/llz/bin/python /root/lz/scripts/a800_throughput_tune.py --out-dir "$OUT" --timeout-sec 600 > "$OUT/run.log" 2>&1
RC=$?
echo "RC=$RC"
tail -n 120 "$OUT/run.log" || true
