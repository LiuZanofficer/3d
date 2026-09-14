#!/usr/bin/env bash
set -euo pipefail
cd /root/lz
RUN_ID="a800_throughput_tune_economy_$(date +%Y%m%d_%H%M%S)"
OUT="/root/lz_outputs/${RUN_ID}"
mkdir -p "$OUT"
echo "RUN_ID=${RUN_ID}"
echo "OUT=${OUT}"
/root/autodl-tmp/conda-envs/llz/bin/python /root/lz/scripts/a800_throughput_tune_economy.py \
  --out-dir "$OUT" \
  --timeout-sec 900 \
  --base-runs /root/lz_outputs/a800_throughput_tune_20260221_154239 /root/lz_outputs/a800_throughput_tune_20260221_153916 \
  > "$OUT/run.log" 2>&1
RC=$?
echo "RC=$RC"
tail -n 120 "$OUT/run.log" || true
