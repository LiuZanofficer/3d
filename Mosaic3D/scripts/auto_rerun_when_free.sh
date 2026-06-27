#!/usr/bin/env bash
# Wait for a GPU with enough free memory, then run the dump-eval and the
# whole-val analyzer automatically. Operational glue (not part of the pipeline).
#
# Usage: bash scripts/auto_rerun_when_free.sh [free_mb_threshold]
# Runs on the HOST (needs nvidia-smi + sudo docker). Logs to logs/auto_rerun.log.
set -u

CONTAINER=mosaic3d-coeus-torch
THRESH="${1:-10240}"            # require >= this many MiB free on one card
POLL=60                          # seconds between probes
LOGDIR=/mnt/sdc/lz/3d/Mosaic3D/logs
LOG="$LOGDIR/auto_rerun.log"
EVAL_LOG="$LOGDIR/rerun_eval_dump.log"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "watcher started: need >= ${THRESH} MiB free, poll ${POLL}s"

while true; do
  # pick the GPU with the most free memory: "<free>,<idx>"
  best=$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits \
         | awk -F', ' '{print $1","$2}' | sort -t, -k1 -rn | head -1)
  free_mb=${best%%,*}
  gpu=${best##*,}
  if [ "${free_mb:-0}" -ge "$THRESH" ]; then
    log "GPU $gpu has ${free_mb} MiB free (>= ${THRESH}); launching dump-eval"
    break
  fi
  log "no card free enough (best: GPU $gpu = ${free_mb} MiB); sleeping ${POLL}s"
  sleep "$POLL"
done

# ---- run eval with extended dump on the chosen GPU ----
sudo docker exec "$CONTAINER" bash -lc \
  "cd /workspace/Mosaic3D && MOSAIC3D_DUMP_EVAL=1 CUDA_VISIBLE_DEVICES=$gpu \
   python src/eval.py experiment=train_spunet_multidata_ppt data=sc+ar+sc++ \
   ckpt_path=./qz/sc+ar+sc++.ckpt" > "$EVAL_LOG" 2>&1
rc=$?
log "eval finished rc=$rc (full log: $EVAL_LOG)"
if [ "$rc" -ne 0 ]; then
  log "eval failed (likely OOM if another job grabbed the card). Re-run this watcher to retry."
  exit "$rc"
fi

# ---- locate newest dump dir and run the analyzer ----
D=$(ls -td /mnt/sdc/lz/3d/Mosaic3D/logs/eval/runs/*/eval_scene_dumps/scannet200 2>/dev/null | head -1)
if [ -z "$D" ] || [ "$(ls -1 "$D"/*.npz 2>/dev/null | wc -l)" -eq 0 ]; then
  log "no dumps found after eval; aborting analyzer"
  exit 1
fi
# host path -> container path
DC=${D/\/mnt\/sdc\/lz\/3d\/Mosaic3D//workspace/Mosaic3D}
log "dumps: $(ls -1 "$D"/*.npz | wc -l) scenes at $D"

sudo docker exec "$CONTAINER" bash -lc \
  "cd /workspace/Mosaic3D && python scripts/analyze_eval_dataset.py \
   --dump-dir $DC --out-dir error_analysis/reports --topn 25" 2>&1 | tee -a "$LOG"
log "analyzer done. Reports in Mosaic3D/error_analysis/reports/ . Check the fg-mIoU self-check above."
