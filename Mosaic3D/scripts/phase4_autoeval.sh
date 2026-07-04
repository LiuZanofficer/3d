#!/usr/bin/env bash
# Phase-4 Method-B eval, GPU-safe. Fires ONLY when the mandated training has
# ended (status != RUNNING) or a GPU is genuinely free (>=20GB free, <20% util,
# 3 consecutive polls). Never runs mid-training to avoid OOM-ing the 8-card run.
set -u
WORK=/root/Mosaic3D_work
RUN=/root/runs/hardneg_full_8gpu_nccl277
CKPT=$RUN/checkpoints/last.ckpt
OUT=$WORK/error_analysis/reports/method_b_autoeval
DUMP_ROOT=/root/runs/phase4_eval
TE_BASE=/workspace/Mosaic3D/error_analysis/reports/text_embeddings.npz
BASE_DUMP=/workspace/Mosaic3D/logs/eval/runs/2026-06-27_06-58-17
mkdir -p "$OUT" "$DUMP_ROOT"
export PROJECT_ROOT=$WORK
cd "$WORK" || exit 1
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$OUT/autoeval.log"; }

free_gpu(){ # echoes index of a safe-to-use GPU or nothing
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
   | awk -F', ' '$2>=20000 && $3<20 {print $1; exit}'
}

log "watcher started; waiting for training end or free GPU"
hits=0; GPU=""
while true; do
  st=$(cat "$RUN/status.txt" 2>/dev/null || echo UNKNOWN)
  g=$(free_gpu)
  if [ "$st" != "RUNNING" ]; then GPU=$(free_gpu); GPU=${GPU:-0}; log "training status=$st -> proceed on GPU ${GPU}"; break; fi
  if [ -n "$g" ]; then hits=$((hits+1)); GPU=$g; else hits=0; fi
  if [ "$hits" -ge 3 ]; then log "free GPU $GPU stable -> proceed"; break; fi
  sleep 120
done

run_mode(){
  local mode=$1; local dd=$DUMP_ROOT/$mode
  rm -rf "$dd"; mkdir -p "$dd"
  log "eval mode=$mode on $CKPT (GPU $GPU) -> $dd"
  CUDA_VISIBLE_DEVICES=$GPU MOSAIC3D_DUMP_EVAL=1 MOSAIC3D_READOUT_MODE=$mode \
    MOSAIC3D_CLUSTER_THRESHOLD=0.90 MOSAIC3D_MAX_CLUSTER_SIZE=8 \
    python src/eval.py experiment=train_spunet_multidata_ppt data=sc+ar+sc++ \
      ckpt_path="$CKPT" trainer.devices=1 hydra.run.dir="$dd" \
      >"$OUT/eval_${mode}.log" 2>&1
  local map=$(grep -oE ">>> mAP: [0-9.]+" "$OUT/eval_${mode}.log" | tail -1)
  local sd=$(find "$dd" -type d -name scannet200 -path "*eval_scene_dumps*" | head -1)
  local miou="NA"
  if [ -n "$sd" ]; then
    miou=$(python scripts/analyze_oracle_real_labelspace.py --dump-dir "$sd" \
             --out-dir "$OUT/oracle_${mode}" --topk 25 2>/dev/null \
           | grep -oE "baseline fg-mIoU=[0-9.]+" | head -1)
  fi
  echo "mode=$mode | $miou | $map" >> "$OUT/summary.txt"
  log "mode=$mode done | $miou | $map"
}

: > "$OUT/summary.txt"
for m in baseline mask_text_vote anchor_decorrelate; do run_mode "$m"; done

# mechanism: cos(d_vis,d_txt) on the trained-B baseline dump
BSD=$(find $DUMP_ROOT/baseline -type d -name scannet200 -path "*eval_scene_dumps*" | head -1)
BIF=$(find $DUMP_ROOT/baseline -type d -name scannet200 -path "*eval_instance_features*" | head -1)
BVM=$(find $DUMP_ROOT/baseline -type f -name scannet200.npz -path "*eval_visual_means*" | head -1)
if [ -n "$BVM" ]; then
  python - "$BVM" "$OUT/text_embeddings_B.npz" <<PY
import numpy as np, sys
d=np.load(sys.argv[1],allow_pickle=True)
np.savez(sys.argv[2], emb=d["emb_target"].astype(np.float32),
         class_names=d["class_names"], fg_class_idx=d["fg_class_idx"])
print("saved B text emb")
PY
fi
if [ -n "$BSD" ] && [ -n "$BIF" ] && [ -f "$OUT/text_embeddings_B.npz" ]; then
  python scripts/analyze_readout_direction.py --dump-dir "$BSD" --feature-dir "$BIF" \
    --text-embeddings "$OUT/text_embeddings_B.npz" --out-dir "$OUT/mechanism_B" \
    >"$OUT/mechanism_B.log" 2>&1
  log "mechanism analysis done"
fi
log "ALL DONE"
