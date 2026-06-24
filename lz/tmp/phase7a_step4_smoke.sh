set -e
source /etc/network_turbo >/dev/null 2>&1 || true
PY=/root/autodl-tmp/conda-envs/llz/bin/python
REPO=/root/lz
OUT=/root/lz_outputs/phase7A_smoke
mkdir -p "$OUT"
cd "$REPO"

SUCCESS=0
USED_BS=0
FALLBACK="none"
STEP_TIME_SEC=0

for BS in 2 1; do
  START_TS=$(date +%s)
  set +e
  CUDA_VISIBLE_DEVICES=0 "$PY" src/train.py \
    model=spunet34c_lz \
    data=sc \
    paths.root_dir=/root/lz \
    paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
    paths.output_dir="$OUT" \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    +trainer.accumulate_grad_batches=1 \
    trainer.gradient_clip_val=1.0 \
    +trainer.precision=16-mixed \
    data.batch_size=$BS \
    sampler.class_freq_path=/root/lz_outputs/class_freq_scannet200.json \
    optim.lr=0.001 \
    model.loss.weights.seg_loss=1.0 \
    model.loss.weights.instance_loss=0.0 \
    model.loss.weights.hpza_loss=0.0 \
    model.loss.seg_loss.lovasz_weight=0.0 \
    data.coord_norm_enable=false \
    data.color_drop_prob=0.0 \
    > "$OUT/train.log" 2>&1
  RC=$?
  set -e
  END_TS=$(date +%s)
  STEP_TIME_SEC=$((END_TS - START_TS))

  if [ $RC -eq 0 ]; then
    SUCCESS=1
    USED_BS=$BS
    break
  fi

  if grep -qi "out of memory" "$OUT/train.log"; then
    FALLBACK="oom_bs${BS}"
    continue
  fi

  echo "Smoke failed with non-OOM error at batch_size=$BS" > "$OUT/smoke_fail_reason.txt"
  tail -n 120 "$OUT/train.log" >> "$OUT/smoke_fail_reason.txt"
  break
done

GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if [ -z "$GPU_MEM_MB" ]; then GPU_MEM_MB=0; fi

cat > "$OUT/smoke_metrics.json" <<JSON
{
  "forward_backward_success": $SUCCESS,
  "used_batch_size": $USED_BS,
  "fallback": "$FALLBACK",
  "single_step_time_sec": $STEP_TIME_SEC,
  "gpu_peak_memory_mb": $GPU_MEM_MB
}
JSON

if [ $SUCCESS -ne 1 ]; then
  exit 1
fi
