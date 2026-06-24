set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true

PY=/root/autodl-tmp/conda-envs/llz/bin/python
REPO=/root/lz
BASE=/root/lz_outputs/phase8B_spunet_bn_recalib
TRAIN_DIR=$BASE/train_full15_ep15
EVAL_A_DIR=$BASE/eval_A_baseline
EVAL_B_DIR=$BASE/eval_B_recalib
CLASS_FREQ=/root/lz_outputs/class_freq_scannet200.json

mkdir -p "$BASE" "$TRAIN_DIR" "$EVAL_A_DIR" "$EVAL_B_DIR"
cd "$REPO"

copy_hydra() {
  local out_dir="$1"
  local latest
  latest=$(find "$out_dir" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort | tail -n1 || true)
  if [ -n "$latest" ] && [ -d "$latest/.hydra" ]; then
    if [ -d "$out_dir/.hydra" ]; then
      mv "$out_dir/.hydra" "$out_dir/.hydra_prev_$(date +%s)"
    fi
    cp -r "$latest/.hydra" "$out_dir/.hydra"
  fi
}

run_eval() {
  local out_dir="$1"
  local ckpt_path="$2"
  local bn_flag="$3"

  CUDA_VISIBLE_DEVICES=0 "$PY" src/eval.py \
    model=spunet34c_lz \
    model.net.backbone.in_channels=6 \
    model.net.backbone.norm_type=bn \
    data=sc \
    +sampler=acbs \
    sampler.class_freq_path="$CLASS_FREQ" \
    ckpt_path="$ckpt_path" \
    paths.root_dir=/root/lz \
    paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
    paths.output_dir="$out_dir" \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    data.batch_size=1 \
    data.train_dataset.split=train \
    data.val_datasets.0.split=val \
    model.loss.weights.seg_loss=1.0 \
    model.loss.weights.instance_loss=0.0 \
    model.loss.weights.hpza_loss=0.0 \
    model.loss.seg_loss.lovasz_weight=0.0 \
    +model.eval_cfg.eval_bn_batch_stats="$bn_flag" \
    > "$out_dir/train.log" 2>&1

  copy_hydra "$out_dir"
}

# Step 1: smoke
set +e
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/phase8B_smoke.py \
  model=spunet34c_lz \
  model.net.backbone.in_channels=6 \
  model.net.backbone.norm_type=bn \
  data=sc \
  sampler.class_freq_path="$CLASS_FREQ" \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$BASE/smoke_run" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data.batch_size=1 \
  data.train_dataset.split=train \
  data.val_datasets.0.split=val \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  +smoke_out="$BASE/smoke_check.json" \
  > "$BASE/smoke.log" 2>&1
SMOKE_RC=$?
set -e

if [ "$SMOKE_RC" -ne 0 ]; then
  "$PY" - <<'PY'
import json
from pathlib import Path
base = Path('/root/lz_outputs/phase8B_spunet_bn_recalib')
smoke = {}
smoke_path = base / 'smoke_check.json'
if smoke_path.exists():
    smoke = json.loads(smoke_path.read_text(encoding='utf-8'))
compare = {
    'smoke': smoke,
    'final_decision': 'BLOCKED_SMOKE_FAIL',
}
(base / 'phase8B_compare.json').write_text(json.dumps(compare, indent=2), encoding='utf-8')
(base / 'PHASE8B_SUMMARY.md').write_text(
    '# PHASE8B SUMMARY\n\n- Final decision: BLOCKED_SMOKE_FAIL\n',
    encoding='utf-8',
)
print('BLOCKED_SMOKE_FAIL')
PY
  exit 0
fi

# Step 2: train full15 ep15
CUDA_VISIBLE_DEVICES=0 "$PY" src/train.py \
  model=spunet34c_lz \
  model.net.backbone.in_channels=6 \
  model.net.backbone.norm_type=bn \
  data=sc \
  sampler.class_freq_path="$CLASS_FREQ" \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$TRAIN_DIR" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=15 \
  +trainer.precision=16-mixed \
  data.batch_size=1 \
  data.train_dataset.split=train \
  data.val_datasets.0.split=val \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  callbacks.model_checkpoint.save_top_k=0 \
  callbacks.model_checkpoint.save_last=true \
  > "$TRAIN_DIR/train.log" 2>&1

copy_hydra "$TRAIN_DIR"

TRAIN_CKPT="$TRAIN_DIR/checkpoints/last.ckpt"
if [ ! -f "$TRAIN_CKPT" ]; then
  echo "Missing checkpoint: $TRAIN_CKPT" >&2
  exit 2
fi

# Step 3A: eval baseline
run_eval "$EVAL_A_DIR" "$TRAIN_CKPT" false

# Step 3B: BN recalib + baseline eval
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/phase8B_bn_recalibrate.py \
  model=spunet34c_lz \
  model.net.backbone.in_channels=6 \
  model.net.backbone.norm_type=bn \
  data=sc \
  +sampler=acbs \
  sampler.class_freq_path="$CLASS_FREQ" \
  ckpt_path="$TRAIN_CKPT" \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$EVAL_B_DIR/recalib_run" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data.batch_size=1 \
  data.train_dataset.split=train \
  data.val_datasets.0.split=val \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  +bn_recalib_iters=300 \
  +recalib_ckpt_out="$EVAL_B_DIR/recalib_ckpt.ckpt" \
  +bn_recalib_stats_out="$EVAL_B_DIR/bn_recalib_stats.json" \
  > "$EVAL_B_DIR/recalib.log" 2>&1

run_eval "$EVAL_B_DIR" "$EVAL_B_DIR/recalib_ckpt.ckpt" false

# Step 4: finalize
"$PY" /root/lz/phase8B_finalize.py > "$BASE/finalize.log" 2>&1

"$PY" - <<'PY'
import json
from pathlib import Path
base = Path('/root/lz_outputs/phase8B_spunet_bn_recalib')
compare = json.loads((base / 'phase8B_compare.json').read_text(encoding='utf-8'))
print(compare['final_decision'])
PY
