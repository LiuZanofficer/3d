set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true

PY=/root/autodl-tmp/conda-envs/llz/bin/python
REPO=/root/lz
BASE=/root/lz_outputs/phase7D_full15_bn_abc
A_DIR=$BASE/A_baseline
B_DIR=$BASE/B_bn_batch_stats
C_DIR=$BASE/C_bn_recalib_eval
BASE_CKPT=/root/lz_outputs/phase7A_recheck_debug1_in6/checkpoints/last.ckpt
CLASS_FREQ=/root/lz_outputs/class_freq_scannet200.json

mkdir -p "$A_DIR" "$B_DIR" "$C_DIR"
cd "$REPO"

copy_hydra() {
  local out_dir="$1"
  local latest
  latest=$(find "$out_dir" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort | tail -n1 || true)
  if [ -n "${latest}" ] && [ -d "${latest}/.hydra" ]; then
    if [ -d "${out_dir}/.hydra" ]; then
      mv "${out_dir}/.hydra" "${out_dir}/.hydra_prev_$(date +%s)"
    fi
    cp -r "${latest}/.hydra" "${out_dir}/.hydra"
  fi
}

run_eval() {
  local out_dir="$1"
  local bn_flag="$2"
  local ckpt_path="$3"

  CUDA_VISIBLE_DEVICES=0 "$PY" src/eval.py \
    model=spunet34c_lz \
    model.net.backbone.in_channels=6 \
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

# A: baseline
run_eval "$A_DIR" false "$BASE_CKPT"

# B: BN batch stats
run_eval "$B_DIR" true "$BASE_CKPT"

# C: BN recalibration + baseline eval
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/phase7D_bn_recalibrate.py \
  model=spunet34c_lz \
  model.net.backbone.in_channels=6 \
  data=sc \
  +sampler=acbs \
  sampler.class_freq_path="$CLASS_FREQ" \
  ckpt_path="$BASE_CKPT" \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$C_DIR" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data.batch_size=1 \
  data.train_dataset.split=train \
  data.val_datasets.0.split=val \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  +model.eval_cfg.eval_bn_batch_stats=false \
  bn_recalib_iters=300 \
  recalib_ckpt_out="$C_DIR/recalib_ckpt.ckpt" \
  bn_recalib_stats_out="$C_DIR/bn_recalib_stats.json" \
  > "$C_DIR/recalib.log" 2>&1

run_eval "$C_DIR" false "$C_DIR/recalib_ckpt.ckpt"

"$PY" /root/lz/phase7D_finalize.py > "$BASE/finalize.log" 2>&1

"$PY" - <<'PY'
import json
from pathlib import Path
base = Path('/root/lz_outputs/phase7D_full15_bn_abc')
compare = json.loads((base / 'phase7D_bn_abc_compare.json').read_text(encoding='utf-8'))
print(compare['final_decision'])
print(json.dumps(compare['decision_basis'], indent=2))
PY
