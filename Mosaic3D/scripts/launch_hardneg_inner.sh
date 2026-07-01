#!/usr/bin/env bash
# Container-side launcher for the elastic Method B (hard-negative caption loss) training.
# Invoked by scripts/gpu_elastic_watcher.py via `docker exec`.
#
# Required env (passed by watcher via docker exec -e):
#   CUDA_VISIBLE_DEVICES : comma list of host GPU indices (== container indices)
#   NDEV                 : number of devices (== count of CUDA_VISIBLE_DEVICES)
#   RESUME               : 1 -> resume from RUN_DIR/checkpoints/last.ckpt (epoch continues)
#                          0 -> cold start from pretrained init_ckpt (epoch from 0)
#   MASTER_PORT          : rendezvous port for this attempt (avoid collisions across relaunches)
# Optional env:
#   RUN_DIR              : fixed hydra run dir (default below) so last.ckpt path is stable
#   MAX_STEPS            : if set, caps trainer.max_steps (used for smoke tests)
#   CKPT_EVERY_STEPS     : step-level checkpoint interval (default 200)

set -u

# Multi-dataset (sc+ar+sc++) DataLoader with several workers opens many shared
# memory handles; default soft nofile=1024 causes "Too many open files". Raise to
# the container hard limit (524288).
ulimit -n 524288 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

PROJECT_DIR=/workspace/Mosaic3D
cd "$PROJECT_DIR"

export PROJECT_ROOT="$PROJECT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29555}"

RUN_DIR="${RUN_DIR:-$PROJECT_DIR/logs/train/runs/hardneg_elastic}"
CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-200}"
NDEV="${NDEV:?NDEV is required}"
RESUME="${RESUME:?RESUME is required}"
# After the sm_120 spconv rebuild + NCCL 2.27.7 upgrade, native multi-GPU NCCL
# works. Restore the full (non-weak) Method B config; these stay env-overridable.
NUM_WORKERS="${NUM_WORKERS:-4}"
BACKEND="${BACKEND:-nccl}"
ALL_GATHER="${ALL_GATHER:-true}"

# libnvrtc-builtins.so.12.8 (cumm-cu128 dep) lives in the nvidia pip lib dirs.
export LD_LIBRARY_PATH="$(ls -d /usr/local/lib/python3.11/dist-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

mkdir -p "$RUN_DIR"

# Base overrides shared by all launches.
ARGS=(
  experiment=train_spunet_multidata_ppt
  data=sc+ar+sc++
  model/loss=caption_siglip_hardneg
  data.train_dataset.repeat=1
  data.num_workers="$NUM_WORKERS"
  trainer.devices="$NDEV"
  +model.loss_cfg.caption_loss.all_gather="$ALL_GATHER"
  test=false
  logger=csv
  hydra.run.dir="$RUN_DIR"
  callbacks.model_checkpoint.every_n_train_steps="$CKPT_EVERY_STEPS"
  callbacks.model_checkpoint.save_last=true
)

# Multi-GPU DDP (native NCCL now that spconv has sm_120 kernels).
if [ "$NDEV" -ge 2 ]; then
  ARGS+=(
    +trainer.strategy._target_=lightning.pytorch.strategies.DDPStrategy
    +trainer.strategy.process_group_backend="$BACKEND"
    +trainer.strategy.find_unused_parameters=true
  )
fi

# Epoch continuity: resume keeps current_epoch/optimizer/scheduler from last.ckpt.
# Cold start loads raw pretrained weights only (epoch from 0).
if [ "$RESUME" = "1" ] && [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
  ARGS+=( ckpt_path="$RUN_DIR/checkpoints/last.ckpt" )
else
  ARGS+=( +init_ckpt_path=./qz/sc+ar+sc++.ckpt ckpt_path=null )
fi

if [ -n "${MAX_STEPS:-}" ]; then
  ARGS+=( +trainer.max_steps="$MAX_STEPS" )
fi

echo "[launch_inner] $(date '+%F %T') NDEV=$NDEV RESUME=$RESUME CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} MASTER_PORT=$MASTER_PORT"
echo "[launch_inner] args: ${ARGS[*]}"

# Record the python PID (this shell PID survives the exec) so the watcher can
# send SIGUSR1 for graceful stop and use kill -0 for liveness checks.
echo $$ > "$RUN_DIR/train.pid"

exec python src/train.py "${ARGS[@]}"
