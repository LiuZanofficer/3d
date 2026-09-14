#!/usr/bin/env bash
set -euo pipefail

PY=/root/autodl-tmp/conda-envs/llz/bin/python

cd /root/lz

echo "[phase7A] GPU check"
nvidia-smi -L

echo "[phase7A] installing dependencies"
"$PY" -m pip install -U pip setuptools wheel
"$PY" -m pip install addict
"$PY" -m pip install spconv-cu121
"$PY" -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

echo "[phase7A] verifying imports"
"$PY" - <<'PY'
import torch
import spconv
import torch_scatter
import timm
import addict
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("spconv ok")
print("torch_scatter ok")
print("timm ok")
print("addict ok")
PY

echo "[phase7A] dependency step done"
