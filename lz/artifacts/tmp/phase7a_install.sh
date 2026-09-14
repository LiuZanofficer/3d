set -e
source /etc/network_turbo >/dev/null 2>&1 || true
if [ -f /root/autodl-tmp/conda/etc/profile.d/conda.sh ]; then
  source /root/autodl-tmp/conda/etc/profile.d/conda.sh
  conda activate /root/autodl-tmp/conda-envs/llz
  PY=python
else
  PY=/root/autodl-tmp/conda-envs/llz/bin/python
fi
echo USING_PY=$PY
$PY -m pip install -U pip setuptools wheel
$PY -m pip install addict
$PY -m pip install spconv-cu121
$PY -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
$PY - <<'"'"'PY'"'"'
import torch
import spconv
import torch_scatter
import timm
import addict
print('"'"'torch'"'"', torch.__version__, '"'"'cuda'"'"', torch.version.cuda, '"'"'cuda_available'"'"', torch.cuda.is_available())
print('"'"'spconv ok'"'"')
print('"'"'torch_scatter ok'"'"')
print('"'"'timm ok'"'"')
print('"'"'addict ok'"'"')
PY