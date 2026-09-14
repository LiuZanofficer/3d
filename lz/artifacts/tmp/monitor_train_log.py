import re
import sys
from pathlib import Path

log = Path(sys.argv[1])
if not log.exists():
    print('log_missing')
    sys.exit(0)
text = log.read_text(encoding='utf-8', errors='ignore')
text = text.replace('\r', '\n')
# keep last chunk for speed
chunk = text[-200000:]

epoch_matches = re.findall(r"Epoch\s+(\d+):\s+([0-9]{1,3})%", chunk)
miou_matches = re.findall(r"val/miou(?:_0)?=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", chunk)
loss_matches = re.findall(r"train/loss_step=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", chunk)

if epoch_matches:
    e, p = epoch_matches[-1]
    print(f'epoch={e} progress={p}%')
else:
    print('epoch=NA progress=NA')

print(f'val_miou_last={miou_matches[-1] if miou_matches else "NA"}')
print(f'loss_step_last={loss_matches[-1] if loss_matches else "NA"}')
print(f'log_size={log.stat().st_size}')
