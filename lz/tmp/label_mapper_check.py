import json
from pathlib import Path
import numpy as np

from src.data.scannet.dataset import ScanNet200Dataset

SPLIT_FILE = Path('/root/lz/src/data/metadata/split_files/scannet_train.txt')
SCANNET_DIR = Path('/root/autodl-tmp/datasets/mosaic3d/data/scannet')
OUT_JSON = Path('/root/lz_outputs/diag_rootcause_20260211_221332/label_mapper_check.json')
OUT_VERDICT = Path('/root/lz_outputs/diag_rootcause_20260211_221332/label_mapper_verdict.txt')

scene_names = []
with SPLIT_FILE.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            scene_names.append(line)

sampled = scene_names[:50]
scene_stats = []
for scene in sampled:
    seg_path = SCANNET_DIR / scene / 'segment200.npy'
    item = {
        'scene': scene,
        'segment_path': str(seg_path),
        'exists': seg_path.exists(),
    }
    if seg_path.exists():
        seg = np.load(seg_path)
        unique_vals = np.unique(seg)
        item.update(
            {
                'shape': list(seg.shape),
                'min': int(seg.min()),
                'max': int(seg.max()),
                'unique_count': int(unique_vals.size),
                'neg_count': int((seg < 0).sum()),
            }
        )
    scene_stats.append(item)

ds = ScanNet200Dataset(data_dir=str(SCANNET_DIR), split='train', transforms=None)
mapper = ds.valid_class_mapper

result = {
    'split_file': str(SPLIT_FILE),
    'scannet_dir': str(SCANNET_DIR),
    'split_total': len(scene_names),
    'sample_size': len(sampled),
    'scene_stats': scene_stats,
    'dataset_info': {
        'ignore_label': int(ds.ignore_label),
        'valid_class_mapper[-1]': int(mapper[-1]),
        'valid_class_mapper[0]': int(mapper[0]),
        'valid_class_mapper[199]': int(mapper[199]),
        'valid_class_mapper_len': int(len(mapper)),
    },
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

ignore_label = int(ds.ignore_label)
mapper_neg1 = int(mapper[-1])
if mapper_neg1 != ignore_label:
    verdict = '严重错误: valid_class_mapper[-1] != ignore_label'
else:
    verdict = '映射负标签逻辑正常: valid_class_mapper[-1] == ignore_label'

OUT_VERDICT.write_text(verdict + '\n', encoding='utf-8')
print('WROTE', OUT_JSON)
print('WROTE', OUT_VERDICT)
print('VERDICT', verdict)
