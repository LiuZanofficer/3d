import json
from pathlib import Path
p = Path('/root/lz_outputs/diag_rootcause_20260211_221332/label_mapper_check.json')
d = json.loads(p.read_text(encoding='utf-8'))
print('split_total', d.get('split_total'))
print('sample_size', d.get('sample_size'))
info = d.get('dataset_info', {})
print('ignore_label', info.get('ignore_label'))
print('mapper_-1', info.get('valid_class_mapper[-1]'))
print('mapper_0', info.get('valid_class_mapper[0]'))
print('mapper_199', info.get('valid_class_mapper[199]'))
scene_stats = d.get('scene_stats', [])
print('sample_exists', sum(1 for x in scene_stats if x.get('exists')))
print('sample_neg_count_sum', sum(int(x.get('neg_count', 0) or 0) for x in scene_stats))
