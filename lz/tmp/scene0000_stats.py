import numpy as np
seg = np.load('/root/autodl-tmp/datasets/mosaic3d/data/scannet/scene0000_00/segment200.npy')
uniq = np.unique(seg)
valid = uniq[uniq >= 0]
print('unique_nonneg_count', int(len(valid)))
print('max_possible_full200_miou', float(len(valid) / 200.0))
print('unique_labels', valid.tolist())
