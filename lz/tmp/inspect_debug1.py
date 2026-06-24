from pathlib import Path

split_file = Path('/root/lz/src/data/metadata/split_files/scannet_debug1.txt')
sc_dir = Path('/root/autodl-tmp/datasets/mosaic3d/data/scannet')
out = Path('/root/lz_outputs/diag_rootcause_20260211_221332/debug1_inspect.txt')

lines = split_file.read_text(encoding='utf-8', errors='ignore').splitlines()
raw = split_file.read_bytes()
scene = lines[0].strip() if lines else ''
scene_dir = sc_dir / scene
coord = scene_dir / 'coord.npy'
color = scene_dir / 'color.npy'
segment = scene_dir / 'segment200.npy'

content = []
content.append(f'split_file={split_file}')
content.append(f'raw_bytes={raw!r}')
content.append(f'lines={lines!r}')
content.append(f'scene={scene!r}')
content.append(f'scene_dir_exists={scene_dir.exists()}')
content.append(f'coord_exists={coord.exists()}')
content.append(f'color_exists={color.exists()}')
content.append(f'segment_exists={segment.exists()}')

out.write_text('\n'.join(content)+'\n', encoding='utf-8')
print('\n'.join(content))
