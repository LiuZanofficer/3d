from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

import numpy as np

from src.data.metadata.scannet import CLASS_LABELS_200, TAIL_CLASSES_200


def load_scene_list(split_file: Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]


def build_tail_ids() -> Set[int]:
    name_to_id = {name: i for i, name in enumerate(CLASS_LABELS_200)}
    return {name_to_id[name] for name in TAIL_CLASSES_200 if name in name_to_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str, help=".../scannet")
    parser.add_argument("--split_file", required=True, type=str, help="scannet_train.txt")
    parser.add_argument("--segment_file", default="segment200.npy", type=str)
    parser.add_argument("--output_file", required=True, type=str)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    scene_names = load_scene_list(Path(args.split_file))
    tail_ids = build_tail_ids()

    selected = []
    for scene in scene_names:
        seg_path = data_dir / scene / args.segment_file
        if not seg_path.exists():
            continue
        labels = np.load(seg_path).astype(np.int64)
        uniq = set(np.unique(labels).tolist())
        if uniq.intersection(tail_ids):
            selected.append(scene)

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for scene in selected:
            f.write(f"{scene}\n")

    print(f"Done. selected={len(selected)} total={len(scene_names)} output={out_file}")


if __name__ == "__main__":
    main()

