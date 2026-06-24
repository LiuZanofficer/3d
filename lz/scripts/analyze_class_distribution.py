from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_scene_list(dataset_name: str, split: str) -> List[str]:
    split_file = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "data"
        / "metadata"
        / "split_files"
        / f"{dataset_name}_{split}.txt"
    )
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("#")]


def analyze_class_frequency(
    data_dir: Path, split: str, segment_file: str
) -> Dict[int, int]:
    dataset_name = data_dir.stem
    scenes = load_scene_list(dataset_name, split)
    counter: Dict[int, int] = {}
    for scene in scenes:
        seg_path = data_dir / scene / segment_file
        if not seg_path.exists():
            continue
        seg = np.load(seg_path)
        values, counts = np.unique(seg, return_counts=True)
        for v, c in zip(values.tolist(), counts.tolist()):
            counter[int(v)] = counter.get(int(v), 0) + int(c)
    return counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--segment_file", default="segment200.npy", type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    freq = analyze_class_frequency(Path(args.data_dir), args.split, args.segment_file)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(freq, f, indent=2)
    print(f"Saved class frequency to {out_path}")


if __name__ == "__main__":
    main()
