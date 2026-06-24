from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List


DATASET_SPECS = {
    "scannet": {
        "segment_candidates": ["segment200.npy", "segment20.npy", "segment.npy"],
        "instance_candidates": ["instance.npy"],
    },
    "matterport3d": {
        "segment_candidates": ["segment.npy"],
        "instance_candidates": [],
    },
    "arkitscenes": {
        "segment_candidates": [],
        "instance_candidates": [],
    },
}


def read_split(split_file: Path) -> List[str]:
    if not split_file.exists():
        return []
    with split_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def build_matterport_alias_map(data_dir: Path) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    if not data_dir.exists():
        return alias_map

    for scene_dir in data_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        actual = scene_dir.name
        alias_map.setdefault(actual, actual)

        match_region = re.match(r"^(?P<stem>.+)_region(?P<rid>\d+)$", actual)
        if match_region is not None:
            stem = match_region.group("stem")
            rid = int(match_region.group("rid"))
            alias_map.setdefault(f"{stem}_{rid}", actual)
            alias_map.setdefault(f"{stem}_{rid:02d}", actual)
            continue

        match_suffix = re.match(r"^(?P<stem>.+)_(?P<rid>\d+)$", actual)
        if match_suffix is None:
            continue
        stem = match_suffix.group("stem")
        rid_txt = match_suffix.group("rid")
        rid = int(rid_txt)
        alias_map.setdefault(f"{stem}_region{rid}", actual)
        alias_map.setdefault(f"{stem}_region{rid_txt}", actual)

    return alias_map


def resolve_scene_name(dataset: str, scene_name: str, alias_map: Dict[str, str]) -> str:
    if dataset != "matterport3d":
        return scene_name
    return alias_map.get(scene_name, scene_name)


def list_anno_sources(scene_dir: Path) -> List[str]:
    sources = []
    prefix = "captions."
    suffix = ".npz"
    for caption_file in scene_dir.glob("captions.*.npz"):
        name = caption_file.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        source = name[len(prefix) : -len(suffix)]
        point_indices_file = scene_dir / f"point_indices.{source}.npz"
        if point_indices_file.exists():
            sources.append(source)
    return sorted(set(sources))


def any_file_exists(scene_dir: Path, candidates: List[str]) -> bool:
    if not candidates:
        return False
    return any((scene_dir / file_name).exists() for file_name in candidates)


def inspect_dataset(data_root: Path, split_root: Path, dataset: str, split: str) -> Dict:
    spec = DATASET_SPECS[dataset]
    split_file = split_root / f"{dataset}_{split}.txt"
    split_scenes = read_split(split_file)

    data_dir = data_root / dataset
    alias_map = build_matterport_alias_map(data_dir) if dataset == "matterport3d" else {}

    scene_exists = 0
    coord_color_exists = 0
    segment_exists = 0
    instance_exists = 0
    anno_counter = Counter()

    for scene_name in split_scenes:
        resolved_scene_name = resolve_scene_name(dataset, scene_name, alias_map)
        scene_dir = data_dir / resolved_scene_name
        if not scene_dir.exists():
            continue
        scene_exists += 1

        if (scene_dir / "coord.npy").exists() and (scene_dir / "color.npy").exists():
            coord_color_exists += 1

        if any_file_exists(scene_dir, spec["segment_candidates"]):
            segment_exists += 1
        if any_file_exists(scene_dir, spec["instance_candidates"]):
            instance_exists += 1

        for source in list_anno_sources(scene_dir):
            anno_counter[source] += 1

    total = len(split_scenes)
    def ratio(count: int) -> float:
        return (count / total) if total > 0 else 0.0

    return {
        "dataset": dataset,
        "split": split,
        "split_file": str(split_file),
        "data_dir": str(data_dir),
        "split_total": total,
        "split_exists": scene_exists,
        "split_exists_ratio": ratio(scene_exists),
        "coord_color_exists": coord_color_exists,
        "coord_color_ratio": ratio(coord_color_exists),
        "segment_exists": segment_exists,
        "segment_ratio": ratio(segment_exists),
        "instance_exists": instance_exists,
        "instance_ratio": ratio(instance_exists),
        "anno_sources_top": anno_counter.most_common(10),
    }


def print_report(report: Dict) -> None:
    print(f"[{report['dataset']}] split={report['split']}")
    print(f"  split_total      : {report['split_total']}")
    print(f"  split_exists     : {report['split_exists']} ({report['split_exists_ratio']:.3f})")
    print(f"  coord+color      : {report['coord_color_exists']} ({report['coord_color_ratio']:.3f})")
    print(f"  segment labels   : {report['segment_exists']} ({report['segment_ratio']:.3f})")
    print(f"  instance labels  : {report['instance_exists']} ({report['instance_ratio']:.3f})")
    if report["anno_sources_top"]:
        top = ", ".join([f"{name}:{count}" for name, count in report["anno_sources_top"]])
        print(f"  anno sources top : {top}")
    else:
        print("  anno sources top : <none>")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight data contract checks for LZ training.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/Mosaic3D/data",
        help="Root directory containing scannet/matterport3d/arkitscenes folders.",
    )
    parser.add_argument(
        "--split_root",
        type=str,
        default="src/data/metadata/split_files",
        help="Directory containing split txt files.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split name to inspect.")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["scannet", "matterport3d", "arkitscenes"],
        choices=sorted(DATASET_SPECS.keys()),
    )
    parser.add_argument(
        "--min_split_exists_ratio",
        type=float,
        default=0.95,
        help="Fail if split_exists_ratio is lower than this value.",
    )
    parser.add_argument(
        "--min_coord_color_ratio",
        type=float,
        default=0.95,
        help="Fail if coord+color availability is lower than this value.",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="",
        help="Optional path to save json report.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    reports = []
    has_failure = False

    for dataset in args.datasets:
        report = inspect_dataset(data_root, split_root, dataset, args.split)
        reports.append(report)
        print_report(report)

        if report["split_total"] == 0:
            print(f"  [FAIL] split file empty or missing: {report['split_file']}")
            has_failure = True
            continue

        if report["split_exists_ratio"] < args.min_split_exists_ratio:
            print(
                f"  [FAIL] split_exists_ratio={report['split_exists_ratio']:.3f} < "
                f"{args.min_split_exists_ratio:.3f}"
            )
            has_failure = True
        if report["coord_color_ratio"] < args.min_coord_color_ratio:
            print(
                f"  [FAIL] coord_color_ratio={report['coord_color_ratio']:.3f} < "
                f"{args.min_coord_color_ratio:.3f}"
            )
            has_failure = True

    if args.report_json:
        output = {"reports": reports}
        output_path = Path(args.report_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Saved] {output_path}")

    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
