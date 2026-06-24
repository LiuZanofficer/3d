from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_datasets(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=str, help="Path containing scannet/matterport3d/arkitscenes")
    parser.add_argument("--datasets", default="scannet,matterport3d,arkitscenes", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--anno_source", default="segment3d-gathered", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--clip_model", default="hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP", type=str)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    script = Path(__file__).resolve().parent / "precompute_text_embeddings.py"
    datasets = parse_datasets(args.datasets)

    for name in datasets:
        data_dir = data_root / name
        if not data_dir.exists():
            msg = f"Dataset not found: {data_dir}"
            if args.skip_missing:
                print(f"[skip] {msg}")
                continue
            raise FileNotFoundError(msg)

        cmd = [
            sys.executable,
            str(script),
            "--data_dir",
            str(data_dir),
            "--split",
            args.split,
            "--anno_source",
            args.anno_source,
            "--batch_size",
            str(args.batch_size),
            "--embed_dim",
            str(args.embed_dim),
            "--device",
            args.device,
            "--clip_model",
            args.clip_model,
        ]
        if args.force:
            cmd.append("--force")

        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

