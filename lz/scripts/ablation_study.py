from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> List[str]:
    items: List[str] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, key))
        else:
            if isinstance(v, bool):
                v = str(v).lower()
            items.append(f"{key}={v}")
    return items


def run_ablation(cfg_path: Path, devices: int, output_dir: Optional[str]) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = flatten_dict(cfg) if isinstance(cfg, dict) else []
    overrides.append(f"trainer.devices={devices}")
    if output_dir:
        overrides.append(f"paths.output_dir={output_dir}")
    cmd = ["python", "src/train.py"] + overrides
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True, type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    args = parser.parse_args()

    cfg_dir = Path(args.config_dir)
    cfg_files = sorted(cfg_dir.glob("*.yaml"))
    if not cfg_files:
        print(f"No config files found in {cfg_dir}")
        return
    for cfg in cfg_files:
        run_ablation(cfg, args.devices, args.output_dir)


if __name__ == "__main__":
    main()
