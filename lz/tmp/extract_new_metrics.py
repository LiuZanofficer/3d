import argparse
import glob
import json
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

TAGS = [
    "val/miou",
    "val/miou_present_gt",
    "val/miou_present_union",
    "val/miou_fg_mosaic",
]

def extract_last_and_best(path: Path):
    files = sorted(glob.glob(str(path / "logs" / "lz" / "version_*" / "events.out.tfevents.*")))
    if not files:
        raise FileNotFoundError(f"No tensorboard event file under {path}")
    event_path = files[-1]
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    scalar_tags = set(ea.Tags().get("scalars", []))
    out = {
        "event_file": event_path,
        "available_tags": sorted(scalar_tags),
        "metrics": {},
    }
    for tag in TAGS:
        if tag not in scalar_tags:
            out["metrics"][tag] = {"last": None, "best": None, "count": 0}
            continue
        vals = ea.Scalars(tag)
        if not vals:
            out["metrics"][tag] = {"last": None, "best": None, "count": 0}
            continue
        values = [x.value for x in vals]
        out["metrics"][tag] = {
            "last": float(values[-1]),
            "best": float(max(values)),
            "count": len(values),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = extract_last_and_best(Path(args.run_dir))
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
