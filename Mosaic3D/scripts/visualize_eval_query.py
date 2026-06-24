"""Interactive viser viewer to query a class on a Mosaic3D eval scene.

Unlike visualize_eval_scene.py (which takes --class-name on the CLI), this opens
a viser GUI with a dropdown to pick the query class live. For the chosen class C:
  - TP (GT=C and pred=C)        -> yellow
  - FN (GT=C, missed)           -> green
  - FP (pred=C, GT!=C)          -> colored by the point's TRUE GT class
  - everything else             -> dimmed original color

A side panel lists the FP points' true-class distribution with the color used,
so you can see "the points wrongly called C are actually wall / door / ...".

Dump files come from language_module._dump_eval_scene_prediction (set
MOSAIC3D_DUMP_EVAL=1 during eval). Read-only; needs numpy + viser (matplotlib
optional, only for nicer palette).
"""

import argparse
import colorsys
import time
from pathlib import Path

import numpy as np


TP_COLOR = np.array([255, 220, 0], dtype=np.uint8)   # yellow: correct
FN_COLOR = np.array([0, 180, 0], dtype=np.uint8)      # green: missed GT
IGNORE_COLOR = np.array([130, 130, 130], dtype=np.uint8)  # gray: unlabeled/ignore GT
BG_DIM = 0.28                                         # background dim factor


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_dump(pred_file: Path):
    data = np.load(pred_file, allow_pickle=True)
    return {key: _decode_scalar(data[key]) for key in data.files}


def class_index(class_names: np.ndarray, class_name: str) -> int:
    matches = np.where(class_names == class_name)[0]
    if len(matches) == 0:
        raise ValueError(f"Unknown class '{class_name}'.")
    return int(matches[0])


def name_of(class_names: np.ndarray, idx: int) -> str:
    if 0 <= idx < len(class_names):
        return str(class_names[idx])
    return f"<ignore:{idx}>"


def distinct_colors(n: int) -> np.ndarray:
    """n maximally-distinct RGB colors (uint8, [n,3]) via evenly-spaced hues.

    Colors are assigned to the classes actually present in the current FP set,
    so a handful of classes always get clearly different colors (no collisions).
    """
    colors = np.zeros((max(n, 1), 3), dtype=np.uint8)
    # Skip the yellow-green hue band [0.10, 0.50] so FP colors never look like
    # TP (yellow) or FN (green). Usable span: cyan -> blue -> magenta -> red -> orange.
    for i in range(n):
        h = (0.50 + (i / max(n, 1)) * 0.60) % 1.0
        s = 0.9 if i % 2 == 0 else 0.65
        v = 1.0 if i % 3 != 0 else 0.8
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def present_classes(dump):
    """Class names that appear in GT or per-point predictions, sorted by GT count desc."""
    class_names = dump["class_names"]
    gt_segment = dump["gt_segment"]
    pred_point_classes = dump["pred_point_classes"]
    rows = []
    for idx, name in enumerate(class_names):
        gt_n = int(np.count_nonzero(gt_segment == idx))
        pred_n = int(np.count_nonzero(pred_point_classes == idx))
        if gt_n or pred_n:
            rows.append((str(name), gt_n))
    rows.sort(key=lambda r: r[1], reverse=True)
    return [name for name, _ in rows]


def compute_overlay(dump, base_colors_dim, coords, class_name, exclude_ignore: bool):
    """Return (overlay uint8 [N,3], info dict) for the chosen class.

    FP points are colored by their TRUE GT class, using a palette assigned only
    to the classes actually present in the FP set (so colors never collide).
    Unlabeled/ignore GT points (label <0 or >=num_classes) are handled per
    `exclude_ignore`: if True they are dropped from FP (match official metric);
    if False they are shown gray and counted.
    """
    class_names = dump["class_names"]
    gt_segment = dump["gt_segment"]
    pred_point_classes = dump["pred_point_classes"]
    num_classes = len(class_names)

    idx = class_index(class_names, class_name)
    gt_mask = gt_segment == idx
    pred_mask = pred_point_classes == idx
    ignore_mask = (gt_segment < 0) | (gt_segment >= num_classes)

    tp = np.logical_and(gt_mask, pred_mask)
    fn = np.logical_and(gt_mask, ~pred_mask)
    fp_all = np.logical_and(~gt_mask, pred_mask)
    fp_ignore = np.logical_and(fp_all, ignore_mask)
    fp_valid = np.logical_and(fp_all, ~ignore_mask)

    overlay = base_colors_dim.copy()
    overlay[fn] = FN_COLOR
    overlay[tp] = TP_COLOR

    # Build a distinct color per present valid-FP true class (by count desc).
    fp_rows = []
    fp_valid_idx = np.where(fp_valid)[0]
    if fp_valid_idx.size:
        values, counts = np.unique(gt_segment[fp_valid_idx], return_counts=True)
        order = np.argsort(counts)[::-1]
        values, counts = values[order], counts[order]
        palette = distinct_colors(len(values))
        color_of = {int(v): palette[k] for k, v in enumerate(values)}
        # color the points
        cols = np.array([color_of[int(g)] for g in gt_segment[fp_valid_idx]], dtype=np.uint8)
        overlay[fp_valid_idx] = cols
        denom = fp_valid_idx.size
        gt_fp = gt_segment[fp_valid_idx]
        for k, v in enumerate(values):
            member = fp_valid_idx[gt_fp == int(v)]
            pts = coords[member]
            centroid = pts.mean(axis=0)
            # robust radius: 90th-percentile distance to centroid
            dists = np.linalg.norm(pts - centroid, axis=1)
            radius = float(np.percentile(dists, 90)) if dists.size else 0.0
            fp_rows.append({
                "name": name_of(class_names, int(v)),
                "count": int(counts[k]),
                "pct": 100.0 * int(counts[k]) / denom,
                "color": palette[k],
                "centroid": centroid.astype(np.float32),
                "radius": max(radius, 0.1),
            })

    n_ignore = int(np.count_nonzero(fp_ignore))
    if not exclude_ignore and n_ignore:
        overlay[fp_ignore] = IGNORE_COLOR

    n_tp = int(np.count_nonzero(tp))
    n_fn = int(np.count_nonzero(fn))
    n_fp_valid = int(fp_valid_idx.size)
    n_fp = n_fp_valid + (0 if exclude_ignore else n_ignore)
    union = n_tp + n_fn + n_fp_valid  # IoU always on valid (official-style)
    info = {
        "n_gt": int(np.count_nonzero(gt_mask)),
        "n_tp": n_tp,
        "n_fn": n_fn,
        "n_fp": n_fp,
        "n_fp_ignore": n_ignore,
        "exclude_ignore": exclude_ignore,
        "iou": n_tp / union if union else 0.0,
        "fp_rows": fp_rows,
    }
    return overlay, info


def make_markdown(class_name, info):
    ig_note = "excluded" if info["exclude_ignore"] else "shown gray"
    lines = [
        f"### Query: `{class_name}`",
        "",
        f"- TP (correct, yellow): **{info['n_tp']}**",
        f"- FN (missed, green): **{info['n_fn']}**",
        f"- FP (wrong): **{info['n_fp']}**",
        f"- point-IoU (valid only): **{info['iou']:.4f}**",
        f"- unlabeled/ignore predicted as this class: **{info['n_fp_ignore']}** ({ig_note})",
        "",
        "**FP points are actually (true class, color = 3D point color):**",
    ]
    if not info["fp_rows"]:
        lines.append("- (none)")
    else:
        for row in info["fp_rows"]:
            col = row["color"]
            hexc = "#{:02x}{:02x}{:02x}".format(int(col[0]), int(col[1]), int(col[2]))
            lines.append(f"- `{row['name']}` - {row['count']} ({row['pct']:.1f}%)  `{hexc}`")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive viser query viewer for a Mosaic3D eval scene dump."
    )
    parser.add_argument("--scene-dir", type=Path, required=True, help="Dir with coord.npy / color.npy.")
    parser.add_argument("--pred-file", type=Path, required=True, help="Path to <scene>.npz dump.")
    parser.add_argument("--point-size", type=float, default=0.02)
    parser.add_argument("--default-class", type=str, default=None, help="Initial query class.")
    args = parser.parse_args()

    import viser

    dump = load_dump(args.pred_file)
    coords = np.load(args.scene_dir / "coord.npy")
    raw_colors = np.load(args.scene_dir / "color.npy").astype(np.float32)
    base_colors_dim = (raw_colors * BG_DIM).astype(np.uint8)

    options = present_classes(dump)
    if not options:
        raise RuntimeError("No classes present in this scene dump.")
    default_class = args.default_class if args.default_class in options else options[0]

    server_cls = getattr(viser, "Server", None) or getattr(viser, "ViserServer", None)
    if server_cls is None:
        raise AttributeError("Installed viser has neither Server nor ViserServer.")
    server = server_cls()

    cloud_name = f"{dump['scene_name']}::query"
    point_size_state = {"value": args.point_size}
    jump_handles = []  # dynamic per-class jump buttons (+ their folder)

    def fly_to(client, center, radius):
        """Recenter & zoom a client's camera onto `center`, keeping view angle."""
        cam = client.camera
        cur_pos = np.asarray(cam.position, dtype=np.float64)
        look = np.asarray(cam.look_at, dtype=np.float64)
        direction = cur_pos - look
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 1.0, 1.0])
            norm = np.linalg.norm(direction)
        direction = direction / norm
        dist = max(radius * 3.0, 0.5)
        center = np.asarray(center, dtype=np.float64)
        cam.look_at = center
        cam.position = center + direction * dist

    def render(class_name):
        overlay, info = compute_overlay(
            dump, base_colors_dim, coords, class_name, exclude_ignore=ignore_cb.value
        )
        server.scene.add_point_cloud(
            name=cloud_name,
            points=coords,
            colors=np.clip(overlay.astype(np.float32) / 255.0, 0.0, 1.0),
            point_size=point_size_state["value"],
        )
        md_handle.content = make_markdown(class_name, info)

        # rebuild clickable "jump to FP class" buttons
        for h in jump_handles:
            h.remove()
        jump_handles.clear()
        if info["fp_rows"]:
            folder = server.gui.add_folder("Jump to FP class")
            jump_handles.append(folder)
            with folder:
                for row in info["fp_rows"]:
                    label = f"{row['name']}  {row['count']} ({row['pct']:.1f}%)"
                    btn = server.gui.add_button(label)
                    center = row["centroid"]
                    radius = row["radius"]

                    @btn.on_click
                    def _(event, center=center, radius=radius):
                        if event.client is not None:
                            fly_to(event.client, center, radius)

        print(
            f"[query] {class_name}: TP={info['n_tp']} FN={info['n_fn']} FP={info['n_fp']} "
            f"(ignore={info['n_fp_ignore']}) IoU={info['iou']:.4f}"
        )
        if info["fp_rows"]:
            top = ", ".join(f"{r['name']}:{r['count']}" for r in info["fp_rows"][:5])
            print(f"         FP true-class: {top}")

    dropdown = server.gui.add_dropdown("Query class", options=options, initial_value=default_class)
    size_slider = server.gui.add_slider(
        "Point size", min=0.005, max=0.05, step=0.001, initial_value=args.point_size
    )
    ignore_cb = server.gui.add_checkbox("Exclude unlabeled (ignore) points", initial_value=True)
    md_handle = server.gui.add_markdown("")

    @dropdown.on_update
    def _(_event):
        render(dropdown.value)

    @size_slider.on_update
    def _(_event):
        point_size_state["value"] = size_slider.value
        render(dropdown.value)

    @ignore_cb.on_update
    def _(_event):
        render(dropdown.value)

    render(default_class)
    print("Colors: yellow=TP, green=FN(missed), FP=colored by true class (see panel & console).")
    print("Click a class under 'Jump to FP class' to fly the camera to those points.")
    print("Open the printed viser URL in a browser. Ctrl+C to quit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
