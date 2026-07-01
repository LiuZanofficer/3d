"""Build caption-supervised visual prototypes without ScanNet200 GT labels.

For each training caption region:
  - pool the model's point `clip_feat` over caption `point_indices`;
  - parse the caption with longest class-name matching;
  - add the pooled region feature to each matched class prototype.

No `segment200.npy`, `gt_segment`, or `gt_instance` is read or used.
"""

from pathlib import Path
from typing import Any
from functools import partial

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_scatter import segment_csr

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.metadata.scannet import CLASS_LABELS_200  # noqa: E402
from src.data.utils.collate import point_collate_fn_with_masks  # noqa: E402
from src.utils.class_term_utils import build_term_matcher, match_class_terms  # noqa: E402


def move_to_device(obj: Any, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def flatten_captions(captions):
    return [str(item) for sublist in captions for item in sublist]


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    out_path = Path(cfg.get("proto", {}).get("out_path", "error_analysis/prototypes/caption_visual_prototypes.npz"))
    max_batches = cfg.get("proto", {}).get("max_batches", None)
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("ckpt_path is required to build prototypes from the reproduced model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    # Prototype extraction is an offline one-pass job. Use num_workers=0 to
    # avoid shared-memory / file-descriptor explosions from large caption tensors.
    loader = DataLoader(
        datamodule.data_train,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(point_collate_fn_with_masks, grid_size=0.02),
    )

    model = hydra.utils.instantiate(cfg.model)
    model.data_cfg = cfg.data
    model.configure_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    for key in list(state_dict.keys()):
        if "emb_target" in key:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    class_names = [str(c) for c in CLASS_LABELS_200]
    pattern, term_to_class = build_term_matcher(class_names)

    proto_sum = None
    proto_count = np.zeros(len(class_names), dtype=np.int64)
    total_regions = 0
    matched_regions = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, device)
            out = model(batch)
            feat = torch.nn.functional.normalize(out["clip_feat"].float(), dim=-1)
            if proto_sum is None:
                proto_sum = np.zeros((len(class_names), feat.shape[1]), dtype=np.float64)

            point_indices = batch["caption_data"]["point_indices"].long()
            offsets = batch["caption_data"]["caption_offsets"].to(device)
            region_feat = segment_csr(feat[point_indices], offsets, reduce="mean")
            region_feat = torch.nn.functional.normalize(region_feat, dim=-1).cpu().numpy()
            captions = flatten_captions(batch["caption_data"]["caption"])

            for cap, rf in zip(captions, region_feat):
                total_regions += 1
                hits = match_class_terms(cap, pattern, term_to_class)
                if not hits:
                    continue
                matched_regions += 1
                for cls_idx in hits:
                    proto_sum[int(cls_idx)] += rf
                    proto_count[int(cls_idx)] += 1

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"batch={batch_idx+1} total_regions={total_regions} "
                    f"matched={matched_regions} classes_with_proto={(proto_count>0).sum()}"
                )

    proto = proto_sum.copy()
    nonzero = proto_count > 0
    proto[nonzero] /= proto_count[nonzero, None]
    proto[nonzero] /= np.maximum(np.linalg.norm(proto[nonzero], axis=1, keepdims=True), 1e-12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        class_names=np.array(class_names, dtype=object),
        proto=proto.astype(np.float32),
        count=proto_count,
        total_regions=np.array(total_regions, dtype=np.int64),
        matched_regions=np.array(matched_regions, dtype=np.int64),
    )
    print("=" * 64)
    print(f"total_regions={total_regions} matched_regions={matched_regions}")
    print(f"classes_with_proto={(proto_count > 0).sum()} / {len(class_names)}")
    print(f"outputs -> {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
