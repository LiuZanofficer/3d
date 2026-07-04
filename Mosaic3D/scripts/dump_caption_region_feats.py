"""Dump per-caption-region pooled clip_feat on TRAIN scannet (annotation-free grouping).
Grouping unit = caption point-groups (Segment3D-derived), NOT gt_instance/segment200 (redline).
For each region we also record: the caption text (for word-mode naming, deployable) and the
segment200 majority class (ORACLE diagnostic only). Baseline ckpt is used (deployable readout model).
Sharded over GPUs by scene index. Output: /root/proto_dump/<scene>.npz
"""
import os, sys, argparse, glob
import numpy as np
import torch

os.chdir("/root/Mosaic3D_work")
import rootutils
rootutils.setup_root("/root/Mosaic3D_work/scripts/dump_caption_region_feats.py", indicator=".project-root", pythonpath=True)

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import hydra
from torch.utils.data import DataLoader
from torch_scatter import segment_csr
from src.data.scannet.dataset import ScanNet200Dataset


def to_device(x, dev):
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, dev) for v in x)
    return x


def build_valid_mapper():
    labels = ScanNet200Dataset.CLASS_LABELS
    valid_idx = [i for i, c in enumerate(labels) if not c.startswith("other")]
    remapper = np.full(256, -100, dtype=np.int64)
    for i in valid_idx:
        remapper[i] = i
    return remapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="/root/proto_dump")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", default="/root/Mosaic3D_work/qz/sc+ar+sc++.ckpt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    with initialize_config_dir(config_dir="/root/Mosaic3D_work/configs", version_base="1.3"):
        cfg = compose(config_name="eval.yaml", overrides=[
            "experiment=train_spunet_multidata_ppt", "data=sc+ar+sc++",
            "trainer.devices=1"])

    # ---- model ----
    model = hydra.utils.instantiate(cfg.model)
    model.configure_model()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    for k in list(sd.keys()):
        if "emb_target" in k:
            del sd[k]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(dev).eval()

    # ---- dataset: train scannet, stripped (deterministic full cloud, keep caption) ----
    dcfg = OmegaConf.to_container(cfg.data.train_dataset.datasets[0], resolve=True)
    keep = {"FilterCaption", "Copy", "CenterShift", "NormalizeColor", "Add", "ToTensor", "Collect"}
    dcfg["transforms"] = [t for t in dcfg["transforms"] if t["type"] in keep]
    dcfg.pop("_partial_", None)
    dcfg.pop("_target_", None)
    dcfg["transforms"] = OmegaConf.create(dcfg["transforms"])
    from src.data.scannet.dataset import ScanNetDataset
    ds = ScanNetDataset(**dcfg)
    all_scenes = list(ds.scene_names)
    ds.scene_names = all_scenes[args.shard::args.nshards]
    if args.limit:
        ds.scene_names = ds.scene_names[:args.limit]
    print(f"[data] shard {args.shard}/{args.nshards} scenes={len(ds.scene_names)}", flush=True)

    collate = hydra.utils.instantiate(cfg.data.collate_fn)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    valid_mapper = build_valid_mapper()
    data_dir = dcfg["data_dir"]

    done = 0
    for i, batch in enumerate(dl):
        scene = ds.scene_names[i]
        outp = os.path.join(args.out, scene + ".npz")
        if os.path.exists(outp):
            done += 1
            continue
        try:
            batch = to_device(batch, dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            cf = out["clip_feat"].float()
            cf = torch.nn.functional.normalize(cf, dim=-1)
            cd = batch["caption_data"]
            pi = cd["point_indices"].long()
            co = cd["caption_offsets"].long()
            seg_feat = segment_csr(cf[pi], co, reduce="mean")  # [B,C]
            seg_feat = torch.nn.functional.normalize(seg_feat, dim=-1).cpu().numpy().astype(np.float16)
            caps = cd["caption"]
            if isinstance(caps, list) and len(caps) and isinstance(caps[0], list):
                caps = caps[0]
            caps = [str(c) for c in caps]
            # oracle GT per region from segment200 (diagnostic only)
            oi = batch.get("origin_idx", None)
            oi = oi.cpu().numpy().astype(np.int64) if torch.is_tensor(oi) else np.arange(cf.shape[0])
            seg_raw = np.load(os.path.join(data_dir, scene, "segment200.npy")).reshape(-1)
            seg_map = np.full(seg_raw.shape[0], -100, dtype=np.int64)
            v = (seg_raw >= 0) & (seg_raw < 256)
            seg_map[v] = valid_mapper[seg_raw[v]]
            pi_np = pi.cpu().numpy(); co_np = co.cpu().numpy()
            B = len(caps)
            region_gt = np.full(B, -100, dtype=np.int64)
            region_np = np.zeros(B, dtype=np.int64)
            for b in range(B):
                rows = pi_np[co_np[b]:co_np[b+1]]
                region_np[b] = len(rows)
                raw = oi[rows] if len(oi) == cf.shape[0] else rows
                raw = raw[(raw >= 0) & (raw < seg_map.shape[0])]
                g = seg_map[raw]; g = g[g >= 0]
                if len(g):
                    region_gt[b] = int(np.bincount(g).argmax())
            np.savez(outp, pooled_feat=seg_feat, caption=np.array(caps, dtype=object),
                     region_gt=region_gt, region_np=region_np)
            done += 1
            if done % 50 == 0:
                print(f"[{args.shard}] {done}/{len(ds.scene_names)} last={scene} B={B}", flush=True)
        except Exception as e:
            print(f"[{args.shard}] ERR {scene}: {repr(e)}", flush=True)
    print(f"[{args.shard}] DONE {done}", flush=True)


if __name__ == "__main__":
    main()
