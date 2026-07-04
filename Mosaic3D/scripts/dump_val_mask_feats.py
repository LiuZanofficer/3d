"""Dump per-Segment3D-mask pooled clip_feat on VAL scannet200 (baseline ckpt), plus mask->GT
alignment (majority GT class, best-IoU, purity) for the deployment-gap decomposition (L1/L2).
"""
import os, argparse, glob
import numpy as np
import torch
os.chdir("/root/Mosaic3D_work")
import rootutils
rootutils.setup_root("/root/Mosaic3D_work/scripts/dump_val_mask_feats.py", indicator=".project-root", pythonpath=True)
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import hydra
from torch.utils.data import DataLoader

def to_device(x, dev):
    if torch.is_tensor(x): return x.to(dev)
    if isinstance(x, dict): return {k: to_device(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(to_device(v, dev) for v in x)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="/root/val_mask_dump")
    ap.add_argument("--ckpt", default="/root/Mosaic3D_work/qz/sc+ar+sc++.ckpt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); dev="cuda"
    with initialize_config_dir(config_dir="/root/Mosaic3D_work/configs", version_base="1.3"):
        cfg = compose(config_name="eval.yaml", overrides=["experiment=train_spunet_multidata_ppt","data=sc+ar+sc++","trainer.devices=1"])
    model = hydra.utils.instantiate(cfg.model); model.configure_model()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    for k in list(sd.keys()):
        if "emb_target" in k: del sd[k]
    model.load_state_dict(sd, strict=False); model = model.to(dev).eval()

    dcfg = OmegaConf.to_container(cfg.data.val_datasets[1], resolve=True)  # ScanNet200 val + masks
    dcfg.pop("_partial_", None); dcfg.pop("_target_", None)
    dcfg["transforms"] = OmegaConf.create(dcfg["transforms"])
    from src.data.scannet.dataset import ScanNet200Dataset
    ds = ScanNet200Dataset(**dcfg)
    ds.scene_names = list(ds.scene_names)[args.shard::args.nshards]
    print(f"[data] shard {args.shard}/{args.nshards} scenes={len(ds.scene_names)}", flush=True)
    collate = hydra.utils.instantiate(cfg.data.collate_fn)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    done=0
    for i, batch in enumerate(dl):
        scene = ds.scene_names[i]; outp=os.path.join(args.out, scene+".npz")
        if os.path.exists(outp): done+=1; continue
        try:
            batch = to_device(batch, dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            cf = torch.nn.functional.normalize(out["clip_feat"].float(), dim=-1)
            seg = batch["segment"].detach().cpu().numpy().astype(np.int64)   # GT class per point (mapped)
            inst = batch["instance"].detach().cpu().numpy().astype(np.int64) # GT instance per point
            masks = batch["masks_binary"][0]
            if torch.is_tensor(masks): masks = masks.detach().cpu().numpy().astype(bool)
            else: masks = np.stack([m.detach().cpu().numpy().astype(bool) for m in masks])
            M = masks.shape[0]
            feats=np.zeros((M,cf.shape[1]),np.float16); mgt=np.full(M,-100,np.int64)
            msize=np.zeros(M,np.int64); mpur=np.zeros(M,np.float32); miou=np.zeros(M,np.float32)
            cfn = cf.cpu().numpy()
            for m in range(M):
                pm = masks[m]
                n = int(pm.sum()); msize[m]=n
                if n==0: continue
                feats[m]=(cfn[pm].mean(0)/max(np.linalg.norm(cfn[pm].mean(0)),1e-12)).astype(np.float16)
                s = seg[pm]; sv = s[s>=0]
                if len(sv): mgt[m]=int(np.bincount(sv).argmax())
                iv = inst[pm]; ivv = iv[iv>=0]
                if len(ivv):
                    gid = int(np.bincount(ivv).argmax())
                    mpur[m]= (iv==gid).sum()/n
                    inst_mask = (inst==gid)
                    inter=(pm & inst_mask).sum(); union=(pm | inst_mask).sum()
                    miou[m]= inter/max(union,1)
            np.savez(outp, mask_feat=feats, mask_gt=mgt, mask_size=msize, mask_purity=mpur, mask_best_iou=miou)
            done+=1
            if done%50==0: print(f"[{args.shard}] {done}/{len(ds.scene_names)} {scene} M={M}", flush=True)
        except Exception as e:
            print(f"[{args.shard}] ERR {scene}: {repr(e)}", flush=True)
    print(f"[{args.shard}] DONE {done}", flush=True)

if __name__=="__main__":
    main()
