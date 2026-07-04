"""Decompose the deployment gap: L0 (GT-instance pooled) vs L1 (Segment3D-mask pooled) sibling
naming + L2 mask->GT alignment stats. Localizes H1(mask quality)/H2(pooling drift)/H3(point) .
"""
import glob, os, json
from collections import defaultdict
import numpy as np
def norm(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

CLU="/root/Mosaic3D_work/error_analysis/reports/task2/clusters_thr0.90.npz"
ANC="/root/Mosaic3D_work/error_analysis/reports/attr_anchors.npz"
GTDIR="/workspace/Mosaic3D/logs/eval/runs/2026-06-27_06-58-17/eval_instance_features/scannet200"
MDIR="/root/val_mask_dump"

cl=np.load(CLU,allow_pickle=True); cid=np.asarray(cl["cluster_id"]).astype(int)
names=[str(x) for x in np.asarray(cl["class_names"])]; fg=np.asarray(cl["fg_class_idx"]).astype(int); fgset=set(fg.tolist())
mem=defaultdict(list)
for c,g in enumerate(cid):
    if c in fgset: mem[g].append(c)
tight={g:ms for g,ms in mem.items() if 2<=len(ms)<=8}; c2m={c:ms for ms in tight.values() for c in ms}; tset=set(c2m)
ne=norm(np.asarray(np.load(ANC,allow_pickle=True)["name_emb"],float))
fgmask=np.full(200,-1e9); fgmask[fg]=0.0

def sib_from(feat,cls,size):
    # cluster-restricted sibNaming for items whose GT class in tight
    tot=cor=0; tp=cp=0; ftot=fcor=0
    for i in range(len(cls)):
        c=int(cls[i])
        if c not in tset: 
            continue
        ms=c2m[c]; sub=feat[i]@ne[ms].T; p=ms[int(np.argmax(sub))]
        tot+=1; tp+=size[i]
        if p==c: cor+=1; cp+=size[i]
        # full-200
        pf=int(np.argmax(feat[i]@ne.T+fgmask)); ftot+=1; fcor+= (pf==c)
    return cor/max(tot,1), cp/max(tp,1), tot, fcor/max(ftot,1)

# L0: GT-instance pooled
gf=[];gc=[];gs=[]
for f in sorted(glob.glob(os.path.join(GTDIR,"*.npz"))):
    d=np.load(f,allow_pickle=True); gf.append(np.asarray(d["pooled_feat"],float)); gc.append(np.asarray(d["true_class"]).reshape(-1).astype(int)); gs.append(np.asarray(d["n_points"]).reshape(-1))
GF=norm(np.concatenate(gf,0)); GC=np.concatenate(gc,0); GS=np.concatenate(gs,0)
l0,l0p,n0,l0full=sib_from(GF,GC,GS)
print("L0 GT-instance  sibNaming=%.4f pts=%.4f (n=%d) | full200 acc=%.4f"%(l0,l0p,n0,l0full))

# L1: Segment3D-mask pooled
mf=[];mc=[];ms_=[];mp=[];mi=[]
for f in sorted(glob.glob(os.path.join(MDIR,"*.npz"))):
    d=np.load(f,allow_pickle=True); mf.append(np.asarray(d["mask_feat"],float)); mc.append(np.asarray(d["mask_gt"]).astype(int))
    ms_.append(np.asarray(d["mask_size"])); mp.append(np.asarray(d["mask_purity"])); mi.append(np.asarray(d["mask_best_iou"]))
MF=norm(np.concatenate(mf,0)); MC=np.concatenate(mc,0); MS=np.concatenate(ms_,0); MP=np.concatenate(mp,0); MI=np.concatenate(mi,0)
l1,l1p,n1,l1full=sib_from(MF,MC,MS)
print("L1 mask-pooled  sibNaming=%.4f pts=%.4f (n=%d) | full200 acc=%.4f"%(l1,l1p,n1,l1full))
print("  L0->L1 drop: sibNaming %.4f -> %.4f (Δ=%.4f) | full200 %.4f -> %.4f (Δ=%.4f)"%(l0,l1,l1-l0,l0full,l1full,l1full-l0full))

# L2: mask alignment quality (all masks, and masks whose GT in tight)
def stats(mask):
    return dict(n=int(mask.sum()),
                iou_med=round(float(np.median(MI[mask])),3), iou_q1=round(float(np.percentile(MI[mask],25)),3),
                pur_med=round(float(np.median(MP[mask])),3),
                iou_gt05=round(float((MI[mask]>0.5).mean()),3), iou_gt025=round(float((MI[mask]>0.25).mean()),3),
                pur_gt08=round(float((MP[mask]>0.8).mean()),3))
allm=np.ones(len(MC),bool); tm=np.array([c in tset for c in MC])
print("\nL2 mask quality (ALL masks):", stats(allm))
print("L2 mask quality (GT in tight):", stats(tm))
print("mask GT valid frac=%.3f  in-fg frac=%.3f  in-tight frac=%.3f"%((MC>=0).mean(), np.isin(MC,list(fgset)).mean(), tm.mean()))

out=dict(L0_sib=round(l0,4),L0_full=round(l0full,4),L0_n=n0,
         L1_sib=round(l1,4),L1_full=round(l1full,4),L1_n=n1,
         drop_sib=round(l1-l0,4),drop_full=round(l1full-l0full,4),
         mask_all=stats(allm),mask_tight=stats(tm))
json.dump(out,open("/root/Mosaic3D_work/error_analysis/reports/task2/deployment_gap.json","w"),indent=1)
print("\nsaved deployment_gap.json")
