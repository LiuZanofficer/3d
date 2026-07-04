"""C1: annotation-free per-cluster GATE for visual-prototype sibling disambiguation (hybrid).
Gate is estimated on TRAIN caption-word pseudo-labels via 2-fold (NO GT). Only clusters where
caption-proto beats text on pseudo-labels are enabled; elsewhere keep text (open-vocab preserved).
Offline (val GT-instance) reports: text-only, all-on(cap), af-gate(cap, DEPLOYABLE), oracle-gate(upper bound).
Also writes gated_proto.npz for the online deployable readout (proto = caption-word centroids only).
"""
import argparse, glob, os, re, json
from collections import defaultdict, Counter
import numpy as np

def norm(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def fg_miou(t,p,w,fg):
    C=200;TP=np.zeros(C);FP=np.zeros(C);FN=np.zeros(C)
    for a,b,n in zip(t,p,w):
        if a==b:TP[a]+=n
        else:FP[b]+=n;FN[a]+=n
    io=[TP[c]/(TP[c]+FP[c]+FN[c]) for c in fg if TP[c]+FP[c]+FN[c]>0]
    return float(np.mean(io)) if io else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump",default="/root/proto_dump")
    ap.add_argument("--feat-dir",default="/workspace/Mosaic3D/logs/eval/runs/2026-06-27_06-58-17/eval_instance_features/scannet200")
    ap.add_argument("--clusters",default="/root/Mosaic3D_work/error_analysis/reports/task2/clusters_thr0.90.npz")
    ap.add_argument("--anchors",default="/root/Mosaic3D_work/error_analysis/reports/attr_anchors.npz")
    ap.add_argument("--margin",type=float,default=0.02)
    ap.add_argument("--protopath",default="/root/Mosaic3D_work/error_analysis/prototypes/gated_proto.npz")
    ap.add_argument("--out",default="/root/Mosaic3D_work/error_analysis/reports/task2/c1_gate.json")
    args=ap.parse_args()

    cl=np.load(args.clusters,allow_pickle=True)
    cid=np.asarray(cl["cluster_id"]).astype(int); names=[str(x) for x in np.asarray(cl["class_names"])]
    fg=np.asarray(cl["fg_class_idx"]).astype(int); fgset=set(fg.tolist())
    mem=defaultdict(list)
    for c,g in enumerate(cid):
        if c in fgset: mem[g].append(c)
    tight={g:ms for g,ms in mem.items() if 2<=len(ms)<=8}
    c2m={c:ms for ms in tight.values() for c in ms}; tset=set(c2m)
    cls2gid={c:g for g,ms in tight.items() for c in ms}
    name2c={names[c]:c for c in range(len(names))}; fgnames=[names[c] for c in fg]
    toks={m:[w for w in re.split(r"[^a-z]+",m.lower()) if w] for m in fgnames}
    def capword(s):
        w=set(re.split(r"[^a-z]+",str(s).lower())); best=None;bs=0
        for m in fgnames:
            tw=toks[m]
            if tw and all(x in w for x in tw) and len(tw)>bs: bs=len(tw);best=m
        return name2c[best] if best is not None else -1

    # ---- load train dump regions (feat, caption-word class, gt class, fold by scene hash) ----
    reg_feat=[]; reg_wc=[]; reg_gt=[]; reg_fold=[]
    for f in sorted(glob.glob(os.path.join(args.dump,"*.npz"))):
        d=np.load(f,allow_pickle=True); pf=norm(np.asarray(d["pooled_feat"],float))
        caps=list(d["caption"]); gt=np.asarray(d["region_gt"]).astype(int)
        fold=hash(os.path.basename(f))&1
        for b in range(len(caps)):
            wc=capword(caps[b])
            reg_feat.append(pf[b]); reg_wc.append(wc); reg_gt.append(int(gt[b])); reg_fold.append(fold)
    reg_feat=np.stack(reg_feat); reg_wc=np.array(reg_wc); reg_gt=np.array(reg_gt); reg_fold=np.array(reg_fold)
    print("train regions=%d"%len(reg_wc))

    def build_proto(feat,label,mask):
        acc=defaultdict(lambda:np.zeros(768)); cnt=Counter()
        for i in np.where(mask)[0]:
            c=int(label[i])
            if c in tset: acc[c]+=feat[i]; cnt[c]+=1
        A=np.zeros((200,768)); H=np.zeros(200,bool)
        for c,v in acc.items():
            if cnt[c]>0: A[c]=v/cnt[c]; H[c]=True
        return norm(A),H,cnt

    an=np.load(args.anchors,allow_pickle=True); ne=norm(np.asarray(an["name_emb"],float))

    # ---- 2-fold gate estimation on caption-word pseudo-labels (annotation-free) ----
    per_cluster_delta=defaultdict(list)
    for train_fold in (0,1):
        trm=(reg_fold==train_fold)&(reg_wc>=0)
        tem=(reg_fold==1-train_fold)&(reg_wc>=0)
        A,H,_=build_proto(reg_feat,reg_wc,trm)
        for g,ms in tight.items():
            idx=[i for i in np.where(tem)[0] if reg_wc[i] in ms]
            if len(idx)<10: continue
            cand=[m for m in ms if H[m]]
            if len(cand)<2: continue
            ct=cp=0
            for i in idx:
                pt=ms[int(np.argmax([reg_feat[i]@ne[m] for m in ms]))]
                pp=cand[int(np.argmax([reg_feat[i]@A[m] for m in cand]))]
                if pt==reg_wc[i]: ct+=1
                if pp==reg_wc[i]: cp+=1
            per_cluster_delta[g].append((cp-ct)/len(idx))
    gate_on=set()
    for g,ds in per_cluster_delta.items():
        if np.mean(ds)>=args.margin: gate_on.add(g)
    print("gate-ON clusters=%d/%d :"%(len(gate_on),len(tight)))
    for g in gate_on:
        print("   ON  [%s]  delta=%.3f"%("|".join(names[c] for c in tight[g]), float(np.mean(per_cluster_delta[g]))))

    # ---- full prototypes on ALL train (deployable cap, and oracle gt) ----
    A_cap,H_cap,cnt_cap=build_proto(reg_feat,reg_wc,np.ones(len(reg_wc),bool))
    A_gt ,H_gt ,_      =build_proto(reg_feat,reg_gt,np.ones(len(reg_gt),bool))

    # ---- val offline scoring ----
    feats=[];tc=[];npt=[]
    for f in sorted(glob.glob(os.path.join(args.feat_dir,"*.npz"))):
        d=np.load(f,allow_pickle=True);feats.append(np.asarray(d["pooled_feat"],float));tc.append(np.asarray(d["true_class"]).reshape(-1).astype(int));npt.append(np.asarray(d["n_points"]).reshape(-1))
    F=norm(np.concatenate(feats,0));tc=np.concatenate(tc,0);npt=np.concatenate(npt,0)
    fgmask=np.full(200,-1e9);fgmask[fg]=0.0
    pred_text_full=(F@ne.T+fgmask[None,:]).argmax(1)
    in_tight=np.array([t in c2m for t in tc])

    def eval_variant(A,H,gate,label):
        pred=pred_text_full.copy(); tot=cor=0; tp=cp=0
        for i in np.where(in_tight)[0]:
            g=cls2gid[tc[i]]; ms=c2m[tc[i]]
            use_proto = (gate is None) or (g in gate)
            if use_proto:
                cand=[m for m in ms if H[m]]
                if cand: p=cand[int(np.argmax([F[i]@A[m] for m in cand]))]
                else: p=ms[int(np.argmax([F[i]@ne[m] for m in ms]))]
            else:
                p=ms[int(np.argmax([F[i]@ne[m] for m in ms]))]
            pred[i]=p; tot+=1; tp+=npt[i]
            if p==tc[i]: cor+=1; cp+=npt[i]
        miou=fg_miou(tc,pred,npt,fg)
        print("%-26s sib=%.4f pts=%.4f miou_incl=%.4f"%(label,cor/tot,cp/tp,miou))
        return dict(sib=round(cor/tot,4),sib_pts=round(cp/tp,4),miou_incl=round(miou,4))

    def eval_text():
        pred=pred_text_full.copy();tot=cor=0;tp=cp=0
        for i in np.where(in_tight)[0]:
            ms=c2m[tc[i]];p=ms[int(np.argmax([F[i]@ne[m] for m in ms]))];pred[i]=p;tot+=1;tp+=npt[i]
            if p==tc[i]:cor+=1;cp+=npt[i]
        print("%-26s sib=%.4f pts=%.4f miou_incl=%.4f"%("text-only",cor/tot,cp/tp,fg_miou(tc,pred,npt,fg)))
        return dict(sib=round(cor/tot,4),miou_incl=round(fg_miou(tc,pred,npt,fg),4))
    def eval_oracle():  # per-cluster pick better text/gt (upper bound)
        pred=pred_text_full.copy();tot=cor=0
        for g,ms in tight.items():
            idx=[i for i in np.where(in_tight)[0] if tc[i] in ms]
            cand=[m for m in ms if H_gt[m]]
            for i in idx:
                pt=ms[int(np.argmax([F[i]@ne[m] for m in ms]))]
                pg=cand[int(np.argmax([F[i]@A_gt[m] for m in cand]))] if cand else pt
                # pick per-cluster better on THIS metric -> approximate by trying both, keep better acc contribution
            # compute cluster acc both ways
            at=sum(1 for i in idx if ms[int(np.argmax([F[i]@ne[m] for m in ms]))]==tc[i])
            ag=sum(1 for i in idx if (cand and cand[int(np.argmax([F[i]@A_gt[m] for m in cand]))]==tc[i]))
            for i in idx:
                if ag>=at and cand: p=cand[int(np.argmax([F[i]@A_gt[m] for m in cand]))]
                else: p=ms[int(np.argmax([F[i]@ne[m] for m in ms]))]
                pred[i]=p; tot+=1
                if p==tc[i]: cor+=1
        print("%-26s sib=%.4f (upper bound)"%("oracle-gate(text/gt)",cor/tot))
        return dict(sib=round(cor/tot,4))

    print("\n--- offline (val GT-instance) ---")
    r_text=eval_text()
    r_allon=eval_variant(A_cap,H_cap,None,"all-on(cap)")
    r_afgate=eval_variant(A_cap,H_cap,gate_on,"af-gate(cap) DEPLOYABLE")
    r_afgate_gt=eval_variant(A_gt,H_gt,gate_on,"af-gate(gt) diag")
    r_oracle=eval_oracle()

    # save deployable proto npz: proto=caption centroids(200,768), count(200), gate map
    os.makedirs(os.path.dirname(args.protopath),exist_ok=True)
    gate_c2cluster={int(c):[int(x) for x in tight[cls2gid[c]]] for c in tset if cls2gid[c] in gate_on}
    np.savez(args.protopath, proto=A_cap.astype(np.float32), count=np.array([cnt_cap[c] for c in range(200)],dtype=np.int64),
             gate_c2cluster=np.array(json.dumps(gate_c2cluster),dtype=object),
             gate_clusters=np.array(sorted(gate_on)))
    print("\nsaved proto+gate ->",args.protopath)
    out=dict(margin=args.margin, gate_on=sorted(int(g) for g in gate_on),
             gate_on_names=["|".join(names[c] for c in tight[g]) for g in sorted(gate_on)],
             text=r_text, all_on_cap=r_allon, af_gate_cap=r_afgate, af_gate_gt=r_afgate_gt, oracle_gate=r_oracle)
    json.dump(out,open(args.out,"w"),indent=1); print("saved",args.out)

if __name__=="__main__":
    main()
