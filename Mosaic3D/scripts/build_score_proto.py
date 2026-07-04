"""Build zero-training visual prototypes from TRAIN caption-region pooled clip_feat, then
score sibling naming on VAL GT-instance pooled features (offline oracle-upperbound protocol,
same as analyze_attr_anchors). Two naming versions on the SAME region pooling:
  (i)  proto_caption : region label = caption class-name word match  (DEPLOYABLE, annotation-free)
  (ii) proto_gt      : region label = segment200 majority over region (ORACLE diagnostic only)
Prototypes are used ONLY for in-tight-cluster disambiguation; global readout stays CLIP-text (name_emb).
"""
import argparse, glob, json, os, re
from collections import defaultdict, Counter
import numpy as np

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def fg_miou(true_c, pred_c, w, fg):
    C=200; TP=np.zeros(C); FP=np.zeros(C); FN=np.zeros(C)
    for t,p,n in zip(true_c,pred_c,w):
        if t==p: TP[t]+=n
        else: FP[p]+=n; FN[t]+=n
    ious=[]
    for c in fg:
        d=TP[c]+FP[c]+FN[c]
        if d>0: ious.append(TP[c]/d)
    return float(np.mean(ious)) if ious else 0.0

def load_val_feats(feat_dir):
    feats,tcls,npts=[],[],[]; cn=None; fg=None
    for f in sorted(glob.glob(os.path.join(feat_dir,"*.npz"))):
        d=np.load(f,allow_pickle=True)
        feats.append(np.asarray(d["pooled_feat"],dtype=np.float64))
        tcls.append(np.asarray(d["true_class"]).reshape(-1).astype(int))
        npts.append(np.asarray(d["n_points"]).reshape(-1).astype(np.int64))
        if cn is None:
            cn=[str(x) for x in np.asarray(d["class_names"])]; fg=np.asarray(d["fg_class_idx"]).reshape(-1).astype(int)
    F=np.concatenate(feats,0); F=norm(F)
    return F,np.concatenate(tcls,0),np.concatenate(npts,0),cn,fg

def name_by_caption(words, cand_names, toks):
    best=None; best_spec=0
    for m in cand_names:
        tw=toks[m]
        if tw and all(w in words for w in tw):
            if len(tw)>best_spec: best_spec=len(tw); best=m
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump", default="/root/proto_dump")
    ap.add_argument("--feat-dir", default="/workspace/Mosaic3D/logs/eval/runs/2026-06-27_06-58-17/eval_instance_features/scannet200")
    ap.add_argument("--clusters", default="/root/Mosaic3D_work/error_analysis/reports/task2/clusters_thr0.90.npz")
    ap.add_argument("--anchors", default="/root/Mosaic3D_work/error_analysis/reports/attr_anchors.npz")
    ap.add_argument("--min-np", type=int, default=25)
    ap.add_argument("--out", default="/root/Mosaic3D_work/error_analysis/reports/task2/zeroshot_proto.json")
    args=ap.parse_args()

    # clusters / tight members
    cl=np.load(args.clusters,allow_pickle=True)
    cid=np.asarray(cl["cluster_id"]).astype(int)
    names=[str(x) for x in np.asarray(cl["class_names"])]
    fg=np.asarray(cl["fg_class_idx"]).astype(int)
    fgset=set(fg.tolist())
    mem=defaultdict(list)
    for c,g in enumerate(cid):
        if c in fgset: mem[g].append(c)
    tight={g:ms for g,ms in mem.items() if 2<=len(ms)<=8}
    c2members={c:ms for ms in tight.values() for c in ms}
    tight_set=set(c2members)
    name2class={names[c]:c for c in range(len(names))}
    fg_names=[names[c] for c in fg]
    toks={m:[w for w in re.split(r"[^a-z]+",m.lower()) if w] for m in fg_names}

    # accumulate prototypes from train dump
    acc_cap=defaultdict(lambda: np.zeros(768,dtype=np.float64)); cnt_cap=Counter()
    acc_gt =defaultdict(lambda: np.zeros(768,dtype=np.float64)); cnt_gt =Counter()
    nfiles=0
    for f in sorted(glob.glob(os.path.join(args.dump,"*.npz"))):
        d=np.load(f,allow_pickle=True)
        pf=np.asarray(d["pooled_feat"],dtype=np.float64); pf=norm(pf)
        caps=list(d["caption"]); gt=np.asarray(d["region_gt"]).astype(int); npx=np.asarray(d["region_np"]).astype(int)
        nfiles+=1
        for b in range(len(caps)):
            if npx[b] < args.min_np: continue
            # (ii) oracle GT label
            g=int(gt[b])
            if g in tight_set:
                acc_gt[g]+=pf[b]; cnt_gt[g]+=1
            # (i) deployable caption word label
            words=set(re.split(r"[^a-z]+", str(caps[b]).lower()))
            wn=name_by_caption(words, fg_names, toks)
            if wn is not None:
                wc=name2class[wn]
                if wc in tight_set:
                    acc_cap[wc]+=pf[b]; cnt_cap[wc]+=1
    print("train dump files=%d" % nfiles)

    def build_A(acc,cnt):
        A=np.zeros((200,768),dtype=np.float64); have=np.zeros(200,bool)
        for c,v in acc.items():
            if cnt[c]>0: A[c]=v/cnt[c]; have[c]=True
        A=norm(A)
        return A,have
    A_cap,have_cap=build_A(acc_cap,cnt_cap)
    A_gt ,have_gt =build_A(acc_gt ,cnt_gt)
    print("tight classes=%d | with caption-proto=%d | with gt-proto=%d"%(len(tight_set),int(have_cap.sum()),int(have_gt.sum())))

    # val feats + name anchor
    F,tcls,npts,cn,fg2=load_val_feats(args.feat_dir)
    an=np.load(args.anchors,allow_pickle=True); name_emb=norm(np.asarray(an["name_emb"],dtype=np.float64))
    fgmask=np.full(200,-1e9); fgmask[fg]=0.0
    Stext=F@name_emb.T + fgmask[None,:]
    pred_full=Stext.argmax(1)
    in_tight=np.array([t in c2members for t in tcls])
    print("val instances=%d in_tight=%d"%(len(tcls),int(in_tight.sum())))

    def score(A,have,label):
        pred=pred_full.copy()
        tot=cor=named=cor_named=0; tpts=cpts=0
        # coverage: true class has proto
        cov_tot=cov_hit=0
        for i in np.where(in_tight)[0]:
            ms=c2members[tcls[i]]
            cand=[m for m in ms if have[m]]
            tot+=1; tpts+=npts[i]
            cov_tot+=1
            if have[tcls[i]]: cov_hit+=1
            if not cand:
                continue  # no prototype in this cluster -> cannot name (counts as wrong)
            sub=A[cand]@F[i]
            p=cand[int(sub.argmax())]
            pred[i]=p
            named+=1
            if p==tcls[i]: cor+=1; cpts+=npts[i]; cor_named+=1
        miou=fg_miou(tcls,pred,npts,fg)
        r=dict(sib_acc=round(cor/max(tot,1),4), sib_acc_pts=round(cpts/max(tpts,1),4),
               among_named=round(cor_named/max(named,1),4), named_frac=round(named/max(tot,1),4),
               miou_incluster=round(miou,4), proto_class_cov=round(cov_hit/max(cov_tot,1),4))
        print("%-14s sib_acc=%.4f pts=%.4f | among_named=%.4f named_frac=%.4f | miou_incl=%.4f | trueclass_hasproto=%.4f"
              %(label,r["sib_acc"],r["sib_acc_pts"],r["among_named"],r["named_frac"],r["miou_incluster"],r["proto_class_cov"]))
        return r

    # reference: name text anchor, same protocol
    def score_text():
        pred=pred_full.copy(); tot=cor=0; tpts=cpts=0
        for i in np.where(in_tight)[0]:
            ms=c2members[tcls[i]]; sub=Stext[i,ms]; p=ms[int(sub.argmax())]; pred[i]=p
            tot+=1; tpts+=npts[i]
            if p==tcls[i]: cor+=1; cpts+=npts[i]
        miou=fg_miou(tcls,pred,npts,fg)
        print("%-14s sib_acc=%.4f pts=%.4f | miou_incl=%.4f"%("name(text)",cor/tot,cpts/tpts,miou))
        return dict(sib_acc=round(cor/tot,4), sib_acc_pts=round(cpts/tpts,4), miou_incluster=round(miou,4))

    print("\n--- offline GT-instance protocol (same as analyze_attr_anchors) ---")
    r_text=score_text()
    r_cap=score(A_cap,have_cap,"proto_caption")
    r_gt =score(A_gt ,have_gt ,"proto_gt(oracle)")
    baseline_full_miou=round(fg_miou(tcls,pred_full,npts,fg),4)
    print("baseline (text full argmax, no incluster override) miou_incl=%.4f"%baseline_full_miou)

    out=dict(train_dump_files=nfiles, min_np=args.min_np,
             tight_classes=len(tight_set), caption_proto_classes=int(have_cap.sum()), gt_proto_classes=int(have_gt.sum()),
             val_in_tight=int(in_tight.sum()),
             ref_name_text=r_text, proto_caption=r_cap, proto_gt_oracle=r_gt,
             baseline_text_full_miou=baseline_full_miou,
             note="offline uses VAL GT-instance pooled feats (oracle upper bound). deployed online sibNaming ref=0.379.")
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    json.dump(out,open(args.out,"w"),indent=1)
    print("\nsaved",args.out)

if __name__=="__main__":
    main()
