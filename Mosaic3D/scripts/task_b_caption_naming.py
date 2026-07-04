"""TASK B: name unsupervised sibling sub-clusters by caption class-name mode word.
Captions exist only for caption-bearing (train) scenes -> this is a naming-signal CEILING
diagnostic (annotation-free naming; GT used only for scoring). anno_sources = gsam2 + seem
(consistent with training default). Compares sibling naming acc vs TASK2 text-anchor 0.377/0.379.
"""
import argparse, glob, json, os, re
from collections import defaultdict, Counter
import numpy as np
def unpack_list_of_np_arrays(filename):
    with np.load(filename, allow_pickle=True) as data:
        packed = data["packed"]
        if "outer_lengths" in data:
            outer_lengths = data["outer_lengths"]; inner_lengths = data["inner_lengths"]
            inner_splits = np.split(packed, np.cumsum(inner_lengths)[:-1])
            def chunks(lst, cs): return [lst[sum(cs[:i]):sum(cs[:i+1])] for i in range(len(cs))]
            return chunks(inner_splits, list(outer_lengths))
        else:
            lengths = data["lengths"]
            return [np.array(arr) for arr in np.split(packed, np.cumsum(lengths)[:-1])]

DATA = "/datasets/mosaic3d/data/scannet"

def scene_caption_bag(scene, sources):
    seg = np.load(os.path.join(DATA, scene, "segment200.npy")).reshape(-1).astype(int)
    inst = np.load(os.path.join(DATA, scene, "instance.npy")).reshape(-1).astype(int)
    N = seg.shape[0]
    inst_caps = defaultdict(list)          # gt_instance_id -> list of caption strings
    inst_class = {}                        # gt_instance_id -> majority true class
    for src in sources:
        cf = os.path.join(DATA, scene, f"captions.{src}.npz")
        pf = os.path.join(DATA, scene, f"point_indices.{src}.npz")
        if not (os.path.exists(cf) and os.path.exists(pf)):
            continue
        caps = unpack_list_of_np_arrays(cf)     # list per object -> array of L_i strings
        pis = unpack_list_of_np_arrays(pf)      # list per object -> list of L_i point-idx arrays
        for capo, pio in zip(caps, pis):
            capo = list(np.asarray(capo).reshape(-1))
            pio = list(pio)
            for cap, pi in zip(capo, pio):       # per-caption point group
                pi = np.asarray(pi).reshape(-1).astype(int)
                pi = pi[(pi >= 0) & (pi < N)]
                if len(pi) == 0:
                    continue
                ii = inst[pi]
                iiv = ii[ii >= 0]
                if len(iiv) == 0:
                    continue
                gid = int(np.bincount(iiv).argmax())
                sel = seg[pi][ii == gid]
                sel = sel[(sel >= 0) & (sel < 200)]
                if len(sel) == 0:
                    continue
                cls = int(np.bincount(sel).argmax())
                inst_class[gid] = cls
                inst_caps[gid].append(str(cap))
    return inst_caps, inst_class

def name_by_caption(caps, member_names):
    # for each candidate member, count captions whose words contain all words of the member name;
    # prefer most specific (more matched words); mode over captions.
    votes = Counter()
    toks = {m: [w for w in re.split(r"[^a-z]+", m.lower()) if w] for m in member_names}
    for cap in caps:
        words = set(re.split(r"[^a-z]+", cap.lower()))
        best = None; best_spec = 0
        for m in member_names:
            tw = toks[m]
            if tw and all(w in words for w in tw):
                if len(tw) > best_spec:
                    best_spec = len(tw); best = m
        if best is not None:
            votes[best] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="/root/Mosaic3D_work/error_analysis/reports/task2/clusters_thr0.90.npz")
    ap.add_argument("--sources", default="gsam2,seem")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/root/Mosaic3D_work/error_analysis/reports/task2/caption_naming.json")
    args = ap.parse_args()
    sources = args.sources.split(",")

    cl = np.load(args.clusters, allow_pickle=True)
    cid = np.asarray(cl["cluster_id"]).astype(int)
    names = [str(x) for x in np.asarray(cl["class_names"])]
    fg = set(np.asarray(cl["fg_class_idx"]).astype(int).tolist())
    mem = defaultdict(list)
    for c, g in enumerate(cid):
        if c in fg: mem[g].append(c)
    tight = {g: ms for g, ms in mem.items() if 2 <= len(ms) <= 8}
    c2members = {c: ms for g, ms in tight.items() for c in ms}
    tight_set = set(c2members)

    scenes = sorted({os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(DATA, "*", "captions.gsam2.npz"))})
    if args.limit: scenes = scenes[:args.limit]
    print("caption scenes=%d, tight clusters=%d covering %d classes" % (len(scenes), len(tight), len(tight_set)))

    tot = 0; correct = 0; named = 0
    per_cluster = defaultdict(lambda: [0, 0, 0])  # gid -> [correct, named, total]
    per_class = defaultdict(lambda: [0, 0, 0])
    for si, scene in enumerate(scenes):
        inst_caps, inst_class = scene_caption_bag(scene, sources)
        for gid, cls in inst_class.items():
            if cls not in tight_set:
                continue
            ms = c2members[cls]; mnames = [names[m] for m in ms]
            pred_name = name_by_caption(inst_caps[gid], mnames)
            gnum = None
            for g, mss in tight.items():
                if cls in mss: gnum = g; break
            tot += 1; per_cluster[gnum][2] += 1; per_class[cls][2] += 1
            if pred_name is not None:
                named += 1; per_cluster[gnum][1] += 1; per_class[cls][1] += 1
                if pred_name == names[cls]:
                    correct += 1; per_cluster[gnum][0] += 1; per_class[cls][0] += 1
        if (si+1) % 200 == 0:
            print("  ..%d scenes, tot=%d acc=%.4f named_frac=%.4f" % (si+1, tot, correct/max(tot,1), named/max(tot,1)))

    acc = correct / max(tot, 1)
    acc_named = correct / max(named, 1)
    print("\n[TASK B] caption class-name mode naming within tight clusters:")
    print("  instances=%d  named(have any class word)=%d (%.3f)" % (tot, named, named/max(tot,1)))
    print("  sibling naming acc (all, unnamed=wrong) = %.4f" % acc)
    print("  sibling naming acc (among named)        = %.4f" % acc_named)
    print("  compare: TASK2 unsup_textname=0.377, baseline pred_majority=0.379")
    out = dict(n_instances=tot, n_named=named, acc_all=round(acc,4), acc_named=round(acc_named,4),
               named_frac=round(named/max(tot,1),4), sources=sources,
               per_cluster={names[[c for c in mem[g]][0]] + "/" + "|".join(names[c] for c in tight[g]):
                            dict(correct=v[0], named=v[1], total=v[2], acc=round(v[0]/max(v[2],1),4))
                            for g, v in per_cluster.items()},
               per_class={names[c]: dict(correct=v[0], named=v[1], total=v[2], acc=round(v[0]/max(v[2],1),4))
                          for c, v in sorted(per_class.items())})
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)

if __name__ == "__main__":
    main()
