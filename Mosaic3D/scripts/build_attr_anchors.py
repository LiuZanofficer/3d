"""Build de-collinearized attribute text anchors (annotation-free).
Encodes LLM-authored discriminative descriptions with the SAME recap CLIP used at eval,
sanity-checks the re-encoded class-name anchor against reports/text_embeddings.npz, and
saves per-description embeddings + class-name anchor + attribute-mean anchor.
"""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
import open_clip

MODEL_ID = "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP"

def load_dict(path, var):
    ns = {}
    exec(open(path).read(), ns)
    return ns[var]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-embeddings", default="/root/Mosaic3D_work/error_analysis/reports/text_embeddings.npz")
    ap.add_argument("--desc1", default="/root/Mosaic3D_work/scripts/attr_descriptions_data.py")
    ap.add_argument("--desc2", default="/root/Mosaic3D_work/scripts/attr_descriptions_data2.py")
    ap.add_argument("--out", default="/root/Mosaic3D_work/error_analysis/reports/attr_anchors.npz")
    ap.add_argument("--out-json", default="/root/Mosaic3D_work/error_analysis/reports/attr_descriptions.json")
    args = ap.parse_args()

    DESC = load_dict(args.desc1, "DESC")
    DESC.update(load_dict(args.desc2, "DESC2"))

    te = np.load(args.text_embeddings, allow_pickle=True)
    class_names = [str(c) for c in np.asarray(te["class_names"])]
    fg = np.asarray(te["fg_class_idx"]).reshape(-1).astype(int)
    name_emb_ref = np.asarray(te["emb"], dtype=np.float64)
    name_emb_ref = name_emb_ref / np.maximum(np.linalg.norm(name_emb_ref, axis=1, keepdims=True), 1e-12)

    missing = [c for c in class_names if c not in DESC]
    if missing:
        raise SystemExit("missing descriptions for: %s" % missing)
    print("classes=%d, all have descriptions" % len(class_names))
    json.dump({c: DESC[c] for c in class_names}, open(args.out_json, "w"), indent=1, ensure_ascii=False)

    device = "cuda"
    model, _ = open_clip.create_model_from_pretrained(MODEL_ID, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(MODEL_ID)

    @torch.no_grad()
    def enc(texts):
        e = model.encode_text(tok(texts).to(device)).float()
        e = F.normalize(e, dim=-1)
        return e.cpu().numpy().astype(np.float64)

    # baseline class-name anchor with the eval prompt (use_prompt=True path)
    name_prompts = [("a %s in a scene" % c) if "other" not in c else "other" for c in class_names]
    name_emb = enc(name_prompts)
    cos_ref = float((name_emb * name_emb_ref).sum(1).mean())
    print("SANITY re-encoded name_emb vs text_embeddings.npz: mean cos = %.4f (expect ~1.0)" % cos_ref)

    emb_all, class_of, mean_list, k_per = [], [], [], []
    for ci, c in enumerate(class_names):
        ds = DESC[c]
        e = enc(ds)
        for j in range(len(ds)):
            emb_all.append(e[j]); class_of.append(ci)
        m = e.mean(0); m = m / max(np.linalg.norm(m), 1e-12)
        mean_list.append(m); k_per.append(len(ds))
    emb_all = np.stack(emb_all); class_of = np.asarray(class_of, dtype=int)
    mean_emb = np.stack(mean_list)

    np.savez(args.out,
             class_names=np.array(class_names, dtype=object),
             fg_class_idx=fg,
             name_emb=name_emb.astype(np.float32),
             mean_emb=mean_emb.astype(np.float32),
             emb_all=emb_all.astype(np.float32),
             class_of=class_of,
             k_per=np.asarray(k_per, dtype=int))
    print("saved %s: emb_all=%s mean_emb=%s name_emb=%s" % (args.out, emb_all.shape, mean_emb.shape, name_emb.shape))
    print("K per class: min=%d max=%d mean=%.1f" % (min(k_per), max(k_per), float(np.mean(k_per))))

if __name__ == "__main__":
    main()
