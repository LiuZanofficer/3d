from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.models.networks.vlm.text_encoder import build_text_encoder, HashTextEncoder
from src.utils.io import pack_list_of_np_arrays, unpack_list_of_np_arrays


TEMPLATES = [
    "a photo of {cap}",
    "a 3d scan of {cap}",
    "a scene with {cap}",
]


def load_scene_list(dataset_name: str, split: str) -> List[str]:
    split_file = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "data"
        / "metadata"
        / "split_files"
        / f"{dataset_name}_{split}.txt"
    )
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("#")]


def augment_captions(captions: List[str]) -> List[str]:
    augmented = []
    for cap in captions:
        augmented.append(cap)
        for t in TEMPLATES:
            augmented.append(t.format(cap=cap))
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--anno_source", default="segment3d-gathered", type=str)
    parser.add_argument("--output_suffix", default="aug", type=str)
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument(
        "--clip_model",
        default="hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP",
        type=str,
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset_name = data_dir.stem
    scenes = load_scene_list(dataset_name, args.split)

    if args.embed:
        try:
            encoder = build_text_encoder(
                {
                    "use_clip": True,
                    "model_id": args.clip_model,
                    "embed_dim": args.embed_dim,
                    "device": "cpu",
                }
            )
        except Exception:
            encoder = HashTextEncoder(embed_dim=args.embed_dim)
    else:
        encoder = None

    for scene in scenes:
        scene_dir = data_dir / scene
        cap_file = scene_dir / f"captions.{args.anno_source}.npz"
        idx_file = scene_dir / f"point_indices.{args.anno_source}.npz"
        if not cap_file.exists() or not idx_file.exists():
            continue

        captions = unpack_list_of_np_arrays(np.load(cap_file, allow_pickle=True))
        point_indices = unpack_list_of_np_arrays(np.load(idx_file, allow_pickle=True))

        new_captions = []
        new_indices = []
        new_embeddings = []
        for caps, idx in zip(captions, point_indices):
            caps_list = [str(c) for c in caps.tolist()] if hasattr(caps, "tolist") else list(caps)
            aug_caps = augment_captions(caps_list)
            new_captions.append(np.array(aug_caps, dtype=object))
            # repeat indices for each augmented caption
            rep_idx = [idx for _ in range(len(aug_caps))]
            new_indices.append(np.array(rep_idx, dtype=object))
            if encoder is not None:
                embeds = encoder.encode(aug_caps).cpu().numpy()
                new_embeddings.append(embeds)

        cap_out = scene_dir / f"captions.{args.output_suffix}.npz"
        idx_out = scene_dir / f"point_indices.{args.output_suffix}.npz"
        np.savez(cap_out, **pack_list_of_np_arrays(new_captions))
        np.savez(idx_out, **pack_list_of_np_arrays(new_indices))

        if encoder is not None and len(new_embeddings) > 0:
            emb_out = scene_dir / f"caption_embeds.{args.output_suffix}.npz"
            np.savez(emb_out, **pack_list_of_np_arrays(new_embeddings))

    print("Caption preprocessing complete.")


if __name__ == "__main__":
    main()
