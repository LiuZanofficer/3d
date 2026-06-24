from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.models.networks.vlm.text_encoder import HashTextEncoder, build_text_encoder
from src.utils.io import pack_list_of_np_arrays, unpack_list_of_np_arrays


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


def to_text_list(arr) -> List[str]:
    if hasattr(arr, "tolist"):
        return [str(x) for x in arr.tolist()]
    return [str(x) for x in list(arr)]


def encode_in_batches(encoder, texts: List[str], device: str, batch_size: int) -> np.ndarray:
    embeds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        with torch.no_grad():
            emb = encoder.encode(chunk, device=torch.device(device))
        embeds.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(embeds, axis=0) if embeds else np.zeros((0, 1), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--anno_source", default="segment3d-gathered", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--clip_model",
        default="hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP",
        type=str,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset_name = data_dir.stem
    scenes = load_scene_list(dataset_name, args.split)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    try:
        encoder = build_text_encoder(
            {
                "use_clip": True,
                "model_id": args.clip_model,
                "embed_dim": args.embed_dim,
                "device": device,
            }
        )
    except Exception:
        encoder = HashTextEncoder(embed_dim=args.embed_dim)
        device = "cpu"

    processed = 0
    skipped = 0
    for scene in scenes:
        scene_dir = data_dir / scene
        cap_file = scene_dir / f"captions.{args.anno_source}.npz"
        out_file = scene_dir / f"caption_embeds.{args.anno_source}.npz"
        if not cap_file.exists():
            skipped += 1
            continue
        if out_file.exists() and not args.force:
            skipped += 1
            continue

        captions = unpack_list_of_np_arrays(np.load(cap_file, allow_pickle=True))
        flat_texts: List[str] = []
        lengths: List[int] = []
        for caps in captions:
            text_list = to_text_list(caps)
            flat_texts.extend(text_list)
            lengths.append(len(text_list))

        if not flat_texts:
            skipped += 1
            continue

        embed_flat = encode_in_batches(
            encoder=encoder,
            texts=flat_texts,
            device=device,
            batch_size=max(int(args.batch_size), 1),
        )

        per_object_embeds = []
        ptr = 0
        for n in lengths:
            per_object_embeds.append(embed_flat[ptr : ptr + n])
            ptr += n

        np.savez(out_file, **pack_list_of_np_arrays(per_object_embeds))
        processed += 1

    print(f"Done. processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()

