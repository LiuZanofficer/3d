from __future__ import annotations

import hashlib
from typing import List, Optional

import torch
import torch.nn as nn


class HashTextEncoder(nn.Module):
    """Fallback text encoder using deterministic hashing."""

    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def encode(self, texts: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little", signed=False)
            g = torch.Generator(device="cpu").manual_seed(seed)
            v = torch.randn(self.embed_dim, generator=g)
            v = v / (v.norm() + 1e-6)
            vectors.append(v)
        out = torch.stack(vectors, dim=0)
        if device is not None:
            out = out.to(device)
        return out


class ClipTextEncoder(nn.Module):
    """CLIP/SigLIP text encoder wrapper (open_clip)."""

    def __init__(
        self,
        model_id: str,
        pretrained: Optional[str] = None,
        embed_dim: int = 768,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.pretrained = pretrained
        self.embed_dim = embed_dim
        self.device = device or "cpu"
        self._clip = None
        self._tokenizer = None
        self._fallback_encoder = None
        self._clip_disabled = False

    def _lazy_init(self):
        if self._clip is not None or self._clip_disabled:
            return
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("open_clip_torch is required for ClipTextEncoder") from exc

        if hasattr(open_clip, "create_model_from_pretrained"):
            created = open_clip.create_model_from_pretrained(self.model_id)
            if isinstance(created, tuple):
                if len(created) == 3:
                    model, _, _ = created
                elif len(created) == 2:
                    model, _ = created
                elif len(created) == 1:
                    model = created[0]
                else:
                    raise RuntimeError("Unexpected return values from create_model_from_pretrained")
            else:
                model = created
        else:
            model, _, _ = open_clip.create_model_and_transforms(self.model_id, pretrained=self.pretrained)
        tokenizer = open_clip.get_tokenizer(self.model_id)

        model = model.to(self.device)
        model.eval()
        self._clip = model
        self._tokenizer = tokenizer

    @torch.no_grad()
    def encode(self, texts: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        if self._fallback_encoder is not None:
            return self._fallback_encoder.encode(texts, device=device)
        try:
            self._lazy_init()
            if self._clip_disabled or self._clip is None or self._tokenizer is None:
                if self._fallback_encoder is None:
                    self._fallback_encoder = HashTextEncoder(embed_dim=self.embed_dim)
                return self._fallback_encoder.encode(texts, device=device)
            assert self._clip is not None
            assert self._tokenizer is not None
            tokens = self._tokenizer(texts)
            tokens = tokens.to(self.device)
            text_emb = self._clip.encode_text(tokens)
            text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-6)
            if device is not None:
                text_emb = text_emb.to(device)
            return text_emb
        except Exception as exc:
            if not self._clip_disabled:
                import logging
                logging.getLogger(__name__).warning(
                    "ClipTextEncoder failed to load (%r). Falling back to HashTextEncoder. "
                    "HPZA loss will train against RANDOM embeddings — set hpza_loss weight=0 "
                    "or run on a node with HF access.", exc,
                )
            self._clip_disabled = True
            self._clip = None
            self._tokenizer = None
            if self._fallback_encoder is None:
                self._fallback_encoder = HashTextEncoder(embed_dim=self.embed_dim)
            return self._fallback_encoder.encode(texts, device=device)


def build_text_encoder(cfg) -> nn.Module:
    use_clip = cfg.get("use_clip", True)
    embed_dim = int(cfg.get("embed_dim", 768))
    if use_clip:
        return ClipTextEncoder(
            model_id=str(cfg.get("model_id")),
            pretrained=cfg.get("pretrained"),
            embed_dim=embed_dim,
            device=cfg.get("device", "cpu"),
        )
    return HashTextEncoder(embed_dim=embed_dim)
