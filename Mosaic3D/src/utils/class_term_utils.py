"""Utilities for class-name term parsing and sibling clusters.

These helpers are annotation-free: they use only class names, aliases, and text
embeddings. They must not depend on ScanNet200 GT labels.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np


def alias_terms(class_names: Iterable[str]) -> dict[str, int]:
    by_name = {str(name).lower(): i for i, name in enumerate(class_names)}
    aliases: dict[str, int] = {}
    for alias, target in [
        ("sofa", "couch"),
        ("couch", "couch"),
        ("trashcan", "trash can"),
        ("garbage can", "trash can"),
        ("tv", "tv"),
        ("television", "tv"),
        ("fridge", "refrigerator"),
        ("refridgerator", "refrigerator"),
        ("white board", "whiteboard"),
    ]:
        if target in by_name:
            aliases[alias] = by_name[target]
    return aliases


def build_term_matcher(class_names: Iterable[str]):
    term_to_class = {str(name).lower(): i for i, name in enumerate(class_names)}
    term_to_class.update(alias_terms(class_names))
    terms = sorted(term_to_class, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b")
    return pattern, term_to_class


def match_class_terms(text: str, pattern, term_to_class: dict[str, int]) -> set[int]:
    hits = set()
    for match in pattern.findall(str(text).lower()):
        hits.add(term_to_class[match])
    return hits


def build_text_clusters(emb: np.ndarray, fg_idx: np.ndarray, threshold: float) -> list[list[int]]:
    """Connected components over text cosine >= threshold, foreground classes only."""
    emb = emb.astype(np.float64)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    fg = [int(i) for i in fg_idx]
    parent = {i: i for i in fg}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cos = emb @ emb.T
    for i, a in enumerate(fg):
        for b in fg[i + 1 :]:
            if cos[a, b] >= threshold:
                union(a, b)

    groups: dict[int, list[int]] = {}
    for c in fg:
        groups.setdefault(find(c), []).append(c)
    return [sorted(v) for v in groups.values()]


def class_to_cluster(clusters: list[list[int]]) -> dict[int, list[int]]:
    out = {}
    for cluster in clusters:
        for c in cluster:
            out[int(c)] = list(cluster)
    return out
