"""Annotation-free within-cluster anchor decorrelation for readout.

Given a set of near-collinear text anchors (a sibling cluster) and an instance
feature, project both into the cluster's principal subspace and whiten by the
anchor singular values. This amplifies the discriminative residual that the raw
cosine readout drowns out, without using any GT labels.
"""

from __future__ import annotations

import numpy as np


def whiten_pick(feat: np.ndarray, anchors: np.ndarray, eps_ratio: float = 1e-2) -> int:
    """Return index (0..k-1) of the whitened-best anchor for ``feat``.

    feat: (d,) instance feature (need not be unit-norm).
    anchors: (k, d) text anchors for the cluster (unit-norm recommended).
    """
    A = np.asarray(anchors, dtype=np.float64)
    f = np.asarray(feat, dtype=np.float64)
    k = A.shape[0]
    if k <= 1:
        return 0
    mu = A.mean(axis=0)
    Ac = A - mu
    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)  # Vt: (r, d)
    if s.size == 0 or s.max() <= 0:
        return int(np.argmax(A @ f))
    eps = eps_ratio * (s.max() + 1e-12)
    inv = 1.0 / (s + eps)
    fw = (Vt @ (f - mu)) * inv            # (r,)
    Aw = (Ac @ Vt.T) * inv                # (k, r)
    scores = Aw @ fw                      # (k,)
    return int(np.argmax(scores))
