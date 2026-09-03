from __future__ import annotations

import numpy as np


def upper_triangle(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("square matrix required")
    return D[np.triu_indices(D.shape[0], 1)]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("shape mismatch")
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average ranks, 0-based; small dependency-free scipy.stats.rankdata subset."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata_average(np.asarray(x).ravel()), rankdata_average(np.asarray(y).ravel()))


def distance_correlation(D1: np.ndarray, D2: np.ndarray, *, rank: bool = False) -> float:
    x = upper_triangle(D1)
    y = upper_triangle(D2)
    return spearman(x, y) if rank else pearson(x, y)


def label_permutation_test(
    predictor: np.ndarray,
    target: np.ndarray,
    *,
    controls: int = 512,
    seed: int = 0,
    rank: bool = False,
) -> dict:
    observed = distance_correlation(predictor, target, rank=rank)
    rng = np.random.default_rng(seed)
    null = np.empty(controls, dtype=float)
    n = predictor.shape[0]
    for i in range(controls):
        p = rng.permutation(n)
        null[i] = distance_correlation(predictor, target[np.ix_(p, p)], rank=rank)
    return {
        "observed": float(observed),
        "null_mean": float(np.mean(null)),
        "null_median": float(np.median(null)),
        "null_std": float(np.std(null, ddof=1)) if controls > 1 else 0.0,
        "p_upper": float((1 + np.sum(null >= observed)) / (controls + 1)),
    }


def within_target_spearman(distance: np.ndarray, damage: np.ndarray) -> np.ndarray:
    """For ordered donor interventions, correlate donor distance with damage per target.

    Diagonal entries are ignored. Returns one rho per target row.
    """
    distance = np.asarray(distance, dtype=float)
    damage = np.asarray(damage, dtype=float)
    if distance.shape != damage.shape or distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("equal square matrices required")
    n = distance.shape[0]
    out = np.empty(n, dtype=float)
    for t in range(n):
        keep = np.arange(n) != t
        out[t] = spearman(distance[t, keep], damage[t, keep])
    return out


def masked_within_target_spearman(distance: np.ndarray, damage: np.ndarray) -> np.ndarray:
    """Per-target Spearman using finite off-diagonal damage entries only."""
    distance = np.asarray(distance, dtype=float)
    damage = np.asarray(damage, dtype=float)
    if distance.shape != damage.shape or distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("equal square matrices required")
    n = distance.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        keep = np.isfinite(damage[t]) & (np.arange(n) != t)
        if np.sum(keep) >= 3:
            out[t] = spearman(distance[t, keep], damage[t, keep])
    return out
