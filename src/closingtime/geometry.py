from __future__ import annotations

import numpy as np


def row_subspace_basis(W: np.ndarray, *, rank: int | None = None) -> np.ndarray:
    """Orthonormal basis (ambient x rank) for the row space of W."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be 2-D")
    _, s, Vt = np.linalg.svd(W, full_matrices=False)
    if rank is None:
        scale = float(s[0]) if len(s) else 0.0
        tol = max(W.shape) * np.finfo(float).eps * max(scale, 1.0)
        rank = int(np.sum(s > tol))
    rank = int(rank)
    if rank <= 0 or rank > min(W.shape):
        raise ValueError("invalid rank")
    return Vt[:rank].T


def projection_kernel(U: np.ndarray, V: np.ndarray) -> float:
    """Projection-kernel overlap ||U^T V||_F^2 for orthonormal bases."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or U.shape[0] != V.shape[0]:
        raise ValueError("bases must share an ambient dimension")
    return float(np.linalg.norm(U.T @ V, ord="fro") ** 2)


def chordal_subspace_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Normalized chordal distance in [0, 1] for equal-rank subspaces."""
    if U.shape[1] != V.shape[1]:
        raise ValueError("equal rank required")
    rank = U.shape[1]
    d2 = max(0.0, rank - projection_kernel(U, V))
    return float(np.sqrt(d2 / rank))


def split_heads(W: np.ndarray, *, num_heads: int) -> list[np.ndarray]:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] % num_heads:
        raise ValueError("projection rows must divide evenly into heads")
    d_head = W.shape[0] // num_heads
    return [W[h*d_head:(h+1)*d_head] for h in range(num_heads)]


def head_subspace_bases(W: np.ndarray, *, num_heads: int) -> list[np.ndarray]:
    blocks = split_heads(W, num_heads=num_heads)
    return [row_subspace_basis(block, rank=block.shape[0]) for block in blocks]


def distance_matrix_from_bases(bases: list[np.ndarray]) -> np.ndarray:
    n = len(bases)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = chordal_subspace_distance(bases[i], bases[j])
            D[i, j] = D[j, i] = d
    return D


def head_subspace_distance_matrix(W: np.ndarray, *, num_heads: int) -> np.ndarray:
    return distance_matrix_from_bases(head_subspace_bases(W, num_heads=num_heads))


def qk_support_distance_matrix(Wq: np.ndarray, Wk: np.ndarray, *, num_heads: int) -> np.ndarray:
    return 0.5 * (
        head_subspace_distance_matrix(Wq, num_heads=num_heads)
        + head_subspace_distance_matrix(Wk, num_heads=num_heads)
    )


def score_operators(Wq: np.ndarray, Wk: np.ndarray, *, num_heads: int) -> np.ndarray:
    """Gauge-invariant per-head score operators Q_h^T K_h / sqrt(d_head)."""
    q = split_heads(Wq, num_heads=num_heads)
    k = split_heads(Wk, num_heads=num_heads)
    d_head = q[0].shape[0]
    return np.stack([qh.T @ kh / np.sqrt(d_head) for qh, kh in zip(q, k)])


def cosine_distance_matrix(rows: np.ndarray, *, center: bool = False, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(rows, dtype=float)
    if X.ndim < 2:
        raise ValueError("rows must have a leading item axis")
    X = X.reshape(X.shape[0], -1)
    if center:
        X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, eps)
    sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    D = 0.5 * (1.0 - sim)
    np.fill_diagonal(D, 0.0)
    return D


def score_operator_distance_matrix(Wq: np.ndarray, Wk: np.ndarray, *, num_heads: int) -> np.ndarray:
    return cosine_distance_matrix(score_operators(Wq, Wk, num_heads=num_heads))


def score_spectrum_distance_matrix(Wq: np.ndarray, Wk: np.ndarray, *, num_heads: int) -> np.ndarray:
    ops = score_operators(Wq, Wk, num_heads=num_heads)
    spectra = np.stack([np.linalg.svd(m, compute_uv=False) for m in ops])
    spectra /= np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-12)
    n = spectra.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(spectra[i] - spectra[j]) / np.sqrt(2.0))
            D[i, j] = D[j, i] = d
    return D


def raw_qk_distance_matrix(Wq: np.ndarray, Wk: np.ndarray, *, num_heads: int) -> np.ndarray:
    q = split_heads(Wq, num_heads=num_heads)
    k = split_heads(Wk, num_heads=num_heads)
    features = np.stack([np.concatenate([qh.ravel(), kh.ravel()]) for qh, kh in zip(q, k)])
    return cosine_distance_matrix(features)
