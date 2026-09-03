import numpy as np

from closingtime.geometry import (
    chordal_subspace_distance,
    projection_kernel,
    qk_support_distance_matrix,
    score_operators,
)


def test_projection_kernel_and_chordal_extremes():
    U = np.eye(6)[:, :2]
    V = U.copy()
    W = np.eye(6)[:, 2:4]
    assert abs(projection_kernel(U, V) - 2.0) < 1e-12
    assert chordal_subspace_distance(U, V) < 1e-12
    assert abs(chordal_subspace_distance(U, W) - 1.0) < 1e-12


def test_qk_support_geometry_invariant_to_joint_head_gauge():
    rng = np.random.default_rng(3)
    Wq = rng.normal(size=(12, 12))
    Wk = rng.normal(size=(12, 12))
    num_heads = 3
    D0 = qk_support_distance_matrix(Wq, Wk, num_heads=num_heads)
    M0 = score_operators(Wq, Wk, num_heads=num_heads)

    d = 4
    q2 = Wq.copy()
    k2 = Wk.copy()
    for h in range(num_heads):
        R, _ = np.linalg.qr(rng.normal(size=(d, d)))
        sl = slice(h*d, (h+1)*d)
        q2[sl] = R @ q2[sl]
        k2[sl] = R @ k2[sl]

    D1 = qk_support_distance_matrix(q2, k2, num_heads=num_heads)
    M1 = score_operators(q2, k2, num_heads=num_heads)
    assert np.allclose(D0, D1, atol=1e-10, rtol=1e-10)
    assert np.allclose(M0, M1, atol=1e-10, rtol=1e-10)
