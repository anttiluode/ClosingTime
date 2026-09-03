import numpy as np

from closingtime.stats import rankdata_average, spearman, within_target_spearman


def test_rankdata_average_ties():
    r = rankdata_average(np.array([10, 5, 5, 20]))
    assert np.allclose(r, [2, 0.5, 0.5, 3])


def test_spearman_monotone():
    x = np.arange(10)
    assert abs(spearman(x, x**3) - 1.0) < 1e-12
    assert abs(spearman(x, -x) + 1.0) < 1e-12


def test_within_target_spearman():
    D = np.array([[0,1,2],[1,0,3],[2,3,0]], float)
    damage = D**2
    r = within_target_spearman(D, damage)
    assert np.allclose(r, 1.0)
