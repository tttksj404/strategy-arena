from __future__ import annotations

import numpy as np

import run_bitget_archive_pbo_search as search


def test_development_moments_exclude_the_locked_test_tail() -> None:
    returns = np.asarray([0.01, 0.02, -0.01, 0.00, 0.03, -0.02, 0.01, 0.00], dtype=float)

    sums, squares, counts = search.development_moments(returns, development_end=6, bins=2)

    assert counts == (3, 3)
    assert sums.shape == (2,)
    assert np.isclose(sums.sum(), returns[:6].sum())
    assert np.isclose(squares.sum(), (returns[:6] ** 2).sum())
