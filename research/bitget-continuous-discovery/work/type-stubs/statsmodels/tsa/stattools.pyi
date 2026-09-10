"""Narrow statsmodels 0.15.0 contract verified from the installed coint source."""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

class CointResult(NamedTuple):
    coint_t: float
    pvalue: float
    critical_values: NDArray[np.float64]


def coint(
    y0: ArrayLike,
    y1: ArrayLike,
    trend: str = "c",
    method: str = "aeg",
    maxlag: int | None = None,
    autolag: str | None = "aic",
    return_results: bool | None = None,
) -> CointResult: ...
