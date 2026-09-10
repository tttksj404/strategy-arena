"""Dimension-aware NumPy types at the research boundary."""

import numpy as np

FloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
FloatVector = np.ndarray[tuple[int], np.dtype[np.float64]]
BoolVector = np.ndarray[tuple[int], np.dtype[np.bool_]]
