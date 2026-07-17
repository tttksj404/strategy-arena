from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing race-grouped screening contract


@dataclass(frozen=True, slots=True)
class Top1View:
    group_codes: np.ndarray
    group_count: int
    winner: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Top1View:
        group_codes, _ = pd.factorize(frame["rk"], sort=False)
        return cls(
            group_codes,
            int(group_codes.max()) + 1,
            frame["win"].to_numpy(dtype=bool),
        )

    def picks(self, score: np.ndarray) -> np.ndarray:
        maxima = np.full(self.group_count, -np.inf)
        np.maximum.at(maxima, self.group_codes, score)
        positions = np.arange(len(score))
        is_best = score == maxima[self.group_codes]
        picks = np.full(self.group_count, len(score), dtype=int)
        np.minimum.at(picks, self.group_codes[is_best], positions[is_best])
        return picks

    def accuracy(self, score: np.ndarray) -> float:
        return float(self.winner[self.picks(score)].mean())

    def switch_rate(self, candidate: np.ndarray, market: np.ndarray) -> float:
        return float(np.mean(self.picks(candidate) != self.picks(market)))
