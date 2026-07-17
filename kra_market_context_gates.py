from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing grouped market contract


@dataclass(frozen=True, slots=True)
class ContextGate:
    name: str
    meet: str | None = None
    distance_minimum: float | None = None
    distance_maximum: float | None = None
    favorite_minimum: float | None = None
    favorite_maximum: float | None = None

    def apply(
        self, frame: pd.DataFrame, candidate: np.ndarray, market: np.ndarray
    ) -> np.ndarray:
        return np.where(self.mask(frame, market), candidate, market)

    def mask(self, frame: pd.DataFrame, market: np.ndarray) -> np.ndarray:
        distance = pd.to_numeric(frame["rcDist"], errors="coerce")
        favorite = pd.Series(market, index=frame.index).groupby(frame["rk"]).transform("max")
        mask = pd.Series(True, index=frame.index)
        if self.meet is not None:
            mask &= frame["meet"].astype(str) == self.meet
        if self.distance_minimum is not None:
            mask &= distance >= self.distance_minimum
        if self.distance_maximum is not None:
            mask &= distance < self.distance_maximum
        if self.favorite_minimum is not None:
            mask &= favorite >= self.favorite_minimum
        if self.favorite_maximum is not None:
            mask &= favorite < self.favorite_maximum
        return mask.to_numpy(dtype=bool)


DISTANCE_GATES: Final = (
    ContextGate("distance_sprint", distance_maximum=1400.0),
    ContextGate("distance_route", distance_minimum=1400.0, distance_maximum=1800.0),
    ContextGate("distance_staying", distance_minimum=1800.0),
)
UNCERTAINTY_GATES: Final = (
    ContextGate("uncertainty_high", favorite_maximum=0.25),
    ContextGate("uncertainty_medium", favorite_minimum=0.25, favorite_maximum=0.40),
    ContextGate("uncertainty_low", favorite_minimum=0.40),
)


def generate_context_gates() -> tuple[ContextGate, ...]:
    gates = [ContextGate("all")]
    meets = ("1", "2", "3")
    gates.extend(ContextGate(f"meet_{meet}", meet=meet) for meet in meets)
    gates.extend(DISTANCE_GATES)
    gates.extend(UNCERTAINTY_GATES)
    for meet in meets:
        gates.extend(
            ContextGate(
                f"meet_{meet}_{gate.name}",
                meet=meet,
                distance_minimum=gate.distance_minimum,
                distance_maximum=gate.distance_maximum,
            )
            for gate in DISTANCE_GATES
        )
        gates.extend(
            ContextGate(
                f"meet_{meet}_{gate.name}",
                meet=meet,
                favorite_minimum=gate.favorite_minimum,
                favorite_maximum=gate.favorite_maximum,
            )
            for gate in UNCERTAINTY_GATES
        )
    return tuple(gates)
