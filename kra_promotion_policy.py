from __future__ import annotations

from typing import Final


MIN_ABSOLUTE_TOP1_LIFT_PP: Final = 5.0


def clears_absolute_top1_lift(top1_lift_pp: float) -> bool:
    return top1_lift_pp >= MIN_ABSOLUTE_TOP1_LIFT_PP
