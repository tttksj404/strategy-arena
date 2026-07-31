#!/usr/bin/env python3
# Wave-43: derive the entry threshold from each symbol's own cost instead of fitting one number for all.
#
# wave42 raised the flat entry bar and found it acts as a risk lever. But a flat bar is answering the
# wrong question. wave38's corrected cost model charges both legs, and the measured one-way rate spans
# 0.00040 to 0.00192 -- a 4.8x range across symbols. Amortising a round trip over L4's measured 4.05-day
# holding period, breakeven APR therefore spans 7.2% (most liquid) to 34.6% (least liquid). A single 15%
# bar is simultaneously too strict for cheap symbols and too permissive for expensive ones, and the
# consequence is measurable: of 54,301 qualifying observations, 23.0% sit BELOW their own breakeven and
# cannot repay the turnover they incur.
#
# So the threshold is not a free parameter. It is
#
#     threshold_i = safety x (2 x cost_rate_i) x 365 / hold_days
#
# which makes every symbol clear its own cost. `safety` and `hold_days` remain parameters -- the first is
# the margin demanded above breakeven, the second the amortisation horizon assumed at entry -- but the
# per-symbol SHAPE is derived from measurement rather than searched.
#
# The hysteresis rule is unchanged from L4: enter above the threshold, exit below half of it. That matters
# for comparability, so hysteresis_position() is verified below to reproduce fam_funding.carry_position
# exactly when handed a constant threshold. If it did not, any difference in results could be the new
# signal code rather than the new idea.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

EXIT_FRACTION: Final = 0.5  # L4's own convention: exit below threshold/2


def hysteresis_position(score: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Per-symbol carry state under a possibly time-varying, per-symbol threshold.

    `score` and `threshold` are both (n_days, n_symbols). Semantics are copied from
    fam_funding.carry_position so that a constant threshold reproduces L4 exactly:
      - turn ON  when score >  threshold
      - turn OFF when score <  threshold * EXIT_FRACTION
      - otherwise hold the previous state
      - NaN scores hold the previous state (never flip on missing data)
      - the whole series is shifted forward one day, because the decision uses the score known at the
        close of the previous bar

    The loop is over days rather than vectorised because the state is path dependent; symbols are
    vectorised, which is where the width is.
    """
    n_days, n_symbols = score.shape
    state = np.zeros(n_symbols, dtype=float)
    out = np.zeros((n_days, n_symbols), dtype=float)
    for day in range(n_days):
        row = score[day]
        bar = threshold[day]
        valid = np.isfinite(row) & np.isfinite(bar)
        turn_on = valid & (row > bar)
        turn_off = valid & (row < bar * EXIT_FRACTION)
        state = np.where(turn_on, 1.0, np.where(turn_off, 0.0, state))
        out[day] = state
    # shift(1): today's position was decided on yesterday's score
    shifted = np.zeros_like(out)
    shifted[1:] = out[:-1]
    return shifted


def breakeven_apr(cost_rate: np.ndarray, hold_days: float) -> np.ndarray:
    """APR at which funding over `hold_days` exactly repays one round trip.

    cost_rate is the ONE-WAY rate from costs_measured, which already covers both legs of the
    delta-neutral pair; a round trip is entering and exiting, hence the factor 2.
    """
    return 2.0 * cost_rate * 365.0 / hold_days


def derived_threshold(cost_rate: np.ndarray, hold_days: float, safety: float) -> np.ndarray:
    """Per-symbol, per-day entry bar: `safety` multiples of that symbol's breakeven APR.

    Where cost is unknown the bar is set to +inf rather than to a default, so an unknown-cost symbol is
    excluded instead of being admitted on an assumption. wave13's lesson applied to a threshold.
    """
    bar = safety * breakeven_apr(cost_rate, hold_days)
    return np.where(np.isfinite(bar), bar, np.inf)


def _verify_matches_carry_position() -> None:
    """hysteresis_position must equal fam_funding.carry_position for a constant threshold."""
    import pandas as pd

    from research.wave1.fam_funding import FundingCandidate, carry_position

    rng = np.random.default_rng(20260731)
    n_days = 500
    scores = rng.normal(0.15, 0.30, size=n_days)
    scores[rng.random(n_days) < 0.1] = np.nan  # exercise the NaN-holds-state path
    threshold = 0.15

    index = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    reference = carry_position(
        pd.Series(scores, index=index),
        FundingCandidate(candidate_id="verify", window_days=7, threshold_apr=threshold, top_k=1),
    ).to_numpy()

    mine = hysteresis_position(
        scores.reshape(-1, 1), np.full((n_days, 1), threshold)
    )[:, 0]

    difference = np.nanmax(np.abs(reference - mine))
    print(f"  carry_position 대조: 최대 불일치 {difference:.3e} ({'일치' if difference == 0 else '불일치'})")
    if difference != 0:
        raise AssertionError("hysteresis_position diverges from the validated carry_position")


def main() -> int:
    print("=== wave43 signal43 자체검증 ===")
    print("시변·종목별 임계값 히스테리시스가 검증된 carry_position 과 (상수 임계값에서) 동일한가")
    _verify_matches_carry_position()

    from research.wave38_breadth.dataio38 import build_panel

    panel = build_panel()
    print("\n=== 유도 임계값의 실제 분포 (보유 4.05일, safety 1.0) ===")
    bar = derived_threshold(panel.cost_rate, 4.05, 1.0)
    finite = bar[np.isfinite(bar)]
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3d} {np.percentile(finite, q):6.1%}")
    print(f"  일괄 15% 대비: 더 엄격한 관측 {float((finite > 0.15).mean()):.1%} · 더 느슨한 관측 {float((finite <= 0.15).mean()):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
