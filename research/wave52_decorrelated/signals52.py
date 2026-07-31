#!/usr/bin/env python3
# Wave-52: build candidates that are actually different, then test whether they are.
#
# wave51 closed the selection-rule axis with an unusually specific diagnosis. Three rules -- argmax,
# equal-weight top-3, equal-weight top-5 -- produced an IDENTICAL 73.9% positive-window share and medians
# within 0.04pp, and minimax demonstrably selected differently (60.9% positive) yet landed within 0.17pp.
# Bet-hedging cannot help when the things being hedged across behave the same, and 1/N's advantage exists
# only when estimation error exceeds true dispersion in expected returns. True dispersion was what was
# missing, because every grid member ranked symbols by the SAME funding level and differed only in
# threshold, k, leg fraction and deployment cap.
#
# So this module builds signal families with different mechanics rather than different parameters, and the
# first thing measured is whether they are in fact decorrelated. That check comes BEFORE any ensemble test:
# if the families still move together there is nothing to hedge and the axis closes cheaply, whereas
# building the ensemble first would spend the effort and then discover the precondition was never met.
#
# Families, all computable from the same single-venue panel so no wave36-style venue mismatch enters:
#
#   LEVEL   -- rank by 7d funding APR. The incumbent, carried over unchanged as the control.
#   CHANGE  -- rank by the CHANGE in funding APR. Enters symbols whose carry is improving rather than
#              already high, so it holds different names at different times even when both are "carry".
#   BASIS   -- rank by the perp-versus-spot price spread. Economically related to funding but a distinct
#              observable: funding is a scheduled payment, basis is a live price gap, and they diverge
#              whenever the market prices future funding differently from current funding.
#   MAJORS  -- LEVEL restricted to the most liquid names; ALTS is its complement. A partition of the
#              universe rather than of the signal, which is the axis wave18's I2/I3 split used.
#
# Every family reuses fam_funding.carry_position for its hysteresis so entry/exit semantics stay identical
# and any measured difference comes from the ranking signal rather than from new state machinery.

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd

from research.wave38_breadth.dataio38 import CarryPanel, _ThresholdCandidate
from research.wave1.fam_funding import carry_position

MAJORS_QUANTILE: Final = 0.80  # top fifth by trailing volume counts as "majors"
CHANGE_WINDOW: Final = 7  # days over which the funding APR change is measured


def _hysteresis_from_score(score: np.ndarray, days, symbols, threshold: float) -> np.ndarray:
    """Apply L4's own carry_position to an arbitrary per-symbol score matrix.

    Reusing the validated state machine rather than writing a new one means a family's results differ from
    the incumbent because its SIGNAL differs, not because its entry/exit logic does.
    """
    frame = pd.DataFrame(score, index=days, columns=list(symbols))
    columns = [
        carry_position(frame[symbol], _ThresholdCandidate(threshold)).to_numpy() for symbol in symbols
    ]
    return np.nan_to_num(np.column_stack(columns), nan=0.0)


def family_level(panel: CarryPanel, threshold: float) -> CarryPanel:
    """Incumbent: rank and gate on the 7d funding APR level."""
    active = _hysteresis_from_score(panel.raw_apr, panel.days, panel.symbols, threshold)
    return dataclasses.replace(panel, active=active)


def family_change(panel: CarryPanel, threshold: float) -> CarryPanel:
    """Rank on the CHANGE in funding APR over CHANGE_WINDOW days.

    A symbol whose funding rose from 5% to 20% qualifies here while one sitting flat at 25% does not, so
    the two families hold different names even in the same regime. The change is computed on raw_apr, whose
    own shift is applied later by carry_position, so nothing here sees the future.
    """
    raw = panel.raw_apr
    change = np.full_like(raw, np.nan)
    change[CHANGE_WINDOW:] = raw[CHANGE_WINDOW:] - raw[:-CHANGE_WINDOW]
    active = _hysteresis_from_score(change, panel.days, panel.symbols, threshold)
    # Ranking must also use the change, not the level, or selection would silently revert to the incumbent.
    ranking = np.full_like(change, np.nan)
    ranking[1:] = change[:-1]
    return dataclasses.replace(panel, active=active, ranking_apr=ranking)


def basis_apr(panel: CarryPanel) -> np.ndarray:
    """Perp premium over spot, annualised, as a carry-comparable rate.

    A perp trading above spot is what funding is supposed to correct, so the premium is the market's live
    statement about carry, whereas funding is the scheduled settlement of it. Expressed per 8h period and
    annualised so the same threshold scale applies as for funding.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        premium = panel.perp_close / panel.spot_close - 1.0
    return premium * 3.0 * 365.0


def family_basis(panel: CarryPanel, threshold: float) -> CarryPanel:
    """Rank and gate on the annualised perp-spot premium instead of on funding."""
    score = basis_apr(panel)
    active = _hysteresis_from_score(score, panel.days, panel.symbols, threshold)
    ranking = np.full_like(score, np.nan)
    ranking[1:] = score[:-1]
    return dataclasses.replace(panel, active=active, ranking_apr=ranking)


def _volume_split(panel: CarryPanel) -> tuple[np.ndarray, np.ndarray]:
    """Per-day boolean masks for the liquid top fifth and its complement.

    panel.quote_volume is already the shifted trailing average dataio38 exports, so the split is knowable
    at decision time.
    """
    volume = panel.quote_volume
    majors = np.zeros_like(volume, dtype=bool)
    for day in range(volume.shape[0]):
        row = volume[day]
        finite = np.isfinite(row)
        if finite.sum() < 5:
            continue
        cutoff = np.quantile(row[finite], MAJORS_QUANTILE)
        majors[day] = finite & (row >= cutoff)
    alts = np.isfinite(volume) & ~majors
    return majors, alts


def family_majors(panel: CarryPanel, threshold: float) -> CarryPanel:
    """Funding level, restricted to the liquid top fifth."""
    majors, _ = _volume_split(panel)
    base = family_level(panel, threshold)
    return dataclasses.replace(base, tradable=base.tradable & majors)


def family_alts(panel: CarryPanel, threshold: float) -> CarryPanel:
    """Funding level, restricted to everything outside the liquid top fifth."""
    _, alts = _volume_split(panel)
    base = family_level(panel, threshold)
    return dataclasses.replace(base, tradable=base.tradable & alts)


FAMILIES: Final = {
    "level": family_level,
    "change": family_change,
    "basis": family_basis,
    "majors": family_majors,
    "alts": family_alts,
}


def main() -> int:
    from research.wave38_breadth.dataio38 import build_panel

    panel = build_panel()
    print("=== wave52 신호 계열별 활성 구조 (임계 0.15) ===")
    print(f"{'계열':>8} {'자격일 비중':>11} {'자격종목/일 중앙':>15} {'거래가능/일 중앙':>15}")
    built = {}
    for name, builder in FAMILIES.items():
        variant = builder(panel, 0.15)
        built[name] = variant
        qualifying = (variant.active > 0.0) & variant.tradable
        counts = qualifying.sum(axis=1)
        print(f"{name:>8} {float((counts >= 1).mean()):10.1%} {np.median(counts):14.0f} "
              f"{np.median(variant.tradable.sum(axis=1)):14.0f}")

    print("\n=== 신호 자체의 상관 (계열 간 자격 종목 집합이 얼마나 겹치나) ===")
    names = list(FAMILIES)
    print("        " + "".join(f"{n:>9}" for n in names))
    for a in names:
        row = f"{a:>8}"
        qa = (built[a].active > 0.0) & built[a].tradable
        for b in names:
            qb = (built[b].active > 0.0) & built[b].tradable
            both = (qa & qb).sum()
            union = (qa | qb).sum()
            row += f"{(both / union if union else 0.0):9.2f}"
        print(row)
    print("\n  (자카드 유사도: 1.00 = 같은 종목을 같은 날 보유, 0.00 = 겹치지 않음)")
    print("  => 이 표가 이미 높으면 앙상블은 검정할 가치가 없다. wave51이 그 경우였다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
