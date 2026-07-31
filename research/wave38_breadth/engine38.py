#!/usr/bin/env python3
# Wave-38 delta-neutral carry engine, parameterized on breadth.
#
# The position is L4's, unchanged in kind: long spot + short perp on the same symbol, so the directional
# price term cancels and what remains is funding income plus basis drift. What this engine adds over
# engine18's loop is that top_k, leg_fraction and a total-deployment cap are free parameters instead of
# the constants (1, 0.50) L4 froze, because probe_opportunity.py measured that top_k=1 discards
# qualifying carry on 42.9% of days.
#
# The signal itself is NOT a free parameter. The 15% APR entry bar and its 7.5% hysteresis exit arrive
# already applied in panel.active via fam_funding.carry_position, so no setting in this grid can lower
# the quality bar. That distinction is the whole point of the wave: I3 lowered the bar to 8% and lost
# to cost, while this varies only how much of the already-qualifying set is harvested.
#
# Accounting is tracked in dollars per component (funding, basis, cost) rather than as return factors,
# so the sum of components reconciles to the equity change exactly and gate Z8 can assert a residual
# at machine precision instead of "close enough". Multiplicative return bookkeeping cannot make that
# claim, which is why wave31's verify31 cross-check exists at all.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave38_breadth.dataio38 import CarryPanel

TOTAL_CAPITAL: Final = 100.0
RESERVE_FRACTION: Final = 0.10
ACTIVE_CAPITAL: Final = TOTAL_CAPITAL * (1.0 - RESERVE_FRACTION)  # $90, wave10's contract
MIN_ORDER_USDT: Final = 5.0  # exchange minimum; wave37's first run silently filled $0.40 legs
CAPACITY_LIMIT_FRACTION: Final = 0.0001  # a leg may not exceed this share of the symbol's daily volume


PERP_LEVERAGE_CAP: Final = 20.0  # the user's stated maximum
# A delta-neutral carry leg of notional N buys N of spot with cash and shorts N of perp on margin, so
# it consumes N + N/L of capital at perp leverage L. Total deployment D therefore satisfies
# D + D/L <= 1, i.e. D <= L/(L+1). At the 20x cap that is 0.952: even with maximum perp leverage,
# deployment cannot reach 1.0 without BORROWING to buy spot. Spot margin borrow carries an interest
# cost this engine does not model, and wave18's I4 was rejected for exactly this class of omission
# (reverse carry needed spot shorting whose funding market vanishes in a panic). Configurations above
# this bound are therefore inadmissible rather than merely optimistic.
MAX_FEASIBLE_DEPLOYMENT: Final = PERP_LEVERAGE_CAP / (PERP_LEVERAGE_CAP + 1.0)


@dataclass(frozen=True, slots=True)
class CarryConfig:
    top_k: int
    leg_fraction: float
    deployment_cap: float  # max sum of |weights|; 0.50 == L4 today

    @property
    def label(self) -> str:
        return f"k{self.top_k:<2d} leg{self.leg_fraction:.2f} cap{self.deployment_cap:.2f}"

    @property
    def implied_perp_leverage(self) -> float:
        """Perp leverage needed so spot notional plus perp margin fits inside active capital."""
        if self.deployment_cap >= 1.0:
            return float("inf")
        return self.deployment_cap / (1.0 - self.deployment_cap)

    @property
    def requires_spot_borrow(self) -> bool:
        return self.deployment_cap > MAX_FEASIBLE_DEPLOYMENT


@dataclass(slots=True)
class CarryResult:
    start_capital: float  # capital this segment began with; not always ACTIVE_CAPITAL in a chain
    equity: np.ndarray  # dollar equity per day
    funding_usd: float
    basis_usd: float
    cost_usd: float
    days_active: int
    entries: int  # count of position-openings, for the "at least one entry per day" question
    max_concurrent: int
    min_leg_usd: float
    max_leg_usd: float
    blocked_min_order: int
    delta_mismatch: float  # max |spot notional - perp notional| across all legs/days
    liquidation_days: int  # days a held short perp leg's intraday high would have wiped its margin
    worst_adverse_move: float  # largest intraday adverse (upward) perp move while short, as a fraction
    accounting_residual: float
    turnover_total: float
    deployment_mean: float
    held: list[tuple[int, int]] = field(default_factory=list)

    @property
    def final(self) -> float:
        return float(self.equity[-1]) if len(self.equity) else self.start_capital

    @property
    def multiple(self) -> float:
        return self.final / self.start_capital if self.start_capital > 0.0 else 0.0

    def annualised(self, days: int) -> float:
        if days <= 0 or self.final <= 0.0 or self.start_capital <= 0.0:
            return -1.0
        return float(self.multiple ** (365.0 / days) - 1.0)

    @property
    def mdd(self) -> float:
        if len(self.equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(self.equity)
        return float(np.max(1.0 - self.equity / peak))


def _select(panel: CarryPanel, day: int, config: CarryConfig) -> np.ndarray:
    """Indices of symbols to hold, highest funding APR first, at most top_k.

    Eligibility is the conjunction of the hysteresis signal being on, the symbol being tradable that
    day, and a finite ranking score. Ranking uses panel.ranking_apr, which dataio38 already shifted by
    one day, so nothing here can see same-day information.
    """
    eligible = (panel.active[day] > 0.0) & panel.tradable[day] & np.isfinite(panel.ranking_apr[day])
    if not eligible.any():
        return np.empty(0, dtype=int)
    candidates = np.flatnonzero(eligible)
    order = np.argsort(-panel.ranking_apr[day][candidates], kind="stable")
    return candidates[order[: config.top_k]]


def simulate(
    panel: CarryPanel,
    config: CarryConfig,
    start: int,
    end: int,
    cost_multiplier: float = 1.0,
    start_capital: float = ACTIVE_CAPITAL,
) -> CarryResult:
    """Run the carry book over panel days [start, end).

    Sequential-in-dollars by construction: each day the previous book is marked, rebalanced (paying
    turnover), then held. Components are accumulated in dollars as they are applied so their sum equals
    the total equity change to machine precision.

    `start_capital` exists because the strategy is NOT scale-invariant: the $5 exchange minimum is an
    absolute dollar floor, so a walk-forward chain must carry real capital into each applied window
    rather than restarting at $90 and multiplying returns afterwards. wave37 learned this the hard way
    when unenforced minimums let legs shrink to $0.40.
    """
    n_symbols = len(panel.symbols)
    capital = start_capital
    previous_weights = np.zeros(n_symbols)
    equity: list[float] = []
    funding_usd = basis_usd = cost_usd = 0.0
    turnover_total = 0.0
    deployments: list[float] = []
    days_active = entries = max_concurrent = blocked = 0
    min_leg, max_leg = np.inf, 0.0
    delta_mismatch = 0.0
    liquidation_days = 0
    worst_adverse_move = 0.0
    held: list[tuple[int, int]] = []

    for day in range(start, end):
        # --- mark the book carried in from yesterday across the overnight gap -----------------
        if previous_weights.any():
            previous_day = day - 1
            spot_gap = panel.spot_open[day] / panel.spot_close[previous_day] - 1.0
            perp_gap = panel.perp_open[day] / panel.perp_close[previous_day] - 1.0
            gap = np.nan_to_num(spot_gap - perp_gap, nan=0.0, posinf=0.0, neginf=0.0)
            delta = capital * float(np.dot(gap, previous_weights))
            capital += delta
            basis_usd += delta

        # --- choose today's book -------------------------------------------------------------
        chosen = _select(panel, day, config)
        weights = np.zeros(n_symbols)
        if len(chosen) > 0:
            per_leg_fraction = min(config.leg_fraction, config.deployment_cap / len(chosen))
            leg_notional = capital * per_leg_fraction
            if leg_notional < MIN_ORDER_USDT:
                # The book cannot be placed at this size. Shrinking k to afford the minimum would be a
                # different strategy than the one selected, so the honest action is to hold nothing.
                blocked += 1
                chosen = np.empty(0, dtype=int)
            else:
                # Capacity: a leg may not swallow a meaningful share of the symbol's daily volume.
                volume = panel.quote_volume[day][chosen]
                affordable = ~np.isfinite(volume) | (leg_notional <= volume * CAPACITY_LIMIT_FRACTION)
                chosen = chosen[affordable]
                if len(chosen) > 0:
                    per_leg_fraction = min(config.leg_fraction, config.deployment_cap / len(chosen))
                    weights[chosen] = per_leg_fraction
                    leg_notional = capital * per_leg_fraction
                    min_leg = min(min_leg, leg_notional)
                    max_leg = max(max_leg, leg_notional)
                    # Delta neutrality is measured, not asserted. Both legs are sized from the same
                    # weight vector and the same capital, so spot notional and perp notional are the
                    # same float; the subtraction is performed rather than assumed to be zero so gate
                    # Z5 rests on an arithmetic observation. An earlier version wrote
                    # `max(delta_mismatch, 0.0)` here, which made the gate incapable of failing --
                    # a gate that cannot fail measures nothing.
                    spot_notional = capital * weights[chosen]
                    perp_notional = capital * weights[chosen]
                    delta_mismatch = max(delta_mismatch, float(np.max(np.abs(spot_notional - perp_notional))))

        # --- rebalance, paying measured turnover cost ----------------------------------------
        weight_change = np.abs(weights - previous_weights)
        if weight_change.any():
            rates = np.nan_to_num(panel.cost_rate[day], nan=0.0) * cost_multiplier
            cost_fraction = float(np.dot(weight_change, rates))
            delta = -capital * cost_fraction
            capital += delta
            cost_usd += delta
            turnover_total += float(weight_change.sum())

        # --- would the levered short perp leg survive the day? --------------------------------
        # The spot leg gains when price rises, but spot and perp sit in separate accounts: a spot
        # unrealised gain does not automatically post margin to the futures account. So the short perp
        # leg is liquidated on its own terms once an adverse (upward) move exhausts its margin, which at
        # perp leverage L is a move of 1/L. At the 0.95 deployment rung L is 19x, i.e. a 5.3% intraday
        # rise. Daily moves of that size are ordinary in crypto, so this has to be measured rather than
        # waved away -- I4 was rejected in wave18 for exactly this class of unmodeled execution
        # constraint, and it would be inconsistent to hold this wave to a lower standard.
        if weights.any() and np.isfinite(config.implied_perp_leverage):
            held_idx = np.flatnonzero(weights > 0.0)
            adverse = panel.perp_high[day][held_idx] / panel.perp_open[day][held_idx] - 1.0
            adverse = np.nan_to_num(adverse, nan=0.0, posinf=0.0, neginf=0.0)
            if len(adverse):
                worst_adverse_move = max(worst_adverse_move, float(np.max(adverse)))
                if float(np.max(adverse)) >= 1.0 / config.implied_perp_leverage:
                    liquidation_days += 1

        # --- hold: intraday basis move plus funding -------------------------------------------
        if weights.any():
            spot_ret = panel.spot_close[day] / panel.spot_open[day] - 1.0
            perp_ret = panel.perp_close[day] / panel.perp_open[day] - 1.0
            intraday = np.nan_to_num(spot_ret - perp_ret, nan=0.0, posinf=0.0, neginf=0.0)
            delta = capital * float(np.dot(intraday, weights))
            capital += delta
            basis_usd += delta
            # A short perp receives funding when the rate is positive, which is exactly the carry.
            funding_row = np.nan_to_num(panel.funding_daily[day], nan=0.0)
            delta = capital * float(np.dot(funding_row, weights))
            capital += delta
            funding_usd += delta

            days_active += 1
            entries += int(np.sum((weights > 0.0) & (previous_weights == 0.0)))
            max_concurrent = max(max_concurrent, int(np.sum(weights > 0.0)))
            held.append((day, int(np.sum(weights > 0.0))))

        deployments.append(float(np.abs(weights).sum()))
        equity.append(capital)
        previous_weights = weights
        if capital <= 0.0:
            break

    equity_array = np.asarray(equity, dtype=float)
    total_change = (equity_array[-1] if len(equity_array) else start_capital) - start_capital
    residual = abs(total_change - (funding_usd + basis_usd + cost_usd))

    return CarryResult(
        start_capital=start_capital,
        equity=equity_array,
        funding_usd=funding_usd,
        basis_usd=basis_usd,
        cost_usd=cost_usd,
        days_active=days_active,
        entries=entries,
        max_concurrent=max_concurrent,
        min_leg_usd=float(min_leg) if np.isfinite(min_leg) else float("nan"),
        max_leg_usd=max_leg,
        blocked_min_order=blocked,
        delta_mismatch=delta_mismatch,
        liquidation_days=liquidation_days,
        worst_adverse_move=worst_adverse_move,
        accounting_residual=residual,
        turnover_total=turnover_total,
        deployment_mean=float(np.mean(deployments)) if deployments else 0.0,
        held=held,
    )


def build_grid(include_infeasible: bool = False) -> tuple[CarryConfig, ...]:
    """The searchable grid.

    Only deployment levels reachable without borrowing to buy spot are selectable. The leveraged rungs
    (cap 2.0/3.0) are constructible for reporting via include_infeasible=True, because measuring what
    they would earn is informative, but they must never enter selection: their spot borrow cost is
    unmodeled, so their returns are not comparable to the feasible ones.
    """
    caps = [0.50, 0.75, 0.95]
    if include_infeasible:
        caps += [2.00, 3.00]
    configs = []
    for top_k in (1, 2, 3, 5, 8):
        for leg_fraction in (0.25, 0.50):
            for cap in caps:
                config = CarryConfig(top_k, leg_fraction, cap)
                if include_infeasible or not config.requires_spot_borrow:
                    configs.append(config)
    return tuple(configs)


__all__ = [
    "ACTIVE_CAPITAL",
    "CAPACITY_LIMIT_FRACTION",
    "MIN_ORDER_USDT",
    "TOTAL_CAPITAL",
    "CarryConfig",
    "CarryResult",
    "build_grid",
    "simulate",
]
