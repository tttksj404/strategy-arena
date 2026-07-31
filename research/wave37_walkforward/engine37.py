# Wave-37 engine: daily cross-sectional funding book, Binance-only, plus a CAUSAL walk-forward.
#
# ---------------------------------------------------------------------------------------
# What is different from wave36 and why it matters
# ---------------------------------------------------------------------------------------
# 1. Single venue. Signal, funding and price all come from the Binance daily cache. wave36's headline
#    was invalidated because its ranking came from one venue and its fills from another, and the two
#    agreed on the top/bottom-k set only ~50% of the time.
# 2. Selection is CAUSAL. Instead of choosing one configuration on in-sample data and testing it once
#    out-of-sample, the walk-forward re-chooses at every step using only data available at that
#    moment, then trades the next 90 days with it. The concatenated result is a single curve on which
#    every point was out-of-sample when it was produced, so there is no holdout to re-open and no
#    multiple-testing correction to argue about.
# 3. The selection score penalises drawdown and low funding share. wave36 maximised walk-forward
#    return alone and therefore selected 3x leverage with a 68% drawdown when 1,785 configurations
#    with sub-25% drawdown existed in the same grid. That was a defect of the selection rule, and it
#    is fixed here rather than discovered again.

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
import pandas as pd  # noqa: PANDAS_OK

from research.wave37_walkforward.dataio37 import DailyPanel

TOTAL_CAPITAL: Final = 100.0
MIN_LEG_USDT: Final = 5.0
LEV_CAP: Final = 3.0
MAINT_MARGIN: Final = 0.005
CAPACITY_LIMIT_USDT: Final = 50_000.0

LOOKBACKS: Final = (3, 7, 14, 30)
K_VALUES: Final = (3, 5, 10, 20)
HOLD_BANDS: Final = (0.10, 0.25, 0.50)
LEVERAGES: Final = (1.0, 2.0, 3.0)
SLEEVE_FRACTIONS: Final = (0.50, 1.00)

TRAIN_DAYS: Final = 365
APPLY_DAYS: Final = 90
MAX_LOOKBACK: Final = max(LOOKBACKS)

DRAWDOWN_TARGET: Final = 0.20
DRAWDOWN_PENALTY: Final = 2.0
FUNDING_SHARE_PENALTY: Final = 0.5


@dataclass(frozen=True)
class Config37:
    lookback_days: int
    k: int
    hold_band: float
    leverage: float
    sleeve_fraction: float

    def to_dict(self) -> dict:
        return {
            "lookback_days": self.lookback_days,
            "k": self.k,
            "hold_band": self.hold_band,
            "leverage": self.leverage,
            "sleeve_fraction": self.sleeve_fraction,
        }


ALL_CONFIGS: Final = tuple(
    Config37(lookback_days=lb, k=k, hold_band=band, leverage=lev, sleeve_fraction=sleeve)
    for lb in LOOKBACKS
    for k in K_VALUES
    for band in HOLD_BANDS
    for lev in LEVERAGES
    for sleeve in SLEEVE_FRACTIONS
)


@dataclass
class Segment:
    start_day: int
    end_day: int
    sleeve_equity: np.ndarray
    funding_usdt: float
    price_pnl_usdt: float
    cost_usdt: float
    margin_call: bool
    min_leg: float
    max_leg: float
    days_in_market: int
    blocked_days: int = 0  # days the book was forced flat by the $5 minimum order size
    held: list[tuple[int, int]] = field(default_factory=list)  # (day, n_legs)


def _signal(panel: DailyPanel, lookback: int) -> np.ndarray:
    """Trailing mean daily funding per symbol, shifted one day: only yesterday's settled funding."""
    frame = pd.DataFrame(panel.funding_daily)
    return frame.rolling(lookback, min_periods=lookback).mean().shift(1).to_numpy()


def simulate(
    panel: DailyPanel,
    config: Config37,
    start_day: int,
    end_day: int,
    signal: np.ndarray,
    starting_sleeve: float | None = None,
    cost_multiplier: float = 1.0,
) -> Segment:
    """Run the book over [start_day, end_day). Decisions use `signal` (already shifted) and fill at
    that day's open, so nothing from the day being traded influences the decision."""
    n_symbols = len(panel.symbols)
    sleeve = TOTAL_CAPITAL * config.sleeve_fraction if starting_sleeve is None else starting_sleeve
    notional = np.zeros(n_symbols)
    curve = np.full(end_day - start_day, np.nan)
    funding_total = price_total = cost_total = 0.0
    margin_call = False
    min_leg, max_leg = np.inf, 0.0
    in_market = 0
    blocked_days = 0
    held_long: list[int] = []
    held_short: list[int] = []
    held_log: list[tuple[int, int]] = []

    for offset, day in enumerate(range(start_day, end_day)):
        # ---- mark existing book on today's move + today's funding ----
        if np.any(notional != 0.0) and day > start_day:
            previous = panel.open[day - 1]
            current = panel.open[day]
            with np.errstate(divide="ignore", invalid="ignore"):
                change = np.where(np.isfinite(previous) & (previous > 0), current / previous - 1.0, 0.0)
            change = np.nan_to_num(change, nan=0.0, posinf=0.0, neginf=0.0)
            price_pnl = float(np.sum(notional * change))
            funding_pnl = float(np.sum(-np.sign(notional) * np.abs(notional) * panel.funding_daily[day]))
            sleeve += price_pnl + funding_pnl
            price_total += price_pnl
            funding_total += funding_pnl
            in_market += 1

        gross = float(np.sum(np.abs(notional)))
        if sleeve <= 0.0 or (gross > 0.0 and sleeve < MAINT_MARGIN * gross):
            charge = float(np.sum(np.abs(notional) * panel.cost_rate * cost_multiplier))
            cost_total += charge
            sleeve = max(0.0, sleeve - charge)
            notional = np.zeros(n_symbols)
            margin_call = True
            curve[offset] = sleeve
            curve[offset:] = sleeve
            break

        # ---- choose today's target book from yesterday's information ----
        row = signal[day]
        usable = np.flatnonzero(np.isfinite(row) & panel.volume_ok[day])
        target_long: list[int] = []
        target_short: list[int] = []
        if len(usable) >= 2 * config.k + 2:
            order = usable[np.argsort(row[usable])]
            width = int(round(config.k + config.hold_band * len(order)))
            width = min(max(width, config.k), len(order))
            long_pool = set(order[:width].tolist())
            short_pool = set(order[-width:].tolist())
            target_long = [i for i in held_long if i in long_pool]
            target_short = [i for i in held_short if i in short_pool]
            for index in order:
                if len(target_long) >= config.k:
                    break
                if index not in target_long and index not in target_short:
                    target_long.append(int(index))
            for index in order[::-1]:
                if len(target_short) >= config.k:
                    break
                if index not in target_short and index not in target_long:
                    target_short.append(int(index))
        held_long, held_short = target_long, target_short

        target = np.zeros(n_symbols)
        if held_long and held_short:
            per_leg = sleeve * config.leverage / (2 * config.k)
            # Exchange minimum order size is a hard constraint, not a reporting statistic. The first
            # run of this wave let legs shrink to $0.40 as the sleeve decayed toward $28 with k=20,
            # which no venue would accept; the book simply cannot be established at that size. Going
            # flat is the honest outcome, and it also removes the false comfort of a book that is
            # profitable only at unplaceable sizes.
            if per_leg < MIN_LEG_USDT:
                held_long, held_short = [], []
                blocked_days += 1
            else:
                for index in held_long:
                    target[index] = per_leg
                for index in held_short:
                    target[index] = -per_leg
                min_leg = min(min_leg, per_leg)
                max_leg = max(max_leg, per_leg)
                held_log.append((day, len(held_long) + len(held_short)))

        turnover = np.abs(target - notional)
        charge = float(np.sum(turnover * panel.cost_rate * cost_multiplier))
        if charge > 0.0:
            cost_total += charge
            sleeve -= charge
        notional = target
        curve[offset] = sleeve

    curve = pd.Series(curve).ffill().bfill().to_numpy()
    return Segment(
        start_day=start_day,
        end_day=end_day,
        sleeve_equity=np.maximum(curve, 0.0),
        funding_usdt=funding_total,
        price_pnl_usdt=price_total,
        cost_usdt=cost_total,
        margin_call=margin_call,
        min_leg=float(min_leg) if np.isfinite(min_leg) else float("nan"),
        max_leg=max_leg,
        days_in_market=in_market,
        blocked_days=blocked_days,
        held=held_log,
    )


def selection_score(segment: Segment, panel: DailyPanel, config: Config37) -> float:
    """SPEC.md frozen score: return, minus 2x the drawdown above 20%, minus a funding-share penalty.

    The funding-share term exists because wave36 earned 56.7% of its P&L from an unvalidated price
    effect while only the funding half had been checked arithmetically. Preferring configurations
    whose profit comes from the verified source is a deliberate bias toward the part we understand.
    """
    if segment.margin_call:
        return -1e9
    curve = segment.sleeve_equity
    if len(curve) < 2 or curve[0] <= 0:
        return -1e9
    days = segment.end_day - segment.start_day
    years = days / 365.25
    ratio = curve[-1] / curve[0]
    annualised = float(ratio ** (1.0 / years) - 1.0) if ratio > 0 and years > 0 else -1.0
    peak = np.maximum.accumulate(curve)
    mdd = float(abs(np.min((curve - peak) / np.maximum(peak, 1e-12))))
    gross = abs(segment.funding_usdt) + abs(segment.price_pnl_usdt)
    funding_share = abs(segment.funding_usdt) / gross if gross > 0 else 0.0
    return (
        annualised
        - DRAWDOWN_PENALTY * max(0.0, mdd - DRAWDOWN_TARGET)
        - FUNDING_SHARE_PENALTY * max(0.0, 1.0 - funding_share)
    )
