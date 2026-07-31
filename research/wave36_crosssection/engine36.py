# Wave-36 cross-sectional market-neutral funding engine.
#
# Structurally different from every prior engine in this repo. engine30 simulates ONE directional
# position at a time and asks whether price moved the right way. This one holds k longs and k shorts
# in equal notional and never expresses a view on direction at all: the return is
# (funding collected on both sides) + (whatever the long/short spread does) - (cost of rotating).
#
# Decisions happen on 8h FUNDING STAMPS, not price bars, because the signal is funding and the
# payment is funding. Signals use only settled stamps shifted by one, so the rate that decides a
# position is money that has already changed hands.
#
# Hysteresis is a first-class part of the design, not a tweak. The pre-registration probe measured
# turnover as the binding cost term: a 1-day lookback rotated 33.9% of the book per stamp and lost
# 38.8% APR to fees, while a hold band cut rotation to 0.9% and turned the same signal into +12.0%
# APR. So `hold_band` is a gene, and the engine implements the band as "keep a leg while it stays
# inside the wider pool, fill free slots from the strict top/bottom k".

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

from research.wave30_qd.dataio30 import MarketCache

TOTAL_CAPITAL: Final = 100.0
MIN_LEG_USDT: Final = 5.0
LEV_CAP: Final = 20.0
MAINT_MARGIN: Final = 0.005
FUNDING_PER_DAY: Final = 3
CAPACITY_LIMIT_USDT: Final = 50_000.0


class InvalidConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config36:
    lookback_days: int
    k: int
    hold_band: float
    leverage: float
    rebalance_every_stamps: int
    min_dispersion_apr: float
    sleeve_fraction: float

    def validate(self) -> "Config36":
        if self.k < 1:
            raise InvalidConfigError("k must be >= 1")
        if not 0.0 <= self.hold_band <= 0.9:
            raise InvalidConfigError("hold_band out of range")
        if not 1.0 <= self.leverage <= LEV_CAP:
            raise InvalidConfigError(f"leverage {self.leverage} outside [1, {LEV_CAP}]")
        if self.rebalance_every_stamps < 1:
            raise InvalidConfigError("rebalance interval must be >= 1 stamp")
        if not 0.0 < self.sleeve_fraction <= 1.0:
            raise InvalidConfigError("sleeve_fraction out of range")
        return self

    def to_dict(self) -> dict:
        return {
            "lookback_days": self.lookback_days,
            "k": self.k,
            "hold_band": self.hold_band,
            "leverage": self.leverage,
            "rebalance_every_stamps": self.rebalance_every_stamps,
            "min_dispersion_apr": self.min_dispersion_apr,
            "sleeve_fraction": self.sleeve_fraction,
        }


@dataclass
class Result36:
    config: Config36
    stamp_index: pd.DatetimeIndex
    sleeve_equity: np.ndarray  # per stamp
    total_equity: np.ndarray  # sleeve + stable leg, $100 basis
    funding_collected_usdt: float
    price_pnl_usdt: float
    cost_paid_usdt: float
    n_rotations: int
    margin_call: bool
    margin_call_stamp: int | None
    max_leg_notional: float
    min_leg_notional: float
    stamps_in_market: int
    book_history: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        if len(self.total_equity) < 2 or self.total_equity[0] <= 0:
            return 0.0
        return float(self.total_equity[-1] / self.total_equity[0] - 1.0)


@dataclass(frozen=True)
class StampPanel:
    """Funding rates and marks sampled at 8h funding stamps.

    Built once and reused by every grid combination -- the grid is 9,720 configurations and rebuilding
    a 5,015 x 20 panel each time would dominate runtime.
    """

    stamps: pd.DatetimeIndex
    symbols: tuple[str, ...]
    funding: np.ndarray  # (n_stamps, n_symbols) realised rate AT that stamp
    price: np.ndarray  # (n_stamps, n_symbols) close at that stamp
    tradable: np.ndarray  # (n_stamps, n_symbols) bool
    cost_rate: np.ndarray  # (n_symbols,) one-way
    day_of_stamp: np.ndarray
    daily_index: pd.DatetimeIndex
    stable_per_dollar: np.ndarray  # per daily index
    is_mask: np.ndarray  # per stamp


def build_stamp_panel(cache: MarketCache) -> StampPanel:
    symbols = tuple(cache.symbols)
    funding_frame = pd.DataFrame(
        {s: pd.Series(cache.arrays[s].funding_at_bar, index=cache.index) for s in symbols}
    )
    # A stamp is any bar where at least one symbol was charged funding.
    stamp_mask = (funding_frame != 0.0).any(axis=1).to_numpy()
    stamps = cache.index[stamp_mask]
    positions = np.flatnonzero(stamp_mask)

    price = np.column_stack([cache.arrays[s].close[positions] for s in symbols])
    tradable = np.column_stack([cache.arrays[s].tradable[positions] for s in symbols])
    funding = funding_frame.to_numpy()[positions]
    return StampPanel(
        stamps=stamps,
        symbols=symbols,
        funding=funding,
        price=price,
        tradable=tradable & np.isfinite(price),
        cost_rate=np.array([cache.arrays[s].cost_rate for s in symbols], dtype=float),
        day_of_stamp=cache.day_of_bar[positions],
        daily_index=cache.daily_index,
        stable_per_dollar=cache.stable_per_dollar,
        is_mask=cache.is_mask[positions],
    )


def _signal(panel: StampPanel, lookback_stamps: int) -> np.ndarray:
    """Trailing mean funding per symbol, shifted one stamp so only settled data is used."""
    frame = pd.DataFrame(panel.funding)
    rolled = frame.rolling(lookback_stamps, min_periods=lookback_stamps).mean().shift(1)
    return rolled.to_numpy()


def run_config(
    panel: StampPanel, config: Config36, cost_multiplier: float = 1.0, is_only: bool = True
) -> Result36:
    config.validate()
    lookback_stamps = max(1, config.lookback_days * FUNDING_PER_DAY)
    signal = _signal(panel, lookback_stamps)
    n_stamps = len(panel.stamps) if not is_only else int(panel.is_mask.sum())

    sleeve_start = TOTAL_CAPITAL * config.sleeve_fraction
    sleeve = sleeve_start
    sleeve_curve = np.full(len(panel.stamps), np.nan)
    notional = np.zeros(len(panel.symbols))  # signed: + long, - short

    funding_total = price_total = cost_total = 0.0
    rotations = 0
    margin_call = False
    margin_call_stamp: int | None = None
    max_leg = 0.0
    min_leg = np.inf
    in_market = 0
    held_long: list[int] = []
    held_short: list[int] = []
    history: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []

    for step in range(n_stamps):
        # ---- 1. mark the existing book to this stamp: price move + funding charged ----
        if step > 0 and np.any(notional != 0.0):
            previous_price = panel.price[step - 1]
            current_price = panel.price[step]
            with np.errstate(divide="ignore", invalid="ignore"):
                change = np.where(previous_price > 0, current_price / previous_price - 1.0, 0.0)
            change = np.nan_to_num(change, nan=0.0, posinf=0.0, neginf=0.0)
            price_pnl = float(np.sum(notional * change))
            # long pays positive funding, short receives it
            funding_pnl = float(np.sum(-np.sign(notional) * np.abs(notional) * panel.funding[step]))
            sleeve += price_pnl + funding_pnl
            price_total += price_pnl
            funding_total += funding_pnl
            in_market += 1

        gross_now = float(np.sum(np.abs(notional)))
        if sleeve <= 0.0 or (gross_now > 0.0 and sleeve < MAINT_MARGIN * gross_now):
            # Margin call: close everything at this stamp and stop. This IS the ruin event for a
            # market-neutral book -- it does not need a single leg to go to zero, only the equity
            # backing the gross exposure to fall below maintenance.
            cost_total += float(np.sum(np.abs(notional) * panel.cost_rate * cost_multiplier))
            sleeve = max(0.0, sleeve - float(np.sum(np.abs(notional) * panel.cost_rate * cost_multiplier)))
            notional = np.zeros(len(panel.symbols))
            margin_call = True
            margin_call_stamp = step
            sleeve_curve[step] = sleeve
            break

        # ---- 2. decide the target book ----
        rebalance = step % config.rebalance_every_stamps == 0
        if rebalance:
            row = signal[step]
            usable = np.flatnonzero(np.isfinite(row) & panel.tradable[step])
            target_long: list[int] = []
            target_short: list[int] = []
            if len(usable) >= 2 * config.k + 2:
                order = usable[np.argsort(row[usable])]
                dispersion_apr = float(
                    (row[order[-config.k :]].mean() - row[order[: config.k]].mean())
                    * FUNDING_PER_DAY
                    * 365
                )
                if dispersion_apr >= config.min_dispersion_apr:
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

            # ---- 3. size to target and charge the turnover ----
            target = np.zeros(len(panel.symbols))
            if held_long and held_short:
                gross = sleeve * config.leverage
                per_leg = gross / (2 * config.k)
                for index in held_long:
                    target[index] = per_leg
                for index in held_short:
                    target[index] = -per_leg
                max_leg = max(max_leg, per_leg)
                min_leg = min(min_leg, per_leg)
            turnover = np.abs(target - notional)
            charge = float(np.sum(turnover * panel.cost_rate * cost_multiplier))
            if charge > 0.0:
                cost_total += charge
                sleeve -= charge
                rotations += int(np.count_nonzero(turnover > 1e-9))
            notional = target
            history.append((step, tuple(panel.symbols[i] for i in held_long), tuple(panel.symbols[i] for i in held_short)))

        sleeve_curve[step] = sleeve

    # forward-fill the curve and add the stable leg
    filled = pd.Series(sleeve_curve).ffill().fillna(sleeve_start).to_numpy()
    filled = np.maximum(filled, 0.0)
    stable_start = TOTAL_CAPITAL - sleeve_start
    stable = stable_start * panel.stable_per_dollar[panel.day_of_stamp]
    total = filled + stable
    return Result36(
        config=config,
        stamp_index=panel.stamps,
        sleeve_equity=filled,
        total_equity=total,
        funding_collected_usdt=funding_total,
        price_pnl_usdt=price_total,
        cost_paid_usdt=cost_total,
        n_rotations=rotations,
        margin_call=margin_call,
        margin_call_stamp=margin_call_stamp,
        max_leg_notional=max_leg,
        min_leg_notional=float(min_leg) if np.isfinite(min_leg) else float("nan"),
        stamps_in_market=in_market,
        book_history=history,
    )
