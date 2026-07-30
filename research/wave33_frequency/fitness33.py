# Wave-33 evaluation: per-ENTRY dollar P&L at fixed $100 sizing, under a frequency constraint.
#
# ---------------------------------------------------------------------------------------
# Why dollars, and why fixed sizing
# ---------------------------------------------------------------------------------------
# The request was "at least $10 per entry, at least one entry a day". Under compounding that
# question has no answer: trade #900's dollars are sized off a different equity than trade #1's,
# so "per-entry dollars" is not a single quantity. engine30.SizingMode(fixed_base=True) pins
# every position to the STARTING sleeve, which makes per-entry P&L one common unit and lets a
# "$10" bar mean the same thing on the first trade and the last.
#
# ---------------------------------------------------------------------------------------
# Why frequency is measured over the ACTIVE span, not the calendar span
# ---------------------------------------------------------------------------------------
# A pre-registration probe found that even low-risk high-frequency configurations drained the
# account to ~$33 (three fixed $33.33 slots) and then stopped trading. Dividing trade count by
# the full 2,214-day IS calendar then reports 0.69 trades/day for something that actually traded
# ~3/day for a few hundred days and died. Both numbers matter and they mean different things, so
# `trades_per_active_day` (count / first-entry-to-last-exit) and `survived_full_span` are kept
# separate, and gate F2 demands both.

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

from research.wave30_qd.dataio30 import MarketCache, OOS_SPLIT
from research.wave30_qd.engine30 import (
    SizingMode,
    Wave30Result,
    max_drawdown,
    run_genome,
)
from research.wave30_qd.fitness30 import LEVERAGE_EDGES, Evaluation
from research.wave30_qd.genome30 import Genome

FIXED_SIZING: Final = SizingMode(fixed_base=True)
TARGET_PER_ENTRY_USDT: Final = 10.0  # the user's bar, frozen in SPEC.md
MIN_TRADES_PER_ACTIVE_DAY: Final = 1.0
INFEASIBLE_PENALTY: Final = 100.0
MIN_TRADES_FOR_STATS: Final = 30

# MAP-Elites axes (SPEC.md)
FREQUENCY_EDGES: Final = (0.0, 0.25, 1.0, 2.0, 4.0, 8.0, np.inf)
BIG_WIN_SHARE_EDGES: Final = (0.0, 0.05, 0.15, 0.30, 0.50, 1.0001)
GRID_SHAPE: Final = (len(FREQUENCY_EDGES) - 1, len(LEVERAGE_EDGES) - 1, len(BIG_WIN_SHARE_EDGES) - 1)


@dataclass(frozen=True)
class EntryProfile:
    """Everything the request asks about, measured in dollars per entry."""

    n_trades: int
    active_days: float
    trades_per_active_day: float
    survived_full_span: bool
    median_usdt: float
    mean_usdt: float
    p05_usdt: float
    p95_usdt: float
    best_usdt: float
    worst_usdt: float
    win_share: float
    share_ge_target: float  # P(entry nets >= +$10)
    share_le_negative_target: float  # P(entry loses >= $10)
    total_usdt: float
    account_final_usdt: float
    account_mdd: float
    capital_for_target_ev: float  # capital at which the EV per entry would be $10

    def as_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "active_days": self.active_days,
            "trades_per_active_day": self.trades_per_active_day,
            "survived_full_span": self.survived_full_span,
            "median_usdt": self.median_usdt,
            "mean_usdt": self.mean_usdt,
            "p05_usdt": self.p05_usdt,
            "p95_usdt": self.p95_usdt,
            "best_usdt": self.best_usdt,
            "worst_usdt": self.worst_usdt,
            "win_share": self.win_share,
            "share_ge_target": self.share_ge_target,
            "share_le_negative_target": self.share_le_negative_target,
            "total_usdt": self.total_usdt,
            "account_final_usdt": self.account_final_usdt,
            "account_mdd": self.account_mdd,
            "capital_for_target_ev": self.capital_for_target_ev,
        }


def _capital_for_target_ev(mean_usdt: float, base_usdt: float) -> float:
    """Capital at which the AVERAGE entry would net $10.

    Per-entry P&L scales linearly with the position base, so if $100 of base yields `mean_usdt`
    on average then `$10 / mean_usdt * base` is the base that yields $10. Returns inf when the
    edge is non-positive, because no amount of capital makes a losing average profitable --
    scaling a negative expectancy only loses money faster.
    """
    if mean_usdt <= 0.0 or base_usdt <= 0.0:
        return float("inf")
    return TARGET_PER_ENTRY_USDT / mean_usdt * base_usdt


def entry_profile(cache: MarketCache, result: Wave30Result, span_start_day: int = 0) -> EntryProfile:
    trades = result.trades
    valid = result.daily_valid
    equity = result.sleeve_equity_daily[valid]
    n_days = int(valid.sum())

    if not trades:
        return EntryProfile(
            0, 0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            float(equity[-1]) if len(equity) else 0.0, 0.0, float("inf"),
        )

    pnl = np.array([t.net_return_on_base * t.base_usdt for t in trades], dtype=float)
    first_day = int(cache.day_of_bar[trades[0].entry_bar])
    last_day = int(cache.day_of_bar[max(t.exit_bar for t in trades)])
    active_days = float(max(1, last_day - first_day + 1))

    # "Survived" = still able to open a full-size position at the very end of the span. The
    # engine stops trading the moment it cannot, so a run whose last exit is far from the span
    # end has died rather than simply gone quiet.
    survived = bool(last_day >= n_days - 1 - 30)

    base = float(trades[0].base_usdt)
    peak = np.maximum.accumulate(equity)
    mdd = float(abs(np.min((equity - peak) / np.maximum(peak, 1e-12)))) if len(equity) else 0.0
    return EntryProfile(
        n_trades=len(trades),
        active_days=active_days,
        trades_per_active_day=len(trades) / active_days,
        survived_full_span=survived,
        median_usdt=float(np.median(pnl)),
        mean_usdt=float(pnl.mean()),
        p05_usdt=float(np.percentile(pnl, 5)),
        p95_usdt=float(np.percentile(pnl, 95)),
        best_usdt=float(pnl.max()),
        worst_usdt=float(pnl.min()),
        win_share=float((pnl > 0).mean()),
        share_ge_target=float((pnl >= TARGET_PER_ENTRY_USDT).mean()),
        share_le_negative_target=float((pnl <= -TARGET_PER_ENTRY_USDT).mean()),
        total_usdt=float(pnl.sum()),
        account_final_usdt=float(equity[-1]),
        account_mdd=mdd,
        capital_for_target_ev=_capital_for_target_ev(float(pnl.mean()), base),
    )


def descriptor_of(trades_per_active_day: float, mean_leverage: float, share_ge_target: float) -> tuple[int, int, int]:
    freq_bin = int(np.clip(np.searchsorted(FREQUENCY_EDGES, trades_per_active_day, side="right") - 1, 0, GRID_SHAPE[0] - 1))
    lev_bin = int(np.clip(np.searchsorted(LEVERAGE_EDGES, max(mean_leverage, 1.0), side="right") - 1, 0, GRID_SHAPE[1] - 1))
    win_bin = int(np.clip(np.searchsorted(BIG_WIN_SHARE_EDGES, share_ge_target, side="right") - 1, 0, GRID_SHAPE[2] - 1))
    return freq_bin, lev_bin, win_bin


def evaluate_frequency(cache: MarketCache, genome: Genome, rng: np.random.Generator) -> Evaluation:
    """IS-only. Signature matches fitness30.evaluate so search30.Evaluator can inject it."""
    result = run_genome(cache, genome, mode="is", sizing=FIXED_SIZING)
    profile = entry_profile(cache, result)

    fitness = profile.median_usdt
    infeasible_reasons: list[str] = []
    if profile.n_trades < MIN_TRADES_FOR_STATS:
        infeasible_reasons.append("too few trades")
    if profile.trades_per_active_day < MIN_TRADES_PER_ACTIVE_DAY:
        infeasible_reasons.append("below 1 entry per active day")
    if not profile.survived_full_span:
        infeasible_reasons.append("account died before the span ended")
    if infeasible_reasons:
        fitness -= INFEASIBLE_PENALTY

    return Evaluation(
        genome=genome,
        fitness=float(fitness),
        fold_cagrs=(),
        is_total_cagr=0.0,  # meaningless under fixed sizing; total_usdt carries the information
        is_total_final=profile.account_final_usdt,
        sleeve_mdd=profile.account_mdd,
        total_mdd=profile.account_mdd,
        trades_per_year=profile.trades_per_active_day * 365.0,
        n_trades=profile.n_trades,
        n_liquidations=result.n_liquidations,
        wipe_probability=0.0,  # F5 computes this once, on the judged candidate only
        descriptor=descriptor_of(profile.trades_per_active_day, result.mean_realized_leverage, profile.share_ge_target),
        mean_leverage=float(result.mean_realized_leverage),
        min_notional_usdt=float(result.min_notional_usdt),
        sleeve_survived=profile.survived_full_span,
        extras={
            "entry_profile": profile.as_dict(),
            "infeasible_reasons": infeasible_reasons,
            "objective_vector": (
                -float(fitness),
                float(profile.share_le_negative_target),
                float(profile.account_mdd),
            ),
        },
    )


def oos_entry_profile(cache: MarketCache, genome: Genome) -> tuple[dict, dict]:
    """THE OOS unsealing. Returns (is_profile, oos_profile) measured on one full-span run.

    Trades are partitioned by exit day rather than re-running the engine on a sliced calendar,
    because a fixed-size account's OOS behaviour depends on the equity it carried in.
    """
    result = run_genome(cache, genome, mode="full", sizing=FIXED_SIZING)
    oos_day = int(cache.daily_index.searchsorted(OOS_SPLIT, side="right"))
    pnl = np.array([t.net_return_on_base * t.base_usdt for t in result.trades], dtype=float)
    exit_days = np.array([int(cache.day_of_bar[t.exit_bar]) for t in result.trades])
    out: dict[str, dict] = {}
    for label, mask in (("is", exit_days < oos_day), ("oos", exit_days >= oos_day)):
        values = pnl[mask]
        if len(values) == 0:
            out[label] = {"n_trades": 0, "median_usdt": 0.0, "mean_usdt": 0.0, "share_ge_target": 0.0,
                          "share_le_negative_target": 0.0, "total_usdt": 0.0, "win_share": 0.0,
                          "trades_per_active_day": 0.0}
            continue
        days = float(max(1, exit_days[mask].max() - exit_days[mask].min() + 1))
        out[label] = {
            "n_trades": int(len(values)),
            "active_days": days,
            "trades_per_active_day": float(len(values) / days),
            "median_usdt": float(np.median(values)),
            "mean_usdt": float(values.mean()),
            "p05_usdt": float(np.percentile(values, 5)),
            "p95_usdt": float(np.percentile(values, 95)),
            "win_share": float((values > 0).mean()),
            "share_ge_target": float((values >= TARGET_PER_ENTRY_USDT).mean()),
            "share_le_negative_target": float((values <= -TARGET_PER_ENTRY_USDT).mean()),
            "total_usdt": float(values.sum()),
            "capital_for_target_ev": _capital_for_target_ev(
                float(values.mean()), float(result.trades[0].base_usdt)
            ),
        }
    full = entry_profile(cache, result)
    out["full"] = full.as_dict()
    out["_meta"] = {
        "n_liquidations": result.n_liquidations,
        "min_notional_usdt": result.min_notional_usdt,
        "max_notional_usdt": float(max((t.notional_usdt for t in result.trades), default=float("nan"))),
        "mean_leverage": result.mean_realized_leverage,
        "base_usdt": float(result.trades[0].base_usdt) if result.trades else 0.0,
        "trade_returns": result.trade_returns.tolist(),
        "account_curve_final": float(result.total_equity_daily[-1]),
    }
    return out, out["_meta"]
