# Wave-35 portfolio assembly + metrics. Takes the per-symbol unit-base trade streams produced by
# sim35.simulate_symbol and combines them under the SPEC.md section-5 capital model:
# $100 fully deployed, fixed notional $100/N per entry, no compounding, no stop.
#
# Point-in-time selection (SPEC.md R3) is applied to ENTRIES only: a trade is taken iff its symbol
# was in the selected set on its entry date. A position already open when its symbol leaves the set
# runs to its natural reversal/end-of-data exit (R4) -- force-closing it would introduce a rule V1
# does not have.

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave35_breadth.sim35 import RESELECT_DAYS, SymbolRun, T35

RUIN_FLOOR_USDT = 50.0   # configs20.G2_RUIN_FLOOR_USDT -- campaign-wide ruin definition
MC_PATHS = 10_000


@dataclass(slots=True)
class PortfolioRun:
    label: str
    n_symbols: int | str
    leverage: float
    base_per_entry: float
    equity: pd.Series
    taken: list[dict] = field(default_factory=list)   # accepted trades, $ terms
    n_candidates: int = 0
    n_rejected_unfunded: int = 0
    symbols_traded: tuple[str, ...] = ()


def build_calendar(runs: dict[str, SymbolRun]) -> pd.DatetimeIndex:
    dates: set[pd.Timestamp] = set()
    for run in runs.values():
        dates.update(run.index)
    return pd.DatetimeIndex(sorted(dates))


def eligibility_frames(runs: dict[str, SymbolRun], calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(eligible bool frame, trailing $-volume frame) on the shared calendar. Both are built only
    from each symbol's own point-in-time series -- no cross-sectional or full-sample statistic."""
    elig = {}
    vol = {}
    for sym, run in runs.items():
        elig[sym] = pd.Series(run.eligible, index=run.index).reindex(calendar).fillna(False).astype(bool)
        vol[sym] = pd.Series(run.trailing_dollar_vol, index=run.index).reindex(calendar)
    return pd.DataFrame(elig), pd.DataFrame(vol)


def selection_frame(elig: pd.DataFrame, vol: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """SPEC.md R3. Every RESELECT_DAYS-th date, rank that date's eligible symbols by their own
    trailing-30d $ volume (already shift(1)-ed, so nothing from date t or later is used) and keep
    the top n. n=None means 'all eligible'. The chosen set is held for the next block."""
    if n is None:
        return elig.copy()
    sel = pd.DataFrame(False, index=elig.index, columns=elig.columns)
    anchors = list(range(0, len(elig.index), RESELECT_DAYS))
    for k, start in enumerate(anchors):
        end = anchors[k + 1] if k + 1 < len(anchors) else len(elig.index)
        d = elig.index[start]
        row_elig = elig.loc[d]
        pool = row_elig[row_elig].index
        if len(pool) == 0:
            continue
        ranked = vol.loc[d, pool].dropna().sort_values(ascending=False)
        chosen = list(ranked.index[:n])
        if chosen:
            sel.iloc[start:end, sel.columns.get_indexer(chosen)] = True
    # a symbol still has to be individually eligible on the day it is entered
    return sel & elig


def assemble(runs: dict[str, SymbolRun], calendar: pd.DatetimeIndex, sel: pd.DataFrame,
             n_slots: int, leverage: float, capital: float = 100.0,
             label: str = "", max_concurrent: int | None = None,
             single_sleeve: bool = False) -> PortfolioRun:
    """Sequential walk over the shared calendar. Trades are accepted in (entry date, symbol) order;
    an entry is taken only if the account can still fund a FULL-size base (shrinking the base would
    silently re-introduce compounding -- SPEC.md 5)."""
    base = capital if single_sleeve else capital / n_slots
    pos_by_date: dict[int, list[tuple[str, T35]]] = {}
    date_pos = {d: i for i, d in enumerate(calendar)}
    n_candidates = 0
    for sym, run in runs.items():
        selcol = sel[sym].to_numpy() if sym in sel.columns else None
        if selcol is None:
            continue
        for tr in run.trades:
            gi = date_pos.get(tr.entry_date)
            if gi is None or not selcol[gi]:
                continue
            n_candidates += 1
            pos_by_date.setdefault(gi, []).append((sym, tr))

    n_days = len(calendar)
    daily_pnl = np.zeros(n_days, dtype=float)
    equity = capital
    open_until: dict[str, int] = {}
    taken: list[dict] = []
    rejected_unfunded = 0
    cap = max_concurrent if max_concurrent is not None else (1 if single_sleeve else n_slots)
    equity_out = np.empty(n_days, dtype=float)

    for gi in range(n_days):
        for sym in [s for s, u in open_until.items() if u < gi]:
            del open_until[sym]
        for sym, tr in sorted(pos_by_date.get(gi, []), key=lambda x: x[0]):
            if sym in open_until:
                continue
            if len(open_until) >= cap:
                continue
            if equity < base:
                rejected_unfunded += 1
                continue
            exit_gi = date_pos.get(tr.exit_date)
            if exit_gi is None:
                continue
            path = tr.mark_path
            prev = 0.0
            for k, m in enumerate(path):
                j = gi + k
                if j >= n_days:
                    break
                daily_pnl[j] += (m - prev) * base
                prev = m
            open_until[sym] = gi + len(path) - 1
            taken.append({
                "symbol": sym, "entry": tr.entry_date, "exit": tr.exit_date,
                "direction": tr.direction, "roi": tr.roi, "pnl": tr.roi * base,
                "reason": tr.reason, "hold_days": len(path),
            })
        equity += daily_pnl[gi]
        equity_out[gi] = equity

    return PortfolioRun(
        label=label, n_symbols=n_slots, leverage=leverage, base_per_entry=base,
        equity=pd.Series(equity_out, index=calendar), taken=taken,
        n_candidates=n_candidates, n_rejected_unfunded=rejected_unfunded,
        symbols_traded=tuple(sorted({t["symbol"] for t in taken})),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _mdd(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((1.0 - equity / peak.replace(0.0, np.nan)).max(skipna=True) or 0.0)


def mc_ruin(pnls: list[float], capital: float = 100.0, paths: int = MC_PATHS,
            floor: float = RUIN_FLOOR_USDT, seed: int = 20260803) -> float:
    """Bootstrap the realised per-entry $ stream (same count, with replacement) and measure
    P(final < $50) -- the campaign's standing ruin definition (configs20.G2_RUIN_FLOOR_USDT)."""
    if not pnls:
        return 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(pnls, dtype=float)
    draws = rng.choice(arr, size=(paths, len(arr)), replace=True)
    finals = capital + draws.cumsum(axis=1)
    # an account that goes to zero mid-path cannot recover -- treat the running minimum <= 0 as dead
    dead = (finals <= 0.0).any(axis=1)
    final = finals[:, -1]
    return float(np.mean(dead | (final < floor)))


def metrics(run: PortfolioRun, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
            capital: float = 100.0) -> dict:
    eq = run.equity
    if start is not None:
        eq = eq[eq.index >= start]
    if end is not None:
        eq = eq[eq.index <= end]
    trades = [t for t in run.taken
              if (start is None or t["entry"] >= start) and (end is None or t["entry"] <= end)]
    pnls = [t["pnl"] for t in trades]

    n_days = len(eq)
    years = n_days / 365.25 if n_days else 0.0
    if len(eq):
        final = float(eq.iloc[-1])
        start_eq = float(eq.iloc[0])
    else:
        final = start_eq = capital
    # rebase so a windowed slice still reads "on $100"
    rebased = capital + (eq - start_eq) if len(eq) else eq
    final_on_100 = capital + (final - start_eq)
    cagr = (final_on_100 / capital) ** (1.0 / years) - 1.0 if years > 0 and final_on_100 > 0 else float("nan")

    total = float(sum(pnls))
    ordered = sorted(pnls, reverse=True)
    gross_profit = float(sum(p for p in pnls if p > 0))
    best_share = (ordered[0] / total) if (ordered and total > 0) else float("nan")
    top5_removed = float(sum(ordered[5:])) if len(ordered) > 5 else 0.0

    out = {
        "label": run.label,
        "n_symbols": run.n_symbols,
        "leverage": run.leverage,
        "base_per_entry": run.base_per_entry,
        "final_on_100": final_on_100,
        "cagr": cagr,
        "mdd": _mdd(rebased) if len(rebased) else 0.0,
        "n_days": n_days,
        "n_trades": len(trades),
        "trades_per_active_day": len(trades) / n_days if n_days else 0.0,
        "win_rate": (sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else float("nan"),
        "mean_per_entry": (total / len(pnls)) if pnls else float("nan"),
        "median_per_entry": statistics.median(pnls) if pnls else float("nan"),
        "p_ge_10": (sum(1 for p in pnls if p >= 10.0) / len(pnls)) if pnls else float("nan"),
        "p_le_m10": (sum(1 for p in pnls if p <= -10.0) / len(pnls)) if pnls else float("nan"),
        "best_entry": max(pnls) if pnls else float("nan"),
        "worst_entry": min(pnls) if pnls else float("nan"),
        "total_pnl": total,
        "gross_profit": gross_profit,
        "best_trade_share_of_profit": best_share,
        "total_ex_top5": top5_removed,
        "ruin_prob": mc_ruin(pnls, capital=capital),
        "n_unique_symbols": len({t["symbol"] for t in trades}),
        "n_candidates": run.n_candidates,
        "n_rejected_unfunded": run.n_rejected_unfunded,
    }
    return out
