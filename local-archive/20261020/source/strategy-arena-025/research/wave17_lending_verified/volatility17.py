# Wave-17 work item 3: lending-yield variability quantification (SPEC.md 방법 3). Pure
# functions over cache/lending_realized.json's already-computed per-ccy stats (fetch17.py's
# `summarize_history` -- NOT recomputed here, just read and ranked/formatted) plus one derived
# view: how much of F1's high-funding-regime lending contribution survives if every coin's
# lendingRate had instead sat at its OWN observed 4-day floor the whole time (a per-coin,
# not-uniformly-discounted, worst-observed-value view -- complements F2's flat 50% haircut).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))


@dataclass(frozen=True, slots=True)
class CoinVolatility:
    ccy: str
    n_samples: int
    span_days: float
    lending_rate_median: float
    lending_rate_mean: float
    lending_rate_std: float
    lending_rate_min: float
    lending_rate_max: float
    lending_rate_range: float
    lending_rate_cv: float | None  # std/mean; None if mean<=0
    range_over_median: float | None  # (max-min)/median -- how big the swing is relative to the level itself


def build_volatility_table(lending_realized: dict[str, Any]) -> tuple[CoinVolatility, ...]:
    """One row per ccy with `history_available=True`, sorted by `lending_rate_cv` descending
    (most unstable RELATIVE to its own level first) -- CV, not raw std, is the right sort key
    because a 1%-APR coin with 0.3%p std is far LESS reliable (proportionally) than a
    20%-APR coin with the same 0.3%p std, even though raw std ties them."""
    rows: list[CoinVolatility] = []
    for ccy, row in lending_realized["by_ccy"].items():
        if not row.get("history_available"):
            continue
        median = float(row["lending_rate_median"])
        range_over_median = (row["lending_rate_range"] / median) if median > 0.0 else None
        rows.append(
            CoinVolatility(
                ccy=ccy,
                n_samples=int(row["n_samples"]),
                span_days=float(row["span_days"]),
                lending_rate_median=median,
                lending_rate_mean=float(row["lending_rate_mean"]),
                lending_rate_std=float(row["lending_rate_std"]),
                lending_rate_min=float(row["lending_rate_min"]),
                lending_rate_max=float(row["lending_rate_max"]),
                lending_rate_range=float(row["lending_rate_range"]),
                lending_rate_cv=row.get("lending_rate_cv"),
                range_over_median=range_over_median,
            )
        )
    rows.sort(key=lambda r: (r.lending_rate_cv if r.lending_rate_cv is not None else -1.0), reverse=True)
    return tuple(rows)


def universe_volatility_summary(rows: tuple[CoinVolatility, ...]) -> dict[str, Any]:
    """Cross-sectional (across-coin) summary of the per-coin (within-4-day-window) volatility
    stats -- answers 'how unstable is this yield source, typically, across the universe'."""
    if not rows:
        return {"n_coins": 0}
    cvs = [r.lending_rate_cv for r in rows if r.lending_rate_cv is not None]
    range_over_medians = [r.range_over_median for r in rows if r.range_over_median is not None]
    import statistics

    return {
        "n_coins": len(rows),
        "cv_median": float(statistics.median(cvs)) if cvs else None,
        "cv_mean": float(statistics.fmean(cvs)) if cvs else None,
        "cv_max": max(cvs) if cvs else None,
        "range_over_median_median": float(statistics.median(range_over_medians)) if range_over_medians else None,
        "range_over_median_max": max(range_over_medians) if range_over_medians else None,
        "most_unstable_5": [r.ccy for r in rows[:5]],
        "most_stable_5": [r.ccy for r in rows[-5:]],
    }


__all__ = ["CoinVolatility", "build_volatility_table", "universe_volatility_summary"]
