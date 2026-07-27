# Wave-22 validation #3 -- regime decomposition. High-funding years (2020, 2021, 2024) vs
# low-funding years (2022, 2023, 2025, 2026) -- task's own bucketing, which happens to match
# research/wave10_carry100/regime.py's own HIGH_FUNDING_YEARS constant for the high side. Per-
# calendar-year annualized return is computed with gates21.yearly_annualized_returns (REUSED,
# not reimplemented -- identical anchoring convention already used for wave21's own H5 gate), for
# both G1 and I5, on the SAME full equity curves validation #2 (rolling.py) already computed.

from __future__ import annotations

from pathlib import Path
import statistics
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave10_carry100.engine import OOS_SPLIT
from research.wave21_ga.gates21 import yearly_annualized_returns

HIGH_FUNDING_YEARS: Final[tuple[int, ...]] = (2020, 2021, 2024)
LOW_FUNDING_YEARS: Final[tuple[int, ...]] = (2022, 2023, 2025, 2026)


def _bucket_for(year: int) -> str:
    if year in HIGH_FUNDING_YEARS:
        return "high_funding"
    if year in LOW_FUNDING_YEARS:
        return "low_funding"
    return "unclassified"


def _bucket_stats(rows: list[dict[str, Any]], bucket_name: str, expected_years: tuple[int, ...]) -> dict[str, Any]:
    bucket_rows = [row for row in rows if row["bucket"] == bucket_name and row["g1_minus_i5_pp"] is not None]
    years_present = [row["year"] for row in bucket_rows]
    years_missing = [year for year in expected_years if year not in years_present]
    if not bucket_rows:
        return {
            "years_expected": list(expected_years), "years_present": [], "years_missing": years_missing,
            "n_years": 0, "mean_g1_minus_i5_pp": None, "median_g1_minus_i5_pp": None,
            "g1_win_count": 0, "g1_win_rate": None,
        }
    deltas = [row["g1_minus_i5_pp"] for row in bucket_rows]
    wins = sum(1 for row in bucket_rows if row["g1_wins"])
    return {
        "years_expected": list(expected_years),
        "years_present": years_present,
        "years_missing": years_missing,
        "n_years": len(bucket_rows),
        "mean_g1_minus_i5_pp": sum(deltas) / len(deltas),
        "median_g1_minus_i5_pp": statistics.median(deltas),
        "g1_win_count": wins,
        "g1_win_rate": wins / len(bucket_rows),
    }


def run(g1_equity: pd.Series, i5_equity: pd.Series) -> dict[str, Any]:
    g1_years = yearly_annualized_returns(g1_equity)
    i5_years = yearly_annualized_returns(i5_equity)
    all_years = sorted(set(g1_years) | set(i5_years))

    data_start_year = min(pd.Timestamp(g1_equity.index[0]).year, pd.Timestamp(i5_equity.index[0]).year)
    data_end_year = max(pd.Timestamp(g1_equity.index[-1]).year, pd.Timestamp(i5_equity.index[-1]).year)
    oos_split_year = pd.Timestamp(OOS_SPLIT).year

    rows: list[dict[str, Any]] = []
    for year in all_years:
        g1_value = g1_years.get(year)
        i5_value = i5_years.get(year)
        delta_pp = (g1_value - i5_value) * 100.0 if (g1_value is not None and i5_value is not None) else None
        rows.append({
            "year": year,
            "bucket": _bucket_for(year),
            "g1_cagr": g1_value,
            "i5_cagr": i5_value,
            "g1_minus_i5_pp": delta_pp,
            "g1_wins": (g1_value > i5_value) if (g1_value is not None and i5_value is not None) else None,
            "is_partial_year": bool(year == data_start_year or year == data_end_year),
            "straddles_oos_split": bool(year == oos_split_year),
        })

    high = _bucket_stats(rows, "high_funding", HIGH_FUNDING_YEARS)
    low = _bucket_stats(rows, "low_funding", LOW_FUNDING_YEARS)

    dominant_regime = None
    if high["mean_g1_minus_i5_pp"] is not None and low["mean_g1_minus_i5_pp"] is not None:
        dominant_regime = "high_funding" if abs(high["mean_g1_minus_i5_pp"]) > abs(low["mean_g1_minus_i5_pp"]) else "low_funding"

    both_positive = (high["mean_g1_minus_i5_pp"] or 0) > 0 and (low["mean_g1_minus_i5_pp"] or 0) > 0
    only_one_positive = None
    if high["mean_g1_minus_i5_pp"] is not None and low["mean_g1_minus_i5_pp"] is not None:
        high_pos, low_pos = high["mean_g1_minus_i5_pp"] > 0, low["mean_g1_minus_i5_pp"] > 0
        if high_pos != low_pos:
            only_one_positive = "high_funding" if high_pos else "low_funding"

    return {
        "methodology": {
            "high_funding_years": list(HIGH_FUNDING_YEARS),
            "low_funding_years": list(LOW_FUNDING_YEARS),
            "per_year_engine": "gates21.yearly_annualized_returns (reused, same anchoring as wave21's own H5 gate)",
            "note": "2019 and 2026 are partial calendar years in this dataset (data starts 2019-09, ends 2026-07); 2025 straddles OOS_SPLIT (2025-09-30) -- part of 2025 was visible to the wave-21 GA during evolution (IS), part was sealed (OOS). A year label answers 'which funding regime' only, not 'was this IS or OOS'.",
        },
        "by_year": rows,
        "high_funding": high,
        "low_funding": low,
        "dominant_regime_by_magnitude": dominant_regime,
        "improvement_only_in_one_regime": only_one_positive,
        "improvement_positive_in_both_regimes": bool(both_positive),
        "limitations": [
            f"only {high['n_years']} high-funding years and {low['n_years']} low-funding years exist in the data -- per-bucket means rest on very few (3-4) independent annual observations each",
            "2019 (partial, Sep-Dec only) is excluded from both buckets by the task's own year lists; 2026 is also partial (through the cache's own last date) and IS included in the low-funding bucket as specified, so its contribution is a part-year annualized figure, not a full year",
            "2025 straddles OOS_SPLIT -- its bucket membership (low-funding) is a funding-regime statement, independent of validation #2's IS/OOS framing; do not conflate the two axes",
        ],
    }


__all__ = ["HIGH_FUNDING_YEARS", "LOW_FUNDING_YEARS", "run"]
