# Wave-22 validation #2 -- time stability (rolling walk-forward). Cuts the full 2019-09~2026-07
# history into NON-OVERLAPPING, consecutive 6-month calendar windows (not a sliding/overlapping
# window -- overlapping windows would pseudo-replicate the same days across many "independent"
# win/loss observations and inflate the apparent sample size behind the win-rate statistic; see
# the "limitations" block in this module's output) and computes G1's and I5's own annualized
# return within each window, using the SAME half-open-interval, last-observation-on-or-before
# anchoring convention research/wave21_ga/gates21.py's yearly_annualized_returns and
# research/wave10_carry100/regime.py's _regime_return already use elsewhere in this repo (so a
# window's return correctly includes its own transition day). Each window's CAGR is obtained by
# calling fitness.cagr() on a 2-point [anchor, window-end] series -- reusing the task-mandated
# engine function rather than reimplementing annualization math.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave10_carry100.engine import OOS_SPLIT
from research.wave21_ga import fitness

WINDOW_MONTHS: Final = 6
MIN_OBS_FOR_CONFIDENCE: Final = 60  # ~2 months of daily obs; windows below this are flagged low-confidence, not dropped


def six_month_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    boundaries = [pd.Timestamp(start)]
    end_ts = pd.Timestamp(end)
    while boundaries[-1] < end_ts:
        boundaries.append(boundaries[-1] + pd.DateOffset(months=WINDOW_MONTHS))
    return list(zip(boundaries[:-1], boundaries[1:]))


def _window_slice(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any] | None:
    mask = (equity.index > start) & (equity.index <= end)
    window = equity[mask]
    if window.empty:
        return None
    pre = equity[equity.index <= start]
    anchor_value = float(pre.iloc[-1]) if len(pre) else float(window.iloc[0])
    anchor_ts = start if len(pre) else pd.Timestamp(window.index[0])
    end_value = float(window.iloc[-1])
    end_ts = pd.Timestamp(window.index[-1])
    two_point = pd.Series([anchor_value, end_value], index=[anchor_ts, end_ts])
    return {
        "cagr": fitness.cagr(two_point),
        "total_return": (end_value / anchor_value - 1.0) if anchor_value > 0 else None,
        "n_obs": int(len(window)),
        "anchor": str(anchor_ts),
        "end": str(end_ts),
    }


def _streaks(flags: list[bool]) -> dict[str, int]:
    best_g1, best_i5, current, current_flag = 0, 0, 0, None
    for flag in flags:
        if flag == current_flag:
            current += 1
        else:
            current, current_flag = 1, flag
        if flag:
            best_g1 = max(best_g1, current)
        else:
            best_i5 = max(best_i5, current)
    return {"longest_g1_win_streak": best_g1, "longest_i5_win_streak": best_i5}


def run(g1_equity: pd.Series, i5_equity: pd.Series) -> dict[str, Any]:
    start = min(g1_equity.index[0], i5_equity.index[0])
    end = max(g1_equity.index[-1], i5_equity.index[-1])
    windows = six_month_windows(start, end)

    rows: list[dict[str, Any]] = []
    for window_start, window_end in windows:
        g1_slice = _window_slice(g1_equity, window_start, window_end)
        i5_slice = _window_slice(i5_equity, window_start, window_end)
        row: dict[str, Any] = {
            "window_start": str(window_start),
            "window_end": str(window_end),
            "contains_oos": bool(window_end > OOS_SPLIT),
            "fully_oos": bool(window_start >= OOS_SPLIT),
            "low_confidence": None,
            "g1_cagr": None,
            "i5_cagr": None,
            "g1_minus_i5_pp": None,
            "g1_wins": None,
            "note": "",
        }
        if g1_slice is None or i5_slice is None:
            row["note"] = "no observations in this window for one or both series -- excluded from win-rate count"
            rows.append(row)
            continue
        g1_wins = bool(g1_slice["cagr"] > i5_slice["cagr"])
        row.update(
            g1_cagr=g1_slice["cagr"],
            i5_cagr=i5_slice["cagr"],
            g1_minus_i5_pp=(g1_slice["cagr"] - i5_slice["cagr"]) * 100.0,
            g1_wins=g1_wins,
            g1_n_obs=g1_slice["n_obs"],
            i5_n_obs=i5_slice["n_obs"],
            low_confidence=bool(min(g1_slice["n_obs"], i5_slice["n_obs"]) < MIN_OBS_FOR_CONFIDENCE),
        )
        rows.append(row)

    counted_rows = [row for row in rows if row["g1_wins"] is not None]
    n_counted = len(counted_rows)
    n_wins = sum(1 for row in counted_rows if row["g1_wins"])
    win_rate = (n_wins / n_counted) if n_counted else None

    pre_oos_rows = [row for row in counted_rows if not row["contains_oos"]]
    oos_touching_rows = [row for row in counted_rows if row["contains_oos"]]
    win_rate_pre_oos = (sum(1 for row in pre_oos_rows if row["g1_wins"]) / len(pre_oos_rows)) if pre_oos_rows else None
    win_rate_oos_touching = (sum(1 for row in oos_touching_rows if row["g1_wins"]) / len(oos_touching_rows)) if oos_touching_rows else None

    midpoint = n_counted // 2
    first_half, second_half = counted_rows[:midpoint], counted_rows[midpoint:]
    win_rate_first_half = (sum(1 for row in first_half if row["g1_wins"]) / len(first_half)) if first_half else None
    win_rate_second_half = (sum(1 for row in second_half if row["g1_wins"]) / len(second_half)) if second_half else None

    streaks = _streaks([bool(row["g1_wins"]) for row in counted_rows])
    n_low_confidence = sum(1 for row in counted_rows if row["low_confidence"])

    return {
        "methodology": {
            "window_definition": f"non-overlapping consecutive {WINDOW_MONTHS}-month calendar windows, half-open (start, end], anchored at the last observation on/before `start` (same convention as gates21.yearly_annualized_returns)",
            "why_non_overlapping": "overlapping/sliding windows would reuse the same days across many nominally-independent win/loss observations, inflating the effective sample size behind the win-rate statistic",
            "cagr_engine": "fitness.cagr() on a 2-point [anchor_value, window_end_value] series",
            "low_confidence_flag": f"a window is flagged low_confidence when either series has fewer than {MIN_OBS_FOR_CONFIDENCE} daily observations in it (annualizing a short realized return amplifies noise)",
        },
        "windows": rows,
        "n_windows_total": len(rows),
        "n_windows_counted": n_counted,
        "n_windows_low_confidence": n_low_confidence,
        "g1_win_rate": win_rate,
        "g1_win_rate_pct": (win_rate * 100.0) if win_rate is not None else None,
        "n_g1_wins": n_wins,
        "win_rate_pre_oos_windows": win_rate_pre_oos,
        "win_rate_oos_touching_windows": win_rate_oos_touching,
        "n_pre_oos_windows": len(pre_oos_rows),
        "n_oos_touching_windows": len(oos_touching_rows),
        "win_rate_first_half_chronological": win_rate_first_half,
        "win_rate_second_half_chronological": win_rate_second_half,
        "streaks": streaks,
        "limitations": [
            f"only {n_counted} independent (non-overlapping) windows exist over the whole 2019-09~2026-07 history -- a win-rate statistic from this few samples has a wide confidence interval (e.g. a true 50% win rate could easily produce 9/14 or 5/14 by chance alone)",
            f"{n_low_confidence} window(s) are flagged low_confidence (fewer than {MIN_OBS_FOR_CONFIDENCE} obs, typically the final partial window)",
            "only 1-2 windows touch the OOS period (2025-10~) at all, since OOS itself is <1 year old -- the OOS-touching win rate is not independently informative beyond what validation #3's regime split already shows",
        ],
    }


__all__ = ["MIN_OBS_FOR_CONFIDENCE", "WINDOW_MONTHS", "run", "six_month_windows"]
