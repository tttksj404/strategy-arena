# Wave-10 funding-regime return breakdown: realized annualized return at the $100 basis,
# split into the historically high-funding years the task asks for (2020, 2021, 2024 --
# these are also the years W2c's own validation.yearly_is_returns shows as its strongest:
# +32.2% / +62.0% / +20.8% at W2c's native 4-pair sizing) versus the current dormant/low
# -funding OOS regime (2025-10-01 onward). All figures come from the *same* realized
# historical equity path each config actually produced -- this is a slice of history, not
# a projection.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd  # noqa: PANDAS_OK

from research.wave10_carry100.engine import ACTIVE_CAPITAL, OOS_SPLIT, Wave10Result

HIGH_FUNDING_YEARS: Final[tuple[int, ...]] = (2020, 2021, 2024)
CURRENT_REGIME_LABEL: Final = "current_low_funding_oos"


@dataclass(frozen=True, slots=True)
class RegimeReturn:
    label: str
    start: str
    end: str
    days: float
    start_usdt: float
    end_usdt: float
    total_return: float
    annualized_return: float
    active_capital_annual_profit_usdt: float  # ACTIVE_CAPITAL x annualized_return
    total100_annual_profit_usdt: float  # same profit, expressed against the full $100 (10% reserve drags it down)


def _slice_return(equity: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None, anchor_equity: pd.Series) -> tuple[float, float, float, float] | None:
    """Returns (start_usdt, end_usdt, days, total_return) for the half-open window
    (start, end] against `anchor_equity` (the full series, used to find the pre-window
    anchor value so a window's return includes the transition day correctly), or None if
    the window has no data."""
    mask = pd.Series(True, index=equity.index)
    if start is not None:
        mask &= equity.index > start
    if end is not None:
        mask &= equity.index <= end
    window = equity[mask]
    if window.empty:
        return None
    if start is not None:
        pre = anchor_equity[anchor_equity.index <= start]
        anchor_value = float(pre.iloc[-1]) if len(pre) else float(window.iloc[0])
    else:
        anchor_value = float(window.iloc[0])
    end_value = float(window.iloc[-1])
    days = max((pd.Timestamp(window.index[-1]) - pd.Timestamp(start if start is not None else window.index[0])).total_seconds() / 86_400.0, 1.0)
    total_return = end_value / anchor_value - 1.0
    return anchor_value, end_value, days, total_return


def _annualize(total_return: float, days: float) -> float:
    growth = 1.0 + total_return
    if growth <= 0.0:
        return -1.0
    return float(growth ** (365.0 / days) - 1.0)


def _regime_return(label: str, equity: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None) -> RegimeReturn | None:
    sliced = _slice_return(equity, start, end, equity)
    if sliced is None:
        return None
    start_usdt, end_usdt, days, total_return = sliced
    annualized = _annualize(total_return, days)
    return RegimeReturn(
        label=label,
        start=(start.isoformat() if start is not None else pd.Timestamp(equity.index[0]).isoformat()),
        end=(end.isoformat() if end is not None else pd.Timestamp(equity.index[-1]).isoformat()),
        days=days,
        start_usdt=start_usdt,
        end_usdt=end_usdt,
        total_return=total_return,
        annualized_return=annualized,
        active_capital_annual_profit_usdt=ACTIVE_CAPITAL * annualized,
        total100_annual_profit_usdt=ACTIVE_CAPITAL * annualized,  # reserve ($10) assumed idle/untouched
    )


def regime_breakdown(result: Wave10Result) -> dict[str, object]:
    equity = result.equity
    by_year: dict[int, RegimeReturn] = {}
    for year in HIGH_FUNDING_YEARS:
        start = pd.Timestamp(f"{year - 1}-12-31T23:59:59Z")
        end = pd.Timestamp(f"{year}-12-31T23:59:59Z")
        regime = _regime_return(str(year), equity, start, end)
        if regime is not None:
            by_year[year] = regime
    current = _regime_return(CURRENT_REGIME_LABEL, equity, OOS_SPLIT, None)
    high_funding_years_available = [by_year[year].annualized_return for year in HIGH_FUNDING_YEARS if year in by_year]
    high_funding_mean_annualized = (
        float(sum(high_funding_years_available) / len(high_funding_years_available)) if high_funding_years_available else None
    )
    return {
        "high_funding_years": {str(year): _asdict(regime) for year, regime in by_year.items()},
        "current_low_funding": _asdict(current) if current is not None else None,
        "high_funding_mean_annualized_return": high_funding_mean_annualized,
        "high_funding_mean_annual_profit_usdt_at_100_basis": (
            ACTIVE_CAPITAL * high_funding_mean_annualized if high_funding_mean_annualized is not None else None
        ),
    }


def _asdict(regime: RegimeReturn) -> dict[str, object]:
    return {
        "label": regime.label,
        "start": regime.start,
        "end": regime.end,
        "days": regime.days,
        "start_usdt": regime.start_usdt,
        "end_usdt": regime.end_usdt,
        "total_return": regime.total_return,
        "annualized_return": regime.annualized_return,
        "active_capital_annual_profit_usdt": regime.active_capital_annual_profit_usdt,
        "total100_annual_profit_usdt": regime.total100_annual_profit_usdt,
    }
