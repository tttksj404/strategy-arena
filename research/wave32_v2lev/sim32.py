# Wave-32: V2 ("꼬리 사냥", funding-extreme directional LONG) under LEVERAGE -- the one axis
# wave-31 flagged as unmeasured. engine20.py / configs20.py are FROZEN; this module imports their
# helpers and rebuilds run_v2's own series, it does not touch them.
#
# TWO POSITION MODELS, and the distinction is the whole point (same trap sim30.py documents):
#
#   "rebalanced" -- equity *= (1 + L*bar_return) every bar. Exposure is reset to L x CURRENT
#                   equity each bar. This is literally what engine20.run_v2 does at L=1, so it is
#                   used ONLY to prove this harness reproduces run_v2 exactly (Gate 0). It cannot
#                   liquidate on a slow grind, which makes it useless for a wipe-risk question.
#   "fixed"      -- equity = entry_equity * (1 + L*(P/P0 - 1) - L*F_cum). Coins held are fixed at
#                   entry, which is what an isolated-margin perp actually is, and the only model
#                   where the liquidation price sits exactly 1/L - maintenance from entry. Every
#                   swept cell uses this.
#
# On "V2 is long-only so the two coincide at L=1": TRUE for the price leg only. prod(1+r_i)
# telescopes to P_exit/P_entry exactly, so with zero funding the models are identical at L=1.
# engine20 mixes funding into the bar return MULTIPLICATIVELY (*= 1 + r - f) while the fixed model
# charges it as a LINEAR drag on notional (- L*F_cum). Those differ by O(F * r) whenever funding is
# nonzero -- and V2 enters precisely when funding is extreme, so the divergence is not hypothetical.
# Gate 0 therefore runs against "rebalanced"; the L=1 "fixed" row is reported separately as the
# like-for-like sweep baseline, and gate0_report() measures the L=1 gap between the two instead of
# asserting it away.

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.wave1.fam_funding import FundingCandidate, carry_position, funding_score
from research.wave13_liquidity import costs_measured
from research.wave20_convex import dataio20
from research.wave20_convex.configs20 import GAMBLE_CAPITAL, V2_CONFIG, WAVE1_CACHE_DIR
from research.wave20_convex.engine20 import one_leg_cost_rate_frame, worst_case_cost

MAINT_MARGIN = 0.005  # Bitget USDT-M maintenance margin
CAPITAL = 100.0       # user's standing principle: full $100 deployed, dollars per entry


@dataclass(frozen=True, slots=True)
class T32:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_equity: float
    pnl: float
    roi: float        # pnl / entry_equity -- "한 번 들어갈 때마다의 ROI"
    reason: str       # rotated | end_of_data | liquidated


@dataclass(frozen=True, slots=True)
class R32:
    leverage: float
    model: str
    equity: pd.Series
    trades: tuple[T32, ...]
    final: float
    liquidations: int


def v2_inputs(stress_multiplier: float = 1.0) -> dict:
    """Exactly the frames run_v2 builds -- same universe, same resample, same lags, same cost
    mapping. `low` is added (run_v2 never needs it) because leveraged liquidation is an intrabar
    event and must be checked against the bar's low, not its close."""
    cfg = V2_CONFIG
    mapping = costs_measured.fit_mapping()
    symbols = tuple(s for s in dataio20.wave1_symbols_with_funding() if s not in cfg.excluded_symbols)
    candidate = FundingCandidate("V2_squeeze", cfg.funding_window_days, cfg.entry_threshold_apr, cfg.top_k)

    closes, opens, lows, funding_daily, raw_scores, active, quote_volumes = {}, {}, {}, {}, {}, {}, {}
    used: list[str] = []
    for symbol in symbols:
        daily = dataio20.try_load_daily(symbol, WAVE1_CACHE_DIR)
        if daily is None:
            continue
        try:
            funding = dataio20.load_funding_rate(symbol, WAVE1_CACHE_DIR)
        except dataio20.DataError:
            continue
        score = funding_score(funding, cfg.funding_window_days).resample("1D").last()
        closes[symbol] = daily["close"]
        opens[symbol] = daily["open"]
        lows[symbol] = daily["low"]
        quote_volumes[symbol] = daily["quote_volume"]
        funding_daily[symbol] = funding.resample("1D").sum()
        raw_scores[symbol] = score
        active[symbol] = carry_position(score, candidate)
        used.append(symbol)
    if not used:
        raise RuntimeError("V2: no symbols with both price and funding cache data")

    close_frame = pd.DataFrame(closes).sort_index()
    return {
        "close": close_frame,
        "open": pd.DataFrame(opens).reindex(close_frame.index),
        "low": pd.DataFrame(lows).reindex(close_frame.index),
        "funding": pd.DataFrame(funding_daily).reindex(close_frame.index).fillna(0.0),
        "active": pd.DataFrame(active).reindex(close_frame.index).fillna(0.0),
        "score": pd.DataFrame(raw_scores).reindex(close_frame.index).shift(1),
        "cost": one_leg_cost_rate_frame(pd.DataFrame(quote_volumes).reindex(close_frame.index), mapping, stress_multiplier),
        "worst_cost": worst_case_cost(mapping, stress_multiplier),
        "top_k": cfg.top_k,
        "symbols": tuple(used),
    }


def simulate(inp: dict, leverage: float = 1.0, model: str = "fixed",
             starting_equity: float = CAPITAL) -> R32:
    """run_v2's rotation logic, bar for bar, with a leverage factor and (for `fixed`) an intrabar
    liquidation check. Bar order is run_v2's own: gap -> rotation decision/fill at open ->
    intraday + funding."""
    close_f, open_f, low_f = inp["close"], inp["open"], inp["low"]
    funding_f, active_f, score_f, cost_f = inp["funding"], inp["active"], inp["score"], inp["cost"]
    worst, top_k = inp["worst_cost"], inp["top_k"]
    linear = model == "fixed"
    if model not in {"fixed", "rebalanced"}:
        raise ValueError(f"unknown model: {model}")

    index = close_f.index
    prior_close_f = close_f.shift(1)

    equity = starting_equity
    held: str | None = None
    entry_price = float("nan")
    entry_time: pd.Timestamp | None = None
    entry_equity = starting_equity
    cum_funding = 0.0          # fixed model: funding accrued on the CURRENT position, in rate terms
    trades: list[T32] = []
    equity_values: list[float] = []
    liquidations = 0
    dead = False

    # Fixed-notional liquidation distance. maintenance margin eats into the usable buffer.
    liq_dist0 = max(0.0, 1.0 / leverage - MAINT_MARGIN)

    def mark(price: float) -> float:
        """fixed model: equity at price `price` for the open position."""
        return entry_equity * (1.0 + leverage * (price / entry_price - 1.0) - leverage * cum_funding)

    def close_trade(exit_time, exit_price: float, new_equity: float, reason: str) -> None:
        nonlocal trades
        pnl = new_equity - entry_equity
        roi = max(pnl / entry_equity, -1.0) if entry_equity > 0 else 0.0
        trades.append(T32(held, entry_time, exit_time, entry_equity, pnl, roi, reason))

    for ts in index:
        if dead:
            equity_values.append(0.0)
            continue

        if held is not None:
            today_open = open_f[held].loc[ts]
            today_low = low_f[held].loc[ts]

            if linear:
                # Intrabar liquidation FIRST: the bar's low is reachable before its close, and on a
                # gap-down open the position is already gone before any rotation decision.
                liq_price = entry_price * (1.0 - max(0.0, liq_dist0 - cum_funding))
                if pd.notna(today_low) and float(today_low) <= liq_price:
                    close_trade(ts, liq_price, 0.0, "liquidated")
                    liquidations += 1
                    equity, dead, held = 0.0, True, None
                    equity_values.append(0.0)
                    continue
                if pd.notna(today_open):
                    equity = mark(float(today_open))
            else:
                prior_close = prior_close_f[held].loc[ts]
                if pd.notna(prior_close) and float(prior_close) > 0.0 and pd.notna(today_open):
                    gap_ret = float(today_open) / float(prior_close) - 1.0
                    equity *= 1.0 + leverage * gap_ret

            if equity <= 0.0:
                exit_price = float(today_open) if pd.notna(today_open) else float(entry_price)
                close_trade(ts, exit_price, 0.0, "liquidated")
                liquidations += 1
                equity, dead, held = 0.0, True, None

        if not dead:
            eligible_row = active_f.loc[ts]
            eligible = eligible_row[eligible_row > 0.0].index
            available = close_f.loc[ts].notna() & open_f.loc[ts].notna()
            eligible = eligible.intersection(available[available].index)
            ranked = score_f.loc[ts, eligible].dropna().nlargest(top_k).index
            new_symbol = str(ranked[0]) if len(ranked) > 0 else None

            if new_symbol != held:
                if held is not None:
                    rate = float(cost_f[held].loc[ts]) if pd.notna(cost_f[held].loc[ts]) else worst
                    fill = float(open_f[held].loc[ts])
                    # Cost on notional = L x equity (L=1 reduces to engine20's equity *= 1 - rate).
                    equity *= 1.0 - leverage * rate
                    close_trade(ts, fill, equity, "rotated")
                if new_symbol is not None:
                    rate_new = float(cost_f[new_symbol].loc[ts]) if pd.notna(cost_f[new_symbol].loc[ts]) else worst
                    equity *= 1.0 - leverage * rate_new
                    entry_price = float(open_f[new_symbol].loc[ts])
                    entry_time = ts
                    entry_equity = equity
                    cum_funding = 0.0
                held = new_symbol

        if held is not None and not dead:
            o = open_f[held].loc[ts]
            c = close_f[held].loc[ts]
            f_today = float(funding_f[held].loc[ts])
            lo = low_f[held].loc[ts]

            if linear:
                cum_funding += f_today
                liq_price = entry_price * (1.0 - max(0.0, liq_dist0 - cum_funding))
                # Entry-bar / holding-bar intrabar check. Daily bars give no intrabar ordering, so
                # the whole bar's low is treated as reachable while the position is open.
                if pd.notna(lo) and float(lo) <= liq_price:
                    close_trade(ts, liq_price, 0.0, "liquidated")
                    liquidations += 1
                    equity, dead, held = 0.0, True, None
                elif pd.notna(c):
                    equity = mark(float(c))
            else:
                if pd.notna(o) and float(o) > 0.0 and pd.notna(c):
                    intraday = float(c) / float(o) - 1.0
                    equity *= 1.0 + leverage * (intraday - f_today)

            if held is not None and equity <= 0.0:
                close_trade(ts, float(c) if pd.notna(c) else entry_price, 0.0, "liquidated")
                liquidations += 1
                equity, dead, held = 0.0, True, None

        equity_values.append(max(equity, 0.0) if dead else equity)

    if held is not None and not dead:
        final_ts = index[-1]
        rate = float(cost_f[held].loc[final_ts]) if pd.notna(cost_f[held].loc[final_ts]) else worst
        equity *= 1.0 - leverage * rate
        close_trade(final_ts, float(close_f[held].loc[final_ts]), equity, "end_of_data")
        equity_values[-1] = equity

    return R32(leverage, model, pd.Series(equity_values, index=index, dtype=float),
               tuple(trades), float(equity), liquidations)


# ---------------------------------------------------------------------------
# Gate 0 -- fidelity. Runs BEFORE any leverage number is allowed to exist.
# ---------------------------------------------------------------------------


def gate0_report(inp: dict | None = None) -> dict:
    from research.wave20_convex import engine20

    inp = v2_inputs() if inp is None else inp
    ref = engine20.run_v2()
    mine = simulate(inp, leverage=1.0, model="rebalanced", starting_equity=GAMBLE_CAPITAL)

    same_len = len(ref.equity) == len(mine.equity)
    eq_err = float(np.nanmax(np.abs(ref.equity.to_numpy(float) - mine.equity.to_numpy(float)))) if same_len else float("nan")
    final_err = abs(float(ref.equity.iloc[-1]) - mine.final)
    trade_diff = len(ref.trades) - len(mine.trades)
    roi_err = float("nan")
    if trade_diff == 0 and len(mine.trades):
        roi_err = float(np.max(np.abs(
            np.asarray([t.pnl_fraction for t in ref.trades]) - np.asarray([t.roi for t in mine.trades]))))

    passed = bool(same_len and trade_diff == 0 and final_err < 1e-6 and eq_err < 1e-6)

    # Scale invariance: everything is a rate, so $100 must be exactly 4x the $25 sleeve.
    at100 = simulate(inp, leverage=1.0, model="rebalanced", starting_equity=CAPITAL)
    scale_err = abs(at100.final / CAPITAL - mine.final / GAMBLE_CAPITAL)

    # The claim worth verifying, not assuming: how far apart are the two models at L=1?
    fixed1 = simulate(inp, leverage=1.0, model="fixed", starting_equity=CAPITAL)
    return {
        "pass": passed,
        "ref_trades": len(ref.trades), "sim_trades": len(mine.trades), "trade_diff": trade_diff,
        "ref_final": float(ref.equity.iloc[-1]), "sim_final": mine.final,
        "final_abs_err": final_err, "equity_max_abs_err": eq_err, "trade_roi_max_abs_err": roi_err,
        "scale_invariance_err": scale_err,
        "fixed_L1_final_per_100": fixed1.final,
        "rebalanced_L1_final_per_100": at100.final,
        "fixed_vs_rebalanced_L1_rel_gap": (fixed1.final - at100.final) / at100.final,
        "fixed_L1_trades": len(fixed1.trades),
        "fixed_L1_liquidations": fixed1.liquidations,
    }


def main() -> int:
    r = gate0_report()
    print("=== Gate 0: fidelity (sim32 rebalanced L=1  vs  engine20.run_v2) ===")
    print(f"  trades      ref={r['ref_trades']}  sim={r['sim_trades']}  diff={r['trade_diff']}")
    print(f"  final $     ref={r['ref_final']:.12f}  sim={r['sim_final']:.12f}  |err|={r['final_abs_err']:.3e}")
    print(f"  equity series max |err| = {r['equity_max_abs_err']:.3e}")
    print(f"  per-trade ROI  max |err| = {r['trade_roi_max_abs_err']:.3e}")
    print(f"  scale invariance ($25 vs $100) err = {r['scale_invariance_err']:.3e}")
    print(f"  GATE 0: {'PASS' if r['pass'] else 'FAIL'}")
    print()
    print("=== funding-convention check (the claim, verified not assumed) ===")
    print(f"  L=1 rebalanced final = ${r['rebalanced_L1_final_per_100']:.4f}")
    print(f"  L=1 fixed      final = ${r['fixed_L1_final_per_100']:.4f}"
          f"   (trades={r['fixed_L1_trades']}, liq={r['fixed_L1_liquidations']})")
    print(f"  relative gap = {r['fixed_vs_rebalanced_L1_rel_gap']*100:+.4f}%"
          "   <- price leg telescopes exactly; this residue is the funding convention only")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
