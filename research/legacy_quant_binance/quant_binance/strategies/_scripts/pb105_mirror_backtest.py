"""
pb105_mirror_backtest.py

PB105 — Hyperliquid Top Trader Mirror PoC: Step 4-5 (simulation + 9-point gates).

DATA REALITY (verified 2026-04-28):
  - Hyperliquid /info has NO leaderboard endpoint (HTTP 422 on all variants tested).
  - userFillsByTime caps at 2000 most-recent fills per address.
  - Both probed candidate addresses (whale_a $6.6M, MM $113M) had only 2-17 hours
    of historical fill data available; no 30-day look-back possible.
  - Of 15 candidate addresses, 0 passed the directional filter
    (open_ratio in 0.20-0.80, fph 0.05-60).

SIMULATION DESIGN:
  Given the data gap, this script runs TWO complementary simulations:

  (A) ATTEMPTED REAL MIRROR — use the captured fills from whale_a as entry events,
      compute forward returns using Bitget 1h klines. Will report n_evaluable
      after time-coverage filtering.

  (B) STRUCTURAL MIRROR SIMULATION — sample synthetic entry timestamps over the
      past 30 days using the same coin distribution observed for whale_a, and
      measure forward returns. This is a NULL-ALPHA reference: it tells us how
      a random-mirror strategy would behave. If the real mirror cannot be
      shown to beat this, no alpha is demonstrated.

  Both runs apply: 5-bps slippage, 5-second latency penalty (0.5 * 1m vol),
  $55 capital * 30% size, 5x leverage, fees 4 bps roundtrip.

OUTPUT: claimed_performance.md (9-point gate table + caveats)
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PB_DIR = ROOT / "strategies" / "_playbook" / "PB105_hyperliquid_leaderboard_mirror"
DATA_DIR = PB_DIR / "data"
KLINES_ROOT = ROOT.parent / "quant_runtime" / "historical_top50"

CAPITAL = 55.0
POSITION_PCT = 0.30  # 30% of capital per trade
LEVERAGE = 5.0
FEE_ROUNDTRIP_BPS = 4.0   # 2 bps each side, taker
SLIPPAGE_BPS = 5.0
LATENCY_SEC = 5.0
HOLD_HOURS = 4  # exit window: top trader avg short hold
RNG_SEED = 42


def load_klines(symbol: str) -> list[dict] | None:
    p = KLINES_ROOT / symbol / "1h.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def find_kline_at(klines: list[dict], ts_ms: int) -> int | None:
    """Returns index of the 1h bar containing ts_ms, or None if out of range."""
    if not klines:
        return None
    # bars are 1h apart; use binary search
    lo, hi = 0, len(klines) - 1
    if ts_ms < klines[0]["open_time"] or ts_ms > klines[-1]["open_time"] + 3600_000:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if klines[mid]["open_time"] + 3600_000 <= ts_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo


def simulate_trade(klines: list[dict], idx: int, side: str, hold_h: int) -> dict | None:
    """side: 'L' (long) or 'S' (short). Returns trade pnl in bps net."""
    if idx is None or idx + hold_h >= len(klines):
        return None
    entry_bar = klines[idx]
    exit_bar = klines[idx + hold_h]
    # entry: open of the bar after signal; exit: close of bar+hold_h
    entry_px = entry_bar["close_price"]   # whale fills at intra-bar; approx with close
    exit_px = exit_bar["close_price"]
    raw_ret_bps = (exit_px - entry_px) / entry_px * 10000.0
    if side == "S":
        raw_ret_bps = -raw_ret_bps
    # latency cost: penalty proportional to bar volatility * latency_share
    bar_range_bps = (entry_bar["high_price"] - entry_bar["low_price"]) / entry_px * 10000.0
    latency_cost_bps = bar_range_bps * (LATENCY_SEC / 3600.0) * 0.5  # half range * latency fraction
    cost_bps = FEE_ROUNDTRIP_BPS + SLIPPAGE_BPS + latency_cost_bps
    net_bps = raw_ret_bps - cost_bps
    return {
        "entry_px": entry_px,
        "exit_px": exit_px,
        "raw_bps": raw_ret_bps,
        "cost_bps": cost_bps,
        "net_bps": net_bps,
    }


# ----------------------------------------------------------------------------
# (A) Real mirror — use captured whale fills
# ----------------------------------------------------------------------------

def coin_to_symbol(coin: str) -> str:
    return f"{coin}USDT"


def run_real_mirror(addresses: list[str]) -> dict:
    klines_cache: dict[str, list[dict] | None] = {}
    fills_total = 0
    overlap_total = 0
    evaluable = 0
    trades: list[dict] = []
    for addr in addresses:
        p = DATA_DIR / f"fills_{addr}.json"
        if not p.exists():
            continue
        with open(p) as f:
            fills = json.load(f)
        fills_total += len(fills)
        for fl in fills:
            if not fl["dir"].startswith("Open"):
                continue
            sym = coin_to_symbol(fl["coin"])
            if sym not in klines_cache:
                klines_cache[sym] = load_klines(sym)
            klines = klines_cache[sym]
            if klines is None:
                continue
            overlap_total += 1
            ts = fl["time"] + int(LATENCY_SEC * 1000)
            idx = find_kline_at(klines, ts)
            side = "L" if "Long" in fl["dir"] else "S"
            tr = simulate_trade(klines, idx, side, HOLD_HOURS)
            if tr is None:
                continue
            tr["coin"] = fl["coin"]
            tr["side"] = side
            tr["t"] = fl["time"]
            trades.append(tr)
            evaluable += 1
    return {
        "fills_total": fills_total,
        "overlap_total": overlap_total,
        "evaluable": evaluable,
        "trades": trades,
    }


# ----------------------------------------------------------------------------
# (B) Structural null — random timestamps over past 30 days, same coin mix
# ----------------------------------------------------------------------------

def run_structural_null(coin_dist: dict[str, int], n_samples: int = 200) -> dict:
    rng = random.Random(RNG_SEED)
    klines_cache: dict[str, list[dict] | None] = {}
    coins, counts = zip(*coin_dist.items())
    # weighted choice
    population = []
    for c, n in coin_dist.items():
        population.extend([c] * n)

    trades: list[dict] = []
    for _ in range(n_samples):
        coin = rng.choice(population)
        sym = coin_to_symbol(coin)
        if sym not in klines_cache:
            klines_cache[sym] = load_klines(sym)
        klines = klines_cache[sym]
        if not klines or len(klines) < HOLD_HOURS + 24:
            continue
        # pick random index in last 30*24 bars (or full range if shorter)
        max_idx = len(klines) - HOLD_HOURS - 1
        min_idx = max(0, max_idx - 30 * 24)
        idx = rng.randint(min_idx, max_idx)
        side = rng.choice(["L", "S"])
        tr = simulate_trade(klines, idx, side, HOLD_HOURS)
        if tr is None:
            continue
        tr["coin"] = coin
        tr["side"] = side
        trades.append(tr)
    return {"trades": trades}


# ----------------------------------------------------------------------------
# 9-point gate evaluator
# ----------------------------------------------------------------------------

def evaluate_gates(trades: list[dict], label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0, "gates": {}, "summary": "EMPTY"}
    n = len(trades)
    rets_bps = [t["net_bps"] for t in trades]
    avg = sum(rets_bps) / n
    wins = sum(1 for r in rets_bps if r > 0)
    wr = wins / n
    # portfolio sim (sum of bps applied to position size)
    notional = CAPITAL * POSITION_PCT * LEVERAGE
    pnl_usd = sum((r / 10000.0) * notional for r in rets_bps)
    # cost-sensitivity: re-sim with double cost
    rets_double = [t["net_bps"] - (FEE_ROUNDTRIP_BPS + SLIPPAGE_BPS) for t in trades]
    avg_double = sum(rets_double) / n
    # liquidation rate at 5x: liquidation if raw_bps < -2000 (20% adverse)
    liq_thresh = -10000.0 / LEVERAGE
    liq_count = sum(1 for t in trades if t["raw_bps"] < liq_thresh)
    liq_rate = liq_count / n
    # quartile time-shuffle (poor-man time-stability): split into 4 quarters by index
    q = n // 4
    quarter_avgs = []
    for i in range(4):
        slc = rets_bps[i * q:(i + 1) * q] if i < 3 else rets_bps[i * q:]
        if slc:
            quarter_avgs.append(sum(slc) / len(slc))
    all_q_pos = all(qa > 0 for qa in quarter_avgs) if quarter_avgs else False

    gates = {
        "1_quarter_avg_pos": all_q_pos,
        "2_portfolio_pnl_pos": pnl_usd > 0,
        "3_avg_ge_50bps": avg >= 50.0,
        "4_wr_ge_65pct": wr >= 0.65,
        "5_axis_fit": True,  # 6-axis user fit was 6/6 in source.md (PB105 design)
        "6_cost_sensitivity_ok": avg_double > 0,
        "7_liq_lt_10pct": liq_rate < 0.10,
        "8_n_ge_50": n >= 50,
        "9_no_warmup": True,  # mirror has no warmup phase
    }
    passed = sum(1 for v in gates.values() if v)
    return {
        "label": label,
        "n": n,
        "avg_net_bps": round(avg, 2),
        "wr": round(wr, 3),
        "pnl_usd": round(pnl_usd, 2),
        "avg_double_cost": round(avg_double, 2),
        "liq_rate": round(liq_rate, 3),
        "quarter_avgs": [round(q, 2) for q in quarter_avgs],
        "gates": gates,
        "passed": f"{passed}/9",
    }


def main() -> None:
    addrs_meta = json.load(open(DATA_DIR / "addresses.json"))
    addresses = [a["address"] for a in addrs_meta]
    print(f"Loaded {len(addresses)} address(es) with non-zero equity")

    print("\n[A] Real mirror simulation...")
    real = run_real_mirror(addresses)
    print(f"  fills_total={real['fills_total']}  "
          f"overlap_with_bitget_alts={real['overlap_total']}  "
          f"evaluable_with_klines={real['evaluable']}")
    real_eval = evaluate_gates(real["trades"], "REAL_MIRROR")

    print("\n[B] Structural null simulation (random entries, same coin mix)...")
    # Build coin distribution from all open events on overlap coins
    coin_counts: dict[str, int] = {}
    for addr in addresses:
        p = DATA_DIR / f"fills_{addr}.json"
        if not p.exists():
            continue
        for fl in json.load(open(p)):
            if not fl["dir"].startswith("Open"):
                continue
            sym = coin_to_symbol(fl["coin"])
            if (KLINES_ROOT / sym / "1h.json").exists():
                coin_counts[fl["coin"]] = coin_counts.get(fl["coin"], 0) + 1
    print(f"  overlap coin distribution: {coin_counts}")
    null_res = run_structural_null(coin_counts, n_samples=200)
    null_eval = evaluate_gates(null_res["trades"], "STRUCTURAL_NULL")

    # alpha test: does real mirror beat null?
    alpha_bps = real_eval.get("avg_net_bps", 0) - null_eval.get("avg_net_bps", 0)

    summary = {
        "real_mirror": real_eval,
        "structural_null": null_eval,
        "alpha_vs_null_bps": round(alpha_bps, 2),
        "config": {
            "capital_usd": CAPITAL,
            "position_pct": POSITION_PCT,
            "leverage": LEVERAGE,
            "fee_roundtrip_bps": FEE_ROUNDTRIP_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "latency_sec": LATENCY_SEC,
            "hold_hours": HOLD_HOURS,
            "rng_seed": RNG_SEED,
        },
    }
    out = DATA_DIR / "backtest_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")
    print(f"\n=== REAL MIRROR ===")
    print(json.dumps(real_eval, indent=2))
    print(f"\n=== STRUCTURAL NULL ===")
    print(json.dumps(null_eval, indent=2))
    print(f"\nAlpha vs null: {alpha_bps:+.2f} bps")


if __name__ == "__main__":
    main()
