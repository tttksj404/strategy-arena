# Wave-35 calibration (SPEC.md section 4, step 2). The broad universe has DAILY bars only, while
# V1's published record is 1H-executed. Before any universe comparison is allowed, measure the
# execution-granularity gap on the ONE symbol where both grids exist: BTCUSDT.
#
# Reference (research/wave30_riskcap/REPORT.md): 1H execution, "fixed" position model, L=1, no
# stop, compounding -> 8.49x (=$212.20 on the $25 sleeve).
# This script runs the same accounting on daily bars and prints the gap. It also prints the
# fixed-notional BTC control ($100 base, no compounding) that the N-sweep actually uses.

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.wave13_liquidity import costs_measured
from research.wave20_convex import dataio20
from research.wave20_convex.configs20 import WAVE3_CACHE_DIR, WAVE6_CACHE_DIR
from research.wave35_breadth.sim35 import simulate_symbol, v1_daily_inputs

HOURLY_REFERENCE_MULTIPLE = 8.49   # wave30_riskcap REPORT.md, L=1 fixed, $25 -> $212.20
RESULTS = Path(__file__).resolve().parent / "results"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    mapping = costs_measured.fit_mapping()

    out: dict = {"reference_hourly_multiple": HOURLY_REFERENCE_MULTIPLE}

    # (a) daily bars resampled from the SAME 1H cache -> isolates granularity, not data source
    hourly = dataio20.load_hourly("BTCUSDT", WAVE6_CACHE_DIR)
    daily_from_hourly = dataio20.resample_hourly_to_daily(hourly)
    inp_a = v1_daily_inputs(daily_from_hourly, mapping)
    run_a = simulate_symbol("BTCUSDT", inp_a, leverage=1.0, compound=True, starting_equity=25.0)
    out["daily_from_1h_cache"] = {
        "multiple": run_a.final_equity / 25.0,
        "final_on_25": run_a.final_equity,
        "n_trades": len(run_a.trades),
        "first": str(inp_a["index"][0]), "last": str(inp_a["index"][-1]), "bars": len(inp_a["index"]),
    }

    # (b) the wave3 daily cache -- the actual data source the universe sweep uses
    daily_cache = dataio20.load_daily("BTCUSDT", WAVE3_CACHE_DIR)
    inp_b = v1_daily_inputs(daily_cache, mapping)
    run_b = simulate_symbol("BTCUSDT", inp_b, leverage=1.0, compound=True, starting_equity=25.0)
    out["daily_wave3_cache"] = {
        "multiple": run_b.final_equity / 25.0,
        "final_on_25": run_b.final_equity,
        "n_trades": len(run_b.trades),
        "first": str(inp_b["index"][0]), "last": str(inp_b["index"][-1]), "bars": len(inp_b["index"]),
    }

    # (c) the fixed-notional control the N-sweep uses ($100 base, no compounding)
    run_c = simulate_symbol("BTCUSDT", inp_b, leverage=1.0, compound=False, starting_equity=1.0)
    pnl = sum(t.roi for t in run_c.trades) * 100.0
    out["btc_fixed_notional_control"] = {
        "final_on_100": 100.0 + pnl,
        "n_trades": len(run_c.trades),
        "trades_per_day": len(run_c.trades) / len(inp_b["index"]),
        "mean_per_entry": pnl / len(run_c.trades) if run_c.trades else None,
    }

    ratio = out["daily_wave3_cache"]["multiple"] / HOURLY_REFERENCE_MULTIPLE
    out["gap_daily_vs_hourly"] = {
        "daily_multiple": out["daily_wave3_cache"]["multiple"],
        "hourly_multiple": HOURLY_REFERENCE_MULTIPLE,
        "ratio_daily_over_hourly": ratio,
    }
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
