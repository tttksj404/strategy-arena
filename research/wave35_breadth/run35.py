# Wave-35 runner: the frozen N-sweep from SPEC.md section 6.
# Checkpoints to results/ after every cell so an interruption loses nothing.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.wave13_liquidity import costs_measured
from research.wave20_convex.configs20 import IS_OOS_SPLIT
from research.wave35_breadth import portfolio35 as pf
from research.wave35_breadth.sim35 import load_universe, simulate_symbol, v1_daily_inputs

RESULTS = Path(__file__).resolve().parent / "results"
N_GRID: list[int | None] = [1, 3, 5, 10, 17, 25, 40, None]   # None = all-eligible
LEVERAGES = [1.0, 2.0]
IS_END = pd.Timestamp(IS_OOS_SPLIT)


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mapping = costs_measured.fit_mapping()

    frames = load_universe()
    print(f"universe: {len(frames)} symbols with >= 405 daily bars", flush=True)

    inputs = {s: v1_daily_inputs(f, mapping) for s, f in frames.items()}
    runs_by_lev: dict[float, dict] = {}
    for lev in LEVERAGES:
        runs = {}
        for k, (sym, inp) in enumerate(inputs.items()):
            runs[sym] = simulate_symbol(sym, inp, leverage=lev, compound=False, starting_equity=1.0)
            if (k + 1) % 100 == 0:
                print(f"  lev {lev:.0f}x simulated {k+1}/{len(inputs)}  ({time.time()-t0:.0f}s)", flush=True)
        runs_by_lev[lev] = runs
        print(f"lev {lev:.0f}x done: {sum(len(r.trades) for r in runs.values())} raw symbol-trades "
              f"({time.time()-t0:.0f}s)", flush=True)

    calendar = pf.build_calendar(runs_by_lev[1.0])
    elig, vol = pf.eligibility_frames(runs_by_lev[1.0], calendar)

    # --- SPEC.md R6: eligible-symbol count over time -------------------------------------------
    counts = elig.sum(axis=1)
    by_year = counts.groupby(counts.index.year).agg(["min", "median", "max"])
    elig_report = {str(y): {k: int(v) for k, v in row.items()} for y, row in by_year.iterrows()}
    (RESULTS / "eligibility.json").write_text(
        json.dumps({"by_year": elig_report,
                    "first_date_with_any": str(counts[counts > 0].index[0]) if (counts > 0).any() else None,
                    "n_symbols_ever_eligible": int((elig.any(axis=0)).sum()),
                    "n_symbols_loaded": int(elig.shape[1])}, indent=2), encoding="utf-8")
    print("eligible symbols by year:\n", by_year.to_string(), flush=True)

    # --- selection frames (shared across leverages; selection depends only on eligibility+volume)
    sel_cache: dict[str, pd.DataFrame] = {}
    for n in N_GRID:
        key = "all" if n is None else str(n)
        if n == 1:
            # SPEC.md: N=1 is the BTC control, not "the single most liquid symbol"
            s = pd.DataFrame(False, index=elig.index, columns=elig.columns)
            if "BTCUSDT" in s.columns:
                s["BTCUSDT"] = elig["BTCUSDT"]
            sel_cache[key] = s
        else:
            sel_cache[key] = pf.selection_frame(elig, vol, n)
        print(f"selection N={key}: mean concurrent slots "
              f"{sel_cache[key].sum(axis=1).mean():.1f}", flush=True)

    rows: list[dict] = []
    detail: dict[str, dict] = {}

    def record(run, tag: str) -> None:
        m_full = pf.metrics(run)
        m_is = pf.metrics(run, end=IS_END)
        rows.append({**m_full, "tag": tag, "window": "FULL"})
        rows.append({**m_is, "tag": tag, "window": "IS"})
        detail[tag] = {
            "full": m_full, "is": m_is,
            "symbols_traded": list(run.symbols_traded),
            "equity_last": float(run.equity.iloc[-1]),
        }
        (RESULTS / "sweep.json").write_text(json.dumps(rows, indent=2, default=_jsonable), encoding="utf-8")
        (RESULTS / "detail.json").write_text(json.dumps(detail, indent=2, default=_jsonable), encoding="utf-8")
        print(f"[{tag}] IS: final ${m_is['final_on_100']:.2f} | {m_is['n_trades']} tr | "
              f"{m_is['trades_per_active_day']:.3f}/day | mean ${m_is['mean_per_entry']:.3f} | "
              f"MDD {m_is['mdd']*100:.1f}% | ruin {m_is['ruin_prob']*100:.1f}%", flush=True)

    for lev in LEVERAGES:
        runs = runs_by_lev[lev]
        for n in N_GRID:
            key = "all" if n is None else str(n)
            sel = sel_cache[key]
            slots = n if n is not None else int(max(1, elig.sum(axis=1).max()))
            run = pf.assemble(runs, calendar, sel, n_slots=slots, leverage=lev,
                              label=f"N{key}_L{lev:.0f}",
                              max_concurrent=None if n is not None else None)
            record(run, f"N{key}_L{lev:.0f}")

        # rotation arm: one position at a time, $100 base, first eligible signal wins
        rot = pf.assemble(runs, calendar, sel_cache["all"], n_slots=1, leverage=lev,
                          label=f"ROT_L{lev:.0f}", single_sleeve=True)
        record(rot, f"ROT_L{lev:.0f}")

    pd.DataFrame(rows).to_csv(RESULTS / "sweep.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s -> {RESULTS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
