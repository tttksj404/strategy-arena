# Wave-35 OOS unseal. Run ONCE, on the single configuration selected on IS (SPEC.md section 8).
#
# Selection rule, stated before running this file:
#   - B1 (>= 1.0 trades/active day) is passed ONLY by the all-eligible arm, and that arm's mean
#     $/entry is $0.015 -- three orders of magnitude below the user's $10 target, so deploying it
#     is pointless regardless of what OOS says.
#   - The configuration this wave would actually recommend is therefore the best IS cell that
#     clears B4 (concentration) and B5 (ruin): N=17, L=2x.
#   -> OOS is opened on N17_L2 only. The BTC-only control (N1_L2) is also evaluated OOS because a
#      concentration/tail comparison without its baseline is meaningless; it is a BASELINE, not a
#      second candidate, and it is not eligible for selection.

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.wave13_liquidity import costs_measured
from research.wave20_convex.configs20 import IS_OOS_SPLIT
from research.wave35_breadth import portfolio35 as pf
from research.wave35_breadth.sim35 import load_universe, simulate_symbol, v1_daily_inputs

RESULTS = Path(__file__).resolve().parent / "results"
IS_END = pd.Timestamp(IS_OOS_SPLIT)
SELECTED = ("17", 17, 2.0)


def main() -> int:
    mapping = costs_measured.fit_mapping()
    frames = load_universe()
    inputs = {s: v1_daily_inputs(f, mapping) for s, f in frames.items()}
    runs = {s: simulate_symbol(s, i, leverage=2.0, compound=False, starting_equity=1.0)
            for s, i in inputs.items()}
    calendar = pf.build_calendar(runs)
    elig, vol = pf.eligibility_frames(runs, calendar)

    out = {}
    for tag, n in (("N17_L2", 17), ("N1_L2_control", 1)):
        if n == 1:
            sel = pd.DataFrame(False, index=elig.index, columns=elig.columns)
            sel["BTCUSDT"] = elig["BTCUSDT"]
        else:
            sel = pf.selection_frame(elig, vol, n)
        run = pf.assemble(runs, calendar, sel, n_slots=n, leverage=2.0, label=tag)
        out[tag] = {
            "IS": pf.metrics(run, end=IS_END),
            "OOS": pf.metrics(run, start=IS_END + pd.Timedelta(days=1)),
        }
        for w in ("IS", "OOS"):
            m = out[tag][w]
            print(f"[{tag} {w}] final ${m['final_on_100']:.2f} | {m['n_trades']} tr | "
                  f"{m['trades_per_active_day']:.3f}/day | mean ${m['mean_per_entry']:.4f} | "
                  f"P(>=+$10) {m['p_ge_10']*100:.2f}% | P(<=-$10) {m['p_le_m10']*100:.2f}% | "
                  f"MDD {m['mdd']*100:.1f}% | ruin {m['ruin_prob']*100:.1f}% | "
                  f"ex-top5 ${m['total_ex_top5']:.2f}", flush=True)

    (RESULTS / "oos.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
