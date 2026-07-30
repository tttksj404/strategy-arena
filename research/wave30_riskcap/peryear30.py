# Wave-30 part 3: per-calendar-year behaviour of a CONSTANT leverage.
#
# Part 2 tests a rule that CHOOSES leverage from recent data. This tests the opposite: a single
# number fixed for all time, never re-fitted. That distinction matters for overfitting -- a
# constant carries almost no selection burden, so if 2x beats 1x in most years and never wipes,
# it is a real property of the payoff shape rather than a fitted parameter.
#
# Reported per year: multiple, worst single trade, and how many entries returned >= +30%
# (the user's own benchmark: "한번 들어갈 때마다 100달러 기준 ROI 30%").

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave30_riskcap.sim30 import simulate, v1_inputs

OUT = Path(__file__).resolve().parent / "results"
SLEEVE = 25.0
CELLS: list[tuple[float, float | None]] = [(1.0, None), (2.0, None), (2.0, 0.15), (3.0, 0.10)]


def main() -> int:
    inp = v1_inputs()
    idx = inp["index"]
    years = sorted({int(t.year) for t in idx})
    table: dict[str, dict] = {}

    for lev, stop in CELLS:
        label = f"{lev:.0f}x/{'없음' if stop is None else f'{stop*100:.0f}%'}"
        equity, per_year = SLEEVE, []
        for y in years:
            start = pd.Timestamp(f"{y}-01-01", tz="UTC")
            end = pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
            if start > idx[-1]:
                continue
            res = simulate(inp, leverage=lev, stop=stop, model="fixed",
                           start_at=start - pd.Timedelta(seconds=1), end_at=end, starting_equity=equity)
            rois = np.asarray([t.roi for t in res.trades], dtype=float)
            step = res.final / max(equity, 1e-9)
            equity = res.final
            per_year.append({
                "year": y, "multiple": float(step), "equity": float(equity),
                "trades": len(rois), "n_ge_30pct": int((rois >= 0.30).sum()) if len(rois) else 0,
                "worst": float(rois.min()) if len(rois) else 0.0,
                "liquidations": sum(1 for t in res.trades if t.reason == "liquidated"),
            })
        table[label] = {
            "per_year": per_year, "final": equity, "multiple": equity / SLEEVE,
            "losing_years": sum(1 for p in per_year if p["multiple"] < 1.0),
            "worst_year": min((p["multiple"] for p in per_year), default=0.0),
            "total_ge_30pct": sum(p["n_ge_30pct"] for p in per_year),
            "worst_trade": min((p["worst"] for p in per_year), default=0.0),
        }

    labels = list(table)
    print(f"{'연도':>6} " + " ".join(f"{l:>13}" for l in labels), flush=True)
    for i, y in enumerate(years):
        cells = []
        for l in labels:
            py = table[l]["per_year"]
            cells.append(f"{py[i]['multiple']:6.2f}x({py[i]['n_ge_30pct']:2d})" if i < len(py) else " " * 13)
        print(f"{y:>6} " + " ".join(f"{c:>13}" for c in cells), flush=True)
    print(f"\n{'최종':>6} " + " ".join(f"{table[l]['multiple']:11.2f}x " for l in labels))
    print(f"{'손실년':>6} " + " ".join(f"{table[l]['losing_years']:>12}개 " for l in labels))
    print(f"{'최악년':>6} " + " ".join(f"{table[l]['worst_year']:11.2f}x " for l in labels))
    print(f"{'+30%':>6} " + " ".join(f"{table[l]['total_ge_30pct']:>12}건 " for l in labels))
    print(f"{'최악1회':>6} " + " ".join(f"{table[l]['worst_trade']*100:11.1f}% " for l in labels))
    print("\n괄호 = 그 해에 1회 진입으로 +30% 이상 낸 횟수")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "peryear.json").write_text(json.dumps(table, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
