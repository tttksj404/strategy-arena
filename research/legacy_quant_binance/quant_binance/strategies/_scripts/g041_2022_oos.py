"""
G041 — 2022-2023 out-of-sample 검증.
2025-2026 (in-sample) 에서 검증된 룰을 BTC bear 시기 ($69k→$15k) 에 적용.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT

DATA_DIR = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2022"
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
            "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT",
            "MATICUSDT", "NEARUSDT", "UNIUSDT", "XRPUSDT"]


def load(sym):
    p = DATA_DIR / sym / "1h.json"
    if not p.exists(): return None
    data = json.loads(p.read_text())
    df = pd.DataFrame(data)
    return df


def gather_entries(threshold=70, hold=72):
    """모든 candidate 진입 (gate 적용 전)."""
    entries = []
    for sym in UNIVERSE:
        df = load(sym)
        if df is None or len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        df["score"] = score
        df["fwd_pct"] = (df["close_price"].shift(-hold) / df["close_price"] - 1) * 10000
        e = df[(df["score"] >= threshold) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym
        e["gross_bps"] = e["fwd_pct"]
        entries.append(e[["open_time", "score", "gross_bps", "sym"]])
    if not entries: return pd.DataFrame()
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True)


def adaptive(all_e, lookback_days, cost_bps):
    """walk-forward gate (recursive: 실제 taken trades 만 history)."""
    LB = lookback_days * 24 * 3600 * 1000
    history = []
    taken = []
    if len(all_e) == 0: return None
    first = all_e["open_time"].min()
    for _, row in all_e.iterrows():
        ts = row["open_time"]
        net = row["gross_bps"] - cost_bps
        if ts - first < LB:
            taken.append((ts, net, row["sym"])); history.append((ts, net))
            continue
        recent = [n for t, n in history if ts - LB <= t < ts]
        if sum(recent) > 0:
            taken.append((ts, net, row["sym"])); history.append((ts, net))
    return taken


def stats(taken_list, label):
    if not taken_list:
        print(f"  {label}: no entries")
        return None
    arr = np.array([t[1] for t in taken_list])
    return {
        "label": label,
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "wr": float((arr > 0).mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def quarterly(taken_list, n_periods=8):
    """2년 데이터 → 8 분기 (3개월씩)."""
    if not taken_list: return []
    first = taken_list[0][0]
    last = taken_list[-1][0]
    span = last - first
    period_size = span / n_periods
    qs = [{"n":0, "net":0.0, "wins":0} for _ in range(n_periods)]
    for ts, net, sym in taken_list:
        q = min(int((ts - first) / period_size), n_periods-1)
        qs[q]["n"] += 1
        qs[q]["net"] += net
        if net > 0: qs[q]["wins"] += 1
    return qs


def main():
    print("=== G041 2022-2023 Out-of-Sample 검증 ===")
    print(f"Universe: {len(UNIVERSE)} symbols (2022 시점에 존재)")
    print(f"Period: 2022-01-01 ~ 2024-01-01 (730일, BTC bear 1년 + 회복 1년)\n")

    all_e = gather_entries(threshold=70, hold=72)
    if len(all_e) == 0:
        print("ERROR: no entries gathered")
        return
    print(f"Total candidates (CH1 ≥70): {len(all_e)}\n")

    # 1. G003 baseline (no gate)
    g003_taken = [(r["open_time"], r["gross_bps"] - 16, r["sym"]) for _, r in all_e.iterrows()]
    g003_stats = stats(g003_taken, "G003 baseline (no gate)")

    # 2. G041 adaptive (30d lookback)
    g041_taken = adaptive(all_e, lookback_days=30, cost_bps=16)
    g041_stats = stats(g041_taken, "G041 adaptive (30d gate)")

    print(f"{'variant':<28} {'n':>6} {'mean':>8} {'std':>8} {'WR':>7} {'P10':>7} {'median':>8} {'P90':>8}")
    for s in (g003_stats, g041_stats):
        if s is None: continue
        print(f"{s['label']:<28} {s['n']:>6} {s['mean']:>+8.1f} {s['std']:>8.0f} {s['wr']*100:>6.1f}% {s['p10']:>+7.0f} {s['median']:>+8.0f} {s['p90']:>+8.0f}")

    # 3. quarterly (8 분기 = 3개월씩)
    print(f"\n=== Quarterly (3개월 × 8 분기) ===")
    q3 = quarterly(g003_taken, 8)
    q4 = quarterly(g041_taken, 8)
    print(f"{'Q':>3} {'G003 n':>7} {'G003 net':>9} {'G003 WR':>8}    {'G041 n':>7} {'G041 net':>9} {'G041 WR':>8}")
    for i in range(8):
        a, b = q3[i], q4[i]
        an = f"{a['net']/a['n']:+.0f}" if a['n']>0 else "--"
        bn = f"{b['net']/b['n']:+.0f}" if b['n']>0 else "--"
        awr = f"{a['wins']/a['n']*100:.0f}%" if a['n']>0 else "--"
        bwr = f"{b['wins']/b['n']*100:.0f}%" if b['n']>0 else "--"
        print(f"Q{i+1:>2} {a['n']:>7} {an:>9} {awr:>8}    {b['n']:>7} {bn:>9} {bwr:>8}")

    # 4. lookback sweep on 2022 data
    print(f"\n=== Lookback robustness (2022-2023) ===")
    print(f"{'lookback':>9} {'n':>6} {'mean':>8} {'WR':>7} {'std':>7}")
    for lb in [7, 14, 30, 60, 90]:
        t = adaptive(all_e, lookback_days=lb, cost_bps=16)
        s = stats(t, f"{lb}d")
        if s:
            print(f"{lb:>5}d {s['n']:>6} {s['mean']:>+8.1f} {s['wr']*100:>6.1f}% {s['std']:>7.0f}")

    # 5. 비교 with in-sample (2025-2026)
    print(f"\n=== In-sample (2025-2026) vs Out-of-sample (2022-2023) ===")
    print(f"  In-sample G041 30d gate:  n=1525  mean=+328 bps  WR=67.7%")
    print(f"  Out-of-sample G041 30d:   n={g041_stats['n']}  mean={g041_stats['mean']:+.0f} bps  WR={g041_stats['wr']*100:.1f}%")

    # save
    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g041_oos_2022_results.json"
    OUT.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "universe": UNIVERSE,
        "period": "2022-01-01 to 2024-01-01",
        "g003_baseline": g003_stats,
        "g041_adaptive": g041_stats,
        "quarterly_g041": q4,
        "in_sample_comparison": {"n":1525, "mean":328, "wr":0.677},
    }, indent=2, ensure_ascii=False, default=float))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
