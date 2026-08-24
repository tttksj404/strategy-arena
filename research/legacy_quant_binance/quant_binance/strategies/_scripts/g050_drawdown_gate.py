"""
G050 — drawdown-based gate (G041 보다 빠른 reaction).

가설: 30-day net 은 너무 늦음. 직전 N개 거래 결과로 더 빠른 detection.

룰:
- 직전 5/10/20 거래의 net 양수 → 진입 OK
- 음수 → 진입 차단
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT
from g041_2022_oos import gather_entries, stats, quarterly, UNIVERSE, DATA_DIR


def adaptive_n_trades(all_e, lookback_n, cost_bps):
    """직전 N 거래 의 net sum > 0 일 때만 진입."""
    if len(all_e) == 0: return None
    history = []
    taken = []
    WARMUP = 30  # 첫 30 거래는 무조건 take
    for _, row in all_e.iterrows():
        net = row["gross_bps"] - cost_bps
        if len(history) < WARMUP:
            taken.append((row["open_time"], net, row["sym"]))
            history.append(net)
            continue
        recent = history[-lookback_n:]
        if sum(recent) > 0:
            taken.append((row["open_time"], net, row["sym"]))
            history.append(net)
    return taken


def adaptive_drawdown(all_e, dd_window_n, dd_threshold_bps, resume_n, cost_bps):
    """직전 N 거래 cumulative net 이 -threshold 이하면 휴면. resume_n 후 재평가."""
    if len(all_e) == 0: return None
    history = []
    taken = []
    paused_until_idx = -1
    WARMUP = 30
    for i, (_, row) in enumerate(all_e.iterrows()):
        net = row["gross_bps"] - cost_bps
        if len(history) < WARMUP:
            taken.append((row["open_time"], net, row["sym"]))
            history.append(net)
            continue
        if i < paused_until_idx:
            continue
        recent = history[-dd_window_n:]
        if sum(recent) < dd_threshold_bps:
            paused_until_idx = i + resume_n
            continue
        taken.append((row["open_time"], net, row["sym"]))
        history.append(net)
    return taken


def test_period(period_label, data_dir, universe):
    """다른 데이터 디렉토리로 entries 수집."""
    entries = []
    for sym in universe:
        p = data_dir / sym / "1h.json"
        if not p.exists(): continue
        df = pd.DataFrame(json.loads(p.read_text()))
        if len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        df["score"] = score
        df["fwd_pct"] = (df["close_price"].shift(-72) / df["close_price"] - 1) * 10000
        e = df[(df["score"] >= 70) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym
        e["gross_bps"] = e["fwd_pct"]
        entries.append(e[["open_time","score","gross_bps","sym"]])
    if not entries: return None
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True)


def main():
    print("=== G050 drawdown-based gate sweep ===\n")

    # 2022-2023 OOS
    print("--- Period: 2022-2023 (Out-of-Sample, BTC bear+recovery) ---")
    e_oos = test_period("2022-2023", DATA_DIR, UNIVERSE)
    print(f"Total candidates: {len(e_oos)}")

    # In-sample for comparison
    archive_dir = Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical"
    UNIVERSE_18 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]
    e_is = test_period("2025-2026", archive_dir, UNIVERSE_18)
    print(f"\n--- Period: 2025-2026 (In-Sample) — {len(e_is)} candidates ---\n")

    print(f"{'gate':<35} {'period':>10} {'n':>5} {'mean':>8} {'WR':>7} {'std':>6}")
    print("-" * 80)

    # baseline (no gate)
    for label_p, e in [("OOS22-23", e_oos), ("IS25-26", e_is)]:
        nogate = [(r["open_time"], r["gross_bps"]-16, r["sym"]) for _, r in e.iterrows()]
        s = stats(nogate, "")
        print(f"{'no gate':<35} {label_p:>10} {s['n']:>5} {s['mean']:>+8.0f} {s['wr']*100:>6.1f}% {s['std']:>6.0f}")

    # G050a: 직전 N 거래 net > 0 (sweep N)
    print()
    for n in [5, 10, 20, 50]:
        for label_p, e in [("OOS22-23", e_oos), ("IS25-26", e_is)]:
            t = adaptive_n_trades(e, n, 16)
            s = stats(t, "")
            if s:
                print(f"{f'last{n}_trades_net>0':<35} {label_p:>10} {s['n']:>5} {s['mean']:>+8.0f} {s['wr']*100:>6.1f}% {s['std']:>6.0f}")

    # G050b: drawdown trigger (recent N 거래 cumulative < -threshold → pause M trades)
    print()
    for window, thr, resume in [(10, -1000, 30), (20, -2000, 50), (10, -2500, 30), (5, -1000, 20)]:
        for label_p, e in [("OOS22-23", e_oos), ("IS25-26", e_is)]:
            t = adaptive_drawdown(e, window, thr, resume, 16)
            s = stats(t, "")
            if s:
                print(f"{f'DD: w{window} thr{thr} pause{resume}':<35} {label_p:>10} {s['n']:>5} {s['mean']:>+8.0f} {s['wr']*100:>6.1f}% {s['std']:>6.0f}")

    # G050c: 결합 - last10 양수 AND drawdown 안전
    print()
    def combined(all_e, lookback_n, dd_window, dd_thr, resume_n, cost):
        if len(all_e) == 0: return None
        history = []; taken = []; paused_until = -1
        WARMUP = 30
        for i, (_, row) in enumerate(all_e.iterrows()):
            net = row["gross_bps"] - cost
            if len(history) < WARMUP:
                taken.append((row["open_time"], net, row["sym"])); history.append(net); continue
            if i < paused_until: continue
            recent_dd = history[-dd_window:]
            if sum(recent_dd) < dd_thr:
                paused_until = i + resume_n
                continue
            recent_n = history[-lookback_n:]
            if sum(recent_n) <= 0:
                continue
            taken.append((row["open_time"], net, row["sym"])); history.append(net)
        return taken

    for label_p, e in [("OOS22-23", e_oos), ("IS25-26", e_is)]:
        t = combined(e, 10, 10, -1500, 30, 16)
        s = stats(t, "")
        if s:
            print(f"{'combined: last10>0 AND DD safe':<35} {label_p:>10} {s['n']:>5} {s['mean']:>+8.0f} {s['wr']*100:>6.1f}% {s['std']:>6.0f}")

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g050_drawdown_gate_results.json"
    OUT.write_text(json.dumps({"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, indent=2))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
