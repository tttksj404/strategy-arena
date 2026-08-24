"""
G050 v2 — hypothetical gate (signal-based, 모든 candidate 의 net 평가).

직전 N일 간 발생한 모든 candidate (taken/skipped 무관) 의 hypothetical net 으로
gate 를 결정. → 휴면 중에도 데이터가 누적되어 재활성화 가능.
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT


def gather(data_dir, universe):
    entries = []
    for sym in universe:
        p = Path(data_dir) / sym / "1h.json"
        if not p.exists(): continue
        df = pd.DataFrame(json.loads(p.read_text()))
        if len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        df["score"] = score
        df["fwd_pct"] = (df["close_price"].shift(-72) / df["close_price"] - 1) * 10000
        e = df[(df["score"] >= 70) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - 16
        entries.append(e[["open_time","score","gross_bps","net_bps","sym"]])
    if not entries: return pd.DataFrame()
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True)


def hypothetical_gate(all_e, lookback_days):
    """직전 N일 간 모든 candidate 의 net sum > 0 일 때만 진입."""
    LB = lookback_days * 24 * 3600 * 1000
    if len(all_e) == 0: return None
    first = all_e["open_time"].min()
    taken = []
    nets_arr = all_e["net_bps"].values
    times_arr = all_e["open_time"].values
    syms_arr = all_e["sym"].values
    for i in range(len(all_e)):
        ts = times_arr[i]
        net = nets_arr[i]
        if ts - first < LB:
            taken.append((ts, net, syms_arr[i]))
            continue
        # all candidates in [ts-LB, ts)
        mask = (times_arr >= ts - LB) & (times_arr < ts)
        if nets_arr[mask].sum() > 0:
            taken.append((ts, net, syms_arr[i]))
    return taken


def hypothetical_drawdown(all_e, dd_window_days, dd_threshold_bps, resume_days):
    """Hypothetical 누적 drawdown 기반. 직전 dd_window_days 의 cumulative net <
    threshold 면 resume_days 동안 휴면."""
    if len(all_e) == 0: return None
    DD_W = dd_window_days * 24 * 3600 * 1000
    RESUME = resume_days * 24 * 3600 * 1000
    first = all_e["open_time"].min()
    times_arr = all_e["open_time"].values
    nets_arr = all_e["net_bps"].values
    syms_arr = all_e["sym"].values
    paused_until = -1
    taken = []
    for i in range(len(all_e)):
        ts = times_arr[i]
        if ts - first < DD_W:
            taken.append((ts, nets_arr[i], syms_arr[i]))
            continue
        if ts < paused_until:
            continue
        mask = (times_arr >= ts - DD_W) & (times_arr < ts)
        recent_net = nets_arr[mask].sum()
        if recent_net < dd_threshold_bps:
            paused_until = ts + RESUME
            continue
        taken.append((ts, nets_arr[i], syms_arr[i]))
    return taken


def stats(taken):
    if not taken or len(taken) == 0: return None
    arr = np.array([t[1] for t in taken])
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "wr": float((arr > 0).mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
    }


def main():
    print("=== G050 v2 — Hypothetical signal-based gate ===\n")

    DATA_22 = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2022"
    DATA_25 = Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical"
    UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
    UNIV_25 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]

    e_22 = gather(DATA_22, UNIV_22)
    e_25 = gather(DATA_25, UNIV_25)
    print(f"Candidates: 2022-2023 = {len(e_22)} / 2025-2026 = {len(e_25)}\n")

    print(f"{'gate':<40} {'period':>10} {'n':>5} {'mean':>8} {'WR':>7} {'std':>6}")
    print("-" * 85)
    # baseline
    for label_p, e in [("OOS22-23", e_22), ("IS25-26", e_25)]:
        s = stats([(0, n, "") for n in e["net_bps"].values])
        print(f"{'no gate':<40} {label_p:>10} {s['n']:>5} {s['mean']:>+8.0f} {s['wr']*100:>6.1f}% {s['std']:>6.0f}")
    print()

    # G041 (30-day lookback)
    for lb in [14, 30, 60]:
        for label_p, e in [("OOS22-23", e_22), ("IS25-26", e_25)]:
            t = hypothetical_gate(e, lb)
            s = stats(t)
            if s:
                print(f"{f'lookback {lb}d net>0':<40} {label_p:>10} {s['n']:>5} {s['mean']:>+8.0f} {s['wr']*100:>6.1f}% {s['std']:>6.0f}")
    print()

    # G050 hypothetical drawdown (다양한 window/threshold)
    for window, thr, resume in [(7, -2000, 14), (7, -3000, 14), (14, -3000, 14), (14, -5000, 14), (30, -5000, 30)]:
        for label_p, e in [("OOS22-23", e_22), ("IS25-26", e_25)]:
            t = hypothetical_drawdown(e, window, thr, resume)
            s = stats(t)
            if s:
                print(f"{f'DD: w{window}d thr{thr}bps pause{resume}d':<40} {label_p:>10} {s['n']:>5} {s['mean']:>+8.0f} {s['wr']*100:>6.1f}% {s['std']:>6.0f}")

    # 결합 — DD 안전 + lookback 양수
    print()
    def combined_v2(all_e, dd_window_days, dd_thr, resume_days, gate_lookback_days):
        DD_W = dd_window_days * 24 * 3600 * 1000
        RESUME = resume_days * 24 * 3600 * 1000
        GATE = gate_lookback_days * 24 * 3600 * 1000
        first = all_e["open_time"].min()
        times = all_e["open_time"].values
        nets = all_e["net_bps"].values
        syms = all_e["sym"].values
        paused_until = -1
        taken = []
        for i in range(len(all_e)):
            ts = times[i]
            if ts - first < max(DD_W, GATE):
                taken.append((ts, nets[i], syms[i]))
                continue
            if ts < paused_until: continue
            dd_mask = (times >= ts - DD_W) & (times < ts)
            if nets[dd_mask].sum() < dd_thr:
                paused_until = ts + RESUME
                continue
            gate_mask = (times >= ts - GATE) & (times < ts)
            if nets[gate_mask].sum() <= 0:
                continue
            taken.append((ts, nets[i], syms[i]))
        return taken

    print("--- combined: DD safe AND gate net>0 ---")
    for dd_w, dd_t, res, g_w in [(7, -3000, 14, 14), (14, -5000, 14, 30), (7, -2000, 14, 14)]:
        for label_p, e in [("OOS22-23", e_22), ("IS25-26", e_25)]:
            t = combined_v2(e, dd_w, dd_t, res, g_w)
            s = stats(t)
            if s:
                print(f"{f'DD w{dd_w} thr{dd_t} pause{res}, gate{g_w}d':<40} {label_p:>10} {s['n']:>5} {s['mean']:>+8.0f} {s['wr']*100:>6.1f}% {s['std']:>6.0f}")

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g050_v2_results.json"
    OUT.write_text(json.dumps({"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, indent=2))


if __name__ == "__main__":
    main()
