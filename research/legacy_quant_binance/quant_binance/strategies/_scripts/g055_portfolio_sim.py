"""G055 Portfolio sim — G050 / G053 양 시기에서 $55 / max3 / 30% / 1x 운용 PnL 추정."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g050_v2_hypothetical import gather

DATA_22 = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2022"
DATA_25 = Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical"
UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]

EQUITY = 55.0
HOLD_MS = 72 * 3600 * 1000


def gate_active_g050(history, ts):
    LB = 14 * 86400 * 1000
    if not history: return True
    first = history[0][0]
    if ts - first < LB: return True
    recent = [n for t, n in history if ts - LB <= t < ts]
    if not recent: return True
    return sum(recent) > 0


def gate_active_g053(history, ts, paused_until):
    """G053: gate14d + DD safety."""
    if ts < paused_until: return False, paused_until
    DD_W = 7 * 86400 * 1000
    LB = 14 * 86400 * 1000
    if not history: return True, paused_until
    first = history[0][0]
    if ts - first < LB: return True, paused_until
    # DD check
    dd_recent = [n for t, n in history if ts - DD_W <= t < ts]
    if dd_recent and sum(dd_recent) < -3000:
        new_pause = ts + 14 * 86400 * 1000
        return False, new_pause
    # gate check
    gate_recent = [n for t, n in history if ts - LB <= t < ts]
    if not gate_recent: return True, paused_until
    return sum(gate_recent) > 0, paused_until


def portfolio_sim(all_e, gate_fn, max_conc, size_pct, label):
    open_pos = []
    history = []
    pnl = 0.0
    taken = 0
    big_wins = 0
    paused_until = -1
    for _, row in all_e.iterrows():
        ts = row["open_time"]
        net_bps = row["net_bps"]
        open_pos = [p for p in open_pos if p[0] > ts]
        if gate_fn == "g050":
            active = gate_active_g050(history, ts)
        else:
            active, paused_until = gate_active_g053(history, ts, paused_until)
        history.append((ts, net_bps))
        if not active: continue
        if any(p[1] == row["sym"] for p in open_pos): continue
        if len(open_pos) >= max_conc: continue
        size = EQUITY * size_pct
        net_pct = net_bps / 10000
        pnl_trade = size * net_pct
        pnl += pnl_trade
        taken += 1
        if net_pct > 0.10: big_wins += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {"label":label,"n_taken":taken,"pnl_usd":round(pnl,2),"pnl_pct":round(pnl/EQUITY*100,1),"big_wins":big_wins}


def main():
    print("=== G055 Portfolio sim ($55, max 3, 30%, 1x) ===\n")
    e_22 = gather(DATA_22, UNIV_22)
    e_25 = gather(DATA_25, UNIV_25)
    print(f"Candidates: 2022-23 = {len(e_22)} / 2025-26 = {len(e_25)}\n")

    # 2024 데이터 추가
    DATA_24 = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2024"
    UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
    e_24 = gather(DATA_24, UNIV_24)
    print(f"+ Candidates 2024 (Jan-Mar2025) = {len(e_24)}\n")

    print(f"{'gate':<6} {'period':>10} {'conc':>4} {'sz%':>4} {'taken':>6} {'PnL$':>8} {'PnL%':>7} {'big':>4} {'days':>5} {'annual':>9}")
    print("-" * 80)
    periods = [("OOS22-23", e_22, 730), ("OOS24-Q1", e_24, 456), ("IS25-26", e_25, 374)]
    configs = [(3, 0.30), (5, 0.20), (5, 0.30), (10, 0.10)]
    for label_p, e, days in periods:
        for conc, sz in configs:
            for gate in ["g050", "g053"]:
                r = portfolio_sim(e, gate, conc, sz, gate.upper())
                annual = round(r["pnl_pct"] / days * 365, 1) if days > 0 else 0
                print(f"{gate.upper():<6} {label_p:>10} {conc:>4} {int(sz*100):>3}% {r['n_taken']:>6} ${r['pnl_usd']:>+7.2f} {r['pnl_pct']:>+6.1f}% {r['big_wins']:>4} {days:>4}d {annual:>+8.1f}%")
        print()

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g055_portfolio_sim_results.json"
    OUT.write_text(json.dumps({"summary": "see stdout"}, indent=2))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
