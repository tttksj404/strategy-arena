"""G070 9-point checklist 검증 — 'production candidate' 자격 확인."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct

DATA_22 = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2022"
DATA_24 = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2024"
DATA_25 = Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical"
UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]

THR = 80
HOLD = 24
LEV = 5.0
SIZE_PCT = 0.30
MAX_CONC = 5
EQUITY = 55.0
ATR_GUARD = 8.0
HOLD_MS = HOLD * 3600 * 1000


def gather(data_dir, universe, threshold=THR, hold=HOLD):
    """G070 candidates with ATR guard + intra-bar drawdown."""
    entries = []
    for sym in universe:
        p = Path(data_dir) / sym / "1h.json"
        if not p.exists(): continue
        df = pd.DataFrame(json.loads(p.read_text()))
        if len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score
        df["atr_pct"] = a
        df["fwd_pct"] = (df["close_price"].shift(-hold) / df["close_price"] - 1) * 10000
        # intra-bar min/max during hold (lookahead within hold window — for 5x lev safety analysis)
        intra_low = []; intra_high = []
        for i in range(len(df)):
            if i + hold >= len(df):
                intra_low.append(np.nan); intra_high.append(np.nan); continue
            window = df.iloc[i:i+hold+1]
            entry = window.iloc[0]["close_price"]
            min_low = window["low_price"].min()
            max_high = window["high_price"].max()
            intra_low.append((min_low / entry - 1) * 10000)
            intra_high.append((max_high / entry - 1) * 10000)
        df["intra_low_bps"] = intra_low
        df["intra_high_bps"] = intra_high
        e = df[(df["score"] >= threshold) & (df["atr_pct"] <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - 16
        entries.append(e[["open_time","score","gross_bps","net_bps","atr_pct","intra_low_bps","intra_high_bps","sym"]])
    if not entries: return pd.DataFrame()
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True)


def portfolio_sim(entries, lev=LEV, days=None):
    """5x lev portfolio sim. liquidation = -100% margin (intra-bar -20% × 5lev = -100%)."""
    if len(entries) == 0: return None
    open_pos = []
    pnl = 0.0; taken = 0; wins = 0; lottery = 0; liquidations = 0
    for _, row in entries.iterrows():
        ts = row["open_time"]
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == row["sym"] for p in open_pos): continue
        if len(open_pos) >= MAX_CONC: continue
        margin = EQUITY * SIZE_PCT
        # 5x lev liquidation 체크: intra-bar -20% (실제 가격) = -100% 마진
        intra_low_pct = row["intra_low_bps"] / 10000
        if intra_low_pct < -0.20:  # -20% 이하 가면 liquidation
            net_pct = -1.0  # 마진 전체 손실
            liquidations += 1
        else:
            net_pct = row["net_bps"] / 10000 * lev
        pnl += margin * net_pct
        taken += 1
        if net_pct > 0: wins += 1
        if net_pct > 0.30: lottery += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {
        "n": taken, "pnl_usd": round(pnl, 2),
        "pnl_pct": round(pnl/EQUITY*100, 1),
        "annual": round(pnl/EQUITY*100/days*365, 1) if days else 0,
        "wr": round(wins/taken, 4) if taken else 0,
        "lottery_30pct": lottery,
        "liquidations": liquidations,
        "trades_per_day": round(taken/days, 3) if days else 0,
    }


def main():
    print("=== G070 9-point Checklist 검증 ===\n")
    e_22 = gather(DATA_22, UNIV_22)
    e_24 = gather(DATA_24, UNIV_24)
    e_25 = gather(DATA_25, UNIV_25)
    n_total = len(e_22) + len(e_24) + len(e_25)
    print(f"Candidates (with ATR guard): 22-23={len(e_22)} / 24-Q1={len(e_24)} / 25-26={len(e_25)} → total {n_total}\n")

    # === Check 1: 3-period trade-level avg net > 0 ===
    print("--- Check 1: 3-period trade-level avg net > 0 ---")
    p1_pass = True
    for label, e in [("22-23", e_22), ("24-Q1", e_24), ("25-26", e_25)]:
        if len(e) == 0:
            print(f"  {label}: no candidates ❌"); p1_pass = False; continue
        avg = e["net_bps"].mean()
        ok = "✓" if avg > 0 else "❌"
        print(f"  {label}: n={len(e)} avg_net={avg:+.0f}bps {ok}")
        if avg <= 0: p1_pass = False
    print(f"  Check 1: {'PASS ✓' if p1_pass else 'FAIL ❌'}\n")

    # === Check 2: portfolio sim 3 periods all annual > 0 ===
    print("--- Check 2: Portfolio sim 5x lev, 3 periods annual > 0 ---")
    p2_pass = True
    for label, e, days in [("22-23", e_22, 730), ("24-Q1", e_24, 456), ("25-26", e_25, 374)]:
        r = portfolio_sim(e, days=days)
        if r is None: print(f"  {label}: no entries ❌"); p2_pass = False; continue
        ok = "✓" if r["annual"] > 0 else "❌"
        print(f"  {label}: n={r['n']} annual={r['annual']:+.1f}% liq={r['liquidations']} {ok}")
        if r["annual"] <= 0: p2_pass = False
    print(f"  Check 2: {'PASS ✓' if p2_pass else 'FAIL ❌'}\n")

    # === Check 3: trade-level avg net ≥ +50 bps ===
    print("--- Check 3: trade-level avg net ≥ +50 bps ---")
    all_e = pd.concat([e_22, e_24, e_25])
    overall_avg = all_e["net_bps"].mean()
    p3_pass = overall_avg >= 50
    print(f"  Overall avg net: {overall_avg:+.0f} bps {'✓' if p3_pass else '❌'}\n")

    # === Check 4: 3-period WR all ≥ 65% (low-freq) ===
    print("--- Check 4: WR all ≥ 65% (low-freq lottery) ---")
    p4_pass = True
    for label, e in [("22-23", e_22), ("24-Q1", e_24), ("25-26", e_25)]:
        if len(e) == 0: continue
        wr = (e["net_bps"] > 0).mean()
        ok = "✓" if wr >= 0.65 else "❌"
        print(f"  {label}: WR={wr*100:.1f}% {ok}")
        if wr < 0.65: p4_pass = False
    print(f"  Check 4: {'PASS ✓' if p4_pass else 'FAIL ❌'}\n")

    # === Check 5: 6축 정합성 ===
    print("--- Check 5: 사용자 6축 정합성 ---")
    avg_per_day = sum(len(e)/d for e, d in [(e_22, 730), (e_24, 456), (e_25, 374)]) / 3
    axis = {
        "lottery (avg/거래≥3%)": overall_avg/100 >= 3,
        "5-10x lev (5x)": True,
        "양방향": False,  # long-only
        "≥3건/일": avg_per_day >= 3,
        "단기 (24-72h)": True,  # 24h
        "빠른 검증": True,  # backtest only
    }
    fit = sum(axis.values())
    for k, v in axis.items():
        print(f"  {k}: {'✓' if v else '❌'}")
    p5_pass = fit >= 5
    print(f"  Check 5: {fit}/6 충족 {'PASS ✓' if p5_pass else 'FAIL ❌'}\n")

    # === Check 6: cost sensitivity 30bps 까지 양수 ===
    print("--- Check 6: Cost sensitivity (16 → 30bps) ---")
    p6_pass = True
    for cost in [16, 20, 25, 30]:
        net = (all_e["gross_bps"] - cost).mean()
        print(f"  cost {cost}bps: avg_net={net:+.1f}")
        if net <= 0: p6_pass = False
    print(f"  Check 6: {'PASS ✓' if p6_pass else 'FAIL ❌'}\n")

    # === Check 7: leverage 안전성 (5x liquidation 빈도) ===
    print("--- Check 7: 5x lev 안전성 (intra-bar -20% liquidation 발생률) ---")
    deep_dd = ((all_e["intra_low_bps"] / 10000) < -0.20).sum()
    p7_pass = deep_dd / max(len(all_e), 1) < 0.10  # 10% 미만
    print(f"  intra-bar -20%+ 거래: {deep_dd}/{len(all_e)} ({deep_dd/max(len(all_e),1)*100:.1f}%) {'✓' if p7_pass else '❌'}\n")

    # === Check 8: sample size n ≥ 50 ===
    print("--- Check 8: Sample size n ≥ 50 (3-period total) ---")
    p8_pass = n_total >= 50
    print(f"  total n={n_total} {'✓' if p8_pass else '❌'}\n")

    # === Check 9: 검증 timeline (백테스트만, lengthy warmup X) ===
    print("--- Check 9: 검증 timeline (백테스트 + 1주 내 검증 가능) ---")
    p9_pass = True  # G070 is purely backtest-validated, no 30-day warmup
    print(f"  백테스트 only, no paper-live warmup ✓\n")

    # === SUMMARY ===
    checks = [p1_pass, p2_pass, p3_pass, p4_pass, p5_pass, p6_pass, p7_pass, p8_pass, p9_pass]
    passed = sum(checks)
    print(f"{'='*60}")
    print(f"=== G070 OVERALL: {passed}/9 PASS ===")
    print(f"{'='*60}")
    if passed == 9:
        print("→ ✅ Production candidate 자격 부여")
    else:
        failed = [i+1 for i, c in enumerate(checks) if not c]
        print(f"→ ❌ DRAFT only — 미통과: Check {failed}")


if __name__ == "__main__":
    main()
