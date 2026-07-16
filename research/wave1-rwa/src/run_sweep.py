"""Run the full RWA strategy sweep and write leaderboard/report artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .engine import Costs, run_backtest
from .gates import gate_result
from .strategies import StrategySpec, cross_section_signals, session_candidates, signal_for, specs

ROOT = Path(__file__).resolve().parents[1]
LEVERS = (2, 3, 5, 7, 10)


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically without shuffling."""
    cut = max(1, int(len(frame) * 0.6))
    return frame.iloc[:cut].reset_index(drop=True), frame.iloc[cut:].reset_index(drop=True)


def run_one(symbol: str, frame: pd.DataFrame, fund: pd.DataFrame, spec: StrategySpec, spread: float, session: tuple[str, set[int], float] | None = None) -> list[dict[str, object]]:
    """Evaluate one symbol/strategy at all leverage levels."""
    train, test = split(frame)
    train_fund = fund[(fund["ts"] >= train["ts"].iloc[0]) & (fund["ts"] <= train["ts"].iloc[-1])]
    test_fund = fund[(fund["ts"] >= test["ts"].iloc[0]) & (fund["ts"] <= test["ts"].iloc[-1])]
    sig = signal_for(spec, frame, session)
    train_sig, test_sig = sig.iloc[:len(train)].reset_index(drop=True), sig.iloc[len(train):].reset_index(drop=True)
    costs = Costs(spread)
    rows: list[dict[str, object]] = []
    leverage_set = (1, 5) if spec.name == "B0_buy_hold" else LEVERS
    for leverage in leverage_set:
        train_result = run_backtest(train, train_sig, train_fund, leverage, costs)
        test_result = run_backtest(test, test_sig, test_fund, leverage, costs)
        train_gates = gate_result(train_result, train_result)
        gates = gate_result(test_result, train_result)
        rows.append({"symbol": symbol, "strategy": spec.name, "params": json.dumps(spec.params, sort_keys=True), "L": leverage,
                     "train_net_return": train_result.net_return, "train_CAGR": train_result.cagr, "train_MDD": train_result.mdd,
                     "train_Sharpe": train_result.sharpe, "train_trades": train_result.trades, "test_net_return": test_result.net_return,
                     "test_CAGR": test_result.cagr, "test_MDD": test_result.mdd, "test_Sharpe": test_result.sharpe,
                     "test_win_rate": test_result.win_rate, "test_trades": test_result.trades, "fees_paid": test_result.fees_paid,
                     "funding_paid": test_result.funding_paid, "funding_coverage": test_result.funding_coverage, "liquidated": test_result.liquidated,
                     "turnover": test_result.turnover, "train_gate": json.dumps(train_gates, sort_keys=True), "gate": json.dumps(gates, sort_keys=True), "train_all_pass": bool(train_gates["all_pass"]), "all_pass": False})
    eligible = [row for row in rows if bool(row["train_all_pass"])]
    selected = max(eligible, key=lambda row: int(row["L"])) if eligible else None
    for row in rows:
        row["selected_L"] = int(selected["L"]) if selected else None
        if selected and int(row["L"]) == int(selected["L"]):
            row["all_pass"] = bool(json.loads(str(row["gate"]))["all_pass"])
    return rows


def metric_row(result: object, prefix: str) -> str:
    """Render report metric fields."""
    return ""


def write_report(df: pd.DataFrame, manifest: list[dict[str, object]], total: int) -> None:
    """Write concise, evidence-backed report."""
    passed = df[df["all_pass"]].sort_values("test_net_return", ascending=False)
    lines = ["# Bitget RWA 선물 전수 백테스트 보고서", "", f"생성시각(UTC): {datetime.now(UTC).isoformat()}", "",
             "## 1. 데이터 커버리지", "", f"계약 목록: {len(manifest)}개, A/B/C: " + ", ".join(f"{tier}={sum(x.get('tier') == tier for x in manifest)}" for tier in "ABC"),
             f"전략-심볼-L 조합 시도 수: {total:,} (B5 포함, 실제 결과 행은 {len(df):,})", "", "## 2. 게이트 통과 상위 10", ""]
    if passed.empty: lines.append("게이트 통과 0개. 억지 승자를 제시하지 않음.")
    else:
        lines.append("|symbol|strategy|L|test net|test MDD|trades|")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, row in passed.head(10).iterrows(): lines.append(f"|{row.symbol}|{row.strategy}|{row.L}|{row.test_net_return:.2%}|{row.test_MDD:.2%}|{int(row.test_trades)}|")
    lines += ["", "## 3. $300 실전 시나리오", ""]
    if passed.empty: lines.append("게이트 통과 전략이 없어 실전 시나리오를 선정하지 않음.")
    else:
        best = passed.iloc[0]
        lines.append(f"최고 1개: {best.symbol} / {best.strategy} / L={int(best.L)}. {best.params} 신호를 종가 확정 후 다음 봉 시가에 체결한다. 테스트 순수익 {best.test_net_return:.2%}, 테스트 거래 {int(best.test_trades)}회. 비용: taker 0.06%/side + 실측 half-spread(최소 1bp) + slippage 1bp를 체결마다 반영; funding_paid={best.funding_paid:.4f}, fees_paid={best.fees_paid:.4f}.")
        bench = df[(df["symbol"] == best.symbol) & (df["strategy"] == "B0_buy_hold") & (df["L"] == 1)]
        if not bench.empty:
            lines.append(f"동일 심볼 B0 L1(무레버리지 보유) test 순수익 {float(bench.iloc[0].test_net_return):.2%}; 선정 전략 초과수익 {float(best.test_net_return - bench.iloc[0].test_net_return):.2%}.")
    lines += ["", "## 4. 벤치마크 대비", "", "B0 buy&hold는 동일 엔진·비용·펀딩으로 평가했다. 선정 전략의 동일 심볼 B0 대비 초과수익이 음수이면 엣지 없음으로 판정한다.", "", "## 5. 한계", "", f"시계열 60/40 분할, test 1회 평가, trade-shuffle bootstrap 1,000회. 다중검정 총 조합 수 N={total:,}. 최신 상장군은 약 5개월이고 전체 표본 범위는 {min(x.get('days', 0) for x in manifest):.1f}~{max(x.get('days', 0) for x in manifest):.1f}일이며, tier B exploratory 결과는 일반화 근거가 약하고 C는 전략을 스킵했다."]
    (ROOT / "out/REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Execute the sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "out/data_manifest.json").read_text(encoding="utf-8"))
    usable = [x for x in manifest if x.get("tier") in {"A", "B"}]
    if args.max_symbols: usable = usable[:args.max_symbols]
    all_frames = {x["symbol"]: pd.read_parquet(ROOT / "data/candles_1h" / f"{x['symbol']}.parquet") for x in usable}
    all_funding = {symbol: pd.read_parquet(ROOT / "data/funding" / f"{symbol}.parquet") for symbol in all_frames}
    cross = cross_section_signals({s: all_frames[s] for s in all_frames if next(x for x in usable if x["symbol"] == s)["tier"] == "A"}) if any(x["tier"] == "A" for x in usable) else {}
    rows: list[dict[str, object]] = []
    attempts = 0
    for meta in usable:
        symbol, frame, fund = meta["symbol"], all_frames[meta["symbol"]], all_funding[meta["symbol"]]
        train, _ = split(frame)
        session_sets = session_candidates(train)
        for spec in specs():
            candidates = session_sets if spec.name == "B3_session" else [None]
            if spec.name == "B0_buy_hold": candidates = [None]
            for session in candidates:
                run_spec = StrategySpec(spec.name, {**spec.params, "which": session[0]}) if session else spec
                batch = run_one(symbol, frame, fund, run_spec, float(meta.get("half_spread_bp", 1.0)), session)
                for row in batch: row["exploratory"] = meta["tier"] == "B"
                rows.extend(batch); attempts += len(batch)
        if meta["tier"] == "A" and symbol in cross:
            spec = StrategySpec("B5_cross_section", {"lookback": 20, "top": 3, "bottom": 3, "rebalance": "weekly"})
            sig = cross[symbol]; train, test = split(frame); train_sig, test_sig = sig.iloc[:len(train)], sig.iloc[len(train):]
            for leverage in LEVERS:
                tr = run_backtest(train, train_sig.reset_index(drop=True), all_funding[symbol], leverage, Costs(float(meta.get("half_spread_bp", 1.0))))
                te = run_backtest(test, test_sig.reset_index(drop=True), all_funding[symbol], leverage, Costs(float(meta.get("half_spread_bp", 1.0))))
                train_gates = gate_result(tr, tr); gates = gate_result(te, tr)
                rows.append({"symbol": symbol, "strategy": spec.name, "params": json.dumps(spec.params), "L": leverage, "train_net_return": tr.net_return, "train_CAGR": tr.cagr, "train_MDD": tr.mdd, "train_Sharpe": tr.sharpe, "train_trades": tr.trades, "test_net_return": te.net_return, "test_CAGR": te.cagr, "test_MDD": te.mdd, "test_Sharpe": te.sharpe, "test_win_rate": te.win_rate, "test_trades": te.trades, "fees_paid": te.fees_paid, "funding_paid": te.funding_paid, "funding_coverage": te.funding_coverage, "liquidated": te.liquidated, "turnover": te.turnover, "train_gate": json.dumps(train_gates), "gate": json.dumps(gates), "train_all_pass": bool(train_gates["all_pass"]), "all_pass": False, "selected_L": None, "exploratory": False}); attempts += 1
            selected = max((r for r in rows if r["symbol"] == symbol and r["strategy"] == spec.name and r["train_all_pass"]), key=lambda r: int(r["L"]), default=None)
            if selected:
                for row in rows:
                    if row["symbol"] == symbol and row["strategy"] == spec.name:
                        row["selected_L"] = selected["L"]
                        if row["L"] == selected["L"]: row["all_pass"] = bool(json.loads(str(row["gate"]))["all_pass"])
    df = pd.DataFrame(rows).sort_values("test_net_return", ascending=False)
    df.to_csv(ROOT / "out/leaderboard.csv", index=False)
    df.to_json(ROOT / "out/leaderboard.json", orient="records", indent=2)
    write_report(df, manifest, attempts)
    with (ROOT / "out/PROGRESS.md").open("a", encoding="utf-8") as f: f.write(f"{datetime.now(UTC).isoformat()} | sweep | rows={len(df)} attempts={attempts}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
