from __future__ import annotations

import argparse
from pathlib import Path

from quant_binance.performance_report import build_runtime_performance_report, write_runtime_performance_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a runtime performance report for the latest or specified live run.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--base-dir", default="quant_runtime")
    parser.add_argument("--output", default="")
    return parser


def _resolve_run_dir(*, base_dir: str | Path, run_dir: str | Path | None) -> Path:
    if run_dir:
        return Path(run_dir)
    mode_root = Path(base_dir) / "output" / "paper-live-shell"
    runs = sorted([p for p in mode_root.iterdir() if p.is_dir() and p.name != "latest"])
    if not runs:
        raise FileNotFoundError("no paper-live-shell runs found")
    return runs[-1]


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_dir = _resolve_run_dir(base_dir=args.base_dir, run_dir=args.run_dir or None)
    report = build_runtime_performance_report(run_dir=run_dir)
    output = Path(args.output) if args.output else run_dir / "performance_report.json"
    write_runtime_performance_report(report=report, output_path=output)
    print(f"run_dir={report.run_dir}")
    print(f"report={output}")
    print(f"closed_trade_count={report.closed_trade_count}")
    print(f"realized_pnl_usd={report.realized_pnl_usd}")
    print("top_symbol_expectancy=")
    for row in report.symbol_expectancy[:5]:
        print(
            {
                "symbol": row.symbol,
                "trade_count": row.trade_count,
                "hit_rate": row.hit_rate,
                "expectancy_usd": row.expectancy_usd,
                "realized_pnl_usd": row.realized_pnl_usd,
            }
        )
    print("regime_performance=")
    for row in report.regime_performance:
        print(
            {
                "mode": row.mode,
                "decision_count": row.decision_count,
                "avg_score": row.avg_score,
                "avg_net_edge_bps": row.avg_net_edge_bps,
                "avg_cost_bps": row.avg_cost_bps,
            }
        )
    print("walk_forward=")
    for row in report.walk_forward:
        print(row)
    print("pruning_recommendations=")
    for row in report.pruning_recommendations[:10]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
