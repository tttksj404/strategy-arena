from __future__ import annotations

import argparse
from pathlib import Path

from quant_binance.validation_report import build_weekly_validation_report, write_weekly_validation_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a weekly validation report for recent live runs.")
    parser.add_argument("--base-dir", default="quant_runtime")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_weekly_validation_report(base_dir=args.base_dir, lookback_days=args.lookback_days)
    output = Path(args.output) if args.output else Path(args.base_dir) / "artifacts" / "weekly_validation_report.json"
    write_weekly_validation_report(report=report, output_path=output)
    print(f"base_dir={report.base_dir}")
    print(f"report={output}")
    print(f"run_count={report.run_count}")
    print(f"total_closed_trade_count={report.total_closed_trade_count}")
    print(f"total_realized_pnl_usd={report.total_realized_pnl_usd}")
    print("symbol_summary=")
    for row in report.symbol_summary[:10]:
        print(row)
    print("regime_summary=")
    for row in report.regime_summary:
        print(row)
    print("criteria=")
    for row in report.criteria:
        print({"category": row.category, "rule": row.rule, "action": row.action})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
