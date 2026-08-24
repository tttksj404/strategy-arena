from __future__ import annotations

import argparse

from quant_binance.backtest.comparison import compare_strategies, render_compact_report, write_comparison_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the current strategy and simple baselines on the same paper-live fixture.")
    parser.add_argument("--config", default="quant_binance/config.example.json")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--equity-usd", type=float, default=10000.0)
    parser.add_argument("--capacity-usd", type=float, default=5000.0)
    parser.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = compare_strategies(
        config_path=args.config,
        fixture_path=args.fixture,
        equity_usd=args.equity_usd,
        capacity_usd=args.capacity_usd,
    )
    if args.output:
        write_comparison_report(args.output, report)
    print(render_compact_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
