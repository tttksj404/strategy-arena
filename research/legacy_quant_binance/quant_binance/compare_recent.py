from __future__ import annotations

import argparse

from quant_binance.backtest.comparison import compare_strategies, render_compact_report, write_comparison_report
from quant_binance.backtest.recent_comparison import (
    build_recent_comparison_services,
    default_recent_comparison_output_root,
    prepare_recent_comparison_fixture,
)
from quant_binance.settings import Settings


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a conservative recent-data comparison fixture from local runtime artifacts and run the strategy comparison."
    )
    parser.add_argument("--config", default="quant_binance/config.example.json")
    parser.add_argument("--base-dir", default="quant_runtime")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--equity-usd", type=float, default=None)
    parser.add_argument("--capacity-usd", type=float, default=None)
    parser.add_argument("--fixture-output", default="")
    parser.add_argument("--prep-output", default="")
    parser.add_argument("--comparison-output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_root = default_recent_comparison_output_root(base_dir=args.base_dir)
    fixture_output = args.fixture_output or str(output_root / "recent_fixture.json")
    prep_output = args.prep_output or str(output_root / "preparation.json")
    comparison_output = args.comparison_output or str(output_root / "comparison.json")

    prepared = prepare_recent_comparison_fixture(
        config_path=args.config,
        base_dir=args.base_dir,
        run_dir=args.run_dir or None,
        fixture_path=fixture_output,
        preparation_report_path=prep_output,
        equity_usd=args.equity_usd,
        capacity_usd=args.capacity_usd,
    )
    settings = Settings.load(args.config)
    services = build_recent_comparison_services(
        settings=settings,
        decisions_path=prepared.source.decisions_path,
    )
    report = compare_strategies(
        config_path=args.config,
        fixture_path=prepared.fixture_path,
        equity_usd=prepared.equity_usd,
        capacity_usd=prepared.capacity_usd,
        services=services,
    )
    write_comparison_report(comparison_output, report)

    print(f"source_run={prepared.source.run_dir}")
    print(f"fixture={prepared.fixture_path}")
    print(f"prep_report={prepared.preparation_report_path}")
    print(f"comparison_report={comparison_output}")
    print("current_strategy_source=recorded decisions.jsonl trace on reconstructed market fixture")
    print(
        "fidelity_limits="
        + "; ".join(prepared.missing_inputs)
    )
    print(render_compact_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
