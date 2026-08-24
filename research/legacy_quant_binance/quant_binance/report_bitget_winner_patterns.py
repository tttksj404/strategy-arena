from __future__ import annotations

import argparse
from pathlib import Path

from quant_binance.bitget_winner_pattern_report import (
    build_winner_pattern_report,
    write_winner_pattern_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Bitget realized winners and summarize pattern clusters.")
    parser.add_argument("--winners-path", default="quant_runtime/artifacts/bitget_realized_winners.json")
    parser.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_winner_pattern_report(winners_path=args.winners_path)
    output = Path(args.output) if args.output else Path(args.winners_path).with_name("bitget_winner_patterns.json")
    write_winner_pattern_report(report=report, output_path=output)
    print(f"report={output}")
    print(f"winner_count={report.winner_count}")
    for line in report.summary:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
