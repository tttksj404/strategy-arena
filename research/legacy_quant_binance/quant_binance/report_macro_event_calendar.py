from __future__ import annotations

import argparse
from pathlib import Path

from quant_binance.macro_event_calendar import write_official_macro_events


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch official macro event schedules and save them as JSON.")
    parser.add_argument("--output", default="quant_runtime/artifacts/official_macro_events.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_path = write_official_macro_events(output_path=Path(args.output))
    print(f"official_macro_events={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
