from __future__ import annotations

import argparse
from pathlib import Path

from quant_binance.news_macro_signal import write_news_macro_signal


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build news + macro signal artifacts for strategy overlays.")
    parser.add_argument("--output", default="quant_runtime/artifacts/news_macro_signal.json")
    parser.add_argument("--macro-output", default="quant_runtime/artifacts/news_macro_inputs.json")
    parser.add_argument("--official-events", default="quant_runtime/artifacts/official_macro_events.json")
    parser.add_argument("--state-output", default="quant_runtime/artifacts/news_macro_signal.state.json")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_path, macro_inputs_path, status = write_news_macro_signal(
        output_path=Path(args.output),
        macro_inputs_output_path=Path(args.macro_output),
        official_events_path=Path(args.official_events),
        state_path=Path(args.state_output),
        force=args.force,
    )
    print(f"news_macro_signal={output_path}")
    print(f"macro_inputs={macro_inputs_path}")
    print(f"status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
