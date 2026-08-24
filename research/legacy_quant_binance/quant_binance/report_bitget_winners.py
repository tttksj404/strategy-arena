from __future__ import annotations

import argparse
from datetime import timezone, datetime, timedelta
from pathlib import Path

from quant_binance.bitget_history_report import build_bitget_realized_winner_report, write_bitget_history_report
from quant_binance.execution.client_factory import build_exchange_rest_client


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Bitget futures realized winners and write a report.")
    parser.add_argument("--base-dir", default="quant_runtime")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--min-pnl-usd", type=float, default=20.0)
    parser.add_argument("--symbol", default="")
    parser.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    client = build_exchange_rest_client(
        exchange="bitget",
        allow_insecure_ssl=True,
        allow_missing_credentials=False,
    )
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=max(int(args.days), 1))
    report = build_bitget_realized_winner_report(
        client=client,
        start_time=start_time,
        end_time=end_time,
        min_realized_pnl_usd=float(args.min_pnl_usd),
        symbol=args.symbol or None,
    )
    output = Path(args.output) if args.output else Path(args.base_dir) / "artifacts" / "bitget_realized_winners.json"
    write_bitget_history_report(report=report, output_path=output)
    print(f"report={output}")
    print(f"winner_count={report.winner_count}")
    for row in report.winners[:20]:
        print(
            {
                "symbol": row.symbol,
                "hold_side": row.hold_side,
                "net_profit_usd": row.net_profit_usd,
                "realized_pnl_usd": row.realized_pnl_usd,
                "open_time": row.open_time,
                "close_time": row.close_time,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
