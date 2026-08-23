# StockPulse Bitget 5x Research Extraction

Python 3.11 or later is required and enforced at package import time.

This package is the standalone, public-data-only strategy core extracted from StockPulse on 2026-08-24. It keeps the reusable algorithmic pieces while intentionally leaving behind the StockPulse Django models, database ledger, browser UI, credentials, and operational commands.

## What is included

- Public Bitget market, historical-candle, and ticker adapters.
- Closed-candle 15m / 5m (and optional 1m) breakout planning.
- Fixed `5x` risk model: an all-in stop-loss cap, fee/spread/slippage model, and minimum post-cost target threshold.
- Chronological candidate replay, validation-only selection, untouched holdout gating, and `NO_RELIABLE_WINNER` behavior.
- Existing technical-score and Bollinger-squeeze filters, used only as causal research filters on the shared exit and cost model.
- The older 4h `trend_atr` reference rule, kept separate from the intraday replay selector.

## What is deliberately excluded

- API keys, secrets, account information, private endpoints, authenticated signing, leverage changes, and all order placement.
- StockPulse's Django models, database ledger, stock/ETF catalog, management commands, UI, logs, and local runtime artifacts.
- Any claim that 5x prevents liquidation or guarantees profit.

## Run the regression checks

```bash
python -m pytest research/bitget_5x/tests -q
```

The adapter is read-only. It can fetch public market data for research, but it cannot create orders or alter a Bitget account.

## Research status

The committed 30-day / four-contract artifact is in [results/2026-08-23-fixed-5x-replay.json](results/2026-08-23-fixed-5x-replay.json). Its conclusion is `NO_RELIABLE_WINNER`: all four precommitted volume variants had a negative post-cost validation return lower bound. The six-candidate contract and current status are also committed under [results](results) and [docs](../../docs/BITGET_5X_CANDIDATE_STATUS_2026-08-24.md). That is a non-promotion result, not an investment recommendation.
