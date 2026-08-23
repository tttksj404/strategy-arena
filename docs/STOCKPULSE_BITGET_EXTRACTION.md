# StockPulse Bitget research extraction

## Purpose

`research/bitget_5x/` makes the reusable Bitget research strategy independently reviewable inside Strategy Arena. It is not a live-trading integration.

## Source map

| StockPulse source family | Strategy Arena destination | Reason |
| --- | --- | --- |
| `pulse/services/bitget_market.py`, `bitget_history.py`, `bitget_ticker.py` | `research/bitget_5x/bitget_*.py` | Public market data and quote parsing. |
| `pulse/services/market_timeframes.py`, `long_forecast.py`, `intraday_long_forecast.py` | same-named files | Candle types, 4h reference rule, and the fixed-5x MTF breakout plan. |
| `pulse/services/intraday_replay.py`, `intraday_replay_engine.py` | same-named files | Causal replay, validation/holdout selector, and cost-adjusted settlement. |
| `pulse/services/trend_continuation_contract.py`, `trend_continuation_features.py` | same-named files | Existing technical-score research filter. |
| `pulse/intraday_signal_models.SignalOutcome` | `signal_outcomes.py` | Pure enum replacement; no database dependency. |

## Explicit exclusions

- StockPulse application models, migrations, database, templates, static UI, and management commands are application glue rather than portable strategy logic.
- Any `.env`, credentials, tokens, API secrets, private account endpoints, user data, logs, caches, output artifacts, and local paths are excluded.
- The source adapter accepts public Bitget market data only. There is no code path for ordering, signing, account inspection, or leverage changes.

## Verification boundary

The embedded fixed-5x artifact retains the correct negative result: the precommitted volume variants did not pass the cost-adjusted chronological validation gate. New candidates must use completed candles, a frozen cost model, validation-only selection, and an untouched later holdout before any promotion claim.
