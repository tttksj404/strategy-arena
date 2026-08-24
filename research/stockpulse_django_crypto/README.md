# StockPulse Crypto Research Source Snapshot

This directory is a source-preserving snapshot of the crypto-specific StockPulse work that remains Django-coupled. It complements the runnable, standalone [Bitget 5x core](../bitget_5x), rather than replacing it.

## Included

- Public Bitget market, candle, ticker, history, stock/ETF-contract screening, and intraday replay research modules.
- Fixed-leverage feasibility, trend-continuation contracts, provenance checks, chronological evaluation, settlement, and forward-collection research.
- The related management commands, migrations, model definitions, and regression tests needed to trace how the work evolved inside StockPulse.

## Boundary

This is a reference archive, not an independently runnable Django app. The parent application's UI, general stock-product code, database, runtime data, templates, static assets, and settings are deliberately not copied. The test suite includes blocked-path assertions containing private/order URLs only to verify that they are rejected; no authenticated client, account read, order placement, or leverage-change code is included.

Any future extraction must preserve the public-data-only contract and the `NO_TRADE` / `NO_RELIABLE_WINNER` gates before it becomes runnable outside StockPulse.
