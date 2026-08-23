# Coin-source inventory — 2026-08-24

## Discovery result

The computer-wide code-workspace scan found one existing GitHub destination for crypto research: `tttksj404/strategy-arena`. Its current `main` already contains the published Binance/Bitget research waves, research contracts, and their tests.

The current StockPulse checkout supplies the new, reusable Bitget intraday 5x research core. It has been extracted into `research/bitget_5x/` in this repository.

## Included

- Existing Strategy Arena crypto research code and documentation already tracked on `main`.
- StockPulse public Bitget adapters, 5x feasibility policy, completed-candle replay engine, technical-score filter, Bollinger-squeeze filter, regression tests, candidate registry, and sanitized research artifacts.

## Deliberately not imported

- Archived tool/settings bundles and historical prompt backups: they are not a canonical strategy source and may contain configuration material.
- Stock-only projects, general finance projects, private documents, browser/application data, and other unrelated workspaces.
- StockPulse Django UI, ORM models, database, and operational signal/ledger wrappers. They are application integration, not portable strategy code.
- Caches, logs, local output, run artifacts, key files, and credentials.

This inventory is intentionally code-focused: it avoids reading private application storage or copying non-source personal data.
