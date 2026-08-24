# Legacy Quant Research Archive

This directory preserves the reusable research portion of the earlier `quant_binance` workspace: market-data adapters, sample fixtures, features, scoring, risk policy, backtests, candidate cards, static experiment results, observability source, and the two historical engine prototypes.

It is deliberately an archive, not a deployment target. Some historical candidate cards use leverage or names that do not match the current 5x research constraint. They are retained so prior exploration remains reviewable, but they are not active candidates and must not be promoted from this archive.

## Deliberately excluded

- Every signed or authenticated exchange adapter, account/order endpoint, live runner, session, daemon, deploy script, remote-control path, environment/settings loader, Telegram notifier, and the Engine B campaign wrapper that imported a live executor.
- Runtime state, logs, output directories, caches, databases, raw market files, downloaded images/documents, and local third-party/user trade datasets.
- Hyperliquid address/fill source data and user-history files.

The remaining files are source/reference material only. They cannot submit an order or recover a credential by themselves.
