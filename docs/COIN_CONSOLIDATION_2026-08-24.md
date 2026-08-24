# Crypto Research Consolidation — 2026-08-24

`strategy-arena` is the single public home for the discovered reusable crypto research. This consolidation keeps the code and evidence that can be safely reviewed and reproduced without publishing credentials, personal activity, raw external collections, or a live-trading control surface.

## Included surfaces

| Surface | Contents | Intended use |
| --- | --- | --- |
| `research/bitget_5x/` | Standalone public Bitget adapters, fixed-5x replay candidates, cost/stop model, tests, and committed replay artifacts | Runnable research package |
| `research/stockpulse_django_crypto/` | StockPulse Bitget/intraday/trend-continuation source, commands, migrations, models, tests, and research note | Reference snapshot of the Django implementation |
| `research/legacy_quant_binance/` | Earlier public-data adapters, sample fixtures, features, strategy/risk/policy and observability source, backtests, candidate cards, static result JSON, and archived engine prototypes | Historical candidate and method archive |
| Existing `research/wave*/`, scanner, and validation directories | Previously published Binance/Bitget research waves already in this repository | Existing chronological research record |

## Explicit publication exclusions

| Source area | Excluded material | Reason |
| --- | --- | --- |
| StockPulse | `.env*`, databases, runtime files, templates/static UI, general app code, Telegram archive tooling, local artifacts | Secrets, private data, or non-portable product surface |
| Legacy quant workspace | `execution/`, `live.py`, `session.py`, `daemon.py`, `runtime.py`, `remote.py`, `service.py`, settings/env loaders, live promotion, Telegram/remote scripts, and Engine B campaign wrappers importing a live executor | Signed/private exchange access or live-operation path |
| Legacy data and strategy inputs | Raw market files, outputs, logs, state, downloaded media, user trade history, Hyperliquid addresses/fills, caches | Privacy, provenance, or reproducibility boundary |
| Every source | API keys, tokens, passphrases, private keys, account/order actions, account reads, leverage changes | Non-negotiable public-repository safety boundary |

The source snapshot intentionally retains guard tests that mention blocked private/order routes. Those strings are test inputs proving rejection, not usable endpoints.

## Current decision status

The current runnable policy is fixed 5x research only. The committed 30-day/four-contract replay has no candidate that clears the validation gate, so the status remains `NO_RELIABLE_WINNER`. This repository provides research evidence, not financial advice, a profit claim, or an instruction to trade.
