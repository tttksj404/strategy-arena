# Candidate inventory and routing decision

| Candidate family | Imported research status | 5x search disposition |
| --- | --- | --- |
| `breakout-volume-*` | Existing 15m/5m completed-candle breakout with four precommitted RVOL thresholds | Benchmarked; prior 30-day/4-contract artifact had negative conservative return for every threshold. |
| `trend_atr_v1` | Existing 4h SMA/RSI/range-break rule | Keep as a separate 4h research reference, not a fair input to the intraday replay because its horizon and cost evaluator differ. |
| `baseline_mtf_v1` | Existing trend-continuation research rule | Conditional pass: its own fee-provenance gate currently returns `historical_fee_policy_coverage_unproven`; do not mix it into a 5x winner claim. |
| `technical_score_v1` | Existing causal 15m/5m score based on direction, RSI, MACD, ROC, RVOL, ATR, and ADX | Added only as a research-only entry filter on the shared 5x breakout exit/cost model. |
| Bollinger squeeze + breakout | New bounded OHLCV-only hypothesis | Added only as a research-only entry filter; it must pass the same chronology and cost gates. |
| 5-minute opening-range breakout | Source-backed intraday hypothesis | Deferred: a 30-day window yields at most about one signal per session per contract, below the 80-observation validation gate without a larger time-safe dataset. |
| Volatility-managed exposure | Risk-sizing method, not an entry algorithm | Not ranked as a signal candidate; fixed-5x replay must stay comparable. |

Primary source anchors:

- Brock, Lakonishok, and LeBaron (1992), moving-average and trading-range-break rules: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04681.x
- Moskowitz, Ooi, and Pedersen (2012), time-series momentum: https://fairmodel.econ.yale.edu/ec439/mosk.pdf
- Moreira and Muir (2017), volatility-managed portfolios: https://onlinelibrary.wiley.com/doi/10.1111/jofi.12513

None of these sources proves profitability for Bitget RWA contracts, this timeframe, or a fixed 5x cost model. They are candidate-generation evidence only.
