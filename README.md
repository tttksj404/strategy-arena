# Strategy Arena

Crypto-quant strategy builder and backtester for Binance public-market-data experiments.

This repository is organized as a portfolio project for finance-domain data analysis, backtesting, and risk-aware validation. It is not positioned as a live trading system. The main goal is to show how a strategy idea can be represented, tested, compared, and explained before any real capital decision.

## Portfolio Positioning

Strategy Arena maps well to requirements seen in financial IT, digital banking, data analysis, and AI/quant-adjacent roles:

- **Python data workflow**: strategy rules, simulation logic, and result calculations.
- **Financial domain understanding**: signals, positions, returns, and backtest interpretation.
- **Risk-aware validation**: separates experiment results from real trading claims.
- **User-facing product design**: exposes strategy composition and results through a simple web interface.

## Core Components

| File | Role |
| --- | --- |
| `app.py` | Web app entrypoint and route layer |
| `engine.py` | Strategy construction, simulation, and backtest logic |
| `templates/` | Browser-facing UI templates |
| `START_HERE.txt` | Local usage guide |

For a project-specific interaction map that follows a strategy idea from UI input through the simulation engine, metrics, and risk-aware interpretation, see [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md). The current research contract is in [docs/RESEARCH_GUARDRAILS.md](docs/RESEARCH_GUARDRAILS.md).

## StockPulse Bitget 5x Research

The reusable StockPulse Bitget strategy core is now separated into [research/bitget_5x](research/bitget_5x). It uses public market data only, keeps leverage fixed at 5x for research, models cost and stop-loss limits, and promotes no candidate without chronological validation plus an untouched holdout. It cannot place orders, change leverage, or read an account. The included replay artifact concludes `NO_RELIABLE_WINNER`, not a buy signal. The code-workspace selection boundary is documented in [docs/COIN_SOURCE_INVENTORY_2026-08-24.md](docs/COIN_SOURCE_INVENTORY_2026-08-24.md).

## Why This Matters for Hiring

Companies hiring for digital finance and data roles usually look for more than model usage. They expect candidates to understand data quality, reproducible evaluation, system boundaries, and business risk. This project is meant to demonstrate that habit:

1. Define a strategy idea as data and rules.
2. Run a reproducible simulation.
3. Compare outcomes with clear assumptions.
4. Treat results as evidence for review, not as investment advice.

## Run

Use Python 3.11 or later. The StockPulse Bitget 5x research package enforces this minimum because it uses modern standard-library enums and pattern matching.

```bash
python -m pip install -r requirements.txt
python app.py
```

Then open the local URL printed by the app.

## Limitations

- Backtest output is for experiment and education only.
- It does not execute live orders.
- Real deployment would require market-data quality controls, transaction-cost modeling, compliance review, monitoring, and strict capital-risk guardrails.
