# Crypto-quant research guardrails

## Research contract

Every experiment records its data source, sampling interval, feature availability time, train/validation/test boundaries, fees, slippage model, and reproducible configuration. A result without those inputs is exploratory only and must not be treated as a deployable strategy.

## Evaluation contract

Use walk-forward or otherwise time-respecting validation. Report total return together with drawdown, turnover, exposure, concentration, benchmark comparison, and sensitivity to transaction costs. Treat a strategy as rejected when it loses its rationale under realistic frictions or out-of-sample evaluation.

## Operational contract

Keep credentials outside version control. Do not place exchange secrets, private account data, or executable trading automation here. Any future production system requires a separate repository, explicit risk limits, human approval, and independent review.
