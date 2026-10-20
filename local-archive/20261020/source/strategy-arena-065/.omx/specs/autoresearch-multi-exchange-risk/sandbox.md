# Wave 35 sandbox

- Public, unauthenticated market-data endpoints only.
- No API key, account endpoint, order endpoint, paper-order sender, or live execution code.
- Isolated worktree: `strategy-arena-profit-harness`; preserve the main dirty checkout and all KRA/KCYCLE paths.
- Current-active-universe survivorship and current-liquidity conditioning must be reported as contamination, not hidden.
- Candidate and request budgets are finite and recorded. Cached snapshots are immutable evidence for replay.
