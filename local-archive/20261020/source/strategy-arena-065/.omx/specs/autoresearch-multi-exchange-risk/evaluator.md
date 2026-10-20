# Wave 35 validator contract

The exact validator command is:

`uv run research/wave35_multi_exchange/run_wave35.py validate`

PASS requires every condition below:

1. The four frozen public exchange responses replay exactly into the complete terminal universe ledger.
2. Every history candidate has one terminal history row; no acquisition error is present.
3. Training, validation, historical stress, and untouched holdout periods are disjoint and fixed before holdout evaluation.
4. The bounded graph records 480 policies, 48 validation finalists, and one frozen policy without post-validation mutation.
5. All 14 abrupt-change scenarios execute and every path-level exposure, concentration, drawdown, and liquidation gate passes.
6. No private endpoint, credential, account, position, or order path is used.
7. A separately composed replay reproduces the final artifact byte-for-byte.
8. Every normalized/raw cache reference is root-confined and hash-verified.
9. The final automated decision remains `research_only` regardless of return.
10. TradingAgents remains outside the alpha path and its pinned commit, decisive files, and 42-test JUnit evidence verify.
11. Luna Max pre-holdout review reports PASS before the primary holdout is evaluated.
12. Machine output exposes artifact/replay correctness and strategy gates separately; overall `passed` is their conjunction and `strategy_approved` remains false for this research harness.

Any missing artifact, alias, mismatch, stale source fingerprint, failed scenario, or non-research decision is FAIL.

An exact replay can therefore pass its narrow correctness check while the evaluator still fails because strategy gates fail. A post-holdout source correction may be approved only as structural remediation for future unseen data; it cannot restore or rerun the consumed holdout.
