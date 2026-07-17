# KCYCLE Pick Source Policy Results — Round 8

Generated: `2026-07-12T08:00:35.329649+00:00`

## Scope
Pure measurement only. `engine.py`, `app.py`, and existing tests were not modified. `data/kcycle_trifecta_snapshots_expansion.jsonl` was read only.

## Counts
- base snapshots kept: 13900 / read 14142
- expansion added after key dedupe: 3372 / read 3408
- entries: 12593
- joined keys: 12593
- scored races: 12593

## Model vs Market
| subset | n | model top1 | market top1 | market-model |
| --- | --- | --- | --- | --- |
| all | 12593 | 60.9% | 64.6% | +3.63pp |
| agree | 10369 | 68.1% | 68.1% | +0.00pp |
| disagree | 2224 | 27.4% | 48.0% | +20.55pp |
| 2026_all | 1198 | 59.0% | 63.8% | +4.76pp |
| 2026_agree | 967 | 67.3% | 67.3% | +0.00pp |
| 2026_disagree | 231 | 24.2% | 48.9% | +24.68pp |

## Year Split
| year | n | model top1 | market top1 | market-model | model top2 | market top2 | agree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2018 | 2259 | 60.4% | 62.6% | +2.21pp | 81.5% | 82.3% | 80.7% |
| 2019 | 1387 | 65.5% | 67.3% | +1.73pp | 85.7% | 84.4% | 83.1% |
| 2020 | 321 | 64.2% | 63.6% | -0.62pp | 82.6% | 85.4% | 83.8% |
| 2021 | 209 | 64.6% | 72.7% | +8.13pp | 84.2% | 86.6% | 83.7% |
| 2022 | 2346 | 60.8% | 64.7% | +3.88pp | 80.1% | 82.9% | 82.4% |
| 2023 | 2350 | 61.9% | 65.0% | +3.11pp | 81.1% | 83.3% | 85.7% |
| 2024 | 770 | 59.9% | 64.3% | +4.42pp | 82.1% | 84.4% | 82.1% |
| 2025 | 1753 | 57.8% | 64.2% | +6.45pp | 79.4% | 83.5% | 80.1% |
| 2026 | 1198 | 59.0% | 63.8% | +4.76pp | 81.3% | 83.1% | 80.7% |

## Grade Split
| grade | n | model top1 | market top1 | market-model | agree |
| --- | --- | --- | --- | --- | --- |
| 선발 | 3996 | 58.3% | 62.0% | +3.75pp | 79.5% |
| 우수 | 5152 | 56.7% | 61.0% | +4.31pp | 80.8% |
| 특선 | 3445 | 70.4% | 72.9% | +2.47pp | 87.9% |

## Policy Simulation
P0 = model always. P1 = board market else model. P2 = P1, but on model/market disagreement with weak market signal (`market_gap_ratio < 1.10` or `market_first_mass < 0.22`) use model.

| split | n | P0 | P1 | P1-P0 | P2 | P2-P0 | P2 fallback n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 12593 | 60.9% | 64.6% | +3.63pp | 64.4% | +3.49pp | 234 |
| 2026 | 1198 | 59.0% | 63.8% | +4.76pp | 63.4% | +4.34pp | 22 |

Decision criterion: P1/P2 is a correction candidate if 2026 lift over P0 is at least +1.0pp.

Verdict: **candidate**. P1 candidate=True, P2 candidate=True.

## Place / Pair Approximation
Overall winner-in-top2: model win top2 81.4%, model plc top2 81.3%, market first_mass top2 83.3%.

Overall pair approximations: exacta model 27.4%, exacta market 35.2%, quinella model 37.7%, quinella market 45.7%.

## Actual Win Favorite Check
2025 local win-odds sample: n=1721, board first_mass matches actual win-odds favorite 93.4%; actual win-odds favorite top1 63.7%; board first_mass top1 on same sample 64.1%.

Payoff DB probe: `/Users/tttksj/keirin/data/keirin.db` rows=59202; limitation: payoff rows expose winning payout values, not all runner win odds; cannot identify actual win favorite from payoff alone..
