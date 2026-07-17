# KRA Round 7 results

generated_at: 2026-07-12T07:47:20+00:00

## Current policy gate
`predict_kra` calls `_kra_prediction_phase`; that returns `live_odds` only when `meta['odds_snapshot_fresh'] is True` and `_kra_market_probabilities(starters)` finds a complete positive win-odds board. `score_kra(..., use_market=True)` is therefore used only for fresh official pre-start odds; missing, partial, stale, post-result, or unproven odds stay on the model path.

## Policy simulation

### all
| policy | races | top1 | top3 | market_source |
|---|---:|---:|---:|---:|
| P0_v4_always | 6249 | 31.85% | 63.32% | 0.00% |
| P1_current_gate | 6249 | 31.85% | 63.32% | 0.00% |
| P2_market_if_odds | 6249 | 36.74% | 69.16% | 100.00% |
| P3_market_except_weak_disagree | 6249 | 36.90% | 68.91% | 92.61% |

### fresh_from_20260622
| policy | races | top1 | top3 | market_source |
|---|---:|---:|---:|---:|
| P0_v4_always | 134 | 32.84% | 66.42% | 0.00% |
| P1_current_gate | 134 | 32.84% | 66.42% | 0.00% |
| P2_market_if_odds | 134 | 39.55% | 70.90% | 100.00% |
| P3_market_except_weak_disagree | 134 | 39.55% | 70.90% | 95.52% |

## Policy decision
- winner: market_if_odds
- fresh_lift_vs_p1_pp: 6.72
- supported: True

## History extension
- status: blocked_dns_resolution_failed
- available_start: 20240105
- probe_error: ERR:<urlopen error [Errno 8] nodename nor servname provided, or not known>

## Training extension
- status: not_run_history_extension_blocked
- enabled: False
- fresh_holdout_used: False
- note: No 2019-2023 rows were available, so v4 architecture retraining and COVID-year ablation were skipped without consuming the final fresh holdout evaluation.
