# KCYCLE surprise top2 experiment

## Summary
- source: snapshot_market_proxy
- races: 13900
- baseline_top2_slot_hit_rate: 0.6760431654676259
- surprise_top2_count: 4955
- surprise_top2_rate: 0.17823741007194244

## Common surprise gates
- gate 2: count=773 share=0.1560
- gate 4: count=765 share=0.1544
- gate 7: count=706 share=0.1425
- gate 3: count=705 share=0.1423
- gate 6: count=681 share=0.1374
- gate 1: count=662 share=0.1336
- gate 5: count=657 share=0.1326

## Diagnostics
- actual_position_counts: {'1': 1440, '2': 3515}
- min_top2_odds_buckets: {'odds_20_50': 1880, 'odds_50_100': 958, 'odds_gt_100': 1007, 'odds_le_20': 1104, 'unknown': 6}
- best20_top2_absent_rate: 0.26579212916246214

## Walk-forward gate boost
- baseline_top2_slot_hit_rate: 0.6760431654676259
- candidate_top2_slot_hit_rate: 0.6761510791366907
- slot_hit_lift: 0.0001079136690647482
- adjusted_races: 271

## Interpretation
- Gate surprise boost improved top2 slot recovery only marginally; treat it as a diagnostic, not a deployable rule.

## Prediction-model cohort
- races: 13
- baseline_top2_slot_hit_rate: 0.5769230769230769
- surprise_top2_count: 7
- surprise_top2_rate: 0.2692307692307692
- gate_boost_slot_hit_lift: 0.0
