# KRA Round 9b Results

- status: enabled=false, production_replace=false
- leakage_assertion: pass (3 horses, future rows deleted and recomputed identical)
- train: 20190101..20250531
- val: 20250601..20260531
- fresh: 20260622..20260711
- fresh_races: 134

## Metrics
- baseline_v4 val top1 30.51%, top3 61.42%, logloss 1.963221
- pace_form val top1 30.43%, top3 63.94%, logloss 1.940648
- baseline_v4 fresh top1 32.84%, top3 63.43%, logloss 1.887947
- pace_form fresh top1 32.09%, top3 61.94%, logloss 1.869373
- R9 expanded_2019_202505 fresh top1 31.34%

## Verdict
- fresh_top1_gate_33pct: False
- val_noninferior: False
- qualifies_candidate: False
- conclusion: missed_gate; KRA fundamental pace-form lever exhausted under this protocol
