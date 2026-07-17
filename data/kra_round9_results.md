# KRA Round 9 Results

- train: 20190101..20250531
- val: 20250601..20260531
- fresh: 20260622..20260711
- fresh races: 134
- sectional features: bu_1fGTime, bu_2fGTime, bu_3fGTime, je_1cTime, je_2cTime, je_3cTime, je_4cTime
- missing: bu_*fGTime and je_*cTime columns are coerced numeric; missing/non-numeric values remain NaN through relative transforms and are median-imputed from the training split only at model fit/predict time.

## Baseline Fresh
- current_v4_overall: top1 37.31%, top3 70.15%, logloss 0.255150
- current_v4_pre_odds_median_filled: top1 17.16%, top3 50.00%, logloss 0.341030

## Fresh Candidates
- sectional_times: top1 44.78%, top3 70.90%, logloss 0.253426
- sectional_recent_weighted: top1 41.79%, top3 70.90%, logloss 0.252921
- rank_average_ensemble_top3: top1 41.79%, top3 71.64%, logloss 1.215085
- expanded_2019_202505: top1 31.34%, top3 57.46%, logloss 0.292218
- no_covid_2020_2021_excluded: top1 31.34%, top3 56.72%, logloss 0.291984
- recent_weighted: top1 29.85%, top3 58.21%, logloss 0.291603

## Ensemble
- selected: sectional_times, sectional_recent_weighted, recent_weighted
- val selection: mean rank of val top1 desc, top3 desc, log_loss asc

## Promotion
- best: sectional_times
- qualifies_for_fable5_review: True
- candidate_artifact: /private/tmp/kra_round9/kra_model_v6_candidate.joblib
- production_replace: false

## Path Limitation
- /Users/tttksj/kra is not writable from this sandbox; staged in /private/tmp/kra_round9.

## Fable5 Verdict (2026-07-12)
- sectional_times/sectional_recent_weighted/ensemble: **FAIL — label leakage** (bu_*fGTime·je_*cTime은 예측 대상 경주의 주행 결과. 스펙 결함: 과거-이력 제한 누락)
- expanded_2019_202505 / no_covid / recent_weighted: 정직 결과 — v4 대비 이득 0 (31.34% ≈ 31.3%). **학습창 확장 레버 소진 확정.**
- v6 candidate joblib: 채택 금지, tmp 보관만.
