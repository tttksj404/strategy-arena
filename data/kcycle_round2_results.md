# KCYCLE prediction uplift Round 2

generated_at: 2026-07-12T02:41:12+00:00
records: 13900
joined_entries: 12593
selection: train/val only; test is reported once after val selection.
metric note: selection_exact is one selected trio; purchase_exact is actual trio inside bought top_k set.

## Gate
- current_axis reference test exact: 0.1606
- gen2_mut_436 top20 selection test exact: 0.1879
- purchase monotonic: pass

## Baselines
| model | top_k | test selection exact | test purchase exact |
|---|---:|---:|---:|
| current_axis_reference | 10 | 0.1606 | n/a |
| market_rank | 10 | 0.1720 | 0.6329 |
| gen2_mut_436 | 10 | 0.1727 | 0.6149 |
| round1_ensemble | 10 | 0.1828 | n/a |
| current_axis_reference | 20 | 0.1606 | n/a |
| market_rank | 20 | 0.1720 | 0.7760 |
| gen2_mut_436 | 20 | 0.1727 | 0.7616 |
| round1_ensemble | 20 | 0.1821 | n/a |
| current_axis_reference | 40 | 0.1606 | n/a |
| market_rank | 40 | 0.1720 | 0.8815 |
| gen2_mut_436 | 40 | 0.1727 | 0.8815 |
| round1_ensemble | 40 | 0.1792 | n/a |

## Val/Test Lift vs Market Rank at top_k=20
| experiment | selected | val purchase lift | test purchase lift | val selection lift | test selection lift |
|---|---|---:|---:|---:|---:|
| blend | beta=0.95 | -0.040pp | -0.217pp | -0.238pp | +0.000pp |
| line_features | alpha=0.01 | -0.198pp | -0.650pp | +0.198pp | -0.289pp |
| conditional_logit | alpha=0.01 | -4.994pp | -4.263pp | -2.774pp | -2.023pp |

## Line Feature Coefficients
| feature | coefficient |
|---|---:|
| neg_log_odds | +0.129126 |
| log_q | +0.128952 |
| neg_odds_ratio_best | -0.048766 |
| rank_score | -0.032921 |
| rec200_rank_sum | +0.022622 |
| pair_mass | +0.021559 |
| entropy_inv | +0.018658 |
| gap110 | -0.016880 |
| unordered_trio_mass | +0.013883 |
| second_mass | -0.013589 |

## Conditional Logit Coefficients
| feature | coefficient |
|---|---:|
| log_first_mass | +0.775097 |
| same_training_teammates | +0.119793 |
| grade_score | +0.095717 |
| high_rate_z | +0.050569 |
| grade_a | -0.050046 |
| age_z | +0.035365 |
| grade_ss | +0.030269 |
| grade_s | +0.027479 |
| gear_z | +0.021906 |
| win_rate_z | -0.015391 |
