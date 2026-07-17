# KCYCLE prediction uplift Round 3

generated_at: 2026-07-12T02:51:01+00:00
records: 13900
joined_entries: 12593
selection: train/val only; test is reported once after val selection.
leakage gate: passed by deleting future races for sampled racers and matching features.

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
| runner_form_conditional_logit | alpha=0.01 | -4.875pp | -4.191pp | -2.774pp | -1.951pp |
| trio_form_ridge | alpha=0.01 | +0.238pp | -0.145pp | +0.119pp | -0.145pp |

## Form Coefficients
| track | feature | coefficient |
|---|---|---:|
| runner | log_first_mass | +0.774487 |
| runner | top3_rate_last10 | +0.053045 |
| runner | streak | +0.040200 |
| runner | meet_top3_rate | +0.039921 |
| runner | top3_rate_last5 | -0.035608 |
| runner | days_since_last | -0.029408 |
| runner | rec200_delta | +0.009126 |
| runner | gear_delta | +0.007847 |
| runner | grade_change | -0.001428 |
| trio | top3_rate_last5_sum | +0.013894 |
| trio | top3_rate_last10_sum | +0.009190 |
| trio | top3_rate_last5_min | -0.008856 |
| trio | meet_top3_rate_sum | +0.008107 |
| trio | top3_rate_last10_min | -0.007154 |
| trio | streak_min | -0.006131 |
| trio | meet_top3_rate_min | +0.005877 |
| trio | grade_change_min | -0.004520 |
| trio | streak_sum | -0.004042 |
| trio | gear_delta_sum | +0.003856 |

## Subgroups
| subgroup | test n | track_b purchase | market purchase | lift |
|---|---:|---:|---:|---:|
| days_since_last_gt30 | 375 | 0.8133 | 0.8133 | +0.000pp |
| gear_up_included | 35 | 0.7143 | 0.7143 | +0.000pp |

## Ensemble Bonus
- top_k=20 add_trio_form_rank_member val_delta=-0.040pp test_delta=-0.072pp
