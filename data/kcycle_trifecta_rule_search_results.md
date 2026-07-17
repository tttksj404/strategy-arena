# KCYCLE trifecta rule search

records: 13900 train: ['2018', '2019', '2020', '2021', '2022', '2023'] holdout: ['2024', '2025', '2026']
promotion_rules: {'min_train_n': 50, 'min_eval_holdout_year_n': 5, 'min_promote_holdout_year_n': 10, 'min_worst_year_hit': 0.5}
predicate_count: 7887 evaluated: 80359
xdom_methods: ['xdom_annealing_escape', 'xdom_drug_scaffold_hop', 'xdom_drug_multi_objective_funnel', 'xdom_clonal_selection_amplify', 'xdom_ecology_predator_prey', 'xdom_bandit_explore_exploit', 'xdom_bayesian_surrogate_focus', 'xdom_bradley_terry_position', 'xdom_harville_order_flow', 'xdom_information_bottleneck', 'xdom_particle_filter_resample']
fifty_watch_or_promote_count: 153 promotion_count: 0
xdom_fifty_watch_or_promote_count: 99 xdom_promotion_count: 0
deduped_fifty_watch_or_promote_count: 9 deduped_xdom_fifty_watch_or_promote_count: 9
pass_count_by_min_year_n: {'5': 153, '10': 0, '20': 0, '30': 0, '50': 0, '100': 0}
directional_lift_count: 79 directional_lift_promote_count: 2 stat_strict_lift_count: 2
risk_flags: {'no_robust_promotion': True, 'low_sample_watch_only': True, 'xdom_duplicate_inflation': True, 'requires_more_outcome_linked_snapshots': True, 'directional_lift_not_stat_strict': False}

Interpretation: xdom recipes are evaluated in the KCYCLE trifecta harness, but current 50%+ holdout passes remain low-sample strong-favorite slices. No candidate is promoted unless every holdout year has n >= 10.

## XDOM diversity versus board_min

| method | holdout_diff_rate | holdout_diff_n | holdout_hit_when_diff |
|---|---:|---:|---:|
| xdom_annealing_escape | 0.0187 | 73 | 0.0822 |
| xdom_drug_scaffold_hop | 0.0978 | 382 | 0.0969 |
| xdom_drug_multi_objective_funnel | 0.0225 | 88 | 0.0909 |
| xdom_clonal_selection_amplify | 0.0788 | 308 | 0.0974 |
| xdom_ecology_predator_prey | 0.0527 | 206 | 0.1117 |
| xdom_bandit_explore_exploit | 0.0289 | 113 | 0.0973 |
| xdom_bayesian_surrogate_focus | 0.0724 | 283 | 0.0919 |
| xdom_bradley_terry_position | 0.0939 | 367 | 0.0872 |
| xdom_harville_order_flow | 0.0456 | 178 | 0.0730 |
| xdom_information_bottleneck | 0.0253 | 99 | 0.0707 |
| xdom_particle_filter_resample | 0.0187 | 73 | 0.0822 |

## Directional lift candidates versus board_min

| status | method | rule | train lift | n24/lift | n25/lift | n26/lift | holdout lift | wins-losses | p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PROMOTE_STAT_STRICT_LIFT | xdom_bradley_terry_position | second_mass_best>=0.380599&third_mass_best>=0.263285 | +0.054pp | 180/+0.556pp | 571/+0.701pp | 533/+0.375pp | +0.545pp | 7-0 | 0.0156 |
| PROMOTE_STAT_STRICT_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599&third_mass_best>=0.263285 | +0.000pp | 180/+0.556pp | 571/+0.525pp | 533/+0.375pp | +0.467pp | 6-0 | 0.0313 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | xdom_bandit_explore_exploit:pair_gap&not_extreme_gap12 | -0.263pp | 54/+1.852pp | 161/+1.863pp | 152/+1.316pp | +1.635pp | 7-1 | 0.0703 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.80531&top3_same_first>=1 | -0.210pp | 64/+1.562pp | 217/+1.382pp | 189/+1.058pp | +1.277pp | 7-1 | 0.0703 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.80531 | -0.200pp | 64/+1.562pp | 228/+1.316pp | 200/+1.000pp | +1.220pp | 7-1 | 0.0703 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=1.66741&pair_gap>=2.80531 | -0.200pp | 64/+1.562pp | 228/+1.316pp | 200/+1.000pp | +1.220pp | 7-1 | 0.0703 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=1.90838&pair_gap>=2.80531 | -0.200pp | 64/+1.562pp | 228/+1.316pp | 200/+1.000pp | +1.220pp | 7-1 | 0.0703 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.24099&pair_gap>=2.80531 | -0.200pp | 64/+1.562pp | 228/+1.316pp | 200/+1.000pp | +1.220pp | 7-1 | 0.0703 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.80531&top5_same_first>=0 | -0.200pp | 64/+1.562pp | 228/+1.316pp | 200/+1.000pp | +1.220pp | 7-1 | 0.0703 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.80531&top3_same_pair>=0 | -0.200pp | 64/+1.562pp | 228/+1.316pp | 200/+1.000pp | +1.220pp | 7-1 | 0.0703 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.80531&top3_same_pair>=1 | -0.473pp | 40/+2.500pp | 139/+0.719pp | 120/+0.833pp | +1.003pp | 4-1 | 0.3750 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.24099&top3_same_first>=1 | -0.181pp | 121/+0.826pp | 390/+0.769pp | 322/+0.621pp | +0.720pp | 8-2 | 0.1094 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.24099 | -0.100pp | 127/+0.787pp | 428/+0.701pp | 353/+0.567pp | +0.661pp | 8-2 | 0.1094 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=1.66741&pair_gap>=2.24099 | -0.100pp | 127/+0.787pp | 428/+0.701pp | 353/+0.567pp | +0.661pp | 8-2 | 0.1094 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=1.90838&pair_gap>=2.24099 | -0.100pp | 127/+0.787pp | 428/+0.701pp | 353/+0.567pp | +0.661pp | 8-2 | 0.1094 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.24099&top5_same_first>=0 | -0.100pp | 127/+0.787pp | 428/+0.701pp | 353/+0.567pp | +0.661pp | 8-2 | 0.1094 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=2.24099&top3_same_pair>=0 | -0.100pp | 127/+0.787pp | 428/+0.701pp | 353/+0.567pp | +0.661pp | 8-2 | 0.1094 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top10_mass>=0.641028&second_mass_best>=0.380599 | +0.000pp | 217/+0.461pp | 640/+0.625pp | 588/+0.510pp | +0.554pp | 11-3 | 0.0574 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top3_mass>=0.350372&second_mass_best>=0.380599 | -0.024pp | 219/+0.457pp | 661/+0.605pp | 598/+0.502pp | +0.541pp | 11-3 | 0.0574 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top5_mass>=0.471047&second_mass_best>=0.380599 | -0.024pp | 220/+0.455pp | 670/+0.597pp | 601/+0.499pp | +0.537pp | 11-3 | 0.0574 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | entropy_inv>=0.336699&second_mass_best>=0.380599 | -0.025pp | 226/+0.442pp | 649/+0.616pp | 587/+0.511pp | +0.547pp | 11-3 | 0.0574 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | gap15>=3.26087&second_mass_best>=0.380599 | -0.117pp | 139/+0.719pp | 510/+0.784pp | 484/+0.413pp | +0.618pp | 7-0 | 0.0156 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | first_mass_best>=0.723268&second_mass_best>=0.380599 | -0.180pp | 177/+0.565pp | 485/+0.412pp | 428/+0.467pp | +0.459pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | second_mass_best>=0.380599 | +0.000pp | 248/+0.403pp | 730/+0.548pp | 655/+0.458pp | +0.490pp | 12-4 | 0.0768 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | second_mass_best>=0.380599&top5_same_first>=0 | +0.000pp | 248/+0.403pp | 730/+0.548pp | 655/+0.458pp | +0.490pp | 12-4 | 0.0768 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | second_mass_best>=0.380599&top3_same_pair>=0 | +0.000pp | 248/+0.403pp | 730/+0.548pp | 655/+0.458pp | +0.490pp | 12-4 | 0.0768 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | gap15>=2.97826&second_mass_best>=0.380599 | -0.024pp | 169/+0.592pp | 565/+0.708pp | 537/+0.372pp | +0.551pp | 8-1 | 0.0391 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | gap110>=6&second_mass_best>=0.380599 | -0.099pp | 192/+0.521pp | 657/+0.609pp | 595/+0.336pp | +0.485pp | 10-3 | 0.0923 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | first_mass_best>=0.668044&second_mass_best>=0.380599 | -0.102pp | 205/+0.488pp | 601/+0.333pp | 511/+0.391pp | +0.380pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | second_mass_best>=0.380599&pair12_mass_best>=0.348628 | +0.000pp | 239/+0.418pp | 712/+0.562pp | 633/+0.316pp | +0.442pp | 11-4 | 0.1185 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top10_mass>=0.641028&second_mass_best>=0.380599 | -0.024pp | 217/+0.461pp | 640/+0.312pp | 588/+0.510pp | +0.415pp | 9-3 | 0.1460 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | entropy_inv>=0.336699&second_mass_best>=0.380599 | -0.049pp | 226/+0.442pp | 649/+0.308pp | 587/+0.511pp | +0.410pp | 9-3 | 0.1460 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | entropy_inv>=0.428239&second_mass_best>=0.380599 | -0.051pp | 71/+1.408pp | 308/+0.974pp | 327/+0.306pp | +0.708pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | gap110>=6&second_mass_best>=0.380599 | -0.099pp | 192/+0.521pp | 657/+0.304pp | 595/+0.336pp | +0.346pp | 8-3 | 0.2266 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top10_mass>=0.755812&second_mass_best>=0.380599 | -0.052pp | 69/+1.449pp | 306/+0.980pp | 329/+0.304pp | +0.710pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top3_mass>=0.350372&second_mass_best>=0.380599 | -0.048pp | 219/+0.457pp | 661/+0.303pp | 598/+0.502pp | +0.406pp | 9-3 | 0.1460 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top5_mass>=0.601577&second_mass_best>=0.380599 | -0.103pp | 62/+1.613pp | 305/+0.984pp | 331/+0.302pp | +0.716pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top5_mass>=0.471047&second_mass_best>=0.380599 | -0.048pp | 220/+0.455pp | 670/+0.299pp | 601/+0.499pp | +0.402pp | 9-3 | 0.1460 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599&pair12_mass_best>=0.348628 | -0.041pp | 239/+0.418pp | 712/+0.281pp | 633/+0.316pp | +0.316pp | 9-4 | 0.2668 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599 | -0.040pp | 248/+0.403pp | 730/+0.274pp | 655/+0.458pp | +0.367pp | 10-4 | 0.1796 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599&top5_same_first>=0 | -0.040pp | 248/+0.403pp | 730/+0.274pp | 655/+0.458pp | +0.367pp | 10-4 | 0.1796 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599&top3_same_pair>=0 | -0.040pp | 248/+0.403pp | 730/+0.274pp | 655/+0.458pp | +0.367pp | 10-4 | 0.1796 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | gap15>=3.85558&second_mass_best>=0.380599 | +0.000pp | 93/+1.075pp | 380/+0.789pp | 375/+0.267pp | +0.590pp | 5-0 | 0.0625 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | third_mass_best>=0.280966&pair_gap>=1.66741 | +0.079pp | 128/+0.781pp | 379/+0.528pp | 378/+0.265pp | +0.452pp | 4-0 | 0.1250 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top10_mass>=0.717127&second_mass_best>=0.380599 | +0.036pp | 116/+0.862pp | 435/+1.149pp | 428/+0.234pp | +0.715pp | 9-2 | 0.0654 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top10_mass>=0.717127&second_mass_best>=0.380599 | +0.000pp | 116/+0.862pp | 435/+0.690pp | 428/+0.234pp | +0.511pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | entropy_inv>=0.395208&second_mass_best>=0.380599 | +0.000pp | 118/+0.847pp | 438/+1.142pp | 429/+0.233pp | +0.711pp | 9-2 | 0.0654 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | entropy_inv>=0.395208&second_mass_best>=0.380599 | -0.072pp | 118/+0.847pp | 438/+0.685pp | 429/+0.233pp | +0.508pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top3_mass>=0.431695&second_mass_best>=0.380599 | -0.071pp | 97/+1.031pp | 429/+0.699pp | 436/+0.229pp | +0.520pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | third_mass_best>=0.263285&pair_gap>=1.66741 | +0.067pp | 157/+0.637pp | 474/+0.633pp | 439/+0.228pp | +0.467pp | 5-0 | 0.0625 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top5_mass>=0.554395&second_mass_best>=0.380599 | -0.036pp | 103/+0.971pp | 446/+1.121pp | 450/+0.222pp | +0.701pp | 9-2 | 0.0654 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top5_mass>=0.554395&second_mass_best>=0.380599 | -0.072pp | 103/+0.971pp | 446/+0.673pp | 450/+0.222pp | +0.501pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599&first_gap>=5.02392 | -0.131pp | 158/+0.633pp | 461/+0.217pp | 406/+0.493pp | +0.390pp | 6-2 | 0.2891 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599&third_mass_best>=0.280966 | +0.000pp | 146/+0.685pp | 467/+0.642pp | 462/+0.216pp | +0.465pp | 5-0 | 0.0625 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | second_mass_best>=0.380599&third_mass_best>=0.280966 | +0.062pp | 146/+0.685pp | 467/+0.642pp | 462/+0.216pp | +0.465pp | 5-0 | 0.0625 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top10_mass>=0.641028&pair_gap>=1.66741 | -0.031pp | 179/+0.559pp | 515/+0.583pp | 469/+0.213pp | +0.430pp | 8-3 | 0.2266 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | entropy_inv>=0.336699&pair_gap>=1.66741 | -0.063pp | 192/+0.521pp | 530/+0.566pp | 471/+0.212pp | +0.419pp | 8-3 | 0.2266 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top3_mass>=0.350372&pair_gap>=1.66741 | -0.061pp | 178/+0.562pp | 530/+0.566pp | 477/+0.210pp | +0.422pp | 8-3 | 0.2266 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top5_mass>=0.471047&pair_gap>=1.66741 | -0.062pp | 178/+0.562pp | 534/+0.562pp | 479/+0.209pp | +0.420pp | 8-3 | 0.2266 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair12_mass_best>=0.348628&pair_gap>=1.66741 | -0.025pp | 202/+0.495pp | 588/+0.510pp | 506/+0.198pp | +0.386pp | 9-4 | 0.2668 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | second_mass_best>=0.380599&pair_gap>=1.66741 | -0.050pp | 199/+0.503pp | 585/+0.513pp | 516/+0.194pp | +0.385pp | 9-4 | 0.2668 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top10_mass>=0.679725&second_mass_best>=0.380599 | +0.000pp | 168/+0.595pp | 549/+0.729pp | 517/+0.193pp | +0.486pp | 9-3 | 0.1460 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top10_mass>=0.679725&second_mass_best>=0.380599 | -0.028pp | 168/+0.595pp | 549/+0.364pp | 517/+0.193pp | +0.324pp | 7-3 | 0.3437 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | entropy_inv>=0.36612&second_mass_best>=0.380599 | +0.000pp | 177/+0.565pp | 561/+0.713pp | 519/+0.193pp | +0.477pp | 9-3 | 0.1460 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | entropy_inv>=0.36612&second_mass_best>=0.380599 | -0.028pp | 177/+0.565pp | 561/+0.357pp | 519/+0.193pp | +0.318pp | 7-3 | 0.3437 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599&best_odds<=4.2 | -0.027pp | 166/+0.602pp | 572/+0.524pp | 534/+0.187pp | +0.393pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top3_mass>=0.389747&second_mass_best>=0.380599 | -0.028pp | 153/+0.654pp | 558/+0.896pp | 535/+0.187pp | +0.562pp | 9-2 | 0.0654 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top3_mass>=0.389747&second_mass_best>=0.380599 | -0.056pp | 153/+0.654pp | 558/+0.538pp | 535/+0.187pp | +0.401pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top5_mass>=0.512852&second_mass_best>=0.380599 | +0.000pp | 163/+0.613pp | 569/+0.703pp | 537/+0.186pp | +0.473pp | 9-3 | 0.1460 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top5_mass>=0.512852&second_mass_best>=0.380599 | -0.028pp | 163/+0.613pp | 569/+0.351pp | 537/+0.186pp | +0.315pp | 7-3 | 0.3437 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599&first_gap>=3.79183 | -0.111pp | 196/+0.510pp | 553/+0.181pp | 478/+0.628pp | +0.407pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=1.66741&top3_same_first>=1 | -0.023pp | 265/+0.377pp | 691/+0.289pp | 558/+0.179pp | +0.264pp | 11-7 | 0.4807 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | top_prob>=0.149916&second_mass_best>=0.380599 | -0.024pp | 209/+0.478pp | 661/+0.756pp | 595/+0.168pp | +0.478pp | 9-2 | 0.0654 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | top_prob>=0.149916&second_mass_best>=0.380599 | -0.048pp | 209/+0.478pp | 661/+0.454pp | 595/+0.168pp | +0.341pp | 7-2 | 0.1797 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | second_mass_best>=0.380599&best_odds<=4.8 | -0.024pp | 211/+0.474pp | 665/+0.602pp | 597/+0.168pp | +0.407pp | 9-3 | 0.1460 |
| WATCH_DIRECTIONAL_LIFT | xdom_drug_scaffold_hop | second_mass_best>=0.380599&best_odds<=4.8 | -0.048pp | 211/+0.474pp | 665/+0.301pp | 597/+0.168pp | +0.272pp | 7-3 | 0.3437 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=1.66741 | +0.020pp | 305/+0.656pp | 811/+0.247pp | 661/+0.151pp | +0.281pp | 12-7 | 0.3593 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=1.66741&top5_same_first>=0 | +0.020pp | 305/+0.656pp | 811/+0.247pp | 661/+0.151pp | +0.281pp | 12-7 | 0.3593 |
| WATCH_DIRECTIONAL_LIFT | xdom_bradley_terry_position | pair_gap>=1.66741&top3_same_pair>=0 | +0.020pp | 305/+0.656pp | 811/+0.247pp | 661/+0.151pp | +0.281pp | 12-7 | 0.3593 |

## 50% absolute hit candidates


| status | method | rule | train_n | train_hit | n24 | hit24 | n25 | hit25 | n26 | hit26 | holdout_n | holdout_hit | worst | min_year_n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WATCH_LOW_SAMPLE_50 | board_min | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | pair_mass | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | first_pair_chain | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_pair | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_first_pair | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_position | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_annealing_escape | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_scaffold_hop | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_clonal_selection_amplify | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_ecology_predator_prey | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bandit_explore_exploit | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bradley_terry_position | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_harville_order_flow | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_information_bottleneck | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_particle_filter_resample | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 | 5 |
| WATCH_LOW_SAMPLE_50 | board_min | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | pair_mass | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | first_pair_chain | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_pair | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_first_pair | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_position | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_annealing_escape | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_scaffold_hop | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_clonal_selection_amplify | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_ecology_predator_prey | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bandit_explore_exploit | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bradley_terry_position | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_harville_order_flow | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_information_bottleneck | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_particle_filter_resample | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 | 5 |
| WATCH_LOW_SAMPLE_50 | board_min | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | pair_mass | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | first_pair_chain | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_pair | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_first_pair | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_position | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_annealing_escape | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_scaffold_hop | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_clonal_selection_amplify | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_ecology_predator_prey | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bandit_explore_exploit | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bradley_terry_position | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_harville_order_flow | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_information_bottleneck | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_particle_filter_resample | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 | 5 |
| WATCH_LOW_SAMPLE_50 | board_min | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | board_min | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | pair_mass | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | pair_mass | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | first_pair_chain | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | first_pair_chain | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_pair | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_pair | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_first_pair | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_first_pair | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_position | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | energy_position | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_annealing_escape | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_annealing_escape | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_scaffold_hop | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_scaffold_hop | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_clonal_selection_amplify | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_clonal_selection_amplify | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_ecology_predator_prey | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_ecology_predator_prey | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bandit_explore_exploit | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bandit_explore_exploit | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bradley_terry_position | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_bradley_terry_position | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
| WATCH_LOW_SAMPLE_50 | xdom_harville_order_flow | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 | 5 |
