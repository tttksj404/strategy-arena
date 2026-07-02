# KCYCLE trifecta rule search

records: 13900 train: ['2018', '2019', '2020', '2021', '2022', '2023'] holdout: ['2024', '2025', '2026']
promotion_rules: {'min_train_n': 50, 'min_eval_holdout_year_n': 5, 'min_promote_holdout_year_n': 10, 'min_worst_year_hit': 0.5}
predicate_count: 7887 evaluated: 80359
xdom_methods: ['xdom_annealing_escape', 'xdom_drug_scaffold_hop', 'xdom_drug_multi_objective_funnel', 'xdom_clonal_selection_amplify', 'xdom_ecology_predator_prey', 'xdom_bandit_explore_exploit', 'xdom_bayesian_surrogate_focus', 'xdom_bradley_terry_position', 'xdom_harville_order_flow', 'xdom_information_bottleneck', 'xdom_particle_filter_resample']
fifty_watch_or_promote_count: 153 promotion_count: 0
xdom_fifty_watch_or_promote_count: 99 xdom_promotion_count: 0
deduped_fifty_watch_or_promote_count: 9 deduped_xdom_fifty_watch_or_promote_count: 9
pass_count_by_min_year_n: {'5': 153, '10': 0, '20': 0, '30': 0, '50': 0, '100': 0}
risk_flags: {'no_robust_promotion': True, 'low_sample_watch_only': True, 'xdom_duplicate_inflation': True, 'requires_more_outcome_linked_snapshots': True}

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
