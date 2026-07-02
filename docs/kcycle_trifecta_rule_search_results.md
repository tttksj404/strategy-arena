# KCYCLE trifecta rule search

records: 13900 train: ['2018', '2019', '2020', '2021', '2022', '2023'] holdout: ['2024', '2025', '2026']
predicate_count: 7883 evaluated: 61399 pass_count: 117
xdom_methods: ['xdom_annealing_escape', 'xdom_drug_scaffold_hop', 'xdom_drug_multi_objective_funnel', 'xdom_clonal_selection_amplify', 'xdom_ecology_predator_prey', 'xdom_bandit_explore_exploit', 'xdom_bayesian_surrogate_focus']
xdom_predicate_count: 7 xdom_pass_count: 63
pass_count_by_min_year_n: {'5': 117, '10': 0, '20': 0, '30': 0, '50': 0, '100': 0}

Interpretation: xdom recipes are evaluated in the KCYCLE trifecta harness, but current 50%+ holdout passes remain low-sample strong-favorite slices. No candidate passes with min yearly holdout n >= 10.

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

| status | method | rule | train_n | train_hit | n24 | hit24 | n25 | hit25 | n26 | hit26 | holdout_n | holdout_hit | worst |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PASS_HOLDOUT_50 | board_min | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | pair_mass | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | first_pair_chain | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | energy_pair | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | energy_first_pair | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | energy_position | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | xdom_annealing_escape | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | xdom_drug_scaffold_hop | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | xdom_clonal_selection_amplify | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | xdom_ecology_predator_prey | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | xdom_bandit_explore_exploit | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&top3_same_first>=1 | 199 | 0.3970 | 5 | 0.8000 | 37 | 0.5946 | 16 | 0.6250 | 58 | 0.6207 | 0.5946 |
| PASS_HOLDOUT_50 | board_min | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | pair_mass | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | first_pair_chain | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | energy_pair | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | energy_first_pair | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | energy_position | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | xdom_annealing_escape | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | xdom_drug_scaffold_hop | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | xdom_clonal_selection_amplify | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | xdom_ecology_predator_prey | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | xdom_bandit_explore_exploit | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&first_gap>=3.79183 | 192 | 0.4271 | 5 | 0.8000 | 34 | 0.6176 | 17 | 0.5882 | 56 | 0.6250 | 0.5882 |
| PASS_HOLDOUT_50 | board_min | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | pair_mass | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | first_pair_chain | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | energy_pair | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | energy_first_pair | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | energy_position | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | xdom_annealing_escape | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | xdom_drug_scaffold_hop | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | xdom_clonal_selection_amplify | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | xdom_ecology_predator_prey | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | xdom_bandit_explore_exploit | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&first_mass_best>=0.723268 | 201 | 0.4279 | 5 | 0.8000 | 35 | 0.5714 | 16 | 0.5625 | 56 | 0.5893 | 0.5625 |
| PASS_HOLDOUT_50 | board_min | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | board_min | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | pair_mass | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | pair_mass | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | first_pair_chain | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | first_pair_chain | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | energy_pair | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | energy_pair | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | energy_first_pair | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | energy_first_pair | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | energy_position | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | energy_position | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_annealing_escape | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_annealing_escape | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_drug_scaffold_hop | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_drug_scaffold_hop | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_clonal_selection_amplify | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_clonal_selection_amplify | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_ecology_predator_prey | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_ecology_predator_prey | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_bandit_explore_exploit | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_bandit_explore_exploit | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&first_mass_best>=0.668044 | 240 | 0.4000 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&second_mass_best>=0.486203 | 245 | 0.4082 | 5 | 0.8000 | 43 | 0.5349 | 19 | 0.5263 | 67 | 0.5522 | 0.5263 |
| PASS_HOLDOUT_50 | board_min | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | pair_mass | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | first_pair_chain | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | energy_pair | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | energy_first_pair | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | energy_position | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | xdom_annealing_escape | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | xdom_drug_scaffold_hop | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | xdom_drug_multi_objective_funnel | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | xdom_clonal_selection_amplify | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | xdom_ecology_predator_prey | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | xdom_bandit_explore_exploit | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | xdom_bayesian_surrogate_focus | gap12>=2.66667&pair12_mass_best>=0.468382 | 242 | 0.4091 | 5 | 0.8000 | 44 | 0.5227 | 18 | 0.5556 | 67 | 0.5522 | 0.5227 |
| PASS_HOLDOUT_50 | board_min | gap12>=2.66667&top3_mass>=0.480963 | 220 | 0.4045 | 5 | 0.8000 | 39 | 0.5385 | 18 | 0.5000 | 62 | 0.5484 | 0.5000 |
| PASS_HOLDOUT_50 | pair_mass | gap12>=2.66667&top3_mass>=0.480963 | 220 | 0.4045 | 5 | 0.8000 | 39 | 0.5385 | 18 | 0.5000 | 62 | 0.5484 | 0.5000 |
