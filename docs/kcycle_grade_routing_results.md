# KCYCLE grade routing experiment

records: 13900
grade_counts: {"선발": 5372, "우수": 4091, "특선": 4437}
train_route: {"선발": "xdom_drug_scaffold_hop", "우수": "board_min", "특선": "energy_position"}
train_top1_route: {"선발": "xdom_clonal_selection_amplify", "우수": "first_pair_chain", "특선": "first_pair_chain"}
selected_policy: baseline
selected_top1_policy: grade_route_train_top1_best
deployable: False
top1_candidate: True

## Test leaderboard

| name | exact | top1 | board_exact | board_top1 | exact_lift_pp | top1_lift_pp | route |
|---|---:|---:|---:|---:|---:|---:|---|
| grade_route_train_top1_best | 0.1763 | 0.6358 | 0.1720 | 0.6344 | +0.434 | +0.145 | {"선발": "xdom_clonal_selection_amplify", "우수": "first_pair_chain", "특선": "first_pair_chain"} |
| single_xdom_ecology_predator_prey | 0.1756 | 0.6395 | 0.1720 | 0.6344 | +0.361 | +0.506 | {"선발": "xdom_ecology_predator_prey", "우수": "xdom_ecology_predator_prey", "특선": "xdom_ecology_predator_prey"} |
| single_xdom_bayesian_surrogate_focus | 0.1756 | 0.6373 | 0.1720 | 0.6344 | +0.361 | +0.289 | {"선발": "xdom_bayesian_surrogate_focus", "우수": "xdom_bayesian_surrogate_focus", "특선": "xdom_bayesian_surrogate_focus"} |
| single_xdom_drug_scaffold_hop | 0.1756 | 0.6322 | 0.1720 | 0.6344 | +0.361 | -0.217 | {"선발": "xdom_drug_scaffold_hop", "우수": "xdom_drug_scaffold_hop", "특선": "xdom_drug_scaffold_hop"} |
| single_xdom_clonal_selection_amplify | 0.1749 | 0.6366 | 0.1720 | 0.6344 | +0.289 | +0.217 | {"선발": "xdom_clonal_selection_amplify", "우수": "xdom_clonal_selection_amplify", "특선": "xdom_clonal_selection_amplify"} |
| single_xdom_bradley_terry_position | 0.1749 | 0.6366 | 0.1720 | 0.6344 | +0.289 | +0.217 | {"선발": "xdom_bradley_terry_position", "우수": "xdom_bradley_terry_position", "특선": "xdom_bradley_terry_position"} |
| grade_route_val_top1_oracle | 0.1741 | 0.6387 | 0.1720 | 0.6344 | +0.217 | +0.434 | {"선발": "first_pair_chain", "우수": "first_pair_chain", "특선": "xdom_particle_filter_resample"} |
| single_pair_mass | 0.1741 | 0.6322 | 0.1720 | 0.6344 | +0.217 | -0.217 | {"선발": "pair_mass", "우수": "pair_mass", "특선": "pair_mass"} |
| single_first_pair_chain | 0.1734 | 0.6395 | 0.1720 | 0.6344 | +0.145 | +0.506 | {"선발": "first_pair_chain", "우수": "first_pair_chain", "특선": "first_pair_chain"} |
| single_xdom_information_bottleneck | 0.1734 | 0.6366 | 0.1720 | 0.6344 | +0.145 | +0.217 | {"선발": "xdom_information_bottleneck", "우수": "xdom_information_bottleneck", "특선": "xdom_information_bottleneck"} |
| single_xdom_bandit_explore_exploit | 0.1734 | 0.6358 | 0.1720 | 0.6344 | +0.145 | +0.145 | {"선발": "xdom_bandit_explore_exploit", "우수": "xdom_bandit_explore_exploit", "특선": "xdom_bandit_explore_exploit"} |
| single_xdom_particle_filter_resample | 0.1727 | 0.6373 | 0.1720 | 0.6344 | +0.072 | +0.289 | {"선발": "xdom_particle_filter_resample", "우수": "xdom_particle_filter_resample", "특선": "xdom_particle_filter_resample"} |
