# KCYCLE Top2 Pair Experiment

- races: 13900
- baseline: best_combo_top2
- deployable candidates: 0

## Top Methods

| method | holdout n | top2 slot | unordered pair | ordered pair | holdout lift pp | worst year lift pp |
|---|---:|---:|---:|---:|---:|---:|
| router_pair_if_gap23_le_0.04 | 3907 | 0.6799 | 0.4287 | 0.3281 | 1.049 | -0.909 |
| router_pair_if_gap23_le_0.0025 | 3907 | 0.6799 | 0.4244 | 0.3256 | 1.049 | -0.844 |
| router_pair_if_entropy_ge_0.9 | 3907 | 0.6798 | 0.4239 | 0.3253 | 1.037 | -0.714 |
| router_pair_if_entropy_ge_0.94 | 3907 | 0.6798 | 0.4239 | 0.3253 | 1.037 | -0.714 |
| slot_mass_top2 | 3907 | 0.6798 | 0.4239 | 0.3253 | 1.037 | -0.714 |
| router_pair_if_gap23_le_0.02 | 3907 | 0.6797 | 0.4264 | 0.3266 | 1.024 | -1.039 |
| router_pair_if_gap23_le_0.01 | 3907 | 0.6797 | 0.4254 | 0.3258 | 1.024 | -0.909 |
| router_pair_if_gap23_le_0.005 | 3907 | 0.6797 | 0.4251 | 0.3258 | 1.024 | -0.844 |
| router_pair_if_entropy_ge_0.86 | 3907 | 0.6795 | 0.4236 | 0.3248 | 1.011 | -0.649 |
| router_pair_if_entropy_ge_0.82 | 3907 | 0.6793 | 0.4239 | 0.3245 | 0.985 | -0.714 |
| router_pair_if_gap23_le_0.08 | 3907 | 0.6781 | 0.4292 | 0.3284 | 0.870 | -0.584 |
| unordered_pair_mass | 3907 | 0.6760 | 0.4297 | 0.3312 | 0.653 | -0.649 |

## Decision

No deployable candidate passed the non-negative holdout-year gate.

## Hybrid Candidates

| method | switched | holdout top2 | unordered lift pp | ordered lift pp | top2 lift pp | worst year lift pp |
|---|---:|---:|---:|---:|---:|---:|
| hybrid_slot_mass_top2__slot_top1_le_0.799966 | 4810 | 0.6794 | 0.307 | -0.179 | 0.998 | 0.519 |
| hybrid_slot_mass_top2__slot_gap12_le_0.385481 | 10955 | 0.6792 | 0.051 | -0.563 | 0.973 | 0.065 |
| hybrid_slot_mass_top2__entropy_ge_0.663301 | 6934 | 0.6785 | -0.026 | -0.410 | 0.909 | 0.130 |
| hybrid_slot_mass_top2__slot_gap12_le_0.317279 | 9098 | 0.6778 | 0.102 | -0.384 | 0.832 | 0.325 |
| hybrid_slot_mass_top2__slot_gap12_le_0.259207 | 6896 | 0.6772 | 0.230 | -0.230 | 0.781 | 0.260 |
| hybrid_slot_mass_top2__slot_top1_le_0.737603 | 2934 | 0.6764 | -0.128 | -0.358 | 0.691 | 0.542 |
| hybrid_unordered_pair_mass__slot_gap12_le_0.385481 | 10955 | 0.6756 | 0.486 | -0.128 | 0.614 | 0.130 |
| hybrid_slot_mass_top2__best_odds_ge_7.3 | 2870 | 0.6756 | -0.051 | -0.205 | 0.614 | 0.390 |
| hybrid_unordered_pair_mass__slot_top1_le_0.799966 | 4810 | 0.6751 | 0.537 | -0.026 | 0.563 | 0.260 |
| hybrid_unordered_pair_mass__entropy_ge_0.663301 | 6934 | 0.6751 | 0.461 | 0.000 | 0.563 | 0.065 |
| hybrid_slot_mass_top2__slot_gap12_le_0.180746 | 4570 | 0.6748 | 0.333 | -0.051 | 0.537 | 0.195 |
| hybrid_slot_mass_top2__unordered_pair_gap12_ge_0.0459598 | 12413 | 0.6748 | 0.051 | -0.435 | 0.537 | 0.195 |
