# KCYCLE fast-evolution trifecta search

records: 13900
top_k: 40
candidates_per_generation: 6000
generations: 3
elapsed_sec: 21.992
deployable_count: 0
strict_deployable_count: 0

selection: validation-only rank, then test is reported as untouched OOS evidence.

| candidate | deployable | strict | val exact/lift | test exact/lift | test top1/lift | formula |
|---|---:|---:|---:|---:|---:|---|
| gen3_mut_3146_guard_r5_m0.00 | false | false | 0.1609 / +1.070pp | 0.1597 / -1.228pp | 0.6351 / +0.072pp | `+0.419*gap110 +0.393*neg_log_odds +0.291*log_q -0.291*first_gap -0.287*gap15 -0.265*second_mass -0.244*top3_same_pair -0.240*first_mass` |
| gen3_mut_2432_guard_r5_m0.00 | false | false | 0.1601 / +0.991pp | 0.1590 / -1.301pp | 0.6366 / +0.217pp | `+0.539*neg_log_odds +0.350*gap110 +0.339*third_share +0.337*log_q +0.273*top3_same_first -0.239*first_gap -0.208*rank_score -0.191*second_mass` |
| gen3_mut_2432_guard_r8_m0.00 | false | false | 0.1601 / +0.991pp | 0.1590 / -1.301pp | 0.6366 / +0.217pp | `+0.539*neg_log_odds +0.350*gap110 +0.339*third_share +0.337*log_q +0.273*top3_same_first -0.239*first_gap -0.208*rank_score -0.191*second_mass` |
| gen3_mut_3146_guard_r8_m0.00 | false | false | 0.1609 / +1.070pp | 0.1597 / -1.228pp | 0.6351 / +0.072pp | `+0.419*gap110 +0.393*neg_log_odds +0.291*log_q -0.291*first_gap -0.287*gap15 -0.265*second_mass -0.244*top3_same_pair -0.240*first_mass` |
| gen3_mut_3503_guard_r5_m0.00 | false | false | 0.1601 / +0.991pp | 0.1626 / -0.939pp | 0.6380 / +0.361pp | `+0.720*neg_log_odds -0.305*rank_score +0.253*third_share +0.247*log_q -0.209*second_mass -0.198*first_mass +0.186*gap110 -0.162*pair_gap` |
| gen3_mut_3503_guard_r8_m0.00 | false | false | 0.1601 / +0.991pp | 0.1626 / -0.939pp | 0.6380 / +0.361pp | `+0.720*neg_log_odds -0.305*rank_score +0.253*third_share +0.247*log_q -0.209*second_mass -0.198*first_mass +0.186*gap110 -0.162*pair_gap` |
| gen3_mut_2084_guard_r5_m0.00 | false | false | 0.1601 / +0.991pp | 0.1626 / -0.939pp | 0.6373 / +0.289pp | `+0.794*neg_log_odds +0.271*entropy_inv +0.210*top3_same_first -0.204*rank_score +0.190*neg_odds_ratio_best -0.181*second_mass +0.173*log_q +0.155*third_share` |
| gen3_mut_5099_guard_r5_m0.00 | false | false | 0.1601 / +0.991pp | 0.1640 / -0.795pp | 0.6358 / +0.145pp | `+0.492*log_q +0.458*entropy_inv +0.376*neg_log_odds +0.302*gap110 -0.239*rank_score +0.233*top3_same_first -0.219*first_gap +0.206*third_share` |
| gen3_mut_2084_guard_r8_m0.00 | false | false | 0.1601 / +0.991pp | 0.1626 / -0.939pp | 0.6373 / +0.289pp | `+0.794*neg_log_odds +0.271*entropy_inv +0.210*top3_same_first -0.204*rank_score +0.190*neg_odds_ratio_best -0.181*second_mass +0.173*log_q +0.155*third_share` |
| gen3_mut_5099_guard_r8_m0.00 | false | false | 0.1601 / +0.991pp | 0.1640 / -0.795pp | 0.6358 / +0.145pp | `+0.492*log_q +0.458*entropy_inv +0.376*neg_log_odds +0.302*gap110 -0.239*rank_score +0.233*top3_same_first -0.219*first_gap +0.206*third_share` |
| gen3_mut_3845_guard_r5_m0.00 | false | false | 0.1597 / +0.951pp | 0.1655 / -0.650pp | 0.6387 / +0.434pp | `+0.749*entropy_inv +0.537*neg_log_odds +0.206*log_q -0.206*rank_score +0.140*neg_odds_ratio_best -0.124*second_mass +0.104*third_share +0.086*pair_gap` |
| gen3_mut_3845_guard_r8_m0.00 | false | false | 0.1597 / +0.951pp | 0.1655 / -0.650pp | 0.6387 / +0.434pp | `+0.749*entropy_inv +0.537*neg_log_odds +0.206*log_q -0.206*rank_score +0.140*neg_odds_ratio_best -0.124*second_mass +0.104*third_share +0.086*pair_gap` |
| gen3_mut_3098_guard_r5_m0.00 | false | false | 0.1601 / +0.991pp | 0.1633 / -0.867pp | 0.6380 / +0.361pp | `+0.432*log_q +0.387*gap110 +0.372*neg_log_odds +0.285*top3_same_first -0.268*first_gap -0.244*second_mass -0.221*first_mass -0.221*top3_same_pair` |
| gen3_mut_930_guard_r5_m0.00 | false | false | 0.1601 / +0.991pp | 0.1611 / -1.084pp | 0.6380 / +0.361pp | `+0.812*neg_log_odds -0.283*rank_score +0.212*neg_odds_ratio_best +0.206*third_share -0.200*second_mass +0.183*gap110 +0.173*log_q -0.159*first_mass` |
| gen3_mut_3098_guard_r8_m0.00 | false | false | 0.1601 / +0.991pp | 0.1633 / -0.867pp | 0.6380 / +0.361pp | `+0.432*log_q +0.387*gap110 +0.372*neg_log_odds +0.285*top3_same_first -0.268*first_gap -0.244*second_mass -0.221*first_mass -0.221*top3_same_pair` |
| gen3_mut_930_guard_r8_m0.00 | false | false | 0.1601 / +0.991pp | 0.1611 / -1.084pp | 0.6380 / +0.361pp | `+0.812*neg_log_odds -0.283*rank_score +0.212*neg_odds_ratio_best +0.206*third_share -0.200*second_mass +0.183*gap110 +0.173*log_q -0.159*first_mass` |
| gen3_mut_1219_guard_r5_m0.00 | false | false | 0.1601 / +0.991pp | 0.1626 / -0.939pp | 0.6380 / +0.361pp | `+0.876*neg_log_odds -0.282*rank_score +0.242*neg_odds_ratio_best +0.180*third_share -0.161*second_mass -0.090*first_mass +0.081*gap110 +0.071*pair_share` |
| gen3_mut_2073_guard_r5_m0.00 | false | false | 0.1601 / +0.991pp | 0.1582 / -1.373pp | 0.6366 / +0.217pp | `+0.464*log_q +0.432*gap110 +0.374*top3_same_first +0.319*neg_log_odds +0.269*third_share -0.236*top3_same_pair -0.236*second_mass -0.194*rank_score` |
| gen3_mut_1219_guard_r8_m0.00 | false | false | 0.1601 / +0.991pp | 0.1626 / -0.939pp | 0.6380 / +0.361pp | `+0.876*neg_log_odds -0.282*rank_score +0.242*neg_odds_ratio_best +0.180*third_share -0.161*second_mass -0.090*first_mass +0.081*gap110 +0.071*pair_share` |
| gen3_mut_5176_guard_r8_m0.00 | false | false | 0.1601 / +0.991pp | 0.1618 / -1.012pp | 0.6380 / +0.361pp | `+0.446*log_q +0.378*gap110 +0.374*neg_log_odds +0.294*top3_same_first -0.277*first_gap -0.252*second_mass -0.232*top3_same_pair +0.225*third_share` |
| gen3_mut_2084_guard_r3_m0.00 | false | false | 0.1597 / +0.951pp | 0.1626 / -0.939pp | 0.6387 / +0.434pp | `+0.794*neg_log_odds +0.271*entropy_inv +0.210*top3_same_first -0.204*rank_score +0.190*neg_odds_ratio_best -0.181*second_mass +0.173*log_q +0.155*third_share` |
| gen3_mut_5099_guard_r3_m0.00 | false | false | 0.1597 / +0.951pp | 0.1640 / -0.795pp | 0.6366 / +0.217pp | `+0.492*log_q +0.458*entropy_inv +0.376*neg_log_odds +0.302*gap110 -0.239*rank_score +0.233*top3_same_first -0.219*first_gap +0.206*third_share` |
| gen3_mut_3845_guard_r3_m0.00 | false | false | 0.1593 / +0.912pp | 0.1655 / -0.650pp | 0.6387 / +0.434pp | `+0.749*entropy_inv +0.537*neg_log_odds +0.206*log_q -0.206*rank_score +0.140*neg_odds_ratio_best -0.124*second_mass +0.104*third_share +0.086*pair_gap` |
| gen3_mut_3503_guard_r3_m0.00 | false | false | 0.1597 / +0.951pp | 0.1626 / -0.939pp | 0.6387 / +0.434pp | `+0.720*neg_log_odds -0.305*rank_score +0.253*third_share +0.247*log_q -0.209*second_mass -0.198*first_mass +0.186*gap110 -0.162*pair_gap` |
| gen3_mut_930_guard_r3_m0.00 | false | false | 0.1597 / +0.951pp | 0.1611 / -1.084pp | 0.6387 / +0.434pp | `+0.812*neg_log_odds -0.283*rank_score +0.212*neg_odds_ratio_best +0.206*third_share -0.200*second_mass +0.183*gap110 +0.173*log_q -0.159*first_mass` |
| gen3_mut_1219_guard_r3_m0.00 | false | false | 0.1597 / +0.951pp | 0.1626 / -0.939pp | 0.6380 / +0.361pp | `+0.876*neg_log_odds -0.282*rank_score +0.242*neg_odds_ratio_best +0.180*third_share -0.161*second_mass -0.090*first_mass +0.081*gap110 +0.071*pair_share` |
| gen3_mut_5176_guard_r3_m0.00 | false | false | 0.1597 / +0.951pp | 0.1618 / -1.012pp | 0.6380 / +0.361pp | `+0.446*log_q +0.378*gap110 +0.374*neg_log_odds +0.294*top3_same_first -0.277*first_gap -0.252*second_mass -0.232*top3_same_pair +0.225*third_share` |
| gen3_mut_3673_guard_r3_m0.00 | false | false | 0.1597 / +0.951pp | 0.1618 / -1.012pp | 0.6387 / +0.434pp | `+0.681*neg_log_odds +0.366*log_q -0.305*rank_score +0.275*gap110 +0.214*third_share -0.178*first_mass -0.177*first_gap -0.155*gap12` |
| gen3_mut_194_guard_r3_m0.00 | false | false | 0.1593 / +0.912pp | 0.1626 / -0.939pp | 0.6395 / +0.506pp | `+0.873*neg_log_odds -0.361*rank_score +0.202*neg_odds_ratio_best -0.168*second_mass +0.146*third_share -0.073*first_mass +0.069*log_q +0.056*gap110` |
| gen3_mut_2179_guard_r3_m0.00 | false | false | 0.1593 / +0.912pp | 0.1618 / -1.012pp | 0.6402 / +0.578pp | `+0.453*third_share +0.432*neg_log_odds +0.429*log_q +0.286*gap110 -0.256*first_gap +0.254*top3_same_first -0.233*rank_score -0.216*second_mass` |
