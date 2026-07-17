# KCYCLE fast-evolution trifecta search

records: 13900
top_k: 20
candidates_per_generation: 5000
generations: 4
elapsed_sec: 20.196
deployable_count: 0
strict_deployable_count: 0

selection: validation-only rank, then test is reported as untouched OOS evidence.

| candidate | deployable | strict | val exact/lift | test exact/lift | test top1/lift | formula |
|---|---:|---:|---:|---:|---:|---|
| gen4_mut_1837_guard_r5_m0.00 | false | false | 0.1613 / +1.110pp | 0.1669 / -0.506pp | 0.6358 / +0.145pp | `+0.522*gap12 +0.493*pair_share +0.420*pair_mass +0.320*log_q -0.208*top5_same_first +0.206*neg_odds_ratio_best -0.203*rank_score +0.176*gap110` |
| gen4_mut_1837_guard_r8_m0.00 | false | false | 0.1613 / +1.110pp | 0.1669 / -0.506pp | 0.6373 / +0.289pp | `+0.522*gap12 +0.493*pair_share +0.420*pair_mass +0.320*log_q -0.208*top5_same_first +0.206*neg_odds_ratio_best -0.203*rank_score +0.176*gap110` |
| gen4_mut_1719 | false | false | 0.1609 / +1.070pp | 0.1676 / -0.434pp | 0.6387 / +0.434pp | `+0.578*gap12 +0.465*pair_share +0.404*pair_mass +0.326*log_q +0.210*neg_odds_ratio_best +0.175*gap110 +0.173*third_share -0.156*top5_same_first` |
| gen4_mut_1719_guard_r8_m0.00 | false | false | 0.1609 / +1.070pp | 0.1676 / -0.434pp | 0.6387 / +0.434pp | `+0.578*gap12 +0.465*pair_share +0.404*pair_mass +0.326*log_q +0.210*neg_odds_ratio_best +0.175*gap110 +0.173*third_share -0.156*top5_same_first` |
| gen4_mut_1719_guard_r5_m0.00 | false | false | 0.1609 / +1.070pp | 0.1676 / -0.434pp | 0.6373 / +0.289pp | `+0.578*gap12 +0.465*pair_share +0.404*pair_mass +0.326*log_q +0.210*neg_odds_ratio_best +0.175*gap110 +0.173*third_share -0.156*top5_same_first` |
| gen4_mut_4166_guard_r5_m0.00 | false | false | 0.1609 / +1.070pp | 0.1597 / -1.228pp | 0.6402 / +0.578pp | `+0.876*gap12 +0.264*first_gap +0.220*log_q -0.206*top3_same_first +0.160*gap110 -0.127*rank_score +0.109*third_share +0.083*neg_odds_ratio_best` |
| gen4_mut_1841_guard_r5_m0.00 | false | false | 0.1609 / +1.070pp | 0.1597 / -1.228pp | 0.6395 / +0.506pp | `+0.847*gap12 +0.323*first_gap +0.236*gap110 +0.216*log_q -0.204*top3_same_first -0.120*rank_score +0.102*third_share +0.069*top3_same_pair` |
| gen4_mut_4166_guard_r8_m0.00 | false | false | 0.1609 / +1.070pp | 0.1597 / -1.228pp | 0.6402 / +0.578pp | `+0.876*gap12 +0.264*first_gap +0.220*log_q -0.206*top3_same_first +0.160*gap110 -0.127*rank_score +0.109*third_share +0.083*neg_odds_ratio_best` |
| gen4_mut_1841_guard_r8_m0.00 | false | false | 0.1609 / +1.070pp | 0.1597 / -1.228pp | 0.6395 / +0.506pp | `+0.847*gap12 +0.323*first_gap +0.236*gap110 +0.216*log_q -0.204*top3_same_first -0.120*rank_score +0.102*third_share +0.069*top3_same_pair` |
| gen4_mut_4166 | false | false | 0.1609 / +1.070pp | 0.1597 / -1.228pp | 0.6402 / +0.578pp | `+0.876*gap12 +0.264*first_gap +0.220*log_q -0.206*top3_same_first +0.160*gap110 -0.127*rank_score +0.109*third_share +0.083*neg_odds_ratio_best` |
| gen4_mut_1841 | false | false | 0.1609 / +1.070pp | 0.1597 / -1.228pp | 0.6395 / +0.506pp | `+0.847*gap12 +0.323*first_gap +0.236*gap110 +0.216*log_q -0.204*top3_same_first -0.120*rank_score +0.102*third_share +0.069*top3_same_pair` |
| gen4_mut_542_guard_r5_m0.00 | false | false | 0.1613 / +1.110pp | 0.1618 / -1.012pp | 0.6351 / +0.072pp | `+0.450*log_q +0.358*pair_mass -0.328*first_mass +0.322*top3_same_first +0.305*pair_share +0.278*gap110 -0.241*top5_same_first -0.188*second_mass` |
| gen4_mut_1446 | false | false | 0.1605 / +1.031pp | 0.1618 / -1.012pp | 0.6402 / +0.578pp | `+0.836*neg_log_odds -0.358*rank_score +0.237*neg_odds_ratio_best +0.169*third_share +0.132*gap15 +0.130*pair_gap -0.101*unordered_trio_mass -0.094*gap12` |
| gen4_mut_545 | false | false | 0.1605 / +1.031pp | 0.1604 / -1.156pp | 0.6416 / +0.723pp | `+0.828*neg_log_odds -0.354*rank_score +0.234*neg_odds_ratio_best +0.186*third_share +0.170*gap15 +0.129*pair_gap -0.100*unordered_trio_mass -0.093*gap12` |
| gen4_mut_1446_guard_r5_m0.00 | false | false | 0.1605 / +1.031pp | 0.1618 / -1.012pp | 0.6402 / +0.578pp | `+0.836*neg_log_odds -0.358*rank_score +0.237*neg_odds_ratio_best +0.169*third_share +0.132*gap15 +0.130*pair_gap -0.101*unordered_trio_mass -0.094*gap12` |
| gen4_mut_545_guard_r5_m0.00 | false | false | 0.1605 / +1.031pp | 0.1604 / -1.156pp | 0.6416 / +0.723pp | `+0.828*neg_log_odds -0.354*rank_score +0.234*neg_odds_ratio_best +0.186*third_share +0.170*gap15 +0.129*pair_gap -0.100*unordered_trio_mass -0.093*gap12` |
| gen4_mut_1446_guard_r8_m0.00 | false | false | 0.1605 / +1.031pp | 0.1618 / -1.012pp | 0.6402 / +0.578pp | `+0.836*neg_log_odds -0.358*rank_score +0.237*neg_odds_ratio_best +0.169*third_share +0.132*gap15 +0.130*pair_gap -0.101*unordered_trio_mass -0.094*gap12` |
| gen4_mut_545_guard_r8_m0.00 | false | false | 0.1605 / +1.031pp | 0.1604 / -1.156pp | 0.6416 / +0.723pp | `+0.828*neg_log_odds -0.354*rank_score +0.234*neg_odds_ratio_best +0.186*third_share +0.170*gap15 +0.129*pair_gap -0.100*unordered_trio_mass -0.093*gap12` |
| gen4_mut_1371_guard_r5_m0.00 | false | false | 0.1609 / +1.070pp | 0.1655 / -0.650pp | 0.6387 / +0.434pp | `-0.414*first_gap +0.355*gap110 +0.345*top3_same_first +0.342*log_q -0.295*rank_score +0.293*neg_log_odds -0.271*second_mass +0.242*third_share` |
| gen4_mut_496 | false | false | 0.1605 / +1.031pp | 0.1611 / -1.084pp | 0.6402 / +0.578pp | `+0.834*neg_log_odds -0.357*rank_score +0.236*neg_odds_ratio_best +0.168*third_share +0.132*gap15 +0.130*pair_gap -0.101*unordered_trio_mass -0.093*gap12` |
| gen4_mut_1596 | false | false | 0.1605 / +1.031pp | 0.1597 / -1.228pp | 0.6416 / +0.723pp | `+0.830*neg_log_odds -0.334*rank_score +0.235*neg_odds_ratio_best +0.172*top3_same_pair +0.167*third_share +0.131*gap15 +0.129*pair_gap -0.101*unordered_trio_mass` |
| gen4_mut_496_guard_r5_m0.00 | false | false | 0.1605 / +1.031pp | 0.1611 / -1.084pp | 0.6402 / +0.578pp | `+0.834*neg_log_odds -0.357*rank_score +0.236*neg_odds_ratio_best +0.168*third_share +0.132*gap15 +0.130*pair_gap -0.101*unordered_trio_mass -0.093*gap12` |
| gen4_mut_1596_guard_r5_m0.00 | false | false | 0.1605 / +1.031pp | 0.1597 / -1.228pp | 0.6416 / +0.723pp | `+0.830*neg_log_odds -0.334*rank_score +0.235*neg_odds_ratio_best +0.172*top3_same_pair +0.167*third_share +0.131*gap15 +0.129*pair_gap -0.101*unordered_trio_mass` |
| gen4_mut_496_guard_r8_m0.00 | false | false | 0.1605 / +1.031pp | 0.1611 / -1.084pp | 0.6402 / +0.578pp | `+0.834*neg_log_odds -0.357*rank_score +0.236*neg_odds_ratio_best +0.168*third_share +0.132*gap15 +0.130*pair_gap -0.101*unordered_trio_mass -0.093*gap12` |
| gen4_mut_1596_guard_r8_m0.00 | false | false | 0.1605 / +1.031pp | 0.1597 / -1.228pp | 0.6416 / +0.723pp | `+0.830*neg_log_odds -0.334*rank_score +0.235*neg_odds_ratio_best +0.172*top3_same_pair +0.167*third_share +0.131*gap15 +0.129*pair_gap -0.101*unordered_trio_mass` |
| gen4_mut_416 | false | false | 0.1609 / +1.070pp | 0.1626 / -0.939pp | 0.6380 / +0.361pp | `+0.451*log_q +0.398*neg_odds_ratio_best +0.359*top3_same_first +0.308*third_share +0.307*neg_log_odds -0.270*second_mass +0.245*gap110 -0.236*first_gap` |
| gen4_mut_416_guard_r8_m0.00 | false | false | 0.1609 / +1.070pp | 0.1626 / -0.939pp | 0.6380 / +0.361pp | `+0.451*log_q +0.398*neg_odds_ratio_best +0.359*top3_same_first +0.308*third_share +0.307*neg_log_odds -0.270*second_mass +0.245*gap110 -0.236*first_gap` |
| gen4_mut_1371_guard_r8_m0.00 | false | false | 0.1609 / +1.070pp | 0.1655 / -0.650pp | 0.6387 / +0.434pp | `-0.414*first_gap +0.355*gap110 +0.345*top3_same_first +0.342*log_q -0.295*rank_score +0.293*neg_log_odds -0.271*second_mass +0.242*third_share` |
| gen4_mut_1371 | false | false | 0.1609 / +1.070pp | 0.1655 / -0.650pp | 0.6387 / +0.434pp | `-0.414*first_gap +0.355*gap110 +0.345*top3_same_first +0.342*log_q -0.295*rank_score +0.293*neg_log_odds -0.271*second_mass +0.242*third_share` |
| gen4_mut_4954 | false | false | 0.1609 / +1.070pp | 0.1684 / -0.361pp | 0.6395 / +0.506pp | `+0.561*gap12 +0.473*pair_share +0.411*pair_mass +0.297*log_q +0.295*neg_odds_ratio_best +0.176*third_share -0.158*top5_same_first -0.153*rank_score` |
