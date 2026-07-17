# KCYCLE fast-evolution trifecta search

records: 13900
top_k: 10
candidates_per_generation: 5000
generations: 4
elapsed_sec: 15.287
deployable_count: 0
strict_deployable_count: 0

selection: validation-only rank, then test is reported as untouched OOS evidence.

| candidate | deployable | strict | val exact/lift | test exact/lift | test top1/lift | formula |
|---|---:|---:|---:|---:|---:|---|
| gen3_mut_2256 | false | false | 0.1633 / +1.308pp | 0.1684 / -0.361pp | 0.6373 / +0.289pp | `+0.708*neg_odds_ratio_best +0.338*neg_log_odds -0.322*top5_same_first -0.301*unordered_trio_mass -0.275*gap12 +0.203*first_gap -0.151*top3_same_first -0.126*gap110` |
| gen3_mut_2256_guard_r8_m0.00 | false | false | 0.1633 / +1.308pp | 0.1684 / -0.361pp | 0.6373 / +0.289pp | `+0.708*neg_odds_ratio_best +0.338*neg_log_odds -0.322*top5_same_first -0.301*unordered_trio_mass -0.275*gap12 +0.203*first_gap -0.151*top3_same_first -0.126*gap110` |
| gen3_mut_2256_guard_r5_m0.00 | false | false | 0.1629 / +1.268pp | 0.1691 / -0.289pp | 0.6380 / +0.361pp | `+0.708*neg_odds_ratio_best +0.338*neg_log_odds -0.322*top5_same_first -0.301*unordered_trio_mass -0.275*gap12 +0.203*first_gap -0.151*top3_same_first -0.126*gap110` |
| gen4_mut_3595 | false | false | 0.1625 / +1.229pp | 0.1684 / -0.361pp | 0.6373 / +0.289pp | `+0.750*neg_odds_ratio_best +0.358*neg_log_odds -0.319*unordered_trio_mass -0.223*gap12 +0.213*first_gap -0.155*gap110 -0.142*top3_same_first -0.133*top5_same_first` |
| gen4_mut_3595_guard_r8_m0.00 | false | false | 0.1625 / +1.229pp | 0.1684 / -0.361pp | 0.6373 / +0.289pp | `+0.750*neg_odds_ratio_best +0.358*neg_log_odds -0.319*unordered_trio_mass -0.223*gap12 +0.213*first_gap -0.155*gap110 -0.142*top3_same_first -0.133*top5_same_first` |
| gen4_mut_3723 | false | false | 0.1621 / +1.189pp | 0.1698 / -0.217pp | 0.6380 / +0.361pp | `+0.718*neg_odds_ratio_best +0.343*neg_log_odds -0.326*top5_same_first -0.305*unordered_trio_mass -0.215*gap12 +0.174*first_gap -0.136*top3_same_first -0.128*gap110` |
| gen4_mut_3723_guard_r8_m0.00 | false | false | 0.1621 / +1.189pp | 0.1698 / -0.217pp | 0.6380 / +0.361pp | `+0.718*neg_odds_ratio_best +0.343*neg_log_odds -0.326*top5_same_first -0.305*unordered_trio_mass -0.215*gap12 +0.174*first_gap -0.136*top3_same_first -0.128*gap110` |
| gen4_mut_3595_guard_r5_m0.00 | false | false | 0.1621 / +1.189pp | 0.1691 / -0.289pp | 0.6380 / +0.361pp | `+0.750*neg_odds_ratio_best +0.358*neg_log_odds -0.319*unordered_trio_mass -0.223*gap12 +0.213*first_gap -0.155*gap110 -0.142*top3_same_first -0.133*top5_same_first` |
| gen4_mut_4160 | false | false | 0.1621 / +1.189pp | 0.1691 / -0.289pp | 0.6366 / +0.217pp | `+0.749*neg_odds_ratio_best +0.343*neg_log_odds -0.326*top5_same_first -0.305*unordered_trio_mass -0.173*gap12 +0.160*first_gap -0.149*top3_same_first +0.093*third_share` |
| gen4_mut_2493_guard_r5_m0.00 | false | false | 0.1621 / +1.189pp | 0.1684 / -0.361pp | 0.6380 / +0.361pp | `+0.710*neg_odds_ratio_best +0.332*neg_log_odds -0.316*top5_same_first -0.295*unordered_trio_mass -0.213*gap110 +0.209*first_gap -0.168*gap12 -0.147*second_mass` |
| gen4_mut_4160_guard_r8_m0.00 | false | false | 0.1621 / +1.189pp | 0.1691 / -0.289pp | 0.6366 / +0.217pp | `+0.749*neg_odds_ratio_best +0.343*neg_log_odds -0.326*top5_same_first -0.305*unordered_trio_mass -0.173*gap12 +0.160*first_gap -0.149*top3_same_first +0.093*third_share` |
| gen4_mut_3974 | false | false | 0.1617 / +1.149pp | 0.1698 / -0.217pp | 0.6395 / +0.506pp | `+0.686*neg_odds_ratio_best +0.317*neg_log_odds -0.301*top5_same_first +0.289*first_gap -0.282*unordered_trio_mass +0.263*first_mass -0.198*gap12 -0.133*top3_same_first` |
| gen4_mut_3974_guard_r8_m0.00 | false | false | 0.1617 / +1.149pp | 0.1698 / -0.217pp | 0.6395 / +0.506pp | `+0.686*neg_odds_ratio_best +0.317*neg_log_odds -0.301*top5_same_first +0.289*first_gap -0.282*unordered_trio_mass +0.263*first_mass -0.198*gap12 -0.133*top3_same_first` |
| gen4_mut_2493 | false | false | 0.1621 / +1.189pp | 0.1676 / -0.434pp | 0.6366 / +0.217pp | `+0.710*neg_odds_ratio_best +0.332*neg_log_odds -0.316*top5_same_first -0.295*unordered_trio_mass -0.213*gap110 +0.209*first_gap -0.168*gap12 -0.147*second_mass` |
| gen4_mut_2493_guard_r8_m0.00 | false | false | 0.1621 / +1.189pp | 0.1676 / -0.434pp | 0.6366 / +0.217pp | `+0.710*neg_odds_ratio_best +0.332*neg_log_odds -0.316*top5_same_first -0.295*unordered_trio_mass -0.213*gap110 +0.209*first_gap -0.168*gap12 -0.147*second_mass` |
| gen4_mut_3723_guard_r5_m0.00 | false | false | 0.1617 / +1.149pp | 0.1705 / -0.145pp | 0.6387 / +0.434pp | `+0.718*neg_odds_ratio_best +0.343*neg_log_odds -0.326*top5_same_first -0.305*unordered_trio_mass -0.215*gap12 +0.174*first_gap -0.136*top3_same_first -0.128*gap110` |
| gen3_mut_2378 | false | false | 0.1617 / +1.149pp | 0.1698 / -0.217pp | 0.6373 / +0.289pp | `+0.699*neg_odds_ratio_best +0.327*neg_log_odds -0.311*top5_same_first -0.291*unordered_trio_mass -0.265*gap12 -0.209*gap110 +0.206*first_gap -0.144*second_mass` |
| gen4_mut_2902_guard_r8_m0.00 | false | false | 0.1617 / +1.149pp | 0.1698 / -0.217pp | 0.6373 / +0.289pp | `+0.699*neg_odds_ratio_best +0.327*neg_log_odds -0.311*top5_same_first -0.291*unordered_trio_mass -0.265*gap12 -0.209*gap110 +0.206*first_gap -0.144*second_mass` |
| gen4_mut_4160_guard_r5_m0.00 | false | false | 0.1617 / +1.149pp | 0.1698 / -0.217pp | 0.6373 / +0.289pp | `+0.749*neg_odds_ratio_best +0.343*neg_log_odds -0.326*top5_same_first -0.305*unordered_trio_mass -0.173*gap12 +0.160*first_gap -0.149*top3_same_first +0.093*third_share` |
| gen4_mut_2011 | false | false | 0.1613 / +1.110pp | 0.1698 / -0.217pp | 0.6380 / +0.361pp | `+0.714*neg_odds_ratio_best +0.340*neg_log_odds -0.323*top5_same_first -0.281*unordered_trio_mass -0.274*gap12 +0.204*first_gap -0.134*top3_same_first -0.120*second_mass` |
| gen4_mut_2011_guard_r8_m0.00 | false | false | 0.1613 / +1.110pp | 0.1698 / -0.217pp | 0.6380 / +0.361pp | `+0.714*neg_odds_ratio_best +0.340*neg_log_odds -0.323*top5_same_first -0.281*unordered_trio_mass -0.274*gap12 +0.204*first_gap -0.134*top3_same_first -0.120*second_mass` |
| gen4_mut_3974_guard_r5_m0.00 | false | false | 0.1613 / +1.110pp | 0.1705 / -0.145pp | 0.6402 / +0.578pp | `+0.686*neg_odds_ratio_best +0.317*neg_log_odds -0.301*top5_same_first +0.289*first_gap -0.282*unordered_trio_mass +0.263*first_mass -0.198*gap12 -0.133*top3_same_first` |
| gen4_mut_2379 | false | false | 0.1613 / +1.110pp | 0.1691 / -0.289pp | 0.6366 / +0.217pp | `+0.704*neg_odds_ratio_best +0.336*neg_log_odds -0.320*top5_same_first -0.304*unordered_trio_mass -0.273*gap12 +0.202*first_gap -0.175*top3_same_first -0.126*gap110` |
| gen4_mut_2379_guard_r8_m0.00 | false | false | 0.1613 / +1.110pp | 0.1691 / -0.289pp | 0.6366 / +0.217pp | `+0.704*neg_odds_ratio_best +0.336*neg_log_odds -0.320*top5_same_first -0.304*unordered_trio_mass -0.273*gap12 +0.202*first_gap -0.175*top3_same_first -0.126*gap110` |
| gen3_mut_3086 | false | false | 0.1609 / +1.070pp | 0.1705 / -0.145pp | 0.6380 / +0.361pp | `+0.710*neg_odds_ratio_best +0.332*neg_log_odds -0.315*top5_same_first -0.295*unordered_trio_mass -0.269*gap12 +0.201*first_mass +0.158*first_gap -0.131*top3_same_first` |
| gen3_mut_3086_guard_r8_m0.00 | false | false | 0.1609 / +1.070pp | 0.1705 / -0.145pp | 0.6380 / +0.361pp | `+0.710*neg_odds_ratio_best +0.332*neg_log_odds -0.315*top5_same_first -0.295*unordered_trio_mass -0.269*gap12 +0.201*first_mass +0.158*first_gap -0.131*top3_same_first` |
| gen4_mut_3231 | false | false | 0.1613 / +1.110pp | 0.1712 / -0.072pp | 0.6366 / +0.217pp | `+0.696*neg_odds_ratio_best +0.325*neg_log_odds -0.309*top5_same_first -0.289*unordered_trio_mass -0.265*gap12 +0.215*first_gap -0.208*gap110 -0.144*second_mass` |
| gen4_mut_4249_guard_r5_m0.00 | false | false | 0.1613 / +1.110pp | 0.1705 / -0.145pp | 0.6380 / +0.361pp | `+0.616*neg_odds_ratio_best -0.480*gap110 +0.288*neg_log_odds -0.285*gap12 -0.274*top5_same_first -0.256*unordered_trio_mass +0.181*first_gap -0.127*second_mass` |
| gen3_mut_2905 | false | false | 0.1609 / +1.070pp | 0.1684 / -0.361pp | 0.6373 / +0.289pp | `+0.711*neg_odds_ratio_best +0.339*neg_log_odds -0.323*top5_same_first -0.302*unordered_trio_mass -0.255*gap12 +0.204*first_gap -0.134*top3_same_first -0.127*gap110` |
| gen4_mut_4645 | false | false | 0.1609 / +1.070pp | 0.1676 / -0.434pp | 0.6366 / +0.217pp | `+0.754*neg_odds_ratio_best +0.354*neg_log_odds -0.315*unordered_trio_mass -0.244*gap12 -0.213*top5_same_first +0.200*first_gap -0.145*gap110 -0.110*second_mass` |
