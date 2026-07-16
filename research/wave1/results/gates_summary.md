# Wave-1 gate summary

| Candidate | Gate | Name | Status | Value |
|---|---:|---|---|---|
| F1a | 1 | data_validation | PASS | True |
| F1a | 2 | overfit_sensitivity | FAIL | dispersion=0.317 |
| F1a | 3 | walk_forward | FAIL | positive_years=28.6% |
| F1a | 4 | oos_after_cost | FAIL | return=-0.0677 |
| F1a | 5 | trade_bootstrap | FAIL | p05=272.15 |
| F1a | 6 | crash_stress | FAIL | return=-0.3679 |
| F1a | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.0915 |
| F1a | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=2; notional=0.00-0.00 |
| F1a | 9 | kelly | FAIL | f=-245.6043; quarter=-61.4011 |
| F1a | 10 | mdd_limit | PASS | mdd=0.0677 |
| F1a | 11 | sharpe | FAIL | sharpe=-2.2205 |
| F1a | 12 | sortino | PASS | sortino=-1.7125 |
| F1a | 13 | calmar | FAIL | calmar=-1.2599 |
| F1a | 14 | profit_factor | FAIL | pf=0.1240 |
| F1a | 15 | recovery_factor | FAIL | recovery=-1.0000 |
| F1a | 16 | bankruptcy | PASS | p=0.0000 |
| F1a | 17 | regime | FAIL | {'2022_bear': -0.3679449446558951, '2024_bull': -0.15923715579499387, '2025_sideways': -0.2657557314185889} |
| F1a | 18 | btc_correlation | PASS | corr=-0.0746 |
| F1a | 19 | factor_exposure | PASS | True |
| F1b | 1 | data_validation | PASS | True |
| F1b | 2 | overfit_sensitivity | FAIL | dispersion=51.694 |
| F1b | 3 | walk_forward | FAIL | positive_years=42.9% |
| F1b | 4 | oos_after_cost | FAIL | return=-0.0274 |
| F1b | 5 | trade_bootstrap | UNDETERMINED | <20 trades |
| F1b | 6 | crash_stress | PASS | return=-0.0042 |
| F1b | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.0309 |
| F1b | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=2; notional=0.00-0.00 |
| F1b | 9 | kelly | FAIL | f=-252.0324; quarter=-63.0081 |
| F1b | 10 | mdd_limit | PASS | mdd=0.0274 |
| F1b | 11 | sharpe | FAIL | sharpe=-0.9335 |
| F1b | 12 | sortino | PASS | sortino=-0.4616 |
| F1b | 13 | calmar | FAIL | calmar=-1.2653 |
| F1b | 14 | profit_factor | FAIL | pf=0.0000 |
| F1b | 15 | recovery_factor | FAIL | recovery=-0.9987 |
| F1b | 16 | bankruptcy | UNDETERMINED | <20 trades |
| F1b | 17 | regime | FAIL | {'2022_bear': -0.00417472117992046, '2024_bull': 0.04671085775958694, '2025_sideways': -0.06520703020990559} |
| F1b | 18 | btc_correlation | PASS | corr=-0.0757 |
| F1b | 19 | factor_exposure | PASS | True |
| F1c | 1 | data_validation | PASS | True |
| F1c | 2 | overfit_sensitivity | FAIL | dispersion=1.518 |
| F1c | 3 | walk_forward | FAIL | positive_years=42.9% |
| F1c | 4 | oos_after_cost | FAIL | return=-0.0367 |
| F1c | 5 | trade_bootstrap | UNDETERMINED | <20 trades |
| F1c | 6 | crash_stress | PASS | return=-0.2017 |
| F1c | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.0459 |
| F1c | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=2; notional=0.00-0.00 |
| F1c | 9 | kelly | FAIL | f=-346.0615; quarter=-86.5154 |
| F1c | 10 | mdd_limit | PASS | mdd=0.0385 |
| F1c | 11 | sharpe | FAIL | sharpe=-0.9640 |
| F1c | 12 | sortino | PASS | sortino=-0.5360 |
| F1c | 13 | calmar | FAIL | calmar=-1.2058 |
| F1c | 14 | profit_factor | FAIL | pf=0.0202 |
| F1c | 15 | recovery_factor | FAIL | recovery=-0.9529 |
| F1c | 16 | bankruptcy | UNDETERMINED | <20 trades |
| F1c | 17 | regime | FAIL | {'2022_bear': -0.20171939893289226, '2024_bull': 0.0005120497611554864, '2025_sideways': -0.15010007662180436} |
| F1c | 18 | btc_correlation | PASS | corr=-0.0603 |
| F1c | 19 | factor_exposure | PASS | True |
| F1d | 1 | data_validation | PASS | True |
| F1d | 2 | overfit_sensitivity | FAIL | dispersion=1.028 |
| F1d | 3 | walk_forward | FAIL | positive_years=57.1% |
| F1d | 4 | oos_after_cost | FAIL | return=-0.0008 |
| F1d | 5 | trade_bootstrap | UNDETERMINED | <20 trades |
| F1d | 6 | crash_stress | PASS | return=-0.0043 |
| F1d | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.0032 |
| F1d | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=2; notional=0.00-0.00 |
| F1d | 9 | kelly | FAIL | f=-89.6539; quarter=-22.4135 |
| F1d | 10 | mdd_limit | PASS | mdd=0.0138 |
| F1d | 11 | sharpe | FAIL | sharpe=-0.0285 |
| F1d | 12 | sortino | PASS | sortino=-0.0092 |
| F1d | 13 | calmar | FAIL | calmar=-0.0769 |
| F1d | 14 | profit_factor | FAIL | pf=0.5709 |
| F1d | 15 | recovery_factor | FAIL | recovery=-0.0605 |
| F1d | 16 | bankruptcy | UNDETERMINED | <20 trades |
| F1d | 17 | regime | FAIL | {'2022_bear': -0.0042834769943771445, '2024_bull': 0.08110675775408338, '2025_sideways': -0.02763181448848262} |
| F1d | 18 | btc_correlation | PASS | corr=-0.0665 |
| F1d | 19 | factor_exposure | PASS | True |
| F1e | 1 | data_validation | PASS | True |
| F1e | 2 | overfit_sensitivity | PASS | dispersion=1.000 |
| F1e | 3 | walk_forward | FAIL | positive_years=42.9% |
| F1e | 4 | oos_after_cost | FAIL | return=0.0000 |
| F1e | 5 | trade_bootstrap | UNDETERMINED | <20 trades |
| F1e | 6 | crash_stress | PASS | return=-0.0070 |
| F1e | 7 | trading_costs | FAIL | measured=True; double_slippage=0.0000 |
| F1e | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=2; notional=0.00-0.00 |
| F1e | 9 | kelly | FAIL | f=0.0000; quarter=0.0000 |
| F1e | 10 | mdd_limit | PASS | mdd=0.0000 |
| F1e | 11 | sharpe | FAIL | sharpe=0.0000 |
| F1e | 12 | sortino | PASS | sortino=0.0000 |
| F1e | 13 | calmar | FAIL | calmar=0.0000 |
| F1e | 14 | profit_factor | FAIL | pf=0.0000 |
| F1e | 15 | recovery_factor | FAIL | recovery=0.0000 |
| F1e | 16 | bankruptcy | UNDETERMINED | <20 trades |
| F1e | 17 | regime | FAIL | {'2022_bear': -0.007000486899365388, '2024_bull': 0.0, '2025_sideways': 0.0} |
| F1e | 18 | btc_correlation | PASS | corr=-0.0671 |
| F1e | 19 | factor_exposure | PASS | True |
| F1f | 1 | data_validation | PASS | True |
| F1f | 2 | overfit_sensitivity | PASS | dispersion=1.011 |
| F1f | 3 | walk_forward | PASS | positive_years=71.4% |
| F1f | 4 | oos_after_cost | FAIL | return=-0.0008 |
| F1f | 5 | trade_bootstrap | UNDETERMINED | <20 trades |
| F1f | 6 | crash_stress | PASS | return=-0.0054 |
| F1f | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.0032 |
| F1f | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=4; notional=0.00-0.00 |
| F1f | 9 | kelly | FAIL | f=-89.6539; quarter=-22.4135 |
| F1f | 10 | mdd_limit | PASS | mdd=0.0138 |
| F1f | 11 | sharpe | FAIL | sharpe=-0.0285 |
| F1f | 12 | sortino | PASS | sortino=-0.0092 |
| F1f | 13 | calmar | FAIL | calmar=-0.0769 |
| F1f | 14 | profit_factor | FAIL | pf=0.5709 |
| F1f | 15 | recovery_factor | FAIL | recovery=-0.0605 |
| F1f | 16 | bankruptcy | UNDETERMINED | <20 trades |
| F1f | 17 | regime | FAIL | {'2022_bear': -0.0053828696826455635, '2024_bull': 0.10515893708294133, '2025_sideways': -0.01855811111723593} |
| F1f | 18 | btc_correlation | PASS | corr=-0.0774 |
| F1f | 19 | factor_exposure | PASS | True |
| F2a | 1 | data_validation | PASS | True |
| F2a | 2 | overfit_sensitivity | FAIL | dispersion=2.794 |
| F2a | 3 | walk_forward | FAIL | positive_years=28.6% |
| F2a | 4 | oos_after_cost | FAIL | return=-0.1100 |
| F2a | 5 | trade_bootstrap | FAIL | p05=209.42 |
| F2a | 6 | crash_stress | PASS | return=-0.2279 |
| F2a | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.1389 |
| F2a | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=3; notional=0.00-0.00 |
| F2a | 9 | kelly | FAIL | f=-2.8947; quarter=-0.7237 |
| F2a | 10 | mdd_limit | PASS | mdd=0.2255 |
| F2a | 11 | sharpe | FAIL | sharpe=-0.3878 |
| F2a | 12 | sortino | FAIL | sortino=-0.6247 |
| F2a | 13 | calmar | FAIL | calmar=-0.6107 |
| F2a | 14 | profit_factor | FAIL | pf=0.9551 |
| F2a | 15 | recovery_factor | FAIL | recovery=-0.4877 |
| F2a | 16 | bankruptcy | PASS | p=0.0005 |
| F2a | 17 | regime | FAIL | {'2022_bear': -0.22786292022649246, '2024_bull': -0.17968076689222034, '2025_sideways': -0.2112600135987165} |
| F2a | 18 | btc_correlation | PASS | corr=-0.0794 |
| F2a | 19 | factor_exposure | PASS | True |
| F2b | 1 | data_validation | PASS | True |
| F2b | 2 | overfit_sensitivity | FAIL | dispersion=3.502 |
| F2b | 3 | walk_forward | FAIL | positive_years=14.3% |
| F2b | 4 | oos_after_cost | FAIL | return=-0.3765 |
| F2b | 5 | trade_bootstrap | FAIL | p05=143.25 |
| F2b | 6 | crash_stress | FAIL | return=-0.2665 |
| F2b | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.3967 |
| F2b | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=3; notional=0.00-0.00 |
| F2b | 9 | kelly | FAIL | f=-15.8795; quarter=-3.9699 |
| F2b | 10 | mdd_limit | FAIL | mdd=0.4120 |
| F2b | 11 | sharpe | FAIL | sharpe=-1.9416 |
| F2b | 12 | sortino | FAIL | sortino=-2.5840 |
| F2b | 13 | calmar | FAIL | calmar=-1.0960 |
| F2b | 14 | profit_factor | FAIL | pf=0.7744 |
| F2b | 15 | recovery_factor | FAIL | recovery=-0.9137 |
| F2b | 16 | bankruptcy | FAIL | p=0.0849 |
| F2b | 17 | regime | FAIL | {'2022_bear': -0.26647558102073265, '2024_bull': -0.19847475258809666, '2025_sideways': -0.13567113835940336} |
| F2b | 18 | btc_correlation | PASS | corr=0.0178 |
| F2b | 19 | factor_exposure | PASS | True |
| F2c | 1 | data_validation | PASS | True |
| F2c | 2 | overfit_sensitivity | FAIL | dispersion=2.765 |
| F2c | 3 | walk_forward | FAIL | positive_years=57.1% |
| F2c | 4 | oos_after_cost | FAIL | return=-0.2043 |
| F2c | 5 | trade_bootstrap | FAIL | p05=205.20 |
| F2c | 6 | crash_stress | FAIL | return=-0.2888 |
| F2c | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.2155 |
| F2c | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=3; notional=0.00-0.00 |
| F2c | 9 | kelly | FAIL | f=-23.8346; quarter=-5.9586 |
| F2c | 10 | mdd_limit | PASS | mdd=0.2299 |
| F2c | 11 | sharpe | FAIL | sharpe=-1.6665 |
| F2c | 12 | sortino | PASS | sortino=-1.6111 |
| F2c | 13 | calmar | FAIL | calmar=-1.0970 |
| F2c | 14 | profit_factor | FAIL | pf=0.7071 |
| F2c | 15 | recovery_factor | FAIL | recovery=-0.8886 |
| F2c | 16 | bankruptcy | PASS | p=0.0000 |
| F2c | 17 | regime | FAIL | {'2022_bear': -0.2888439216111157, '2024_bull': 0.04241497659233895, '2025_sideways': -0.09750513764734126} |
| F2c | 18 | btc_correlation | PASS | corr=0.5945 |
| F2c | 19 | factor_exposure | PASS | True |
| F2d | 1 | data_validation | PASS | True |
| F2d | 2 | overfit_sensitivity | FAIL | dispersion=8.135 |
| F2d | 3 | walk_forward | FAIL | positive_years=57.1% |
| F2d | 4 | oos_after_cost | FAIL | return=-0.3107 |
| F2d | 5 | trade_bootstrap | FAIL | p05=179.75 |
| F2d | 6 | crash_stress | FAIL | return=-0.2702 |
| F2d | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.3177 |
| F2d | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=3; notional=0.00-0.00 |
| F2d | 9 | kelly | FAIL | f=-52.2864; quarter=-13.0716 |
| F2d | 10 | mdd_limit | FAIL | mdd=0.3500 |
| F2d | 11 | sharpe | FAIL | sharpe=-3.0919 |
| F2d | 12 | sortino | PASS | sortino=-2.1255 |
| F2d | 13 | calmar | FAIL | calmar=-1.0771 |
| F2d | 14 | profit_factor | FAIL | pf=0.4390 |
| F2d | 15 | recovery_factor | FAIL | recovery=-0.8877 |
| F2d | 16 | bankruptcy | PASS | p=0.0000 |
| F2d | 17 | regime | FAIL | {'2022_bear': -0.2701870059720486, '2024_bull': 0.017984980280241203, '2025_sideways': -0.12209737690087363} |
| F2d | 18 | btc_correlation | PASS | corr=0.6313 |
| F2d | 19 | factor_exposure | PASS | True |
| F2e | 1 | data_validation | PASS | True |
| F2e | 2 | overfit_sensitivity | FAIL | dispersion=11.017 |
| F2e | 3 | walk_forward | FAIL | positive_years=28.6% |
| F2e | 4 | oos_after_cost | FAIL | return=-0.1690 |
| F2e | 5 | trade_bootstrap | FAIL | p05=220.16 |
| F2e | 6 | crash_stress | PASS | return=-0.1416 |
| F2e | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.1732 |
| F2e | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=3; notional=0.00-0.00 |
| F2e | 9 | kelly | FAIL | f=-34.8270; quarter=-8.7067 |
| F2e | 10 | mdd_limit | PASS | mdd=0.2232 |
| F2e | 11 | sharpe | FAIL | sharpe=-1.6333 |
| F2e | 12 | sortino | PASS | sortino=-0.8895 |
| F2e | 13 | calmar | FAIL | calmar=-0.9397 |
| F2e | 14 | profit_factor | FAIL | pf=0.5772 |
| F2e | 15 | recovery_factor | FAIL | recovery=-0.7571 |
| F2e | 16 | bankruptcy | PASS | p=0.0000 |
| F2e | 17 | regime | FAIL | {'2022_bear': -0.14164151559432225, '2024_bull': -0.06888210130766348, '2025_sideways': -0.3018211477286783} |
| F2e | 18 | btc_correlation | PASS | corr=0.6011 |
| F2e | 19 | factor_exposure | PASS | True |
| F2f | 1 | data_validation | PASS | True |
| F2f | 2 | overfit_sensitivity | FAIL | dispersion=1.539 |
| F2f | 3 | walk_forward | FAIL | positive_years=28.6% |
| F2f | 4 | oos_after_cost | FAIL | return=-0.2656 |
| F2f | 5 | trade_bootstrap | FAIL | p05=168.38 |
| F2f | 6 | crash_stress | PASS | return=0.0436 |
| F2f | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.2895 |
| F2f | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=3; notional=0.00-0.00 |
| F2f | 9 | kelly | FAIL | f=-10.1218; quarter=-2.5304 |
| F2f | 10 | mdd_limit | FAIL | mdd=0.3285 |
| F2f | 11 | sharpe | FAIL | sharpe=-1.2015 |
| F2f | 12 | sortino | FAIL | sortino=-1.7155 |
| F2f | 13 | calmar | FAIL | calmar=-0.9884 |
| F2f | 14 | profit_factor | FAIL | pf=0.8509 |
| F2f | 15 | recovery_factor | FAIL | recovery=-0.8085 |
| F2f | 16 | bankruptcy | PASS | p=0.0107 |
| F2f | 17 | regime | FAIL | {'2022_bear': 0.0435854863548768, '2024_bull': -0.0669370466683008, '2025_sideways': -0.257534187204693} |
| F2f | 18 | btc_correlation | PASS | corr=0.0955 |
| F2f | 19 | factor_exposure | PASS | True |
| F3-H1 | 1 | data_validation | PASS | True |
| F3-H1 | 2 | overfit_sensitivity | FAIL | dispersion=0.031 |
| F3-H1 | 3 | walk_forward | FAIL | positive_years=0.0% |
| F3-H1 | 4 | oos_after_cost | FAIL | return=-0.4707 |
| F3-H1 | 5 | trade_bootstrap | FAIL | p05=155.62 |
| F3-H1 | 6 | crash_stress | UNDETERMINED | no 2022 data |
| F3-H1 | 7 | trading_costs | FAIL | measured=True; double_slippage=-0.6031 |
| F3-H1 | 8 | capacity_and_sizing | FAIL | quarter_kelly=0.0000; leverage=0.0000; concurrent=1; notional=0.00-0.00 |
| F3-H1 | 9 | kelly | FAIL | f=-4305.2212; quarter=-1076.3053 |
| F3-H1 | 10 | mdd_limit | FAIL | mdd=0.4707 |
| F3-H1 | 11 | sharpe | FAIL | sharpe=-58.9875 |
| F3-H1 | 12 | sortino | FAIL | sortino=-61.0389 |
| F3-H1 | 13 | calmar | FAIL | calmar=-1.1785 |
| F3-H1 | 14 | profit_factor | FAIL | pf=0.0003 |
| F3-H1 | 15 | recovery_factor | FAIL | recovery=-1.0000 |
| F3-H1 | 16 | bankruptcy | PASS | p=0.0000 |
| F3-H1 | 17 | regime | FAIL | {'2025_sideways': -0.27822596113601983} |
| F3-H1 | 18 | btc_correlation | PASS | corr=0.0011 |
| F3-H1 | 19 | factor_exposure | PASS | True |
| F3-H2 | 1 | data_validation | PASS | True |
| F3-H2 | 2 | overfit_sensitivity | FAIL | dispersion=0.595 |
| F3-H2 | 3 | walk_forward | PASS | positive_years=100.0% |
| F3-H2 | 4 | oos_after_cost | PASS | return=0.2012 |
| F3-H2 | 5 | trade_bootstrap | FAIL | p05=34.33 |
| F3-H2 | 6 | crash_stress | UNDETERMINED | no 2022 data |
| F3-H2 | 7 | trading_costs | PASS | measured=True; double_slippage=0.1519 |
| F3-H2 | 8 | capacity_and_sizing | PASS | quarter_kelly=0.1364; leverage=0.1364; concurrent=1; notional=40.91-40.91 |
| F3-H2 | 9 | kelly | PASS | f=0.5455; quarter=0.1364 |
| F3-H2 | 10 | mdd_limit | FAIL | mdd=0.5273 |
| F3-H2 | 11 | sharpe | FAIL | sharpe=0.9307 |
| F3-H2 | 12 | sortino | PASS | sortino=1.0511 |
| F3-H2 | 13 | calmar | FAIL | calmar=0.5058 |
| F3-H2 | 14 | profit_factor | FAIL | pf=1.3434 |
| F3-H2 | 15 | recovery_factor | FAIL | recovery=0.3815 |
| F3-H2 | 16 | bankruptcy | FAIL | p=0.2764 |
| F3-H2 | 17 | regime | PASS | {'2025_sideways': 0.24227198553188245} |
| F3-H2 | 18 | btc_correlation | PASS | corr=0.0017 |
| F3-H2 | 19 | factor_exposure | PASS | True |
| F3-H3 | 1 | data_validation | PASS | True |
| F3-H3 | 2 | overfit_sensitivity | FAIL | dispersion=27229480679569.895 |
| F3-H3 | 3 | walk_forward | FAIL | positive_years=0.0% |
| F3-H3 | 4 | oos_after_cost | PASS | return=0.0190 |
| F3-H3 | 5 | trade_bootstrap | PASS | p05=304.47 |
| F3-H3 | 6 | crash_stress | UNDETERMINED | no 2022 data |
| F3-H3 | 7 | trading_costs | PASS | measured=True; double_slippage=0.0171 |
| F3-H3 | 8 | capacity_and_sizing | PASS | quarter_kelly=1828.2318; leverage=0.5000; concurrent=1; notional=150.00-150.00 |
| F3-H3 | 9 | kelly | PASS | f=7312.9273; quarter=1828.2318 |
| F3-H3 | 10 | mdd_limit | PASS | mdd=0.0010 |
| F3-H3 | 11 | sharpe | PASS | sharpe=27.2295 |
| F3-H3 | 12 | sortino | FAIL | sortino=0.0000 |
| F3-H3 | 13 | calmar | PASS | calmar=55.9019 |
| F3-H3 | 14 | profit_factor | PASS | pf=9.5275 |
| F3-H3 | 15 | recovery_factor | PASS | recovery=19.8811 |
| F3-H3 | 16 | bankruptcy | PASS | p=0.0000 |
| F3-H3 | 17 | regime | UNDETERMINED | {} |
| F3-H3 | 18 | btc_correlation | PASS | corr=-0.1856 |
| F3-H3 | 19 | factor_exposure | PASS | True |
