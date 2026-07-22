# Wave-5 report

Wave-5 uses only the frozen wave-1 cache. No fetch stage or network call is part of this pipeline.

Single-candidate tie-break selection: **W5a**.

| Candidate | Family | Verdict | OOS label |
|---|---|---|---|
| W2c | F1 | FAIL |  |
| W5a | F5 | FAIL |  |
| W5b | F5 | FAIL |  |
| W5c | F5 | FAIL |  |
| W5d | F5 | FAIL |  |
| W5e | F5 | FAIL |  |
| W5f | F4 | OOS_CONTAMINATED_IS_ONLY | OOS_CONTAMINATED_IS_ONLY |
| W5g | F5 | FAIL |  |

## W5g combination decision

- Selected candidate: `W5a`
- Correlation: `-0.037937513327918254`; pass=`True`
- MDD: `0.35039989651410697` vs baseline `0.017222042019532036`; pass=`False`
- CAGR: `-0.04158357592602513` vs baseline `0.18081147700275957`; pass=`False`
- OOS return: `0.01103224288571858` vs baseline `0.006394883566836818`; pass=`True`
- Final verdict: **FAIL**

A missing OOS interval remains untested; it is not converted into a performance win.
