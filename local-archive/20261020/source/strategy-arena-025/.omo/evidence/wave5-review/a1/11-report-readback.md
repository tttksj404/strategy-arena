REPORT:

# Wave-5 report

Wave-5 uses only the frozen wave-1 cache. No fetch stage or network call is part of this pipeline.

Single-candidate tie-break selection: **W5a**.

| Candidate | Family | Verdict | OOS label |
|---|---|---|---|
| W2c | F1 | FAIL |  |
| W5a | F5 | FAIL |  |
| W5b | F5 | FAIL |  |
| W5c | F5 | UNTESTED_IN_OOS | UNTESTED_IN_OOS |
| W5d | F5 | FAIL |  |
| W5e | F5 | FAIL |  |
| W5f | F4 | OOS_CONTAMINATED_IS_ONLY | OOS_CONTAMINATED_IS_ONLY |
| W5g | F5 | FAIL |  |

## W5g combination decision

- Selected candidate: `W5a`
- Correlation: `-0.036415845381116835`; pass=`True`
- MDD: `0.3220575502840052` vs baseline `0.016125151123011827`; pass=`False`
- CAGR: `-0.02128773750460089` vs baseline `0.18419143534537863`; pass=`False`
- OOS return: `0.034242255420069645` vs baseline `0.00639488356683704`; pass=`True`
- Final verdict: **FAIL**

A missing OOS interval remains untested; it is not converted into a performance win.
