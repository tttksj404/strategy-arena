# KRA wide greedy drug-discovery search

The search treats an ensemble as a genome across 20 model families. Each cycle
assays at least 100,000 candidates (`20,000 × 5 generations`) using sparse and
dense random libraries, mutation, crossover, and a greedy discovery-only beam.
The library includes winner classifiers, full-order Plackett-Luce models, and
all-precedence pairwise rankers.

Safety and validation contracts:

- current-race market weight is capped at 15%;
- frontier selection uses only the first three chronological discovery folds;
- the fourth fold is confirmation-only and cannot affect candidate selection;
- historical parity requires every fold within 0.5 percentage points of market
  and every fold at least 5.0 points above production v4;
- production promotion remains false until a new pristine forward holdout passes;
- state, frontier, cache, report, and append-only ledger live under `runs/`.

The persistent runner is `tools/kra_drug_discovery_loop.py`. It refuses narrow
cycles, resumes from the saved frontier, and stops broad exploration only after
historical market parity, an explicit cycle limit, or the presence of
`runs/kra_drug_discovery.stop`. A parity result moves the work to pristine
forward validation; it does not enable production scoring.
