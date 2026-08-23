# Reproducible research artifacts

- [2026-08-23-fixed-5x-replay.json](2026-08-23-fixed-5x-replay.json) is the completed 30-day / four-contract baseline replay. Every precommitted volume threshold failed the positive conservative-return requirement, so the result is `NO_RELIABLE_WINNER`.
- [2026-08-24-candidate-contract.json](2026-08-24-candidate-contract.json) freezes all six current candidates: four breakout-volume thresholds, the existing technical-score filter, and the Bollinger-squeeze filter.

The technical-score and Bollinger filters were added to the replay code and exercised by deterministic regression tests. Their public-data sample is not yet large enough to satisfy the precommitted 80 validation / 20 held-out signal minima; neither is promoted. A 7-day, one-contract smoke replay observed zero technical-score and one Bollinger signal, which is explicitly insufficient for selection.
