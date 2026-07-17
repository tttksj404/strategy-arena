#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    sport: str
    title: str
    objective: str
    combination: tuple[str, ...]
    why_next: str
    already_tested_boundary: str
    validation_gate: str
    expected_accuracy_gain: int
    novelty: int
    deployability: int
    leakage_safety: int
    overfit_resistance: int

    @property
    def score(self) -> int:
        return (
            self.expected_accuracy_gain * 3
            + self.novelty * 2
            + self.deployability
            + self.leakage_safety
            + self.overfit_resistance
        )


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        exp_id="KRA-DUAL-PHASE-ODDS",
        sport="horse",
        title="KRA pre-race/live dual-phase model router",
        objective="Improve displayed top-1 and calibration by separating pure pre-race inference from live-odds inference.",
        combination=(
            "pure intrinsic model without winOdds/plcOdds columns",
            "live model/blend only when a fresh odds snapshot exists",
            "phase flag in API response: pre_race vs live_odds",
            "per-meet calibration check",
        ),
        why_next=(
            "KRA OOS evidence says market top-1 beat intrinsic GBM by a large margin "
            "(market 0.381 vs model 0.242 in runs/model_backtest_results.md). "
            "The deployed artifact already contains odds columns, so phase separation is the highest-safety accuracy fix."
        ),
        already_tested_boundary="Do not claim +EV; prior KRA single/exotic betting gates failed across 35+ OOS tests.",
        validation_gate="2026 holdout: top1 >= max(pure_model, market_blend_baseline) and ECE not worse; paired bootstrap by race.",
        expected_accuracy_gain=5,
        novelty=4,
        deployability=4,
        leakage_safety=5,
        overfit_resistance=4,
    ),
    Experiment(
        exp_id="KEIRIN-META-ROUTER",
        sport="keirin",
        title="Learned model router for base/final/11R/special models",
        objective="Raise conditional hit rate by choosing the right existing specialist model per race.",
        combination=(
            "base keirin model",
            "final-day model",
            "11R+ model",
            "special+11R model",
            "OOF meta-router using race_no, day_tcnt, grade, field size, pwin gap",
        ),
        why_next=(
            "The app currently uses hard thresholds. Existing specialist comments report much higher conditional top1 "
            "(final 78%, 11R+ 66%, special 69%), so the next gain is routing quality rather than a new monolithic model."
        ),
        already_tested_boundary="General ensembles in final_validation_results.md were slightly negative, so router must be regime-gated OOF, not average blending.",
        validation_gate="Walk-forward OOS: top1 +0.5pp over current rule router, no regression in plc-top2; paired bootstrap by race.",
        expected_accuracy_gain=4,
        novelty=4,
        deployability=5,
        leakage_safety=4,
        overfit_resistance=4,
    ),
    Experiment(
        exp_id="KEIRIN-REST-EXPANDING-LTR",
        sport="keirin",
        title="Promote the best rest/expanding/LambdaRank feature bundle",
        objective="Lock the strongest measured hit-rate feature gains into a reproducible candidate artifact.",
        combination=(
            "rest_days",
            "racer expanding form with shift",
            "all relative features",
            "LambdaRank/listwise scorer",
            "GBM fallback if listwise underperforms by regime",
        ),
        why_next=(
            "feature_experiment_results.md already showed small but real-looking gains: "
            "all_lambdarank +0.44pp top1 and +0.49pp plc-top2 on holdout."
        ),
        already_tested_boundary="Do not retest the same feature table without seed/time-split stability and CI.",
        validation_gate="Repeat across at least 3 time splits/seeds; deploy only if mean gain >0 and lower CI is not meaningfully negative.",
        expected_accuracy_gain=3,
        novelty=2,
        deployability=4,
        leakage_safety=4,
        overfit_resistance=4,
    ),
    Experiment(
        exp_id="BOTH-CONDITIONAL-CALIBRATION",
        sport="both",
        title="Regime-specific calibration and confidence policy",
        objective="Make probabilities more predictable by calibrating by race regime and abstaining on chaotic races.",
        combination=(
            "sport/meet calibration buckets",
            "race_no bucket",
            "field_size bucket",
            "top1-top2 gap quantile",
            "confidence labels: high/normal/chaos",
        ),
        why_next=(
            "Prior work found strong conditional differences: high-confidence gaps and final/11R regimes behave differently. "
            "Even when top1 accuracy barely moves, calibrated confidence improves user decisions and avoids false certainty."
        ),
        already_tested_boundary="This is an accuracy/UX policy, not a betting edge policy.",
        validation_gate="ECE and Brier improve in each large bucket; high-confidence bucket hit rate materially exceeds all-race hit rate.",
        expected_accuracy_gain=3,
        novelty=3,
        deployability=5,
        leakage_safety=5,
        overfit_resistance=5,
    ),
    Experiment(
        exp_id="BOTH-DIRECT-COMBO-RANKER",
        sport="both",
        title="Direct pair/trio combination ranker for 7 bet types",
        objective="Improve 7-ticket hit-rate display by scoring pairs/trios directly instead of relying only on Harville order.",
        combination=(
            "pair features: p_i, p_j, gap, grade/rating spread",
            "trio features: top concentration, entropy, field size",
            "pool-specific labels for quinella/exacta/trio",
            "Harville as baseline and calibration feature",
        ),
        why_next=(
            "Current 7-ticket logic is Harville from win probabilities. It is simple and stable, but pair/trio interactions "
            "are a genuine unmodeled surface for hit-rate, separate from ROI."
        ),
        already_tested_boundary="KRA/keirin exotic +EV failed; this can only be a hit-rate/ranking improvement unless payout gates pass separately.",
        validation_gate="Per pool: hit-rate > Harville baseline with race-level bootstrap; no deployment if only ROI improves via outliers.",
        expected_accuracy_gain=3,
        novelty=5,
        deployability=3,
        leakage_safety=4,
        overfit_resistance=3,
    ),
    Experiment(
        exp_id="BOTH-STRICT-SKIP-POLICY",
        sport="both",
        title="Strict skip/hold policy optimized for conditional accuracy",
        objective="Increase prediction usefulness by showing final picks only when the model state is historically reliable.",
        combination=(
            "top1-top2 gap",
            "calibrated top probability",
            "market agreement/disagreement if available",
            "race regime",
            "minimum sample-size bucket",
        ),
        why_next=(
            "If the user wants higher probability rather than more picks, abstention is the cleanest lever. "
            "It raises conditional hit rate without pretending every race is equally predictable."
        ),
        already_tested_boundary="Do not backfit thresholds on the same holdout; freeze thresholds before forward validation.",
        validation_gate="Conditional top1/plc hit rate rises with coverage reported; forward candidates logged before evaluation.",
        expected_accuracy_gain=4,
        novelty=3,
        deployability=5,
        leakage_safety=5,
        overfit_resistance=4,
    ),
)


def ranked(sport: str | None = None) -> list[Experiment]:
    items = [
        exp for exp in EXPERIMENTS
        if sport in (None, "all") or exp.sport in (sport, "both")
    ]
    return sorted(items, key=lambda e: (-e.score, e.exp_id))


def as_markdown(items: list[Experiment]) -> str:
    lines = [
        "# Prediction Improvement Matrix",
        "",
        "Scores prioritize hit-rate/calibration improvement, novelty, deployment safety, leakage safety, and overfit resistance.",
        "",
        "| rank | id | sport | score | objective | validation gate |",
        "|---:|---|---|---:|---|---|",
    ]
    for idx, exp in enumerate(items, 1):
        lines.append(
            f"| {idx} | {exp.exp_id} | {exp.sport} | {exp.score} | "
            f"{exp.objective} | {exp.validation_gate} |"
        )
    lines.append("")
    for exp in items:
        lines.extend([
            f"## {exp.exp_id} - {exp.title}",
            f"- Why next: {exp.why_next}",
            f"- Combination: {', '.join(exp.combination)}",
            f"- Boundary: {exp.already_tested_boundary}",
            "",
        ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=("all", "keirin", "horse", "both"), default="all")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    items = ranked(args.sport)
    if args.format == "json":
        print(json.dumps([asdict(exp) | {"score": exp.score} for exp in items], ensure_ascii=False, indent=2))
    else:
        print(as_markdown(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
