#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — chronological grouped research contract


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_market_pairwise_challenger import ChallengerSpec, fit_challenger_model  # noqa: E402
from kra_market_context_gates import generate_context_gates  # noqa: E402
from kra_model_evaluation import as_dict, market_probability, metrics, paired_bootstrap_top1  # noqa: E402
from kra_top1_screening import Top1View  # noqa: E402
from tools.kra_diversified_search import CONDITIONING_ARCHIVE, ENTRY_ARCHIVE, HEALTH_ARCHIVE, build_diversified_frame  # noqa: E402
from tools.kra_dual_phase_experiment import DEFAULT_DB  # noqa: E402
from tools.kra_favorite_challenger_search import _runner_columns  # noqa: E402
from tools.kra_max_winrate_search import FRESH_FROM, _fold_data  # noqa: E402


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_max_winrate_h20_results.json"
TOP_KS: Final = (2, 3)
BALANCE_POWERS: Final = (0.25, 0.5, 0.75, 1.0)
THRESHOLDS: Final = tuple(float(value) for value in np.arange(0.05, 1.0, 0.05))


@dataclass(frozen=True, slots=True)
class Evidence:
    top_k: int
    balance_power: float
    threshold: float
    top1_lifts_pp: tuple[float, ...]
    switch_rates: tuple[float, ...]
    gate: str = "all"

    @property
    def key(self) -> tuple[int, float, float]:
        return self.top_k, self.balance_power, self.threshold


@dataclass(frozen=True, slots=True)
class SearchInputs:
    db_path: Path
    entry_archive: Path
    conditioning_archive: Path
    health_archive: Path


def _select(candidates: tuple[Evidence, ...]) -> Evidence:
    return max(
        candidates,
        key=lambda item: (
            min(item.top1_lifts_pp[:3]),
            float(np.mean(item.top1_lifts_pp[:3])),
            -float(np.mean(item.switch_rates[:3])),
        ),
    )


def _forward_candidate(candidates: tuple[Evidence, ...]) -> Evidence | None:
    qualifying = tuple(
        candidate for candidate in candidates if min(candidate.top1_lifts_pp) > 0.0
    )
    return max(qualifying, key=lambda item: min(item.top1_lifts_pp)) if qualifying else None


def run_search(inputs: SearchInputs) -> dict:  # noqa: DICT_OK — JSON report boundary
    frame, baseline_columns, candidate_columns = build_diversified_frame(
        inputs.db_path,
        inputs.entry_archive,
        inputs.conditioning_archive,
        inputs.health_archive,
    )
    runner_columns = _runner_columns(candidate_columns)
    folds = _fold_data(frame, baseline_columns)
    predictions: list[dict[tuple[int, float, float], np.ndarray]] = []
    for fold in folds:
        fold_predictions: dict[tuple[int, float, float], np.ndarray] = {}
        for top_k in TOP_KS:
            for balance_power in BALANCE_POWERS:
                spec = ChallengerSpec(top_k, balance_power, 0.0)
                model = fit_challenger_model(fold.train, runner_columns, spec)
                for threshold, score in model.predict_many(
                    fold.test, THRESHOLDS
                ).items():
                    fold_predictions[(top_k, balance_power, threshold)] = score
        predictions.append(fold_predictions)

    gates = generate_context_gates()
    markets = [market_probability(fold.test) for fold in folds]
    views = [Top1View.from_frame(fold.test) for fold in folds]
    market_top1 = [view.accuracy(market) for view, market in zip(views, markets)]
    gate_masks = [
        {gate.name: gate.mask(fold.test, market) for gate in gates}
        for fold, market in zip(folds, markets)
    ]
    evidence: list[Evidence] = []
    for key in predictions[0]:
        for gate in gates:
            top1_lifts: list[float] = []
            switch_rates: list[float] = []
            for index, fold_predictions in enumerate(predictions):
                market = markets[index]
                candidate = np.where(
                    gate_masks[index][gate.name], fold_predictions[key], market
                )
                top1_lifts.append(
                    (views[index].accuracy(candidate) - market_top1[index]) * 100.0
                )
                switch_rates.append(views[index].switch_rate(candidate, market))
            evidence.append(
                Evidence(
                    key[0],
                    key[1],
                    key[2],
                    tuple(top1_lifts),
                    tuple(switch_rates),
                    gate.name,
                )
            )

    selected = _select(tuple(evidence))
    selected_gate = next(gate for gate in gates if gate.name == selected.gate)
    selected_scores = [
        np.where(gate_masks[index][selected.gate], fold_predictions[selected.key], markets[index])
        for index, fold_predictions in enumerate(predictions)
    ]
    selected_top3_lifts: list[float] = []
    selected_logloss_deltas: list[float] = []
    for fold, market, candidate in zip(folds, markets, selected_scores):
        market_metrics = metrics(fold.test, market)
        candidate_metrics = metrics(fold.test, candidate)
        selected_top3_lifts.append(
            (candidate_metrics.top3 - market_metrics.top3) * 100.0
        )
        selected_logloss_deltas.append(
            candidate_metrics.race_logloss - market_metrics.race_logloss
        )
    pooled_frame = pd.concat([fold.test for fold in folds], ignore_index=True)
    pooled_candidate = np.concatenate(selected_scores)
    pooled_market = np.concatenate([market_probability(fold.test) for fold in folds])
    pooled_v4 = np.concatenate([fold.baseline_probability for fold in folds])
    pooled_bootstrap = paired_bootstrap_top1(
        pooled_frame, pooled_candidate, pooled_market, samples=10_000
    )

    fresh_train = frame[frame["rcDate"] < FRESH_FROM]
    fresh = frame[frame["rcDate"] >= FRESH_FROM].copy()
    fresh_model = fit_challenger_model(
        fresh_train,
        runner_columns,
        ChallengerSpec(selected.top_k, selected.balance_power, selected.threshold),
    )
    fresh_market = market_probability(fresh)
    fresh_candidate = selected_gate.apply(
        fresh,
        fresh_model.predict(fresh, selected.threshold),
        fresh_market,
    )
    fresh_bootstrap = paired_bootstrap_top1(
        fresh, fresh_candidate, fresh_market, samples=10_000
    )
    historical_market_beating = bool(
        min(selected.top1_lifts_pp) > 0.0
        and pooled_bootstrap["ci95_low_pp"] > 0.0
        and min(selected_top3_lifts) >= 0.0
        and max(selected_logloss_deltas) <= 0.0
    )
    forward_candidate = _forward_candidate(tuple(evidence))
    forward_payload = None
    if forward_candidate is not None:
        forward_gate = next(gate for gate in gates if gate.name == forward_candidate.gate)
        forward_scores = [
            np.where(gate_masks[index][forward_candidate.gate], fold_predictions[forward_candidate.key], markets[index])
            for index, fold_predictions in enumerate(predictions)
        ]
        pooled_forward = np.concatenate(forward_scores)
        forward_fresh_model = fit_challenger_model(
            fresh_train,
            runner_columns,
            ChallengerSpec(forward_candidate.top_k, forward_candidate.balance_power, forward_candidate.threshold),
        )
        forward_fresh = forward_gate.apply(
            fresh,
            forward_fresh_model.predict(fresh, forward_candidate.threshold),
            fresh_market,
        )
        forward_payload = {
            **asdict(forward_candidate),
            "selection_uses_confirmation": True,
            "is_post_selection_only": True,
            "pooled_bootstrap_vs_market": paired_bootstrap_top1(
                pooled_frame, pooled_forward, pooled_market, samples=10_000
            ),
            "observed_holdout_bootstrap_vs_market": paired_bootstrap_top1(
                fresh, forward_fresh, fresh_market, samples=10_000
            ),
        }
    return {
        "method": "market_pairwise_challenger_context_gate_search",
        "baseline": "pure_normalized_win_market",
        "candidate_count": len(evidence),
        "context_gate_count": len(gates),
        "runner_feature_count": len(runner_columns),
        "selection_folds": [fold.name for fold in folds[:3]],
        "confirmation_fold": folds[3].name,
        "selection_uses_confirmation": False,
        "selected": {
            **asdict(selected),
            "top3_lifts_pp": selected_top3_lifts,
            "logloss_deltas": selected_logloss_deltas,
        },
        "selected_result": {
            "pooled": {
                "v4": as_dict(metrics(pooled_frame, pooled_v4)),
                "market": as_dict(metrics(pooled_frame, pooled_market)),
                "candidate": as_dict(metrics(pooled_frame, pooled_candidate)),
                "bootstrap_vs_market": pooled_bootstrap,
            },
            "observed_holdout": {
                "from": FRESH_FROM,
                "market": as_dict(metrics(fresh, fresh_market)),
                "candidate": as_dict(metrics(fresh, fresh_candidate)),
                "bootstrap_vs_market": fresh_bootstrap,
                "is_pristine": False,
            },
        },
        "historical_market_beating_pass": historical_market_beating,
        "post_selection_all_fold_candidate": forward_payload,
        "fresh_timestamped_forward_validation_required": True,
        "promotion_pass": False,
        "candidates": [asdict(candidate) for candidate in evidence],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--entry-archive", type=Path, default=ENTRY_ARCHIVE)
    parser.add_argument("--conditioning-archive", type=Path, default=CONDITIONING_ARCHIVE)
    parser.add_argument("--health-archive", type=Path, default=HEALTH_ARCHIVE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_search(
        SearchInputs(
            args.db,
            args.entry_archive,
            args.conditioning_archive,
            args.health_archive,
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
