from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kra_drug_assay import FoldTensor, screen_genomes, screen_top1_matrix
from kra_drug_discovery import AssayScore, Genome


@dataclass(frozen=True, slots=True)
class StageReport:
    fold_count: int
    input_count: int
    output_count: int
    assay_evaluations: int
    best_market_gap_pp: float


def _quality(top1: np.ndarray, folds: tuple[FoldTensor, ...]) -> np.ndarray:
    gaps = top1 - np.asarray([fold.market_top1 for fold in folds])
    v4 = top1 - np.asarray([fold.v4_top1 for fold in folds])
    return 100.0 * (
        np.min(gaps, axis=1)
        + 0.15 * np.mean(gaps, axis=1)
        + 0.05 * np.min(v4, axis=1)
    )


def _objectives(top1: np.ndarray, folds: tuple[FoldTensor, ...]) -> np.ndarray:
    gaps = top1 - np.asarray([fold.market_top1 for fold in folds])
    v4 = top1 - np.asarray([fold.v4_top1 for fold in folds])
    return np.column_stack((
        np.min(gaps, axis=1),
        np.mean(gaps, axis=1),
        np.min(v4, axis=1),
        np.mean(v4, axis=1),
    ))


def _pareto_pool(
    quality: np.ndarray,
    top1: np.ndarray,
    folds: tuple[FoldTensor, ...],
    pool_size: int,
) -> np.ndarray:
    pool = np.argsort(quality)[-pool_size:]
    objectives = _objectives(top1[pool], folds)
    dominates = np.all(
        objectives[:, None, :] >= objectives[None, :, :], axis=2
    ) & np.any(
        objectives[:, None, :] > objectives[None, :, :], axis=2
    )
    nondominated = ~np.any(dominates, axis=0)
    pareto = pool[nondominated]
    if len(pareto) >= min(pool_size, 64):
        return pareto
    supplement = pool[~np.isin(pool, pareto)]
    return np.concatenate((pareto, supplement))


def _select_diverse(
    genomes: tuple[Genome, ...],
    top1: np.ndarray,
    folds: tuple[FoldTensor, ...],
    beam_width: int,
) -> tuple[int, ...]:
    if len(genomes) <= beam_width:
        return tuple(range(len(genomes)))
    quality = _quality(top1, folds)
    weights = np.asarray([(*genome.weights, genome.market_weight) for genome in genomes])
    pool_size = min(len(genomes), max(beam_width * 16, 256))
    pool = _pareto_pool(quality, top1, folds, pool_size)
    candidate_mask = np.zeros(len(genomes), dtype=bool)
    candidate_mask[pool] = True
    selected = [int(pool[np.argmax(quality[pool])])]
    min_distance = np.full(len(genomes), np.inf)
    while len(selected) < beam_width:
        latest = weights[selected[-1]]
        min_distance = np.minimum(
            min_distance,
            np.abs(weights - latest[None, :]).sum(axis=1),
        )
        novelty = min_distance.copy()
        novelty[~candidate_mask] = -np.inf
        novelty[np.asarray(selected)] = -np.inf
        score = quality + 0.01 * novelty
        selected.append(int(np.argmax(score)))
    return tuple(selected)


def hierarchical_screen(
    genomes: tuple[Genome, ...],
    folds: tuple[FoldTensor, ...],
    batch_size: int,
    stage_beams: tuple[int, ...],
) -> tuple[tuple[AssayScore, ...], tuple[StageReport, ...]]:
    if len(folds) < 4:
        raise ValueError("hierarchical screening requires three discovery folds and confirmation")
    if not stage_beams:
        raise ValueError("at least one screening beam is required")
    survivors = genomes
    reports: list[StageReport] = []
    for stage_index, requested_beam in enumerate(stage_beams):
        fold_count = min(stage_index + 1, 3)
        stage_folds = folds[:fold_count]
        matrix = screen_top1_matrix(survivors, stage_folds, batch_size)
        beam = min(max(1, requested_beam), len(survivors))
        indices = _select_diverse(survivors, matrix, stage_folds, beam)
        best_index = int(np.argmax(_quality(matrix, stage_folds)))
        best_gap = float(
            np.min(
                matrix[best_index]
                - np.asarray([fold.market_top1 for fold in stage_folds])
            )
            * 100.0
        )
        reports.append(StageReport(
            fold_count=fold_count,
            input_count=len(survivors),
            output_count=len(indices),
            assay_evaluations=len(survivors) * fold_count,
            best_market_gap_pp=best_gap,
        ))
        survivors = tuple(survivors[index] for index in indices)
    final_scores = screen_genomes(survivors, folds, batch_size)
    return final_scores, tuple(reports)
