from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kra_drug_discovery import AssayScore, Genome


@dataclass(frozen=True, slots=True)
class FoldTensor:
    name: str
    model_scores: np.ndarray
    market_scores: np.ndarray
    winner_positions: np.ndarray
    v4_top1: float
    market_top1: float


def build_fold_tensor(
    name: str,
    frame: pd.DataFrame,
    model_scores: np.ndarray,
    market_scores: np.ndarray,
    v4_model_index: int,
) -> FoldTensor:
    races = list(frame.groupby("rk", sort=False))
    maximum_field = max(len(race) for _, race in races)
    tensor = np.full(
        (model_scores.shape[0], len(races), maximum_field),
        -1.0,
        dtype=float,
    )
    market_tensor = np.full((len(races), maximum_field), -1.0, dtype=float)
    winner_positions = np.zeros(len(races), dtype=int)
    row_positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for race_index, (_, race) in enumerate(races):
        positions = row_positions.loc[race.index].to_numpy(dtype=int)
        field_size = len(positions)
        tensor[:, race_index, :field_size] = model_scores[:, positions]
        market_tensor[race_index, :field_size] = market_scores[positions]
        winners = np.flatnonzero(race["win"].to_numpy(dtype=int) == 1)
        winner_positions[race_index] = int(winners[0])
    v4_picks = tensor[v4_model_index].argmax(axis=1)
    market_picks = market_tensor.argmax(axis=1)
    return FoldTensor(
        name=name,
        model_scores=tensor,
        market_scores=market_tensor,
        winner_positions=winner_positions,
        v4_top1=float(np.mean(v4_picks == winner_positions)),
        market_top1=float(np.mean(market_picks == winner_positions)),
    )


def _top1_batch(
    genomes: tuple[Genome, ...],
    fold: FoldTensor,
) -> np.ndarray:
    weights = np.asarray([genome.weights for genome in genomes], dtype=float)
    market_weights = np.asarray(
        [genome.market_weight for genome in genomes], dtype=float
    )
    combined = np.einsum("bm,mrf->brf", weights, fold.model_scores)
    combined += market_weights[:, None, None] * fold.market_scores[None, :, :]
    picks = combined.argmax(axis=2)
    return np.mean(picks == fold.winner_positions[None, :], axis=1)


def screen_genomes(
    genomes: tuple[Genome, ...],
    folds: tuple[FoldTensor, ...],
    batch_size: int,
) -> tuple[AssayScore, ...]:
    fold_top1 = screen_top1_matrix(genomes, folds, batch_size)
    scores = []
    for index, genome in enumerate(genomes):
        market_gaps = tuple(
            (fold_top1[index, fold_index] - fold.market_top1) * 100.0
            for fold_index, fold in enumerate(folds)
        )
        v4_lifts = tuple(
            (fold_top1[index, fold_index] - fold.v4_top1) * 100.0
            for fold_index, fold in enumerate(folds)
        )
        scores.append(AssayScore(
            genome=genome,
            discovery_market_gaps_pp=market_gaps[:-1],
            discovery_v4_lifts_pp=v4_lifts[:-1],
            confirmation_market_gap_pp=market_gaps[-1],
            confirmation_v4_lift_pp=v4_lifts[-1],
        ))
    return tuple(scores)


def screen_top1_matrix(
    genomes: tuple[Genome, ...],
    folds: tuple[FoldTensor, ...],
    batch_size: int,
) -> np.ndarray:
    fold_top1 = np.zeros((len(genomes), len(folds)), dtype=float)
    for start in range(0, len(genomes), batch_size):
        stop = min(start + batch_size, len(genomes))
        batch = genomes[start:stop]
        for fold_index, fold in enumerate(folds):
            fold_top1[start:stop, fold_index] = _top1_batch(batch, fold)
    return fold_top1
