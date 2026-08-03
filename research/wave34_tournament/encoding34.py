# Wave-34 vector encoding. A fixed-length real vector in [0,1]^D <-> wave30 Genome.
# (Do not write the word "coding" followed by a colon on line 1 or 2 -- PEP 263 reads it as a
#  source-encoding cookie and the file stops importing.)
#
# Every gradient-free optimizer in this tournament (CMA-ES, PSO, SA, TPE, MCTS) natively
# searches a real box. The wave30 Genome is a mix of categorical, integer-choice and coupled
# continuous genes, so the tournament needs ONE shared bijection-ish map. Two rules make the
# comparison honest:
#
#   1. EVERY point of [0,1]^D decodes to a genome that passes Genome.validate(). This is not
#      achieved by repairing an invalid draw -- repair would silently move different methods'
#      proposals to different places and make the "same search space" claim false. It is
#      achieved by REPARAMETERISING the coupled pair: dimension `stop_u` is not stop_pct, it
#      is the position INSIDE the feasible stop interval [min_stop(risk), max_stop(risk)]
#      that genome30 already solves in closed form. So the constraint cannot be violated.
#   2. Decoding is deterministic and stateless. Same vector -> same genome, for every method.
#
# Categorical genes use floor(u * k) with u clipped below 1, i.e. equal-width bins, so a
# uniform draw over the box is a uniform draw over the categories -- the random control is
# then exactly the same distribution genome30.random_genome samples (verified in tests).

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final, Sequence

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.genome30 import (
    CONCURRENCY_CHOICES,
    COOLDOWN_CHOICES,
    ENTRY_THRESHOLD_RANGE,
    Genome,
    LOOKBACK_CHOICES,
    MAX_HOLD_CHOICES,
    RISK_FRAC_RANGE,
    SIGNAL_FAMILIES,
    SLEEVE_FRACTION_CHOICES,
    SYMBOL_SETS,
    TARGET_R_RANGE,
    _max_stop_for_risk,
    _min_stop_for_risk,
)

# Dimension order is frozen: every optimizer, every checkpoint and every report row uses it.
DIM_NAMES: Final = (
    "signal_family",
    "lookback_bars",
    "entry_threshold",
    "risk_frac",
    "stop_u",  # position inside the feasible stop interval, NOT stop_pct
    "target_r",
    "trail_enabled",
    "max_hold_bars",
    "allow_short",
    "symbols",
    "max_concurrent",
    "cooldown_bars_after_loss",
    "sleeve_fraction",
)
DIM: Final = len(DIM_NAMES)


def _pick(u: float, options: Sequence):
    """Equal-width binning of [0,1] onto `options`. u==1.0 maps to the last bin, not out."""
    index = int(np.floor(np.clip(u, 0.0, 1.0 - 1e-12) * len(options)))
    return options[index]


def _lerp(u: float, low: float, high: float) -> float:
    return float(low + np.clip(u, 0.0, 1.0) * (high - low))


def clip_vector(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def decode(x: np.ndarray) -> Genome:
    """[0,1]^DIM -> a validated Genome. Never raises for an in-box vector."""
    u = clip_vector(x)
    if u.shape != (DIM,):
        raise ValueError(f"expected shape ({DIM},), got {u.shape}")

    risk_frac = _lerp(u[3], *RISK_FRAC_RANGE)
    low = _min_stop_for_risk(risk_frac)
    high = _max_stop_for_risk(risk_frac)
    # low <= high holds over the whole risk range (checked in the feasibility probe); the max()
    # is a belt-and-braces guard so a future range change degrades to the boundary rather than
    # to an inverted interval.
    stop_pct = _lerp(u[4], low, max(low, high))

    genome = Genome(
        signal_family=_pick(u[0], SIGNAL_FAMILIES),
        lookback_bars=int(_pick(u[1], LOOKBACK_CHOICES)),
        entry_threshold=_lerp(u[2], *ENTRY_THRESHOLD_RANGE),
        stop_pct=stop_pct,
        target_r=_lerp(u[5], *TARGET_R_RANGE),
        trail_enabled=bool(u[6] >= 0.5),
        risk_frac=risk_frac,
        max_hold_bars=int(_pick(u[7], MAX_HOLD_CHOICES)),
        allow_short=bool(u[8] >= 0.5),
        symbols=_pick(u[9], SYMBOL_SETS),
        max_concurrent=int(_pick(u[10], CONCURRENCY_CHOICES)),
        cooldown_bars_after_loss=int(_pick(u[11], COOLDOWN_CHOICES)),
        sleeve_fraction=float(_pick(u[12], SLEEVE_FRACTION_CHOICES)),
    )
    return genome.validate()


def encode(genome: Genome) -> np.ndarray:
    """Genome -> a vector that decodes back to it. Used to seed optimizers from a known point.

    Categorical dimensions return the bin CENTRE, so a subsequent small perturbation stays in
    the same category instead of flipping on the first mutation.
    """
    def centre(value, options) -> float:
        index = list(options).index(value)
        return (index + 0.5) / len(options)

    def inv(value: float, low: float, high: float) -> float:
        return float(np.clip((value - low) / (high - low), 0.0, 1.0)) if high > low else 0.5

    low = _min_stop_for_risk(genome.risk_frac)
    high = _max_stop_for_risk(genome.risk_frac)
    return np.array(
        [
            centre(genome.signal_family, SIGNAL_FAMILIES),
            centre(genome.lookback_bars, LOOKBACK_CHOICES),
            inv(genome.entry_threshold, *ENTRY_THRESHOLD_RANGE),
            inv(genome.risk_frac, *RISK_FRAC_RANGE),
            inv(genome.stop_pct, low, high),
            inv(genome.target_r, *TARGET_R_RANGE),
            0.75 if genome.trail_enabled else 0.25,
            centre(genome.max_hold_bars, MAX_HOLD_CHOICES),
            0.75 if genome.allow_short else 0.25,
            centre(genome.symbols, SYMBOL_SETS),
            centre(genome.max_concurrent, CONCURRENCY_CHOICES),
            centre(genome.cooldown_bars_after_loss, COOLDOWN_CHOICES),
            centre(genome.sleeve_fraction, SLEEVE_FRACTION_CHOICES),
        ],
        dtype=float,
    )


def feasibility_probe(n: int = 200_000, seed: int = 34_000) -> dict:
    """Decode n uniform points and count validate() failures. Reported in SPEC/REPORT."""
    rng = np.random.default_rng(seed)
    failures = 0
    first_failure = None
    leverages = np.empty(n, dtype=float)
    for i in range(n):
        x = rng.random(DIM)
        try:
            genome = decode(x)
        except Exception as exc:  # noqa: BLE001 -- probe reports whatever escapes
            failures += 1
            if first_failure is None:
                first_failure = (x.tolist(), repr(exc))
            leverages[i] = np.nan
            continue
        leverages[i] = genome.leverage
    return {
        "n": n,
        "failures": failures,
        "feasible_rate": 1.0 - failures / n,
        "first_failure": first_failure,
        "leverage_min": float(np.nanmin(leverages)),
        "leverage_median": float(np.nanmedian(leverages)),
        "leverage_max": float(np.nanmax(leverages)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(feasibility_probe(int(sys.argv[1]) if len(sys.argv) > 1 else 200_000), indent=2))
