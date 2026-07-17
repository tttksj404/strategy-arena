import numpy as np
import pandas as pd  # noqa: PANDAS_OK — project model fixtures use pandas

from kra_market_context_gates import generate_context_gates


def test_context_gate_changes_only_matching_races() -> None:
    frame = pd.DataFrame(
        {
            "rk": ["a", "a", "b", "b"],
            "meet": ["1", "1", "2", "2"],
            "rcDist": [1200, 1200, 1800, 1800],
            "winOdds": [2.0, 4.0, 2.0, 5.0],
        }
    )
    market = np.asarray([0.6, 0.4, 0.7, 0.3])
    candidate = np.asarray([0.4, 0.6, 0.3, 0.7])
    gate = next(gate for gate in generate_context_gates() if gate.name == "meet_1")

    gated = gate.apply(frame, candidate, market)

    assert gated.tolist() == [0.4, 0.6, 0.7, 0.3]
    assert gate.mask(frame, market).tolist() == [True, True, False, False]


def test_context_library_is_broad_and_contains_interactions() -> None:
    names = {gate.name for gate in generate_context_gates()}

    assert len(names) >= 20
    assert "all" in names
    assert "meet_1_distance_sprint" in names
    assert "meet_3_uncertainty_high" in names
