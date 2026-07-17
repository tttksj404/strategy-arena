import numpy as np
import pandas as pd  # noqa: PANDAS_OK — project model fixtures use pandas

from kra_full_order_rankers import (
    FullOrderSpec,
    build_full_order_pairs,
    fit_full_order_model,
)


def _training_frame() -> pd.DataFrame:
    rows = []
    for race_number in range(12):
        for order, strength in enumerate((3.0, 2.0, 1.0), start=1):
            rows.append(
                {
                    "rk": f"race-{race_number}",
                    "ord": order,
                    "strength": strength,
                }
            )
    return pd.DataFrame(rows)


def test_build_full_order_pairs_uses_every_observed_precedence() -> None:
    frame = _training_frame().iloc[:3].copy()

    pairs = build_full_order_pairs(frame, ["strength"])

    assert len(pairs.values) == 6
    assert pairs.targets.tolist() == [1, 0, 1, 0, 1, 0]
    assert pairs.values.iloc[0, 0] > 0.0


def test_plackett_luce_uses_finishing_order_to_recover_race_ranking() -> None:
    frame = _training_frame()
    spec = FullOrderSpec.plackett_luce("plackett_test", ridge=0.1, maximum_rank=3)

    model = fit_full_order_model(frame, ["strength"], spec)
    probability = model.predict(frame.iloc[:3].copy())

    assert np.argsort(-probability).tolist() == [0, 1, 2]
    assert np.isclose(probability.sum(), 1.0)


def test_full_order_pairwise_uses_finishing_order_to_recover_race_ranking() -> None:
    frame = _training_frame()
    spec = FullOrderSpec.pairwise("pairwise_test", depth=2, minimum_leaf=2)

    model = fit_full_order_model(frame, ["strength"], spec)
    probability = model.predict(frame.iloc[:3].copy())

    assert np.argsort(-probability).tolist() == [0, 1, 2]
    assert np.isclose(probability.sum(), 1.0)
