import numpy as np
import pandas as pd  # noqa: PANDAS_OK — project model fixtures use pandas

from kra_market_pairwise_challenger import (
    ChallengerSpec,
    build_challenger_rows,
    fit_challenger_model,
)


def _frame() -> pd.DataFrame:
    rows = []
    for race_number in range(30):
        winner = 1 if race_number % 3 else 2
        for saddlecloth, (odds, strength) in enumerate(
            ((2.0, 0.4), (3.0, 0.9), (6.0, 0.1)), start=1
        ):
            rows.append(
                {
                    "rk": f"race-{race_number}",
                    "ord": 1 if saddlecloth == winner else saddlecloth + 1,
                    "win": int(saddlecloth == winner),
                    "winOdds": odds,
                    "strength": strength,
                }
            )
    return pd.DataFrame(rows)


def test_build_challenger_rows_keeps_one_row_per_nonfavorite() -> None:
    rows, race_keys, positions = build_challenger_rows(
        _frame().iloc[:3].copy(), ["strength"], top_k=3
    )

    assert len(rows) == 2
    assert race_keys == ("race-0", "race-0")
    assert positions == (1, 2)
    assert rows["target"].tolist() == [1, 0]
    assert rows["d_strength"].iloc[0] > 0.0


def test_pairwise_challenger_can_override_favorite_with_stronger_runner() -> None:
    frame = _frame()
    spec = ChallengerSpec(top_k=3, balance_power=0.5, threshold=0.2)

    model = fit_challenger_model(frame, ["strength"], spec)
    predictions = model.predict_many(frame.iloc[:3].copy(), thresholds=(0.2, 1.1))

    assert int(np.argmax(predictions[0.2])) == 1
    assert int(np.argmax(predictions[1.1])) == 0
    assert np.isclose(predictions[0.2].sum(), 1.0)
