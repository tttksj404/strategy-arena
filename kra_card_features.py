from __future__ import annotations

from typing import Final

import pandas as pd  # noqa: PANDAS_OK — shared scikit-learn feature-frame contract


OWNER_BASELINE: Final = 0.10
CARD_FEATURES: Final = (
    "jk_weight_allowance",
    "owner_win_prior",
    "tool_mesh_eye",
    "tool_mesh",
    "tool_eye_mask",
    "tool_tongue_tie",
    "tool_approved_shoe",
    "tool_triabit",
)


def _owner_prior(frame: pd.DataFrame) -> pd.Series:
    daily = (
        frame.groupby(["owNo", "rcDate"], dropna=False)["win"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(["owNo", "rcDate"])
    )
    grouped = daily.groupby("owNo", dropna=False)
    prior_wins = grouped["sum"].cumsum() - daily["sum"]
    prior_starts = grouped["count"].cumsum() - daily["count"]
    daily["prior"] = (prior_wins / prior_starts).fillna(OWNER_BASELINE)
    lookup = daily.set_index(["owNo", "rcDate"])["prior"]
    values = [
        lookup.get((owner, date), OWNER_BASELINE)
        for owner, date in zip(frame["owNo"], frame["rcDate"])
    ]
    return pd.Series(values, index=frame.index, dtype=float)


def add_card_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["jk_weight_allowance"] = pd.to_numeric(
        result.get("wgJk"), errors="coerce"
    ).fillna(0.0)
    result["owner_win_prior"] = _owner_prior(result)
    tools = result.get("hrTool", pd.Series("", index=result.index)).fillna("").astype(str)
    patterns = {
        "tool_mesh_eye": "망사눈",
        "tool_mesh": "망사",
        "tool_eye_mask": "눈가면",
        "tool_tongue_tie": "혀끈",
        "tool_approved_shoe": "승인편자",
        "tool_triabit": "Triabit",
    }
    for output, pattern in patterns.items():
        result[output] = tools.str.contains(pattern, regex=False).astype(float)
    return result
