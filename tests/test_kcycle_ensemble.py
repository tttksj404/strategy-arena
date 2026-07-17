#!/usr/bin/env python3
import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine


def make_trifecta_board(best="1-2-3", best_odds=10.0, second="1-2-4", second_odds=11.0):
    board = {}
    base = 100.0
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                if len({a, b, c}) == 3:
                    board[f"{a}-{b}-{c}"] = base
                    base += 0.5
    board[best] = best_odds
    board[second] = second_odds
    return board


def test_kcycle_ensemble_pick_is_deterministic_for_fixed_board():
    board = make_trifecta_board(best="4-6-7", best_odds=9.1, second="4-7-6", second_odds=10.0)

    first = engine.kcycle_ensemble_trifecta_rank(board)
    second = engine.kcycle_ensemble_trifecta_rank(board)
    tier = engine.kcycle_trifecta_confidence_tier(board)

    assert first
    assert first[:5] == second[:5]
    assert first[0]["combo"] == second[0]["combo"]
    assert tier["tier"] == "T0_base"


def test_kcycle_confidence_tier_boundaries():
    t0 = make_trifecta_board(best="1-2-3", best_odds=3.01, second="1-2-4", second_odds=3.70)
    t1 = make_trifecta_board(best="1-2-3", best_odds=2.80, second="1-2-4", second_odds=3.50)
    t2 = make_trifecta_board(best="1-2-3", best_odds=1.50, second="1-2-4", second_odds=4.50)

    assert engine.kcycle_trifecta_confidence_tier(t0)["tier"] == "T0_base"
    assert engine.kcycle_trifecta_confidence_tier(t1)["tier"] == "T1_strong"
    assert engine.kcycle_trifecta_confidence_tier(t2)["tier"] == "T2_top16"


def test_kcycle_ensemble_artifact_integrity():
    payload = json.loads(Path("static/models/kcycle_trifecta_ensemble_v1.json").read_text(encoding="utf-8"))

    assert payload["schema"] == "kcycle_trifecta_ensemble_v1"
    assert len(payload["formulas"]) == 20
    assert payload["selection"]["criteria"].startswith("val-only")
    assert payload["strong_pull_tiers"]["description"] == "백테스트 13,900경주 기준 티어별 적중률"
    for formula in payload["formulas"]:
        assert set(payload["feature_names"]) == set(formula["weights"])
        assert all(math.isfinite(float(value)) for value in formula["weights"].values())


def test_live_decision_keeps_harville_fallback_when_board_missing():
    base_model = {
        "kind": "ok",
        "rows": [
            {"bno": 1, "name": "선두", "grade": "선발", "pwin": 0.60, "pplc": 0.90},
            {"bno": 2, "name": "상대1", "grade": "선발", "pwin": 0.30, "pplc": 0.70},
            {"bno": 3, "name": "상대2", "grade": "선발", "pwin": 0.20, "pplc": 0.60},
        ],
    }

    with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
         patch.object(engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2999-01-01T10:00:00")), \
         patch.object(engine, "fetch_kcycle_trifecta_board_with_ts", return_value=({}, "2999-01-01T10:00:00")):
        decision = engine.compute_live_decision("keirin", "2999-01-01", "광명", "1", base_model_out=base_model)

    by_code = {pick["code"]: pick for pick in decision["picks"]}
    assert decision["trifecta_ensemble"] is None
    assert by_code["TRI"]["selection"] == "1-2-3"


def test_live_decision_adds_ensemble_payload_when_board_exists():
    board = make_trifecta_board(best="4-6-7", best_odds=9.1, second="4-7-6", second_odds=10.0)
    base_model = {
        "kind": "ok",
        "rows": [
            {"bno": 4, "name": "선두", "grade": "선발", "pwin": 0.60, "pplc": 0.90},
            {"bno": 6, "name": "상대1", "grade": "선발", "pwin": 0.30, "pplc": 0.70},
            {"bno": 7, "name": "상대2", "grade": "선발", "pwin": 0.20, "pplc": 0.60},
        ],
    }

    with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
         patch.object(engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2999-01-01T10:00:00")), \
         patch.object(engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(board, "2999-01-01T10:00:00")), \
         patch.object(engine, "save_kcycle_trifecta_snapshot", return_value=True) as save_snapshot:
        decision = engine.compute_live_decision("keirin", "2999-01-01", "광명", "1", base_model_out=base_model)

    save_snapshot.assert_called_once()
    ensemble = decision["trifecta_ensemble"]
    assert ensemble["source"] == "ensemble_v1"
    assert ensemble["pick"] == engine.kcycle_ensemble_trifecta_rank(board)[0]["combo"]
    assert ensemble["selection"] == "ensemble_v1_top1"
    assert ensemble["board_complete"] is True
    picks = {pick["code"]: pick for pick in decision["picks"]}
    assert picks["TRI"]["selection"] == ensemble["pick"]
    assert picks["TRB"]["selection"] == "-".join(sorted(ensemble["pick"].split("-"), key=int))
    assert len(ensemble["top5"]) == 5
    assert ensemble["tier"] in {"T0_base", "T1_strong", "T2_top16"}
