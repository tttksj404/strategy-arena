#!/usr/bin/env python3
"""경륜 삼쌍 보드 기반 단승/연승 픽 소스 정책 테스트."""
import copy
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine


def keirin_rows_fixture():
    return [
        {"bno": 1, "name": "모델축", "grade": "", "pwin": 0.55, "pplc": 0.80, "rank_score": 0.55},
        {"bno": 2, "name": "모델연승", "grade": "", "pwin": 0.20, "pplc": 0.76, "rank_score": 0.20},
        {"bno": 3, "name": "모델삼번", "grade": "", "pwin": 0.10, "pplc": 0.70, "rank_score": 0.10},
        {"bno": 4, "name": "시장축", "grade": "", "pwin": 0.06, "pplc": 0.45, "rank_score": 0.06},
        {"bno": 5, "name": "상대오", "grade": "", "pwin": 0.04, "pplc": 0.40, "rank_score": 0.04},
        {"bno": 6, "name": "시장연승", "grade": "", "pwin": 0.03, "pplc": 0.38, "rank_score": 0.03},
        {"bno": 7, "name": "상대칠", "grade": "", "pwin": 0.02, "pplc": 0.35, "rank_score": 0.02},
    ]


def complete_trifecta_board_fixture():
    board = {}
    for first in range(1, 8):
        for second in range(1, 8):
            for third in range(1, 8):
                if len({first, second, third}) == 3:
                    odds = 1000.0
                    if first == 4:
                        odds = 10.0
                    elif first == 6:
                        odds = 20.0
                    board[f"{first}-{second}-{third}"] = odds
    return board


def exotic_market_board_fixture():
    board = {}
    for first in range(1, 8):
        for second in range(1, 8):
            for third in range(1, 8):
                if len({first, second, third}) == 3:
                    odds = 10000.0
                    if {first, second, third} == {3, 5, 7}:
                        odds = 4.0
                    if first == 5 and second == 7:
                        odds = 1.5
                    board[f"{first}-{second}-{third}"] = odds
    return board


def predict_with_rows(monkeypatch, rows, board):
    monkeypatch.setattr(engine, "load_final_model", lambda: (None, "missing"))
    monkeypatch.setattr(engine, "load_special_11r_model", lambda: (None, "missing"))
    monkeypatch.setattr(engine, "load_11r_model", lambda: (None, "missing"))
    monkeypatch.setattr(engine, "load_cross_model", lambda: (None, "missing"))
    monkeypatch.setattr(engine, "score_keirin", lambda starters, meta=None: (copy.deepcopy(rows), None))
    return engine.predict([{"racer_no": "1"}], meta={"race_no": "1", "trifecta_board": board})


def test_predict_uses_market_for_win_and_place_when_board_complete(monkeypatch):
    rows = keirin_rows_fixture()

    monkeypatch.setenv("KEIRIN_PICK_POLICY", "market_if_board")
    result = predict_with_rows(monkeypatch, rows, complete_trifecta_board_fixture())

    assert result["pick_source"] == "market"
    assert result["picks"][0]["code"] == "단승"
    assert result["picks"][0]["pick"] == ["4번 시장축"]
    assert result["picks"][1]["code"] == "연승"
    assert result["picks"][1]["pick"] == ["4번 시장축", "6번 시장연승"]
    assert "삼쌍 배당 기준 근사" in result["picks"][0]["prob"]
    assert result["picks"][2]["code"] == "복승"
    assert result["picks"][2]["pick"] == ["4번 시장축 ↔ 6번 시장연승"]
    assert result["picks"][2]["pick_source"] == "market"


def test_predict_falls_back_to_model_when_board_missing_or_incomplete(monkeypatch):
    rows = keirin_rows_fixture()
    complete = complete_trifecta_board_fixture()
    incomplete = dict(list(complete.items())[:209])
    zero = dict(complete)
    zero["4-1-2"] = 0.0
    negative = dict(complete)
    negative["4-1-2"] = -1.0

    monkeypatch.setenv("KEIRIN_PICK_POLICY", "market_if_board")
    for board in (None, incomplete, zero, negative):
        result = predict_with_rows(monkeypatch, rows, board)
        assert result["pick_source"] == "model"
        assert result["picks"][0]["pick"] == ["1번 모델축"]
        assert result["picks"][1]["pick"] == ["1번 모델축", "2번 모델연승"]
        assert result["picks"][2]["pick_source"] == "model"


def test_model_always_keeps_model_pick_source(monkeypatch):
    rows = keirin_rows_fixture()

    monkeypatch.setenv("KEIRIN_PICK_POLICY", "model_always")
    result = predict_with_rows(monkeypatch, rows, complete_trifecta_board_fixture())

    assert result["pick_source"] == "model"
    assert result["picks"][0]["pick"] == ["1번 모델축"]
    assert result["picks"][1]["pick"] == ["1번 모델축", "2번 모델연승"]
    assert result["picks"][2]["pick"] == ["1번 모델축 ↔ 2번 모델연승"]


def test_predict_uses_trifecta_board_marginalization_for_exotics(monkeypatch):
    rows = keirin_rows_fixture()

    monkeypatch.setenv("KEIRIN_PICK_POLICY", "market_if_board")
    result = predict_with_rows(monkeypatch, rows, exotic_market_board_fixture())
    by_code = {pick["code"]: pick for pick in result["picks"]}

    assert result["pick_source"] == "market"
    assert by_code["복승"]["pick"] == ["5번 상대오 ↔ 7번 상대칠"]
    assert by_code["쌍승"]["pick"] == ["5번 상대오 → 7번 상대칠"]
    assert by_code["삼복"]["pick"] == ["3번 모델삼번 ↔ 5번 상대오 ↔ 7번 상대칠"]
    assert by_code["쌍복"]["pick"] == ["5번 상대오 ↔ 7번 상대칠"]
    assert by_code["복승"]["prob"] == "삼쌍 배당 기준 근사 · 1-2슬롯 무순 pair_mass 1위"
    assert by_code["쌍승"]["prob"] == "삼쌍 배당 기준 근사 · 1-2슬롯 ordered pair_mass 1위"
    assert by_code["삼복"]["prob"] == "삼쌍 배당 기준 근사 · 3슬롯 무순 trio_mass 1위"
    assert by_code["쌍복"]["prob"] == "삼쌍 배당 기준 근사 · 1-2슬롯 무순 pair_mass 1위"


def test_live_decision_response_uses_fetched_complete_board(monkeypatch):
    base_model = {
        "rows": keirin_rows_fixture(),
        "picks": [],
        "pick_source": "model",
    }

    monkeypatch.setenv("KCYCLE_ENABLED", "1")
    monkeypatch.setenv("KEIRIN_PICK_POLICY", "market_if_board")
    with patch.object(engine, "fetch_kcycle_odds_with_ts", return_value=(None, None)), \
         patch.object(engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(complete_trifecta_board_fixture(), "2999-01-01T00:00:00")), \
         patch.object(engine, "save_kcycle_trifecta_snapshot", return_value=True):
        result = engine.compute_live_decision(
            "keirin",
            "2999-01-01",
            "광명",
            "1",
            base_model_out=base_model,
        )

    assert result["pick_source"] == "market"
    assert result["picks"][0]["code"] == "TOP1"
    assert result["picks"][0]["selection"] == "4"
    assert result["picks"][1]["code"] == "QNL"
    assert result["picks"][1]["selection"] == "4-6"
