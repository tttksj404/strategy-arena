import json
from pathlib import Path

import engine


def _starters(odds: list[float]) -> list[dict[str, str]]:
    return [
        {"chulNo": str(index), "hrName": f"H{index}", "winOdds": str(odd)}
        for index, odd in enumerate(odds, start=1)
    ]


def _base_live_out(starters: list[dict[str, str]]) -> dict:
    return {
        "rows": [
            {"bno": 1, "name": "H1", "pwin": 0.55, "pplc": 0.8, "rank_score": 0.55},
            {"bno": 2, "name": "H2", "pwin": 0.45, "pplc": 0.7, "rank_score": 0.45},
        ],
        "top": {"bno": 1, "name": "H1", "pwin": 0.55, "pplc": 0.8, "rank_score": 0.55},
        "picks": [],
        "_participant_sources": {},
        "_kra_starters": starters,
        "roster_verification": {"state": "unverified", "official_names": [], "checked_at": ""},
    }


def test_kra_confidence_tier_boundaries_and_field_buckets() -> None:
    cases = [
        ("very_strong_pull", "field_le_7", [1.8, 2.7, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ("strong_pull", "field_8_10", [2.5, 3.25, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        ("price_short", "field_8_10", [2.0, 2.4, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ("gap_wide", "field_11_plus", [3.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
        ("weak_or_open", "field_11_plus", [4.01, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
    ]

    for tier, bucket, odds in cases:
        result = engine.kra_confidence_tier(_starters(odds))

        assert result is not None
        assert result["tier"] == tier
        assert result["field_bucket"] == bucket

    assert engine.kra_confidence_tier(_starters([1.8001, 2.70015, 5.0]))["tier"] == "strong_pull"
    assert engine.kra_confidence_tier(_starters([2.5001, 3.25013, 5.0]))["tier"] == "all"
    assert engine.kra_confidence_tier(_starters([2.0001, 2.4, 5.0]))["tier"] == "all"
    assert engine.kra_confidence_tier(_starters([3.0, 4.499, 5.0]))["tier"] == "all"
    assert engine.kra_confidence_tier(_starters([4.0, 4.4, 5.0]))["tier"] == "all"


def test_kra_confidence_tier_returns_none_for_missing_or_invalid_win_odds() -> None:
    invalid_cases = [
        [],
        [{"chulNo": "1", "hrName": "H1"}],
        [{"chulNo": "1", "winOdds": "2.0"}, {"chulNo": "2"}],
        [{"chulNo": "1", "winOdds": "0"}, {"chulNo": "2", "winOdds": "3.0"}],
        [{"chulNo": "1", "winOdds": "-1"}, {"chulNo": "2", "winOdds": "3.0"}],
    ]

    for starters in invalid_cases:
        assert engine.kra_confidence_tier(starters) is None


def test_kra_confidence_tier_artifact_integrity() -> None:
    payload = json.loads(Path(engine.KRA_CONFIDENCE_TIERS_PATH).read_text(encoding="utf-8"))
    thresholds = payload["thresholds"]

    assert thresholds["very_strong_pull"]["favorite_odds_max"] <= thresholds["price_short"]["favorite_odds_max"]
    assert thresholds["price_short"]["favorite_odds_max"] <= thresholds["strong_pull"]["favorite_odds_max"]
    assert thresholds["weak_or_open"]["ratio12_max_exclusive"] < thresholds["strong_pull"]["ratio12_min"]
    assert thresholds["strong_pull"]["ratio12_min"] < thresholds["gap_wide"]["ratio12_min"]
    assert payload["corpus"]["races"] == 6249
    assert payload["corpus"]["entries"] == 64548

    for bucket in payload["win_market"]["by_field_size"].values():
        for metrics in bucket.values():
            assert 0.0 <= metrics["coverage"] <= 1.0
            assert 0.0 <= metrics["top1"] <= 1.0
            assert 0.0 <= metrics["top3"] <= 1.0


def test_kra_live_decision_adds_market_confidence_when_win_odds_complete() -> None:
    decision = engine.compute_live_decision(
        "horse",
        "2026-07-12",
        "서울",
        "1",
        base_model_out=_base_live_out(_starters([1.8, 2.7, 5.0, 6.0, 7.0, 8.0, 9.0])),
    )

    assert decision["market_confidence"] == {
        "tier": "very_strong_pull",
        "field_bucket": "field_le_7",
        "historical_top1": 0.4583333333333333,
        "historical_top3": 0.7083333333333334,
        "coverage": 0.3287671232876712,
        "source": "kra_tiers_v1",
    }


def test_kra_live_decision_omits_market_confidence_when_win_odds_incomplete() -> None:
    starters = [{"chulNo": "1", "winOdds": "2.0"}, {"chulNo": "2"}]

    decision = engine.compute_live_decision(
        "horse",
        "2026-07-12",
        "서울",
        "1",
        base_model_out=_base_live_out(starters),
    )

    assert "market_confidence" not in decision
