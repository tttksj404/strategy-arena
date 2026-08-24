from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_binance.models import DecisionIntent


@dataclass(frozen=True)
class OracleSegment:
    start: datetime
    end: datetime
    expected_mode: str
    expected_side: str
    note: str


@dataclass(frozen=True)
class OracleReport:
    matched_segments: int
    total_segments: int
    segment_accuracy: float


def load_oracle(path: str | Path) -> list[OracleSegment]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        OracleSegment(
            start=datetime.fromisoformat(item["start"]),
            end=datetime.fromisoformat(item["end"]),
            expected_mode=item["expected_mode"],
            expected_side=item["expected_side"],
            note=item.get("note", ""),
        )
        for item in payload["segments"]
    ]


def compare_decisions_to_oracle(
    decisions: list[DecisionIntent] | tuple[DecisionIntent, ...],
    oracle_segments: list[OracleSegment],
) -> OracleReport:
    matched = 0
    for segment in oracle_segments:
        segment_decisions = [
            decision
            for decision in decisions
            if segment.start <= decision.timestamp <= segment.end
        ]
        if not segment_decisions:
            continue
        if all(
            decision.final_mode == segment.expected_mode
            and decision.side == segment.expected_side
            for decision in segment_decisions
        ):
            matched += 1
    total = len(oracle_segments)
    accuracy = matched / total if total else 0.0
    return OracleReport(matched_segments=matched, total_segments=total, segment_accuracy=accuracy)
