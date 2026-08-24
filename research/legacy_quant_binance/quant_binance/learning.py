from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_binance.models import DecisionIntent
from quant_binance.strategy.edge import ConditionalEdgeLookup


@dataclass(frozen=True)
class LearningUpdate:
    observation_count: int
    symbols: tuple[str, ...]
    updated_path: Path


class OnlineEdgeLearner:
    def __init__(self, *, min_observations: int = 5) -> None:
        self.lookup = ConditionalEdgeLookup(min_observations=min_observations)

    def load_from_export(self, path: str | Path) -> int:
        """Restore learning state from a previous edge_table.json export.
        Returns number of observations restored."""
        export_path = Path(path)
        if not export_path.exists():
            return 0
        try:
            raw = export_path.read_text(encoding="utf-8")
            if not raw.strip():
                return 0
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return 0
        if not isinstance(data, dict):
            return 0
        symbols = data.get("symbols", {})
        if not isinstance(symbols, dict):
            return 0
        # Sanity check: observation_count should be positive
        if int(data.get("observation_count", 0) or 0) <= 0:
            return 0
        restored = 0
        for symbol, rows in symbols.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                mode = row.get("mode", "")
                bucket = row.get("bucket", 0)
                direction = row.get("trend_direction", 1)
                observations = row.get("observations", [])
                if not isinstance(observations, list) or not mode:
                    continue
                for obs in observations:
                    if isinstance(obs, (int, float)):
                        self.lookup.add_observation(
                            symbol=symbol,
                            mode=mode,
                            predictability_score=bucket * 10 + 5,  # bucket midpoint
                            trend_direction=direction,
                            forward_return_bps=float(obs),
                        )
                        restored += 1
        return restored

    def ingest_decisions(
        self,
        decisions: list[DecisionIntent] | tuple[DecisionIntent, ...],
        *,
        realized_return_override_bps: dict[str, float] | None = None,
    ) -> int:
        count = 0
        for decision in decisions:
            if decision.final_mode not in {"spot", "futures"}:
                continue
            realized_bps = (
                realized_return_override_bps.get(decision.decision_id)
                if realized_return_override_bps is not None
                else None
            )
            if realized_bps is None:
                # Skip: do NOT inject prediction as realized value.
                # Real observations come from ingest_closed_trade() instead.
                continue
            self.lookup.add_observation(
                symbol=decision.symbol,
                mode=decision.final_mode,
                predictability_score=decision.predictability_score,
                trend_direction=decision.trend_direction or 1,
                forward_return_bps=realized_bps,
            )
            count += 1
        return count

    def ingest_closed_trade(
        self,
        *,
        symbol: str,
        mode: str,
        side: str,
        entry_predictability_score: float,
        realized_return_bps: float,
    ) -> None:
        """Feed actual realized return from a closed trade into the edge lookup."""
        if mode not in {"spot", "futures"}:
            return
        trend_direction = 1 if side == "long" else -1
        self.lookup.add_observation(
            symbol=symbol,
            mode=mode,
            predictability_score=entry_predictability_score,
            trend_direction=trend_direction,
            forward_return_bps=realized_return_bps,
        )

    def export(self, path: str | Path) -> LearningUpdate:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        symbol_rows: dict[str, list[dict[str, Any]]] = {}
        observation_count = 0
        for (symbol, mode, bucket, direction), values in self.lookup._symbol_buckets.items():
            symbol_rows.setdefault(symbol, []).append(
                {
                    "mode": mode,
                    "bucket": bucket,
                    "trend_direction": direction,
                    "observations": values,
                    "median_bps": sorted(values)[len(values) // 2],
                }
            )
            observation_count += len(values)
        output_path.write_text(
            json.dumps(
                {
                    "observation_count": observation_count,
                    "symbols": symbol_rows,
                    "diagnostics": self.lookup.diagnostics(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return LearningUpdate(
            observation_count=observation_count,
            symbols=tuple(sorted(symbol_rows.keys())),
            updated_path=output_path,
        )
