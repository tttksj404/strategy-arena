from __future__ import annotations

from collections import defaultdict
from statistics import median


def score_bucket(score: float) -> int:
    capped = max(0.0, min(score, 99.9999))
    return int(capped // 10)


class ConditionalEdgeLookup:
    def __init__(self, min_observations: int) -> None:
        self.min_observations = min_observations
        self._symbol_buckets: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
        self._pooled_buckets: dict[tuple[str, int, int], list[float]] = defaultdict(list)

    def add_observation(
        self,
        *,
        symbol: str,
        mode: str,
        predictability_score: float,
        trend_direction: int,
        forward_return_bps: float,
    ) -> None:
        bucket = score_bucket(predictability_score)
        key = (symbol, mode, bucket, trend_direction)
        pooled_key = (mode, bucket, trend_direction)
        self._symbol_buckets[key].append(forward_return_bps)
        self._pooled_buckets[pooled_key].append(forward_return_bps)

    def expected_edge_bps(
        self,
        *,
        symbol: str,
        mode: str,
        predictability_score: float,
        trend_direction: int,
    ) -> float | None:
        bucket = score_bucket(predictability_score)
        key = (symbol, mode, bucket, trend_direction)
        values = self._symbol_buckets.get(key, [])
        if len(values) >= self.min_observations:
            return float(median(values))
        pooled_values = self._pooled_buckets.get((mode, bucket, trend_direction), [])
        if len(pooled_values) >= self.min_observations:
            return float(median(pooled_values))
        return None

    def diagnostics(self) -> dict[str, object]:
        """Return observation counts for monitoring learning health."""
        total_symbol = sum(len(v) for v in self._symbol_buckets.values())
        total_pooled = sum(len(v) for v in self._pooled_buckets.values())
        qualified_symbol = sum(1 for v in self._symbol_buckets.values() if len(v) >= self.min_observations)
        qualified_pooled = sum(1 for v in self._pooled_buckets.values() if len(v) >= self.min_observations)
        return {
            "total_observations": total_symbol,
            "symbol_buckets": len(self._symbol_buckets),
            "symbol_buckets_qualified": qualified_symbol,
            "pooled_buckets": len(self._pooled_buckets),
            "pooled_buckets_qualified": qualified_pooled,
            "min_observations_threshold": self.min_observations,
            "learning_active": qualified_symbol > 0 or qualified_pooled > 0,
        }
