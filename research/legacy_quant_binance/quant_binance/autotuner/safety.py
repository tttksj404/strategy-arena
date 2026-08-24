"""Safety gates and revert monitoring for auto-tuning."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass
class RevertMonitorState:
    change_id: str = ""
    applied_at: str = ""
    deltas: list[dict[str, Any]] = field(default_factory=list)
    pre_change_avg_pnl: float = 0.0
    pre_change_max_consecutive_loss: int = 0
    pre_change_avg_loss_magnitude: float = 0.0
    trades_since: int = 0
    pnl_since: float = 0.0
    max_drawdown_since: float = 0.0
    peak_equity_since: float = 0.0
    consecutive_losses: int = 0
    revert_triggered: bool = False
    revert_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "applied_at": self.applied_at,
            "deltas": self.deltas,
            "pre_change_avg_pnl": self.pre_change_avg_pnl,
            "pre_change_max_consecutive_loss": self.pre_change_max_consecutive_loss,
            "pre_change_avg_loss_magnitude": self.pre_change_avg_loss_magnitude,
            "trades_since": self.trades_since,
            "pnl_since": self.pnl_since,
            "max_drawdown_since": self.max_drawdown_since,
            "peak_equity_since": self.peak_equity_since,
            "consecutive_losses": self.consecutive_losses,
            "revert_triggered": self.revert_triggered,
            "revert_reason": self.revert_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RevertMonitorState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


REVERT_TRADE_THRESHOLD = 30


class RevertMonitor:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.state = self._load()

    def _load(self) -> RevertMonitorState:
        if not self.state_path.exists():
            return RevertMonitorState()
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            return RevertMonitorState.from_dict(d)
        except (json.JSONDecodeError, OSError):
            return RevertMonitorState()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @property
    def active(self) -> bool:
        return bool(self.state.change_id) and not self.state.revert_triggered

    def start_monitoring(
        self,
        *,
        change_id: str,
        deltas: list[dict[str, Any]],
        pre_trades: list[dict[str, Any]],
    ) -> None:
        pnls = [float(t.get("realized_pnl_usd_estimate", 0) or 0) for t in pre_trades]
        losses = [p for p in pnls if p < 0]
        # Compute max consecutive losses
        max_consec = 0
        consec = 0
        for p in pnls:
            if p < 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        self.state = RevertMonitorState(
            change_id=change_id,
            applied_at=datetime.now(tz=timezone.utc).isoformat(),
            deltas=deltas,
            pre_change_avg_pnl=sum(pnls) / max(len(pnls), 1),
            pre_change_max_consecutive_loss=max_consec,
            pre_change_avg_loss_magnitude=sum(abs(l) for l in losses) / max(len(losses), 1),
        )
        self._save()

    def record_trade(self, pnl_usd: float) -> str | None:
        """Record a trade and check revert triggers. Returns reason if revert needed."""
        if not self.active:
            return None

        self.state.trades_since += 1
        self.state.pnl_since += pnl_usd
        self.state.peak_equity_since = max(self.state.peak_equity_since, self.state.pnl_since)
        drawdown = self.state.peak_equity_since - self.state.pnl_since
        self.state.max_drawdown_since = max(self.state.max_drawdown_since, drawdown)

        if pnl_usd < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Check revert triggers only after threshold trades
        reason = None
        if self.state.trades_since >= REVERT_TRADE_THRESHOLD:
            avg_pnl_post = self.state.pnl_since / self.state.trades_since
            # Trigger 1: avg PnL dropped to < 50% of pre-change
            if self.state.pre_change_avg_pnl > 0 and avg_pnl_post < self.state.pre_change_avg_pnl * 0.5:
                reason = f"avg_pnl_dropped: {avg_pnl_post:.4f} < {self.state.pre_change_avg_pnl * 0.5:.4f}"

        # Trigger 2: 5 consecutive losses (if pre-change max was < 3)
        if self.state.consecutive_losses >= 5 and self.state.pre_change_max_consecutive_loss < 3:
            reason = f"consecutive_losses: {self.state.consecutive_losses} (pre: {self.state.pre_change_max_consecutive_loss})"

        # Trigger 3: single loss > 2x pre-change average loss
        if pnl_usd < 0 and self.state.pre_change_avg_loss_magnitude > 0:
            if abs(pnl_usd) > self.state.pre_change_avg_loss_magnitude * 2:
                reason = f"extreme_loss: {pnl_usd:.4f} > 2x avg {self.state.pre_change_avg_loss_magnitude:.4f}"

        if reason:
            self.state.revert_triggered = True
            self.state.revert_reason = reason

        self._save()
        return reason

    def clear(self) -> None:
        self.state = RevertMonitorState()
        self._save()


def validate_deltas(
    deltas: list[dict[str, Any]],
    *,
    total_trades: int,
    audit_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter deltas through safety gates. Returns (approved, rejection_reasons)."""
    approved = []
    reasons = []

    # Gate 1: Global minimum trades
    if total_trades < 50:
        return [], [f"insufficient_trades: {total_trades} < 50"]

    # Load recent audit for revert history
    recent_reverts: set[str] = set()
    if audit_path.exists():
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=48)
        try:
            for line in audit_path.open(encoding="utf-8"):
                entry = json.loads(line)
                if entry.get("action") == "revert":
                    ts = entry.get("timestamp", "")
                    if ts and datetime.fromisoformat(ts) > cutoff:
                        for d in entry.get("deltas", []):
                            recent_reverts.add(str(d.get("path", "")))
        except (json.JSONDecodeError, OSError):
            pass

    has_tier1 = False
    has_tier3 = False

    for delta in deltas:
        path_str = str(delta.get("path", ""))
        param_min_trades = delta.get("min_trades", 50)
        confidence = delta.get("confidence", 0.0)
        risk_tier = delta.get("risk_tier", "medium")

        # Gate 2: Per-param minimum trades
        if total_trades < param_min_trades:
            reasons.append(f"{path_str}: trades {total_trades} < {param_min_trades}")
            continue

        # Gate 3: Confidence threshold
        threshold = 0.8 if risk_tier == "high" else 0.6
        if confidence < threshold:
            reasons.append(f"{path_str}: confidence {confidence:.2f} < {threshold}")
            continue

        # Gate 4: Recent revert check
        if path_str in recent_reverts:
            reasons.append(f"{path_str}: recently reverted, blocked for 48h")
            continue

        # Track tiers for mutual exclusion
        if "mode_thresholds" in path_str or "futures_exposure" in path_str:
            has_tier1 = True
        if "risk" in path_str and "per_trade" in path_str:
            has_tier3 = True

        approved.append(delta)

    # Gate 5: Mutual exclusion (entry thresholds + sizing)
    if has_tier1 and has_tier3:
        approved = [d for d in approved if "per_trade" not in str(d.get("path", ""))]
        reasons.append("mutual_exclusion: removed Tier3 (sizing) because Tier1 (entry) present")

    # Rate limit: max 3 changes per cycle
    if len(approved) > 3:
        approved = approved[:3]
        reasons.append(f"rate_limit: capped to 3 (had {len(deltas)})")

    return approved, reasons
