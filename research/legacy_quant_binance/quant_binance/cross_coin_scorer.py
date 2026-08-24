"""Cross-coin OI quadrant EV scorer.

Loads the EV table (`quant_runtime_paper50/cross_coin_ev_table.json`),
computes each candidate symbol's quadrant from the most recent two cycle
snapshots, looks up the matching scenario, returns ranked candidates.

Conflict rules (committed 2026-04-26):
- 1: most-specific match wins (leader_quadrant > leader_dir > none)
- 2: drop unmatched candidates this cycle (no queue, recompute next cycle)
- 3: hold during open trade — caller decides; this module returns scores only
- 4: 15-min staleness expiry on cycle snapshot
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _quadrant(price_up: bool, oi_up: bool) -> str:
    if price_up and oi_up:
        return "newLongs"
    if price_up and not oi_up:
        return "shortCover"
    if not price_up and oi_up:
        return "newShorts"
    return "longUnwind"


def _direction(price_up: bool) -> str:
    return "up" if price_up else "down"


@dataclass(frozen=True)
class CycleSnapshot:
    """A single (symbol, ts) cycle row from bitget_external_alpha_shadow."""

    symbol: str
    ts_ms: int
    open_interest: float
    last_price: float


@dataclass(frozen=True)
class SymbolState:
    """Computed state for one symbol from the latest two cycles."""

    symbol: str
    cycle_ts_ms: int
    own_quadrant: str
    own_dir: str
    oi_delta_pct: float
    price_delta_bps: float


@dataclass(frozen=True)
class Score:
    """Result of scoring one candidate."""

    symbol: str
    side: str
    ev_bps: float
    rank: int
    n: int
    fwd_60m_bps: float
    winrate: float
    matched_scenario: dict[str, Any]
    own_state: SymbolState
    leader_state: SymbolState | None
    stale: bool
    blocker_match: dict[str, Any] | None  # if matched a blocker scenario


@dataclass(frozen=True)
class Blocker:
    """A negative-EV scenario that explicitly forbids entry."""

    symbol: str
    reason: str
    fwd_60m_bps: float
    own_state: SymbolState
    leader_state: SymbolState | None


def compute_state(
    cycles: list[CycleSnapshot],
) -> SymbolState | None:
    """Given an ordered list of cycle snapshots for one symbol (oldest→newest),
    return the SymbolState computed from the last two."""
    if len(cycles) < 2:
        return None
    cur = cycles[-1]
    prv = cycles[-2]
    if (
        cur.open_interest <= 0.0
        or prv.open_interest <= 0.0
        or cur.last_price <= 0.0
        or prv.last_price <= 0.0
    ):
        return None
    oi_d = (cur.open_interest / prv.open_interest - 1.0) * 100.0
    p_d = (cur.last_price / prv.last_price - 1.0) * 10000.0
    return SymbolState(
        symbol=cur.symbol,
        cycle_ts_ms=cur.ts_ms,
        own_quadrant=_quadrant(p_d > 0, oi_d > 0),
        own_dir=_direction(p_d > 0),
        oi_delta_pct=oi_d,
        price_delta_bps=p_d,
    )


def _match_specificity(scenario: dict[str, Any]) -> int:
    """Higher = more specific. leader_quadrant present > leader_dir > none."""
    score = 0
    if scenario.get("leader_quadrant"):
        score += 2
    if scenario.get("leader_dir"):
        score += 1
    return score


def _matches_scenario(
    scenario: dict[str, Any],
    own: SymbolState,
    leader: SymbolState | None,
) -> bool:
    if scenario["symbol"] != own.symbol:
        return False
    if scenario["own_quadrant"] != own.own_quadrant:
        return False
    leader_quad = scenario.get("leader_quadrant")
    leader_dir = scenario.get("leader_dir")
    if leader_quad:
        if leader is None or leader.own_quadrant != leader_quad:
            return False
    if leader_dir:
        if leader is None or leader.own_dir != leader_dir:
            return False
    return True


def score_symbol(
    table: dict[str, Any],
    own: SymbolState,
    leader: SymbolState | None,
    *,
    now_ms: int,
) -> Score | None:
    """Find best matching scenario for a symbol, return Score or None.

    Returns None if no scenario matches AND no blocker matches.
    Returns Score with blocker_match set if a blocker matches (caller treats as veto).
    """
    stale_minutes = int(table.get("stale_minutes", 15))
    age_min = (now_ms - own.cycle_ts_ms) / 60_000.0
    stale = age_min > stale_minutes

    # Blocker check first
    blocker_hit = None
    for b in table.get("blockers", []):
        if b["symbol"] != own.symbol:
            continue
        if b.get("own_quadrant") != own.own_quadrant:
            continue
        if b.get("leader_quadrant") and (leader is None or leader.own_quadrant != b["leader_quadrant"]):
            continue
        if b.get("leader_dir") and (leader is None or leader.own_dir != b["leader_dir"]):
            continue
        blocker_hit = b
        break

    # Scenario match (most-specific wins)
    candidates = [
        s for s in table.get("scenarios", []) if _matches_scenario(s, own, leader)
    ]
    if not candidates and not blocker_hit:
        return None

    if candidates:
        best = max(candidates, key=_match_specificity)
        return Score(
            symbol=own.symbol,
            side=best["side"],
            ev_bps=float(best["ev_bps"]),
            rank=int(best["rank"]),
            n=int(best["n"]),
            fwd_60m_bps=float(best["fwd_60m_bps"]),
            winrate=float(best["winrate"]),
            matched_scenario=best,
            own_state=own,
            leader_state=leader,
            stale=stale,
            blocker_match=blocker_hit,
        )

    # Only blocker, no positive scenario
    return Score(
        symbol=own.symbol,
        side="block",
        ev_bps=float(blocker_hit["fwd_60m_bps"]),
        rank=999,
        n=int(blocker_hit["n"]),
        fwd_60m_bps=float(blocker_hit["fwd_60m_bps"]),
        winrate=float(blocker_hit["winrate"]),
        matched_scenario={},
        own_state=own,
        leader_state=leader,
        stale=stale,
        blocker_match=blocker_hit,
    )


def rank_candidates(
    table: dict[str, Any],
    states_by_symbol: dict[str, SymbolState],
    *,
    now_ms: int,
) -> list[Score]:
    """Compute ranked Score list across the universe.

    Filters out:
    - stale snapshots
    - blocker matches (still returned for visibility but with score below threshold)
    - candidates below EV threshold

    Returns top candidates in EV-descending order; the caller picks #1.
    """
    threshold = float(table.get("ev_threshold_bps", 5.0))
    leader_map: dict[str, str] = table.get("leader_per_symbol", {})
    out: list[Score] = []
    for sym, own in states_by_symbol.items():
        leader_sym = leader_map.get(sym)
        leader = states_by_symbol.get(leader_sym) if leader_sym else None
        score = score_symbol(table, own, leader, now_ms=now_ms)
        if score is None:
            continue
        if score.stale:
            continue
        if score.blocker_match is not None:
            # blocked: do not promote regardless of scenario
            continue
        if score.ev_bps < threshold:
            continue
        out.append(score)
    out.sort(key=lambda s: s.ev_bps, reverse=True)
    return out


def load_ev_table(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_latest_states(
    shadow_root: Path,
    universe: list[str],
) -> dict[str, SymbolState]:
    """Read the two most-recent cycle_*/metrics.json under shadow_root and
    compute SymbolState for each symbol in `universe`.

    Returns {} on any IO error or insufficient data — caller must treat empty
    as "no cross-coin context available, do not block on this layer".
    """
    try:
        cycle_dirs = sorted(shadow_root.glob("cycle_*"))[-2:]
    except (OSError, ValueError):
        return {}
    if len(cycle_dirs) < 2:
        return {}

    snaps_by_sym: dict[str, list[CycleSnapshot]] = {}
    for cdir in cycle_dirs:
        mfile = cdir / "metrics.json"
        if not mfile.exists():
            return {}
        try:
            payload = json.loads(mfile.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        for row in payload.get("rows", []):
            sym = (row.get("symbol") or "").upper()
            if sym not in universe:
                continue
            ts_str = row.get("timestamp")
            if not ts_str:
                continue
            try:
                # Avoid heavy datetime imports here — caller's parser is fine
                from datetime import datetime
                ts_norm = ts_str.replace("Z", "+00:00") if ts_str.endswith("Z") else ts_str
                ts_ms = int(datetime.fromisoformat(ts_norm).timestamp() * 1000)
            except (ValueError, AttributeError):
                continue
            try:
                oi = float(row.get("open_interest") or 0.0)
                px = float(row.get("last_price") or 0.0)
            except (TypeError, ValueError):
                continue
            snaps_by_sym.setdefault(sym, []).append(
                CycleSnapshot(symbol=sym, ts_ms=ts_ms, open_interest=oi, last_price=px)
            )

    out: dict[str, SymbolState] = {}
    for sym, snaps in snaps_by_sym.items():
        snaps_sorted = sorted(snaps, key=lambda s: s.ts_ms)
        st = compute_state(snaps_sorted)
        if st is not None:
            out[sym] = st
    return out


def is_blocked(
    table: dict[str, Any],
    states_by_symbol: dict[str, SymbolState],
    target_symbol: str,
    *,
    now_ms: int,
) -> tuple[bool, dict[str, Any] | None]:
    """Standalone check: is `target_symbol` currently blocked by a blocker rule?

    Returns (is_blocked, matching_blocker_dict_or_None).
    """
    own = states_by_symbol.get(target_symbol)
    if own is None:
        return (False, None)
    leader_sym = table.get("leader_per_symbol", {}).get(target_symbol)
    leader = states_by_symbol.get(leader_sym) if leader_sym else None
    score = score_symbol(table, own, leader, now_ms=now_ms)
    if score is None:
        return (False, None)
    if score.blocker_match is not None:
        return (True, score.blocker_match)
    return (False, None)
