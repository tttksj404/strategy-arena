# Wave-34 reinforcement-learning arm: tabular Q-learning over a state-dependent policy.
#
# ---------------------------------------------------------------------------------------
# Why this is a separate arm and not a seventh tournament entry
# ---------------------------------------------------------------------------------------
# The five metaheuristics and the random control all search the SAME object: a fixed parameter
# vector describing a static rule. Every wave in this repository from wave1 to wave33 searched
# that object. Reinforcement learning is the only method on the user's list that changes the
# object itself -- it learns a MAPPING from market state to action, so the same symbol can be
# traded long, short, or not at all depending on conditions, which no genome in genome30 can
# express. Comparing it on equal evaluation budget would be meaningless because an "evaluation"
# is not the same unit. So it is reported alongside, not inside, the tournament.
#
# ---------------------------------------------------------------------------------------
# Why tabular Q-learning rather than DQN/PPO/SAC
# ---------------------------------------------------------------------------------------
# SPEC.md freezes tabular Q-learning for a reason that is about evidence, not convenience: with
# 81 states and 3 actions the learned policy is a table a human can read and audit, so a positive
# result can be inspected for whether it encodes something economic or something spurious. A DQN
# on the same 60,000 hourly bars would add a function approximator, a replay buffer, a target
# network and their hyperparameters -- more capacity to overfit and less ability to see it. If the
# tabular version cannot find an edge, a deeper one finding one would be the more suspicious
# outcome, not the more convincing one.
#
# ---------------------------------------------------------------------------------------
# Leakage discipline
# ---------------------------------------------------------------------------------------
# All state features are built from bars strictly BEFORE the bar being acted on, using the same
# shift discipline as dataio30. Training touches IS bars only. The Q-table is frozen before OOS
# is replayed, and the replay is greedy (no learning, no exploration), so OOS cannot influence
# the policy.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave30_qd.dataio30 import MarketCache, OOS_SPLIT

MOMENTUM_BUCKETS: Final = 3
VOLATILITY_BUCKETS: Final = 3
FUNDING_BUCKETS: Final = 3
POSITION_STATES: Final = 3  # flat, long, short
N_STATES: Final = MOMENTUM_BUCKETS * VOLATILITY_BUCKETS * FUNDING_BUCKETS * POSITION_STATES
N_ACTIONS: Final = 3  # 0 = flat, 1 = long, 2 = short
ACTION_POSITION: Final = np.array([0.0, 1.0, -1.0])

LEVERAGE_GRID: Final = (1.0, 3.0, 5.0, 10.0, 20.0)
MOMENTUM_LOOKBACK: Final = 24
VOLATILITY_LOOKBACK: Final = 168
LEARNING_RATE: Final = 0.10
DISCOUNT: Final = 0.95
EPOCHS: Final = 8
EPSILON_START: Final = 0.30
EPSILON_END: Final = 0.02
MAINT_MARGIN: Final = 0.005


@dataclass(frozen=True)
class SymbolFeatures:
    symbol: str
    state_base: np.ndarray  # state index ignoring position, per bar (-1 where unusable)
    bar_return: np.ndarray  # close-to-close simple return realised AT this bar
    funding_at_bar: np.ndarray
    cost_rate: float
    tradable: np.ndarray


def build_features(cache: MarketCache) -> dict[str, SymbolFeatures]:
    features: dict[str, SymbolFeatures] = {}
    for symbol, arrays in cache.arrays.items():
        close = pd.Series(np.where(arrays.tradable, arrays.close, np.nan))
        simple_return = (close / close.shift(1) - 1.0).to_numpy(dtype=float)

        # Momentum: z-score of the trailing 24-bar return, measured through the PREVIOUS bar.
        momentum = (close / close.shift(MOMENTUM_LOOKBACK) - 1.0)
        z = ((momentum - momentum.rolling(VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean())
             / momentum.rolling(VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).std(ddof=1))
        z = z.shift(1).to_numpy(dtype=float)

        # Volatility: 24-bar realised vol against its own trailing median, through previous bar.
        log_return = np.log(close / close.shift(1))
        realised = log_return.rolling(MOMENTUM_LOOKBACK, min_periods=MOMENTUM_LOOKBACK).std(ddof=1)
        reference = realised.rolling(VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).median()
        ratio = (realised / reference).shift(1).to_numpy(dtype=float)

        rates = arrays.funding_at_bar
        known = np.where(rates != 0.0, rates, np.nan)
        index = np.arange(len(known))
        valid = ~np.isnan(known)
        positions = np.maximum.accumulate(np.where(valid, index, 0))
        last_funding = np.where(valid.any(), known[positions], 0.0)
        last_funding = np.nan_to_num(last_funding, nan=0.0)

        momentum_bucket = np.select([z < -0.5, z > 0.5], [0, 2], default=1)
        volatility_bucket = np.select([ratio < 0.8, ratio > 1.25], [0, 2], default=1)
        funding_bucket = np.select([last_funding < 0.0, last_funding > 0.0], [0, 2], default=1)

        usable = np.isfinite(z) & np.isfinite(ratio) & arrays.tradable & np.isfinite(simple_return)
        base = (momentum_bucket * VOLATILITY_BUCKETS + volatility_bucket) * FUNDING_BUCKETS + funding_bucket
        base = np.where(usable, base, -1)
        features[symbol] = SymbolFeatures(
            symbol=symbol,
            state_base=base.astype(int),
            bar_return=np.nan_to_num(simple_return, nan=0.0),
            funding_at_bar=rates,
            cost_rate=arrays.cost_rate,
            tradable=arrays.tradable,
        )
    return features


def _state_index(state_base: int, position: float) -> int:
    position_code = 0 if position == 0.0 else (1 if position > 0.0 else 2)
    return state_base * POSITION_STATES + position_code


def _step_reward(
    feature: SymbolFeatures, bar: int, previous_position: float, new_position: float, leverage: float
) -> float:
    """Log growth contributed by this bar.

    Cost is charged on the TURNOVER of notional: going flat->long crosses one leg, long->short
    crosses two. Funding is charged on the position held through the bar. Both are multiplied by
    leverage, the same convention engine30 uses, so the RL arm and the tournament arm pay
    identical frictions.
    """
    turnover = abs(new_position - previous_position)
    cost = turnover * feature.cost_rate * leverage
    funding = new_position * feature.funding_at_bar[bar] * leverage
    gross = new_position * feature.bar_return[bar] * leverage
    growth = 1.0 + gross - cost - funding
    return float(np.log(max(growth, 1e-6)))


def train_q_table(
    features: dict[str, SymbolFeatures], horizon: int, leverage: float, seed: int
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    symbols = list(features)
    updates = 0
    for epoch in range(EPOCHS):
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * epoch / max(1, EPOCHS - 1)
        for symbol in symbols:
            feature = features[symbol]
            position = 0.0
            for bar in range(1, horizon - 1):
                base = feature.state_base[bar]
                if base < 0:
                    position = 0.0
                    continue
                state = _state_index(int(base), position)
                if rng.random() < epsilon:
                    action = int(rng.integers(N_ACTIONS))
                else:
                    action = int(np.argmax(q_table[state]))
                new_position = float(ACTION_POSITION[action])
                reward = _step_reward(feature, bar + 1, position, new_position, leverage)

                next_base = feature.state_base[bar + 1]
                if next_base < 0:
                    target = reward
                else:
                    next_state = _state_index(int(next_base), new_position)
                    target = reward + DISCOUNT * float(np.max(q_table[next_state]))
                q_table[state, action] += LEARNING_RATE * (target - q_table[state, action])
                updates += 1
                position = new_position
    return q_table, {"updates": updates, "epochs": EPOCHS, "leverage": leverage, "seed": seed}


def replay_greedy(
    features: dict[str, SymbolFeatures], q_table: np.ndarray, start: int, end: int, leverage: float
) -> dict:
    """Frozen-policy replay. No learning, no exploration. Each symbol runs its own book with an
    equal share of capital; the liquidation band is checked per bar against the position's own
    adverse move, mirroring engine30's isolated-margin treatment."""
    symbols = list(features)
    share = 1.0 / len(symbols)
    liquidation_band = max(0.0, 1.0 / leverage - MAINT_MARGIN)

    total_log = 0.0
    entries = 0
    liquidations = 0
    bars_in_market = 0
    per_symbol: dict[str, dict] = {}
    for symbol in symbols:
        feature = features[symbol]
        position = 0.0
        symbol_log = 0.0
        symbol_entries = 0
        for bar in range(max(1, start), min(end, len(feature.state_base) - 1)):
            base = feature.state_base[bar]
            if base < 0:
                position = 0.0
                continue
            state = _state_index(int(base), position)
            action = int(np.argmax(q_table[state]))
            new_position = float(ACTION_POSITION[action])
            if new_position != position and new_position != 0.0:
                symbol_entries += 1
            adverse = -new_position * feature.bar_return[bar + 1]
            if new_position != 0.0 and adverse >= liquidation_band:
                symbol_log += np.log(1e-6)
                liquidations += 1
                position = 0.0
                continue
            symbol_log += _step_reward(feature, bar + 1, position, new_position, leverage)
            if new_position != 0.0:
                bars_in_market += 1
            position = new_position
        per_symbol[symbol] = {"log_growth": symbol_log, "entries": symbol_entries}
        total_log += share * symbol_log
        entries += symbol_entries
    return {
        "log_growth": float(total_log),
        "growth_multiple": float(np.exp(total_log)),
        "entries": entries,
        "liquidations": liquidations,
        "bars_in_market": bars_in_market,
    }


def run_rl_arm(cache: MarketCache, seeds: tuple[int, ...]) -> dict:
    features = build_features(cache)
    horizon = int(cache.is_mask.sum())
    oos_start = horizon
    n_bars = cache.n_bars
    is_days = float(int(cache.day_of_bar[horizon - 1]) + 1)
    oos_days = float(len(cache.daily_index) - is_days)

    rows: list[dict] = []
    for leverage in LEVERAGE_GRID:
        for seed in seeds:
            q_table, meta = train_q_table(features, horizon, leverage, seed)
            in_sample = replay_greedy(features, q_table, 1, horizon, leverage)
            out_sample = replay_greedy(features, q_table, oos_start, n_bars, leverage)
            rows.append(
                {
                    "leverage": leverage,
                    "seed": seed,
                    "updates": meta["updates"],
                    "is_growth_multiple": in_sample["growth_multiple"],
                    "is_entries": in_sample["entries"],
                    "is_entries_per_day": in_sample["entries"] / is_days,
                    "is_liquidations": in_sample["liquidations"],
                    "oos_growth_multiple": out_sample["growth_multiple"],
                    "oos_entries": out_sample["entries"],
                    "oos_entries_per_day": out_sample["entries"] / max(oos_days, 1.0),
                    "oos_liquidations": out_sample["liquidations"],
                    "distinct_actions_used": int(len(np.unique(np.argmax(q_table, axis=1)))),
                }
            )
    summary: dict[str, dict] = {}
    for leverage in LEVERAGE_GRID:
        subset = [r for r in rows if r["leverage"] == leverage]
        summary[f"{leverage:g}x"] = {
            "is_growth_median": float(np.median([r["is_growth_multiple"] for r in subset])),
            "is_growth_best": float(max(r["is_growth_multiple"] for r in subset)),
            "oos_growth_median": float(np.median([r["oos_growth_multiple"] for r in subset])),
            "oos_growth_best": float(max(r["oos_growth_multiple"] for r in subset)),
            "is_entries_per_day_median": float(np.median([r["is_entries_per_day"] for r in subset])),
            "oos_entries_per_day_median": float(np.median([r["oos_entries_per_day"] for r in subset])),
            "liquidations_total": int(sum(r["is_liquidations"] + r["oos_liquidations"] for r in subset)),
        }
    return {"rows": rows, "summary": summary, "n_states": N_STATES, "n_actions": N_ACTIONS,
            "is_days": is_days, "oos_days": oos_days}
