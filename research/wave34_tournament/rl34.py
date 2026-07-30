# Wave-34 arm 7: tabular Q-learning on hourly bars.
#
# This arm is reported SEPARATELY from the six optimizers and is not ranked against them.
# The six all search the same static rule space (the wave30 genome); Q-learning does not --
# its policy can condition on the current bar's state and change its mind hour by hour, so it
# is a different hypothesis class. Comparing it on the same leaderboard would answer neither
# "does the search method matter" (it is not searching the same space) nor "is RL better"
# (it has a different number of effective parameters). It gets its own section.
#
# ---------------------------------------------------------------------------------------
# What is deliberately NOT hidden
# ---------------------------------------------------------------------------------------
# The learner trades ONE symbol with a fixed notional and a fixed leverage, and it pays the
# same measured costs the engine pays: `arrays.cost_rate` per side on every position CHANGE,
# and the realised funding of the bar while holding. Costs are charged on transitions only,
# which is what makes churning expensive -- an agent rewarded on unlevered price moves with
# free switching would learn a beautiful strategy that does not exist.
#
# Reward is the log change in account equity over the bar, so maximising cumulative reward is
# exactly maximising the final account -- the same objective as the six.
#
# The Q-table is learned on IS bars only. OOS is never touched here; the runner decides
# whether this arm is even eligible for the single OOS unsealing.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import MarketCache

ACTIONS: Final = (0.0, 1.0, -1.0)  # flat, long, short
N_ACTIONS: Final = len(ACTIONS)
MOMENTUM_BINS: Final = 5
VOL_BINS: Final = 4
FUNDING_BINS: Final = 3
POSITION_STATES: Final = 3
N_STATES: Final = MOMENTUM_BINS * VOL_BINS * FUNDING_BINS * POSITION_STATES

START_CAPITAL: Final = 100.0
RL_LEVERAGE: Final = 5.0  # frozen; a learner that can also pick leverage is a different study
RL_SYMBOL: Final = "BTCUSDT"
RL_LOOKBACK: Final = 24
SURVIVAL_FLOOR_USDT: Final = 1.0  # below this the account cannot fund any position


@dataclass(frozen=True)
class RLResult:
    episodes: int
    final_usdt: float
    n_position_changes: int
    trades_per_active_day: float
    active_days: float
    survived: bool
    mdd: float
    action_mix: tuple[float, float, float]
    greedy_equity_curve_final: float
    mean_usdt_per_change: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "final_usdt": self.final_usdt,
            "n_position_changes": self.n_position_changes,
            "trades_per_active_day": self.trades_per_active_day,
            "active_days": self.active_days,
            "survived": self.survived,
            "mdd": self.mdd,
            "action_mix_flat_long_short": list(self.action_mix),
            "mean_usdt_per_change": self.mean_usdt_per_change,
        }


def build_features(cache: MarketCache, symbol: str = RL_SYMBOL, is_only: bool = True) -> dict[str, np.ndarray]:
    """Discretise the market into the state features. All quantile edges come from IS bars
    only, so the OOS replay (if ever run) uses IS-derived bins rather than bins fitted on the
    data it is being tested on."""
    arrays = cache.arrays[symbol]
    horizon = int(cache.is_mask.sum()) if is_only else cache.n_bars

    momentum = arrays.ret[RL_LOOKBACK]
    volatility = arrays.vol[RL_LOOKBACK]
    funding = arrays.funding_at_bar

    fit = slice(0, int(cache.is_mask.sum()))
    momentum_edges = np.nanquantile(momentum[fit], np.linspace(0, 1, MOMENTUM_BINS + 1)[1:-1])
    vol_edges = np.nanquantile(volatility[fit], np.linspace(0, 1, VOL_BINS + 1)[1:-1])
    funding_edges = np.array([-1e-5, 1e-5])

    momentum_bin = np.clip(np.searchsorted(momentum_edges, np.nan_to_num(momentum)), 0, MOMENTUM_BINS - 1)
    vol_bin = np.clip(np.searchsorted(vol_edges, np.nan_to_num(volatility)), 0, VOL_BINS - 1)
    funding_bin = np.clip(np.searchsorted(funding_edges, np.nan_to_num(funding)), 0, FUNDING_BINS - 1)

    close = arrays.close
    bar_return = np.zeros(cache.n_bars)
    bar_return[1:] = np.nan_to_num(close[1:] / close[:-1] - 1.0, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "horizon": horizon,
        "momentum_bin": momentum_bin.astype(np.int32),
        "vol_bin": vol_bin.astype(np.int32),
        "funding_bin": funding_bin.astype(np.int32),
        "bar_return": bar_return,
        "funding": np.nan_to_num(funding),
        "tradable": arrays.tradable,
        "cost_rate": float(arrays.cost_rate),
        "first_bar": int(np.argmax(arrays.tradable)) + RL_LOOKBACK + 1,
    }


def _state_index(momentum: int, vol: int, funding: int, position: int) -> int:
    return ((momentum * VOL_BINS + vol) * FUNDING_BINS + funding) * POSITION_STATES + position


def _bar_log_reward(
    features: dict[str, np.ndarray],
    bar: int,
    previous_position: float,
    position: float,
) -> float:
    """Log equity change over bar->bar+1 for a position held at RL_LEVERAGE, minus the cost of
    having CHANGED position at this bar and minus the funding charged while holding."""
    gross = position * features["bar_return"][bar + 1] * RL_LEVERAGE
    turnover = abs(position - previous_position)
    cost = turnover * features["cost_rate"] * RL_LEVERAGE
    funding = position * features["funding"][bar + 1] * RL_LEVERAGE
    factor = 1.0 + gross - cost - funding
    return float(np.log(max(factor, 1e-6)))


def train_q_learning(
    cache: MarketCache,
    seed: int,
    episodes: int,
    alpha: float = 0.15,
    gamma: float = 0.999,
    epsilon0: float = 0.5,
    epsilon_end: float = 0.02,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Epsilon-greedy tabular Q-learning, one episode = one full IS pass.

    gamma is near 1 because the reward is log equity and the objective is the FULL-span
    product; discounting hard would optimise a myopic proxy nobody asked for.
    """
    rng = np.random.default_rng(seed)
    features = build_features(cache, is_only=True)
    horizon = int(features["horizon"]) - 1
    start = int(features["first_bar"])

    q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    momentum_bin = features["momentum_bin"]
    vol_bin = features["vol_bin"]
    funding_bin = features["funding_bin"]

    for episode in range(episodes):
        fraction = episode / max(1, episodes - 1)
        epsilon = epsilon0 * (epsilon_end / epsilon0) ** fraction
        position = 0.0
        position_state = 0
        state = _state_index(momentum_bin[start], vol_bin[start], funding_bin[start], position_state)

        for bar in range(start, horizon):
            if rng.random() < epsilon:
                action = int(rng.integers(N_ACTIONS))
            else:
                action = int(np.argmax(q[state]))
            new_position = ACTIONS[action]
            reward = _bar_log_reward(features, bar, position, new_position)

            new_position_state = 0 if new_position == 0.0 else (1 if new_position > 0 else 2)
            next_state = _state_index(
                momentum_bin[bar + 1], vol_bin[bar + 1], funding_bin[bar + 1], new_position_state
            )
            q[state, action] += alpha * (reward + gamma * float(np.max(q[next_state])) - q[state, action])

            state = next_state
            position = new_position
            position_state = new_position_state

    return q, features


def replay_greedy(
    cache: MarketCache,
    q: np.ndarray,
    features: dict[str, np.ndarray],
    is_only: bool = True,
) -> RLResult:
    """Run the learned greedy policy once and MEASURE the account, in dollars, with the same
    cost and funding charges used in training. Equity floors at zero and trading stops when it
    is gone -- so a policy that blows the account cannot report a recovery."""
    horizon = (int(cache.is_mask.sum()) if is_only else cache.n_bars) - 1
    start = int(features["first_bar"])
    momentum_bin = features["momentum_bin"]
    vol_bin = features["vol_bin"]
    funding_bin = features["funding_bin"]

    equity = START_CAPITAL
    position = 0.0
    position_state = 0
    changes = 0
    first_change_bar = None
    last_change_bar = None
    action_counts = np.zeros(N_ACTIONS)
    curve = []

    for bar in range(start, horizon):
        if equity <= 1e-6:
            break
        state = _state_index(momentum_bin[bar], vol_bin[bar], funding_bin[bar], position_state)
        action = int(np.argmax(q[state]))
        action_counts[action] += 1
        new_position = ACTIONS[action]
        if new_position != position:
            changes += 1
            if first_change_bar is None:
                first_change_bar = bar
            last_change_bar = bar
        reward = _bar_log_reward(features, bar, position, new_position)
        equity = max(0.0, equity * float(np.exp(reward)))
        curve.append(equity)
        position = new_position
        position_state = 0 if new_position == 0.0 else (1 if new_position > 0 else 2)

    equity_curve = np.array(curve) if curve else np.array([START_CAPITAL])
    peak = np.maximum.accumulate(equity_curve)
    mdd = float(abs(np.min((equity_curve - peak) / np.maximum(peak, 1e-12))))

    if first_change_bar is None:
        active_days, per_day, survived = 0.0, 0.0, False
    else:
        first_day = int(cache.day_of_bar[first_change_bar])
        last_day = int(cache.day_of_bar[last_change_bar])
        active_days = float(max(1, last_day - first_day + 1))
        per_day = changes / active_days
        # "Survived" must mean the account can still trade, not merely that the float is
        # above zero. A $0.005 account is dead; calling it survived would let a wiped-out
        # policy pass the survival constraint on a rounding artefact.
        survived = bool(last_day >= int(cache.day_of_bar[horizon - 1]) - 30 and equity >= SURVIVAL_FLOOR_USDT)

    return RLResult(
        episodes=0,
        final_usdt=float(equity_curve[-1]),
        n_position_changes=changes,
        trades_per_active_day=float(per_day),
        active_days=active_days,
        survived=survived,
        mdd=mdd,
        action_mix=tuple(float(v) for v in action_counts / max(1.0, action_counts.sum())),
        greedy_equity_curve_final=float(equity_curve[-1]),
        mean_usdt_per_change=float((equity_curve[-1] - START_CAPITAL) / changes) if changes else 0.0,
    )
