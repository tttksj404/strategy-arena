from __future__ import annotations

from typing import Final

from .intraday_replay import ReplayEntryFilter, ReplayRule


PRECOMMITTED_REPLAY_RULES: Final = (
    ReplayRule("breakout-volume-1.0", 1.0),
    ReplayRule("breakout-volume-1.25", 1.25),
    ReplayRule("breakout-volume-1.5", 1.5),
    ReplayRule("breakout-volume-2.0", 2.0),
    ReplayRule("breakout-technical-score-v1-volume-1.0", 1.0, ReplayEntryFilter.TECHNICAL_SCORE),
    ReplayRule("breakout-bollinger-squeeze-v1-volume-1.0", 1.0, ReplayEntryFilter.BOLLINGER_SQUEEZE),
)
