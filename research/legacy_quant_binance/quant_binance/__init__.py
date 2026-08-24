"""Core package for the Binance regime-switching quant system."""

from .bootstrap import WorkspaceLayout, initialize_workspace
from .live import EventDispatcher, LivePaperRuntime
from .models import DecisionIntent, FeatureVector, MarketSnapshot
from .paths import RunPaths, prepare_run_paths
from .settings import Settings

__all__ = [
    "DecisionIntent",
    "EventDispatcher",
    "FeatureVector",
    "LivePaperRuntime",
    "MarketSnapshot",
    "RunPaths",
    "Settings",
    "WorkspaceLayout",
    "initialize_workspace",
    "prepare_run_paths",
]
