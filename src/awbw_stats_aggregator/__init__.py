"""Primary entrypoints for AWBW stats aggregation utilities."""

from __future__ import annotations

from awbw_stats_aggregator.completed_games import get_completed_games
from awbw_stats_aggregator.stats import (
    player_stats,
    plot_player_win_rates,
    plot_win_rate_table,
    win_rate_matrix,
)

__all__ = [
    "get_completed_games",
    "player_stats",
    "plot_player_win_rates",
    "plot_win_rate_table",
    "win_rate_matrix",
]
