"""Export Elo data and summaries to JSON for the static site."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from awbw_stats_aggregator import get_completed_games
from awbw_stats_aggregator.stats import (
    BASE_TEAM_ELO,
    team_combo_elo,
    team_elo,
)


def _datetime_to_str(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _build_player_series(df: pd.DataFrame, base_rating: float) -> list[dict]:
    series_list: list[dict] = []
    for player, sub in df.groupby("player"):
        ordered = sub.sort_values(["ended_on", "day", "game_id"])
        points = [
            {
                "game_index": 0,
                "rating": base_rating,
                "game_id": None,
                "day": None,
                "ended_on": None,
            }
        ]
        for idx, row in enumerate(ordered.itertuples(index=False), start=1):
            points.append(
                {
                    "game_index": idx,
                    "rating": float(row.rating_after),
                    "game_id": int(row.game_id),
                    "day": int(row.day),
                    "ended_on": _datetime_to_str(row.ended_on),
                }
            )
        series_list.append({"player": player, "points": points})
    return series_list


def _build_team_series(df: pd.DataFrame, base_rating: float) -> list[dict]:
    series_list: list[dict] = []
    for team, sub in df.groupby("team"):
        ordered = sub.sort_values(["ended_on", "day", "game_id"])
        points = [
            {
                "game_index": 0,
                "rating": base_rating,
                "game_id": None,
                "day": None,
                "ended_on": None,
            }
        ]
        for idx, row in enumerate(ordered.itertuples(index=False), start=1):
            points.append(
                {
                    "game_index": idx,
                    "rating": float(row.rating_after),
                    "game_id": int(row.game_id),
                    "day": int(row.day),
                    "ended_on": _datetime_to_str(row.ended_on),
                }
            )
        series_list.append(
            {
                "team": team,
                "members": row.members if hasattr(row, "members") else "",
                "points": points,
            }
        )
    return series_list


def _latest_by_group(df: pd.DataFrame, group_col: str, sort_value: str) -> pd.DataFrame:
    return (
        df.sort_values(["ended_on", "day", "game_id"])
        .groupby(group_col, as_index=False)
        .tail(1)
        .sort_values(sort_value, ascending=False)
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Elo data for the static site.")
    parser.add_argument("--username", required=True, help="AWBW username to fetch games from")
    parser.add_argument(
        "--players",
        nargs="+",
        required=True,
        help="List of player usernames to include",
    )
    parser.add_argument("--team-size", type=int, default=2, help="Team size (default: 2)")
    parser.add_argument("--min-days", type=int, default=2, help="Minimum day filter (default: 2)")
    parser.add_argument("--max-pages", type=int, default=200, help="Max pages to scrape (default: 200)")
    parser.add_argument("--k-factor", type=float, default=32.0, help="Elo K factor (default: 32)")
    parser.add_argument("--scale", type=float, default=400.0, help="Elo scale (default: 400)")
    parser.add_argument(
        "--round-mode",
        choices=["none", "nearest"],
        default="nearest",
        help="Rating rounding mode (default: nearest, AoE2-style)",
    )
    parser.add_argument(
        "--output",
        default="docs/data/elo.json",
        help="Path to write JSON output (default: docs/data/elo.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    games = get_completed_games(args.username, max_pages=args.max_pages)

    elo_df = team_elo(
        games,
        args.players,
        team_size=args.team_size,
        min_days=args.min_days,
        k_factor=args.k_factor,
        scale=args.scale,
        round_mode=args.round_mode,  # AoE2-style rounding by default
    )

    team_elo_df = team_combo_elo(
        games,
        args.players,
        team_size=args.team_size,
        min_days=args.min_days,
        k_factor=args.k_factor,
        scale=args.scale,
        round_mode=args.round_mode,
    )

    latest_players = _latest_by_group(elo_df, "player", "rating_after")[
        ["player", "rating_after", "game_id", "day", "ended_on"]
    ].assign(ended_on=lambda df: df["ended_on"].map(_datetime_to_str))
    latest_teams = _latest_by_group(team_elo_df, "team", "rating_after")[
        ["team", "members", "rating_after", "game_id", "day", "ended_on"]
    ].assign(ended_on=lambda df: df["ended_on"].map(_datetime_to_str))

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "username": args.username,
            "players": args.players,
            "team_size": args.team_size,
            "min_days": args.min_days,
            "max_pages": args.max_pages,
            "k_factor": args.k_factor,
            "scale": args.scale,
            "round_mode": args.round_mode,
            "base_rating": BASE_TEAM_ELO,
        },
        "latest_player_elo": latest_players.to_dict(orient="records"),
        "latest_team_elo": latest_teams.to_dict(orient="records"),
        "player_series": _build_player_series(elo_df, BASE_TEAM_ELO),
        "team_series": _build_team_series(team_elo_df, BASE_TEAM_ELO),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()


