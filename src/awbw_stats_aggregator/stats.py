"""Utilities for generating player statistics from completed AWBW games."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pydantic import BaseModel, ConfigDict

from awbw_stats_aggregator.completed_games import CompletedGame, CompletedGamePlayer


class _PlayerMatchResult(BaseModel):
    """Internal representation of a single player's outcome in a game."""

    model_config = ConfigDict(frozen=True)

    player: str
    opponent: Optional[str]
    result: str  # "win", "loss", or "draw"


class _PlayerIdentity(BaseModel):
    """Normalized representation of a player username."""

    model_config = ConfigDict(frozen=True)

    original: str
    normalized: str


class _HeadToHeadCount(BaseModel):
    """Cumulative match outcomes between two players."""

    player_a: str
    player_b: str
    player_a_wins: int = 0
    player_b_wins: int = 0
    draws: int = 0


class _TeamIdentity(BaseModel):
    """Represents a fixed-size team of players."""

    model_config = ConfigDict(frozen=True)

    members: tuple[_PlayerIdentity, ...]
    normalized_members: tuple[str, ...]
    label: str

    @property
    def member_set(self) -> set[str]:
        return set(self.normalized_members)


class _TeamHeadToHeadCount(BaseModel):
    """Aggregated results between two teams."""

    team_a: tuple[str, ...]
    team_b: tuple[str, ...]
    team_a_wins: int = 0
    team_b_wins: int = 0
    draws: int = 0


def player_stats(
    games: Iterable[CompletedGame],
    player_username: str,
    *,
    opponents: Iterable[str] | str | None = None,
    min_days: int | None = None,
) -> pd.DataFrame:
    """Return aggregate match results for ``player_username`` as a DataFrame.

    Parameters
    ----------
    games:
        An iterable of :class:`~awbw_stats_aggregator.completed_games.CompletedGame`
        instances to analyze.
    player_username:
        The player whose record should be summarized.
    opponents:
        Optional iterable (or single string) of opponent usernames to restrict the
        statistics to. When provided, the returned dataframe contains one row per
        opponent. When omitted, a single row with the player's overall record is
        returned.
    min_days:
        Exclude games that finished before this day count. When ``None`` (default),
        all games are considered.
    """

    if not player_username:
        raise ValueError("player_username must be provided")

    normalized_opponents = _normalize_opponents(opponents)
    games_list = _filter_games_by_day(games, min_days)

    rows: List[dict[str, object]] = []
    if normalized_opponents:
        for opponent in normalized_opponents:
            rows.append(
                _summarize_results(
                    games_list,
                    player_username,
                    opponent_username=opponent,
                )
            )
    else:
        rows.append(
            _summarize_results(
                games_list,
                player_username,
                opponent_username=None,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "player",
            "opponent",
            "matches",
            "wins",
            "losses",
            "draws",
            "win_rate",
        ],
    )


def win_rate_matrix(
    games: Iterable[CompletedGame],
    players: Iterable[str],
    *,
    min_days: int | None = None,
) -> pd.DataFrame:
    """Return a square DataFrame of win rates between the provided players."""

    identities = _normalize_players(players)
    if not identities:
        raise ValueError("At least one player must be provided")

    games_list = _filter_games_by_day(games, min_days)
    identity_map: Dict[str, _PlayerIdentity] = {
        identity.normalized: identity for identity in identities
    }
    counts = _aggregate_head_to_head(games_list, identity_map)

    player_names = [identity.original for identity in identities]
    matrix = pd.DataFrame(
        data=np.nan,
        index=player_names,
        columns=player_names,
        dtype=float,
    )

    for (player_a_norm, player_b_norm), record in counts.items():
        identity_a = identity_map[player_a_norm]
        identity_b = identity_map[player_b_norm]

        matches = record.player_a_wins + record.player_b_wins + record.draws
        if matches == 0:
            continue

        matrix.at[identity_a.original, identity_b.original] = (
            record.player_a_wins / matches
        )
        matrix.at[identity_b.original, identity_a.original] = (
            record.player_b_wins / matches
        )

    return matrix


def plot_player_win_rates(
    games: Iterable[CompletedGame],
    player_username: str,
    *,
    opponents: Iterable[str] | str,
    min_days: int | None = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot win rates for ``player_username`` against given opponents."""

    stats_df = player_stats(
        games,
        player_username,
        opponents=opponents,
        min_days=min_days,
    )

    if stats_df.empty:
        raise ValueError("No games match the provided criteria")

    plot_ax = ax or plt.subplots(figsize=(max(4, len(stats_df) * 1.2), 4))[1]

    x_labels = stats_df["opponent"].fillna("Unknown opponent")
    win_rates = stats_df["win_rate"].fillna(0.0)

    plot_ax.bar(x_labels, win_rates, color="#1f77b4")
    plot_ax.set_ylim(0, 1)
    plot_ax.set_ylabel("Win Rate")
    plot_ax.set_xlabel("Opponent")
    plot_ax.set_title(f"Win Rate for {player_username}")
    plot_ax.set_xticklabels(x_labels, rotation=45, ha="right")
    plot_ax.grid(axis="y", linestyle="--", alpha=0.3)

    return plot_ax


def plot_win_rate_table(
    games: Iterable[CompletedGame],
    players: Iterable[str],
    *,
    min_days: int | None = None,
    ax: Optional[Axes] = None,
    cmap: str = "Blues",
) -> Axes:
    """Visualize win rates between players as a heatmap-style table."""

    matrix = win_rate_matrix(games, players, min_days=min_days)
    if matrix.empty:
        raise ValueError("No players provided")

    plot_ax = (
        ax
        or plt.subplots(
            figsize=(max(4, len(matrix.columns) * 1.2), max(4, len(matrix.index) * 0.8))
        )[1]
    )

    data = matrix.applymap(lambda value: np.nan if value is None else value).to_numpy(
        dtype=float
    )
    im = plot_ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)

    players_list = list(matrix.index)
    plot_ax.set_xticks(
        range(len(players_list)), labels=players_list, rotation=45, ha="right"
    )
    plot_ax.set_yticks(range(len(players_list)), labels=players_list)
    plot_ax.set_xlabel("Opponent")
    plot_ax.set_ylabel("Player")
    plot_ax.set_title("Head-to-Head Win Rates")

    for i in range(len(players_list)):
        for j in range(len(players_list)):
            value = data[i, j]
            if np.isnan(value):
                text = "—"
            else:
                text = f"{value:.0%}"
            text_color = "black" if np.isnan(value) or value < 0.6 else "white"
            plot_ax.text(j, i, text, ha="center", va="center", color=text_color)

    plot_ax.figure.colorbar(im, ax=plot_ax, fraction=0.046, pad=0.04, label="Win Rate")
    plot_ax.set_xlim(-0.5, len(players_list) - 0.5)
    plot_ax.set_ylim(len(players_list) - 0.5, -0.5)
    plot_ax.grid(False)

    return plot_ax


def team_win_rates(
    games: Iterable[CompletedGame],
    players: Iterable[str],
    *,
    team_size: int = 2,
    min_days: int | None = None,
) -> pd.DataFrame:
    """Return win rates for every unique team pairing that actually occurs in games.

    Parameters
    ----------
    games:
        Completed game records to analyze.
    players:
        Iterable of player usernames to consider when forming teams.
    team_size:
        Number of players per team (default 2).
    min_days:
        Minimum day threshold to include games (matches shorter than this are ignored).

    Returns
    -------
    pd.DataFrame
        A dataframe with one row per disjoint team matchup, including match counts,
        wins/losses/draws for each side, and win rates. Only matchups that occurred
        in the provided games are included.
    """

    if team_size < 1:
        raise ValueError("team_size must be at least 1")

    identities = _normalize_players(players)
    if len(identities) < team_size:
        raise ValueError("Not enough players provided to form teams")

    teams = _build_team_identities(identities, team_size)
    if not teams:
        raise ValueError("No valid teams could be constructed from provided players")

    games_list = _filter_games_by_day(games, min_days)
    counts = _aggregate_team_head_to_head(games_list, teams)

    team_map = {team.normalized_members: team for team in teams}
    rows: List[dict[str, object]] = []

    for (team_a_norm, team_b_norm), record in counts.items():
        team_a = team_map[team_a_norm]
        team_b = team_map[team_b_norm]

        matches = record.team_a_wins + record.team_b_wins + record.draws
        if matches == 0:
            continue

        rows.append(
            {
                "team_a": team_a.label,
                "team_b": team_b.label,
                "matches": matches,
                "team_a_wins": record.team_a_wins,
                "team_b_wins": record.team_b_wins,
                "draws": record.draws,
                "team_a_win_rate": record.team_a_wins / matches,
                "team_b_win_rate": record.team_b_wins / matches,
                "draw_rate": record.draws / matches,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "team_a",
            "team_b",
            "matches",
            "team_a_wins",
            "team_b_wins",
            "draws",
            "team_a_win_rate",
            "team_b_win_rate",
            "draw_rate",
        ],
    )


def _summarize_results(
    games: Iterable[CompletedGame],
    player_username: str,
    *,
    opponent_username: Optional[str],
) -> dict[str, object]:
    records = _collect_player_results(games, player_username, opponent_username)
    counts = Counter(record.result for record in records)
    matches = len(records)
    wins = counts.get("win", 0)
    return {
        "player": player_username,
        "opponent": opponent_username,
        "matches": matches,
        "wins": wins,
        "losses": counts.get("loss", 0),
        "draws": counts.get("draw", 0),
        "win_rate": wins / matches if matches else None,
    }


def _collect_player_results(
    games: Iterable[CompletedGame],
    player_username: str,
    opponent_username: Optional[str],
) -> List[_PlayerMatchResult]:
    results: List[_PlayerMatchResult] = []

    normalized_player = _normalize(player_username)
    normalized_opponent = _normalize(opponent_username) if opponent_username else None

    for game in games:
        player_entry = _find_player(game.players, normalized_player)
        if player_entry is None:
            continue

        if normalized_opponent:
            opponent_entry = _find_player(game.players, normalized_opponent)
            if opponent_entry is None:
                continue
            if _same_team(player_entry, opponent_entry):
                continue
        else:
            opponent_entry = _pick_opponent_on_other_team(game.players, player_entry)
            if opponent_entry is None:
                continue

        result = _determine_result(player_entry, game.players)
        results.append(
            _PlayerMatchResult(
                player=player_entry.username,
                opponent=opponent_entry.username if opponent_entry else None,
                result=result,
            )
        )

    return results


def _find_player(
    players: Iterable[CompletedGamePlayer],
    normalized_username: str,
) -> Optional[CompletedGamePlayer]:
    for player in players:
        if _normalize(player.username) == normalized_username:
            return player
    return None


def _determine_result(
    player: CompletedGamePlayer,
    all_players: Iterable[CompletedGamePlayer],
) -> str:
    if player.is_winner:
        return "win"
    opponent_won = any(other.is_winner for other in all_players if other is not player)
    return "loss" if opponent_won else "draw"


def _normalize(name: Optional[str]) -> str:
    return name.strip().lower() if name else ""


def _normalize_opponents(opponents: Iterable[str] | str | None) -> List[str]:
    if opponents is None:
        return []

    if isinstance(opponents, str):
        iterator = [opponents]
    else:
        iterator = opponents

    normalized: List[str] = []
    for opponent in iterator:
        clean = opponent.strip()
        if clean:
            normalized.append(clean)
    return normalized


def _normalize_players(players: Iterable[str]) -> List[_PlayerIdentity]:
    identities: List[_PlayerIdentity] = []
    seen: set[str] = set()
    for name in players:
        clean = name.strip()
        if not clean:
            continue
        normalized = _normalize(clean)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        identities.append(_PlayerIdentity(original=clean, normalized=normalized))
    return identities


def _build_team_identities(
    identities: Sequence[_PlayerIdentity],
    team_size: int,
) -> List[_TeamIdentity]:
    teams: List[_TeamIdentity] = []
    for combo in combinations(identities, team_size):
        normalized_members = tuple(sorted(member.normalized for member in combo))
        label = " & ".join(member.original for member in combo)
        teams.append(
            _TeamIdentity(
                members=tuple(combo),
                normalized_members=normalized_members,
                label=label,
            )
        )
    return teams


def _aggregate_head_to_head(
    games: Iterable[CompletedGame],
    identity_map: Dict[str, _PlayerIdentity],
) -> Dict[tuple[str, str], _HeadToHeadCount]:
    counts: Dict[tuple[str, str], _HeadToHeadCount] = {}
    relevant = set(identity_map.keys())

    for game in games:
        players_in_game: Dict[str, CompletedGamePlayer] = {}
        for participant in game.players:
            normalized = _normalize(participant.username)
            if normalized in relevant:
                players_in_game[normalized] = participant

        if len(players_in_game) < 2:
            continue

        for norm_a, norm_b in combinations(sorted(players_in_game.keys()), 2):
            participant_a = players_in_game[norm_a]
            participant_b = players_in_game[norm_b]

            if _same_team(participant_a, participant_b):
                continue

            key = (norm_a, norm_b)
            record = counts.get(key)
            if record is None:
                record = _HeadToHeadCount(player_a=norm_a, player_b=norm_b)
                counts[key] = record

            if participant_a.is_winner and participant_b.is_winner:
                record.draws += 1
            elif participant_a.is_winner:
                record.player_a_wins += 1
            elif participant_b.is_winner:
                record.player_b_wins += 1
            else:
                record.draws += 1

    return counts


def _aggregate_team_head_to_head(
    games: Iterable[CompletedGame],
    teams: Sequence[_TeamIdentity],
) -> Dict[tuple[tuple[str, ...], tuple[str, ...]], _TeamHeadToHeadCount]:
    counts: Dict[tuple[tuple[str, ...], tuple[str, ...]], _TeamHeadToHeadCount] = {}
    relevant_players = {
        normalized for team in teams for normalized in team.normalized_members
    }

    for game in games:
        players_in_game: Dict[str, CompletedGamePlayer] = {}
        for participant in game.players:
            normalized = _normalize(participant.username)
            if normalized in relevant_players:
                players_in_game[normalized] = participant

        if len(players_in_game) < 2:
            continue

        for team_a, team_b in combinations(teams, 2):
            if team_a.member_set & team_b.member_set:
                continue

            members_a = [
                players_in_game.get(normalized)
                for normalized in team_a.normalized_members
            ]
            if any(member is None for member in members_a):
                continue

            members_b = [
                players_in_game.get(normalized)
                for normalized in team_b.normalized_members
            ]
            if any(member is None for member in members_b):
                continue

            team_name_a = _shared_team_name(members_a)
            team_name_b = _shared_team_name(members_b)

            if not team_name_a or not team_name_b:
                continue
            if team_name_a == team_name_b:
                continue

            team_a_won = any(member.is_winner for member in members_a)
            team_b_won = any(member.is_winner for member in members_b)

            key = (team_a.normalized_members, team_b.normalized_members)
            record = counts.get(key)
            if record is None:
                record = _TeamHeadToHeadCount(
                    team_a=team_a.normalized_members, team_b=team_b.normalized_members
                )
                counts[key] = record

            if team_a_won and team_b_won:
                record.draws += 1
            elif team_a_won:
                record.team_a_wins += 1
            elif team_b_won:
                record.team_b_wins += 1
            else:
                record.draws += 1

    return counts


def _shared_team_name(players: Sequence[CompletedGamePlayer]) -> Optional[str]:
    team_names = {_normalize(player.team) for player in players if player.team}
    if len(team_names) != 1:
        return None
    return next(iter(team_names))


def _pick_opponent_on_other_team(
    players: Iterable[CompletedGamePlayer],
    player_entry: CompletedGamePlayer,
) -> Optional[CompletedGamePlayer]:
    player_team = _normalize(player_entry.team)
    for candidate in players:
        if candidate is player_entry:
            continue
        if _normalize(candidate.team) != player_team:
            return candidate
    return None


def _same_team(player_a: CompletedGamePlayer, player_b: CompletedGamePlayer) -> bool:
    return _normalize(player_a.team) == _normalize(player_b.team)


def _filter_games_by_day(
    games: Iterable[CompletedGame],
    min_days: int | None,
) -> List[CompletedGame]:
    if min_days is None:
        return list(games)
    if min_days < 0:
        raise ValueError("min_days cannot be negative")
    return [game for game in games if game.day >= min_days]
