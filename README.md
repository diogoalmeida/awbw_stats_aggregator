# awbw_stats_aggregator
Collects game stats from AWBW (Advance Wars By Web): https://awbw.amarriner.com/

## Installation (uv)

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install runtime dependencies
uv sync

# (Optional) install notebook/tooling extras
uv sync --group dev
```

To work inside a shell with the project environment activated:

```bash
uv run -- jupyter notebook  # or any other command
```

## Usage

```python
from awbw_stats_aggregator import (
    get_completed_games,
    player_stats,
    win_rate_matrix,
)

games = get_completed_games("ExampleUser", max_pages=1)

# Overall record for a player
overall = player_stats(games, "ExampleUser")

# Head-to-head against specific rivals
head_to_head = player_stats(
    games,
    "ExampleUser",
    opponents=["Rival123", "AnotherOpponent"],
)

# Symmetric win-rate matrix for multiple players
players = ["ExampleUser", "Rival123", "AnotherOpponent"]
matrix = win_rate_matrix(games, players)

print(overall)
print(head_to_head)
print(matrix)
```
