"""Fetch and parse completed AWBW games for a specific user."""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field
from awbw_stats_aggregator.lookups import co_name_from_path, country_from_image, team_from_image


BASE_URL = "https://awbw.amarriner.com/"
COMPLETED_GAMES_PATH = "gamescompleted.php"
PAGE_SIZE = 50


class CompletedGamePlayer(BaseModel):
    """Summary information for a participant in a completed game."""

    username: str
    profile_url: str
    is_winner: bool = False
    team: str = ""
    status_text: str = ""
    status_indicator: str = ""
    co_name: str = ""
    co_image: str = ""
    country: str = ""
    country_image: str = ""


class CompletedGame(BaseModel):
    """Parsed representation of a completed game entry."""

    game_id: int
    name: str
    game_url: str
    day: int = 0
    ended_on: datetime = datetime.min
    map_id: int = 0
    map_name: str = ""
    map_url: str = ""
    map_image: str = ""
    replay_url: str = ""
    players: List[CompletedGamePlayer] = Field(default_factory=list)


def get_completed_games(
    username: str,
    *,
    session: Optional[requests.Session] = None,
    max_pages: Optional[int] = None,
) -> List[CompletedGame]:
    """Retrieve all completed games for the given user by scraping AWBW."""

    if not username:
        raise ValueError("username must be provided")

    own_session = session is None
    sess = session or requests.Session()

    games: List[CompletedGame] = []
    start = 1
    pages_fetched = 0

    try:
        while True:
            params = {"username": username, "start": start}
            resp = sess.get(urljoin(BASE_URL, COMPLETED_GAMES_PATH), params=params, timeout=15)
            resp.raise_for_status()

            parsed = _parse_completed_games_page(resp.text)
            games.extend(parsed)
            pages_fetched += 1

            if len(parsed) < PAGE_SIZE:
                break

            if max_pages is not None and pages_fetched >= max_pages:
                break

            start += PAGE_SIZE
    finally:
        if own_session:
            sess.close()

    return games


def _parse_completed_games_page(html: str) -> List[CompletedGame]:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", attrs={"name": lambda x: x and x.startswith("game_")})

    games: List[CompletedGame] = []
    for idx, anchor in enumerate(anchors):
        next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else None
        games.append(_parse_game(anchor, next_anchor))

    return games


def _parse_game(anchor: Tag, next_anchor: Optional[Tag]) -> CompletedGame:
    game_link = anchor.find_next_sibling("a", href=lambda s: s and "game.php?games_id=" in s)
    if game_link is None:
        raise ValueError("Unable to locate game link for anchor", anchor)

    game_id = _extract_query_int(game_link["href"], "games_id")
    name = _extract_game_name(game_link)
    game_url = urljoin(BASE_URL, game_link["href"])

    map_info: Tuple[int, str, str, str] = (0, "", "", "")
    replay_url: str = ""
    day: int = 0
    ended_on: datetime = datetime.min
    players: List[CompletedGamePlayer] = []

    for elem in _iter_between(anchor, next_anchor):
        if isinstance(elem, Tag):
            if map_info == (0, "", "", "") and elem.name == "a" and elem.get("href") and "prevmaps.php" in elem["href"]:
                map_info = _parse_map_link(elem)
            elif elem.name == "a" and elem.get("href") and "game.php" in elem["href"] and "ndx=" in elem["href"]:
                replay_url = urljoin(BASE_URL, elem["href"])
            elif day == 0 and elem.name == "b" and elem.get_text(strip=True).startswith("Day"):
                parsed_day = _parse_day(elem.get_text(strip=True))
                if parsed_day is not None:
                    day = parsed_day
            elif ended_on == datetime.min and elem.name == "span" and "Ended on" in elem.get_text():
                parsed_date = _parse_date(elem.get_text(strip=True))
                if parsed_date is not None:
                    ended_on = parsed_date
            elif elem.name == "div" and elem.get("id") == "do-game-player-row":
                players.append(_parse_player(elem))

    map_id, map_name, map_url, map_image = map_info

    return CompletedGame(
        game_id=game_id,
        name=name,
        game_url=game_url,
        day=day,
        ended_on=ended_on,
        map_id=map_id,
        map_name=map_name,
        map_url=map_url,
        map_image=map_image,
        replay_url=replay_url,
        players=players,
    )


def _iter_between(start: Tag, end: Optional[Tag]) -> Iterator:
    for elem in start.next_elements:
        if elem is end:
            break
        yield elem


def _extract_query_int(url: str, key: str) -> int:
    parsed = urlparse(url)
    try:
        return int(parse_qs(parsed.query)[key][0])
    except (KeyError, IndexError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not extract integer query param '{key}' from {url}") from exc


def _extract_game_name(link: Tag) -> str:
    # The link text is typically "1." and "game name" in separate spans; fall back to full text.
    bold_spans = link.find_all("b")
    if len(bold_spans) >= 2:
        return bold_spans[1].get_text(strip=True)
    return link.get_text(strip=True)


def _parse_day(text: str) -> Optional[int]:
    parts = text.split()
    for part in parts:
        if part.isdigit():
            try:
                return int(part)
            except ValueError:  # pragma: no cover - defensive
                return None
    return None


def _parse_date(text: str) -> Optional[datetime]:
    # Example: "Ended on 10/14/2025"
    tokens = text.split("Ended on", 1)
    if len(tokens) != 2:
        return None
    raw_date = tokens[1].strip()
    try:
        return datetime.strptime(raw_date, "%m/%d/%Y")
    except ValueError:
        return None


def _parse_map_link(link: Tag) -> Tuple[int, str, str, str]:
    map_id = _extract_query_int(link["href"], "maps_id")
    map_name = link.get_text(strip=True)
    map_url = urljoin(BASE_URL, link["href"])
    img = link.find("img")
    map_image = urljoin(BASE_URL, img["src"]) if img and img.get("src") else ""
    return map_id, map_name, map_url, map_image


def _parse_player(container: Tag) -> CompletedGamePlayer:
    username_anchor = container.select_one(".do-game-username a")
    if not username_anchor:
        raise ValueError("Player row lacks username link")

    username = username_anchor.get_text(strip=True)
    profile_url = urljoin(BASE_URL, username_anchor["href"])

    is_winner = bool(container.select_one('.do-game-co-image img[title="Winner"]')) or bool(
        username_anchor.find("b")
    )

    extras = container.find("div", class_="do-game-extras")
    status_text = ""
    status_indicator = ""
    team = ""
    if extras:
        status_span = extras.find("span", class_="game-tools-btn-text")
        if status_span:
            status_text = status_span.get_text(strip=True)
        indicator_span = extras.find("span", class_=lambda c: c and c.startswith("dot_"))
        if indicator_span and indicator_span.get("class"):
            for cls in indicator_span["class"]:
                if cls.startswith("dot_"):
                    status_indicator = cls
                    break
        team_img = extras.find("img")
        if team_img and team_img.get("src"):
            team = team_from_image(team_img["src"])

    co_img = container.select_one(".do-game-co-image img.co_portrait")
    co_image = urljoin(BASE_URL, co_img["src"]) if co_img and co_img.get("src") else ""
    co_name = co_name_from_path(co_img["src"]) if co_img and co_img.get("src") else ""

    country_img = container.select_one(".do-game-country-logo img")
    country_image = urljoin(BASE_URL, country_img["src"]) if country_img and country_img.get("src") else ""
    country = country_from_image(country_img["src"]) if country_img and country_img.get("src") else ""

    return CompletedGamePlayer(
        username=username,
        profile_url=profile_url,
        is_winner=is_winner,
        team=team,
        status_text=status_text,
        status_indicator=status_indicator,
        co_name=co_name,
        co_image=co_image,
        country=country,
        country_image=country_image,
    )


