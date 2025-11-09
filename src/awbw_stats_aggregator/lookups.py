"""Utility helpers for interpreting AWBW asset filenames."""

from __future__ import annotations

from urllib.parse import urlparse

COUNTRY_MAP = {
    "oslogo": "Orange Star",
    "bmlogo": "Blue Moon",
    "yclogo": "Yellow Comet",
    "gelogo": "Green Earth",
    "pclogo": "Black Hole",
    "nelogo": "Neotropolis",
    "gslogo": "Green Sky",
    "rflogo": "Rubenelle",
    "tglogo": "Tundra",
    "cilogo": "Cobalt Ice",
    "bdlogo": "Beast",
    "bhlogo": "Black Hole",
    "aalogo": "Amber Blaze",
    "arlogo": "Archaic",
    "gs_ablogo": "Amber Blitz",
    "gs_bdlogo": "Beast (GS)",
    "gs_bhlogo": "Black Hole (GS)",
    "gs_bmlogo": "Blue Moon (GS)",
    "gs_cilogo": "Cobalt Ice (GS)",
    "gs_gelogo": "Green Earth (GS)",
    "gs_gslogo": "Green Sky (GS)",
    "gs_jslogo": "Jade Sun",
    "gs_nelogo": "Neotropolis (GS)",
    "gs_oslogo": "Orange Star (GS)",
    "gs_pclogo": "Black Hole (GS)",
    "gs_pllogo": "Celestial",
    "gs_rflogo": "Rubenelle (GS)",
    "gs_tglogo": "Tundra (GS)",
    "gs_wnlogo": "White Nova",
    "gs_yclogo": "Yellow Comet (GS)",
}


def team_from_image(src: str) -> str:
    """Derive a team identifier from an AWBW asset path."""

    filename = urlparse(src).path.split("/")[-1]
    if not filename:
        return ""
    name = filename.split(".")[0]
    return name.upper()


def co_name_from_path(src: str) -> str:
    """Convert a CO portrait path into a human-readable name."""

    path = urlparse(src).path.split("/")[-1]
    base = path.split(".")[0]
    if not base:
        return ""
    return base.replace("_", " ").title()


def country_from_image(src: str) -> str:
    """Map a country logo image path to its faction name."""

    filename = urlparse(src).path.split("/")[-1]
    key = filename.split(".")[0]
    return COUNTRY_MAP.get(key, key.replace("_", " ").title())
