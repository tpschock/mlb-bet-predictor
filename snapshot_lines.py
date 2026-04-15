"""
snapshot_lines.py — Save opening DraftKings lines for today's MLB games.

Run this BEFORE the main predictor (e.g. 7 AM ET via GitHub Actions).
The predictor then compares these saved lines against the 10 AM lines
to detect significant line movement driven by sharp money.

Saved to: data/opening_lines_YYYYMMDD.json
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE    = "https://api.the-odds-api.com/v4/sports"
SPORT        = "baseball_mlb"
BOOKMAKER    = "draftkings"


def fetch_opening_lines() -> list[dict]:
    if not ODDS_API_KEY:
        log.warning("No ODDS_API_KEY — skipping opening lines snapshot.")
        return []

    url    = f"{ODDS_BASE}/{SPORT}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "h2h",
        "oddsFormat": "american",
        "bookmakers": BOOKMAKER,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        games = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info(f"Opening lines snapshot: {len(games)} games. API requests remaining: {remaining}")
        return games
    except requests.RequestException as exc:
        log.error(f"Opening lines fetch failed: {exc}")
        return []


def extract_lines(games: list[dict]) -> dict:
    """
    Returns a flat dict keyed by game id:
    {
      "game_id": {
        "home_team": "Houston Astros",
        "away_team": "Colorado Rockies",
        "home_odds": -175,
        "away_odds": +148,
        "snapshot_time": "2026-04-15T12:00:00+00:00"
      }
    }
    """
    snapshot_time = datetime.now(timezone.utc).isoformat()
    lines = {}

    for game in games:
        dk = next((b for b in game.get("bookmakers", []) if b["key"] == BOOKMAKER), None)
        if not dk:
            continue
        h2h = next((m for m in dk.get("markets", []) if m["key"] == "h2h"), None)
        if not h2h:
            continue

        home = game["home_team"]
        away = game["away_team"]
        home_out = next((o for o in h2h["outcomes"] if o["name"] == home), None)
        away_out = next((o for o in h2h["outcomes"] if o["name"] == away), None)
        if not home_out or not away_out:
            continue

        lines[game["id"]] = {
            "home_team":     home,
            "away_team":     away,
            "home_odds":     home_out["price"],
            "away_odds":     away_out["price"],
            "snapshot_time": snapshot_time,
        }

    return lines


def main():
    log.info("Taking opening lines snapshot…")
    games = fetch_opening_lines()

    if not games:
        log.info("No games to snapshot. Exiting.")
        return

    lines    = extract_lines(games)
    today    = datetime.now().strftime("%Y%m%d")
    out_file = Path("data") / f"opening_lines_{today}.json"

    # Don't overwrite if already snapshotted today — first snapshot wins
    if out_file.exists():
        log.info(f"Opening lines already saved for {today}. Skipping overwrite.")
        return

    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(lines, f, indent=2)

    log.info(f"Opening lines saved → {out_file}  ({len(lines)} games)")

    # Print summary
    print(f"\n📸  Opening lines snapshot — {today}\n")
    for gid, g in lines.items():
        print(f"  {g['away_team']} @ {g['home_team']}"
              f"   {g['away_odds']:+d} / {g['home_odds']:+d}")
    print()


if __name__ == "__main__":
    main()
