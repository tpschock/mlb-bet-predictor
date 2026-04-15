"""
update_ratings.py — Fetch current MLB standings + run differentials
and build a simple power-rating file used by predictor.py.

Run this once per day (before predictor.py) via GitHub Actions.

Data source: MLB Stats API (free, no key required).
"""

import json
import logging
from pathlib import Path

import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MLB_STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings"
LEAGUE_IDS        = [103, 104]   # AL=103, NL=104

# Map MLB Stats API team names → DraftKings display names
TEAM_NAME_MAP = {
    "Arizona Diamondbacks":   "Arizona Diamondbacks",
    "Atlanta Braves":         "Atlanta Braves",
    "Baltimore Orioles":      "Baltimore Orioles",
    "Boston Red Sox":         "Boston Red Sox",
    "Chicago Cubs":           "Chicago Cubs",
    "Chicago White Sox":      "Chicago White Sox",
    "Cincinnati Reds":        "Cincinnati Reds",
    "Cleveland Guardians":    "Cleveland Guardians",
    "Colorado Rockies":       "Colorado Rockies",
    "Detroit Tigers":         "Detroit Tigers",
    "Houston Astros":         "Houston Astros",
    "Kansas City Royals":     "Kansas City Royals",
    "Los Angeles Angels":     "Los Angeles Angels",
    "Los Angeles Dodgers":    "Los Angeles Dodgers",
    "Miami Marlins":          "Miami Marlins",
    "Milwaukee Brewers":      "Milwaukee Brewers",
    "Minnesota Twins":        "Minnesota Twins",
    "New York Mets":          "New York Mets",
    "New York Yankees":       "New York Yankees",
    "Oakland Athletics":      "Athletics",
    "Athletics":              "Athletics",
    "Philadelphia Phillies":  "Philadelphia Phillies",
    "Pittsburgh Pirates":     "Pittsburgh Pirates",
    "San Diego Padres":       "San Diego Padres",
    "San Francisco Giants":   "San Francisco Giants",
    "Seattle Mariners":       "Seattle Mariners",
    "St. Louis Cardinals":    "St. Louis Cardinals",
    "Tampa Bay Rays":         "Tampa Bay Rays",
    "Texas Rangers":          "Texas Rangers",
    "Toronto Blue Jays":      "Toronto Blue Jays",
    "Washington Nationals":   "Washington Nationals",
}


def fetch_standings() -> list[dict]:
    records = []
    for league in LEAGUE_IDS:
        params = {"leagueId": league, "season": 2026, "standingsTypes": "regularSeason"}
        try:
            r = requests.get(MLB_STANDINGS_URL, params=params, timeout=10)
            r.raise_for_status()
            for division in r.json().get("records", []):
                for team_rec in division.get("teamRecords", []):
                    records.append(team_rec)
        except requests.RequestException as e:
            log.error(f"Standings fetch error (league {league}): {e}")
    return records


def compute_ratings(records: list[dict]) -> dict[str, float]:
    """
    Power rating = 50 + (run_diff_per_game * 5), scaled to 0-100.
    Teams with more games played are weighted more heavily.
    """
    ratings: dict[str, float] = {}
    for rec in records:
        name      = rec["team"]["name"]
        dk_name   = TEAM_NAME_MAP.get(name, name)
        wins      = rec.get("wins", 0)
        losses    = rec.get("losses", 0)
        games     = wins + losses
        run_diff  = rec.get("runDifferential", 0)

        if games == 0:
            ratings[dk_name] = 50.0
            continue

        rd_per_game = run_diff / games
        # Confidence weight: ramp up over first 20 games
        confidence  = min(games / 20, 1.0)
        raw_rating  = 50 + (rd_per_game * 5)
        # Blend toward 50 for early season small samples
        rating      = confidence * raw_rating + (1 - confidence) * 50
        ratings[dk_name] = round(max(10, min(90, rating)), 2)

    return ratings


def main():
    log.info("Fetching MLB standings…")
    records = fetch_standings()

    if not records:
        log.warning("No standings data. Using default ratings.")
        return

    ratings = compute_ratings(records)
    log.info(f"Computed ratings for {len(ratings)} teams.")

    # Sort by rating descending for readability
    sorted_r = dict(sorted(ratings.items(), key=lambda x: x[1], reverse=True))

    out = Path("data") / "power_ratings.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(sorted_r, f, indent=2)
    log.info(f"Power ratings saved → {out}")

    print("\n🏟️  Current MLB Power Ratings (season-to-date):\n")
    for i, (team, rating) in enumerate(sorted_r.items(), 1):
        bar = "█" * int(rating / 5)
        print(f"  {i:2}. {team:<28} {rating:5.1f}  {bar}")
    print()


if __name__ == "__main__":
    main()
