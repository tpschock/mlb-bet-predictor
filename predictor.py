"""
MLB Bet Predictor — DraftKings Odds + Starting Pitcher Model
Fetches today's DraftKings MLB moneyline/totals odds, probable starters,
and pitcher stats, then applies a multi-factor model to recommend bets.

Usage:
    python predictor.py                  # predict today's games
    python predictor.py --backtest       # backtest saved history
    python predictor.py --output json    # output as JSON
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs") / f"run_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE    = "https://api.the-odds-api.com/v4/sports"
SPORT        = "baseball_mlb"
REGIONS      = "us"
MARKETS      = "h2h,totals"
BOOKMAKER    = "draftkings"

MIN_EDGE     = 5.0   # raised from 3% — only bet strong edges
KELLY_FRAC   = 0.25  # fractional Kelly (conservative)
BANKROLL     = 1000

MLB_API      = "https://statsapi.mlb.com/api/v1"

# Blending weights — must sum to 1.0
W_MARKET     = 0.50  # market fair probability
W_POWER      = 0.25  # team power rating
W_PITCHER    = 0.25  # starting pitcher advantage


# ── Odds helpers ──────────────────────────────────────────────────────────────

def american_to_prob(american: float) -> float:
    if american > 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def american_to_decimal(american: float) -> float:
    if american > 0:
        return american / 100 + 1
    return 100 / abs(american) + 1


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def kelly_criterion(fair_prob: float, decimal_odds: float, fraction: float = KELLY_FRAC) -> float:
    b = decimal_odds - 1
    q = 1 - fair_prob
    kelly = (b * fair_prob - q) / b
    return max(0.0, kelly * fraction)


# ── Odds fetching ─────────────────────────────────────────────────────────────

def fetch_odds() -> list[dict]:
    if not ODDS_API_KEY:
        log.warning("No ODDS_API_KEY set — using sample data.")
        return _sample_odds()
    url = f"{ODDS_BASE}/{SPORT}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    REGIONS,
        "markets":    MARKETS,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKER,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info(f"Odds API: {len(resp.json())} games. Requests remaining: {remaining}")
        return resp.json()
    except requests.RequestException as exc:
        log.error(f"Odds fetch failed: {exc}")
        return []


# ── MLB Stats API — schedule + probable pitchers ──────────────────────────────

def fetch_probable_pitchers() -> dict[str, dict]:
    """
    Returns a dict keyed by MLB team name:
      { "Minnesota Twins": {"name": "Pablo Lopez", "id": 641154}, ... }
    Uses today's schedule endpoint which includes probablePitcher when available.
    """
    today  = datetime.now().strftime("%Y-%m-%d")
    url    = f"{MLB_API}/schedule"
    params = {
        "sportId": 1,
        "date":    today,
        "hydrate": "probablePitcher(note),team",
        "fields":  "dates,games,teams,team,name,probablePitcher,id,fullName",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.warning(f"Schedule fetch failed: {exc}")
        return {}

    pitchers: dict[str, dict] = {}
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            for side in ("home", "away"):
                team_info    = game.get("teams", {}).get(side, {})
                team_name    = team_info.get("team", {}).get("name", "")
                probable     = team_info.get("probablePitcher", {})
                pitcher_name = probable.get("fullName", "TBD")
                pitcher_id   = probable.get("id")
                if team_name:
                    pitchers[team_name] = {"name": pitcher_name, "id": pitcher_id}

    log.info(f"Probable pitchers fetched for {len(pitchers)} team slots.")
    return pitchers


def fetch_pitcher_stats(pitcher_id: int) -> dict:
    """
    Fetch current season stats for a pitcher from the MLB Stats API.
    Returns era, whip, k_per9, bb_per9, innings_pitched.
    Falls back to league-average defaults if unavailable.
    """
    LEAGUE_AVG = {
        "era": 4.20, "whip": 1.28, "k_per9": 8.5,
        "bb_per9": 3.1, "innings_pitched": 0,
    }

    if not pitcher_id:
        return LEAGUE_AVG

    url    = f"{MLB_API}/people/{pitcher_id}/stats"
    params = {"stats": "season", "group": "pitching", "season": datetime.now().year}
    try:
        resp   = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data   = resp.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return LEAGUE_AVG
        s = splits[0].get("stat", {})
        return {
            "era":             float(s.get("era",               LEAGUE_AVG["era"])),
            "whip":            float(s.get("whip",              LEAGUE_AVG["whip"])),
            "k_per9":          float(s.get("strikeoutsPer9Inn", LEAGUE_AVG["k_per9"])),
            "bb_per9":         float(s.get("walksPer9Inn",      LEAGUE_AVG["bb_per9"])),
            "innings_pitched": float(s.get("inningsPitched",    0)),
        }
    except Exception as exc:
        log.debug(f"Pitcher stats fetch failed for id {pitcher_id}: {exc}")
        return LEAGUE_AVG


def pitcher_advantage(home_stats: dict, away_stats: dict) -> tuple[float, float]:
    """
    Compare the two starters and return a win probability for each team.
    Lower ERA and WHIP = better pitcher = higher win probability for that side.
    Returns (home_pitcher_prob, away_pitcher_prob) summing to 1.0.
    """
    LEAGUE_ERA  = 4.20
    LEAGUE_WHIP = 1.28

    def pitcher_score(stats: dict) -> float:
        # Invert ERA and WHIP so lower values produce higher scores
        era_score  = max(0.1, (LEAGUE_ERA  * 2) - stats["era"])  * 1.5
        whip_score = max(0.1, (LEAGUE_WHIP * 2) - stats["whip"]) * 1.0
        k_score    = stats["k_per9"] * 0.3
        # Blend toward neutral (5.0) for pitchers with small sample sizes
        ip_confidence = min(stats["innings_pitched"] / 30, 1.0)
        base = era_score + whip_score + k_score
        return base * ip_confidence + (1 - ip_confidence) * 5.0

    home_score = pitcher_score(home_stats)
    away_score = pitcher_score(away_stats)
    total      = home_score + away_score
    if total == 0:
        return 0.5, 0.5
    return home_score / total, away_score / total


# ── Team power ratings ────────────────────────────────────────────────────────

def build_power_ratings() -> dict:
    ratings_file = Path("data") / "power_ratings.json"
    if ratings_file.exists():
        with open(ratings_file) as f:
            return json.load(f)
    log.warning("power_ratings.json not found — using neutral 50 for all teams.")
    return {}


def team_power_prob(home: str, away: str, ratings: dict) -> tuple[float, float]:
    home_r = ratings.get(home, 50.0)
    away_r = ratings.get(away, 50.0)
    total  = home_r + away_r
    home_p = (home_r / total) if total else 0.5
    # Home-field advantage: ~54% historical win rate at home in MLB
    home_p = min(0.95, home_p * 1.04)
    return home_p, 1 - home_p


# ── Core model ────────────────────────────────────────────────────────────────

def blend_probabilities(
    home_fair:    float,
    away_fair:    float,
    home_power:   float,
    away_power:   float,
    home_pitcher: float,
    away_pitcher: float,
) -> tuple[float, float]:
    """
    Weighted blend of three independent signals:
      50% market fair probability  (vig-stripped DK odds)
      25% team power rating        (season run differential)
      25% starting pitcher matchup (ERA / WHIP / K9)
    """
    home_blend = (W_MARKET * home_fair) + (W_POWER * home_power) + (W_PITCHER * home_pitcher)
    return home_blend, 1 - home_blend


# ── Game analysis ─────────────────────────────────────────────────────────────

def analyze_game(game: dict, ratings: dict, probable_pitchers: dict) -> dict | None:
    home = game["home_team"]
    away = game["away_team"]

    dk = next((b for b in game.get("bookmakers", []) if b["key"] == BOOKMAKER), None)
    if not dk:
        return None

    markets = {m["key"]: m["outcomes"] for m in dk.get("markets", [])}

    # ── Moneyline ──
    h2h          = markets.get("h2h", [])
    home_outcome = next((o for o in h2h if o["name"] == home), None)
    away_outcome = next((o for o in h2h if o["name"] == away), None)
    if not home_outcome or not away_outcome:
        return None

    home_odds = home_outcome["price"]
    away_odds = away_outcome["price"]
    home_fair, away_fair = remove_vig(american_to_prob(home_odds), american_to_prob(away_odds))

    # ── Power ratings ──
    home_power, away_power = team_power_prob(home, away, ratings)

    # ── Pitcher stats ──
    home_pitcher_info  = probable_pitchers.get(home, {"name": "TBD", "id": None})
    away_pitcher_info  = probable_pitchers.get(away, {"name": "TBD", "id": None})
    home_pitcher_stats = fetch_pitcher_stats(home_pitcher_info.get("id"))
    away_pitcher_stats = fetch_pitcher_stats(away_pitcher_info.get("id"))
    home_pitcher_prob, away_pitcher_prob = pitcher_advantage(home_pitcher_stats, away_pitcher_stats)

    # ── Blend ──
    home_blend, away_blend = blend_probabilities(
        home_fair, away_fair,
        home_power, away_power,
        home_pitcher_prob, away_pitcher_prob,
    )

    home_edge  = (home_blend - home_fair) * 100
    away_edge  = (away_blend - away_fair) * 100
    home_kelly = kelly_criterion(home_blend, american_to_decimal(home_odds))
    away_kelly = kelly_criterion(away_blend, american_to_decimal(away_odds))

    # ── Totals ──
    totals    = markets.get("totals", [])
    over_out  = next((o for o in totals if o["name"] == "Over"),  None)
    under_out = next((o for o in totals if o["name"] == "Under"), None)
    total_line = over_out["point"] if over_out else None
    over_imp   = american_to_prob(over_out["price"])  if over_out  else None
    under_imp  = american_to_prob(under_out["price"]) if under_out else None

    # ── Picks ──
    picks = []
    if home_edge >= MIN_EDGE:
        picks.append({
            "bet":   f"{home} ML",
            "odds":  home_odds,
            "edge":  round(home_edge, 2),
            "kelly": round(home_kelly * 100, 1),
            "wager": round(BANKROLL * home_kelly, 2),
        })
    if away_edge >= MIN_EDGE:
        picks.append({
            "bet":   f"{away} ML",
            "odds":  away_odds,
            "edge":  round(away_edge, 2),
            "kelly": round(away_kelly * 100, 1),
            "wager": round(BANKROLL * away_kelly, 2),
        })

    return {
        "game":              f"{away} @ {home}",
        "commence":          game.get("commence_time", ""),
        "home_starter":      home_pitcher_info["name"],
        "away_starter":      away_pitcher_info["name"],
        "home_starter_era":  round(home_pitcher_stats["era"],  2),
        "away_starter_era":  round(away_pitcher_stats["era"],  2),
        "home_starter_whip": round(home_pitcher_stats["whip"], 2),
        "away_starter_whip": round(away_pitcher_stats["whip"], 2),
        "home_starter_k9":   round(home_pitcher_stats["k_per9"], 1),
        "away_starter_k9":   round(away_pitcher_stats["k_per9"], 1),
        "home_odds":         home_odds,
        "away_odds":         away_odds,
        "home_fair_p":       round(home_fair         * 100, 1),
        "away_fair_p":       round(away_fair         * 100, 1),
        "home_power_p":      round(home_power        * 100, 1),
        "away_power_p":      round(away_power        * 100, 1),
        "home_pitcher_p":    round(home_pitcher_prob * 100, 1),
        "away_pitcher_p":    round(away_pitcher_prob * 100, 1),
        "home_blend_p":      round(home_blend        * 100, 1),
        "away_blend_p":      round(away_blend        * 100, 1),
        "home_edge":         round(home_edge, 2),
        "away_edge":         round(away_edge, 2),
        "total_line":        total_line,
        "over_imp":          round(over_imp  * 100, 1) if over_imp  else None,
        "under_imp":         round(under_imp * 100, 1) if under_imp else None,
        "value_picks":       picks,
    }


# ── Console report ────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    today = datetime.now().strftime("%A %B %d, %Y")
    print(f"\n{'═'*64}")
    print(f"  MLB BET PREDICTOR — {today}")
    print(f"  DraftKings Odds + Pitcher Model | Min Edge: {MIN_EDGE}%")
    print(f"{'═'*64}\n")

    value_games = [r for r in results if r["value_picks"]]
    no_value    = [r for r in results if not r["value_picks"]]

    if value_games:
        print(f"🎯  VALUE BETS ({len(value_games)} game(s))\n")
        for r in value_games:
            print(f"  {r['game']}")
            print(f"  Starters — Away: {r['away_starter']}  "
                  f"({r['away_starter_era']} ERA | {r['away_starter_whip']} WHIP | {r['away_starter_k9']} K/9)")
            print(f"             Home: {r['home_starter']}  "
                  f"({r['home_starter_era']} ERA | {r['home_starter_whip']} WHIP | {r['home_starter_k9']} K/9)")
            print(f"  Market fair:   Away {r['away_fair_p']}%   | Home {r['home_fair_p']}%")
            print(f"  Power rating:  Away {r['away_power_p']}%  | Home {r['home_power_p']}%")
            print(f"  Pitcher edge:  Away {r['away_pitcher_p']}% | Home {r['home_pitcher_p']}%")
            print(f"  Final blend:   Away {r['away_blend_p']}%   | Home {r['home_blend_p']}%")
            if r["total_line"]:
                print(f"  Total: {r['total_line']}  (Over {r['over_imp']}% / Under {r['under_imp']}%)")
            for pick in r["value_picks"]:
                print(f"  ✅  {pick['bet']}  {pick['odds']:+d}  "
                      f"edge={pick['edge']}%  Kelly={pick['kelly']}%  wager=${pick['wager']}")
            print()
    else:
        print("  No value bets found today.\n")

    if no_value:
        print(f"⛔  NO EDGE ({len(no_value)} game(s)):")
        for r in no_value:
            best = max(r["home_edge"], r["away_edge"])
            print(f"     {r['game']}  "
                  f"(best edge: {best:.1f}%  |  {r['away_starter']} vs {r['home_starter']})")

    print(f"\n{'─'*64}")
    print("  ⚠️  For entertainment/research only. Gamble responsibly.")
    print(f"{'─'*64}\n")


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest() -> None:
    pred_dir = Path("predictions")
    files    = sorted(pred_dir.glob("predictions_*.json"))
    if not files:
        log.warning("No saved prediction files found.")
        return

    total_bets = wins = losses = push = 0
    total_wagered = total_return = 0.0

    for f in files:
        with open(f) as fh:
            saved = json.load(fh)
        date   = saved.get("date", f.stem.split("_")[-1])
        result = Path("data") / f"results_{date}.json"
        if not result.exists():
            continue
        with open(result) as fh:
            actuals = json.load(fh)

        for game in saved.get("results", []):
            for pick in game.get("value_picks", []):
                team   = pick["bet"].replace(" ML", "")
                wager  = pick["wager"]
                winner = actuals.get(game["game"])
                if winner is None:
                    push += 1
                    continue
                total_bets    += 1
                total_wagered += wager
                if winner == team:
                    wins         += 1
                    total_return += wager * american_to_decimal(pick["odds"])
                else:
                    losses += 1

    if total_bets == 0:
        print("No completed bets to evaluate yet.")
        return

    roi = (total_return - total_wagered) / total_wagered * 100
    print(f"\n📊  BACKTEST RESULTS  ({len(files)} days)")
    print(f"  Bets: {total_bets}  |  W-L: {wins}-{losses}  |  Push: {push}")
    print(f"  Hit rate: {wins / total_bets * 100:.1f}%")
    print(f"  Wagered: ${total_wagered:.2f}  |  Returned: ${total_return:.2f}")
    print(f"  ROI: {roi:+.1f}%\n")


# ── Sample data (no API key) ──────────────────────────────────────────────────

def _sample_odds() -> list[dict]:
    return [
        {
            "id": "sample1",
            "home_team": "Minnesota Twins",
            "away_team": "Boston Red Sox",
            "commence_time": datetime.now(timezone.utc).isoformat(),
            "bookmakers": [{"key": "draftkings", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Minnesota Twins", "price": -145},
                    {"name": "Boston Red Sox",  "price": +125},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over",  "price": -110, "point": 8.5},
                    {"name": "Under", "price": -110, "point": 8.5},
                ]},
            ]}],
        },
        {
            "id": "sample2",
            "home_team": "Houston Astros",
            "away_team": "Colorado Rockies",
            "commence_time": datetime.now(timezone.utc).isoformat(),
            "bookmakers": [{"key": "draftkings", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Houston Astros",   "price": -175},
                    {"name": "Colorado Rockies", "price": +148},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over",  "price": -115, "point": 9.5},
                    {"name": "Under", "price": -105, "point": 9.5},
                ]},
            ]}],
        },
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB Bet Predictor")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--output", choices=["text", "json"], default="text")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
        return

    log.info("Fetching odds…")
    games = fetch_odds()

    log.info("Fetching probable pitchers…")
    probable_pitchers = fetch_probable_pitchers()

    log.info("Loading power ratings…")
    ratings = build_power_ratings()

    results = []
    for g in games:
        r = analyze_game(g, ratings, probable_pitchers)
        if r:
            results.append(r)

    if args.output == "json":
        print(json.dumps({"date": datetime.now().strftime("%Y%m%d"), "results": results}, indent=2))
    else:
        print_report(results)

    out_file = Path("predictions") / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out_file, "w") as f:
        json.dump({"date": datetime.now().strftime("%Y%m%d"), "results": results}, f, indent=2)
    log.info(f"Predictions saved → {out_file}")


if __name__ == "__main__":
    main()
