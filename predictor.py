"""
MLB Bet Predictor — DraftKings Odds-Based Algorithm
Fetches today's DraftKings MLB moneyline/totals odds and scores,
then applies a multi-factor model to recommend bets.

Usage:
    python predictor.py                  # predict today's games
    python predictor.py --backtest       # backtest last 7 days
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

# ── Logging ──────────────────────────────────────────────────────────────────
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
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")          # https://the-odds-api.com  (free tier: 500 req/mo)
ODDS_BASE    = "https://api.the-odds-api.com/v4/sports"
SPORT        = "baseball_mlb"
REGIONS      = "us"
MARKETS      = "h2h,totals"                            # moneyline + over/under
BOOKMAKER    = "draftkings"

MIN_EDGE     = 3.0   # minimum edge % to flag as a value bet
KELLY_FRAC   = 0.25  # fractional Kelly (conservative)
BANKROLL     = 100  # hypothetical bankroll for sizing


# ── Odds helpers ──────────────────────────────────────────────────────────────

def american_to_prob(american: float) -> float:
    """Convert American odds to implied probability (0-1)."""
    if american > 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def american_to_decimal(american: float) -> float:
    if american > 0:
        return american / 100 + 1
    return 100 / abs(american) + 1


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Strip the bookmaker's vig to get fair probabilities."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def kelly_criterion(fair_prob: float, decimal_odds: float, fraction: float = KELLY_FRAC) -> float:
    """Return fraction of bankroll to wager (fractional Kelly)."""
    b = decimal_odds - 1
    q = 1 - fair_prob
    kelly = (b * fair_prob - q) / b
    return max(0.0, kelly * fraction)


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_odds() -> list[dict]:
    """Pull today's MLB odds from The Odds API (DraftKings focus)."""
    if not ODDS_API_KEY:
        log.warning("No ODDS_API_KEY set — using sample data. Set env var for live odds.")
        return _sample_odds()

    url = f"{ODDS_BASE}/{SPORT}/odds"
    params = {
        "apiKey":  ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKER,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info(f"Odds API: {len(resp.json())} games fetched. Requests remaining: {remaining}")
        return resp.json()
    except requests.RequestException as exc:
        log.error(f"Odds fetch failed: {exc}")
        return []


def _sample_odds() -> list[dict]:
    """Hardcoded sample so the script runs without an API key."""
    return [
        {
            "id": "sample1",
            "home_team": "Minnesota Twins",
            "away_team": "Boston Red Sox",
            "commence_time": datetime.now(timezone.utc).isoformat(),
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Minnesota Twins",  "price": -145},
                                {"name": "Boston Red Sox",   "price": +125},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over",  "price": -110, "point": 8.5},
                                {"name": "Under", "price": -110, "point": 8.5},
                            ],
                        },
                    ],
                }
            ],
        },
        {
            "id": "sample2",
            "home_team": "Houston Astros",
            "away_team": "Colorado Rockies",
            "commence_time": datetime.now(timezone.utc).isoformat(),
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Houston Astros",    "price": -175},
                                {"name": "Colorado Rockies",  "price": +148},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over",  "price": -115, "point": 9.5},
                                {"name": "Under", "price": -105, "point": 9.5},
                            ],
                        },
                    ],
                }
            ],
        },
    ]


# ── Model ─────────────────────────────────────────────────────────────────────

def team_power_rating(team_name: str, ratings: dict) -> float:
    """
    Look up a pre-built power rating (0–100 scale).
    Replace / extend this with your own model (e.g. ELO, FanGraphs WAR).
    """
    return ratings.get(team_name, 50.0)


def build_power_ratings() -> dict:
    """
    Season-to-date simple power ratings derived from run differential.
    Refreshed daily by update_ratings.py.
    If the file doesn't exist, fall back to league-average (50).
    """
    ratings_file = Path("data") / "power_ratings.json"
    if ratings_file.exists():
        with open(ratings_file) as f:
            return json.load(f)
    log.warning("power_ratings.json not found — using neutral 50 for all teams.")
    return {}


def model_win_probability(home: str, away: str, ratings: dict) -> tuple[float, float]:
    """
    Blend DK implied probability with our power-rating model.
    Returns (home_win_prob, away_win_prob) as fair probabilities.
    """
    home_r = team_power_rating(home, ratings)
    away_r = team_power_rating(away, ratings)
    total   = home_r + away_r
    home_model = home_r / total if total else 0.5
    # Home-field bump: historically ~54% win rate at home in MLB
    home_model = min(0.95, home_model * 1.04)
    away_model = 1 - home_model
    return home_model, away_model


def analyze_game(game: dict, ratings: dict) -> dict | None:
    """Run the full model on a single game and return a recommendation dict."""
    home = game["home_team"]
    away = game["away_team"]

    # Pull DraftKings markets
    dk = next((b for b in game.get("bookmakers", []) if b["key"] == BOOKMAKER), None)
    if not dk:
        log.debug(f"No DK lines for {away} @ {home}")
        return None

    markets = {m["key"]: m["outcomes"] for m in dk.get("markets", [])}

    # ── Moneyline ──
    h2h = markets.get("h2h", [])
    home_outcome = next((o for o in h2h if o["name"] == home), None)
    away_outcome = next((o for o in h2h if o["name"] == away), None)
    if not home_outcome or not away_outcome:
        return None

    home_odds   = home_outcome["price"]
    away_odds   = away_outcome["price"]
    home_imp    = american_to_prob(home_odds)
    away_imp    = american_to_prob(away_odds)
    home_fair, away_fair = remove_vig(home_imp, away_imp)

    # Model probabilities
    home_model, away_model = model_win_probability(home, away, ratings)

    # Blend: 60% market, 40% our model
    home_blended = 0.60 * home_fair + 0.40 * home_model
    away_blended = 1 - home_blended

    home_edge = (home_blended - home_fair) * 100
    away_edge = (away_blended - away_fair) * 100

    # Kelly sizing
    home_kelly = kelly_criterion(home_blended, american_to_decimal(home_odds))
    away_kelly = kelly_criterion(away_blended, american_to_decimal(away_odds))

    # ── Totals ──
    totals = markets.get("totals", [])
    over_out  = next((o for o in totals if o["name"] == "Over"),  None)
    under_out = next((o for o in totals if o["name"] == "Under"), None)

    total_line = over_out["point"] if over_out else None
    over_imp   = american_to_prob(over_out["price"])  if over_out  else None
    under_imp  = american_to_prob(under_out["price"]) if under_out else None

    # ── Recommendation ──
    picks = []
    if home_edge >= MIN_EDGE:
        wager = round(BANKROLL * home_kelly, 2)
        picks.append({
            "bet":   f"{home} ML",
            "odds":  home_odds,
            "edge":  round(home_edge, 2),
            "kelly": round(home_kelly * 100, 1),
            "wager": wager,
        })
    if away_edge >= MIN_EDGE:
        wager = round(BANKROLL * away_kelly, 2)
        picks.append({
            "bet":   f"{away} ML",
            "odds":  away_odds,
            "edge":  round(away_edge, 2),
            "kelly": round(away_kelly * 100, 1),
            "wager": wager,
        })

    return {
        "game":         f"{away} @ {home}",
        "commence":     game.get("commence_time", ""),
        "home_odds":    home_odds,
        "away_odds":    away_odds,
        "home_fair_p":  round(home_fair * 100, 1),
        "away_fair_p":  round(away_fair * 100, 1),
        "home_model_p": round(home_model * 100, 1),
        "away_model_p": round(away_model * 100, 1),
        "home_blend_p": round(home_blended * 100, 1),
        "away_blend_p": round(away_blended * 100, 1),
        "home_edge":    round(home_edge, 2),
        "away_edge":    round(away_edge, 2),
        "total_line":   total_line,
        "over_imp":     round(over_imp * 100, 1) if over_imp else None,
        "under_imp":    round(under_imp * 100, 1) if under_imp else None,
        "value_picks":  picks,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    today = datetime.now().strftime("%A %B %d, %Y")
    print(f"\n{'═'*60}")
    print(f"  MLB BET PREDICTOR — {today}")
    print(f"  Powered by DraftKings Odds | Min Edge: {MIN_EDGE}%")
    print(f"{'═'*60}\n")

    value_games = [r for r in results if r["value_picks"]]
    no_value    = [r for r in results if not r["value_picks"]]

    if value_games:
        print(f"🎯  VALUE BETS ({len(value_games)} game(s))\n")
        for r in value_games:
            print(f"  {r['game']}")
            print(f"  Moneyline: {r['away_odds']:+d} / {r['home_odds']:+d}")
            print(f"  Fair probs (vig-free): Away {r['away_fair_p']}% | Home {r['home_fair_p']}%")
            print(f"  Model blend:           Away {r['away_blend_p']}% | Home {r['home_blend_p']}%")
            if r["total_line"]:
                print(f"  Total: {r['total_line']}  (Over imp {r['over_imp']}% / Under imp {r['under_imp']}%)")
            for pick in r["value_picks"]:
                print(f"  ✅ BET: {pick['bet']}  {pick['odds']:+d}  edge={pick['edge']}%  "
                      f"Kelly={pick['kelly']}%  wager=${pick['wager']}")
            print()
    else:
        print("  No value bets found today.\n")

    if no_value:
        print(f"⛔  NO EDGE ({len(no_value)} game(s)): "
              + ", ".join(r["game"] for r in no_value))

    print(f"\n{'─'*60}")
    print("  ⚠️  For entertainment/research only. Gamble responsibly.")
    print(f"{'─'*60}\n")


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest() -> None:
    """
    Load saved daily prediction JSON files and compare against recorded results.
    Results are pulled from data/results_YYYYMMDD.json (populated by results_fetcher.py).
    """
    pred_dir = Path("predictions")
    files    = sorted(pred_dir.glob("predictions_*.json"))
    if not files:
        log.warning("No saved prediction files found. Run the predictor daily to build history.")
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
            actuals = json.load(fh)  # {game_str: winner_team}

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
                    dec_odds      = american_to_decimal(pick["odds"])
                    total_return += wager * dec_odds
                else:
                    losses       += 1

    if total_bets == 0:
        print("No completed bets to evaluate yet.")
        return

    roi = (total_return - total_wagered) / total_wagered * 100
    print(f"\n📊  BACKTEST RESULTS  ({len(files)} days)")
    print(f"  Bets: {total_bets}  |  W-L: {wins}-{losses}  |  Push: {push}")
    print(f"  Wagered: ${total_wagered:.2f}  |  Returned: ${total_return:.2f}")
    print(f"  ROI: {roi:+.1f}%\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB Bet Predictor")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on saved predictions")
    parser.add_argument("--output",   choices=["text", "json"], default="text")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
        return

    log.info("Fetching odds…")
    games   = fetch_odds()
    ratings = build_power_ratings()

    results = []
    for g in games:
        r = analyze_game(g, ratings)
        if r:
            results.append(r)

    if args.output == "json":
        payload = {"date": datetime.now().strftime("%Y%m%d"), "results": results}
        print(json.dumps(payload, indent=2))
    else:
        print_report(results)

    # Always save predictions for backtesting
    out_file = Path("predictions") / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out_file, "w") as f:
        json.dump({"date": datetime.now().strftime("%Y%m%d"), "results": results}, f, indent=2)
    log.info(f"Predictions saved → {out_file}")


if __name__ == "__main__":
    main()
