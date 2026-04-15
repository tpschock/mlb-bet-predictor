# ⚾ MLB Bet Predictor

A **daily automated betting algorithm** that pulls live DraftKings odds, applies a power-rating model, strips the vig, and surfaces value bets — all running for free on GitHub Actions.

> ⚠️ For research and entertainment only. Gamble responsibly.

---

## How It Works

```
GitHub Actions (daily noon ET)
        │
        ▼
update_ratings.py        ← MLB Stats API → power_ratings.json
        │
        ▼
predictor.py             ← The Odds API (DraftKings) → analyze each game
        │
        ├── Strip the vig from moneyline odds
        ├── Blend market probability (60%) with our model (40%)
        ├── Calculate edge = blend_prob − fair_market_prob
        ├── Flag bets with edge ≥ 3%
        └── Size with fractional Kelly criterion
        │
        ▼
predictions/predictions_YYYYMMDD.json  (auto-committed)
```

---

## Setup

### 1. Fork & clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/mlb-bet-predictor.git
cd mlb-bet-predictor
pip install -r requirements.txt
```

### 2. Get a free Odds API key

Sign up at **https://the-odds-api.com** — free tier gives 500 requests/month (plenty for daily use).

### 3. Add your key as a GitHub Secret

`Settings → Secrets and variables → Actions → New repository secret`

- Name: `ODDS_API_KEY`
- Value: your key from step 2

### 4. Enable GitHub Actions

The workflow runs automatically every day at noon ET.  
You can also trigger it manually via **Actions → MLB Daily Bet Predictions → Run workflow**.

---

## Running Locally

```bash
# Update power ratings first
python update_ratings.py

# Run predictions (uses sample data if no API key)
python predictor.py

# Output as JSON
python predictor.py --output json

# Backtest against saved results
python predictor.py --backtest
```

Set your API key locally:
```bash
export ODDS_API_KEY="your_key_here"
python predictor.py
```

---

## Files

| File | Purpose |
|---|---|
| `predictor.py` | Main engine — fetches odds, runs model, outputs picks |
| `update_ratings.py` | Pulls MLB standings, computes team power ratings |
| `requirements.txt` | Python dependencies |
| `.github/workflows/daily_predictions.yml` | GitHub Actions cron job |
| `data/power_ratings.json` | Auto-updated daily team ratings (0–100 scale) |
| `predictions/predictions_YYYYMMDD.json` | Daily pick output, committed by bot |
| `data/results_YYYYMMDD.json` | *(You populate)* Actual game results for backtesting |

---

## The Model

### Power Ratings (`update_ratings.py`)
Built from **run differential per game** (season-to-date), with a confidence ramp over the first 20 games to handle small early-season samples. Ratings are on a 0–100 scale, centered at 50.

### Prediction Blend (`predictor.py`)
```
blended_prob = 0.60 × market_fair_prob + 0.40 × power_rating_prob
```
- **60% market** — respects the wisdom of the crowd
- **40% model** — adds signal from run differential
- **Home field** — a 4% multiplicative boost to home team probability

### Vig Removal
```
fair_home = implied_home / (implied_home + implied_away)
```
Strips the bookmaker's juice before comparing to our model.

### Kelly Criterion
```
kelly_fraction = (b × p − q) / b   × 0.25 (fractional)
```
Using 25% Kelly (conservative) on a hypothetical $1,000 bankroll.

---

## Backtesting

Populate `data/results_YYYYMMDD.json` with game outcomes:
```json
{
  "Boston Red Sox @ Minnesota Twins": "Minnesota Twins",
  "Colorado Rockies @ Houston Astros": "Houston Astros"
}
```
Then run:
```bash
python predictor.py --backtest
```

---

## Improving the Model

Ideas to increase predictive accuracy:

- **Starting pitcher ERA/FIP** — biggest single-game factor in MLB
- **Bullpen fatigue** — days of rest, recent usage
- **Park factors** — Coors Field, etc.
- **Weather** — wind speed/direction, temperature
- **Lineup data** — injuries, day-of lineup scraping
- **Line movement** — track opening vs. closing odds

Pull requests welcome!

---

## License

MIT — use freely, no warranty expressed or implied.
