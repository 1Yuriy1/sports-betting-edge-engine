# Sports Betting / Conspiracy Model Code

[![CI](https://github.com/1Yuriy1/sports-betting-edge-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/1Yuriy1/sports-betting-edge-engine/actions/workflows/ci.yml)

This folder contains the standalone **Hermes Conspiracy Model V2** code we built for NFL moneyline slate analysis.

> **Entertainment only. This is not betting advice. Superstition/conspiracy terms are unvalidated. Do not bet blindly from this model.**

## Contents

```text
engine/odds.py                             # The Odds API client (h2h, retries, credit tracking)
engine/store.py                            # Append-only SQLite snapshot store (schema v1)
engine/ledger.py                           # Bet ledger: log, grade, CLV report
scripts/hermes_conspiracy_model.py         # Main model + CLI
scripts/snapshot_odds.py                   # Odds snapshot CLI (data spine)
scripts/bet_log.py                         # Bet-log CLI (log/grade/report)
.github/workflows/ci.yml                   # Lint + test pipeline
.github/workflows/snapshot.yml             # Scheduled odds snapshots
references/v2-market-edge-filter.md        # V2 market-edge workflow
references/conspiracy-model-skill.md       # Hermes skill documentation
examples/week2_fox_odds.csv                # Week 2 odds example slate
examples/sample_teams.csv                  # Example team-rating table
examples/run_slate.py                      # Helper to run the model across a slate CSV
```

## Requirements

The script uses Python with `pandas` and `numpy`. On this Mac, use `uv` so nothing has to be globally installed:

```bash
uv run --with pandas --with numpy scripts/hermes_conspiracy_model.py --help
```

## Development

CI (`.github/workflows/ci.yml`) lints and tests on Python 3.13 with [uv](https://docs.astral.sh/uv/). To run the same checks locally:

```bash
uv venv
uv pip install -r requirements-dev.txt
source .venv/bin/activate

ruff check scripts examples tests
pytest -q
```

## Quick test

```bash
cd /Users/yuriy/sports_betting_code
uv run --with pandas --with numpy scripts/hermes_conspiracy_model.py --write-samples examples/generated
```

## Run the example Week 2 slate

```bash
cd /Users/yuriy/sports_betting_code
uv run --with pandas --with numpy examples/run_slate.py \
  --teams examples/sample_teams.csv \
  --slate examples/week2_fox_odds.csv
```

## Single matchup with V2 odds filter

```bash
cd /Users/yuriy/sports_betting_code
uv run --with pandas --with numpy scripts/hermes_conspiracy_model.py \
  --teams examples/sample_teams.csv \
  --matchup DET BUF \
  --ml-a 170 \
  --ml-b -205 \
  --spread 4.5 \
  --week 2 \
  --json
```

## What V2 outputs

Look for:

```json
"v2_edge": {
  "model_pick": "BUF",
  "model_pick_prob": 0.72,
  "market_pick_prob": 0.64,
  "edge_pct": 8.0,
  "recommendation": "PLAY",
  "stake_units": 0.75,
  "trap_flags": ["public_chalk_tax"]
}
```

## Bet log, grading, and CLV

Bets live in the same SQLite file as the odds snapshots (`bets` table, schema
v1: `placed_at`, `game_id`, `market`, `pick`, `odds`, `stake`,
`sportsbook`, `closing_odds`, `status`, `profit`, `settled_at`). Log a bet
when you take it, log the closing price when the market closes, and grade
manually once the game is final — that is the v1 workflow.

```bash
uv run python scripts/bet_log.py log --game-id nfl-w2-det-buf --pick DET \
    --odds -150 --stake 100 --sportsbook pinnacle --closing-odds -160
uv run python scripts/bet_log.py grade --game-id nfl-w2-det-buf --outcome won
uv run python scripts/bet_log.py report --db ./odds.sqlite
```

Grading outcomes are `won`, `lost`, and `pushed`; profit follows the frozen
model's `bet_profit` semantics (won at +150 pays `stake * 1.5`, won at -150
pays `stake * 100 / 150`, a loss costs the stake, a push returns nothing).

The report prints per-bet and mean closing line value in percent:

```text
clv_pct = (decimal_odds_taken / decimal_odds_close - 1) * 100
```

Positive means you beat the close — the market moved toward your pick after
you bet. CLV is the honest early signal: it shows an edge in tens of bets,
long before profit proves anything. Pending bets are excluded from the mean
and reported by count (`bets_ungraded`), and graded bets without logged
closing odds are counted as `bets_missing_closing_odds`.

## Core idea

V2 is not meant to pick every game. It is meant to find:

```text
model probability - no-vig market probability = edge
```

Then classify:

```text
PASS_NO_EDGE
PASS_EXPENSIVE_FAVORITE
SMALL_LEAN
LEAN
PLAY
DOG_VALUE_PLAY
```

## Guardrail

Separate:

- **Facts:** odds, scores, spreads, injuries, weather, official data
- **Assumptions:** hand-built team ratings and weights
- **Parody:** conspiracy / superstition features

Do not treat this as a proven edge without a large held-out backtest.
