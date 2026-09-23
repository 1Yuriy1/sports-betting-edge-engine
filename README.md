# The Discipline Engine

[![CI](https://github.com/1Yuriy1/sports-betting-edge-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/1Yuriy1/sports-betting-edge-engine/actions/workflows/ci.yml)

**Auditable sports-betting discipline — deterministic staking, closing-line feedback, human-in-the-loop. Not picks, not guarantees, not a betting bot.**

Betting apps have largely solved the *what to bet* problem: research, stats, market exploration, bet-building, all in the app. Almost nothing solves the *how* problem — bankroll discipline, stake sizing, journaling, and honest feedback about whether you are actually beating the closing line. This repo is a working prototype of that missing layer, built to the standard a regulated operator would require.

> **The model never computes a number.** Every probability, edge, stake, and grade comes from deterministic, test-covered Python. A conversational agent sits on top to route, explain, and narrate — it has no tool that produces a figure of its own, and no authority over stakes. A human places every bet.

## The concept

Betting products compete on *what* to bet. The unsolved problem — and the one responsible-gaming programs and regulators actually care about — is *how*:

- **Stake sizing a customer can audit, line by line.** Quarter Kelly by default, hard-capped at 5% of bankroll, zero stake whenever the edge is non-positive. Discipline lives in the math, not a settings toggle.
- **Honest feedback.** The ledger grades every bet against the closing price, so you learn whether you beat the market — the standard serious bettors already trust. CLV shows signal in a few hundred logged bets; proving ROI from raw profit needs thousands of graded games.
- **A narrating agent, not a betting agent.** The conversational layer (an Anthropic tool runner) wraps eleven deterministic tools and re-applies mandatory disclaimers to every output. It can research, explain, and summarize. It cannot size a stake or place a bet.
- **Operator-shaped.** The architecture is built so a sportsbook's own assistant could adopt the layer wholesale — discipline guidance rather than suggested wagers, which is the category regulators are drawing lines around.

The repo began as the **Hermes Conspiracy Model**, a parody win-probability engine for NFL slates. Its narrative layer survives as a clearly labelled `narrative` field on edge rows and as the agent's voice — display only, never an input to staking.

## Contents

```text
engine/odds.py                             # The Odds API client (h2h, retries, credit tracking)
engine/store.py                            # Append-only SQLite snapshot store (schema v1)
engine/pricing.py                          # Moneyline-to-implied and no-vig pair helpers
engine/anchor.py                           # Sharp-book anchoring, edges, line movement
engine/kelly.py                            # Fractional Kelly staking
engine/ledger.py                           # Bet ledger: log, grade, CLV report
engine/agent.py                            # Hermes agent: tool runner over the engine
scripts/hermes_conspiracy_model.py         # Main model + CLI
scripts/snapshot_odds.py                   # Odds snapshot CLI (data spine)
scripts/bet_log.py                         # Bet-log CLI (log/grade/report)
scripts/agent_slate.py                     # One agent turn -> structured slate report
scripts/agent_chat.py                      # Interactive one-spot reasoning with the agent
.github/workflows/ci.yml                   # Lint + test pipeline
.github/workflows/snapshot.yml             # Scheduled odds snapshots
references/v2-market-edge-filter.md        # V2 market-edge workflow
references/conspiracy-model-skill.md       # Hermes skill documentation
examples/week2_fox_odds.csv                # Week 2 odds example slate
examples/sample_teams.csv                  # Example team-rating table
examples/run_slate.py                      # Helper to run the model across a slate CSV
```

## Requirements

Everything runs through [uv](https://docs.astral.sh/uv/) so nothing has to be installed globally. All commands below run from the repository root:

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
uv run --with pandas --with numpy scripts/hermes_conspiracy_model.py --write-samples examples/generated
```

## Run the example Week 2 slate

```bash
uv run --with pandas --with numpy examples/run_slate.py \
  --teams examples/sample_teams.csv \
  --slate examples/week2_fox_odds.csv
```

## Single matchup with V2 odds filter

```bash
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

## Engine core: snapshots, sharp anchor, and staking

The `engine/` package is the deterministic core — every number the
product relies on is computed there, everything runs offline from the
SQLite snapshot store, and the conspiracy model is never an input to
staking.

### Odds snapshots (The Odds API)

`engine/odds.py` wraps The Odds API (the provider locked in the spec):
h2h moneylines, retry/backoff on 429/5xx, and credit accounting from the
response headers — the free tier is roughly 500 credits/month, so every
read of remaining credits matters. The snapshot CLI fetches and appends
one snapshot; the cadence belongs to your scheduler (see
`.github/workflows/snapshot.yml`), not the tool:

```bash
uv run python scripts/snapshot_odds.py --sport americanfootball_nfl
```

| Env var | Default | Meaning |
|---|---|---|
| `ODDS_API_KEY` | — | The Odds API key (required; never hardcoded, never in tests) |
| `ODDS_DB_PATH` | `./odds.sqlite` | SQLite file for odds snapshots and bets |
| `SNAPSHOT_INTERVAL_HOURS` | `12` | Minimum cadence the scheduler should allow on the free tier |

Snapshots are append-only and idempotent: running the same fetch twice
appends zero duplicate rows. `--help` shows per-sport and per-book
filtering.

### Sharp anchor, edges, and fractional Kelly

`engine/anchor.py` anchors every price on the sharp books (Pinnacle and
Circa by default): the no-vig pair is devigged per book and averaged
across sharp books quoting both sides. If no sharp book quotes both
sides, the anchor is unavailable — soft books are never substituted.
Edges are `sharp_prob − implied(best soft price)`, ranked, and sized by
`engine/kelly.py` (quarter Kelly by default, capped at 5% of bankroll,
0 whenever the edge is non-positive — a pass). Line movement reports
open→now movement in seam-aware cents (a one-tick move across ±100 is
10 cents, not 210) with a `steam` flag at 10 cents:

```python
from engine.anchor import find_edges, line_move, sharp_anchor
from engine.kelly import kelly_stake
from engine.odds import OddsClient
from engine.store import OddsStore

store = OddsStore("./odds.sqlite")
client = OddsClient()  # reads ODDS_API_KEY; inject a transport in tests
anchor = sharp_anchor(store, "det-buf-2026-09-21")
rows = find_edges(store, client, min_edge=0.02, bankroll=1000.0)
moves = line_move(store, "det-buf-2026-09-21")
```

The conspiracy model's output rides along only as the clearly labelled
`narrative` field on each edge row — display, never a staking input.

> **Live-key gate:** a real `ODDS_API_KEY` fetch plus a credit-header
> check is a manual verification step before upgrading from the free
> tier. CI and the test suite never touch the live API.

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

## Agent: Hermes over the engine

`engine/agent.py` is the conversational layer: an Anthropic tool runner
whose eleven tools (`get_slate`, `get_odds`, `fair_price`, `sharp_anchor`,
`find_edges`, `line_move`, `conspiracy_score`, `narrate`, `log_bet`,
`grade_bets`, `clv_report`) are thin wrappers over the deterministic engine
functions above. The model never computes a number — every figure it cites
comes from a tool result, and the conspiracy layer stays narration-only,
never an input to stakes.

Two entry points:

```bash
# One turn over the current slate -> structured report (medium effort)
uv run python scripts/agent_slate.py --db ./odds.sqlite

# Interactive reasoning about one spot (high effort)
uv run python scripts/agent_chat.py --db ./odds.sqlite
```

### Setup

The agent needs an Anthropic API key from the environment — it is never
hardcoded and never used in tests:

| Env var | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required; export it before running) |
| `AGENT_MODEL` | `claude-opus-5` | Model id used by the tool runner |
| `ODDS_DB_PATH` | `./odds.sqlite` | Same SQLite file as the data spine |
| `HERMES_TEAMS_CSV` | `examples/sample_teams.csv` | Team ratings for the narrative layer |

The system prompt carries the Hermes voice plus hard guardrails: 21+ only,
entertainment framing, no guarantees, and the mandatory disclaimer is
re-applied to every output (structured or chat) even if the model omits
it. Slate reports at `medium` effort; one-spot reasoning runs at `high`.

> **Live-key gate:** like the odds client, a real `ANTHROPIC_API_KEY` call
> is a manual verification step. CI and the test suite stay key-free — the
> agent tests stub the SDK surface and execute the real tool objects
> offline against a seeded temp store.

## Core idea

The product path does not predict. It anchors:

```text
sharp-book no-vig probability - implied probability of the best soft price = edge
```

Then sizes the edge by fractional Kelly (capped, zero on a non-positive edge), logs the bet, and grades it against the close. The legacy conspiracy path keeps its original identity — `model probability - no-vig market probability = edge` — and its classifier (`PASS_NO_EDGE`, `PASS_EXPENSIVE_FAVORITE`, `SMALL_LEAN`, `LEAN`, `PLAY`, `DOG_VALUE_PLAY`), frozen as entertainment.

## What this is not

- **Not affiliated.** A personal prototype — not affiliated with, endorsed by, or reviewed by any sportsbook or operator.
- **Not a betting bot.** It cannot place bets and does not connect to any sportsbook account. Execution is always manual.
- **Not a pick seller.** No guarantees and no claim of a proven edge. The only success metric this repo tracks is CLV against the closing line, and validating that takes hundreds of logged bets.
- **Not betting advice.** 21+ only.

## Guardrail

Separate:

- **Facts:** odds, scores, spreads, injuries, weather, official data
- **Assumptions:** sharp-book prices, hand-tuned narrative weights
- **Parody:** conspiracy / superstition features

## Responsible gambling

Betting should be entertainment, not income. The staking caps are a ceiling, not a target. If gambling stops being fun, call **1-800-GAMBLER** (US) or your local helpline.
