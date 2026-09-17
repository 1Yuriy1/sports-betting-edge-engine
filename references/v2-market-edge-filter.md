# Conspiracy Model V2: Market Edge Filter

## Purpose

V2 turns the parody win-probability engine into a more disciplined slate reviewer:

> Do not pick every game. Compare the model to the market, flag real disagreement, and pass on thin/expensive chalk.

It remains **entertainment only**. The market-edge filter is for analysis and logging, not betting advice.

## Core Additions

1. **Moneyline implied probability**
   - Negative odds: `abs(odds) / (abs(odds) + 100)`
   - Positive odds: `100 / (odds + 100)`
   - V2 normalizes both sides to no-vig probabilities.

2. **Edge**
   - `edge = model_pick_probability - no_vig_market_probability`
   - Positive edge means the model thinks its picked side is more likely than the market says.

3. **Recommendations**
   - `PASS_NO_EDGE`: edge below 2 points
   - `PASS_EXPENSIVE_FAVORITE`: favorite at `-250` or worse without 12+ point edge
   - `SMALL_LEAN`: 2-5 point edge
   - `LEAN`: 5-7 point edge
   - `PLAY`: 7+ point edge on a favorite or non-dog value
   - `DOG_VALUE_PLAY`: 7+ point edge on an underdog/market fade

4. **Stake units**
   - `0.00`: pass
   - `0.25`: tiny lean
   - `0.50`: lean
   - `0.75`: stronger edge
   - `1.00`: 12+ point edge

5. **Trap flags**
   - `expensive_favorite`
   - `road_favorite`
   - `big_spread_favorite`
   - `divisional_chaos`
   - `week1_uncertainty`
   - `public_chalk_tax`

## Single Matchup Command

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py \
  --teams teams.csv \
  --matchup NYJ TEN \
  --game jets_game.json \
  --ml-a 105 \
  --ml-b -125 \
  --spread 1.5 \
  --week 1
```

Machine-readable:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py \
  --teams teams.csv \
  --matchup NYJ TEN \
  --game jets_game.json \
  --ml-a 105 \
  --ml-b -125 \
  --spread 1.5 \
  --json
```

Look for:

```json
"v2_edge": {
  "model_pick": "NYJ",
  "model_pick_prob": 0.64,
  "market_pick_prob": 0.49,
  "edge_pct": 15.0,
  "recommendation": "DOG_VALUE_PLAY",
  "stake_units": 1.0,
  "trap_flags": ["week1_uncertainty"]
}
```

## Backtest With Moneylines

If `results.csv` includes `ml_a`, `ml_b`, optional `spread`, and optional `week`, `--backtest` now reports V2 market-filter performance.

Required columns:

```csv
team_a,team_b,a_won
```

Optional V2 columns:

```csv
ml_a,ml_b,spread,week
```

Context columns still work:

```csv
ctx_a_primetime,ctx_b_home,ctx_divisional_game
```

Run:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py \
  --teams teams.csv \
  --backtest results.csv
```

Backtest output includes:

- `flat_100_profit_if_all_model_picks`
- `recommended_staked`
- `recommended_profit`
- `recommended_roi_pct`
- `plays`
- `play_record`
- per-game rows with recommendation and profit

## Analysis Rules for Hermes

When Yuriy asks for a slate:

1. Pull current games, moneylines, spreads, and results if grading.
2. Run V2 with moneylines whenever possible.
3. Report **straight-up record** separately from **V2 recommended plays**.
4. Highlight market fades, especially `DOG_VALUE_PLAY`.
5. Call out passes on expensive favorites.
6. Separate facts from assumptions:
   - Facts: odds, scores, line movement, injuries if fetched.
   - Assumptions: team rating table, hand-tuned weights.
   - Parody: superstition/conspiracy features.
7. Include the disclaimer.

## Interpretation

V2 should produce fewer plays. That is the point.

Good output style:

| Game | Model pick | Market prob | Model prob | Edge | V2 | Traps |
|---|---:|---:|---:|---:|---|---|
| NYJ @ TEN | NYJ | 48.8% | 64.0% | +15.2 | DOG_VALUE_PLAY | week1_uncertainty |
| ARI @ LAC | LAC | 83.3% | 91.8% | +8.5 | PASS_EXPENSIVE_FAVORITE | expensive_favorite, big_spread |

## Guardrail

Even with V2:

> Entertainment only. Superstition terms are unvalidated. Do not bet.

Do not claim the model has a real edge without a large held-out backtest.
