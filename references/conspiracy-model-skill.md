---
name: conspiracy-model
description: Use when Yuriy asks for a playful sports matchup “conspiracy model,” football win-probability narrative, superstition-weighted prediction, feature registry, slate ranking, or backtest against an Elo baseline. Entertainment-only; always include the no-betting disclaimer and separate real signal from parody signal.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [sports, football, prediction, parody, betting-guardrail, elo, model]
    related_skills: []
---

# Conspiracy Model

## Overview

This skill adds a reusable **entertainment-only football matchup model**: real football signal plus intentionally silly “conspiracy”/superstition features blended into a logistic matchup model.

Use it as a talker and narrative generator, not as betting advice. The script explicitly backtests against a plain Elo baseline and should usually lose to Elo if the superstition terms are noise.

## When to Use

Use when the user asks for:

- a humorous “scripted league” / “conspiracy” prediction
- NFL/football matchup probabilities with vibes and superstition
- feature breakdowns like ref bias, full moon, Madden curse, primetime narrative, etc.
- an all-matchup slate ranking
- an honesty check/backtest against results
- machine-readable output for another agent

Do **not** use as a real betting edge. If the user asks for betting guidance, present this as entertainment only and prefer grounded stats, odds, injury reports, line movement, and risk management.

## Script

Linked script:

```bash
~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py
```

Because the model uses `pandas` and `numpy`, run it with `uv` so dependencies do not need to be globally installed:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py --help
```

## Quick Start

For slate reviews against odds/results, use `references/v2-market-edge-filter.md` for the recommended V2 workflow: moneyline implied probability, model-vs-market edge, trap flags, play/lean/pass, and stake units.

Create sample files in the current directory:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py --write-samples
```

Run one matchup:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py \
  --teams teams.csv \
  --matchup KC BUF \
  --game sample_game.json
```

Machine-readable output:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py \
  --teams teams.csv \
  --matchup KC BUF \
  --json
```

All pairings, sorted by confidence:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py \
  --teams teams.csv \
  --slate
```

List feature registry and weights:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py \
  --list-features
```

Backtest:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py \
  --teams teams.csv \
  --backtest results.csv
```

V2 market-edge filter for one matchup:

```bash
uv run --with pandas --with numpy ~/.hermes/skills/gaming/conspiracy-model/scripts/hermes_conspiracy_model.py \
  --teams teams.csv \
  --matchup NYJ TEN \
  --game game.json \
  --ml-a 105 \
  --ml-b -125 \
  --spread 1.5 \
  --week 1 \
  --json
```

If `results.csv` includes `ml_a`, `ml_b`, optional `spread`, and optional `week`, `--backtest` also reports `v2_market_filter` with flat $100 profit, recommended staking profit, play count, and play record.

## Agent Entry Point

```python
from hermes_conspiracy_model import ConspiracyModel

m = ConspiracyModel.from_csv("teams.csv")
out = m.predict("KC", "BUF", game_context={"shared": {"full_moon": 1}})
print(out["prob_a"], out["narrative"])
```

## Data Format

`teams.csv` needs a `team` column. Optional columns include:

- real-ish signal: `elo`, `point_diff`, `star_qb`
- league-conspiracy tier: `popularity`, `ref_bias`, `farewell_tour`
- curses: `madden_cover`, `si_cover`, `sb_hangover`, `heisman_qb`, `qb_new_relationship`
- motivation/vibes: `contract_year`, `coach_hot_seat`, `owner_controversy`, `color_rush_record`

Game context JSON shape:

```json
{
  "a": {"primetime": 1, "short_week_road": 1},
  "b": {"home": 1, "cold_weather_edge": 1},
  "shared": {"full_moon": 1, "wind_over_15mph": 1}
}
```

- `a` applies to `TEAM_A`
- `b` applies to `TEAM_B`
- `shared` applies to both or to chaos terms

## Guardrails

Always include:

> Entertainment only. Superstition terms are unvalidated. Do not bet.

Cognitive surrender guardrail:

- Facts: Elo, point differential, home/road, weather/travel signals may be partly grounded.
- Assumptions: weights are hand-built and not calibrated unless backtested.
- Parody: curses, moon, relationship, narrative gravity, and most conspiracy terms are for entertainment.
- Opposing view: a plain Elo baseline is usually more defensible.

## Common Pitfalls

1. **Treating the probability as real betting advice.** It is not. Use the output as entertainment and narrative.
2. **Overfitting by vibes.** Tune only with held-out backtests, not by making the last game “look right.”
3. **Double-counting mirrored penalties.** Opponent mirror features are off by default because the head-to-head difference already gives the other side the relative gain.
4. **Using two-team min-max scaling.** The script fits normalization on the whole team table first, then reuses it for matchups.
5. **Forgetting shared chaos terms.** Full moon/wind/snow compress the favorite’s edge; they should not help one team directly.

## Verification Checklist

- [ ] `--write-samples` creates `teams.csv`, `sample_game.json`, and `results.csv`
- [ ] `--matchup KC BUF` prints a narrative and disclaimer
- [ ] `--json` returns structured probabilities and breakdowns
- [ ] `--list-features` prints the registry
- [ ] `--backtest results.csv` compares model vs Elo baseline
- [ ] `--matchup ... --ml-a ... --ml-b ... --json` returns `market` and `v2_edge`
- [ ] Backtests with `ml_a`/`ml_b` return `v2_market_filter`
- [ ] V2 reports passes on expensive favorites and highlights true market fades
