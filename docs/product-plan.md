# Betting Agent Product Plan

As of 2026-09-17. Living version: https://claude.ai/code/artifact/6eb6a4b3-7593-4180-92a3-c73a07fdcdc2

Turning the Hermes conspiracy-model repo into something with a buyer.

## Asset audit

The repo holds two assets glued together, and only one has a buyer.

| Asset | In the code | Verdict |
| --- | --- | --- |
| Moneyline math, no-vig conversion, edge classifier, stake ladder, backtest harness | `moneyline_to_implied`, `no_vig_pair`, `bet_profit`, `market_edge`, `backtest` | **Sellable.** This is a discipline engine, and discipline is what people pay for |
| Hand-tuned conspiracy and superstition weights | `score_side`, the feature registry | **Not sellable as prediction.** Valuable as entertainment IP and a marketing hook |

The README already says the second one is unvalidated and should lose to a plain Elo baseline. Selling picks off it means selling an edge that does not exist. That carries legal exposure (tout services are regulated in several US states) and guarantees churn the first time a slate goes 4-12.

So the conspiracy layer stays — as the product's voice, not its brain.

## The pivot that makes it a product

Swap one input and the existing pipeline starts producing real edges.

Today the code computes:

```text
hand-built model prob − no-vig market prob = edge
```

That edge is noise, because the first term is guesswork. Change the first term to a devigged price from a sharp book (Pinnacle, Circa):

```text
sharp-book no-vig prob − soft-book offered prob = edge
```

Same functions, same classifier, same staking. But the "model" is now the most accurate public probability estimate that exists, and the strategy becomes ordinary +EV betting rather than a superstition bet.

The proof method changes too. Instead of needing thousands of graded games to show ROI, you measure closing line value: did you beat the number the market closed at? CLV shows signal in a few hundred bets and is the standard the serious audience already trusts.

Two other changes come with it:

- Replace the 0.25 / 0.50 / 0.75 / 1.00 unit ladder with fractional Kelly sized off the actual edge and price.
- Keep the conspiracy probability as a separate, clearly labelled output field, never folded into the staking decision.

## Go to market

Start B2B: five customers is a business, and no one is buying picks from you.

| Shape | What you sell | Price | Reality |
| --- | --- | --- | --- |
| B2B / white-label | Slate engine + narrative segments for affiliate sites, podcasts, fantasy apps | $500–3,000 / mo per client | **Start here.** ~5 sales instead of ~1,000, zero pick liability, and "entertainment only" becomes a feature |
| B2C SaaS | +EV scanner, AI analyst chat, bet log, CLV report | $29–49 / mo | Real market, but you are against OddsJam, Unabated and Outlier, who have data budgets you do not |
| Content flywheel | Daily conspiracy slate → newsletter and short-form video | Affiliate revenue | Cheapest to start and genuinely shareable, but monetizes at scale, not at launch |

The sequencing matters. B2B pays the data bill and builds the engine. The content flywheel runs alongside it as distribution, using the conspiracy narrative as the hook. B2C becomes the upsell once you have a logged CLV track record worth pointing at.

A podcast buying a weekly conspiracy segment does not care whether the model beats Elo. That is the whole reason to sell there first.

## Agent architecture

One rule governs the design: the model never computes a number.

Every probability, edge, and stake comes from deterministic Python. The agent routes, explains and narrates. That keeps the output auditable, cheap, and defensible when a customer asks where a number came from.

### Tool surface

| Tool | Returns | Status |
| --- | --- | --- |
| `get_slate(league, week)` | Games, kickoff, venue | New, thin wrapper |
| `get_odds(game_id, markets, books)` | Per-book prices | New, the data dependency |
| `fair_price(ml_a, ml_b)` | No-vig pair | Already built |
| `sharp_anchor(game_id, market)` | Devigged sharp-book probability | New, and the core of the product |
| `find_edges(min_edge, books, bankroll)` | Ranked +EV list with Kelly stakes | New, assembled from the above |
| `line_move(game_id, market)` | Open to now, steam flags | New, needs stored snapshots |
| `conspiracy_score(team_a, team_b, ctx)` | Narrative probability and features | Already built |
| `narrate(game_id, tone)` | The entertainment layer | Already built |
| `log_bet`, `grade_bets`, `clv_report` | The user's history and their CLV | New, and the retention moat |

`clv_report` is what stops churn. Nobody cancels a tool that holds their entire betting history.

### The loop

Python, official Anthropic SDK, `client.beta.messages.tool_runner` with `@beta_tool` functions, so you write tools and the SDK drives the loop.

- Model `claude-opus-5` with `thinking: {"type": "adaptive"}`.
- `output_config: {"effort": "medium"}` for routine slate reports, `"high"` when a user asks it to reason about one spot.
- Prompt caching ordered tools → system → today's odds snapshot; the user's question and any timestamp go last so they do not invalidate the prefix.
- Structured outputs (`output_config.format`) for the report payload, so the frontend renders fields instead of parsing prose.
- Nightly precompute of every game, book and narrative through the Batch API at 50% cost.

### Cost

Opus 5 is $5 per MTok in, $25 out. A 16-game slate report at roughly 25K input and 4K output tokens costs about **$0.22**, less once caching lands. An interactive chat turn is one to three cents.

Model spend is a rounding error. Odds data is the real cost and the real competitive floor: $30–100 per month at the entry tier, four figures for low-latency multi-book feeds.

## First two weeks

Build the data spine before the agent, in this order.

1. Wire one odds API and persist hourly snapshots to SQLite. Line history cannot be backfilled, so this is day one or never.
2. Add `sharp_anchor()` and use it in place of the model probability inside `market_edge()`. Keep the conspiracy output as its own labelled field.
3. Replace the unit ladder with fractional Kelly.
4. Build `log_bet`, `grade_bets` and `clv_report`. This is the product, not the picks.
5. Only then wrap the `tool_runner` agent around it and give it the Hermes voice.

Steps 1–4 are worth selling without any agent at all. Step 5 is what makes it demo well.

## What will kill it

Three things, in order of how likely they are to end the project.

**Data cost and latency.** Soft-book +EV lines get taken within seconds. A loop running five minutes behind the market is finding edges that no longer exist. This is the single most common reason these products fail, and the fix costs money rather than code.

**Regulatory surface.** You need 21+ gating, state-by-state affiliate licensing where you take referral revenue, and responsible-gambling disclosures. App stores are hostile to gambling-adjacent apps. Going B2B and web-first avoids most of this.

**Unproven edge.** You cannot claim an edge until CLV is logged across a few hundred bets. Until then, sell the tooling, the discipline and the narrative — never a guarantee. The existing disclaimer is not a legal formality, it is the honest position, and it should stay in the product until the data replaces it.

### Open question

Which odds provider to start with. That choice sets both the monthly floor and the latency ceiling, so it is worth deciding before step 1.
