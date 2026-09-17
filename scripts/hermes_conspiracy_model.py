#!/usr/bin/env python3
"""
hermes_conspiracy_model.py

A parody win-probability engine: real football signal + a pile of superstition
and conspiracy “features,” blended into a logistic matchup model.

FOR ENTERTAINMENT ONLY. The superstition terms have no demonstrated predictive
power. Run --backtest and you should watch this model lose to a plain Elo
baseline. If it ever beats Elo by a lot, you overfit. Do not bet money on this.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGIT_SCALE = 3.0
HOME_FIELD = 0.035
MAX_CHAOS = 0.45

# V2 market-edge filter defaults. These intentionally make the model more selective:
# it should find disagreement with price, not pick every game.
EDGE_PASS = 0.02
EDGE_LEAN = 0.05
EDGE_PLAY = 0.07
EDGE_BIG_PLAY = 0.12
EXPENSIVE_FAVORITE_CUTOFF = -250
BIG_SPREAD = 7.5


@dataclass(frozen=True)
class Feature:
    key: str
    weight: float
    kind: str
    scale: str
    label: str
    rationale: str
    default: float = 0.0


REGISTRY: Tuple[Feature, ...] = (
    Feature("elo", 0.45, "team", "z", "Team strength (Elo)", "Good teams win. This is the only term carrying real weight."),
    Feature("point_diff", 0.12, "team", "z", "Season point differential", "Cheap proxy for true strength; correlated with Elo on purpose."),
    Feature("star_qb", 0.10, "team", "minmax", "Star quarterback", "Partly real, partly ‘the script favors stars.’"),
    Feature("popularity", 0.15, "team", "minmax", "Popularity / market size", "The league allegedly 'wants' the big brand to advance."),
    Feature("ref_bias", 0.05, "team", "z", "Ref crew flag lean", "Crew-level penalty differential. Real but tiny and unstable."),
    Feature("primetime", 0.05, "game", "raw", "National primetime game", "Popular teams supposedly get the calls under the lights."),
    Feature("farewell_tour", 0.05, "team", "raw", "Farewell tour narrative", "League reportedly enjoys a happy ending."),
    Feature("ref_from_opp_state", -0.03, "game", "raw", "Ref crew from opponent's state", "Conspiracy classic. Zero evidence."),
    Feature("sharp_money_on", 0.04, "game", "raw", "Sharp line move toward this side", "'They know something.' The one entry here with a real track record."),
    Feature("public_fade", 0.03, "game", "raw", "Fading heavy public money", "Contrarian tax on the crowd's favorite."),
    Feature("madden_cover", -0.10, "team", "raw", "Madden cover curse", "Cover athlete gets hurt. Survivorship bias, but load-bearing lore."),
    Feature("si_cover", -0.05, "team", "raw", "SI cover jinx", "Regression to the mean wearing a costume."),
    Feature("sb_hangover", -0.05, "team", "raw", "Super Bowl hangover", "Short offseason, long emotional debt."),
    Feature("heisman_qb", -0.04, "team", "raw", "Heisman winner under center", "The curse. Mostly just rookie QBs being rookies."),
    Feature("announcer_jinx", -0.03, "game", "raw", "Broadcast booth jinx", "Romo praises the kicker, kicker shanks it."),
    Feature("qb_new_relationship", -0.07, "team", "raw", "QB in a new relationship", "Tabloid causality. See note on mirrored features."),
    Feature("contract_year", 0.03, "team", "raw", "Contract-year motivation", "Play for the bag."),
    Feature("coach_hot_seat", -0.03, "team", "raw", "Coach on the hot seat", "Coin flip between rally and quit; modeled as mild drag."),
    Feature("owner_controversy", -0.04, "team", "raw", "Owner in the headlines", "Distraction penalty."),
    Feature("backup_qb_chaos", 0.02, "game", "raw", "Backup QB, nothing to lose", "Nobody has tape on him."),
    Feature("halftime_hometown", 0.02, "game", "raw", "Halftime act from this city", "Narrative gravity."),
    Feature("short_week_road", -0.04, "game", "raw", "Thursday road game / short week", "Fatigue. One of the few situational terms with support."),
    Feature("timezones_crossed", -0.015, "game", "raw", "Time zones crossed", "Scaled by count, not binary. Body clock tax."),
    Feature("early_window_west_coast", -0.03, "game", "raw", "West coast team at 1pm ET", "10am body clock. Modest real effect."),
    Feature("london_game", -0.03, "game", "raw", "London / international game", "Jet lag and strange kicking."),
    Feature("dome_team_outdoors_bad", -0.03, "game", "raw", "Dome team outdoors in weather", "Climate-controlled team meets actual weather."),
    Feature("cold_weather_edge", 0.03, "game", "raw", "Cold-weather team in a snow game", "Home in the elements."),
    Feature("turf_speed_edge", 0.02, "game", "raw", "Speed team on turf", "Faster surface, faster team."),
    Feature("color_rush_record", 0.02, "team", "raw", "Undefeated in color rush", "Dumb luck formalized."),
    Feature("coin_toss_winner", 0.01, "game", "raw", "Won the coin toss", "Literally 50/50. Included because people swear by it."),
    Feature("full_moon", 0.02, "chaos", "raw", "Full moon", "Chaos multiplier: compresses the spread toward a coin flip."),
    Feature("mercury_retrograde", 0.02, "chaos", "raw", "Mercury retrograde", "Joke penalty on certainty itself."),
    Feature("wind_over_15mph", 0.05, "chaos", "raw", "Wind above 15 mph", "Kicking chaos, shortened playbooks, variance up. Real."),
    Feature("snow_game", 0.03, "chaos", "raw", "Snow game", "Ground game, fewer possessions, wider outcome spread."),
    Feature("divisional_game", 0.03, "chaos", "raw", "Divisional matchup", "Familiarity compresses talent gaps. Mildly real."),
)

BY_KEY: Dict[str, Feature] = {f.key: f for f in REGISTRY}
MIRROR_FEATURES = ("qb_new_relationship", "madden_cover", "si_cover")


@dataclass
class LeagueStats:
    """Normalization parameters fit on the full league table."""

    minmax: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    zparams: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "LeagueStats":
        stats = cls()
        for feat in REGISTRY:
            if feat.kind != "team" or feat.key not in df.columns:
                continue
            col = pd.to_numeric(df[feat.key], errors="coerce").dropna()
            if col.empty:
                continue
            if feat.scale == "minmax":
                lo, hi = float(col.min()), float(col.max())
                stats.minmax[feat.key] = (lo, hi if hi > lo else lo + 1.0)
            elif feat.scale == "z":
                mu, sd = float(col.mean()), float(col.std(ddof=0))
                stats.zparams[feat.key] = (mu, sd if sd > 1e-9 else 1.0)
        return stats

    def apply(self, key: str, value: float) -> float:
        feat = BY_KEY[key]
        if feat.scale == "minmax" and key in self.minmax:
            lo, hi = self.minmax[key]
            return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
        if feat.scale == "z" and key in self.zparams:
            mu, sd = self.zparams[key]
            return float(np.tanh((value - mu) / (2 * sd)))
        return float(value)


@dataclass
class Contribution:
    key: str
    label: str
    raw: float
    scaled: float
    weight: float
    points: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "raw": round(self.raw, 4),
            "scaled": round(self.scaled, 4),
            "weight": self.weight,
            "points": round(self.points, 4),
        }


def moneyline_to_implied(odds: float) -> float:
    """Convert American odds to raw implied probability, including vig."""
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def no_vig_pair(ml_a: float, ml_b: float) -> Tuple[float, float]:
    """Return no-vig implied probabilities for a two-sided moneyline market."""
    pa = moneyline_to_implied(ml_a)
    pb = moneyline_to_implied(ml_b)
    s = pa + pb
    if s <= 0:
        return 0.5, 0.5
    return pa / s, pb / s


def bet_profit(stake: float, odds: float, won: bool) -> float:
    """Net profit for a stake at American odds."""
    if not won:
        return -stake
    if odds > 0:
        return stake * odds / 100.0
    return stake * 100.0 / abs(odds)


def _stake_units(edge: float, recommendation: str) -> float:
    if recommendation.startswith("PASS"):
        return 0.0
    if edge >= EDGE_BIG_PLAY:
        return 1.0
    if edge >= 0.08:
        return 0.75
    if edge >= EDGE_LEAN:
        return 0.50
    if edge >= EDGE_PASS:
        return 0.25
    return 0.0


class ConspiracyModel:
    def __init__(
        self,
        teams: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        logit_scale: float = LOGIT_SCALE,
        use_mirrors: bool = False,
    ) -> None:
        if "team" not in teams.columns:
            raise ValueError("teams table needs a 'team' column")
        teams = teams.copy()
        teams["team"] = teams["team"].astype(str).str.strip().str.upper()
        self.teams = teams.set_index("team", drop=False)
        self.league = LeagueStats.fit(teams)
        self.weights = {f.key: f.weight for f in REGISTRY}
        if weights:
            unknown = set(weights) - set(self.weights)
            if unknown:
                raise ValueError(f"unknown feature(s) in weight override: {sorted(unknown)}")
            self.weights.update(weights)
        self.logit_scale = logit_scale
        self.use_mirrors = use_mirrors

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: Any) -> "ConspiracyModel":
        return cls(pd.read_csv(path), **kwargs)

    def _team_value(self, team: str, key: str) -> float:
        row = self.teams.loc[team]
        if key not in self.teams.columns:
            return BY_KEY[key].default
        val = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
        return BY_KEY[key].default if pd.isna(val) else float(val)

    def score_side(
        self,
        team: str,
        side_ctx: Dict[str, Any],
        opponent: Optional[str] = None,
    ) -> Tuple[float, List[Contribution]]:
        team = team.upper()
        if team not in self.teams.index:
            raise KeyError(f"unknown team '{team}'. known: {sorted(self.teams.index)}")

        contribs: List[Contribution] = []
        total = 0.0

        for feat in REGISTRY:
            if feat.kind == "chaos":
                continue
            if feat.kind == "team":
                raw = self._team_value(team, feat.key)
            else:
                raw = float(side_ctx.get(feat.key, feat.default) or 0.0)
            if raw == 0.0 and feat.scale == "raw":
                continue
            scaled = self.league.apply(feat.key, raw) if feat.scale != "raw" else raw
            w = self.weights[feat.key]
            pts = scaled * w
            total += pts
            contribs.append(Contribution(feat.key, feat.label, raw, scaled, w, pts))

        if self.use_mirrors and opponent is not None:
            opponent = opponent.upper()
            for key in MIRROR_FEATURES:
                raw = self._team_value(opponent, key)
                if raw == 0.0:
                    continue
                w = -self.weights[key]
                pts = raw * w
                total += pts
                contribs.append(Contribution(f"opp_{key}", f"Opponent: {BY_KEY[key].label}", raw, raw, w, pts))

        if side_ctx.get("home"):
            total += HOME_FIELD
            contribs.append(Contribution("home", "Home field", 1.0, 1.0, HOME_FIELD, HOME_FIELD))

        contribs.sort(key=lambda c: abs(c.points), reverse=True)
        return total, contribs

    def chaos_factor(self, shared: Dict[str, Any]) -> Tuple[float, List[Contribution]]:
        contribs: List[Contribution] = []
        total = 0.0
        for feat in REGISTRY:
            if feat.kind != "chaos":
                continue
            raw = float(shared.get(feat.key, feat.default) or 0.0)
            if raw == 0.0:
                continue
            w = self.weights[feat.key]
            total += raw * w
            contribs.append(Contribution(feat.key, feat.label, raw, raw, w, raw * w))
        total = min(total, MAX_CHAOS)
        contribs.sort(key=lambda c: abs(c.points), reverse=True)
        return 1.0 - total, contribs

    def predict(
        self,
        team_a: str,
        team_b: str,
        game_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        team_a, team_b = team_a.upper(), team_b.upper()
        ctx = game_context or {}
        ctx_a = dict(ctx.get("a", {}))
        ctx_b = dict(ctx.get("b", {}))
        shared = dict(ctx.get("shared", {}))

        for key, val in shared.items():
            if key in BY_KEY and BY_KEY[key].kind == "game":
                ctx_a.setdefault(key, val)
                ctx_b.setdefault(key, val)

        score_a, cont_a = self.score_side(team_a, ctx_a, team_b)
        score_b, cont_b = self.score_side(team_b, ctx_b, team_a)
        raw_diff = score_a - score_b
        factor, chaos_cont = self.chaos_factor(shared)
        diff = raw_diff * factor
        prob_a = 1.0 / (1.0 + math.exp(-self.logit_scale * diff))

        elo_a = self.league.apply("elo", self._team_value(team_a, "elo"))
        elo_b = self.league.apply("elo", self._team_value(team_b, "elo"))
        base_diff = (elo_a - elo_b) * self.weights["elo"]
        prob_baseline = 1.0 / (1.0 + math.exp(-self.logit_scale * base_diff))

        return {
            "team_a": team_a,
            "team_b": team_b,
            "prob_a": round(prob_a, 4),
            "prob_b": round(1 - prob_a, 4),
            "score_a": round(score_a, 4),
            "score_b": round(score_b, 4),
            "raw_score_diff": round(raw_diff, 4),
            "chaos_factor": round(factor, 4),
            "adjusted_diff": round(diff, 4),
            "baseline_prob_a_elo_only": round(prob_baseline, 4),
            "superstition_swing_pct": round((prob_a - prob_baseline) * 100, 2),
            "breakdown_a": [c.as_dict() for c in cont_a],
            "breakdown_b": [c.as_dict() for c in cont_b],
            "chaos_terms": [c.as_dict() for c in chaos_cont],
            "narrative": self._narrate(team_a, team_b, prob_a, prob_baseline, cont_a, cont_b, chaos_cont),
            "disclaimer": "Entertainment only. Superstition terms are unvalidated. Do not bet.",
        }

    def market_edge(
        self,
        team_a: str,
        team_b: str,
        ml_a: float,
        ml_b: float,
        game_context: Optional[Dict[str, Any]] = None,
        spread: Optional[float] = None,
        week: int = 1,
    ) -> Dict[str, Any]:
        """V2: compare model probability to no-vig market probability.

        `spread` is absolute points attached to the market favorite when known.
        Team B is treated as home if game_context['b']['home'] is truthy.
        """
        result = self.predict(team_a, team_b, game_context)
        prob_a, prob_b = result["prob_a"], result["prob_b"]
        mkt_a, mkt_b = no_vig_pair(ml_a, ml_b)
        pick = team_a.upper() if prob_a >= prob_b else team_b.upper()
        pick_prob = prob_a if pick == team_a.upper() else prob_b
        pick_market = mkt_a if pick == team_a.upper() else mkt_b
        pick_odds = float(ml_a if pick == team_a.upper() else ml_b)
        edge = pick_prob - pick_market

        favorite = team_a.upper() if moneyline_to_implied(ml_a) >= moneyline_to_implied(ml_b) else team_b.upper()
        is_favorite_pick = pick == favorite
        ctx = game_context or {}
        home_team = team_b.upper() if dict(ctx.get("b", {})).get("home") else None
        is_road_favorite = is_favorite_pick and home_team is not None and pick != home_team

        trap_flags: List[str] = []
        if pick_odds <= EXPENSIVE_FAVORITE_CUTOFF:
            trap_flags.append("expensive_favorite")
        if is_road_favorite:
            trap_flags.append("road_favorite")
        if is_favorite_pick and spread is not None and abs(float(spread)) >= BIG_SPREAD:
            trap_flags.append("big_spread_favorite")
        if dict(ctx.get("shared", {})).get("divisional_game"):
            trap_flags.append("divisional_chaos")
        if week <= 1:
            trap_flags.append("week1_uncertainty")
        if is_favorite_pick and pick_odds <= -150:
            trap_flags.append("public_chalk_tax")

        if edge < EDGE_PASS:
            recommendation = "PASS_NO_EDGE"
        elif pick_odds <= EXPENSIVE_FAVORITE_CUTOFF and edge < EDGE_BIG_PLAY:
            recommendation = "PASS_EXPENSIVE_FAVORITE"
        elif not is_favorite_pick and edge >= EDGE_PLAY:
            recommendation = "DOG_VALUE_PLAY"
        elif edge >= EDGE_PLAY:
            recommendation = "PLAY"
        elif edge >= EDGE_LEAN:
            recommendation = "LEAN"
        else:
            recommendation = "SMALL_LEAN"

        stake_units = _stake_units(edge, recommendation)
        if recommendation.startswith("PASS"):
            confidence = "pass"
        elif edge >= EDGE_BIG_PLAY:
            confidence = "high-entertainment"
        elif edge >= EDGE_PLAY:
            confidence = "medium-high"
        elif edge >= EDGE_LEAN:
            confidence = "medium"
        else:
            confidence = "low"

        market = {
            "ml_a": float(ml_a),
            "ml_b": float(ml_b),
            "raw_implied_a": round(moneyline_to_implied(ml_a), 4),
            "raw_implied_b": round(moneyline_to_implied(ml_b), 4),
            "no_vig_implied_a": round(mkt_a, 4),
            "no_vig_implied_b": round(mkt_b, 4),
            "favorite": favorite,
            "spread": spread,
        }
        result.update({
            "market": market,
            "v2_edge": {
                "model_pick": pick,
                "model_pick_prob": round(pick_prob, 4),
                "market_pick_prob": round(pick_market, 4),
                "edge": round(edge, 4),
                "edge_pct": round(edge * 100, 2),
                "pick_moneyline": pick_odds,
                "recommendation": recommendation,
                "stake_units": stake_units,
                "confidence": confidence,
                "trap_flags": trap_flags,
            },
        })
        result["narrative"] = result["narrative"] + [
            f"Market no-vig says {pick} {pick_market * 100:.1f}%; model says {pick_prob * 100:.1f}% (edge {edge * 100:+.1f} pts).",
            f"V2 filter: {recommendation}, stake_units={stake_units}, traps={', '.join(trap_flags) or 'none'}.",
        ]
        return result

    def _narrate(
        self,
        a: str,
        b: str,
        prob_a: float,
        prob_base: float,
        cont_a: List[Contribution],
        cont_b: List[Contribution],
        chaos: List[Contribution],
    ) -> List[str]:
        lines = [f"{a} {prob_a * 100:.1f}% — {b} {(1 - prob_a) * 100:.1f}%"]

        def top_spooky(conts: List[Contribution], team: str) -> List[str]:
            out = []
            for c in conts:
                if c.key in ("elo", "point_diff", "home"):
                    continue
                if abs(c.points) < 0.005:
                    continue
                verb = "helps" if c.points > 0 else "hurts"
                out.append(f"  {c.label} {verb} {team} ({c.points:+.3f})")
            return out[:4]

        lines += top_spooky(cont_a, a)
        lines += top_spooky(cont_b, b)
        for c in chaos:
            lines.append(f"  {c.label} compresses the edge (-{c.points * 100:.1f}% of the gap)")
        swing = (prob_a - prob_base) * 100
        lines.append(
            f"Elo alone says {a} {prob_base * 100:.1f}%. "
            f"The nonsense moved it {swing:+.1f} points, and the nonsense is noise."
        )
        return lines

    def slate(self, game_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        rows = []
        names = list(self.teams.index)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                r = self.predict(a, b, game_context)
                rows.append({
                    "team_a": a,
                    "team_b": b,
                    "prob_a": r["prob_a"],
                    "elo_only": r["baseline_prob_a_elo_only"],
                    "swing_pct": r["superstition_swing_pct"],
                })
        out = pd.DataFrame(rows)
        return out.reindex(out["prob_a"].sub(0.5).abs().sort_values(ascending=False).index)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probs, 1e-9, 1 - 1e-9)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def backtest(model: ConspiracyModel, results_csv: str | Path) -> Dict[str, Any]:
    games = pd.read_csv(results_csv)
    required = {"team_a", "team_b", "a_won"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"results file missing columns: {sorted(missing)}")

    model_p, base_p, outcomes = [], [], []
    v2_rows: List[Dict[str, Any]] = []
    has_market = {"ml_a", "ml_b"}.issubset(games.columns)
    for _, g in games.iterrows():
        ctx: Dict[str, Any] = {"a": {}, "b": {}, "shared": {}}
        for col in games.columns:
            if col.startswith("ctx_a_"):
                ctx["a"][col[6:]] = g[col]
            elif col.startswith("ctx_b_"):
                ctx["b"][col[6:]] = g[col]
            elif col.startswith("ctx_"):
                ctx["shared"][col[4:]] = g[col]
        team_a, team_b = str(g["team_a"]).upper(), str(g["team_b"]).upper()
        r = model.predict(team_a, team_b, ctx)
        model_p.append(r["prob_a"])
        base_p.append(r["baseline_prob_a_elo_only"])
        outcome = float(g["a_won"])
        outcomes.append(outcome)

        if has_market:
            spread = float(g["spread"]) if "spread" in games.columns and pd.notna(g["spread"]) else None
            week = int(g["week"]) if "week" in games.columns and pd.notna(g["week"]) else 1
            edge = model.market_edge(team_a, team_b, float(g["ml_a"]), float(g["ml_b"]), ctx, spread=spread, week=week)
            v2 = edge["v2_edge"]
            pick_is_a = v2["model_pick"] == team_a
            won = bool(outcome == 1.0) if pick_is_a else bool(outcome == 0.0)
            flat_profit = bet_profit(100.0, v2["pick_moneyline"], won)
            rec_stake = 100.0 * float(v2["stake_units"])
            rec_profit = bet_profit(rec_stake, v2["pick_moneyline"], won) if rec_stake else 0.0
            v2_rows.append({
                "game": f"{team_a} vs {team_b}",
                "pick": v2["model_pick"],
                "moneyline": v2["pick_moneyline"],
                "edge_pct": v2["edge_pct"],
                "recommendation": v2["recommendation"],
                "stake_units": v2["stake_units"],
                "won": won,
                "flat_100_profit": round(flat_profit, 2),
                "recommended_profit": round(rec_profit, 2),
            })

    mp, bp, y = np.array(model_p), np.array(base_p), np.array(outcomes)
    report: Dict[str, Any] = {
        "n_games": int(len(y)),
        "conspiracy_model": {"log_loss": round(log_loss(mp, y), 4), "brier": round(brier(mp, y), 4)},
        "elo_baseline": {"log_loss": round(log_loss(bp, y), 4), "brier": round(brier(bp, y), 4)},
        "coin_flip": {"log_loss": round(log_loss(np.full_like(y, 0.5), y), 4), "brier": 0.25},
    }
    if "vegas_prob_a" in games.columns:
        vp = games["vegas_prob_a"].to_numpy(dtype=float)
        report["vegas"] = {"log_loss": round(log_loss(vp, y), 4), "brier": round(brier(vp, y), 4)}
    if v2_rows:
        plays = [r for r in v2_rows if not str(r["recommendation"]).startswith("PASS")]
        report["v2_market_filter"] = {
            "flat_100_profit_if_all_model_picks": round(sum(r["flat_100_profit"] for r in v2_rows), 2),
            "recommended_staked": round(sum(100.0 * float(r["stake_units"]) for r in v2_rows), 2),
            "recommended_profit": round(sum(r["recommended_profit"] for r in v2_rows), 2),
            "recommended_roi_pct": round((sum(r["recommended_profit"] for r in v2_rows) / max(sum(100.0 * float(r["stake_units"]) for r in v2_rows), 1e-9)) * 100, 2),
            "plays": len(plays),
            "play_record": f"{sum(r['won'] for r in plays)}-{len(plays) - sum(r['won'] for r in plays)}",
            "rows": v2_rows,
        }

    beat = report["conspiracy_model"]["log_loss"] < report["elo_baseline"]["log_loss"]
    report["verdict"] = (
        "Conspiracy model beat Elo. On this sample size that is almost certainly overfitting or luck, not discovery. Re-run on a held-out season."
        if beat
        else "Elo baseline wins, as expected. The superstition terms are costing you accuracy. Working as intended."
    )
    return report


SAMPLE_TEAMS = """team,elo,point_diff,popularity,star_qb,ref_bias,qb_new_relationship,madden_cover,si_cover,sb_hangover,contract_year,farewell_tour,coach_hot_seat,owner_controversy,heisman_qb,color_rush_record
KC,1650,95,0.90,1.00,0.10,0,0,1,1,0,0,0,0,0,0
BUF,1620,88,0.70,0.90,0.00,0,0,0,0,1,0,0,0,0,0
DAL,1550,40,1.00,0.70,0.20,1,0,0,0,0,0,1,1,0,0
SF,1605,102,0.80,0.85,-0.05,0,1,0,0,0,0,0,0,0,1
NYG,1480,-55,0.60,0.50,-0.10,0,0,0,0,0,0,1,0,0,0
GB,1540,25,0.75,0.75,0.05,0,0,0,0,0,1,0,0,0,1
DET,1560,60,0.55,0.70,0.00,0,0,0,0,1,0,0,0,0,0
MIA,1510,10,0.50,0.65,-0.05,0,0,0,0,0,0,0,0,0,0
"""

SAMPLE_GAME = {
    "a": {"primetime": 1, "short_week_road": 1, "timezones_crossed": 2, "sharp_money_on": 1},
    "b": {"primetime": 1, "home": 1, "cold_weather_edge": 1},
    "shared": {"full_moon": 1, "mercury_retrograde": 1, "wind_over_15mph": 1, "divisional_game": 0},
}

SAMPLE_RESULTS = """team_a,team_b,a_won,vegas_prob_a,ml_a,ml_b,spread,week,ctx_a_primetime,ctx_b_home,ctx_full_moon,ctx_divisional_game
KC,BUF,1,0.55,-125,105,1.5,1,1,1,0,0
DAL,NYG,0,0.62,-162,136,3.0,1,1,1,0,1
SF,GB,1,0.68,-190,160,4.0,1,1,0,0,0
DET,MIA,0,0.60,-145,125,2.5,1,0,1,0,0
BUF,DAL,1,0.62,-118,-102,1.5,1,1,0,1,0
KC,SF,0,0.51,-110,-110,0.0,1,1,1,0,0
GB,NYG,1,0.70,-250,205,5.5,1,0,0,0,0
MIA,KC,0,0.28,142,-170,3.0,1,0,1,1,0
"""


def write_samples(dest: Path) -> List[Path]:
    files = []
    for name, content in (
        ("teams.csv", SAMPLE_TEAMS),
        ("sample_game.json", json.dumps(SAMPLE_GAME, indent=2) + "\n"),
        ("results.csv", SAMPLE_RESULTS),
    ):
        p = dest / name
        p.write_text(content)
        files.append(p)
    return files


def print_features() -> None:
    order = {"team": 0, "game": 1, "chaos": 2}
    print(f"{'KEY':<26}{'WEIGHT':>8}  {'KIND':<6}{'SCALE':<7}LABEL")
    print("-" * 96)
    for feat in sorted(REGISTRY, key=lambda f: (order[f.kind], -abs(f.weight))):
        print(f"{feat.key:<26}{feat.weight:>+8.3f}  {feat.kind:<6}{feat.scale:<7}{feat.label}")
        print(f"{'':<26}{'':>8}  {'':<13}{feat.rationale}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parody conspiracy win-probability model.")
    ap.add_argument("--teams", help="path to teams CSV")
    ap.add_argument("--matchup", nargs=2, metavar=("TEAM_A", "TEAM_B"))
    ap.add_argument("--game", help="path to game-context JSON")
    ap.add_argument("--slate", action="store_true", help="score every pairing")
    ap.add_argument("--backtest", metavar="RESULTS_CSV")
    ap.add_argument("--list-features", action="store_true")
    ap.add_argument("--write-samples", metavar="DIR", nargs="?", const=".")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--ml-a", type=float, help="American moneyline for TEAM_A; enables V2 market-edge filter with --ml-b")
    ap.add_argument("--ml-b", type=float, help="American moneyline for TEAM_B; enables V2 market-edge filter with --ml-a")
    ap.add_argument("--spread", type=float, help="absolute spread attached to the market favorite, for trap flags")
    ap.add_argument("--week", type=int, default=1, help="NFL week number, used for uncertainty flags")
    ap.add_argument("--scale", type=float, default=LOGIT_SCALE)
    ap.add_argument("--mirrors", action="store_true", help="enable opponent mirror features (double-counts; off by default)")
    args = ap.parse_args(argv)

    if args.list_features:
        print_features()
        return 0

    if args.write_samples:
        dest = Path(args.write_samples)
        dest.mkdir(parents=True, exist_ok=True)
        for p in write_samples(dest):
            print(f"wrote {p}")
        return 0

    if not args.teams:
        ap.error("--teams is required (or use --write-samples / --list-features)")

    model = ConspiracyModel.from_csv(args.teams, logit_scale=args.scale, use_mirrors=args.mirrors)

    if args.backtest:
        print(json.dumps(backtest(model, args.backtest), indent=2))
        return 0

    ctx = json.loads(Path(args.game).read_text()) if args.game else None

    if args.slate:
        print(model.slate(ctx).to_string(index=False))
        return 0

    if args.matchup:
        a, b = (t.strip().upper() for t in args.matchup)
        try:
            if (args.ml_a is None) ^ (args.ml_b is None):
                ap.error("--ml-a and --ml-b must be supplied together")
            if args.ml_a is not None and args.ml_b is not None:
                result = model.market_edge(a, b, args.ml_a, args.ml_b, ctx, spread=args.spread, week=args.week)
            else:
                result = model.predict(a, b, ctx)
        except KeyError as exc:
            print(str(exc).strip('"'), file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for line in result["narrative"]:
                print(line)
            print(f"\n{result['disclaimer']}")
        return 0

    ap.error("nothing to do: pass --matchup, --slate, or --backtest")
    return 1


if __name__ == "__main__":
    sys.exit(main())
