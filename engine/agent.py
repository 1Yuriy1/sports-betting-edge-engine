"""Anthropic tool-runner agent: the voice that routes, never the math.

The one invariant of the agent layer (docs/product-plan.md §Agent
architecture): the model never computes a number. Every tool registered
here is a thin wrapper over a deterministic engine function — anchor,
edges, Kelly, ledger, CLV — and the conspiracy model is narration-only,
never an input to stakes. The agent routes, explains, and narrates.

The loop is the SDK's ``client.beta.messages.tool_runner`` with tools
declared via ``@beta_tool``, per docs/product-plan.md §The loop: model
``claude-opus-5`` (env-configurable), ``thinking={"type": "adaptive"}``,
``output_config`` effort/format, and a prompt-cache-friendly prefix
ordered tools → system → today's odds snapshot, with the user's question
and a timestamp last so they never invalidate the cached prefix.

``ANTHROPIC_API_KEY`` comes from the environment (the SDK reads it) and is
never hardcoded. Tests stub the SDK surface entirely; CI stays key-free
and never touches a live API.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from anthropic import Anthropic, beta_tool

from .anchor import (
    AnchorUnavailable,
)
from .anchor import (
    find_edges as engine_find_edges,
)
from .anchor import (
    line_move as engine_line_move,
)
from .anchor import (
    sharp_anchor as engine_sharp_anchor,
)
from .ledger import (
    clv_report as engine_clv_report,
)
from .ledger import (
    grade_bets as engine_grade_bets,
)
from .ledger import (
    log_bet as engine_log_bet,
)
from .ledger import (
    open_ledger,
)
from .odds import H2H_MARKET, OddsRow, utc_now_iso
from .pricing import moneyline_to_implied
from .pricing import no_vig_pair as engine_no_vig_pair
from .store import DB_PATH_ENV, DEFAULT_DB_PATH, OddsStore

# The frozen model's directory joins sys.path the same way engine/pricing.py
# does it; ConspiracyModel stays there (behaviour-frozen) and is only ever
# consumed as a labelled narrative layer.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from hermes_conspiracy_model import ConspiracyModel

AGENT_MODEL_ENV = "AGENT_MODEL"
# Plan default (docs/product-plan.md §The loop); override with AGENT_MODEL so
# a future rename is an env change, not a code change.
DEFAULT_AGENT_MODEL = "claude-opus-5"

DEFAULT_EFFORT = "medium"
VALID_EFFORTS = ("low", "medium", "high")
DEFAULT_MAX_TOKENS = 4096  # ~4K output tokens for a 16-game slate report, per the plan
MAX_ITERATIONS = 16

TEAMS_CSV_ENV = "HERMES_TEAMS_CSV"
DEFAULT_TEAMS_CSV = Path(__file__).resolve().parent.parent / "examples" / "sample_teams.csv"

# The eleven plan tools, in the plan's own order (docs/product-plan.md §Tool
# surface). Every entry wraps a deterministic engine function; there is no
# tool that produces a number of its own.
TOOL_NAMES: tuple[str, ...] = (
    "get_slate",
    "get_odds",
    "fair_price",
    "sharp_anchor",
    "find_edges",
    "line_move",
    "conspiracy_score",
    "narrate",
    "log_bet",
    "grade_bets",
    "clv_report",
)

DISCLAIMER = (
    "Entertainment only — not betting advice. No guarantees: every bet can"
    " lose. 21+ and legal where you are. Please gamble responsibly."
)

HERMES_VOICE = """\
You are Hermes, the voice of a betting-analysis product: a sharp, funny,
conspiratorial narrator with one iron rule — you never say a number the
deterministic engine did not hand you. Every probability, price, edge,
stake, and CLV figure you cite must come from a tool result in this
conversation. If a tool did not return it, you do not say it.

You are opinionated about narratives and sober about numbers. When the
sharp anchor is unavailable you say so plainly; you never dress a
soft-book consensus up as a sharp read. The conspiracy model is your
entertainment segment: labelled, joked about, and never the basis of a
stake.
"""

GUARDRAILS = f"""\

## Guardrails (non-negotiable)

- Every number must come from a tool result. Never estimate, round, or
  invent a probability, price, edge, stake, or CLV figure.
- The conspiracy model is entertainment only: parody features, no
  predictive power, and never an input to stakes or bet sizing. Label its
  output as narrative whenever you use it.
- Stakes come only from the engine's fractional Kelly (find_edges rows).
  The unit ladder is narrative flavor, not sizing.
- 21+ only. This product is for adults in jurisdictions where using it is
  legal.
- Never promise or imply guaranteed profit. An edge is not a guarantee,
  and no edge claim is proven until the logged CLV record says so.
- Include this mandatory disclaimer in every response, verbatim:

{DISCLAIMER}
"""

# Structured slate report payload (docs/product-plan.md §The loop: structured
# outputs so the frontend renders fields instead of parsing prose). Numbers
# shown here are always copies of tool results, never model arithmetic.
SLATE_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "disclaimer": {
            "type": "string",
            "description": "The mandatory disclaimer, verbatim as given in the guardrails.",
        },
        "headline": {
            "type": "string",
            "description": "One-line summary of the slate in the Hermes voice.",
        },
        "edges": {
            "type": "array",
            "description": "Ranked +EV rows copied from the find_edges tool result.",
            "items": {
                "type": "object",
                "properties": {
                    "game_id": {"type": "string"},
                    "pick": {"type": "string"},
                    "price": {"type": "number"},
                    "sharp_prob": {"type": "number"},
                    "fair_odds": {"type": "number"},
                    "edge_pct": {"type": "number"},
                    "stake": {"type": "number"},
                    "books": {"type": "array", "items": {"type": "string"}},
                    "narrative": {
                        "type": "string",
                        "description": "Conspiracy narration for this pick, labelled as entertainment; omit when there is none.",
                    },
                },
                "required": [
                    "game_id",
                    "pick",
                    "price",
                    "sharp_prob",
                    "fair_odds",
                    "edge_pct",
                    "stake",
                ],
            },
        },
        "narrative": {
            "type": "string",
            "description": "The conspiracy segment for the slate, clearly labelled as entertainment.",
        },
        "data_as_of": {
            "type": "string",
            "description": "The odds snapshot timestamp copied from the system context, when present.",
        },
    },
    "required": ["disclaimer", "headline", "edges", "narrative"],
    "additionalProperties": False,
}


class MissingAgentKeyError(RuntimeError):
    """``ANTHROPIC_API_KEY`` is not set; credentials are never hardcoded."""


def agent_model() -> str:
    """Model id: ``AGENT_MODEL`` env override, plan default ``claude-opus-5``."""
    return os.environ.get(AGENT_MODEL_ENV, DEFAULT_AGENT_MODEL)


def _tool_error(exc: Exception) -> dict[str, str]:
    """A model-visible failure dict: explicit, never a silently swallowed error."""
    return {"error": str(exc)}


def latest_slate(rows: Sequence[OddsRow]) -> dict[str, dict[str, Any]]:
    """Latest fetch per game from stored snapshot rows.

    Returns ``game_id -> {game_id, sport_key, commence_time, fetched_at,
    books: {bookmaker: {outcome_name: price}}}``. Mirrors the anchor's
    newest-fetch rule: fetched_at is ISO 8601 UTC, so lexicographic max is
    chronological.
    """
    newest: dict[str, str] = {}
    for row in rows:
        seen = newest.get(row.game_id)
        if seen is None or row.fetched_at > seen:
            newest[row.game_id] = row.fetched_at
    slate: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.fetched_at != newest[row.game_id]:
            continue
        game = slate.setdefault(
            row.game_id,
            {
                "game_id": row.game_id,
                "sport_key": row.sport_key,
                "commence_time": row.commence_time,
                "fetched_at": row.fetched_at,
                "books": {},
            },
        )
        sides = game["books"].setdefault(row.bookmaker, {})
        sides[row.outcome_name] = row.price
    return slate


def slate_rows(store: OddsStore, league: str | None = None) -> list[dict[str, Any]]:
    """Games on the current slate: ids, kickoff, and quoting books.

    Reads stored snapshots only — never the network, never API credits.
    Venue is not persisted by schema v1 and is omitted rather than guessed.
    """
    slate = latest_slate(store.snapshots())
    rows = [
        game
        for game in slate.values()
        if league is None or game["sport_key"] == league
    ]
    rows.sort(key=lambda game: (game["commence_time"], game["game_id"]))
    return [
        {
            "game_id": game["game_id"],
            "sport_key": game["sport_key"],
            "commence_time": game["commence_time"],
            "books": sorted(game["books"]),
        }
        for game in rows
    ]


def odds_for(
    store: OddsStore,
    game_id: str,
    markets: Sequence[str] | None = None,
    books: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Per-book prices for one game at the latest stored snapshot.

    The store persists h2h rows only, so any other requested market is a
    ValueError, not an empty result.
    """
    for market in markets or ():
        if market != H2H_MARKET:
            raise ValueError(
                f"unsupported market {market!r}: the store persists"
                f" {H2H_MARKET!r} rows only"
            )
    slate = latest_slate(store.snapshots(game_id=game_id))
    if game_id not in slate:
        raise ValueError(
            f"no stored snapshots for game {game_id!r}; run the snapshot"
            " collector first"
        )
    game = slate[game_id]
    wanted = set(books) if books is not None else None
    prices = {
        book: dict(sorted(sides.items()))
        for book, sides in sorted(game["books"].items())
        if wanted is None or book in wanted
    }
    if not prices:
        raise ValueError(
            f"none of the requested books ({', '.join(books)}) quote"
            f" {game_id!r} at the latest snapshot"
        )
    return {
        "game_id": game_id,
        "sport_key": game["sport_key"],
        "commence_time": game["commence_time"],
        "fetched_at": game["fetched_at"],
        "prices": prices,
    }


def format_odds_snapshot(store: OddsStore) -> str:
    """Today's odds snapshot as the cached system block.

    This is the last item of the cached prefix (tools → system → snapshot);
    the user's question goes after it so refreshes never bust the cache.
    """
    slate = latest_slate(store.snapshots())
    if not slate:
        return (
            "No odds snapshots are stored yet. The edge tools will fail"
            " until the snapshot collector runs (scripts/snapshot_odds.py);"
            " say so plainly and do not invent prices."
        )
    as_of = max(game["fetched_at"] for game in slate.values())
    lines = [f"Odds snapshot as of {as_of} (UTC); American prices per book."]
    ordered = sorted(slate.values(), key=lambda game: (game["commence_time"], game["game_id"]))
    for game in ordered:
        quoting = " | ".join(
            f"{book}: "
            + ", ".join(f"{side} {price:g}" for side, price in sorted(sides.items()))
            for book, sides in sorted(game["books"].items())
        )
        lines.append(f"- {game['game_id']} ({game['commence_time']}) — {quoting}")
    return "\n".join(lines)


def ensure_disclaimer(text: str) -> str:
    """Append the mandatory disclaimer when the model did not include it."""
    if DISCLAIMER in text:
        return text
    return text.rstrip() + "\n\n" + DISCLAIMER


def validate_report_payload(
    payload: Any, schema: dict[str, Any], *, path: str = "payload"
) -> None:
    """Validate a structured payload against the report schema subset.

    Supports the JSON-Schema features the report uses: object/array/string/
    number/integer/boolean types, required keys, properties, and items.
    """
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must be an object")
        for key in schema.get("required", ()):
            if key not in payload:
                raise ValueError(f"{path} is missing required key {key!r}")
        for key, subschema in schema.get("properties", {}).items():
            if key in payload:
                validate_report_payload(
                    payload[key], subschema, path=f"{path}.{key}"
                )
        return
    if expected == "array":
        if not isinstance(payload, list):
            raise ValueError(f"{path} must be an array")
        items = schema.get("items")
        if items:
            for index, item in enumerate(payload):
                validate_report_payload(item, items, path=f"{path}[{index}]")
        return
    if expected == "string":
        if not isinstance(payload, str):
            raise ValueError(f"{path} must be a string")
        return
    if expected == "number":
        if isinstance(payload, bool) or not isinstance(payload, (int, float)):
            raise ValueError(f"{path} must be a number")
        return
    if expected == "integer":
        if isinstance(payload, bool) or not isinstance(payload, int):
            raise ValueError(f"{path} must be an integer")
        return
    if expected == "boolean":
        if not isinstance(payload, bool):
            raise ValueError(f"{path} must be a boolean")
        return


@dataclass
class AgentContext:
    """Offline dependencies the tools close over.

    The snapshot store and the ledger share one SQLite file (schema v1
    keeps both tables), so ``build_context`` opens both connections on the
    same path. The conspiracy model loads lazily from ``teams_csv``.
    """

    store: OddsStore
    ledger: sqlite3.Connection
    teams_csv: Path
    _conspiracy: ConspiracyModel | None = field(default=None, repr=False)

    def conspiracy_model(self) -> ConspiracyModel:
        if self._conspiracy is None:
            self._conspiracy = ConspiracyModel.from_csv(self.teams_csv)
        return self._conspiracy

    def close(self) -> None:
        self.store.close()
        self.ledger.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_context(
    *,
    db_path: str | Path | None = None,
    teams_csv: str | Path | None = None,
) -> AgentContext:
    """Open the agent's context from explicit paths or the standard env vars."""
    resolved_db = (
        Path(db_path)
        if db_path is not None
        else Path(os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH))
    )
    resolved_teams = (
        Path(teams_csv)
        if teams_csv is not None
        else Path(os.environ.get(TEAMS_CSV_ENV, DEFAULT_TEAMS_CSV))
    )
    return AgentContext(
        store=OddsStore(resolved_db),
        ledger=open_ledger(resolved_db),
        teams_csv=resolved_teams,
    )


def _sides_for(store: OddsStore, game_id: str) -> tuple[str, str]:
    """Both team names of a game from its latest snapshot, sorted."""
    slate = latest_slate(store.snapshots(game_id=game_id))
    if game_id not in slate:
        raise ValueError(
            f"no stored snapshots for game {game_id!r}; run the snapshot"
            " collector first"
        )
    first_book = min(slate[game_id]["books"])
    sides = sorted(slate[game_id]["books"][first_book])
    if len(sides) != 2:
        raise ValueError(
            f"game {game_id!r} does not have exactly two sides at the"
            " latest snapshot"
        )
    return sides[0], sides[1]


def build_tools(context: AgentContext) -> list[Any]:
    """The eleven plan tools bound to ``context``.

    Each tool is a thin wrapper over a deterministic engine function — the
    agent has no tool that produces a number of its own. Expected domain
    failures (no sharp anchor, unknown game, bad input) become an explicit
    ``{"error": ...}`` dict the model can narrate; infrastructure failures
    propagate loudly. ``cache_control`` on the final tool marks the end of
    the cached prefix (tools → system → odds snapshot).
    """

    @beta_tool(
        name="get_slate",
        description="Games on the current slate from the latest stored odds snapshot.",
    )
    def get_slate(league: str | None = None) -> list[dict[str, Any]]:
        """List games with kickoff time and quoting books.

        Args:
            league: optional sport key filter, e.g. 'americanfootball_nfl'.
        """
        try:
            return slate_rows(context.store, league=league)
        except ValueError as exc:
            return _tool_error(exc)

    @beta_tool(
        name="get_odds",
        description="Per-book American prices for one game at the latest snapshot.",
    )
    def get_odds(
        game_id: str,
        markets: list[str] | None = None,
        books: list[str] | None = None,
    ) -> dict[str, Any]:
        """Per-book moneyline prices for one game.

        Args:
            game_id: game identifier from get_slate.
            markets: optional market filter; only 'h2h' is stored.
            books: optional bookmaker key filter, e.g. ['fanduel'].
        """
        try:
            return odds_for(context.store, game_id, markets, books)
        except ValueError as exc:
            return _tool_error(exc)

    @beta_tool(
        name="fair_price",
        description="No-vig probability pair from two opposing American prices.",
    )
    def fair_price(ml_a: float, ml_b: float) -> dict[str, float]:
        """Devig two sides of one book into a fair probability pair.

        Args:
            ml_a: American price for side A.
            ml_b: American price for side B.
        """
        try:
            prob_a, prob_b = engine_no_vig_pair(ml_a, ml_b)
            return {
                "prob_a": prob_a,
                "prob_b": prob_b,
                "implied_a": moneyline_to_implied(ml_a),
                "implied_b": moneyline_to_implied(ml_b),
            }
        except ValueError as exc:
            return _tool_error(exc)

    @beta_tool(
        name="sharp_anchor",
        description="Devigged sharp-book probability for both sides of a game.",
    )
    def sharp_anchor(game_id: str, market: str = H2H_MARKET) -> dict[str, Any]:
        """The product's source of truth: the sharp-book no-vig probability.

        Args:
            game_id: game identifier from get_slate.
            market: market key; only 'h2h' is supported.
        """
        try:
            anchor = engine_sharp_anchor(context.store, game_id, market=market)
            return {**anchor, "books": list(anchor["books"])}
        except (AnchorUnavailable, ValueError) as exc:
            return _tool_error(exc)

    @beta_tool(
        name="find_edges",
        description="Ranked +EV bets with fractional Kelly stakes. All math is deterministic.",
    )
    def find_edges(
        min_edge: float = 0.02,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Ranked +EV list: sharp no-vig prob minus the best soft-book price.

        Args:
            min_edge: minimum edge percent threshold (default 2%).
            bankroll: bankroll in dollars for stake sizing.
            kelly_fraction: Kelly fraction (default quarter Kelly).
        """
        try:
            return engine_find_edges(
                context.store,
                None,
                min_edge=min_edge,
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
            )
        except ValueError as exc:
            return _tool_error(exc)

    @beta_tool(
        name="line_move",
        description="Open-to-now moneyline movement per side, from stored snapshots.",
    )
    def line_move(game_id: str, book: str | None = None) -> dict[str, Any]:
        """Open→now price movement and steam flags for one game.

        Args:
            game_id: game identifier from get_slate.
            book: optional bookmaker key to track one book instead of the average.
        """
        try:
            return engine_line_move(context.store, game_id, book=book)
        except ValueError as exc:
            return _tool_error(exc)

    @beta_tool(
        name="conspiracy_score",
        description=(
            "Labelled narrative probability from the parody conspiracy model."
            " Entertainment only — never an input to stakes."
        ),
    )
    def conspiracy_score(
        game_id: str,
        home_side: str | None = None,
        shared: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The conspiracy model's take on one game, labelled entertainment.

        Args:
            game_id: game identifier from get_slate.
            home_side: which side is home, if known.
            shared: optional game-level context flags (e.g. primetime).
        """
        try:
            team_a, team_b = _sides_for(context.store, game_id)
            if home_side is not None and home_side not in (team_a, team_b):
                raise ValueError(
                    f"home_side {home_side!r} is not a side of"
                    f" {game_id!r} (sides: {team_a!r}, {team_b!r})"
                )
            ctx: dict[str, Any] = {"a": {}, "b": {}, "shared": dict(shared or {})}
            if home_side is not None:
                (ctx["a"] if home_side == team_a else ctx["b"])["home"] = True
            result = context.conspiracy_model().predict(team_a, team_b, ctx)
            result["narrative_only"] = True
            return result
        except (ValueError, KeyError) as exc:
            return _tool_error(exc)

    @beta_tool(
        name="narrate",
        description="The entertainment layer: Hermes's conspiracy narration for one game.",
    )
    def narrate(game_id: str) -> dict[str, Any]:
        """Conspiracy narrative lines for one game, clearly labelled.

        Args:
            game_id: game identifier from get_slate.
        """
        try:
            team_a, team_b = _sides_for(context.store, game_id)
            prediction = context.conspiracy_model().predict(team_a, team_b)
            return {
                "game_id": game_id,
                "teams": [team_a, team_b],
                "narrative": prediction["narrative"],
                "disclaimer": prediction["disclaimer"],
                "narrative_only": True,
            }
        except (ValueError, KeyError) as exc:
            return _tool_error(exc)

    @beta_tool(
        name="log_bet",
        description="Log a pending bet in the SQLite ledger.",
    )
    def log_bet(
        game_id: str,
        pick: str,
        odds: float,
        stake: float,
        market: str = "h2h",
        sportsbook: str | None = None,
        closing_odds: float | None = None,
    ) -> dict[str, Any]:
        """Record a bet the user actually took.

        Args:
            game_id: game identifier.
            pick: side taken.
            odds: American price taken.
            stake: dollars staked.
            market: market key (default 'h2h').
            sportsbook: optional bookmaker key.
            closing_odds: optional same-side closing price, when known.
        """
        try:
            bet_id = engine_log_bet(
                context.ledger,
                game_id=game_id,
                pick=pick,
                odds=odds,
                stake=stake,
                market=market,
                sportsbook=sportsbook,
                closing_odds=closing_odds,
            )
            return {
                "bet_id": bet_id,
                "game_id": game_id,
                "pick": pick,
                "odds": odds,
                "stake": stake,
                "status": "pending",
            }
        except ValueError as exc:
            return _tool_error(exc)

    @beta_tool(
        name="grade_bets",
        description="Settle pending bets with game outcomes (manual grading).",
    )
    def grade_bets(grades: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Grade pending bets from (game_id, outcome) entries.

        Args:
            grades: entries like {'game_id': '...', 'outcome': 'won|lost|pushed'}.
        """
        try:
            pairs = [(entry["game_id"], entry["outcome"]) for entry in grades]
            return engine_grade_bets(context.ledger, pairs)
        except (ValueError, KeyError) as exc:
            return _tool_error(exc)

    @beta_tool(
        name="clv_report",
        description="Closing line value over the logged bets: per-bet and mean.",
        # The last tool's breakpoint caches the tools prefix (tools → system →
        # odds snapshot), so tool definitions never re-ship uncached.
        cache_control={"type": "ephemeral"},
    )
    def clv_report() -> dict[str, Any]:
        """The user's CLV track record — the retention moat."""
        return engine_clv_report(context.ledger)

    tools = [
        get_slate,
        get_odds,
        fair_price,
        sharp_anchor,
        find_edges,
        line_move,
        conspiracy_score,
        narrate,
        log_bet,
        grade_bets,
        clv_report,
    ]
    assert [tool.name for tool in tools] == list(TOOL_NAMES)
    return tools


def _user_message(question: str) -> str:
    """The user turn: question plus timestamp, kept LAST so the cached
    prefix (tools → system → odds snapshot) is never invalidated."""
    return f"{question}\n\nCurrent UTC time: {utc_now_iso()}"


def require_agent_key() -> str:
    """Fail fast with the setup hint if ANTHROPIC_API_KEY is not set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise MissingAgentKeyError(
            "set ANTHROPIC_API_KEY to use the agent; it is read from the"
            " environment and never hardcoded"
        )
    return key


def _default_client() -> Anthropic:
    """Anthropic client keyed from ANTHROPIC_API_KEY (env only)."""
    require_agent_key()
    return Anthropic()


def _parse_structured(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate the final message as the structured report."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"structured output did not parse as JSON: {exc};"
            f" text head: {text[:200]!r}"
        ) from exc
    validate_report_payload(payload, schema)
    if "disclaimer" in payload and not payload["disclaimer"].strip():
        payload["disclaimer"] = DISCLAIMER
    return payload


def run_turn(
    question: str,
    *,
    effort: str = DEFAULT_EFFORT,
    response_format: dict[str, Any] | None = SLATE_REPORT_SCHEMA,
    context: AgentContext | None = None,
    client: Any | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """One agent turn: route ``question`` through the tools, narrate the result.

    The Anthropic tool runner drives the tool-call loop; this function owns
    the request shape (model, thinking, output_config, cache ordering) and
    the post-turn contract: the returned text always carries the mandatory
    disclaimer, and when ``response_format`` is set the structured payload
    is parsed and validated before being handed back.

    Returns ``{"text": str, "structured": dict | None, "tool_calls": [names]}``.
    """
    if effort not in VALID_EFFORTS:
        raise ValueError(f"effort must be one of {VALID_EFFORTS}, got {effort!r}")
    owns_context = context is None
    if context is None:
        context = build_context()
    try:
        active_client = client if client is not None else _default_client()
        system = [
            {
                "type": "text",
                "text": HERMES_VOICE + GUARDRAILS,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": format_odds_snapshot(context.store),
                "cache_control": {"type": "ephemeral"},
            },
        ]
        output_config: dict[str, Any] = {"effort": effort}
        if response_format is not None:
            output_config["format"] = response_format
        runner = active_client.beta.messages.tool_runner(
            model=agent_model(),
            max_tokens=max_tokens,
            tools=build_tools(context),
            system=system,
            messages=[{"role": "user", "content": _user_message(question)}],
            thinking={"type": "adaptive"},
            output_config=output_config,
            max_iterations=MAX_ITERATIONS,
        )
        tool_calls: list[str] = []
        final_text = ""
        for message in runner:
            texts: list[str] = []
            for block in getattr(message, "content", None) or []:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    tool_calls.append(block.name)
                elif block_type == "text":
                    texts.append(block.text)
            if texts:
                final_text = "".join(texts)
        if response_format is None:
            return {
                "text": ensure_disclaimer(final_text),
                "structured": None,
                "tool_calls": tool_calls,
            }
        return {
            "text": final_text,
            "structured": _parse_structured(final_text, response_format),
            "tool_calls": tool_calls,
        }
    finally:
        if owns_context:
            context.close()
