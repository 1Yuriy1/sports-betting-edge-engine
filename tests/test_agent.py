"""Fully offline tests for the tool-runner agent layer.

The Anthropic SDK surface is stubbed (``StubAnthropic`` scripts the model's
turns and executes the *real* ``@beta_tool`` objects by name, mirroring the
real ``BetaToolRunner``), so the tool-call sequence is routed to the real
engine functions over a seeded temp SQLite store — no network, no API key,
no live model anywhere in CI.

Canonical fixture (shared with test_anchor): Pinnacle +150/-200 devigs to
exactly (0.375, 0.625); FanDuel's latest DET +190 prices the DET side at
implied 100/290, so the canned edge is 0.375 - 100/290 and the quarter-Kelly
stake off it (bankroll 1000, capped at 50).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

import engine.agent as agent_module
from engine.agent import (
    DEFAULT_AGENT_MODEL,
    DISCLAIMER,
    GUARDRAILS,
    HERMES_VOICE,
    MAX_ITERATIONS,
    SLATE_REPORT_SCHEMA,
    TOOL_NAMES,
    build_context,
    build_tools,
    ensure_disclaimer,
    format_odds_snapshot,
    latest_slate,
    run_turn,
    slate_rows,
    validate_report_payload,
)
from engine.anchor import find_edges as engine_find_edges
from engine.anchor import line_move as engine_line_move
from engine.anchor import sharp_anchor as engine_sharp_anchor
from engine.odds import OddsRow
from engine.store import OddsStore

SPORT = "americanfootball_nfl"
COMMENCE = "2026-09-21T17:00:00Z"
GAME = "det-buf-2026-09-21"
T1 = "2026-09-17T12:00:00Z"
T2 = "2026-09-17T18:00:00Z"
DET = "Detroit Lions"
BUF = "Buffalo Bills"


def _row(fetched_at: str, bookmaker: str, outcome_name: str, price: float) -> OddsRow:
    return OddsRow(
        fetched_at=fetched_at,
        sport_key=SPORT,
        game_id=GAME,
        commence_time=COMMENCE,
        bookmaker=bookmaker,
        outcome_name=outcome_name,
        price=float(price),
        book_updated_at=None,
    )


def _seed(store: OddsStore) -> None:
    store.insert_rows(
        [
            _row(T1, "pinnacle", DET, 150),
            _row(T1, "pinnacle", BUF, -200),
            _row(T1, "fanduel", DET, 200),
            _row(T1, "fanduel", BUF, -260),
            _row(T2, "pinnacle", DET, 150),
            _row(T2, "pinnacle", BUF, -200),
            _row(T2, "fanduel", DET, 190),
            _row(T2, "fanduel", BUF, -240),
        ]
    )


@pytest.fixture
def teams_csv(tmp_path):
    path = tmp_path / "teams.csv"
    path.write_text("team,elo\nDetroit Lions,1500\nBuffalo Bills,1600\n")
    return path


@pytest.fixture
def context(tmp_path, teams_csv):
    ctx = build_context(db_path=tmp_path / "odds.sqlite", teams_csv=teams_csv)
    _seed(ctx.store)
    yield ctx
    ctx.close()


@pytest.fixture
def tools(context):
    return {tool.name: tool for tool in build_tools(context)}


class _StubRunner:
    """Scripted stand-in for BetaToolRunner: executes real tools by name."""

    def __init__(self, script, kwargs, executed):
        self._script = script
        self._kwargs = kwargs
        self._executed = executed
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._script):
            raise StopIteration
        turn = self._script[self._index]
        self._index += 1
        blocks = []
        for name, tool_input in turn.get("tool_use", ()):
            tool = {t.name: t for t in self._kwargs["tools"]}[name]
            result = tool(**tool_input)
            self._executed.append(
                {"name": name, "input": tool_input, "result": result}
            )
            blocks.append(
                SimpleNamespace(type="tool_use", name=name, input=tool_input)
            )
        if "text" in turn:
            blocks.append(SimpleNamespace(type="text", text=turn["text"]))
        return SimpleNamespace(content=blocks)


class StubAnthropic:
    """Offline stand-in for anthropic.Anthropic.

    Records every tool_runner kwargs dict for request-shape assertions and
    drives the real tool objects exactly like the real runner does.
    """

    def __init__(self, script):
        self.script = script
        self.tool_runner_calls = []
        self.executed = []
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(tool_runner=self._tool_runner)
        )

    def _tool_runner(self, **kwargs):
        self.tool_runner_calls.append(kwargs)
        return _StubRunner(self.script, kwargs, self.executed)


def _report_payload(**overrides):
    """A structured slate report the scripted model would emit: every number
    is a copy of the canned find_edges row, plus the mandatory disclaimer."""
    payload = {
        "disclaimer": DISCLAIMER,
        "headline": "One +EV spot, and it is not the one the public likes.",
        "edges": [
            {
                "game_id": GAME,
                "pick": DET,
                "price": 190.0,
                "sharp_prob": 0.375,
                "fair_odds": 167,
                "edge_pct": 3.02,
                "stake": 11.51,
            }
        ],
        "narrative": "The league wants Buffalo in primetime. The sharp book disagrees.",
    }
    payload.update(overrides)
    return payload


class TestToolRegistry:
    def test_exactly_the_eleven_plan_tools_in_plan_order(self, context):
        tools = build_tools(context)
        assert [tool.name for tool in tools] == list(TOOL_NAMES)

    def test_every_tool_declares_a_schema_and_description(self, tools):
        for name in TOOL_NAMES:
            tool = tools[name]
            assert tool.description, name
            assert tool.input_schema.get("type") == "object", name

    def test_final_tool_carries_the_cache_breakpoint(self, context):
        last_tool = build_tools(context)[-1]
        assert last_tool._cache_control == {"type": "ephemeral"}


class TestToolsWrapTheEngine:
    def test_fair_price_devigs_canonical_pair(self, tools):
        result = tools["fair_price"](150.0, -200.0)
        assert result["prob_a"] == pytest.approx(0.375)
        assert result["prob_b"] == pytest.approx(0.625)

    def test_sharp_anchor_routes_to_engine(self, context, tools):
        result = tools["sharp_anchor"](GAME)
        expected = engine_sharp_anchor(context.store, GAME)
        assert result["probs"] == expected["probs"]
        assert result["books"] == list(expected["books"])

    def test_find_edges_routes_to_engine(self, context, tools):
        result = tools["find_edges"]()
        assert result == engine_find_edges(context.store, None)

    def test_find_edges_canned_row(self, tools):
        (row,) = tools["find_edges"]()
        assert row["game_id"] == GAME
        assert row["pick"] == DET
        assert row["price"] == 190.0
        assert row["sharp_prob"] == pytest.approx(0.375)
        assert row["fair_odds"] == 167
        assert row["edge_pct"] == pytest.approx(100 * (0.375 - 100 / 290), abs=1e-9)
        assert row["stake"] == pytest.approx(11.5131578947, abs=1e-6)
        assert row["stake"] <= 0.05 * 1000  # Kelly cap holds
        assert row["narrative"] is None  # narration never feeds stakes

    def test_line_move_routes_to_engine(self, context, tools):
        assert tools["line_move"](GAME) == engine_line_move(context.store, GAME)

    def test_get_odds_returns_latest_per_book(self, tools):
        result = tools["get_odds"](GAME)
        assert result["prices"]["pinnacle"] == {DET: 150.0, BUF: -200.0}
        assert result["prices"]["fanduel"] == {DET: 190.0, BUF: -240.0}
        assert result["fetched_at"] == T2

    def test_get_odds_rejects_unstored_market(self, tools):
        result = tools["get_odds"](GAME, markets=["totals"])
        assert "error" in result
        assert "h2h" in result["error"]

    def test_get_odds_book_filter(self, tools):
        result = tools["get_odds"](GAME, books=["fanduel"])
        assert list(result["prices"]) == ["fanduel"]

    def test_get_slate_lists_games_with_books(self, tools):
        result = tools["get_slate"]()
        assert result == [
            {
                "game_id": GAME,
                "sport_key": SPORT,
                "commence_time": COMMENCE,
                "books": ["fanduel", "pinnacle"],
            }
        ]

    def test_get_slate_league_filter(self, tools):
        assert tools["get_slate"](league="icehockey_nhl") == []

    def test_sharp_anchor_unknown_game_is_an_error_dict_not_a_crash(self, tools):
        result = tools["sharp_anchor"]("no-such-game")
        assert "error" in result
        assert "no stored snapshots" in result["error"]

    def test_conspiracy_score_is_labelled_narrative(self, tools):
        result = tools["conspiracy_score"](GAME)
        assert result["narrative_only"] is True
        assert result["prob_a"] == pytest.approx(1 - result["prob_b"])
        assert "Entertainment" in result["disclaimer"]

    def test_narrate_is_labelled_narrative(self, tools):
        result = tools["narrate"](GAME)
        assert result["teams"] == sorted([DET, BUF])
        assert result["narrative"]
        assert result["narrative_only"] is True
        assert "Entertainment" in result["disclaimer"]

    def test_conspiracy_never_touches_stakes(self, tools):
        """The invariant: even after running the conspiracy tools, the edge
        rows' stakes come from Kelly alone — narrative stays None."""
        tools["conspiracy_score"](GAME)
        tools["narrate"](GAME)
        for row in tools["find_edges"]():
            assert row["narrative"] is None
            assert row["stake"] <= 0.05 * 1000

    def test_ledger_roundtrip_through_tools(self, tools):
        logged = tools["log_bet"](
            game_id=GAME, pick=DET, odds=190, stake=25, closing_odds=150
        )
        assert logged["bet_id"] == 1
        assert logged["status"] == "pending"
        graded = tools["grade_bets"]([{"game_id": GAME, "outcome": "won"}])
        assert graded == [{"game_id": GAME, "outcome": "won", "graded": 1}]
        report = tools["clv_report"]()
        assert report["bets_graded"] == 1
        # decimal 2.9 taken vs decimal 2.5 close = exactly +16%
        assert report["mean_clv_pct"] == pytest.approx(16.0)

    def test_log_bet_rejects_bad_stake(self, tools):
        result = tools["log_bet"](game_id=GAME, pick=DET, odds=190, stake=0)
        assert "error" in result


class TestRunTurnRequestShape:
    def test_model_thinking_output_config_and_cache_order(self, context):
        stub = StubAnthropic(script=[{"text": json.dumps(_report_payload())}])
        run_turn("Where are the edges?", context=context, client=stub)
        kwargs = stub.tool_runner_calls[0]
        assert kwargs["model"] == DEFAULT_AGENT_MODEL
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {
            "effort": "medium",
            "format": SLATE_REPORT_SCHEMA,
        }
        assert kwargs["max_iterations"] == MAX_ITERATIONS
        # Cache prefix: tools → system (voice + guardrails, then snapshot).
        assert [tool.name for tool in kwargs["tools"]] == list(TOOL_NAMES)
        assert kwargs["system"][0]["text"] == HERMES_VOICE + GUARDRAILS
        assert GAME in kwargs["system"][1]["text"]  # today's odds snapshot
        assert "pinnacle" in kwargs["system"][1]["text"]
        # The user's question and timestamp go last, after the cached prefix.
        (user_message,) = kwargs["messages"]
        assert user_message["content"].startswith("Where are the edges?")
        assert "UTC" in user_message["content"]

    def test_system_prompt_carries_guardrails(self, context):
        stub = StubAnthropic(script=[{"text": json.dumps(_report_payload())}])
        run_turn("q", context=context, client=stub)
        system_text = stub.tool_runner_calls[0]["system"][0]["text"]
        assert "21+" in system_text
        assert "never an input to stakes" in system_text
        assert DISCLAIMER in system_text

    def test_agent_model_env_override(self, context, monkeypatch):
        monkeypatch.setenv("AGENT_MODEL", "claude-sonnet-test")
        stub = StubAnthropic(script=[{"text": json.dumps(_report_payload())}])
        run_turn("q", context=context, client=stub)
        assert stub.tool_runner_calls[0]["model"] == "claude-sonnet-test"

    def test_default_model_is_the_plan_default(self, monkeypatch):
        monkeypatch.delenv("AGENT_MODEL", raising=False)
        assert agent_module.agent_model() == "claude-opus-5"

    def test_chat_mode_omits_the_format(self, context):
        stub = StubAnthropic(script=[{"text": "Talk it through, then."}])
        run_turn(
            "Reason through one spot.",
            context=context,
            client=stub,
            response_format=None,
        )
        assert stub.tool_runner_calls[0]["output_config"] == {"effort": "medium"}

    def test_high_effort_for_one_spot_reasoning(self, context):
        stub = StubAnthropic(script=[{"text": json.dumps(_report_payload())}])
        run_turn("q", effort="high", context=context, client=stub)
        assert stub.tool_runner_calls[0]["output_config"]["effort"] == "high"

    def test_invalid_effort_is_rejected(self, context):
        stub = StubAnthropic(script=[])
        with pytest.raises(ValueError, match="effort"):
            run_turn("q", effort="extreme", context=context, client=stub)


class TestRunTurnToolSequence:
    def test_tool_calls_are_routed_and_recorded_in_order(self, context):
        stub = StubAnthropic(
            script=[
                {"tool_use": [("sharp_anchor", {"game_id": GAME})]},
                {
                    "tool_use": [
                        ("find_edges", {}),
                        ("conspiracy_score", {"game_id": GAME}),
                    ]
                },
                {"text": json.dumps(_report_payload())},
            ]
        )
        result = run_turn("Slate report.", context=context, client=stub)
        assert result["tool_calls"] == [
            "sharp_anchor",
            "find_edges",
            "conspiracy_score",
        ]
        # The executed sequence is the real engine, not a stub: each tool's
        # result equals the direct engine output on the same store.
        by_name = {call["name"]: call["result"] for call in stub.executed}
        assert by_name["find_edges"] == engine_find_edges(context.store, None)
        assert by_name["sharp_anchor"]["probs"] == engine_sharp_anchor(
            context.store, GAME
        )["probs"]
        assert by_name["conspiracy_score"]["narrative_only"] is True


class TestRunTurnStructuredOutput:
    def test_valid_payload_round_trips_with_disclaimer(self, context):
        payload = _report_payload()
        stub = StubAnthropic(script=[{"text": json.dumps(payload)}])
        result = run_turn("q", context=context, client=stub)
        assert result["structured"] == payload
        assert result["structured"]["disclaimer"] == DISCLAIMER
        # The payload's numbers are the canned engine numbers.
        (tool_row,) = engine_find_edges(context.store, None)
        (reported,) = result["structured"]["edges"]
        assert reported["stake"] == pytest.approx(tool_row["stake"], abs=0.01)

    def test_empty_disclaimer_is_replaced_with_the_mandatory_line(self, context):
        stub = StubAnthropic(
            script=[{"text": json.dumps(_report_payload(disclaimer="  "))}]
        )
        result = run_turn("q", context=context, client=stub)
        assert result["structured"]["disclaimer"] == DISCLAIMER

    def test_missing_required_key_fails_loudly(self, context):
        payload = _report_payload()
        del payload["edges"]
        stub = StubAnthropic(script=[{"text": json.dumps(payload)}])
        with pytest.raises(ValueError, match="edges"):
            run_turn("q", context=context, client=stub)

    def test_unparseable_structured_output_fails_loudly(self, context):
        stub = StubAnthropic(script=[{"text": "not json at all"}])
        with pytest.raises(ValueError, match="structured output"):
            run_turn("q", context=context, client=stub)

    def test_wrong_typed_edge_field_fails(self, context):
        payload = _report_payload()
        payload["edges"][0]["stake"] = "a lot"
        stub = StubAnthropic(script=[{"text": json.dumps(payload)}])
        with pytest.raises(ValueError, match="stake"):
            run_turn("q", context=context, client=stub)


class TestDisclaimerAlwaysPresent:
    def test_chat_text_gets_disclaimer_appended(self, context):
        stub = StubAnthropic(script=[{"text": "Detroit at +190 is the spot."}])
        result = run_turn(
            "q", context=context, client=stub, response_format=None
        )
        assert result["structured"] is None
        assert result["text"].endswith(DISCLAIMER)

    def test_chat_text_with_disclaimer_is_unchanged(self, context):
        text = f"Take a walk.\n\n{DISCLAIMER}"
        stub = StubAnthropic(script=[{"text": text}])
        result = run_turn(
            "q", context=context, client=stub, response_format=None
        )
        assert result["text"] == text

    def test_ensure_disclaimer_is_idempotent(self):
        once = ensure_disclaimer("hello")
        assert once.endswith(DISCLAIMER)
        assert ensure_disclaimer(once) == once


class TestKeyHandling:
    def test_missing_key_raises_a_clear_error(self, context, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(
            agent_module.MissingAgentKeyError, match="ANTHROPIC_API_KEY"
        ):
            run_turn("q", context=context)

    def test_error_is_not_raised_when_client_is_injected(
        self, context, monkeypatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        stub = StubAnthropic(script=[{"text": json.dumps(_report_payload())}])
        result = run_turn("q", context=context, client=stub)
        assert result["tool_calls"] == []


class TestReportPayloadValidator:
    def test_valid_payload_passes(self):
        validate_report_payload(_report_payload(), SLATE_REPORT_SCHEMA)

    def test_missing_required_key(self):
        with pytest.raises(ValueError, match="headline"):
            validate_report_payload(
                {"disclaimer": DISCLAIMER}, SLATE_REPORT_SCHEMA
            )

    def test_non_object_payload(self):
        with pytest.raises(ValueError, match="object"):
            validate_report_payload([1, 2], SLATE_REPORT_SCHEMA)

    def test_array_item_type_is_checked_with_path(self):
        payload = _report_payload()
        payload["edges"][0]["sharp_prob"] = "high"
        with pytest.raises(ValueError, match=r"payload\.edges\[0\]\.sharp_prob"):
            validate_report_payload(payload, SLATE_REPORT_SCHEMA)


class TestSnapshotFormatting:
    def test_snapshot_text_lists_games_and_prices(self, context):
        text = format_odds_snapshot(context.store)
        assert text.startswith(f"Odds snapshot as of {T2} (UTC)")
        assert GAME in text
        assert "pinnacle: Buffalo Bills -200, Detroit Lions 150" in text

    def test_empty_store_gets_honest_guidance(self, tmp_path, teams_csv):
        ctx = build_context(
            db_path=tmp_path / "empty.sqlite", teams_csv=teams_csv
        )
        try:
            text = format_odds_snapshot(ctx.store)
            assert "No odds snapshots" in text
            assert "do not invent prices" in text
        finally:
            ctx.close()

    def test_latest_slate_keeps_only_the_newest_fetch(self, context):
        slate = latest_slate(context.store.snapshots())
        assert list(slate) == [GAME]
        assert slate[GAME]["fetched_at"] == T2

    def test_slate_rows_reads_the_store(self, context):
        assert [row["game_id"] for row in slate_rows(context.store)] == [GAME]


class TestContextWiring:
    def test_store_and_ledger_share_one_file(self, tmp_path, teams_csv):
        db = tmp_path / "odds.sqlite"
        ctx = build_context(db_path=db, teams_csv=teams_csv)
        try:
            assert ctx.store.path == db
            assert ctx.conspiracy_model() is ctx.conspiracy_model()  # lazy+cached
        finally:
            ctx.close()


class TestAgentClis:
    """Smoke tests mirroring test_cli.py: the entry points run in CI
    offline — stubbed client in-process, no key needed."""

    def test_agent_slate_help_exits_zero(self):
        import agent_slate

        with pytest.raises(SystemExit) as excinfo:
            agent_slate.parse_args(["--help"])
        assert excinfo.value.code == 0

    def test_agent_slate_runs_structured_report_offline(
        self, context, monkeypatch, capsys
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ci-offline-stub")
        monkeypatch.setattr(
            agent_module,
            "_default_client",
            lambda: StubAnthropic(
                script=[{"text": json.dumps(_report_payload())}]
            ),
        )
        import agent_slate

        assert agent_slate.main(["--effort", "medium"]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["disclaimer"] == DISCLAIMER

    def test_agent_slate_without_key_returns_2(
        self, tmp_path, teams_csv, monkeypatch, capsys
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        import agent_slate

        rc = agent_slate.main(
            ["--db", str(tmp_path / "odds.sqlite"), "--teams", str(teams_csv)]
        )
        assert rc == 2
        assert "ANTHROPIC_API_KEY" in capsys.readouterr().err

    def test_agent_chat_replies_with_disclaimer_offline(
        self, context, monkeypatch, capsys
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ci-offline-stub")
        monkeypatch.setattr(
            agent_module,
            "_default_client",
            lambda: StubAnthropic(
                script=[{"text": "Detroit at +190 is the sharp-anchored spot."}]
            ),
        )
        import agent_chat

        with mock.patch(
            "builtins.input", side_effect=["Reason through DET.", "quit"]
        ):
            assert agent_chat.main([]) == 0
        out = capsys.readouterr().out
        assert "Detroit at +190" in out
        assert DISCLAIMER in out

    def test_agent_chat_without_key_returns_2(
        self, tmp_path, teams_csv, monkeypatch, capsys
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        import agent_chat

        rc = agent_chat.main(
            ["--db", str(tmp_path / "odds.sqlite"), "--teams", str(teams_csv)]
        )
        assert rc == 2
        assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
