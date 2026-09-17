"""Offline tests for the Odds API client: parsing, retries, credit headers.

All HTTP is respx-mocked (or served by an explicit ``httpx.MockTransport``);
no test touches the network and no test needs a real API key.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from engine.odds import (
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    MissingAPIKeyError,
    OddsAPIError,
    OddsClient,
    RetryPolicy,
    parse_events,
)

ODDS_URL = f"{DEFAULT_BASE_URL}/sports/americanfootball_nfl/odds/"
CREDIT_HEADERS = {"x-requests-remaining": "499", "x-requests-used": "1"}

FIXTURE_EVENTS = [
    {
        "id": "evt-lions-bills",
        "sport_key": "americanfootball_nfl",
        "sport_title": "NFL",
        "commence_time": "2026-09-21T17:00:00Z",
        "home_team": "Detroit Lions",
        "away_team": "Buffalo Bills",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": "2026-09-17T18:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2026-09-17T18:00:00Z",
                        "outcomes": [
                            {"name": "Detroit Lions", "price": -125},
                            {"name": "Buffalo Bills", "price": 105},
                        ],
                    },
                    {
                        "key": "spreads",
                        "last_update": "2026-09-17T18:00:00Z",
                        "outcomes": [
                            {"name": "Detroit Lions", "price": -110, "point": 4.5},
                        ],
                    },
                ],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "last_update": "2026-09-17T17:55:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2026-09-17T17:55:00Z",
                        "outcomes": [
                            {"name": "Detroit Lions", "price": -120},
                            {"name": "Buffalo Bills", "price": 100},
                        ],
                    },
                ],
            },
        ],
    },
    {
        # No bookmakers yet — contributes zero rows, must not crash.
        "id": "evt-no-books",
        "sport_key": "americanfootball_nfl",
        "sport_title": "NFL",
        "commence_time": "2026-09-22T01:15:00Z",
        "home_team": "Denver Broncos",
        "away_team": "Las Vegas Raiders",
        "bookmakers": [],
    },
]


def make_client(**overrides) -> OddsClient:
    kwargs = {"api_key": "test-key", "sleep_fn": lambda _seconds: None}
    kwargs.update(overrides)
    return OddsClient(**kwargs)


class TestParseEvents:
    def test_h2h_rows_are_normalized_one_row_per_side(self):
        rows = parse_events(
            FIXTURE_EVENTS, sport_key="americanfootball_nfl", fetched_at="t0"
        )
        # 2 books x 2 sides; spreads ignored, no-books event empty.
        assert len(rows) == 4
        first = rows[0]
        assert (first.fetched_at, first.game_id, first.commence_time) == (
            "t0",
            "evt-lions-bills",
            "2026-09-21T17:00:00Z",
        )
        assert (first.bookmaker, first.outcome_name, first.price) == (
            "pinnacle",
            "Detroit Lions",
            -125.0,
        )
        assert first.book_updated_at == "2026-09-17T18:00:00Z"
        # The spreads market never becomes a row.
        assert not any(
            row.bookmaker == "pinnacle" and row.price == -110.0 for row in rows
        )

    def test_non_h2h_markets_are_skipped_explicitly(self):
        rows = parse_events(
            FIXTURE_EVENTS, sport_key="americanfootball_nfl", fetched_at="t0"
        )
        fanduel_rows = [row for row in rows if row.bookmaker == "fanduel"]
        assert [row.outcome_name for row in fanduel_rows] == [
            "Detroit Lions",
            "Buffalo Bills",
        ]


class TestFetchH2H:
    @respx.mock
    def test_fetch_parses_expected_rows_and_reports_credits(self):
        route = respx.get(ODDS_URL).mock(
            return_value=httpx.Response(200, json=FIXTURE_EVENTS, headers=CREDIT_HEADERS)
        )
        client = make_client()
        result = client.fetch_h2h("americanfootball_nfl")

        assert route.call_count == 1
        assert result.events == 2
        assert len(result.rows) == 4
        assert result.requests_remaining == 499.0
        assert result.requests_used == 1.0
        # Every row of one fetch shares its fetched_at timestamp.
        assert {row.fetched_at for row in result.rows} == {result.fetched_at}

    @respx.mock
    def test_request_carries_key_regions_market_and_books_filter(self):
        route = respx.get(ODDS_URL).mock(
            return_value=httpx.Response(200, json=[], headers=CREDIT_HEADERS)
        )
        client = make_client()
        client.fetch_h2h("americanfootball_nfl", books=["pinnacle", "fanduel"])

        sent = route.calls.last.request.url.params
        assert sent["apiKey"] == "test-key"
        assert sent["regions"] == "us"
        assert sent["markets"] == "h2h"
        assert sent["bookmakers"] == "pinnacle,fanduel"

    @respx.mock
    def test_credit_headers_missing_surfaced_as_none(self):
        respx.get(ODDS_URL).mock(return_value=httpx.Response(200, json=FIXTURE_EVENTS))
        client = make_client()
        result = client.fetch_h2h("americanfootball_nfl")
        assert result.requests_remaining is None
        assert result.requests_used is None


class TestRetryAndErrors:
    @respx.mock
    def test_retry_on_http_500_then_success(self):
        sleeps: list[float] = []
        route = respx.get(ODDS_URL).mock(
            side_effect=[
                httpx.Response(500, text="boom"),
                httpx.Response(200, json=FIXTURE_EVENTS, headers=CREDIT_HEADERS),
            ]
        )
        client = make_client(sleep_fn=sleeps.append)
        result = client.fetch_h2h("americanfootball_nfl")

        assert route.call_count == 2
        assert result.events == 2
        assert sleeps == [0.5]  # one backoff, base delay, before the retry

    @respx.mock
    def test_retries_exhausted_raises_odds_api_error(self):
        route = respx.get(ODDS_URL).mock(
            return_value=httpx.Response(503, text="unavailable")
        )
        client = make_client(retry=RetryPolicy(max_attempts=3))
        with pytest.raises(OddsAPIError, match="failed after 3 attempts.*503"):
            client.fetch_h2h("americanfootball_nfl")
        assert route.call_count == 3

    @respx.mock
    def test_non_retryable_404_fails_immediately(self):
        route = respx.get(ODDS_URL).mock(
            return_value=httpx.Response(404, text="unknown sport")
        )
        client = make_client()
        with pytest.raises(OddsAPIError, match="404"):
            client.fetch_h2h("americanfootball_nfl")
        assert route.call_count == 1

    @respx.mock
    def test_429_honors_retry_after_header(self):
        sleeps: list[float] = []
        route = respx.get(ODDS_URL).mock(
            side_effect=[
                httpx.Response(429, text="slow down", headers={"Retry-After": "7"}),
                httpx.Response(200, json=[], headers=CREDIT_HEADERS),
            ]
        )
        client = make_client(sleep_fn=sleeps.append)
        client.fetch_h2h("americanfootball_nfl")

        assert route.call_count == 2
        assert sleeps == [7.0]  # server's Retry-After beats the base backoff

    @respx.mock
    def test_transport_error_is_retried(self):
        route = respx.get(ODDS_URL).mock(
            side_effect=[
                httpx.ConnectError("connection refused"),
                httpx.Response(200, json=FIXTURE_EVENTS, headers=CREDIT_HEADERS),
            ]
        )
        client = make_client()
        result = client.fetch_h2h("americanfootball_nfl")
        assert route.call_count == 2
        assert result.events == 2
        assert len(result.rows) == 4


class TestClientConfiguration:
    def test_missing_api_key_raises_before_any_request(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        with pytest.raises(MissingAPIKeyError, match=API_KEY_ENV):
            OddsClient()

    def test_injectable_transport_serves_the_request(self):
        # No respx here: a custom transport alone must be enough to work
        # offline, proving the client's transport seam.
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["apiKey"] == "injected-key"
            return httpx.Response(200, json=FIXTURE_EVENTS, headers=CREDIT_HEADERS)

        client = make_client(
            api_key="injected-key", transport=httpx.MockTransport(handler)
        )
        result = client.fetch_h2h("americanfootball_nfl")
        assert len(result.rows) == 4
        assert result.requests_remaining == 499.0
        client.close()
