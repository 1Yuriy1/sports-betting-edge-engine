"""The Odds API client: h2h moneylines with retry/backoff and credit tracking.

The HTTP transport is injectable (``httpx.BaseTransport``) so every test runs
offline. The API key comes from the ``ODDS_API_KEY`` environment variable and
is never hardcoded.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Self

import httpx

DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "us"
H2H_MARKET = "h2h"
API_KEY_ENV = "ODDS_API_KEY"

# Burn a retry only on our own network trouble or the API's transient
# refusals (rate limits, blips). Everything else fails fast and loud.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class OddsAPIError(RuntimeError):
    """An Odds API request failed after retries, or with a fatal status."""


class MissingAPIKeyError(OddsAPIError):
    """``ODDS_API_KEY`` is not set; credentials are never guessed or hardcoded."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.5  # seconds; doubles with each retry


@dataclass(frozen=True)
class OddsRow:
    """One normalized moneyline price: one side, one book, one fetch."""

    fetched_at: str  # ISO 8601 UTC, shared by every row of one fetch
    sport_key: str  # e.g. 'americanfootball_nfl'
    game_id: str  # The Odds API event id
    commence_time: str  # ISO 8601 UTC
    bookmaker: str  # e.g. 'pinnacle'
    outcome_name: str  # team name; one row per side
    price: float  # American odds, e.g. -125 or 105
    book_updated_at: str | None  # bookmaker's own last_update, when reported


@dataclass(frozen=True)
class FetchResult:
    """Parsed fetch: normalized rows plus the credit headers the API reported."""

    rows: tuple[OddsRow, ...]
    events: int  # games seen in the response
    fetched_at: str
    requests_remaining: float | None  # x-requests-remaining: monthly credits left
    requests_used: float | None  # x-requests-used: credits spent this month


def utc_now_iso() -> str:
    """Current UTC time as ISO 8601 with explicit offset — the store's clock."""
    return datetime.now(timezone.utc).isoformat()


def parse_events(
    events: Iterable[dict], *, sport_key: str, fetched_at: str
) -> tuple[OddsRow, ...]:
    """Flatten an odds response into normalized h2h rows.

    Only h2h markets are stored (spreads/totals are ignored). Parsing is
    strict: a malformed response raises instead of silently storing a
    partial slate — a snapshot collector must fail visibly.
    """
    rows: list[OddsRow] = []
    for event in events:
        game_id = event["id"]
        commence_time = event["commence_time"]
        event_sport = event.get("sport_key", sport_key)
        for bookmaker in event.get("bookmakers", []):
            bookmaker_key = bookmaker["key"]
            book_updated_at = bookmaker.get("last_update")
            for market in bookmaker.get("markets", []):
                if market["key"] != H2H_MARKET:
                    continue
                for outcome in market["outcomes"]:
                    rows.append(
                        OddsRow(
                            fetched_at=fetched_at,
                            sport_key=event_sport,
                            game_id=game_id,
                            commence_time=commence_time,
                            bookmaker=bookmaker_key,
                            outcome_name=outcome["name"],
                            price=float(outcome["price"]),
                            book_updated_at=book_updated_at,
                        )
                    )
    return tuple(rows)


class OddsClient:
    """Minimal The Odds API client for scheduled h2h snapshotting."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        retry: RetryPolicy | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise MissingAPIKeyError(
                f"set {API_KEY_ENV} to your The Odds API key"
                " (https://the-odds-api.com); it is never hardcoded"
            )
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._retry = retry if retry is not None else RetryPolicy()
        # Late-bound so tests (and callers) can substitute a recorder.
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep
        self._http = httpx.Client(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch_h2h(
        self,
        sport_key: str,
        *,
        regions: str = DEFAULT_REGIONS,
        books: Iterable[str] | None = None,
    ) -> FetchResult:
        """Fetch current h2h odds for a sport and parse them into rows."""
        params = {
            "apiKey": self._api_key,
            "regions": regions,
            "markets": H2H_MARKET,
        }
        bookmakers = ",".join(books) if books else None
        if bookmakers:
            params["bookmakers"] = bookmakers
        fetched_at = utc_now_iso()
        response = self._get_with_retry(f"/sports/{sport_key}/odds/", params=params)
        events = response.json()
        rows = parse_events(events, sport_key=sport_key, fetched_at=fetched_at)
        return FetchResult(
            rows=rows,
            events=len(events),
            fetched_at=fetched_at,
            requests_remaining=_header_float(response, "x-requests-remaining"),
            requests_used=_header_float(response, "x-requests-used"),
        )

    def _get_with_retry(self, path: str, *, params: dict[str, str]) -> httpx.Response:
        """GET with exponential backoff on 5xx/429 and transport errors."""
        last_error = ""
        for attempt in range(self._retry.max_attempts):
            try:
                response = self._http.get(path, params=params)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
            else:
                if response.status_code not in RETRYABLE_STATUSES:
                    if response.is_error:
                        raise OddsAPIError(
                            f"The Odds API returned HTTP {response.status_code}:"
                            f" {response.text[:200]}"
                        )
                    return response
                last_error = f"HTTP {response.status_code}"
                # Sleep only when another attempt remains; a final failure
                # should raise immediately, not idle.
                if attempt + 1 < self._retry.max_attempts:
                    self._sleep(self._retry_delay(response, attempt))
        raise OddsAPIError(
            f"The Odds API request failed after {self._retry.max_attempts}"
            f" attempts ({last_error})"
        )

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Exponential backoff, stretched to a server-provided Retry-After."""
        delay = self._retry.base_delay * (2**attempt)
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass  # HTTP-date form; plain backoff is the safe fallback
        return delay


def _header_float(response: httpx.Response, name: str) -> float | None:
    """Read a credit header as float; unparseable/missing means unknown."""
    raw = response.headers.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None  # the API reported something we cannot interpret
