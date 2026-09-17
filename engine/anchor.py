"""Sharp-book anchoring, +EV edge finding, and line movement.

The anchor is the devigged no-vig probability from sharp books only
(Pinnacle, Circa by default). It is the product's source of truth: when
no sharp book quotes both sides of a game the anchor is *unavailable*,
and soft books are never substituted — a soft consensus would silently
replace the market's sharpest prices and overstate every edge built on
it. Edges compare the sharp anchor against the best soft-book price and
size with fractional Kelly.

All reads come from the append-only snapshot store; nothing here touches
the network or spends API credits. ``client`` exists on ``find_edges``
for the agent tool surface, which passes its shared client through.
"""
from __future__ import annotations

from typing import TypedDict

from .kelly import kelly_stake
from .odds import H2H_MARKET, OddsClient, OddsRow
from .pricing import moneyline_to_implied, no_vig_pair
from .store import OddsStore

DEFAULT_SHARP_BOOKS = ("pinnacle", "circa")


class AnchorUnavailable(RuntimeError):
    """No sharp book quotes both sides of the game — the anchor is unknown.

    Raised instead of degrading to soft-book consensus on purpose: a soft
    anchor would overstate every edge computed from it.
    """


class Anchor(TypedDict):
    """Return shape of :func:`sharp_anchor`."""

    game_id: str
    market: str
    books: tuple[str, ...]  # sharp books actually used
    probs: dict[str, float]  # outcome_name -> devigged probability


class EdgeRow(TypedDict):
    """One ranked +EV row, per the approved spec."""

    game_id: str
    pick: str
    price: float  # best soft-book American price for the pick
    sharp_prob: float
    fair_odds: int  # American odds at sharp_prob, zero vig
    edge_pct: float  # (sharp_prob - implied(price)) * 100
    stake: float  # fractional Kelly dollars
    books: list[str]  # soft books offering the best price
    narrative: dict[str, object] | None  # conspiracy output, labelled only


class LineMove(TypedDict):
    """Open→now price movement for one side."""

    open: float | None
    now: float | None
    delta: float | None


def _require_h2h(market: str) -> None:
    """The store persists h2h rows only (parse_events drops everything else)."""
    if market != H2H_MARKET:
        raise ValueError(
            f"unsupported market {market!r}: the store persists"
            f" {H2H_MARKET!r} rows only"
        )


def _latest_per_game(
    rows: list[OddsRow],
) -> dict[str, dict[str, dict[str, float]]]:
    """Group rows by game, keeping each game's newest fetch only.

    Returns ``game_id -> bookmaker -> {outcome_name: price}``. fetched_at
    is ISO 8601 UTC, so lexicographic comparison is chronological.
    """
    newest: dict[str, str] = {}
    for row in rows:
        seen = newest.get(row.game_id)
        if seen is None or row.fetched_at > seen:
            newest[row.game_id] = row.fetched_at
    latest: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row.fetched_at != newest[row.game_id]:
            continue
        sides = latest.setdefault(row.game_id, {}).setdefault(row.bookmaker, {})
        sides[row.outcome_name] = row.price
    return latest


def _anchor_from_books(
    book_sides: dict[str, dict[str, float]],
    sharp_books: tuple[str, ...],
    game_id: str,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Average the no-vig pair across sharp books that quote both sides.

    Pure: no store, no I/O. Books quoting anything but exactly two sides
    (incomplete feeds, three-way soccer markets) cannot form a two-sided
    no-vig pair and are skipped.
    """
    prob_sums: dict[str, float] = {}
    books_used: list[str] = []
    for book in sharp_books:
        sides = book_sides.get(book)
        if sides is None or len(sides) != 2:
            continue
        first, second = sorted(sides)
        pa, pb = no_vig_pair(sides[first], sides[second])
        prob_sums[first] = prob_sums.get(first, 0.0) + pa
        prob_sums[second] = prob_sums.get(second, 0.0) + pb
        books_used.append(book)
    if not books_used:
        raise AnchorUnavailable(
            f"no sharp book ({', '.join(sharp_books) or 'none configured'})"
            f" quotes both sides of {game_id!r} at the latest snapshot;"
            " soft-book prices are never used to anchor"
        )
    n = len(books_used)
    probs = {name: total / n for name, total in prob_sums.items()}
    return tuple(books_used), probs


def sharp_anchor(
    store: OddsStore,
    game_id: str,
    sharp_books: tuple[str, ...] = DEFAULT_SHARP_BOOKS,
    market: str = H2H_MARKET,
) -> Anchor:
    """Devigged sharp-book probability for both sides of a game.

    Averages the no-vig pair across available sharp books so one book's
    stale number can't own the anchor. Raises AnchorUnavailable if no
    sharp book has both sides — never falls back to soft books.
    """
    _require_h2h(market)
    rows = store.snapshots(game_id=game_id)
    if not rows:
        raise AnchorUnavailable(
            f"no stored snapshots for game {game_id!r}; run the snapshot"
            " collector first"
        )
    book_sides = _latest_per_game(rows).get(game_id, {})
    books, probs = _anchor_from_books(book_sides, sharp_books, game_id)
    return {"game_id": game_id, "market": market, "books": books, "probs": probs}


def _decimal_payout(price: float) -> float:
    """Total return per unit staked at American ``price``."""
    b = price / 100.0 if price > 0.0 else 100.0 / abs(price)
    return 1.0 + b


def _fair_american_odds(prob: float) -> int:
    """American odds carrying zero vig at ``prob`` — the fair price."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"no finite fair price for probability {prob}")
    if prob >= 0.5:
        return round(-100.0 * prob / (1.0 - prob))
    return round(100.0 * (1.0 - prob) / prob)


def find_edges(
    store: OddsStore,
    client: OddsClient | None,
    *,
    min_edge: float = 0.02,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
) -> list[EdgeRow]:
    """Ranked +EV list: for every game, sharp no-vig prob minus the
    best soft-book price; stake = fractional Kelly off the true edge.

    Rows follow the spec shape:
    {"game_id": "det-buf-2026-09-21", "pick": "Detroit Lions",
     "price": 118, "sharp_prob": 0.571, "fair_odds": -133,
     "edge_pct": 4.8, "stake": 21.5, "books": ["fanduel"],
     "narrative": None}

    ``narrative`` is reserved for the agent layer's conspiracy output —
    a clearly labelled field that is never an input to stake. Edges are
    computed purely from stored snapshots: ``client`` is accepted for the
    tool surface and never used, so a sweep never spends API credits.
    Games without a sharp anchor are skipped — they have no product edge,
    and inventing one from soft consensus is exactly the failure mode
    AnchorUnavailable exists to prevent.
    """
    edges: list[EdgeRow] = []
    sharp_set = frozenset(DEFAULT_SHARP_BOOKS)
    latest = _latest_per_game(store.snapshots())
    for game_id in sorted(latest):
        book_sides = latest[game_id]
        try:
            _, probs = _anchor_from_books(book_sides, DEFAULT_SHARP_BOOKS, game_id)
        except AnchorUnavailable:
            continue
        for pick in sorted(probs):
            prob = probs[pick]
            prices = {
                book: sides[pick]
                for book, sides in book_sides.items()
                if book not in sharp_set and pick in sides
            }
            if not prices:
                continue
            best_price = max(prices.values(), key=_decimal_payout)
            edge = prob - moneyline_to_implied(best_price)
            if edge < min_edge:
                continue
            edges.append(
                EdgeRow(
                    game_id=game_id,
                    pick=pick,
                    price=best_price,
                    sharp_prob=prob,
                    fair_odds=_fair_american_odds(prob),
                    edge_pct=edge * 100.0,
                    stake=kelly_stake(
                        prob, best_price, bankroll, fraction=kelly_fraction
                    ),
                    books=sorted(
                        book for book, price in prices.items() if price == best_price
                    ),
                    narrative=None,
                )
            )
    edges.sort(key=lambda row: (-row["edge_pct"], row["game_id"], row["pick"]))
    return edges


def line_move(
    store: OddsStore,
    game_id: str,
    *,
    book: str | None = None,
    market: str = H2H_MARKET,
) -> dict[str, LineMove]:
    """Open→now price movement per side, from stored snapshots.

    ``open`` is the price at the game's earliest snapshot, ``now`` at the
    latest; ``delta`` is ``now - open`` in American price points. With
    ``book=None`` prices are averaged across bookmakers at each endpoint
    (a late-joining book pulls the average); pass a bookmaker key to
    track one book. Sides missing an endpoint report None — movement is
    never guessed from a single observation. Unknown games return {}.
    """
    _require_h2h(market)
    rows = store.snapshots(game_id=game_id)
    if not rows:
        return {}

    def prices_at(timestamp: str) -> dict[str, float]:
        pooled: dict[str, list[float]] = {}
        for row in rows:
            if row.fetched_at != timestamp:
                continue
            if book is not None and row.bookmaker != book:
                continue
            pooled.setdefault(row.outcome_name, []).append(row.price)
        return {name: sum(prices) / len(prices) for name, prices in pooled.items()}

    timestamps = sorted({row.fetched_at for row in rows})
    opening = prices_at(timestamps[0])
    current = prices_at(timestamps[-1])
    movement: dict[str, LineMove] = {}
    for name in sorted(set(opening) | set(current)):
        open_price = opening.get(name)
        now_price = current.get(name)
        delta = (
            now_price - open_price
            if open_price is not None and now_price is not None
            else None
        )
        movement[name] = {"open": open_price, "now": now_price, "delta": delta}
    return movement
