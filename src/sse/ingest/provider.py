"""Yahoo Finance price provider.

Design points that matter downstream:

* **Explicit adjustment flags.**  ``auto_adjust=False`` keeps raw OHLC *and*
  ``Adj Close`` in the same frame, and ``actions=True`` adds dividends and
  splits.  Storing both raw and adjusted is what allows the research layer to
  compute returns on adjusted prices (roadmap §3) while the API can still serve
  the traded price a backtest needs for fills.
* **End dates are inclusive at every level above this module.**  Yahoo's
  ``end`` is exclusive; the +1 day is added here, once, and never leaks upward.
* **Failures are values, not exceptions.**  ``fetch_many`` yields a
  :class:`FetchResult` per ticker so the caller can persist a failed-ticker list
  and retry it, rather than losing a six-hour backfill to one bad symbol.
* **Bounded concurrency plus a token bucket.**  Yahoo rate-limits aggressively
  and responds to abuse with hour-long 429s, which would cost more time than
  the politeness saves.

The :class:`PriceProvider` protocol exists so a second source (AkShare, an SSE
vendor feed) can be added in Task 2/3 without touching the backfill script.
"""

from __future__ import annotations

import datetime as dt
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Protocol, Sequence

from ..codec import BarSeries
from ..config import get_settings
from ..logging_setup import setup_logging

logger = setup_logging(__name__)

_RATE_LIMIT_MARKERS = ("429", "too many requests", "rate limit", "temporarily unavailable")


@dataclass
class FetchResult:
    ticker: str
    bars: BarSeries | None = None
    error: str | None = None
    attempts: int = 0
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def n_bars(self) -> int:
        return len(self.bars) if self.bars is not None else 0


class PriceProvider(Protocol):
    def fetch(self, ticker: str, start: dt.date, end: dt.date) -> BarSeries: ...


class RateLimiter:
    """Thread-safe sliding-window limiter shared by every worker thread."""

    def __init__(self, per_minute: int) -> None:
        self._per_minute = max(1, per_minute)
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0] > 60.0:
                    self._events.popleft()
                if len(self._events) < self._per_minute:
                    self._events.append(now)
                    return
                wait = 60.0 - (now - self._events[0]) + 0.01
            time.sleep(max(wait, 0.01))


class YahooPriceProvider:
    """Daily bars from Yahoo Finance via ``yfinance``."""

    def __init__(
        self,
        *,
        rate_limiter: RateLimiter | None = None,
        repair: bool = False,
        settings: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.limiter = rate_limiter or RateLimiter(self.settings.requests_per_minute)
        #: yfinance's `repair` heuristically fixes 100x unit errors and bad splits.
        #: Off by default: silently rewriting vendor data would undermine the
        #: quality report, which is meant to *measure* Yahoo, not launder it.
        self.repair = repair

    # -- single ticket ------------------------------------------------------

    def fetch(self, ticker: str, start: dt.date, end: dt.date) -> BarSeries:
        """Fetch ``[start, end]`` inclusive.  Raises on unrecoverable failure."""
        import yfinance as yf  # imported lazily: the API process never needs it

        exclusive_end = end + dt.timedelta(days=1)
        last_error: Exception | None = None

        for attempt in range(1, self.settings.max_retries + 2):
            self.limiter.acquire()
            try:
                frame = yf.Ticker(ticker).history(
                    start=start.isoformat(),
                    end=exclusive_end.isoformat(),
                    interval="1d",
                    auto_adjust=False,   # keep raw OHLC alongside Adj Close
                    back_adjust=False,
                    actions=True,        # dividends and splits
                    repair=self.repair,
                    timeout=self.settings.request_timeout,
                    raise_errors=True,
                )
                return BarSeries.from_frame(frame, ticker=ticker)
            except Exception as exc:  # noqa: BLE001 - provider errors are heterogeneous
                last_error = exc
                if attempt > self.settings.max_retries:
                    break
                time.sleep(self._backoff(attempt, exc))

        raise RuntimeError(f"{ticker}: {last_error}") from last_error

    def _backoff(self, attempt: int, exc: Exception) -> float:
        """Exponential backoff with jitter; rate-limit responses wait longer."""
        message = str(exc).lower()
        base = self.settings.backoff_base ** attempt
        if any(marker in message for marker in _RATE_LIMIT_MARKERS):
            base *= 4
        delay = min(base, self.settings.backoff_cap)
        return delay * (0.5 + random.random())  # jitter: avoid thundering herd

    # -- cross-section ------------------------------------------------------

    def fetch_many(
        self,
        tickers: Sequence[str],
        start: dt.date,
        end: dt.date,
        *,
        max_workers: int | None = None,
    ) -> Iterator[FetchResult]:
        """Yield results as they complete.  Never raises for a single ticker."""
        workers = max_workers or self.settings.max_workers
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="yf") as pool:
            futures = {pool.submit(self._fetch_result, t, start, end): t for t in tickers}
            for future in as_completed(futures):
                yield future.result()

    def _fetch_result(self, ticker: str, start: dt.date, end: dt.date) -> FetchResult:
        started = time.monotonic()
        try:
            bars = self.fetch(ticker, start, end)
            return FetchResult(
                ticker=ticker, bars=bars, attempts=1, elapsed=time.monotonic() - started
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch failed for %s: %s", ticker, exc)
            return FetchResult(
                ticker=ticker,
                error=str(exc)[:500],
                attempts=self.settings.max_retries + 1,
                elapsed=time.monotonic() - started,
            )


def symbol_for(code: str | int, suffix: str = ".SS") -> str:
    """``600000`` -> ``600000.SS``.  SSE codes are always six digits."""
    return f"{int(code):06d}{suffix}"


#: SSE code blocks are allocated by the exchange, so the first three digits
#: identify the *instrument type* deterministically — no vendor metadata
#: required.  This matters because Yahoo's screener answers "exchange == SHH"
#: with every listed instrument, not just common shares: ETFs, LOFs, REITs,
#: convertible bonds and repos all carry six-digit codes ending in ``.SS`` and
#: are indistinguishable from equities by symbol shape alone.
INSTRUMENT_BLOCKS: dict[str, str] = {
    # common shares
    "600": "equity_main", "601": "equity_main", "603": "equity_main", "605": "equity_main",
    "688": "equity_star", "689": "equity_star",
    "900": "equity_b_share",
    # funds
    "500": "fund_closed", "501": "fund_lof", "502": "fund_structured",
    "505": "fund_lof", "506": "fund_lof", "507": "fund_lof",
    "508": "reit",
    "510": "etf", "511": "etf_bond", "512": "etf", "513": "etf_qdii", "515": "etf",
    "516": "etf", "517": "etf", "518": "etf_gold", "588": "etf_star",
    "560": "etf", "561": "etf", "562": "etf", "563": "etf",
    # debt and money market
    "110": "bond_convertible", "111": "bond_convertible", "113": "bond_convertible",
    "118": "bond_convertible",
    "010": "bond_treasury", "018": "bond_treasury", "019": "bond_treasury", "020": "bond_treasury",
    "122": "bond_corporate", "124": "bond_corporate", "126": "bond_corporate",
    "127": "bond_corporate", "128": "bond_corporate", "132": "bond_corporate",
    "136": "bond_corporate", "163": "bond_corporate", "175": "bond_corporate",
    "204": "repo",
    # 000001.SS is the SSE Composite Index, not a security
    "000": "index",
}

#: Instrument types that belong in the security master.
EQUITY_INSTRUMENTS: frozenset[str] = frozenset(
    {"equity_main", "equity_star", "equity_b_share"}
)


def classify_code(code: str | int) -> str:
    """Instrument type for an SSE code, or ``'unknown'`` for an unallocated block."""
    return INSTRUMENT_BLOCKS.get(f"{int(code):06d}"[:3], "unknown")


def is_equity(code: str | int, quote_type: str | None = None) -> bool:
    """Is this code a common share?

    **Whitelist only.**  The SSE equity code space is fixed by the exchange —
    600/601/603/605 main board, 688/689 STAR, 900 B-share — and nothing else on
    this exchange is a common share.

    An earlier version fell back to Yahoo's ``quoteType`` for unrecognised
    blocks, on the theory that a newly allocated code range should not be
    silently dropped.  That was the wrong trade.  Measured on the 12 Aug run,
    the fallback admitted **78 instruments** from newer fund blocks that Yahoo
    labels ``EQUITY``; Yahoo's ``quoteType`` is simply unreliable for Chinese
    funds.  The hypothetical cost of missing a future equity block is now paid
    instead by :func:`unknown_blocks`, which reports unrecognised codes loudly
    so a human notices — excluded, but never invisible.

    ``quote_type`` is retained in the signature (and ignored) so that callers
    and stored documents need no change.
    """
    return classify_code(code) in EQUITY_INSTRUMENTS


def unknown_blocks(codes: "Iterable[str | int]") -> dict[str, list[str]]:
    """Group codes whose block this table does not recognise, by 3-digit prefix.

    A non-empty result means either the exchange allocated a new range or the
    table is stale.  Either way it is a fact for a human to check, not something
    to resolve by guessing from vendor metadata.
    """
    out: dict[str, list[str]] = {}
    for code in codes:
        text = f"{int(str(code).split('.')[0]):06d}"
        if classify_code(text) == "unknown":
            out.setdefault(text[:3], []).append(text)
    return {k: sorted(v) for k, v in sorted(out.items())}


def board_of(code: str) -> str:
    """Classify an SSE code by board — drives price limits and the research filter."""
    prefix = str(code)[:3]
    if prefix in {"600", "601", "603", "605"}:
        return "main"
    if prefix in {"688", "689"}:
        return "star"
    if prefix == "900":
        return "b_share"
    return "other"
