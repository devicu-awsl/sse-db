"""Universe discovery for the Shanghai Stock Exchange.

The problem
-----------
Yahoo has no "list every SSE listing" endpoint.  The screener will answer
``exchange == SHH``, but it stops honouring ``offset`` after a few hundred rows
and truncates **silently** — a flat paged query returns perhaps 1,000 of ~2,300
names and looks entirely successful.  Building a universe on that would quietly
drop a third of the market, and every correlation, cluster and backtest number
downstream would inherit the omission without a single error being raised.

The approach (roadmap §2, conflict 2 — adopt both, in this order)
----------------------------------------------------------------
1. **Flat screen.**  Cheap; establishes a baseline and reveals the truncation.
2. **Saturation detection + band bisection.**  When a query hits the paging
   ceiling it is declared *saturated*, split into two market-cap bands, and each
   half re-screened.  Recursion continues until every leaf band returns fewer
   rows than the ceiling, so no band is ever truncated.  Market cap is the
   partition key because it is close to continuous and Yahoo sorts on it
   reliably.
3. **Code-range probe.**  Independent evidence, not merely a bigger screen: SSE
   codes are allocated in known blocks (600/601/603/605 main board, 688/689
   STAR, 900 B-share), so probing the code space finds listings the screener
   never mentions.  Agreement between two unrelated methods is the actual
   diligence claim on the universe slide; disagreement is a finding worth
   reporting either way.
4. **Union across snapshots.**  A name that vanishes from a later screen is
   marked inactive, never deleted (see README, survivorship bias).

Every function that talks to Yahoo is injectable, so the whole search strategy
is unit-testable without the network — and a yfinance API change is a one-line
repair rather than a rewrite.
"""

from __future__ import annotations

import datetime as dt
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from ..config import get_settings
from ..logging_setup import setup_logging
from .provider import (RateLimiter, board_of, classify_code, is_equity,
                       symbol_for, unknown_blocks)

logger = setup_logging(__name__)


class ProbeOutcome(str, Enum):
    """Tri-state result of a code probe."""

    PRESENT = "present"    # Yahoo returned bars: the listing exists
    ABSENT = "absent"      # Yahoo answered and has no such symbol
    UNKNOWN = "unknown"    # Yahoo did not answer; absence is NOT established


#: Message fragments that mean "Yahoo answered: no such listing".
_ABSENT_MARKERS = ("possibly delisted", "no price data found", "no data found",
                   "symbol may be delisted", "404")
#: Message fragments that mean "the request itself failed".
_TRANSIENT_MARKERS = ("timeout", "timed out", "connection", "failed to perform",
                      "too many requests", "429", "500", "502", "503", "504",
                      "temporarily unavailable", "ssl", "reset by peer", "curl")

#: (query, offset, size) -> raw Yahoo screener response
ScreenerFn = Callable[[Any, int, int], Mapping[str, Any]]
#: symbol -> ProbeOutcome (a plain bool is accepted for backwards compatibility)
ProbeFn = Callable[[str], "ProbeOutcome | bool"]

_CODE_RE = re.compile(r"^(\d{6})\.(SS|SH)$", re.IGNORECASE)
_MARKET_CAP_CEILING = 5e12  # CNY; comfortably above the largest A-share
#: Below this, market cap has stopped being a usable partition key: Yahoo
#: reports no market cap at all for most funds and for freshly-listed shares,
#: and every one of them lands in the same bottom band.  Observed in the
#: 2026-08-11 run: the band [0, 1.0] returned the paging ceiling at every one of
#: ten successive bisections, because a geometric split of [0, h] can never
#: escape 1.0.  Such a band is declared unresolvable-by-cap and handed to the
#: code-range probe, which does not depend on any vendor metadata.
_MIN_RESOLVABLE_CAP = 1e6
#: A band whose bounds differ by less than this ratio can no longer separate
#: anything, whatever the population.
_MIN_BAND_RATIO = 1.01


@dataclass
class Band:
    """One market-cap interval in the bisection search."""

    low: float
    high: float | None
    depth: int = 0
    n_returned: int = 0
    saturated: bool = False
    unresolvable: str | None = None

    def midpoint(self) -> float:
        """Geometric midpoint — market cap spans six orders of magnitude."""
        high = self.high if self.high is not None else _MARKET_CAP_CEILING
        low = max(self.low, 1.0)
        return (low * high) ** 0.5

    def splittable(self) -> str | None:
        """Reason this band cannot usefully be split, or ``None`` if it can.

        Note what is *not* here: a rule comparing a child's row count to its
        parent's.  That was tried and rejected — when a band is truncated at the
        paging ceiling, a child hitting the same ceiling is not evidence of a
        failed split, and abandoning on that basis loses real names.  Only
        properties of the interval itself are used.
        """
        if self.high is not None and self.high <= _MIN_RESOLVABLE_CAP:
            return "no_reported_market_cap"
        if self.high is not None and self.low > 0 and self.high / self.low < _MIN_BAND_RATIO:
            return "interval_exhausted"
        if self.high is not None and self.midpoint() <= self.low * 1.000001:
            return "interval_exhausted"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "depth": self.depth,
            "n_returned": self.n_returned,
            "saturated": self.saturated,
            "unresolvable": self.unresolvable,
        }


@dataclass
class DiscoveryResult:
    symbols: set[str] = field(default_factory=set)
    screener_symbols: set[str] = field(default_factory=set)
    probe_symbols: set[str] = field(default_factory=set)
    unknown_symbols: set[str] = field(default_factory=set)
    non_equity: set[str] = field(default_factory=set)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    bands: list[dict[str, Any]] = field(default_factory=list)
    unresolved_bands: list[dict[str, Any]] = field(default_factory=list)
    truncation_detected: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def probe_only(self) -> set[str]:
        """Equities the screener missed — the number that justifies the probe."""
        return self.probe_symbols - self.screener_symbols

    @property
    def screener_only(self) -> set[str]:
        """Equities the probe missed — outside the probed code blocks."""
        return self.screener_symbols - self.probe_symbols

    def unknown_blocks(self) -> dict[str, list[str]]:
        """Unrecognised code blocks the screener returned — excluded, but visible."""
        return unknown_blocks(
            r["code"] for r in self.metadata.values() if r.get("instrument") == "unknown"
        )

    def instrument_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.metadata.values():
            key = record.get("instrument", "unknown")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def reconciliation(self) -> dict[str, Any]:
        """Like-for-like comparison of the two discovery methods.

        Counts cover **equities only**.  The screener answers "exchange ==
        SHH", which includes ETFs, LOFs, REITs and bonds; comparing that raw
        total against a probe of the equity code blocks measures the difference
        between two populations, not the disagreement between two methods.  In
        the 2026-08-11 run that inflated apparent disagreement from 1.5% to
        29.8% and tripped the tolerance gate for no reason.
        """
        union = self.symbols
        return {
            "screener_equities": len(self.screener_symbols),
            "probe_equities": len(self.probe_symbols),
            "union_equities": len(union),
            "probe_only": len(self.probe_only),
            "screener_only": len(self.screener_only),
            "agreement": len(self.screener_symbols & self.probe_symbols),
            "disagreement_fraction": (
                max(len(self.probe_only), len(self.screener_only)) / len(union) if union else 0.0
            ),
            "non_equity_excluded": len(self.non_equity),
            "probe_unresolved": len(self.unknown_symbols),
            "truncation_detected": self.truncation_detected,
            "bands_searched": len(self.bands),
            "bands_saturated": sum(1 for b in self.bands if b["saturated"]),
            "bands_unresolvable": len(self.unresolved_bands),
            "instruments": self.instrument_counts(),
            "unknown_blocks": {k: len(v) for k, v in self.unknown_blocks().items()},
        }


# --------------------------------------------------------------------------- yahoo adapters


def default_screener() -> ScreenerFn:
    """Adapter over ``yfinance.screen``, isolated so the API can change alone."""

    def _screen(query: Any, offset: int, size: int) -> Mapping[str, Any]:
        import yfinance as yf

        return yf.screen(query, offset=offset, size=size, sortField="intradaymarketcap",
                         sortAsc=False)

    return _screen


def build_query(exchange: str, low: float | None = None, high: float | None = None) -> Any:
    """``exchange == SHH`` optionally intersected with a market-cap band."""
    import yfinance as yf

    clauses = [yf.EquityQuery("eq", ["exchange", exchange])]
    if low is not None or high is not None:
        lo = low if low is not None else 0.0
        hi = high if high is not None else _MARKET_CAP_CEILING
        clauses.append(yf.EquityQuery("btwn", ["intradaymarketcap", lo, hi]))
    return clauses[0] if len(clauses) == 1 else yf.EquityQuery("and", clauses)


def default_probe() -> ProbeFn:
    """Existence check for a candidate symbol.

    Three outcomes, not two.  The original version returned a bool and caught
    every exception as "absent", which meant a transient network failure
    silently deleted a real listing from the universe — the 2026-08-11 run had
    nine such failures among 7,000 probes.  A timeout is not evidence of
    absence, so it is retried and, if it persists, reported as ``UNKNOWN`` for a
    human to look at.

    A one-year window rather than a few days: a suspended stock prints no recent
    bars but is certainly listed, and mistaking a halt for a non-existent
    listing is exactly the error the probe exists to avoid.
    """

    def _probe(symbol: str) -> ProbeOutcome:
        import yfinance as yf

        settings = get_settings()
        for attempt in range(settings.max_retries + 1):
            try:
                frame = yf.Ticker(symbol).history(
                    period="1y",
                    interval="1d",
                    timeout=settings.request_timeout,
                    raise_errors=True,
                )
                return ProbeOutcome.PRESENT if frame is not None and len(frame) > 0 \
                    else ProbeOutcome.ABSENT
            except Exception as exc:  # noqa: BLE001 - yfinance raises many types
                if not _is_transient(exc):
                    return ProbeOutcome.ABSENT
                if attempt >= settings.max_retries:
                    logger.debug("probe for %s gave up after %d attempts: %s",
                                 symbol, attempt + 1, exc)
                    return ProbeOutcome.UNKNOWN
                time.sleep(min(settings.backoff_base ** (attempt + 1), settings.backoff_cap)
                           * (0.5 + random.random()))
        return ProbeOutcome.UNKNOWN

    return _probe


def _is_transient(exc: Exception) -> bool:
    """Distinguish "Yahoo says no such symbol" from "Yahoo did not answer".

    Yahoo reports an unallocated code with a "possibly delisted / no price data"
    message; a network or rate-limit failure looks entirely different.  Getting
    this backwards in either direction is costly: treat a real outage as absence
    and the universe shrinks silently; treat absence as an outage and 4,600
    unallocated codes are retried four times each.
    """
    message = str(exc).lower()
    if any(marker in message for marker in _ABSENT_MARKERS):
        return False
    return any(marker in message for marker in _TRANSIENT_MARKERS) or isinstance(
        exc, (TimeoutError, ConnectionError, OSError)
    )


@contextmanager
def quiet_yfinance() -> Iterator[None]:
    """Silence yfinance's own logging for the duration of the probe.

    Probing 7,000 candidate codes to find ~2,300 listings means ~4,700 negative
    results, each of which yfinance logs at ERROR level.  Those lines are the
    expected output of a search, not faults, and burying the two dozen genuine
    warnings under 4,640 false alarms is how a real problem goes unnoticed.
    Outcomes are determined here, from return values and exception types, so
    nothing is lost by muting the vendor's commentary.
    """
    yf_logger = logging.getLogger("yfinance")
    previous_level, previous_disabled = yf_logger.level, yf_logger.disabled
    yf_logger.setLevel(logging.CRITICAL)
    yf_logger.disabled = True
    try:
        yield
    finally:
        yf_logger.setLevel(previous_level)
        yf_logger.disabled = previous_disabled


# --------------------------------------------------------------------------- screening


def _quotes(response: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """Yahoo has moved this payload around; accept the shapes seen in the wild."""
    if not response:
        return []
    if isinstance(response, Mapping):
        for key in ("quotes", "records", "result"):
            value = response.get(key)
            if isinstance(value, list):
                return [q for q in value if isinstance(q, Mapping)]
        finance = response.get("finance")
        if isinstance(finance, Mapping):
            result = finance.get("result")
            if isinstance(result, list) and result:
                return list(result[0].get("quotes", []))
    return []


def _reported_total(response: Mapping[str, Any] | None) -> int | None:
    if isinstance(response, Mapping):
        for key in ("total", "count"):
            value = response.get(key)
            if isinstance(value, int):
                return value
    return None


def screen_band(
    band: Band,
    *,
    screener: ScreenerFn,
    exchange: str,
    page_size: int,
    max_offset: int,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Page one band to exhaustion or to the ceiling.

    Returns ``(metadata_by_symbol, saturated)``.  ``saturated`` means the result
    set was cut off — the band's true size is unknown and it must be split.
    """
    query = build_query(exchange, band.low, band.high)
    found: dict[str, dict[str, Any]] = {}
    offset = 0
    received = 0
    saturated = False

    while True:
        try:
            response = screener(query, offset, page_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("screener failed at offset %d for band %s: %s", offset, band, exc)
            saturated = True  # unknown remainder -> treat as truncated, bisect
            break

        quotes = _quotes(response)
        received += len(quotes)
        for quote in quotes:
            record = _normalise_quote(quote)
            if record:
                found[record["ticker"]] = record

        total = _reported_total(response)
        offset += page_size

        if len(quotes) < page_size:
            # Short page: normally the end of the result set.  The comparison
            # must be against the rows actually *received*, not against the
            # offset — the offset advances by the requested page size, so a
            # server that truncates below that size would otherwise look
            # complete.  Measured cost of getting this wrong: 168 of 600
            # equities silently dropped in simulation.
            if total is not None and total > received:
                saturated = True
            break
        if offset >= max_offset:
            saturated = True
            break

    band.n_returned = len(found)
    band.saturated = saturated
    return found, saturated


def screen_universe(
    *,
    screener: ScreenerFn | None = None,
    exchange: str | None = None,
    page_size: int | None = None,
    max_offset: int | None = None,
    max_depth: int = 12,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], bool, list[dict[str, Any]]]:
    """Flat screen, then bisect every saturated band until nothing truncates.

    Returns ``(metadata, bands_searched, truncation_detected, unresolved_bands)``.

    A band that cannot be resolved by market cap is reported rather than retried
    to exhaustion — see :meth:`Band.splittable`.  Those bands are precisely
    where the code-range probe earns its cost.
    """
    settings = get_settings()
    screener = screener or default_screener()
    exchange = exchange or settings.exchange_code
    page_size = page_size or settings.screener_page_size
    max_offset = max_offset or settings.screener_max_offset

    metadata: dict[str, dict[str, Any]] = {}
    searched: list[dict[str, Any]] = []
    truncation_detected = False
    unresolved: list[dict[str, Any]] = []

    queue: list[Band] = [Band(low=0.0, high=None, depth=0)]
    while queue:
        band = queue.pop(0)
        found, saturated = screen_band(
            band,
            screener=screener,
            exchange=exchange,
            page_size=page_size,
            max_offset=max_offset,
        )
        metadata.update(found)
        logger.info(
            "band [%.3g, %s) depth=%d -> %d symbols%s",
            band.low,
            f"{band.high:.3g}" if band.high is not None else "inf",
            band.depth,
            len(found),
            " SATURATED" if saturated else "",
        )

        if not saturated:
            searched.append(band.as_dict())
            continue

        truncation_detected = True
        reason = band.splittable()
        if reason is None and band.depth >= max_depth:
            reason = "max_depth_reached"

        if reason is not None:
            band.unresolvable = reason
            searched.append(band.as_dict())
            unresolved.append(band.as_dict())
            logger.warning(
                "band [%.3g, %s) abandoned (%s): the screener cannot enumerate it; "
                "the code-range probe is the authority here",
                band.low,
                f"{band.high:.3g}" if band.high is not None else "inf",
                reason,
            )
            continue

        searched.append(band.as_dict())
        middle = band.midpoint()
        queue.append(Band(low=band.low, high=middle, depth=band.depth + 1))
        queue.append(Band(low=middle, high=band.high, depth=band.depth + 1))

    return metadata, searched, truncation_detected, unresolved


def _normalise_quote(quote: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalise one screener row.

    Nothing is filtered here: non-equities are *labelled* and dropped later, so
    that the reconciliation report can state how many funds the screener
    returned rather than making them vanish silently.
    """
    symbol = str(quote.get("symbol", "")).strip()
    match = _CODE_RE.match(symbol)
    if not match:
        return None
    code = match.group(1)
    quote_type = str(quote.get("quoteType") or "")
    return {
        "ticker": f"{code}.SS",
        "code": code,
        "board": board_of(code),
        "instrument": classify_code(code),
        "is_equity": is_equity(code, quote_type),
        "name": quote.get("longName") or quote.get("shortName"),
        "exchange": quote.get("fullExchangeName") or quote.get("exchange"),
        "currency": quote.get("currency"),
        "quote_type": quote_type,
        "market_cap": quote.get("marketCap"),
        "source": "screener",
    }


# --------------------------------------------------------------------------- probing


def candidate_codes(prefixes: Sequence[str]) -> list[str]:
    """Every six-digit code in the given SSE blocks."""
    codes: list[str] = []
    for prefix in prefixes:
        prefix = str(prefix)
        width = 6 - len(prefix)
        codes.extend(f"{prefix}{i:0{width}d}" for i in range(10**width))
    return codes


def probe_code_ranges(
    prefixes: Sequence[str] | None = None,
    *,
    probe: ProbeFn | None = None,
    skip: Iterable[str] = (),
    max_workers: int | None = None,
    rate_limiter: RateLimiter | None = None,
    progress_every: int = 250,
    quiet: bool = True,
) -> tuple[set[str], set[str], list[str]]:
    """Probe the SSE code space.  Returns ``(present, unknown, errors)``.

    ``unknown`` is the set the caller must not treat as absent: those codes were
    never actually answered.  Anything in it should be re-probed before the
    universe is declared complete.

    This is the expensive half of discovery (~7,000 candidate codes), so it is
    resumable: pass previously-probed symbols in ``skip``.
    """
    settings = get_settings()
    prefixes = list(prefixes or settings.code_ranges)
    probe = probe or default_probe()
    limiter = rate_limiter or RateLimiter(settings.requests_per_minute)
    skip_set = set(skip)

    symbols = [symbol_for(code, settings.yahoo_suffix) for code in candidate_codes(prefixes)]
    todo = [s for s in symbols if s not in skip_set]
    logger.info(
        "probing %d candidate codes (%d skipped); most will be unallocated and that is "
        "the expected result, not a failure",
        len(todo),
        len(symbols) - len(todo),
    )

    present: set[str] = set()
    unknown: set[str] = set()
    errors: list[str] = []

    def _check(symbol: str) -> tuple[str, ProbeOutcome, str | None]:
        limiter.acquire()
        try:
            outcome = probe(symbol)
            if isinstance(outcome, bool):  # legacy bool-returning probe
                outcome = ProbeOutcome.PRESENT if outcome else ProbeOutcome.ABSENT
            return symbol, outcome, None
        except Exception as exc:  # noqa: BLE001
            return symbol, ProbeOutcome.UNKNOWN, str(exc)[:200]

    context = quiet_yfinance() if quiet else nullcontext()
    with context, ThreadPoolExecutor(max_workers=max_workers or settings.max_workers) as pool:
        futures = [pool.submit(_check, s) for s in todo]
        for i, future in enumerate(as_completed(futures), start=1):
            symbol, outcome, error = future.result()
            if outcome is ProbeOutcome.PRESENT:
                present.add(symbol)
            elif outcome is ProbeOutcome.UNKNOWN:
                unknown.add(symbol)
                if error:
                    errors.append(f"{symbol}: {error}")
            if progress_every and i % progress_every == 0:
                logger.info("probe %d/%d, %d listings found, %d unresolved",
                            i, len(todo), len(present), len(unknown))

    logger.info(
        "probe complete: %d listings, %d unallocated codes, %d unresolved",
        len(present),
        len(todo) - len(present) - len(unknown),
        len(unknown),
    )
    if unknown:
        logger.warning(
            "%d codes could not be resolved (network failures, not absences) — re-probe "
            "them before treating the universe as complete: %s",
            len(unknown),
            ", ".join(sorted(unknown)[:10]),
        )
    return present, unknown, errors


# --------------------------------------------------------------------------- orchestration


def discover(
    *,
    screener: ScreenerFn | None = None,
    probe: ProbeFn | None = None,
    run_probe: bool = True,
    prefixes: Sequence[str] | None = None,
    skip_probed: Iterable[str] = (),
) -> DiscoveryResult:
    """Full discovery: screen (bisecting as needed), probe, reconcile, union.

    Only common shares reach ``symbols``.  The screener returns every instrument
    listed on the exchange — in the 2026-08-11 run, 1,000 of its 3,325 hits were
    ETFs, LOFs and REITs — and admitting those to the security master would send
    the backfill to download a thousand funds, spending roughly 30% of the
    Yahoo request budget and of the 512 MiB storage cap on instruments that can
    never enter a pairs trade.
    """
    result = DiscoveryResult()

    metadata, bands, truncated, unresolved = screen_universe(screener=screener)
    result.metadata.update(metadata)
    result.bands = bands
    result.unresolved_bands = unresolved
    result.truncation_detected = truncated

    equities = {t for t, record in metadata.items() if record.get("is_equity")}
    result.screener_symbols = equities
    result.non_equity = set(metadata) - equities
    logger.info(
        "screener returned %d symbols across %d bands: %d equities, %d other instruments (%s)",
        len(metadata),
        len(bands),
        len(equities),
        len(result.non_equity),
        ", ".join(f"{k}={v}" for k, v in list(result.instrument_counts().items())[:6]),
    )
    if unresolved:
        logger.warning(
            "%d band(s) the screener cannot enumerate: %s",
            len(unresolved),
            ", ".join(sorted({b["unresolvable"] for b in unresolved})),
        )
    blocks = result.unknown_blocks()
    if blocks:
        logger.warning(
            "%d symbol(s) in %d unrecognised code block(s), all EXCLUDED — check whether the "
            "exchange has allocated a new range: %s",
            sum(len(v) for v in blocks.values()),
            len(blocks),
            ", ".join(f"{k}xxx({len(v)}: {v[0]}...)" for k, v in list(blocks.items())[:6]),
        )

    if run_probe:
        present, unknown, errors = probe_code_ranges(prefixes, probe=probe, skip=skip_probed)
        result.probe_symbols = {s for s in present if is_equity(s.split(".")[0])}
        result.unknown_symbols = unknown
        result.errors.extend(errors[:100])
        for symbol in result.probe_symbols:
            if symbol not in result.metadata:
                code = symbol.split(".")[0]
                result.metadata[symbol] = {
                    "ticker": symbol,
                    "code": code,
                    "board": board_of(code),
                    "instrument": classify_code(code),
                    "is_equity": True,
                    "name": None,
                    "source": "probe",
                }
        logger.info(
            "probe found %d equities, %d of which the screener missed",
            len(result.probe_symbols),
            len(result.probe_symbols - result.screener_symbols),
        )

    result.symbols = result.screener_symbols | result.probe_symbols
    return result


def persist(
    result: DiscoveryResult,
    securities: Any,
    snapshots: Any,
    *,
    run_id: str,
    mark_missing: bool = True,
) -> dict[str, Any]:
    """Write discovery results and, when complete, mark vanished names inactive.

    A full screener+probe run is authoritative and may mark previously active
    names absent from the union as inactive.  A screener-only refresh is not:
    it cannot see probe-only names, so callers such as the nightly updater must
    pass ``mark_missing=False`` or they would create false delistings.
    """
    records = [result.metadata[s] for s in sorted(result.symbols) if s in result.metadata]
    stats = securities.upsert_many(records, source="discovery")

    inactivated = 0
    if mark_missing:
        known = set(securities.list_tickers(status=None))
        vanished = known - result.symbols
        inactivated = securities.mark_inactive(vanished)

    snapshots.record(
        source="screener",
        symbols=sorted(result.screener_symbols),
        run_id=run_id,
        details={
            "bands": result.bands,
            "unresolved_bands": result.unresolved_bands,
            "truncation_detected": result.truncation_detected,
            "non_equity_excluded": sorted(result.non_equity)[:500],
            "instruments": result.instrument_counts(),
        },
    )
    if result.probe_symbols:
        snapshots.record(
            source="probe",
            symbols=sorted(result.probe_symbols),
            run_id=run_id,
            details={
                "errors": result.errors[:50],
                "unresolved": sorted(result.unknown_symbols)[:200],
            },
        )

    return {
        **stats,
        "inactivated": inactivated,
        "reconciliation": result.reconciliation(),
        "taken_at": dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat(),
    }
