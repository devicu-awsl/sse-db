"""Data-quality layer.

The purpose is to **measure** Yahoo, not to launder it.  Nothing here fills,
smooths or interpolates: a missing bar in an A-share series is usually a real
trading suspension, and a forward fill would manufacture a flat price series
that a mean-reversion strategy would happily "trade" and profit from.  Rows are
dropped only when they are structurally impossible (duplicate date, no usable
price), and every drop is counted and reported.

Checks split into two severities:

``ERROR``   the row cannot be stored as-is — duplicate date, non-finite or
            non-positive price, OHLC inequality violated.
``WARNING`` the row is stored and flagged — zero volume, a move beyond the
            exchange's daily price limit, a suspected suspension gap.

The price-limit check is specific to this market and is the most informative of
them all: SSE main-board stocks are limited to ±10% and STAR-market stocks to
±20% per session, so a larger close-to-close move with no dividend or split on
that date is either a data error or a resumption after a long halt.  Either way
it must not enter a spread calculation unexamined, and the same limits become
the tradability mask in the Phase 2 backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..codec import PRICE_FIELDS, BarSeries
from ..logging_setup import setup_logging

logger = setup_logging(__name__)

#: Daily price limits by board.  ST/*ST names are limited to 5% but Yahoo does
#: not expose the ST flag, so those show up as warnings — recorded, not assumed.
PRICE_LIMITS: dict[str, float] = {"main": 0.10, "star": 0.20, "b_share": 0.10, "other": 0.20}
#: Tolerance above the limit before flagging (ticks, rounding, IPO-day auctions).
LIMIT_TOLERANCE = 0.005


@dataclass
class QualityReport:
    ticker: str
    n_input: int = 0
    n_output: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return self.n_input - self.n_output

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, name: str, count: int, **detail: Any) -> None:
        if count:
            self.errors[name] = self.errors.get(name, 0) + int(count)
            self.details.update({f"{name}.{k}": v for k, v in detail.items()})

    def warn(self, name: str, count: int, **detail: Any) -> None:
        if count:
            self.warnings[name] = self.warnings.get(name, 0) + int(count)
            self.details.update({f"{name}.{k}": v for k, v in detail.items()})

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_dropped": self.n_dropped,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }

    def summary(self) -> str:
        parts = [f"{self.ticker}: {self.n_output}/{self.n_input} bars"]
        if self.errors:
            parts.append("errors=" + ", ".join(f"{k}:{v}" for k, v in sorted(self.errors.items())))
        if self.warnings:
            parts.append("warnings=" + ", ".join(f"{k}:{v}" for k, v in sorted(self.warnings.items())))
        return " | ".join(parts)


def validate(bars: BarSeries, *, board: str = "main", ticker: str | None = None) -> QualityReport:
    """Inspect a series without modifying it."""
    name = ticker or bars.ticker or "?"
    report = QualityReport(ticker=name, n_input=len(bars), n_output=len(bars))
    if len(bars) == 0:
        return report

    dates = bars.dates
    order = np.argsort(dates.astype("int64"), kind="stable")
    if not np.array_equal(order, np.arange(len(bars))):
        report.error("unsorted_dates", 1)
    unique, counts = np.unique(dates, return_counts=True)
    duplicates = int(np.sum(counts - 1))
    if duplicates:
        report.error("duplicate_dates", duplicates,
                     examples=[str(d) for d in unique[counts > 1][:5]])

    for column in PRICE_FIELDS:
        values = getattr(bars, column)
        if values is None:
            continue
        nan_count = int(np.count_nonzero(np.isnan(values)))
        if nan_count:
            report.warn(f"nan_{column}", nan_count)
        finite = values[np.isfinite(values)]
        non_positive = int(np.count_nonzero(finite <= 0))
        if non_positive:
            report.error(f"non_positive_{column}", non_positive)

    if bars.volume is not None:
        finite_volume = bars.volume[np.isfinite(bars.volume)]
        negative = int(np.count_nonzero(finite_volume < 0))
        if negative:
            report.error("negative_volume", negative)
        zero_volume = int(np.count_nonzero(finite_volume == 0))
        if zero_volume:
            report.warn("zero_volume", zero_volume)

    violations = ohlc_violations(bars)
    if violations.any():
        report.error("ohlc_inconsistent", int(violations.sum()),
                     examples=[str(d) for d in bars.dates[violations][:5]])

    limit_breaks = price_limit_breaks(bars, board=board)
    if limit_breaks.any():
        report.warn("price_limit_exceeded", int(limit_breaks.sum()),
                    examples=[str(d) for d in bars.dates[limit_breaks][:5]])

    locked = limit_locked_days(bars)
    if locked.any():
        report.warn("limit_locked_candidate", int(locked.sum()))

    gaps = calendar_gaps(bars.dates)
    if gaps:
        report.warn("long_gaps", len(gaps), examples=gaps[:5])

    return report


def ohlc_violations(bars: BarSeries) -> np.ndarray:
    """``high >= max(open, close, low)`` and ``low <= min(open, close, high)``."""
    n = len(bars)
    if bars.high is None or bars.low is None:
        return np.zeros(n, dtype=bool)

    stack = [bars.open, bars.close, bars.low]
    upper = np.nanmax(np.vstack([a for a in stack if a is not None]), axis=0)
    stack = [bars.open, bars.close, bars.high]
    lower = np.nanmin(np.vstack([a for a in stack if a is not None]), axis=0)

    with np.errstate(invalid="ignore"):
        bad_high = bars.high < upper - 1e-6
        bad_low = bars.low > lower + 1e-6
    return np.nan_to_num(bad_high, nan=False) | np.nan_to_num(bad_low, nan=False)


def price_limit_breaks(bars: BarSeries, *, board: str = "main") -> np.ndarray:
    """Close-to-close moves beyond the board's daily limit, excluding action days.

    Raw close is used deliberately: the limit applies to the traded price.  Days
    carrying a dividend or split are excluded because the raw series legitimately
    jumps on those dates.
    """
    n = len(bars)
    if bars.close is None or n < 2:
        return np.zeros(n, dtype=bool)

    limit = PRICE_LIMITS.get(board, 0.20) + LIMIT_TOLERANCE
    close = bars.close
    with np.errstate(invalid="ignore", divide="ignore"):
        change = np.abs(close[1:] / close[:-1] - 1.0)
    flagged = np.zeros(n, dtype=bool)
    flagged[1:] = np.nan_to_num(change, nan=0.0) > limit

    for actions in (bars.dividends, bars.splits):
        if actions is not None:
            flagged &= np.nan_to_num(actions, nan=0.0) == 0.0
    return flagged


def limit_locked_days(bars: BarSeries) -> np.ndarray:
    """Sessions where open == high == low == close: a locked limit or a halt.

    Not an error — but these bars are untradeable, which the backtest must know.
    """
    if any(getattr(bars, f) is None for f in ("open", "high", "low", "close")):
        return np.zeros(len(bars), dtype=bool)
    same = (bars.high == bars.low) & (bars.open == bars.close) & (bars.high == bars.close)
    return np.nan_to_num(same, nan=False)


def calendar_gaps(dates: np.ndarray, *, min_days: int = 30) -> list[str]:
    """Gaps long enough to be a real suspension rather than a holiday week."""
    if dates.size < 2:
        return []
    spacing = np.diff(dates.astype("datetime64[D]")).astype("int64")
    gaps = np.flatnonzero(spacing >= min_days)
    return [f"{dates[i]}..{dates[i + 1]} ({spacing[i]}d)" for i in gaps]


def sanitize(bars: BarSeries, *, board: str = "main", ticker: str | None = None
             ) -> tuple[BarSeries, QualityReport]:
    """Return a storable series plus the report describing what was removed.

    Removal rules, in order:
      1. sort by date;
      2. drop duplicate dates, keeping the last occurrence (Yahoo's corrections
         arrive as later duplicates);
      3. drop rows with no usable close, or with a non-positive price;
      4. drop rows violating the OHLC inequalities.

    Nothing is ever filled in.
    """
    name = ticker or bars.ticker or "?"
    if len(bars) == 0:
        return bars, QualityReport(ticker=name)

    before = validate(bars, board=board, ticker=name)

    order = np.argsort(bars.dates.astype("int64"), kind="stable")
    ordered = bars._take(order)

    keep = np.ones(len(ordered), dtype=bool)
    unique_dates, last_index = np.unique(ordered.dates[::-1], return_index=True)
    keep_indices = len(ordered) - 1 - last_index
    keep[:] = False
    keep[keep_indices] = True

    if ordered.close is not None:
        keep &= np.isfinite(ordered.close) & (ordered.close > 0)
    for column in PRICE_FIELDS:
        values = getattr(ordered, column)
        if values is not None:
            with np.errstate(invalid="ignore"):
                keep &= ~np.nan_to_num(values <= 0, nan=False)

    if ordered.volume is not None:
        # Negative volume is structurally impossible.  Zero volume remains a
        # warning (often a halt/locked limit), while a negative value must not
        # reach storage even on the backfill path.
        with np.errstate(invalid="ignore"):
            keep &= ~np.nan_to_num(ordered.volume < 0, nan=False)

    keep &= ~ohlc_violations(ordered)

    cleaned = ordered._take(keep)
    after = validate(cleaned, board=board, ticker=name)
    after.n_input = len(bars)
    after.n_output = len(cleaned)
    after.details["dropped_by_sanitize"] = int(len(bars) - len(cleaned))
    after.details["pre_sanitize_errors"] = dict(before.errors)
    return cleaned, after


def census(reports: Sequence[QualityReport] | Mapping[str, QualityReport]) -> dict[str, Any]:
    """Aggregate report across the universe — the table in notebook 01."""
    items = list(reports.values()) if isinstance(reports, Mapping) else list(reports)
    if not items:
        return {"tickers": 0}

    error_totals: dict[str, int] = {}
    warning_totals: dict[str, int] = {}
    for report in items:
        for key, value in report.errors.items():
            error_totals[key] = error_totals.get(key, 0) + value
        for key, value in report.warnings.items():
            warning_totals[key] = warning_totals.get(key, 0) + value

    return {
        "tickers": len(items),
        "tickers_clean": sum(1 for r in items if r.ok and not r.warnings),
        "tickers_with_errors": sum(1 for r in items if r.errors),
        "bars_in": sum(r.n_input for r in items),
        "bars_out": sum(r.n_output for r in items),
        "bars_dropped": sum(r.n_dropped for r in items),
        "errors": dict(sorted(error_totals.items())),
        "warnings": dict(sorted(warning_totals.items())),
        "worst": sorted(
            ({"ticker": r.ticker, "dropped": r.n_dropped} for r in items if r.n_dropped),
            key=lambda row: -row["dropped"],
        )[:20],
    }
