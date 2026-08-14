"""SSE trading-calendar utilities.

The exchange calendar is *inferred from the data* rather than hard-coded.  Two
reasons:

1. Chinese public holidays are set by State Council announcement each autumn
   for the following year, include lunar-calendar dates, and are routinely
   shifted to create the Spring Festival and National Day "golden weeks" — with
   working Saturdays used to compensate.  A hand-written table covering
   2010-2026 would be long, unverifiable and wrong somewhere.
2. What the quality layer actually needs is "was the exchange open?", and the
   most direct evidence is that most of the market printed a bar.

A date is a session when a quorum of the covered universe has a bar on it.
This distinguishes a market holiday (nobody trades) from a suspension (one
stock is halted while the rest of the market trades) — which matters, because
the first is not a gap and the second is a real event that must never be
forward-filled.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

#: Fixed-date PRC public holidays, used only as a sanity check on the inferred
#: calendar — never as the calendar itself.  Lunar holidays (Spring Festival,
#: Qingming, Dragon Boat, Mid-Autumn) move and are deliberately absent.
FIXED_HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 1),    # New Year's Day
    (5, 1),    # Labour Day
    (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),  # National Day
)


def is_weekend(day: dt.date) -> bool:
    return day.weekday() >= 5


def weekdays_between(start: dt.date, end: dt.date) -> np.ndarray:
    """Inclusive weekday range — the crude upper bound on sessions."""
    days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1)
    weekday = (days.astype("datetime64[D]").astype(int) + 4) % 7  # 1970-01-01 was a Thursday
    return days[weekday < 5]


def infer_sessions(
    dates_by_ticker: Mapping[str, Sequence[np.datetime64]] | Iterable[Sequence[np.datetime64]],
    *,
    quorum: float = 0.5,
) -> np.ndarray:
    """Infer trading sessions as dates where >= ``quorum`` of active names printed.

    The denominator is per-date: the number of tickers whose own coverage span
    contains that date.  Using a global denominator would misclassify every
    session before the STAR market existed.
    """
    series = list(dates_by_ticker.values()) if isinstance(dates_by_ticker, Mapping) else list(dates_by_ticker)
    observed: Counter[np.datetime64] = Counter()
    spans: list[tuple[np.datetime64, np.datetime64]] = []

    for dates in series:
        arr = np.asarray(dates, dtype="datetime64[D]")
        if arr.size == 0:
            continue
        observed.update(arr.tolist())
        spans.append((arr.min(), arr.max()))

    if not observed:
        return np.zeros(0, dtype="datetime64[D]")

    candidates = np.array(sorted(observed), dtype="datetime64[D]")
    starts = np.array([s for s, _ in spans], dtype="datetime64[D]")
    ends = np.array([e for _, e in spans], dtype="datetime64[D]")

    sessions = []
    for day in candidates:
        live = int(np.count_nonzero((starts <= day) & (ends >= day)))
        if live and observed[day.item()] / live >= quorum:
            sessions.append(day)
    return np.array(sessions, dtype="datetime64[D]")


def sessions_between(calendar: np.ndarray, start: dt.date, end: dt.date) -> np.ndarray:
    """Inclusive slice of an inferred calendar."""
    calendar = np.asarray(calendar, dtype="datetime64[D]")
    mask = (calendar >= np.datetime64(start, "D")) & (calendar <= np.datetime64(end, "D"))
    return calendar[mask]


def missing_sessions(
    ticker_dates: Sequence[np.datetime64],
    calendar: np.ndarray,
    *,
    within_coverage_only: bool = True,
) -> np.ndarray:
    """Sessions the exchange was open but this ticker did not print.

    These are candidate suspensions.  They are reported, never filled.
    """
    dates = np.asarray(ticker_dates, dtype="datetime64[D]")
    calendar = np.asarray(calendar, dtype="datetime64[D]")
    if dates.size == 0:
        return calendar.copy() if not within_coverage_only else np.zeros(0, dtype="datetime64[D]")
    if within_coverage_only:
        calendar = calendar[(calendar >= dates.min()) & (calendar <= dates.max())]
    return np.setdiff1d(calendar, dates)


def calendar_sanity(calendar: np.ndarray) -> dict[str, list[str]]:
    """Cross-check an inferred calendar against things that must be true."""
    calendar = np.asarray(calendar, dtype="datetime64[D]")
    days = [d.item() for d in calendar]
    weekend = [d.isoformat() for d in days if is_weekend(d)]
    fixed = [d.isoformat() for d in days if (d.month, d.day) in FIXED_HOLIDAYS]
    return {
        # Expected to be EMPTY.  China designates compensating working Saturdays
        # around the golden weeks, but those apply to offices and schools — the
        # exchange stays shut.  SSE sessions are Monday-Friday minus holidays,
        # without exception, so any entry here is a data error, not a quirk.
        "sessions_on_weekend": weekend,
        "sessions_on_fixed_holiday": fixed,
    }


def year_bounds(year: int) -> tuple[dt.date, dt.date]:
    return dt.date(year, 1, 1), dt.date(year, 12, 31)


def to_dates(days: np.ndarray) -> list[dt.date]:
    return [d.item() for d in np.asarray(days, dtype="datetime64[D]")]
