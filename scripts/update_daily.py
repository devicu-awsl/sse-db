#!/usr/bin/env python3
"""Keep the database current.  This is the "script to keep updating the data".

    python scripts/update_daily.py                 # nightly incremental
    python scripts/update_daily.py --full-refresh  # weekly adjustment resync
    python scripts/update_daily.py --skip-universe

What it does, in order:

1. **Refresh the universe** (screener only — the code probe is too slow for a
   nightly job) so new listings are picked up.  Because this is not the full
   screener+probe union, it does not mark missing names inactive.
2. **Backfill new listings** from ``history_start``: a stock that IPO'd today has
   no watermark, so an incremental window would fetch nothing.
3. **Re-download an overlap window** — the last ``update_overlap_days`` sessions,
   not just yesterday.  Yahoo silently corrects recent bars, and a strictly
   forward-only updater freezes the first version it ever saw.
4. **Merge the overlap into affected chunks.**  Rows outside the downloaded
   window are retained, refreshed dates replace their old values, and checksums
   are compared before writing so an unchanged year is not written.

``--full-refresh`` re-downloads complete history for every ticker.  This is not
paranoia: a split retroactively rewrites the *entire* ``adj_close`` series, so a
delta-only updater drifts silently out of sync with the vendor forever after.
Run it weekly (roadmap §5.8).  Because chunks are compared by checksum, the
sweep is cheap in writes even though it is expensive in requests.

Exit codes: 0 ok, 1 failures above tolerance or validation errors, 2 storage
halt, 3 database down.  Non-zero is what makes a cron failure visible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Sequence

import _path  # noqa: F401

from sse.config import get_settings
from sse.db import client as db_client
from sse.db.repository import (
    MongoPriceRepository,
    Outcome,
    RunCounters,
    RunRepository,
    SecurityRepository,
    SnapshotRepository,
)
from sse.db.schema import format_storage, storage_stats
from sse.ingest import quality, universe
from sse.ingest.provider import YahooPriceProvider, board_of
from sse.logging_setup import setup_logging

logger = setup_logging("update")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-refresh", action="store_true",
                        help="re-download complete history (weekly adjustment resync)")
    parser.add_argument("--skip-universe", action="store_true")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--overlap-days", type=int, help="default: SSE_UPDATE_OVERLAP_DAYS")
    parser.add_argument("--end", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--failure-tolerance", type=float, default=0.05)
    parser.add_argument("--report", type=Path, default=Path("reports/update_last.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()

    try:
        db_client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.error("database unavailable: %s", db_client.redact(str(exc)))
        return 3

    db = db_client.get_db()
    prices = MongoPriceRepository(db)
    securities = SecurityRepository(db)
    snapshots = SnapshotRepository(db)
    runs = RunRepository(db)

    kind = "full_refresh" if args.full_refresh else "update"
    run_id = f"{kind}-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}"
    runs.start(run_id, kind, {"full_refresh": args.full_refresh})
    counters = RunCounters()
    end = args.end or dt.date.today()
    overlap = args.overlap_days or settings.update_overlap_days

    # 1. universe refresh -----------------------------------------------------
    if not args.skip_universe and not args.tickers:
        try:
            result = universe.discover(run_probe=False)
            # The nightly refresh intentionally uses the screener only.  It is
            # therefore incomplete relative to the full screener+probe union
            # and must never turn probe-only equities into false delistings.
            stats = universe.persist(
                result, securities, snapshots, run_id=run_id, mark_missing=False
            )
            logger.info("universe: %d symbols, %d newly inactive",
                        stats["reconciliation"]["union_equities"], stats["inactivated"])
        except Exception as exc:  # noqa: BLE001
            # A screener outage must not stop the price update.
            logger.warning("universe refresh failed, continuing with the stored master: %s", exc)
            counters.errors.append({"stage": "universe", "error": str(exc)[:300]})

    tickers = [t.upper() for t in args.tickers] if args.tickers else securities.list_tickers("active")
    if args.limit:
        tickers = tickers[: args.limit]
    if not tickers:
        logger.error("no active tickers — run scripts/discover_universe.py first")
        runs.finish(run_id, status="failed", counters=counters, note="empty universe")
        return 1

    # 2. per-ticker start dates ----------------------------------------------
    watermarks = prices.last_dates(tickers)
    new_listings = [t for t in tickers if t not in watermarks]
    logger.info("%d tickers: %d with history, %d new listings%s",
                len(tickers), len(watermarks), len(new_listings),
                " (FULL REFRESH)" if args.full_refresh else "")

    counters.tickers_total = len(tickers)
    reports: list[quality.QualityReport] = []
    provider = YahooPriceProvider()
    halted = False
    validation_failures: list[str] = []

    groups: dict[dt.date, list[str]] = {}
    for ticker in tickers:
        if args.full_refresh or ticker in new_listings:
            start = settings.history_start
        else:
            start = watermarks[ticker] - dt.timedelta(days=overlap)
            start = max(start, settings.history_start)
        groups.setdefault(start, []).append(ticker)

    # 3. fetch and store ------------------------------------------------------
    for start, batch in sorted(groups.items()):
        if start > end:
            counters.tickers_done += len(batch)
            continue
        logger.info("fetching %d tickers from %s to %s", len(batch), start, end)
        stream = provider.fetch_many(batch, start, end, max_workers=args.workers)
        try:
            for result in stream:
                failure = _handle(
                    result,
                    prices,
                    counters,
                    reports,
                    securities,
                    run_id,
                    merge_existing=(not args.full_refresh and result.ticker in watermarks),
                )
                if failure:
                    validation_failures.append(failure)
                processed = counters.tickers_done + counters.tickers_failed
                if processed % settings.storage_check_every == 0:
                    stats = storage_stats(db)
                    if stats["over_threshold"]:
                        logger.error("storage guard tripped: %s", format_storage(stats))
                        halted = True
                        break
        finally:
            stream.close()
        runs.heartbeat(run_id, counters)
        if halted:
            break

    # 4. finish ---------------------------------------------------------------
    storage = _storage(db)
    census = quality.census(reports)
    failure_rate = counters.tickers_failed / max(counters.tickers_total, 1)
    status = (
        "halted_storage" if halted
        else "failed" if failure_rate > args.failure_tolerance or validation_failures
        else "ok"
    )
    runs.finish(run_id, status=status, counters=counters, storage=storage,
                note=f"validation_failures={len(validation_failures)}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "kind": kind,
                "status": status,
                "counters": counters.as_dict(),
                "census": census,
                "storage": storage,
                "validation_failures": validation_failures[:50],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info("--- %s %s ---", kind, status)
    logger.info("  tickers ok/failed  : %d/%d", counters.tickers_done, counters.tickers_failed)
    logger.info("  chunks written     : %d new, %d rewritten, %d unchanged",
                counters.chunks_inserted, counters.chunks_updated, counters.chunks_unchanged)
    if counters.chunks_updated and not args.full_refresh:
        logger.info("  (rewrites are Yahoo corrections inside the %d-day overlap window)", overlap)
    if storage:
        logger.info("  storage            : %s", format_storage(storage))

    if halted:
        return 2
    return 0 if status == "ok" else 1


def _handle(
    result,
    prices,
    counters,
    reports,
    securities,
    run_id,
    *,
    merge_existing: bool = False,
) -> str | None:
    """Returns a description of a validation failure, or None."""
    ticker = result.ticker
    if not result.ok:
        counters.tickers_failed += 1
        counters.errors.append({"ticker": ticker, "error": result.error})
        return None

    board = board_of(ticker.split(".")[0])
    cleaned, report = quality.sanitize(result.bars, board=board, ticker=ticker)
    reports.append(report)

    if not report.ok:
        # sanitize() should have removed every error-severity row; anything left
        # is a bug or a new failure mode, and must not pass silently.
        counters.tickers_failed += 1
        return f"{ticker}: {report.summary()}"

    if len(cleaned) == 0:
        counters.tickers_done += 1
        return None

    outcomes = (
        prices.merge_series(ticker, cleaned)
        if merge_existing
        else prices.upsert_series(ticker, cleaned)
    )
    counters.chunks_inserted += outcomes[Outcome.INSERTED]
    counters.chunks_updated += outcomes[Outcome.UPDATED]
    counters.chunks_unchanged += outcomes[Outcome.UNCHANGED]
    counters.bars_written += len(cleaned)
    counters.tickers_done += 1
    securities.set_ingest_state(ticker, "ok", run_id=run_id,
                                last=cleaned.dates[-1].item().isoformat(), bars=len(cleaned))
    return None


def _storage(db) -> dict:
    try:
        return storage_stats(db)
    except Exception:  # noqa: BLE001
        return {}


if __name__ == "__main__":
    sys.exit(main())
