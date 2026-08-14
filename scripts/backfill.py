#!/usr/bin/env python3
"""Backfill daily history for the whole SSE universe.

    python scripts/backfill.py                        # everything, from 2010
    python scripts/backfill.py --limit 50             # 50-ticker development slice
    python scripts/backfill.py --board main --workers 4
    python scripts/backfill.py --resume RUN_ID        # continue after a crash
    python scripts/backfill.py --tickers 600000.SS 600519.SS

This is the long pole of the week and it is I/O-bound on Yahoo's rate limits,
not on the machine.  Three properties make an eight-hour unattended run
survivable:

**Idempotent.**  A chunk's ``_id`` is ``{ticker}:{year}`` and its checksum is
compared before writing, so re-running changes nothing that has not changed.
Interrupt it at any point and start it again.

**Restartable.**  Per-ticker state is written to ``securities.ingest``, so
``--resume`` picks up exactly the tickers that never completed.

**Bounded.**  Storage is polled every ``storage_check_every`` tickers and the
run halts cleanly at 90% of the M0 cap rather than dying mid-write at 100%.

Exit codes: 0 ok, 1 failures above tolerance, 2 halted on storage, 3 database down.
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
)
from sse.db.schema import format_storage, storage_stats
from sse.ingest import quality
from sse.ingest.provider import YahooPriceProvider, board_of
from sse.logging_setup import setup_logging

logger = setup_logging("backfill")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", help="explicit symbols instead of the master list")
    parser.add_argument("--board", choices=["main", "star", "b_share", "other"])
    parser.add_argument("--limit", type=int, help="cap the number of tickers (development slice)")
    parser.add_argument("--start", type=dt.date.fromisoformat, help="default: SSE_HISTORY_START")
    parser.add_argument("--end", type=dt.date.fromisoformat, help="default: today")
    parser.add_argument("--workers", type=int, help="default: SSE_MAX_WORKERS")
    parser.add_argument("--batch-size", type=int, default=100, help="tickers between checkpoints")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    parser.add_argument("--missing-only", action="store_true",
                        help="only tickers with no stored bars (new listings, recovered names)")
    parser.add_argument("--failure-tolerance", type=float, default=0.05,
                        help="fraction of tickers allowed to fail before exit code 1")
    parser.add_argument("--dry-run", action="store_true", help="fetch and validate, do not write")
    parser.add_argument("--report", type=Path, default=Path("reports/backfill_quality.json"))
    return parser.parse_args(argv)


def select_tickers(
    args: argparse.Namespace,
    securities: SecurityRepository,
    prices: MongoPriceRepository,
) -> list[str]:
    if args.tickers:
        return [t.upper() for t in args.tickers]
    if args.missing_only:
        # Discovery can add names long after the last backfill — the code probe
        # recovered 35 on 11 Aug that the screener had never returned.  Fetching
        # only what is absent costs 35 requests instead of 2,360.
        active = securities.list_tickers("active", board=args.board)
        stored = set(prices.tickers())
        missing = [t for t in active if t not in stored]
        logger.info("%d of %d active tickers have no stored bars", len(missing), len(active))
        return missing[: args.limit] if args.limit else missing
    if args.resume:
        pending = securities.pending(args.resume)
        logger.info("resuming run %s: %d tickers outstanding", args.resume, len(pending))
        return pending[: args.limit] if args.limit else pending
    tickers = securities.list_tickers(status="active", board=args.board)
    if not tickers:
        logger.error("security master is empty — run scripts/discover_universe.py first")
    return tickers[: args.limit] if args.limit else tickers


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
    runs = RunRepository(db)

    start = args.start or settings.history_start
    end = args.end or dt.date.today()
    tickers = select_tickers(args, securities, prices)
    if not tickers:
        return 1

    run_id = args.resume or f"backfill-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}"
    runs.start(run_id, "backfill",
               {"start": start.isoformat(), "end": end.isoformat(), "n_tickers": len(tickers),
                "board": args.board, "dry_run": args.dry_run})

    counters = RunCounters(tickers_total=len(tickers))
    reports: list[quality.QualityReport] = []
    provider = YahooPriceProvider()
    halted = False

    logger.info("backfill %s: %d tickers, %s..%s, %d workers",
                run_id, len(tickers), start, end, args.workers or settings.max_workers)

    for offset in range(0, len(tickers), args.batch_size):
        batch = tickers[offset : offset + args.batch_size]
        stream = provider.fetch_many(batch, start, end, max_workers=args.workers)
        try:
            for result in stream:
                _handle(result, prices, securities, counters, reports, run_id, args.dry_run)

                if (counters.tickers_done + counters.tickers_failed) % settings.storage_check_every == 0:
                    stats = storage_stats(db)
                    logger.info("storage: %s", format_storage(stats))
                    if stats["over_threshold"]:
                        logger.error(
                            "storage guard tripped at %.1f%% of the M0 cap — halting cleanly",
                            stats["used_fraction"] * 100,
                        )
                        halted = True
                        break
        finally:
            stream.close()  # cancel outstanding futures promptly on an early exit

        runs.heartbeat(run_id, counters)
        done = counters.tickers_done + counters.tickers_failed
        logger.info("progress %d/%d tickers, %d bars written",
                    done, len(tickers), counters.bars_written)
        if halted:
            break

    stats = _final_storage(db)
    status = "halted_storage" if halted else ("ok" if not counters.tickers_failed else "partial")
    runs.finish(run_id, status=status, counters=counters, storage=stats)

    census = quality.census(reports)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"run_id": run_id, "census": census, "storage": stats}, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("--- backfill %s ---", status)
    logger.info("  tickers ok/failed : %d/%d", counters.tickers_done, counters.tickers_failed)
    logger.info("  chunks i/u/unchanged: %d/%d/%d", counters.chunks_inserted,
                counters.chunks_updated, counters.chunks_unchanged)
    logger.info("  bars written      : %d", counters.bars_written)
    logger.info("  bars dropped by QC: %d", census.get("bars_dropped", 0))
    if stats:
        logger.info("  storage           : %s", format_storage(stats))
    logger.info("  report            : %s", args.report)
    if counters.errors:
        logger.warning("  first failures    : %s",
                       ", ".join(e["ticker"] for e in counters.errors[:10]))

    if halted:
        return 2
    if counters.tickers_total and counters.tickers_failed / counters.tickers_total > args.failure_tolerance:
        logger.error("failure rate %.1f%% exceeds tolerance",
                     100 * counters.tickers_failed / counters.tickers_total)
        return 1
    return 0


def _handle(result, prices, securities, counters, reports, run_id, dry_run) -> None:
    ticker = result.ticker
    if not result.ok:
        counters.tickers_failed += 1
        counters.errors.append({"ticker": ticker, "error": result.error})
        securities.set_ingest_state(ticker, "failed", run_id=run_id, error=result.error)
        return

    board = board_of(ticker.split(".")[0])
    cleaned, report = quality.sanitize(result.bars, board=board, ticker=ticker)
    reports.append(report)

    if len(cleaned) == 0:
        counters.tickers_done += 1
        securities.set_ingest_state(ticker, "empty", run_id=run_id, bars=0)
        logger.debug("%s: no usable bars", ticker)
        return

    if dry_run:
        counters.tickers_done += 1
        counters.bars_written += len(cleaned)
        return

    outcomes = prices.upsert_series(ticker, cleaned)
    counters.chunks_inserted += outcomes[Outcome.INSERTED]
    counters.chunks_updated += outcomes[Outcome.UPDATED]
    counters.chunks_unchanged += outcomes[Outcome.UNCHANGED]
    counters.bars_written += len(cleaned)
    counters.tickers_done += 1
    securities.set_ingest_state(
        ticker,
        "ok",
        run_id=run_id,
        bars=len(cleaned),
        first=cleaned.dates[0].item().isoformat(),
        last=cleaned.dates[-1].item().isoformat(),
        dropped=report.n_dropped,
        warnings=sorted(report.warnings),
    )


def _final_storage(db) -> dict:
    try:
        return storage_stats(db)
    except Exception:  # noqa: BLE001
        return {}


if __name__ == "__main__":
    sys.exit(main())
