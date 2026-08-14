#!/usr/bin/env python3
"""Verify what is actually in the database.

    python scripts/verify_integrity.py              # full sweep
    python scripts/verify_integrity.py --sample 200 # quick check

Four independent checks, because "the backfill exited 0" is not evidence that
the data is right:

1. **Checksums.**  Every chunk is decoded and its sha256 recomputed, catching a
   truncated write, a partially-applied update or storage-level corruption.
2. **Row-count reconciliation.**  The bar count summed from chunk metadata must
   equal the count obtained by decoding the date columns.  A mismatch means a
   chunk's ``n`` disagrees with its payload — the acceptance check in §9.
3. **Calendar.**  The trading calendar is inferred from the cross-section, then
   sanity-checked (sessions on a Sunday, sessions on National Day) and used to
   census suspensions per ticker.
4. **Quality.**  The same validators the ingestion path uses, re-run against
   stored data, so the numbers in the deck come from the database rather than
   from a transient in-memory object.

Exit codes: 0 clean, 1 integrity or reconciliation failure, 3 database down.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import Counter
from pathlib import Path

import _path  # noqa: F401

from sse import calendar_utils
from sse.codec import decode_chunk, verify_chunk
from sse.db import client as db_client
from sse.db.repository import MongoPriceRepository, SecurityRepository
from sse.db.schema import format_storage, storage_stats
from sse.ingest import quality
from sse.ingest.provider import board_of
from sse.logging_setup import setup_logging

logger = setup_logging("verify")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, help="check a random subset of tickers")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quorum", type=float, default=0.5,
                        help="fraction of live names needed to call a date a session")
    parser.add_argument("--report", type=Path, default=Path("reports/integrity.json"))
    args = parser.parse_args()

    try:
        db_client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.error("database unavailable: %s", db_client.redact(str(exc)))
        return 3

    db = db_client.get_db()
    prices = MongoPriceRepository(db)
    securities = SecurityRepository(db)

    tickers = prices.tickers()
    if args.sample and args.sample < len(tickers):
        random.seed(args.seed)
        tickers = sorted(random.sample(tickers, args.sample))
    logger.info("verifying %d tickers", len(tickers))

    checksum_failures: list[str] = []
    count_mismatches: list[str] = []
    reports: list[quality.QualityReport] = []
    dates_by_ticker: dict[str, list] = {}
    metadata_bars = 0
    decoded_bars = 0
    chunks = 0
    schema_versions: Counter[int] = Counter()

    for ticker in tickers:
        series_dates = []
        for doc in prices.iter_chunks(ticker):
            chunks += 1
            schema_versions[int(doc.get("schema_version", 0))] += 1
            if not verify_chunk(doc):
                checksum_failures.append(str(doc.get("_id")))
            bars = decode_chunk(doc)
            metadata_bars += int(doc.get("n", 0))
            decoded_bars += len(bars)
            if int(doc.get("n", -1)) != len(bars):
                count_mismatches.append(str(doc.get("_id")))
            series_dates.append(bars.dates)

        if not series_dates:
            continue
        full = prices.get_series(ticker)
        dates_by_ticker[ticker] = full.dates
        reports.append(quality.validate(full, board=board_of(ticker.split(".")[0]), ticker=ticker))

    calendar = calendar_utils.infer_sessions(dates_by_ticker, quorum=args.quorum)
    sanity = calendar_utils.calendar_sanity(calendar)
    suspensions = {
        ticker: int(calendar_utils.missing_sessions(dates, calendar).size)
        for ticker, dates in dates_by_ticker.items()
    }

    census = quality.census(reports)
    storage = storage_stats(db)
    summary = {
        "checked_at": dt.datetime.now(dt.UTC).isoformat(),
        "tickers": len(tickers),
        "chunks": chunks,
        "schema_versions": dict(schema_versions),
        "bars_metadata": metadata_bars,
        "bars_decoded": decoded_bars,
        "reconciled": metadata_bars == decoded_bars,
        "checksum_failures": checksum_failures[:50],
        "checksum_failure_count": len(checksum_failures),
        "count_mismatches": count_mismatches[:50],
        "calendar_sessions": int(calendar.size),
        "calendar_first": str(calendar[0]) if calendar.size else None,
        "calendar_last": str(calendar[-1]) if calendar.size else None,
        "calendar_sanity": {k: v[:10] for k, v in sanity.items()},
        "suspension_sessions_total": int(sum(suspensions.values())),
        "most_suspended": sorted(suspensions.items(), key=lambda kv: -kv[1])[:20],
        "quality_census": census,
        "storage": storage,
        "securities": securities.count_by_status(),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    logger.info("--- integrity ---")
    logger.info("  chunks              : %d across %d tickers", chunks, len(tickers))
    logger.info("  bars (meta/decoded) : %d / %d %s", metadata_bars, decoded_bars,
                "OK" if summary["reconciled"] else "MISMATCH")
    logger.info("  checksum failures   : %d", len(checksum_failures))
    logger.info("  trading sessions    : %d (%s..%s)", calendar.size,
                summary["calendar_first"], summary["calendar_last"])
    weekend = len(sanity["sessions_on_weekend"])
    logger.info("  weekend sessions    : %d (must be 0 — the exchange never trades a weekend)%s",
                weekend, "  <- DATA ERROR" if weekend else "")
    logger.info("  suspension sessions : %d", summary["suspension_sessions_total"])
    logger.info("  bars with warnings  : %s", census.get("warnings"))
    logger.info("  storage             : %s", format_storage(storage))
    logger.info("  report              : %s", args.report)

    if checksum_failures or count_mismatches or not summary["reconciled"]:
        logger.error("integrity check FAILED")
        return 1
    logger.info("integrity check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
