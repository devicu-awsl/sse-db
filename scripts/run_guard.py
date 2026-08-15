#!/usr/bin/env python3
"""Refuse to start if another updater is already running, anywhere.

    python scripts/run_guard.py            # exit 0 free, 75 busy, 3 db down
    python scripts/run_guard.py --wait 20  # poll for up to 20 minutes first

The scheduled job and a manual `update_daily.py` on a laptop are two writers
against one Atlas cluster.  GitHub's `concurrency:` key only serialises runs
*within GitHub*; it knows nothing about a terminal window in another country.

Chunk writes are idempotent and checksum-compared, so two updaters carrying the
same vendor data converge rather than corrupt.  What does not converge is
`merge_series`: it reads a year chunk, splices new rows in and writes the whole
chunk back, so two interleaved merges can silently drop the rows that landed
between one writer's read and its write.

The lock therefore lives where both writers can see it -- the `ingestion_runs`
collection that `update_daily.py` already maintains.  A run is considered live
when its status is `running` and it has been heard from within `--stale-after`
minutes; that timeout matters because a laptop closed mid-run would otherwise
hold the lock for ever.

Nothing is written here.  This only reads, so it cannot itself become the thing
that corrupts a run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from typing import Any, Sequence

import _path  # noqa: F401

from sse.db import client as db_client
from sse.db.schema import INGESTION_RUNS
from sse.logging_setup import setup_logging

logger = setup_logging("guard")

BUSY = 75  # distinct from update_daily's 0/1/2/3 so a skip is never read as success


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stale-after", type=int, default=180,
                        help="minutes without a heartbeat before a run is presumed dead")
    parser.add_argument("--wait", type=int, default=0,
                        help="minutes to wait for the lock before giving up")
    parser.add_argument("--poll", type=int, default=60, help="seconds between polls")
    return parser.parse_args(argv)


def active_run(stale_after_minutes: int) -> dict[str, Any] | None:
    """The live updater run, if there is one."""
    db = db_client.get_db()
    cutoff = (dt.datetime.now(dt.UTC).replace(tzinfo=None)
              - dt.timedelta(minutes=stale_after_minutes))
    for doc in db[INGESTION_RUNS].find({"status": "running"}).sort("started_at", -1):
        # `heartbeat_at` only appears after the first batch, so fall back to the
        # start time -- otherwise a run in its first minutes looks dead.
        seen = doc.get("heartbeat_at") or doc.get("started_at")
        if seen and seen >= cutoff:
            return dict(doc)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        db_client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.error("database unavailable: %s", db_client.redact(str(exc)))
        return 3

    deadline = time.monotonic() + args.wait * 60
    while True:
        run = active_run(args.stale_after)
        if run is None:
            logger.info("no other updater is running — clear to start")
            return 0

        seen = run.get("heartbeat_at") or run.get("started_at")
        counters = run.get("counters") or {}
        logger.warning(
            "another updater holds the lock: %s (kind=%s, started %s, last seen %s, "
            "%s/%s tickers done)",
            run.get("_id"), run.get("kind"), run.get("started_at"), seen,
            counters.get("tickers_done", "?"), counters.get("tickers_total", "?"),
        )
        if time.monotonic() >= deadline:
            logger.warning("giving up; skipping this run rather than racing it")
            return BUSY
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
