#!/usr/bin/env python3
"""Discover the SSE universe and write it to ``securities`` + ``universe_snapshots``.

    python scripts/discover_universe.py                 # screener + code probe
    python scripts/discover_universe.py --no-probe      # screener only (fast)
    python scripts/discover_universe.py --prefixes 600 601 603 605 688

The screener alone silently truncates.  The code-range probe is the independent
cross-check, and the reconciliation table it produces — how many names each
method found, and how many only one of them found — is the evidence behind the
universe slide.  It is slow (~7,000 candidate codes), so it is skippable during
development and run once for real.

Exit codes: 0 ok, 2 the two methods disagree beyond tolerance, 3 database down.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import _path  # noqa: F401

from sse.db import client as db_client
from sse.db.repository import RunCounters, RunRepository, SecurityRepository, SnapshotRepository
from sse.ingest import universe
from sse.logging_setup import setup_logging

logger = setup_logging("discover")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-probe", action="store_true", help="skip the code-range cross-check")
    parser.add_argument("--prefixes", nargs="*", default=None, help="code blocks to probe")
    parser.add_argument("--resume-probe", action="store_true",
                        help="skip codes already confirmed in an earlier snapshot")
    parser.add_argument("--disagreement-tolerance", type=float, default=0.10,
                        help="fraction of the union either method may miss before exit code 2")
    parser.add_argument("--report", type=Path, default=Path("reports/universe.json"))
    args = parser.parse_args()

    try:
        db_client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.error("database unavailable: %s", db_client.redact(str(exc)))
        return 3

    db = db_client.get_db()
    securities = SecurityRepository(db)
    snapshots = SnapshotRepository(db)
    runs = RunRepository(db)

    run_id = f"universe-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}"
    runs.start(run_id, "universe", {"probe": not args.no_probe, "prefixes": args.prefixes})
    counters = RunCounters()

    already = snapshots.union() if args.resume_probe else set()

    try:
        result = universe.discover(
            run_probe=not args.no_probe,
            prefixes=args.prefixes,
            skip_probed=already,
        )
        stats = universe.persist(result, securities, snapshots, run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("discovery failed")
        counters.errors.append({"stage": "discover", "error": str(exc)[:500]})
        runs.finish(run_id, status="failed", counters=counters, note=str(exc)[:500])
        return 1

    reconciliation = stats["reconciliation"]
    counters.tickers_total = reconciliation["union_equities"]
    counters.tickers_done = reconciliation["union_equities"]
    runs.finish(run_id, status="ok", counters=counters, note=json.dumps(reconciliation))

    logger.info("--- universe reconciliation (equities only) ---")
    for key, value in reconciliation.items():
        if key in ("instruments", "unknown_blocks"):
            continue
        logger.info("  %-22s %s", key, value)
    logger.info("  instruments returned by the screener:")
    for name, count in reconciliation["instruments"].items():
        marker = "  <- kept" if name.startswith("equity") else ""
        logger.info("      %-20s %5d%s", name, count, marker)
    for band in result.unresolved_bands:
        logger.warning(
            "  screener could not enumerate band [%.3g, %s): %s",
            band["low"],
            f"{band['high']:.3g}" if band["high"] is not None else "inf",
            band["unresolvable"],
        )
    logger.info("securities upserted=%s inactivated=%s", stats.get("seen"), stats.get("inactivated"))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "taken_at": stats["taken_at"],
                "reconciliation": reconciliation,
                "bands": result.bands,
                "probe_only": sorted(result.probe_only)[:200],
                "screener_only": sorted(result.screener_only)[:200],
                "unresolved_bands": result.unresolved_bands,
                "probe_unresolved": sorted(result.unknown_symbols)[:200],
                "non_equity_excluded": sorted(result.non_equity)[:200],
                "instruments": result.instrument_counts(),
                "unknown_blocks": result.unknown_blocks(),
                "by_board": _board_counts(result),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", args.report)

    if not args.no_probe:
        # Compared equity-to-equity: the screener's fund hits are excluded before
        # this ratio is formed, so it measures method disagreement rather than the
        # difference between two populations.
        worst_miss = reconciliation["disagreement_fraction"]
        if worst_miss > args.disagreement_tolerance:
            logger.warning(
                "methods disagree on %.1f%% of the equity union — inspect %s before backfilling",
                worst_miss * 100,
                args.report,
            )
            return 2
        if reconciliation["probe_unresolved"]:
            logger.warning(
                "%d codes were never resolved (network failures, not absences); re-run "
                "with --resume-probe before treating the universe as complete",
                reconciliation["probe_unresolved"],
            )
            return 2
    return 0


def _board_counts(result: universe.DiscoveryResult) -> dict[str, int]:
    """Boards of the *kept* universe — equities only."""
    counts: dict[str, int] = {}
    for symbol in result.symbols:
        board = result.metadata.get(symbol, {}).get("board", "other")
        counts[board] = counts.get(board, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    sys.exit(main())
