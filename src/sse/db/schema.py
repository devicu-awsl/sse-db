"""Physical schema: collections, indexes and storage accounting.

Four collections, fixed by the pre-coding checklist and not to be added to
without a schema-version bump:

``securities``
    One document per Yahoo symbol ever seen on the SSE.  The point-in-time
    universe cannot be reconstructed from Yahoo, so a symbol that disappears
    from the screener is marked ``inactive`` and *never deleted* — the union
    across snapshots is the best available approximation of the historical
    universe, and deleting would make survivorship bias worse than it already
    is (see README §Known limitations).

``universe_snapshots``
    The raw result of one discovery run, kept verbatim: which query produced
    it, which market-cap bands saturated, how many symbols the independent
    code-range probe found.  This is the audit trail behind the universe slide.

``price_chunks``
    One document per ticker-year of packed binary columns (see ``sse.codec``).

``ingestion_runs``
    One document per backfill/update/discovery run: counters, failures,
    storage reading at the end.  Makes every ingestion reproducible and gives
    the updater a place to record a non-zero exit.
"""

from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.database import Database
from pymongo.errors import OperationFailure

from ..config import get_settings
from ..logging_setup import setup_logging

logger = setup_logging(__name__)

SECURITIES = "securities"
UNIVERSE_SNAPSHOTS = "universe_snapshots"
PRICE_CHUNKS = "price_chunks"
INGESTION_RUNS = "ingestion_runs"

COLLECTIONS: tuple[str, ...] = (SECURITIES, UNIVERSE_SNAPSHOTS, PRICE_CHUNKS, INGESTION_RUNS)

INDEXES: dict[str, list[IndexModel]] = {
    SECURITIES: [
        IndexModel([("ticker", ASCENDING)], name="uq_ticker", unique=True),
        IndexModel([("status", ASCENDING), ("board", ASCENDING)], name="ix_status_board"),
        IndexModel([("code", ASCENDING)], name="ix_code"),
        # supports "which tickers still need backfilling?" without a collection scan
        IndexModel([("ingest.state", ASCENDING), ("ticker", ASCENDING)], name="ix_ingest_state"),
    ],
    UNIVERSE_SNAPSHOTS: [
        IndexModel([("taken_at", DESCENDING)], name="ix_taken_at"),
        IndexModel([("source", ASCENDING), ("taken_at", DESCENDING)], name="ix_source_taken_at"),
    ],
    PRICE_CHUNKS: [
        # _id is already "{ticker}:{year}" and unique; the compound unique index is
        # what the *queries* use (range scan over years for one ticker) and it
        # guarantees the invariant independently of the _id convention.
        IndexModel([("ticker", ASCENDING), ("year", ASCENDING)], name="uq_ticker_year", unique=True),
        IndexModel([("last", DESCENDING)], name="ix_last"),
        IndexModel([("schema_version", ASCENDING)], name="ix_schema_version"),
    ],
    INGESTION_RUNS: [
        IndexModel([("started_at", DESCENDING)], name="ix_started_at"),
        IndexModel([("kind", ASCENDING), ("started_at", DESCENDING)], name="ix_kind_started_at"),
        IndexModel([("status", ASCENDING)], name="ix_status"),
    ],
}


def ensure_collections(db: Database) -> list[str]:
    """Create any missing collection.  Safe to run repeatedly."""
    existing = set(db.list_collection_names())
    created = []
    for name in COLLECTIONS:
        if name not in existing:
            db.create_collection(name)
            created.append(name)
            logger.info("created collection %s", name)
    return created


def ensure_indexes(db: Database) -> dict[str, list[str]]:
    """Create indexes idempotently; report what each collection now has."""
    result: dict[str, list[str]] = {}
    for collection, models in INDEXES.items():
        try:
            db[collection].create_indexes(models)
        except OperationFailure as exc:
            # An index that already exists with different options must be dropped
            # deliberately — never silently, or a uniqueness guarantee could vanish.
            logger.error("index creation failed on %s: %s", collection, exc)
            raise
        result[collection] = sorted(db[collection].index_information())
        logger.info("indexes on %-20s -> %s", collection, ", ".join(result[collection]))
    return result


def storage_stats(db: Database) -> dict[str, Any]:
    """Storage accounting against the Atlas M0 cap.

    M0 bills *compressed* on-disk size, so ``storageSize + indexSize`` is the
    figure that matters, not the uncompressed ``dataSize`` — though both are
    reported because the ratio between them is the compression evidence for the
    storage slide.
    """
    settings = get_settings()
    stats = db.command("dbStats")
    data_size = int(stats.get("dataSize", 0))
    storage_size = int(stats.get("storageSize", 0))
    index_size = int(stats.get("indexSize", 0))
    used = storage_size + index_size
    cap = settings.storage_cap_bytes
    return {
        "data_size": data_size,
        "storage_size": storage_size,
        "index_size": index_size,
        "used_bytes": used,
        "cap_bytes": cap,
        "used_fraction": used / cap if cap else 0.0,
        "halt_fraction": settings.storage_halt_fraction,
        "over_threshold": used >= settings.storage_halt_bytes,
        "compression_ratio": (data_size / storage_size) if storage_size else 0.0,
        "objects": int(stats.get("objects", 0)),
    }


def format_storage(stats: dict[str, Any]) -> str:
    mib = 1024 * 1024
    return (
        f"{stats['used_bytes'] / mib:.1f} MiB of {stats['cap_bytes'] / mib:.0f} MiB "
        f"({stats['used_fraction']:.1%}; data {stats['data_size'] / mib:.1f} MiB, "
        f"index {stats['index_size'] / mib:.1f} MiB, "
        f"compression {stats['compression_ratio']:.2f}x)"
    )
