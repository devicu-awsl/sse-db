#!/usr/bin/env python3
"""Create collections and indexes.  Idempotent — safe to re-run.

    python scripts/init_db.py [--drop-indexes]
"""

from __future__ import annotations

import argparse
import sys

import _path  # noqa: F401

from sse.db import client as db_client
from sse.db.schema import COLLECTIONS, ensure_collections, ensure_indexes
from sse.logging_setup import setup_logging

logger = setup_logging("init_db")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drop-indexes",
        action="store_true",
        help="drop non-_id indexes before recreating them (needed after a definition change)",
    )
    args = parser.parse_args()

    try:
        db_client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.error("database unavailable: %s", db_client.redact(str(exc)))
        return 3

    db = db_client.get_db()
    created = ensure_collections(db)
    logger.info("collections present: %s", ", ".join(COLLECTIONS))
    if created:
        logger.info("created: %s", ", ".join(created))

    if args.drop_indexes:
        for collection in COLLECTIONS:
            db[collection].drop_indexes()
            logger.warning("dropped indexes on %s", collection)

    ensure_indexes(db)
    logger.info("schema ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
