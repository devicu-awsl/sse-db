#!/usr/bin/env python3
"""Connection check — the first box on the pre-coding checklist.

Verifies the Atlas cluster answers, the Stable API is pinned, the target
database is reachable and reports current storage against the M0 cap.
Prints nothing that could leak a credential.

    python scripts/ping_atlas.py
"""

from __future__ import annotations

import argparse
import sys

import _path  # noqa: F401

from sse.config import get_settings
from sse.db import client as db_client
from sse.db.schema import format_storage, storage_stats
from sse.logging_setup import setup_logging

logger = setup_logging("ping")


def main() -> int:
    # Parsed first so `--help` works without a configured .env or a reachable
    # cluster: a diagnostic whose help text needs the thing it diagnoses is no use.
    argparse.ArgumentParser(description=__doc__).parse_args()
    settings = get_settings()
    try:
        settings.require_uri()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 3

    try:
        db_client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.error("could not reach the cluster: %s", db_client.redact(str(exc)))
        return 3

    info = db_client.server_info()
    logger.info("pinged your deployment — connected to MongoDB %s", info.get("version"))

    db = db_client.get_db()
    logger.info("database: %s", settings.db_name)
    logger.info("collections: %s", ", ".join(sorted(db.list_collection_names())) or "(none yet)")
    try:
        logger.info("storage: %s", format_storage(storage_stats(db)))
    except Exception as exc:  # noqa: BLE001 - dbStats needs an existing database
        logger.info("storage: not available yet (%s)", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
