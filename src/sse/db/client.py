"""MongoDB connection management.

One client per process, held for the process lifetime: ``MongoClient`` owns a
connection pool and is thread-safe, so creating one per call would defeat
pooling and exhaust Atlas M0's 500-connection ceiling during a threaded
backfill.

``ServerApi("1")`` pins the Stable API, which is what makes it safe to promise
the supervisor that "the database will be extended in future tasks" — a server
upgrade cannot change behaviour under a pinned API version.
"""

from __future__ import annotations

import re
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from pymongo.server_api import ServerApi

from ..config import get_settings
from ..logging_setup import setup_logging

logger = setup_logging(__name__)

_client: MongoClient | None = None
_CREDENTIAL_RE = re.compile(r"(mongodb(?:\+srv)?://)([^:@/\s]+):([^@/\s]+)@", re.IGNORECASE)


class DatabaseUnavailable(RuntimeError):
    """Raised when the cluster cannot be reached — surfaces as HTTP 503."""


def redact(text: str) -> str:
    """Mask the password in any connection string before it is shown or logged."""
    return _CREDENTIAL_RE.sub(r"\1\2:***@", text)


def get_client() -> MongoClient:
    """Return the process-wide client, creating it on first use."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = MongoClient(
            settings.require_uri(),
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=settings.server_selection_timeout_ms,
            connectTimeoutMS=settings.server_selection_timeout_ms,
            retryWrites=True,
            appname="sse-statarb",
            # Backfill runs `max_workers` threads plus the main thread; a small
            # pool keeps M0 well inside its connection limit.
            maxPoolSize=max(8, settings.max_workers + 4),
            tz_aware=False,
        )
        logger.debug("MongoClient created for %s", redact(settings.require_uri()))
    return _client


def get_db(name: str | None = None) -> Database:
    return get_client()[name or get_settings().db_name]


def ping(*, raise_on_failure: bool = True) -> bool:
    """Round-trip the cluster.  Used by scripts at start-up and by ``/v1/health``."""
    try:
        get_client().admin.command("ping")
        return True
    except (ServerSelectionTimeoutError, PyMongoError) as exc:
        message = redact(str(exc))
        logger.error("MongoDB ping failed: %s", message)
        if raise_on_failure:
            raise DatabaseUnavailable(message) from None
        return False


def server_info() -> dict[str, Any]:
    info = get_client().server_info()
    return {"version": info.get("version"), "ok": info.get("ok")}


def close_client() -> None:
    """Close the pool — for tests and for clean script shutdown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
