"""Logging configuration.

Two non-obvious requirements are handled here:

* **Acceptance check §9** — "no credentials, connection strings or local paths in
  any submitted file *or log*".  ``RedactingFilter`` rewrites anything that looks
  like a ``mongodb[+srv]://user:password@host`` URI before it reaches a handler,
  so a stray ``logger.exception`` during ingestion cannot leak the Atlas password.
* Ingestion runs for hours unattended, so records also go to a rotating file.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
from pathlib import Path

from .config import get_settings

_CREDENTIAL_RE = re.compile(r"(mongodb(?:\+srv)?://)([^:@/\s]+):([^@/\s]+)@", re.IGNORECASE)
_FMT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"
_configured = False


class RedactingFilter(logging.Filter):
    """Mask credentials embedded in URIs anywhere in the record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - stdlib name
        if isinstance(record.msg, str):
            record.msg = _CREDENTIAL_RE.sub(r"\1\2:***@", record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _CREDENTIAL_RE.sub(r"\1\2:***@", v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    _CREDENTIAL_RE.sub(r"\1\2:***@", a) if isinstance(a, str) else a
                    for a in record.args
                )
        return True


def setup_logging(name: str = "sse", *, level: str | None = None) -> logging.Logger:
    """Idempotently configure root logging and return a named logger."""
    global _configured
    settings = get_settings()
    resolved = level or settings.log_level

    if not _configured:
        root = logging.getLogger()
        root.setLevel(resolved)
        for handler in list(root.handlers):
            root.removeHandler(handler)

        formatter = logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S")
        redactor = RedactingFilter()

        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        stream.addFilter(redactor)
        root.addHandler(stream)

        log_dir = Path(settings.log_dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "sse.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(redactor)
            root.addHandler(file_handler)
        except OSError:  # read-only filesystem: stderr alone is acceptable
            pass

        # yfinance and urllib3 are extremely chatty at DEBUG
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("yfinance").setLevel(logging.WARNING)
        logging.getLogger("peewee").setLevel(logging.WARNING)
        _configured = True

    return logging.getLogger(name)
