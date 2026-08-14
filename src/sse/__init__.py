"""SSE statistical-arbitrage data system.

A system for fetching, storing and serving the complete daily history of
equities traded on the Shanghai Stock Exchange, built as the foundation for a
pairs-trading study of the China A-share market.

Layers, from the outside in::

    api/        FastAPI surface, mounted at /v1
    db/         repositories — the seam that hides physical storage
    codec       binary column encoding for one ticker-year
    ingest/     universe discovery, Yahoo provider, quality layer
    research/   implemented Phase 2 correlation, clustering and backtest layer

Nothing above ``db.repository`` knows that prices are stored as packed binary
columns; that is what allows the storage design to change in a later task
without touching the API or the research code.
"""

from __future__ import annotations

#: Stable public package/API contract version.  This is intentionally distinct
#: from the Python distribution version (``0.1.0`` in pyproject.toml) and from
#: archive labels such as v1.11, which identify review snapshots rather than a
#: new public contract.
__version__ = "1.0.0"
__all__ = ["__version__"]
