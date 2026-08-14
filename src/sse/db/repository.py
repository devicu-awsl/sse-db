"""Repositories — the seam between physical storage and everything above it.

The brief says the database and API "will be extended in future tasks", so the
one architectural commitment worth making now is that *nothing outside this
module knows that prices live in binary chunks*.  The API, the research loader
and the notebooks all speak :class:`~sse.codec.BarSeries` and see only the
:class:`PriceStore` protocol.  Swapping chunks for a time-series collection, a
Parquet lake or an intraday store in Task 2/3 means writing one new class here
and changing one line of wiring.

Everything is idempotent by construction: chunk ``_id`` is ``{ticker}:{year}``,
so a re-run overwrites rather than duplicates, and a chunk whose checksum is
unchanged is not written at all.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
from pymongo import ReplaceOne, UpdateOne
from pymongo.database import Database

from .. import codec
from ..codec import BarSeries, encode_chunk
from ..logging_setup import setup_logging
from .schema import INGESTION_RUNS, PRICE_CHUNKS, SECURITIES, UNIVERSE_SNAPSHOTS

logger = setup_logging(__name__)

#: BSON fields that must accompany any projected read for the chunk to be decodable.
_CHUNK_META_FIELDS = ("_id", "ticker", "year", "n", "first", "last", "codec", "schema_version",
                      "price_scale", "sha256", "date")


class Outcome(str, Enum):
    INSERTED = "inserted"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class Coverage:
    ticker: str
    first: dt.date | None
    last: dt.date | None
    bars: int
    years: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "first": self.first.isoformat() if self.first else None,
            "last": self.last.isoformat() if self.last else None,
            "bars": self.bars,
            "years": self.years,
        }


class PriceStore(Protocol):
    """The only contract the API and research layers may depend on."""

    def get_series(
        self,
        ticker: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
        fields: Sequence[str] | None = None,
    ) -> BarSeries: ...

    def get_many(
        self,
        tickers: Sequence[str],
        start: dt.date | None = None,
        end: dt.date | None = None,
        fields: Sequence[str] | None = None,
    ) -> dict[str, BarSeries]: ...

    def merge_series(self, ticker: str, bars: BarSeries) -> dict[Outcome, int]: ...

    def coverage(self, ticker: str) -> Coverage | None: ...

    def tickers(self) -> list[str]: ...


# --------------------------------------------------------------------------- prices


class MongoPriceRepository:
    """Chunked binary-column storage backed by ``price_chunks``."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._chunks = db[PRICE_CHUNKS]

    # -- reads --------------------------------------------------------------

    @staticmethod
    def _projection(fields: Sequence[str] | None) -> dict[str, int]:
        """Pull only the requested binary columns off the wire.

        A ten-year, one-column request should not transfer ten years of six
        columns.  ``date`` and the metadata fields are always included because
        the chunk is undecodable without them.
        """
        projection = {name: 1 for name in _CHUNK_META_FIELDS}
        wanted = tuple(fields) if fields else codec.DATA_FIELDS
        for name in wanted:
            if name in codec.DATA_FIELDS:
                projection[name] = 1
        return projection

    def _year_filter(self, start: dt.date | None, end: dt.date | None) -> dict[str, Any]:
        constraint: dict[str, int] = {}
        if start is not None:
            constraint["$gte"] = start.year
        if end is not None:
            constraint["$lte"] = end.year
        return {"year": constraint} if constraint else {}

    def get_series(
        self,
        ticker: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
        fields: Sequence[str] | None = None,
    ) -> BarSeries:
        query: dict[str, Any] = {"ticker": ticker, **self._year_filter(start, end)}
        cursor = self._chunks.find(query, self._projection(fields)).sort("year", 1)
        decoded = [codec.decode_chunk(doc, fields=fields) for doc in cursor]
        if not decoded:
            return BarSeries.empty(ticker)
        series = codec.concat(decoded)
        series.ticker = ticker
        return series.slice_dates(start, end)

    def get_many(
        self,
        tickers: Sequence[str],
        start: dt.date | None = None,
        end: dt.date | None = None,
        fields: Sequence[str] | None = None,
    ) -> dict[str, BarSeries]:
        """One round trip for the whole cross-section.

        The research layer needs ~2,000 tickers at once; issuing 2,000 sequential
        queries (or HTTP calls) is not a design.
        """
        if not tickers:
            return {}
        query: dict[str, Any] = {"ticker": {"$in": list(tickers)}, **self._year_filter(start, end)}
        cursor = self._chunks.find(query, self._projection(fields)).sort([("ticker", 1), ("year", 1)])

        grouped: dict[str, list[BarSeries]] = {}
        for doc in cursor:
            grouped.setdefault(doc["ticker"], []).append(codec.decode_chunk(doc, fields=fields))

        out: dict[str, BarSeries] = {}
        for ticker in tickers:
            parts = grouped.get(ticker)
            if not parts:
                out[ticker] = BarSeries.empty(ticker)
                continue
            series = codec.concat(parts)
            series.ticker = ticker
            out[ticker] = series.slice_dates(start, end)
        return out

    def get_chunk(self, ticker: str, year: int) -> Mapping[str, Any] | None:
        """Full chunk document, checksum included — used by the integrity audit."""
        return self._chunks.find_one({"_id": codec.chunk_id(ticker, year)})

    def coverage(self, ticker: str) -> Coverage | None:
        rows = self._coverage_pipeline({"ticker": ticker})
        return rows.get(ticker)

    def coverage_many(self, tickers: Sequence[str] | None = None) -> dict[str, Coverage]:
        match = {"ticker": {"$in": list(tickers)}} if tickers else {}
        return self._coverage_pipeline(match)

    def _coverage_pipeline(self, match: dict[str, Any]) -> dict[str, Coverage]:
        pipeline: list[dict[str, Any]] = []
        if match:
            pipeline.append({"$match": match})
        pipeline.append(
            {
                "$group": {
                    "_id": "$ticker",
                    "first": {"$min": "$first"},
                    "last": {"$max": "$last"},
                    "bars": {"$sum": "$n"},
                    "years": {"$sum": 1},
                }
            }
        )
        out: dict[str, Coverage] = {}
        for row in self._chunks.aggregate(pipeline):
            out[row["_id"]] = Coverage(
                ticker=row["_id"],
                first=row["first"].date() if row.get("first") else None,
                last=row["last"].date() if row.get("last") else None,
                bars=int(row.get("bars", 0)),
                years=int(row.get("years", 0)),
            )
        return out

    def last_date(self, ticker: str) -> dt.date | None:
        """Watermark for the incremental updater."""
        doc = self._chunks.find_one({"ticker": ticker}, {"last": 1}, sort=[("year", -1)])
        return doc["last"].date() if doc and doc.get("last") else None

    def last_dates(self, tickers: Sequence[str] | None = None) -> dict[str, dt.date]:
        return {t: c.last for t, c in self.coverage_many(tickers).items() if c.last}

    def tickers(self) -> list[str]:
        return sorted(self._chunks.distinct("ticker"))

    def stored_bar_count(self) -> int:
        rows = list(self._chunks.aggregate([{"$group": {"_id": None, "bars": {"$sum": "$n"}}}]))
        return int(rows[0]["bars"]) if rows else 0

    def iter_chunks(self, ticker: str | None = None) -> Iterable[Mapping[str, Any]]:
        """Full documents, for checksum verification sweeps."""
        query = {"ticker": ticker} if ticker else {}
        return self._chunks.find(query).sort([("ticker", 1), ("year", 1)])

    # -- writes -------------------------------------------------------------

    def upsert_chunk(self, ticker: str, year: int, bars: BarSeries) -> Outcome:
        """Write one ticker-year, skipping the write when nothing changed.

        Yahoo rewrites ``adj_close`` history retroactively after a split, so the
        weekly full refresh re-encodes every year of every ticker.  Comparing
        checksums first turns that sweep from ~34,000 writes into a handful.
        """
        doc = encode_chunk(ticker, year, bars)
        existing = self._chunks.find_one({"_id": doc["_id"]}, {"sha256": 1})
        if existing and existing.get("sha256") == doc["sha256"]:
            return Outcome.UNCHANGED
        doc["updated_at"] = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        self._chunks.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        return Outcome.UPDATED if existing else Outcome.INSERTED

    def upsert_series(self, ticker: str, bars: BarSeries) -> dict[Outcome, int]:
        """Split a complete series into chunks and replace each ticker-year."""
        counts = {outcome: 0 for outcome in Outcome}
        by_year = bars.split_by_year()
        if not by_year:
            return counts

        encoded = {
            year: encode_chunk(ticker, year, part) for year, part in sorted(by_year.items())
        }
        known = {
            row["_id"]: row.get("sha256")
            for row in self._chunks.find({"_id": {"$in": [d["_id"] for d in encoded.values()]}},
                                         {"sha256": 1})
        }

        now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        operations = []
        for doc in encoded.values():
            previous = known.get(doc["_id"], ...)
            if previous == doc["sha256"]:
                counts[Outcome.UNCHANGED] += 1
                continue
            doc["updated_at"] = now
            operations.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
            counts[Outcome.INSERTED if previous is ... else Outcome.UPDATED] += 1

        if operations:
            self._chunks.bulk_write(operations, ordered=False)
        return counts

    @staticmethod
    def _merge_year(existing: BarSeries, incoming: BarSeries, ticker: str) -> BarSeries:
        """Merge a refreshed window into one ticker-year without losing history.

        The incremental updater deliberately downloads only a recent overlap
        window.  A normal ``upsert_series`` replaces a whole ticker-year chunk,
        which would otherwise discard every older bar in that year.  Concatenating
        existing rows first and refreshed rows second, followed by a stable sort
        and last-row-per-date selection, makes the refreshed value win on an
        overlapping date while retaining rows outside the window.
        """
        if len(existing) == 0:
            incoming.ticker = ticker
            return incoming
        if len(incoming) == 0:
            existing.ticker = ticker
            return existing

        dates = np.concatenate([existing.dates, incoming.dates])
        order = np.argsort(dates.astype("int64"), kind="stable")
        sorted_dates = dates[order]
        keep = np.ones(len(sorted_dates), dtype=bool)
        # Existing rows precede incoming rows, so the final row at a duplicate
        # date is the refreshed value.
        keep[:-1] = sorted_dates[:-1] != sorted_dates[1:]

        kwargs: dict[str, np.ndarray] = {}
        for name in codec.DATA_FIELDS:
            old = getattr(existing, name)
            new = getattr(incoming, name)
            if old is None and new is None:
                continue
            old_values = np.full(len(existing), np.nan) if old is None else old
            new_values = np.full(len(incoming), np.nan) if new is None else new
            values = np.concatenate([old_values, new_values])[order]
            kwargs[name] = values[keep]

        return BarSeries(dates=sorted_dates[keep], ticker=ticker, **kwargs)

    def merge_series(self, ticker: str, bars: BarSeries) -> dict[Outcome, int]:
        """Merge partial refreshed data, preserving rows outside the window.

        This is the write path for incremental updates.  It intentionally reads
        only the affected years; full refreshes should continue to use
        :meth:`upsert_series` so that a vendor deletion is not masked by stale
        local rows.
        """
        counts = {outcome: 0 for outcome in Outcome}
        for year, incoming in sorted(bars.split_by_year().items()):
            doc = self._chunks.find_one({"_id": codec.chunk_id(ticker, year)})
            existing = codec.decode_chunk(doc) if doc else BarSeries.empty(ticker)
            merged = self._merge_year(existing, incoming, ticker)
            outcome = self.upsert_chunk(ticker, year, merged)
            counts[outcome] += 1
        return counts

    def delete_ticker(self, ticker: str) -> int:
        return self._chunks.delete_many({"ticker": ticker}).deleted_count


# --------------------------------------------------------------------------- securities


class SecurityRepository:
    """The security master.  Symbols are marked inactive, never deleted."""

    def __init__(self, db: Database) -> None:
        self._col = db[SECURITIES]

    def upsert_many(self, securities: Sequence[Mapping[str, Any]], *, source: str) -> dict[str, int]:
        """Merge discovery results into the master.

        ``$set`` + ``$setOnInsert``, never ``ReplaceOne``: a replace would delete
        ``first_seen`` and — far worse — the per-ticker ``ingest`` watermark that
        makes ``backfill.py --resume`` work.  The nightly updater refreshes the
        universe before it fetches prices, so a replace here would silently reset
        the ingestion state of the entire universe every night.
        """
        if not securities:
            return {"seen": 0, "inserted": 0, "modified": 0}
        now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        operations = []
        for security in securities:
            ticker = security["ticker"]
            payload = {k: v for k, v in security.items() if k != "ticker" and v is not None}
            operations.append(
                UpdateOne(
                    {"ticker": ticker},
                    {
                        "$set": {
                            **payload,
                            "status": "active",
                            "last_seen": now,
                            "last_source": source,
                        },
                        "$setOnInsert": {"ticker": ticker, "first_seen": now},
                        # a name that reappears is active again, and the old
                        # deactivation date would be misleading
                        "$unset": {"inactive_since": ""},
                    },
                    upsert=True,
                )
            )
        result = self._col.bulk_write(operations, ordered=False)
        return {
            "seen": len(securities),
            "inserted": result.upserted_count,
            "modified": result.modified_count,
        }

    def mark_inactive(self, missing: Iterable[str]) -> int:
        missing = list(missing)
        if not missing:
            return 0
        return self._col.update_many(
            {"ticker": {"$in": missing}, "status": "active"},
            {"$set": {"status": "inactive", "inactive_since": dt.datetime.now(dt.UTC).replace(tzinfo=None)}},
        ).modified_count

    def list_tickers(self, status: str | None = "active", *, board: str | None = None) -> list[str]:
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        if board:
            query["board"] = board
        return sorted(doc["ticker"] for doc in self._col.find(query, {"ticker": 1}))

    def get(self, ticker: str) -> Mapping[str, Any] | None:
        return self._col.find_one({"ticker": ticker}, {"_id": 0})

    def find(
        self,
        *,
        status: str | None = None,
        board: str | None = None,
        query: str | None = None,
        limit: int = 100,
        skip: int = 0,
    ) -> tuple[list[Mapping[str, Any]], int]:
        filters: dict[str, Any] = {}
        if status:
            filters["status"] = status
        if board:
            filters["board"] = board
        if query:
            filters["$or"] = [
                {"ticker": {"$regex": query, "$options": "i"}},
                {"name": {"$regex": query, "$options": "i"}},
            ]
        total = self._col.count_documents(filters)
        cursor = self._col.find(filters, {"_id": 0}).sort("ticker", 1).skip(skip).limit(limit)
        return list(cursor), total

    def exists(self, ticker: str) -> bool:
        return self._col.count_documents({"ticker": ticker}, limit=1) > 0

    def set_ingest_state(self, ticker: str, state: str, **extra: Any) -> None:
        """Per-ticker ingestion watermark — what makes the backfill restartable."""
        payload = {f"ingest.{k}": v for k, v in extra.items()}
        payload["ingest.state"] = state
        payload["ingest.updated_at"] = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        self._col.update_one({"ticker": ticker}, {"$set": payload})

    def pending(self, run_id: str) -> list[str]:
        """Tickers not yet completed *within this run* — resume after a crash."""
        return sorted(
            doc["ticker"]
            for doc in self._col.find(
                {"status": "active", "$or": [
                    {"ingest.run_id": {"$ne": run_id}},
                    {"ingest.state": {"$nin": ["ok", "empty"]}},
                ]},
                {"ticker": 1},
            )
        )

    def count_by_status(self) -> dict[str, int]:
        rows = self._col.aggregate([{"$group": {"_id": "$status", "n": {"$sum": 1}}}])
        return {row["_id"] or "unknown": int(row["n"]) for row in rows}


# --------------------------------------------------------------------------- snapshots


class SnapshotRepository:
    """Every universe discovery run is kept verbatim as evidence."""

    def __init__(self, db: Database) -> None:
        self._col = db[UNIVERSE_SNAPSHOTS]

    def record(
        self,
        *,
        source: str,
        symbols: Sequence[str],
        run_id: str,
        details: Mapping[str, Any] | None = None,
    ) -> str:
        doc = {
            "run_id": run_id,
            "source": source,
            "taken_at": dt.datetime.now(dt.UTC).replace(tzinfo=None),
            "n_symbols": len(symbols),
            "symbols": sorted(symbols),
            "details": dict(details or {}),
        }
        return str(self._col.insert_one(doc).inserted_id)

    def latest(self, source: str | None = None) -> Mapping[str, Any] | None:
        query = {"source": source} if source else {}
        return self._col.find_one(query, sort=[("taken_at", -1)])

    def union(self) -> set[str]:
        """The universe is the union across all snapshots, never the latest one."""
        symbols: set[str] = set()
        for doc in self._col.find({}, {"symbols": 1}):
            symbols.update(doc.get("symbols", []))
        return symbols

    def history(self, limit: int = 20) -> list[Mapping[str, Any]]:
        return list(self._col.find({}, {"symbols": 0}).sort("taken_at", -1).limit(limit))


# --------------------------------------------------------------------------- runs


@dataclass
class RunCounters:
    tickers_total: int = 0
    tickers_done: int = 0
    tickers_failed: int = 0
    chunks_inserted: int = 0
    chunks_updated: int = 0
    chunks_unchanged: int = 0
    bars_written: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tickers_total": self.tickers_total,
            "tickers_done": self.tickers_done,
            "tickers_failed": self.tickers_failed,
            "chunks_inserted": self.chunks_inserted,
            "chunks_updated": self.chunks_updated,
            "chunks_unchanged": self.chunks_unchanged,
            "bars_written": self.bars_written,
            "errors": self.errors[:200],  # keep the document well under 16 MB
            "error_count": len(self.errors),
        }


class RunRepository:
    """Ingestion run log — makes every load reproducible and auditable."""

    def __init__(self, db: Database) -> None:
        self._col = db[INGESTION_RUNS]

    def start(self, run_id: str, kind: str, params: Mapping[str, Any] | None = None) -> None:
        self._col.replace_one(
            {"_id": run_id},
            {
                "_id": run_id,
                "kind": kind,
                "status": "running",
                "started_at": dt.datetime.now(dt.UTC).replace(tzinfo=None),
                "params": dict(params or {}),
            },
            upsert=True,
        )

    def heartbeat(self, run_id: str, counters: RunCounters) -> None:
        self._col.update_one(
            {"_id": run_id},
            {"$set": {"counters": counters.as_dict(),
                      "heartbeat_at": dt.datetime.now(dt.UTC).replace(tzinfo=None)}},
        )

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        counters: RunCounters,
        storage: Mapping[str, Any] | None = None,
        note: str | None = None,
    ) -> None:
        self._col.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": status,
                    "finished_at": dt.datetime.now(dt.UTC).replace(tzinfo=None),
                    "counters": counters.as_dict(),
                    "storage": dict(storage or {}),
                    "note": note,
                }
            },
        )

    def latest(self, kind: str | None = None) -> Mapping[str, Any] | None:
        query = {"kind": kind} if kind else {}
        return self._col.find_one(query, sort=[("started_at", -1)])

    def recent(self, limit: int = 10) -> list[Mapping[str, Any]]:
        return list(self._col.find({}, {"counters.errors": 0}).sort("started_at", -1).limit(limit))
