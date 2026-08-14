"""Binary column codec for daily bars (``cols-v1``).

Why a codec at all
------------------
Atlas M0 gives 512 MiB *including indexes*.  A row-per-bar document costs
~130-180 B once BSON field names, the ``_id`` and the compound index are paid
for; at ~2,300 tickers x ~15 years x ~243 sessions (~8.4 M bars) that does not
fit.  Storing each ticker-year as a set of packed binary columns costs::

    date       int32   4 B      days since 1970-01-01
    open       int32   4 B      price x 10_000
    high       int32   4 B
    low        int32   4 B
    close      int32   4 B
    adj_close  int32   4 B
    volume     int64   8 B
                     ----
                      32 B/bar   + ~250 B of per-document BSON overhead
                                   amortised over ~243 bars  ->  ~33 B/bar

Roadmap §2 conflict 1 settled this against Parquet-per-chunk (51.5 B/bar):
Parquet's footer and row-group metadata do not amortise over only ~243 rows.

Why fixed point rather than float32
-----------------------------------
float32 has 24 bits of mantissa, so a price near 1,800 CNY (Kweichow Moutai
trades on the SSE) resolves to ~1e-4 *at best* and rounds unpredictably.
int32 scaled by 10,000 is exact to 4 dp for anything below 214,748.36 and costs
the same 4 bytes.  Exactness matters: the chunk checksum must be reproducible.

Missing data
------------
Never forward-filled and never zero-filled — an A-share suspension is a real
event, not a gap to paper over.  A bar that Yahoo reports with a NaN field is
stored with ``INT32_MISSING`` / ``INT64_MISSING`` sentinels and decodes back to
NaN; a session with no bar at all simply has no entry in the date column.

Integrity
---------
``sha256`` is taken over the concatenated buffers in a fixed order, each
prefixed with its name and length so that two different column layouts can
never hash alike.  A truncated write, a flipped bit or a partially-applied
update is therefore detectable — see ``verify_chunk``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from bson.binary import Binary

SCHEMA_VERSION = 1
CODEC_ID = "cols-v1"

PRICE_SCALE = 10_000
ACTION_SCALE = 10_000

INT32_MISSING = np.iinfo(np.int32).min  # -2147483648, reserved as the NaN sentinel
INT64_MISSING = np.iinfo(np.int64).min
_PRICE_ABS_MAX = np.iinfo(np.int32).max - 1  # sentinel excluded from the value range

PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close")
ACTION_FIELDS: tuple[str, ...] = ("dividends", "splits")
DATA_FIELDS: tuple[str, ...] = PRICE_FIELDS + ("volume",) + ACTION_FIELDS
ALL_FIELDS: tuple[str, ...] = ("date",) + DATA_FIELDS

#: canonical hashing order — changing this changes every checksum, so it is
#: pinned to the schema version rather than derived from dict ordering.
_CHECKSUM_ORDER: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends.i",
    "dividends.v",
    "splits.i",
    "splits.v",
)

_EPOCH = np.datetime64("1970-01-01", "D")


class CodecError(ValueError):
    """Raised when data cannot be represented, or a stored chunk is inconsistent."""


# --------------------------------------------------------------------------- model


@dataclass(slots=True)
class BarSeries:
    """Columnar daily bars for one ticker, in memory.

    All arrays share length ``n`` and are aligned to ``dates``.  Prices and
    volume are float64 so that NaN can represent "Yahoo reported this bar but
    not this field"; the codec narrows them on the way to storage.
    """

    dates: np.ndarray  # datetime64[D]
    open: np.ndarray | None = None
    high: np.ndarray | None = None
    low: np.ndarray | None = None
    close: np.ndarray | None = None
    adj_close: np.ndarray | None = None
    volume: np.ndarray | None = None
    dividends: np.ndarray | None = None
    splits: np.ndarray | None = None
    ticker: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dates = np.asarray(self.dates, dtype="datetime64[D]")
        for name in DATA_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != self.dates.shape:
                raise CodecError(
                    f"column {name!r} has length {arr.size}, expected {self.dates.size}"
                )
            setattr(self, name, arr)

    def __len__(self) -> int:
        return int(self.dates.size)

    @property
    def present(self) -> tuple[str, ...]:
        return tuple(f for f in DATA_FIELDS if getattr(self, f) is not None)

    # -- construction -------------------------------------------------------

    @classmethod
    def empty(cls, ticker: str | None = None) -> "BarSeries":
        z = np.zeros(0, dtype=np.float64)
        return cls(
            dates=np.zeros(0, dtype="datetime64[D]"),
            open=z.copy(),
            high=z.copy(),
            low=z.copy(),
            close=z.copy(),
            adj_close=z.copy(),
            volume=z.copy(),
            dividends=z.copy(),
            splits=z.copy(),
            ticker=ticker,
        )

    @classmethod
    def from_frame(cls, frame: Any, ticker: str | None = None) -> "BarSeries":
        """Build from a yfinance/pandas frame indexed by date.

        Column names are matched case-insensitively and space-insensitively so
        that both ``Adj Close`` and ``adj_close`` work, and both ``Stock
        Splits`` and ``splits``.
        """
        if frame is None or len(frame) == 0:
            return cls.empty(ticker)

        lookup = {str(c).lower().replace(" ", "_"): c for c in frame.columns}
        alias = {
            "open": ("open",),
            "high": ("high",),
            "low": ("low",),
            "close": ("close",),
            "adj_close": ("adj_close", "adjclose"),
            "volume": ("volume",),
            "dividends": ("dividends",),
            "splits": ("stock_splits", "splits"),
        }

        index = frame.index
        # tz-aware DatetimeIndex -> naive calendar dates in exchange local terms
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        dates = np.asarray(index, dtype="datetime64[D]")

        columns: dict[str, np.ndarray] = {}
        for target, candidates in alias.items():
            for candidate in candidates:
                if candidate in lookup:
                    columns[target] = np.asarray(frame[lookup[candidate]], dtype=np.float64)
                    break
        # A split of "no split" is reported as 0.0 by yfinance; keep that convention.
        return cls(dates=dates, ticker=ticker, **columns)

    # -- operations ---------------------------------------------------------

    def slice_dates(self, start: dt.date | None, end: dt.date | None) -> "BarSeries":
        """Inclusive on both ends — the API contract, applied once, here."""
        mask = np.ones(len(self), dtype=bool)
        if start is not None:
            mask &= self.dates >= np.datetime64(start, "D")
        if end is not None:
            mask &= self.dates <= np.datetime64(end, "D")
        return self._take(mask)

    def _take(self, mask: np.ndarray) -> "BarSeries":
        kwargs = {f: getattr(self, f)[mask] for f in self.present}
        return BarSeries(dates=self.dates[mask], ticker=self.ticker, meta=dict(self.meta), **kwargs)

    def split_by_year(self) -> dict[int, "BarSeries"]:
        if len(self) == 0:
            return {}
        years = self.dates.astype("datetime64[Y]").astype(int) + 1970
        return {int(y): self._take(years == y) for y in np.unique(years)}

    def to_columns(self, fields: Sequence[str] | None = None) -> dict[str, list[Any]]:
        """JSON-friendly columnar dict; NaN becomes ``None``, volume becomes int."""
        wanted = tuple(fields) if fields else self.present
        out: dict[str, list[Any]] = {
            "date": [d.item().isoformat() for d in self.dates.astype("datetime64[D]")]
        }
        for name in wanted:
            arr = getattr(self, name)
            if arr is None:
                continue
            if name == "volume":
                out[name] = [None if np.isnan(v) else int(v) for v in arr]
            else:
                out[name] = [None if np.isnan(v) else float(v) for v in arr]
        return out


def concat(series: Iterable[BarSeries]) -> BarSeries:
    """Concatenate chunks in date order, keeping only columns present in all."""
    items = [s for s in series if len(s) > 0]
    if not items:
        return BarSeries.empty()
    common = set(items[0].present)
    for s in items[1:]:
        common &= set(s.present)
    ordered = sorted(items, key=lambda s: s.dates[0])
    kwargs = {f: np.concatenate([s.__getattribute__(f) for s in ordered]) for f in common}
    dates = np.concatenate([s.dates for s in ordered])
    order = np.argsort(dates, kind="stable")
    return BarSeries(
        dates=dates[order],
        ticker=ordered[0].ticker,
        **{k: v[order] for k, v in kwargs.items()},
    )


# --------------------------------------------------------------------------- pack


def _pack_scaled(values: np.ndarray, scale: int, name: str) -> bytes:
    finite = np.isfinite(values)
    scaled = np.full(values.shape, INT32_MISSING, dtype=np.int64)
    scaled[finite] = np.rint(values[finite] * scale).astype(np.int64)
    if finite.any():
        peak = int(np.max(np.abs(scaled[finite]))) if finite.any() else 0
        if peak > _PRICE_ABS_MAX:
            raise CodecError(
                f"column {name!r} value {peak / scale:.4f} overflows int32 fixed point "
                f"(max {_PRICE_ABS_MAX / scale:.4f})"
            )
    return scaled.astype("<i4").tobytes()


def _unpack_scaled(buf: bytes, scale: int) -> np.ndarray:
    raw = np.frombuffer(buf, dtype="<i4")
    out = raw.astype(np.float64) / scale
    out[raw == INT32_MISSING] = np.nan
    return out


def _pack_volume(values: np.ndarray) -> bytes:
    finite = np.isfinite(values)
    packed = np.full(values.shape, INT64_MISSING, dtype=np.int64)
    packed[finite] = np.rint(values[finite]).astype(np.int64)
    return packed.astype("<i8").tobytes()


def _unpack_volume(buf: bytes) -> np.ndarray:
    raw = np.frombuffer(buf, dtype="<i8")
    out = raw.astype(np.float64)
    out[raw == INT64_MISSING] = np.nan
    return out


def _pack_dates(dates: np.ndarray) -> bytes:
    days = (dates.astype("datetime64[D]") - _EPOCH).astype("int64")
    return days.astype("<i4").tobytes()


def _unpack_dates(buf: bytes) -> np.ndarray:
    return _EPOCH + np.frombuffer(buf, dtype="<i4").astype("timedelta64[D]")


def _pack_sparse(values: np.ndarray | None) -> dict[str, Binary] | None:
    """Dividends and splits are zero on >99% of sessions: store only the hits."""
    if values is None:
        return None
    nonzero = np.flatnonzero(np.nan_to_num(values, nan=0.0) != 0.0)
    if nonzero.size == 0:
        return None
    return {
        "i": Binary(nonzero.astype("<i4").tobytes()),
        "v": Binary(_pack_scaled(values[nonzero], ACTION_SCALE, "action")),
    }


def _unpack_sparse(doc: Mapping[str, Any] | None, n: int) -> np.ndarray:
    dense = np.zeros(n, dtype=np.float64)
    if not doc:
        return dense
    idx = np.frombuffer(bytes(doc["i"]), dtype="<i4")
    dense[idx] = _unpack_scaled(bytes(doc["v"]), ACTION_SCALE)
    return dense


def _checksum(buffers: Mapping[str, bytes]) -> str:
    """Length-prefixed, name-prefixed hash so column layouts cannot collide."""
    digest = hashlib.sha256()
    for name in _CHECKSUM_ORDER:
        buf = buffers.get(name)
        if buf is None:
            continue
        digest.update(name.encode("ascii"))
        digest.update(struct.pack("<I", len(buf)))
        digest.update(buf)
    return digest.hexdigest()


# --------------------------------------------------------------------------- api


def encode_chunk(ticker: str, year: int, bars: BarSeries) -> dict[str, Any]:
    """Encode one ticker-year into a BSON-ready document.

    ``ticker``, ``year``, ``first``, ``last``, ``n`` and ``sha256`` stay ordinary
    indexed BSON fields: the database must remain queryable, not become a bucket
    of opaque blobs.
    """
    if len(bars) == 0:
        raise CodecError(f"refusing to encode an empty chunk for {ticker} {year}")

    years = np.unique(bars.dates.astype("datetime64[Y]").astype(int) + 1970)
    if years.size != 1 or int(years[0]) != year:
        raise CodecError(f"chunk for {ticker} {year} contains dates from years {years.tolist()}")
    if not np.all(np.diff(bars.dates.astype("int64")) > 0):
        raise CodecError(f"chunk for {ticker} {year} has unsorted or duplicated dates")

    buffers: dict[str, bytes] = {"date": _pack_dates(bars.dates)}
    for name in PRICE_FIELDS:
        values = getattr(bars, name)
        if values is not None:
            buffers[name] = _pack_scaled(values, PRICE_SCALE, name)
    if bars.volume is not None:
        buffers["volume"] = _pack_volume(bars.volume)

    doc: dict[str, Any] = {
        "_id": chunk_id(ticker, year),
        "ticker": ticker,
        "year": int(year),
        "n": len(bars),
        "first": _to_datetime(bars.dates[0]),
        "last": _to_datetime(bars.dates[-1]),
        "schema_version": SCHEMA_VERSION,
        "codec": CODEC_ID,
        "price_scale": PRICE_SCALE,
        "columns": [k for k in buffers if k != "date"],
    }
    for name, buf in buffers.items():
        doc[name] = Binary(buf)

    for name in ACTION_FIELDS:
        sparse = _pack_sparse(getattr(bars, name))
        if sparse is not None:
            doc[name] = sparse
            buffers[f"{name}.i"] = bytes(sparse["i"])
            buffers[f"{name}.v"] = bytes(sparse["v"])

    doc["sha256"] = _checksum(buffers)
    return doc


def decode_chunk(
    doc: Mapping[str, Any],
    fields: Sequence[str] | None = None,
    *,
    verify: bool = False,
) -> BarSeries:
    """Decode a stored chunk, optionally only the requested columns.

    ``verify=True`` requires a complete document; a projected read cannot be
    checksummed because the hash covers every column.
    """
    codec = doc.get("codec", CODEC_ID)
    if codec != CODEC_ID:
        raise CodecError(f"unsupported codec {codec!r}; this build reads {CODEC_ID!r}")
    version = int(doc.get("schema_version", SCHEMA_VERSION))
    if version > SCHEMA_VERSION:
        raise CodecError(f"chunk schema_version {version} is newer than this build ({SCHEMA_VERSION})")

    if verify:
        verify_chunk(doc, raise_on_mismatch=True)

    dates = _unpack_dates(bytes(doc["date"]))
    n = dates.size
    if "n" in doc and int(doc["n"]) != n:
        raise CodecError(f"chunk {doc.get('_id')!r}: n={doc['n']} but date column holds {n}")

    wanted = tuple(fields) if fields else DATA_FIELDS
    scale = int(doc.get("price_scale", PRICE_SCALE))
    columns: dict[str, np.ndarray] = {}
    for name in wanted:
        if name in PRICE_FIELDS and name in doc:
            columns[name] = _unpack_scaled(bytes(doc[name]), scale)
        elif name == "volume" and "volume" in doc:
            columns[name] = _unpack_volume(bytes(doc["volume"]))
        elif name in ACTION_FIELDS:
            columns[name] = _unpack_sparse(doc.get(name), n)

    return BarSeries(
        dates=dates,
        ticker=doc.get("ticker"),
        meta={"year": doc.get("year"), "sha256": doc.get("sha256")},
        **columns,
    )


def verify_chunk(doc: Mapping[str, Any], *, raise_on_mismatch: bool = False) -> bool:
    """Recompute the checksum of a *complete* chunk document."""
    stored = doc.get("sha256")
    if stored is None:
        if raise_on_mismatch:
            raise CodecError(f"chunk {doc.get('_id')!r} carries no sha256")
        return False

    buffers: dict[str, bytes] = {}
    for name in ("date",) + PRICE_FIELDS + ("volume",):
        if name in doc:
            buffers[name] = bytes(doc[name])
    for name in ACTION_FIELDS:
        sparse = doc.get(name)
        if sparse:
            buffers[f"{name}.i"] = bytes(sparse["i"])
            buffers[f"{name}.v"] = bytes(sparse["v"])

    actual = _checksum(buffers)
    if actual != stored:
        if raise_on_mismatch:
            raise CodecError(
                f"checksum mismatch for chunk {doc.get('_id')!r}: "
                f"stored {stored[:12]}..., computed {actual[:12]}..."
            )
        return False
    return True


def chunk_id(ticker: str, year: int) -> str:
    """Deterministic ``_id`` — makes every write an idempotent upsert."""
    return f"{ticker}:{year}"


def bytes_per_bar(doc: Mapping[str, Any]) -> float:
    """Measured payload cost, used for the storage table in the deck."""
    n = int(doc["n"])
    payload = sum(
        len(bytes(doc[k]))
        for k in ("date",) + PRICE_FIELDS + ("volume",)
        if k in doc
    )
    for name in ACTION_FIELDS:
        sparse = doc.get(name)
        if sparse:
            payload += len(bytes(sparse["i"])) + len(bytes(sparse["v"]))
    return payload / n if n else 0.0


def _to_datetime(day: np.datetime64) -> dt.datetime:
    """BSON has no date type — store midnight UTC and never attach a timezone."""
    d = day.astype("datetime64[D]").item()
    return dt.datetime(d.year, d.month, d.day)
