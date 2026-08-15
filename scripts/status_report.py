#!/usr/bin/env python3
"""Publish the last update as a table in README.md.

    python scripts/status_report.py                       # rewrite the README block
    python scripts/status_report.py --print               # stdout only, touch nothing
    python scripts/status_report.py --release-notes o.md  # also write release notes

Renders what `update_daily.py` logs at the end of a run, plus the standing
database facts that `verify_integrity.py` reports, as one markdown block bounded
by HTML comments so surrounding prose is never touched.  If the README has no
block yet -- or no README exists at all -- one is created.

Two timestamps are published because they are different facts:

``last run``     when the updater last executed.  Says the automation is alive.
``data through`` the newest session actually stored.  Says the *data* is current.

A job that runs every night against a market that was shut for a week is green
on the first and eight days stale on the second, and only the second matters to
a reader.  The badge colour therefore follows the data, not the exit code.

Exit codes: 0 ok, 1 no report to summarise, 3 database unreachable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import _path  # noqa: F401

from sse.logging_setup import setup_logging  # isort: skip

logger = setup_logging("status")

BEGIN = "<!-- data-status:begin -->"
END = "<!-- data-status:end -->"
MIB = 1024 * 1024

COLOURS = {"ok": "brightgreen", "failed": "red", "halted_storage": "orange",
           "stale": "yellow", "unknown": "lightgrey"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=Path("reports/update_last.json"),
                        help="run report written by update_daily.py")
    parser.add_argument("--integrity", type=Path, default=Path("reports/integrity.json"),
                        help="optional report from verify_integrity.py")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--print", dest="print_only", action="store_true")
    parser.add_argument("--release-notes", type=Path)
    parser.add_argument("--stale-after-days", type=int, default=5,
                        help="days of silence before the badge stops saying ok")
    parser.add_argument("--no-db", action="store_true",
                        help="skip the live queries (offline rendering)")
    parser.add_argument("--timezone-offset", type=int, default=8,
                        help="hours east of UTC for the local-time column (default: CST)")
    return parser.parse_args(argv)


def live_facts() -> dict[str, Any]:
    """Standing database facts -- the subset of verify_integrity worth showing daily."""
    from sse.db import client as db_client
    from sse.db.repository import MongoPriceRepository, SecurityRepository

    db = db_client.get_db()
    prices = MongoPriceRepository(db)
    securities = SecurityRepository(db)
    coverage = prices.coverage_many()
    firsts = [c.first for c in coverage.values() if c.first]
    lasts = [c.last for c in coverage.values() if c.last]
    return {
        "watermark": max(lasts) if lasts else None,
        "earliest": min(firsts) if firsts else None,
        "tickers_stored": len(coverage),
        "bars_stored": prices.stored_bar_count(),
        "securities": securities.count_by_status(),
    }


def _run_time(report: dict[str, Any]) -> dt.datetime | None:
    """`update_daily.py` stamps the UTC time into run_id, so it always survives."""
    for key in ("finished_at", "generated_at", "checked_at"):
        if report.get(key):
            try:
                return dt.datetime.fromisoformat(str(report[key]).replace("Z", "+00:00"))
            except ValueError:
                pass
    stamp = str(report.get("run_id", "")).rsplit("-", 1)[-1]
    try:
        return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def _ago(delta: dt.timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86_400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86_400} d ago"


def _shield(label: str, message: str, colour: str) -> str:
    def esc(text: str) -> str:
        return str(text).replace("-", "--").replace("_", "__").replace(" ", "_")

    return f"https://img.shields.io/badge/{esc(label)}-{esc(message)}-{colour}"


def render(report: dict[str, Any], facts: dict[str, Any],
           integrity: dict[str, Any] | None, *, stale_after_days: int,
           now: dt.datetime, tz_offset: int) -> str:
    counters = report.get("counters", {}) or {}
    storage = report.get("storage", {}) or {}
    census = report.get("census", {}) or {}
    status = report.get("status", "unknown")
    watermark = facts.get("watermark")

    # The badge follows the data, not the exit code: a successful run against a
    # market that has not traded for a week is not "ok" to a reader.
    badge = status
    if watermark is not None and (now.date() - watermark).days > stale_after_days:
        badge = "stale"
    colour = COLOURS.get(badge, COLOURS["unknown"])

    when = _run_time(report)
    ran_utc = when.strftime("%Y-%m-%d %H:%M UTC") if when else "unknown"
    ran_local = ((when + dt.timedelta(hours=tz_offset)).strftime("%Y-%m-%d %H:%M")
                 + f" UTC+{tz_offset}") if when else "unknown"
    through = watermark.isoformat() if watermark else "unknown"

    ok = counters.get("tickers_done", 0)
    failed = counters.get("tickers_failed", 0)
    total = counters.get("tickers_total", 0)
    ins = counters.get("chunks_inserted", 0)
    upd = counters.get("chunks_updated", 0)
    unch = counters.get("chunks_unchanged", 0)

    rows: list[tuple[str, str]] = [
        ("Data current through", f"`{through}`"),
        ("Last update run", f"{ran_local} &nbsp;/&nbsp; {ran_utc}"
                            + (f" ({_ago(now - when)})" if when else "")),
        ("Run kind / status", f"`{report.get('kind', '?')}` / **{status}**"),
        ("Tickers ok / failed", f"{ok:,} / {failed:,}" + (f" of {total:,}" if total else "")),
        ("Chunks written", f"{ins:,} new, {upd:,} rewritten, {unch:,} unchanged"),
        ("Bars written this run", f"{counters.get('bars_written', 0):,}"),
    ]
    if upd and report.get("kind") != "full_refresh":
        rows.append(("", "<sub>rewrites are Yahoo corrections inside the overlap "
                         "window, not new sessions</sub>"))

    if facts.get("tickers_stored"):
        rows.append(("Stored", f"{facts['tickers_stored']:,} tickers, "
                               f"{facts.get('bars_stored', 0):,} bars, "
                               f"{facts.get('earliest') or '?'} to {through}"))
    if facts.get("securities"):
        rows.append(("Securities master",
                     ", ".join(f"{v:,} {k}" for k, v in sorted(facts["securities"].items()))))

    if storage.get("cap_bytes"):
        detail = (f"data {storage.get('data_size', 0) / MIB:.1f} MiB, "
                  f"index {storage.get('index_size', 0) / MIB:.1f} MiB, "
                  f"compression {storage.get('compression_ratio', 0):.2f}x")
        rows.append(("Atlas M0 storage",
                     f"{storage['used_bytes'] / MIB:.1f} MiB of "
                     f"{storage['cap_bytes'] / MIB:.0f} MiB "
                     f"({storage['used_fraction']:.1%}; {detail})"))
    if census.get("tickers"):
        rows.append(("Quality census",
                     f"{census.get('bars_out', 0):,} bars kept, "
                     f"{census.get('bars_dropped', 0):,} dropped, "
                     f"{census.get('tickers_with_errors', 0):,} tickers with errors"))
    for failure in (report.get("validation_failures") or [])[:3]:
        rows.append(("Validation failure", f"`{failure}`"))

    # Integrity is a weekly fact, not a nightly one, so it carries its own
    # timestamp rather than being passed off as part of tonight's run.
    if integrity:
        checked = str(integrity.get("checked_at", ""))[:16].replace("T", " ")
        rows += [
            (f"**Integrity** <sub>{checked} UTC</sub>",
             f"{integrity.get('chunks', 0):,} chunks, "
             f"{integrity.get('bars_decoded', 0):,} bars decoded, "
             f"reconciled: **{integrity.get('reconciled')}**"),
            ("Checksum failures", f"**{integrity.get('checksum_failure_count', 0):,}**"),
            ("Trading calendar", f"{integrity.get('calendar_sessions', 0):,} sessions "
                                 f"({integrity.get('calendar_first')} to "
                                 f"{integrity.get('calendar_last')})"),
            ("Suspension sessions", f"{integrity.get('suspension_sessions_total', 0):,}"),
        ]

    lines = [
        BEGIN,
        "## Data status",
        "",
        f"![data through]({_shield('data through', through, colour)}) "
        f"![last run]({_shield('last run', ran_utc, colour)}) "
        f"![status]({_shield('status', badge, colour)})",
        "",
        "| | |",
        "|---|---|",
    ]
    lines += [f"| {label} | {value} |" for label, value in rows]
    lines += [
        "",
        "<sub>Generated by `scripts/status_report.py` after each scheduled run — "
        "do not edit by hand; anything between the markers is overwritten. "
        "*Data current through* is the newest stored session and is the number that "
        "matters; *last update run* only says the automation fired.</sub>",
        END,
    ]
    return "\n".join(lines)


def splice(readme: str, block: str) -> str:
    """Replace the delimited block, appending one if the README has none."""
    start, end = readme.find(BEGIN), readme.find(END)
    if start == -1 or end == -1 or end < start:
        if not readme.strip():
            return block + "\n"
        sep = "" if readme.endswith("\n\n") else ("\n" if readme.endswith("\n") else "\n\n")
        return readme + sep + block + "\n"
    return readme[:start] + block + readme[end + len(END):]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.report.exists():
        logger.error("no run report at %s — has update_daily.py run yet?", args.report)
        return 1
    report = json.loads(args.report.read_text(encoding="utf-8"))

    integrity = None
    if args.integrity and args.integrity.exists():
        try:
            integrity = json.loads(args.integrity.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("ignoring unreadable %s", args.integrity)

    facts: dict[str, Any] = {}
    if not args.no_db:
        try:
            facts = live_facts()
        except Exception as exc:  # noqa: BLE001
            from sse.db import client as db_client

            logger.error("could not read the database: %s", db_client.redact(str(exc)))
            return 3

    block = render(report, facts, integrity,
                   stale_after_days=args.stale_after_days,
                   now=dt.datetime.now(dt.UTC), tz_offset=args.timezone_offset)

    if args.release_notes:
        args.release_notes.parent.mkdir(parents=True, exist_ok=True)
        args.release_notes.write_text(
            block.replace(BEGIN, "").replace(END, "").strip() + "\n", encoding="utf-8")
        logger.info("wrote release notes %s", args.release_notes)

    if args.print_only:
        print(block)
        return 0

    original = args.readme.read_text(encoding="utf-8") if args.readme.exists() else ""
    updated = splice(original, block)
    if updated == original:
        logger.info("status block unchanged — nothing to commit")
        return 0
    args.readme.parent.mkdir(parents=True, exist_ok=True)
    args.readme.write_text(updated, encoding="utf-8")
    logger.info("updated the status block in %s (data through %s)",
                args.readme, facts.get("watermark") or "unknown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
