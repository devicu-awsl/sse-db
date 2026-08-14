#!/usr/bin/env python3
"""Publish "when was the data last updated" where a reader will actually see it.

    python scripts/status_report.py                      # rewrite the README block
    python scripts/status_report.py --print              # stdout only, touch nothing
    python scripts/status_report.py --release-notes out.md

Two different timestamps get confused constantly, so both are reported:

``last_run``    when the updater last executed.  Says the automation is alive.
``data_through``the newest session actually stored, taken from the chunk
                watermarks.  Says the *data* is current.

They are not the same fact.  A job that runs every night against a market that
was closed for a week is green on the first and eight days stale on the second,
and only the second one matters to anyone reading the README.

The README block is delimited by HTML comments so the surrounding prose is never
touched.  Exit codes: 0 ok, 1 no report to summarise, 3 database unreachable.
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

#: Shields.io renders these; the colour is the honest signal, so a stale or
#: failed run must not be able to show green.
COLOURS = {"ok": "brightgreen", "failed": "red", "halted_storage": "orange",
           "stale": "yellow", "unknown": "lightgrey"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=Path("reports/update_last.json"),
                        help="run report written by update_daily.py")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="write nothing; print the block to stdout")
    parser.add_argument("--release-notes", type=Path,
                        help="also write the same summary here, for `gh release`")
    parser.add_argument("--stale-after-days", type=int, default=5,
                        help="sessions of silence before the badge stops saying ok")
    parser.add_argument("--no-db", action="store_true",
                        help="skip the watermark query (offline rendering)")
    return parser.parse_args(argv)


def data_watermark() -> tuple[dt.date | None, int, int]:
    """Newest stored session, ticker count and bar count, straight from the chunks."""
    from sse.db import client as db_client
    from sse.db.repository import MongoPriceRepository

    prices = MongoPriceRepository(db_client.get_db())
    coverage = prices.coverage_many()
    latest = max((c.last for c in coverage.values() if c.last), default=None)
    return latest, len(coverage), prices.stored_bar_count()


def _humanise(delta: dt.timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86_400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86_400} d ago"


def render(report: dict[str, Any], watermark: dt.date | None, n_tickers: int,
           n_bars: int, *, stale_after_days: int, now: dt.datetime) -> str:
    counters = report.get("counters", {})
    storage = report.get("storage", {})
    status = report.get("status", "unknown")

    ran_at = report.get("finished_at") or report.get("generated_at")
    when = None
    if ran_at:
        try:
            when = dt.datetime.fromisoformat(str(ran_at).replace("Z", "+00:00"))
        except ValueError:
            when = None
    if when is None:
        # `update_daily.py` stamps the run id, so the time survives even when the
        # report carries no explicit timestamp.
        run_id = str(report.get("run_id", ""))
        stamp = run_id.rsplit("-", 1)[-1]
        try:
            when = dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC)
        except ValueError:
            when = None

    # The badge reflects the *data*, not merely the exit code: a run that
    # succeeds while the market data is a week old is not "ok".
    badge_status = status
    if watermark is not None and (now.date() - watermark).days > stale_after_days:
        badge_status = "stale"
    colour = COLOURS.get(badge_status, COLOURS["unknown"])

    through = watermark.isoformat() if watermark else "unknown"
    ran = when.strftime("%Y-%m-%d %H:%M UTC") if when else "unknown"
    ago = f" ({_humanise(now - when)})" if when else ""

    used = ""
    if storage.get("cap_bytes"):
        mib = 1024 * 1024
        used = (f"{storage['used_bytes'] / mib:.1f} / "
                f"{storage['cap_bytes'] / mib:.0f} MiB "
                f"({storage['used_fraction']:.1%})")

    lines = [
        BEGIN,
        f"![data]({_shield('data through', through, colour)}) "
        f"![updated]({_shield('last run', ran, colour)})",
        "",
        "| | |",
        "|---|---|",
        f"| **Data current through** | `{through}` |",
        f"| **Last update run** | {ran}{ago} |",
        f"| **Run kind / status** | `{report.get('kind', '?')}` / `{status}` |",
        f"| **Tickers ok / failed** | {counters.get('tickers_done', 0)} / "
        f"{counters.get('tickers_failed', 0)} of {counters.get('tickers_total', 0)} |",
        f"| **Chunks new / rewritten / unchanged** | {counters.get('chunks_inserted', 0)} / "
        f"{counters.get('chunks_updated', 0)} / {counters.get('chunks_unchanged', 0)} |",
    ]
    if n_tickers:
        lines.append(f"| **Stored** | {n_tickers:,} tickers, {n_bars:,} bars |")
    if used:
        lines.append(f"| **Atlas M0 storage** | {used} |")
    lines += [
        "",
        "<sub>Written by `scripts/status_report.py` after each scheduled run. "
        "*Data current through* is the newest stored session, which is the number "
        "that matters; *last update run* only says the automation fired.</sub>",
        END,
    ]
    return "\n".join(lines)


def _shield(label: str, message: str, colour: str) -> str:
    def esc(text: str) -> str:
        return str(text).replace("-", "--").replace("_", "__").replace(" ", "_")

    return f"https://img.shields.io/badge/{esc(label)}-{esc(message)}-{colour}"


def splice(readme: str, block: str) -> str:
    """Replace the delimited block, or append one if the README has none."""
    start, end = readme.find(BEGIN), readme.find(END)
    if start == -1 or end == -1 or end < start:
        separator = "" if readme.endswith("\n\n") else "\n\n"
        return readme + separator + block + "\n"
    return readme[:start] + block + readme[end + len(END):]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.report.exists():
        logger.error("no run report at %s — has update_daily.py run yet?", args.report)
        return 1
    report = json.loads(args.report.read_text(encoding="utf-8"))

    watermark: dt.date | None = None
    n_tickers = n_bars = 0
    if not args.no_db:
        try:
            watermark, n_tickers, n_bars = data_watermark()
        except Exception as exc:  # noqa: BLE001
            from sse.db import client as db_client

            logger.error("could not read the watermark: %s", db_client.redact(str(exc)))
            return 3

    block = render(report, watermark, n_tickers, n_bars,
                   stale_after_days=args.stale_after_days,
                   now=dt.datetime.now(dt.UTC))

    if args.release_notes:
        args.release_notes.parent.mkdir(parents=True, exist_ok=True)
        args.release_notes.write_text(
            block.replace(BEGIN, "").replace(END, "").strip() + "\n", encoding="utf-8")
        logger.info("wrote release notes %s", args.release_notes)

    if args.print_only:
        print(block)
        return 0

    if not args.readme.exists():
        logger.error("no README at %s", args.readme)
        return 1
    original = args.readme.read_text(encoding="utf-8")
    updated = splice(original, block)
    if updated == original:
        logger.info("status block unchanged — nothing to commit")
        return 0
    args.readme.write_text(updated, encoding="utf-8")
    logger.info("updated the status block in %s (data through %s)",
                args.readme, watermark or "unknown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
