"""CLI: report ``skew_25d`` movement around a scheduled event, from the
history job's own committed rows (Day 7).

    python -m volsurface.event_skew --underlying RELIANCE --event-date 2026-10-15

Splits history rows by ``data_date`` - the date the numbers actually
describe, never ``run_date`` (see ``volsurface/history.py``'s own module
docstring for why the two are kept separate) - into a pre-event and
post-event window around ``--event-date``, and reports the mean
``skew_25d`` on each side plus the before -> after change.

This is a query over whatever history already exists; it never runs the
pipeline itself. Run ``volsurface.history`` first to build up rows.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean

from volsurface.history import HISTORY_DIR, HistoryError, load_history

DEFAULT_WINDOW_DAYS = 5


@dataclass
class EventSkewReport:
    event_date: str
    window_days: int
    pre_rows: int
    post_rows: int
    pre_mean_skew: float | None
    post_mean_skew: float | None
    change: float | None  # post_mean_skew - pre_mean_skew
    note: str  # always set: explains the number, or explains why there isn't one


def _rows_with_skew(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("skew_25d") is not None]


def event_skew_report(
    rows: list[dict], event_date: str | date, window_days: int = DEFAULT_WINDOW_DAYS
) -> EventSkewReport:
    """Compare mean ``skew_25d`` in the ``window_days`` before ``event_date``
    against the ``window_days`` on or after it, using ``data_date`` (not
    ``run_date``) to place each row in time.

    A row with ``skew_25d`` missing (the value :func:`volsurface.history.compute_daily_summary`
    leaves ``None`` when the 25-delta strike wasn't found in the fitted
    domain that day) is excluded from both windows rather than treated as
    zero movement.
    """
    event = date.fromisoformat(event_date) if isinstance(event_date, str) else event_date
    scored = _rows_with_skew(rows)

    pre = [
        r for r in scored
        if 0 < (event - date.fromisoformat(r["data_date"])).days <= window_days
    ]
    post = [
        r for r in scored
        if 0 <= (date.fromisoformat(r["data_date"]) - event).days <= window_days
    ]

    pre_mean = mean(r["skew_25d"] for r in pre) if pre else None
    post_mean = mean(r["skew_25d"] for r in post) if post else None
    change = post_mean - pre_mean if pre_mean is not None and post_mean is not None else None

    distinct_dates = {r["data_date"] for r in scored}
    if change is not None:
        note = f"{len(pre)} pre-event row(s), {len(post)} post-event row(s) with a usable skew_25d."
    elif len(distinct_dates) <= 1:
        only = next(iter(distinct_dates)) if distinct_dates else "none"
        note = (
            f"every history row with a skew_25d shares one data_date ({only}) - this sandbox's "
            "only working provider is the fixture, which never advances data_date (README Day 1/6 "
            "Findings), so there is no real before/after to compare yet. This command's logic is "
            "exercised by tests/test_event_skew.py against synthetic multi-day history instead."
        )
    else:
        note = (
            f"only {len(pre)} pre-event and {len(post)} post-event row(s) with a usable skew_25d "
            f"within {window_days} day(s) of {event.isoformat()} - not enough on at least one side."
        )

    return EventSkewReport(
        event_date=event.isoformat(),
        window_days=window_days,
        pre_rows=len(pre),
        post_rows=len(post),
        pre_mean_skew=pre_mean,
        post_mean_skew=post_mean,
        change=change,
        note=note,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", required=True, help="e.g. RELIANCE")
    parser.add_argument("--event-date", required=True, dest="event_date", help="YYYY-MM-DD")
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS, dest="window_days",
        help="days either side of --event-date to average over",
    )
    parser.add_argument(
        "--history-file", type=Path, default=None, dest="history_file",
        help="defaults to fixtures/history/<UNDERLYING>.json",
    )
    args = parser.parse_args(argv)

    path = args.history_file or (HISTORY_DIR / f"{args.underlying.upper()}.json")
    try:
        rows = load_history(path)
    except HistoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print(f"error: no history rows at {path} - run volsurface.history first", file=sys.stderr)
        return 1

    report = event_skew_report(rows, args.event_date, args.window_days)
    print(
        f"{args.underlying.upper()} event={report.event_date} window=+/-{report.window_days}d: "
        f"pre skew_25d={report.pre_mean_skew} ({report.pre_rows} row(s)), "
        f"post skew_25d={report.post_mean_skew} ({report.post_rows} row(s)), "
        f"change={report.change}"
    )
    print(report.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
