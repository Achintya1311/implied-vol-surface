"""CLI: run the full pipeline once and append a small daily summary row to
a per-underlying history file (Day 6).

    python -m volsurface.history --underlying RELIANCE

Every later day that needs a trailing window (``iv_rank_1y`` in the v0.6
contract most directly) needs a history to look back over first - this is
the job that builds it, one row per run.

Two dates are kept per row, deliberately not collapsed into one:

- ``run_date``: when this job executed (defaults to today, UTC).
- ``data_date``: the valuation date of the chain snapshot the row was
  computed from (``ChainSnapshot.timestamp``'s date) - the date the numbers
  actually describe.

A row is keyed on ``(run_date, underlying)`` for idempotency: running the
job twice on the same day updates that day's row in place rather than
appending a duplicate. It is *not* keyed on ``data_date``, because the only
provider that currently works from this sandbox (``fixture``) always
carries the same fixed ``data_date`` - keying on it would collapse every
day's row into one, defeating the point of a daily job. See the README's
Findings for what that means for the numbers themselves.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from volsurface.chain import PROVIDERS, ProviderError, get_provider
from volsurface.iv import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, _valuation_date, solve_chain_ivs
from volsurface.plot import term_structure_points
from volsurface.surface import (
    DEFAULT_MAX_RELATIVE_SPREAD,
    DEFAULT_MIN_OPEN_INTEREST,
    SurfaceResult,
    build_surface,
)

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "fixtures" / "history"

# Nearest fitted expiry to this tenor stands in for "atm_iv_30d". Named for
# the v0.6 contract field it will eventually feed, not a guarantee that any
# given day's nearest expiry is actually close to 30 days out.
TARGET_TENOR_DAYS = 30.0
TARGET_TENOR_YEARS = TARGET_TENOR_DAYS / 365.0

HISTORY_ROW_KEYS = [
    "run_date",
    "data_date",
    "underlying",
    "source",
    "spot",
    "n_expiries",
    "atm_iv_30d",
    "atm_iv_near",
    "term_slope",
    "n_violations",
]


class HistoryError(ValueError):
    """A history file on disk is malformed, or a summary can't be built."""


@dataclass
class DailySummary:
    run_date: str  # YYYY-MM-DD, when this job ran
    data_date: str  # YYYY-MM-DD, the chain snapshot's own valuation date
    underlying: str
    source: str  # ChainSnapshot.source - "fixture", "nse", "yfinance", ...
    spot: float
    n_expiries: int  # expiries with a fitted smile, regardless of ATM coverage
    atm_iv_30d: float | None  # nearest-to-30d fitted expiry, IV at spot
    atm_iv_near: float | None  # nearest fitted expiry, IV at spot
    term_slope: float | None  # farthest atm_iv - nearest atm_iv; None if <2 usable expiries
    n_violations: int  # butterfly + calendar violations found on the fitted surface

    def to_dict(self) -> dict:
        return asdict(self)


def compute_daily_summary(
    result: SurfaceResult,
    spot: float,
    underlying: str,
    source: str,
    data_date: date,
    run_date: date,
) -> DailySummary:
    """Reduce a :class:`SurfaceResult` to one small, JSON-able row.

    Reuses :func:`volsurface.plot.term_structure_points` rather than
    re-deriving ATM IV, so this row and the Day 5 term-structure chart can
    never silently disagree about what "ATM" means for a given expiry.
    """
    points = term_structure_points(result.expiries, spot)  # sorted by tenor, ATM-in-domain only

    atm_iv_near: float | None = None
    atm_iv_30d: float | None = None
    term_slope: float | None = None

    if points:
        atm_iv_near = points[0][2]
        atm_iv_30d = min(points, key=lambda p: abs(p[1] - TARGET_TENOR_YEARS))[2]
        if len(points) >= 2:
            term_slope = points[-1][2] - points[0][2]

    return DailySummary(
        run_date=run_date.isoformat(),
        data_date=data_date.isoformat(),
        underlying=underlying.upper(),
        source=source,
        spot=spot,
        n_expiries=len(result.expiries),
        atm_iv_30d=atm_iv_30d,
        atm_iv_near=atm_iv_near,
        term_slope=term_slope,
        n_violations=len(result.violations),
    )


def _validate_row(row: dict, path: Path) -> None:
    missing = [k for k in ("run_date", "underlying") if k not in row]
    if missing:
        raise HistoryError(f"{path}: row is missing keys {missing}: {row!r}")


def load_history(path: Path) -> list[dict]:
    """Read a history file, or ``[]`` if it doesn't exist yet.

    Raises :class:`HistoryError` on anything that parses but isn't the
    expected shape (a JSON array of objects, each carrying at least
    ``run_date`` and ``underlying``) - a silently-ignored malformed file
    would quietly reset the history instead of failing loudly.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HistoryError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, list):
        raise HistoryError(f"{path}: expected a JSON array of rows, got {type(data).__name__}")
    for row in data:
        if not isinstance(row, dict):
            raise HistoryError(f"{path}: expected each row to be an object, got {row!r}")
        _validate_row(row, path)
    return data


def append_history(summary: DailySummary, path: Path) -> list[dict]:
    """Load, upsert ``summary`` keyed on ``(run_date, underlying)``, write
    back sorted by ``run_date``, and return the resulting rows.

    Upsert rather than append: re-running the job twice in one day (a retry
    after a transient provider failure, or a manual re-run) replaces that
    day's row instead of leaving two rows for the same day.
    """
    rows = load_history(path)
    new_row = summary.to_dict()
    rows = [
        r
        for r in rows
        if not (r["run_date"] == new_row["run_date"] and r["underlying"] == new_row["underlying"])
    ]
    rows.append(new_row)
    rows.sort(key=lambda r: (r["run_date"], r["underlying"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", required=True, help="e.g. RELIANCE")
    parser.add_argument("--provider", default="fixture", choices=sorted(PROVIDERS))
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="annualised risk-free rate")
    parser.add_argument(
        "--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD, dest="dividend_yield"
    )
    parser.add_argument(
        "--min-open-interest", type=int, default=DEFAULT_MIN_OPEN_INTEREST, dest="min_open_interest"
    )
    parser.add_argument(
        "--max-relative-spread",
        type=float,
        default=DEFAULT_MAX_RELATIVE_SPREAD,
        dest="max_relative_spread",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        dest="as_of",
        help="override run_date (YYYY-MM-DD); defaults to today (UTC). For tests and backfills.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="defaults to fixtures/history/<UNDERLYING>.json"
    )
    args = parser.parse_args(argv)

    run_date = datetime.now(timezone.utc).date() if args.as_of is None else date.fromisoformat(args.as_of)
    out_path = args.out or (HISTORY_DIR / f"{args.underlying.upper()}.json")

    try:
        snapshot = get_provider(args.provider).fetch(args.underlying)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    table = solve_chain_ivs(
        snapshot.quotes, snapshot.spot, snapshot.timestamp, r=args.rate, q=args.dividend_yield
    )
    data_date = _valuation_date(snapshot.timestamp)
    result = build_surface(
        table,
        snapshot.quotes,
        snapshot.spot,
        data_date,
        args.rate,
        args.dividend_yield,
        min_open_interest=args.min_open_interest,
        max_relative_spread=args.max_relative_spread,
    )

    summary = compute_daily_summary(
        result, snapshot.spot, args.underlying, snapshot.source, data_date, run_date
    )

    try:
        history = append_history(summary, out_path)
    except HistoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{summary.underlying} run_date={summary.run_date} data_date={summary.data_date} "
        f"source={summary.source}: atm_iv_30d={summary.atm_iv_30d}, atm_iv_near={summary.atm_iv_near}, "
        f"term_slope={summary.term_slope}, {summary.n_violations} violation(s) "
        f"-> {out_path} ({len(history)} row(s) total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
