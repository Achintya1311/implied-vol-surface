"""CLI: solve implied volatility for every contract in a chain snapshot.

    python -m volsurface.solve --underlying RELIANCE --provider fixture

Reads a chain via any registered provider (``fixture`` by default, so this
runs with no network), solves each row's bid/ask mid for implied vol, prints
a summary, and optionally writes the full per-contract table as CSV.

``--greeks`` (Day 3) attaches analytic delta/gamma/vega/theta to every solved
row and prints the largest analytic-vs-finite-difference gap seen across the
chain - the same cross-check ``tests/test_greeks.py`` runs per-parameter, but
against this chain's actual solved IVs rather than a hand-picked grid.

``--surface`` (Day 4) filters illiquid strikes, fits an OTM smile per
expiry, and checks the fitted surface for butterfly and calendar arbitrage.

``--plot`` (Day 5) builds the surface (implying ``--surface``) and writes a
per-expiry smile PNG, an ATM term-structure PNG, a static 3D surface PNG,
and a standalone interactive 3D surface HTML to ``outputs/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from volsurface.chain import PROVIDERS, ProviderError, get_provider
from volsurface.greeks import GREEK_COLUMNS, compute_chain_greeks
from volsurface.iv import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, _valuation_date, solve_chain_ivs
from volsurface.plot import PlotError, render_all
from volsurface.surface import (
    DEFAULT_MAX_RELATIVE_SPREAD,
    DEFAULT_MIN_OPEN_INTEREST,
    build_surface,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", required=True, help="e.g. RELIANCE")
    parser.add_argument("--provider", default="fixture", choices=sorted(PROVIDERS))
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="annualised risk-free rate")
    parser.add_argument(
        "--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD, dest="dividend_yield"
    )
    parser.add_argument("--out", type=Path, default=None, help="write the full table as CSV here")
    parser.add_argument(
        "--greeks", action="store_true", help="attach analytic Greeks plus a finite-difference cross-check"
    )
    parser.add_argument(
        "--surface", action="store_true",
        help="filter illiquid strikes, fit an OTM smile per expiry, and check for butterfly/calendar arbitrage",
    )
    parser.add_argument(
        "--min-open-interest", type=int, default=DEFAULT_MIN_OPEN_INTEREST, dest="min_open_interest",
    )
    parser.add_argument(
        "--max-relative-spread", type=float, default=DEFAULT_MAX_RELATIVE_SPREAD, dest="max_relative_spread",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="build the surface and write smile/term-structure/3D-surface charts to outputs/ "
        "(implies --surface)",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=None, dest="plot_dir",
        help="directory for --plot output (default: <repo root>/outputs)",
    )
    args = parser.parse_args(argv)
    if args.plot:
        args.surface = True

    try:
        snapshot = get_provider(args.provider).fetch(args.underlying)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    table = solve_chain_ivs(
        snapshot.quotes, snapshot.spot, snapshot.timestamp, r=args.rate, q=args.dividend_yield
    )

    solved = table["error"].isna()
    n_newton = int((table["method"] == "newton").sum())
    n_brent = int((table["method"] == "brent").sum())
    print(
        f"{snapshot.underlying} @ spot {snapshot.spot} ({snapshot.timestamp}), "
        f"r={args.rate:.4f} q={args.dividend_yield:.4f}"
    )
    print(f"{solved.sum()}/{len(table)} contracts solved ({n_newton} newton, {n_brent} brent fallback)")
    if (~solved).any():
        print(f"{(~solved).sum()} contract(s) had no solution:")
        for _, row in table[~solved].iterrows():
            print(f"  {row['expiry']} {row['strike']} {row['option_type']}: {row['error']}")

    if args.greeks:
        table = compute_chain_greeks(table, snapshot.spot, r=args.rate, q=args.dividend_yield)
        greek_rows = table.loc[solved, GREEK_COLUMNS].dropna()
        if not greek_rows.empty:
            worst = greek_rows["fd_max_abs_diff"].max()
            print(
                f"greeks: {len(greek_rows)}/{solved.sum()} solved contracts got analytic Greeks "
                f"(worst analytic-vs-finite-difference gap: {worst:.2e})"
            )

    if args.surface:
        result = build_surface(
            table,
            snapshot.quotes,
            snapshot.spot,
            _valuation_date(snapshot.timestamp),
            args.rate,
            args.dividend_yield,
            min_open_interest=args.min_open_interest,
            max_relative_spread=args.max_relative_spread,
        )
        print(
            f"surface: liquidity filter (open interest >= {args.min_open_interest}, "
            f"relative spread <= {args.max_relative_spread:.2f})"
        )
        for expiry, counts in sorted(result.dropped_counts.items()):
            print(f"  {expiry}: {counts['kept']}/{counts['total']} strikes kept")
        if not result.expiries:
            print("  no expiry had enough surviving strikes to fit a smile")
        elif result.violations:
            print(f"{len(result.violations)} arbitrage violation(s) found:")
            for v in result.violations:
                print(f"  [{v.kind}] {v.expiry} K={v.strike:.2f}: {v.detail}")
        else:
            print("no butterfly or calendar arbitrage violations found in the fitted surface")

        if args.plot:
            if not result.expiries:
                print("plot: skipped - no expiry had enough surviving strikes to fit a smile")
            else:
                try:
                    paths = render_all(result, snapshot.spot, args.underlying, args.plot_dir)
                except PlotError as exc:
                    print(f"plot: error: {exc}", file=sys.stderr)
                    return 1
                print("plot: wrote")
                for label, path in paths.items():
                    print(f"  {label}: {path}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
