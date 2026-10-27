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

``--forward parity`` (R25) replaces the single r/q forward with a per-expiry
forward from put-call parity on the chain's own two-sided quotes (see
``volsurface.forward``), so every count can be compared under both.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from volsurface import DISCLAIMER
from volsurface.chain import PROVIDERS, ProviderError, get_provider
from volsurface.forward import implied_dividend_yields, parity_forwards
from volsurface.greeks import GREEK_COLUMNS, compute_chain_greeks
from volsurface.iv import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_RATE,
    AT_OR_PAST_EXPIRY,
    NO_TWO_SIDED_QUOTE,
    _valuation_date,
    solve_chain_ivs,
    time_to_expiry,
)
from volsurface.plot import PlotError, render_all
from volsurface.surface import (
    BUTTERFLY_GRID_POINTS,
    DEFAULT_MAX_RELATIVE_SPREAD,
    DEFAULT_MIN_OPEN_INTEREST,
    build_surface,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", required=True, help="e.g. RELIANCE")
    parser.add_argument("--provider", default="fixture", choices=sorted(PROVIDERS))
    parser.add_argument(
        "--capture",
        metavar="STAMP",
        default=None,
        help="with --provider nse_raw, read this one raw capture batch (its <stamp>, e.g. "
        "2026-10-16T2319) instead of the newest; for reproducing a dated finding (P73)",
    )
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="annualised risk-free rate")
    parser.add_argument(
        "--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD, dest="dividend_yield"
    )
    parser.add_argument(
        "--forward", choices=["model", "parity"], default="model",
        help="model: one forward from --rate/--dividend-yield (default); parity: a per-expiry "
        "forward from put-call parity on two-sided quotes near spot (R25)",
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
    print(DISCLAIMER)
    if args.plot:
        args.surface = True

    try:
        snapshot = get_provider(args.provider, capture=args.capture).fetch(args.underlying)
    except (ProviderError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.forward == "parity" and args.greeks:
        print("error: --greeks takes one dividend yield; it is not combined with --forward parity",
              file=sys.stderr)
        return 1

    forwards = q_by_expiry = None
    if args.forward == "parity":
        valuation = _valuation_date(snapshot.timestamp)
        parity = parity_forwards(snapshot.quotes, snapshot.spot, valuation, args.rate)
        q_by_expiry = implied_dividend_yields(parity, snapshot.spot, valuation, args.rate)
        forwards = {e: f.forward for e, f in parity.items()}

    table = solve_chain_ivs(
        snapshot.quotes, snapshot.spot, snapshot.timestamp, r=args.rate, q=args.dividend_yield,
        q_by_expiry=q_by_expiry,
    )

    solved = table["error"].isna()
    n_newton = int((table["method"] == "newton").sum())
    n_brent = int((table["method"] == "brent").sum())
    print(
        f"{snapshot.underlying} @ spot {snapshot.spot} ({snapshot.timestamp}), "
        f"r={args.rate:.4f} q={args.dividend_yield:.4f}"
    )
    if args.forward == "parity":
        print("forward: put-call parity per expiry (median over two-sided strikes nearest spot)")
        for expiry in sorted(table["expiry"].unique()):
            if expiry in parity:
                f = parity[expiry]
                model_f = snapshot.spot * math.exp(
                    (args.rate - args.dividend_yield) * time_to_expiry(valuation, expiry)
                )
                print(
                    f"  {expiry}: F={f.forward:.2f} (model {model_f:.2f}; implied q={q_by_expiry[expiry]:.4f}; "
                    f"{f.pairs} strike(s))"
                )
            else:
                print(f"  {expiry}: no strike quoted two-sided on both legs - model forward kept")
    print(f"{solved.sum()}/{len(table)} contracts solved ({n_newton} newton, {n_brent} brent fallback)")
    if (~solved).any():
        # D59: split the unsolved rows by reason, so a missing quote is not
        # read as a price below the no-arbitrage floor.
        print(f"{(~solved).sum()} contract(s) had no solution:")
        errors = table["error"].fillna("")
        floor = errors.str.contains("below the no-arbitrage floor")
        no_quote = errors == NO_TWO_SIDED_QUOTE
        expiring = errors.str.startswith(AT_OR_PAST_EXPIRY)
        quotes = snapshot.quotes.reset_index(drop=True)
        itm = ((quotes["option_type"] == "CE") & (quotes["strike"] < snapshot.spot)) | (
            (quotes["option_type"] == "PE") & (quotes["strike"] > snapshot.spot)
        )
        one_sided = (quotes["bid"].fillna(0) <= 0) != (quotes["ask"].fillna(0) <= 0)
        print(
            f"  {floor.sum()} below the no-arbitrage floor ({(floor & itm).sum()} in the money at spot; "
            f"{(floor & one_sided).sum()} priced off a one-sided quote)"
        )
        print(f"  {no_quote.sum()} with no two-sided quote (bid 0, ask 0)")
        if expiring.any():
            # P131: e.g. the 27 Oct 2026 batch, quoted on the 27-Oct expiry day.
            print(f"  {expiring.sum()} at or past expiry on the quote date (T <= 0 in whole days)")
        other = ~solved & ~floor & ~no_quote & ~expiring
        if other.any():
            print(f"  {other.sum()} for another reason")

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
            forwards=forwards,
            q_by_expiry=q_by_expiry,
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
        if result.expiries:
            # R26: a fitted-smile count is grid points, so also say how many
            # bends they make and what the quoted strikes themselves show.
            n_grid = sum(c.points for c in result.clusters)
            print(
                f"butterfly: {n_grid} grid point(s) in {len(result.clusters)} cluster(s) on the fitted smile "
                f"({BUTTERFLY_GRID_POINTS}-point grid per expiry)"
            )
            for c in result.clusters:
                print(
                    f"  {c.expiry} K {c.k_low:.2f}-{c.k_high:.2f}: {c.points} point(s), "
                    f"between quoted strikes {c.quoted_low:g} and {c.quoted_high:g}"
                )
            qb = result.quoted
            print(
                f"butterfly on quoted strikes (liquid, consecutive triples, calls and puts): "
                f"{qb.negative_mid}/{qb.triples} negative at mid, "
                f"{qb.negative_executable}/{qb.triples} negative at executable bid/ask"
            )
            for e in (e for e in qb.examples if e.executable_cost < 0):
                k1, k2, k3 = e.strikes
                print(
                    f"  {e.expiry} {e.option_type} {k1:g}/{k2:g}/{k3:g}: mid cost {e.mid_cost:.4f}, "
                    f"executable cost {e.executable_cost:.4f}"
                )

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
