"""CLI: solve implied volatility for every contract in a chain snapshot.

    python -m volsurface.solve --underlying RELIANCE --provider fixture

Reads a chain via any registered provider (``fixture`` by default, so this
runs with no network), solves each row's bid/ask mid for implied vol, prints
a summary, and optionally writes the full per-contract table as CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from volsurface.chain import PROVIDERS, ProviderError, get_provider
from volsurface.iv import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, solve_chain_ivs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", required=True, help="e.g. RELIANCE")
    parser.add_argument("--provider", default="fixture", choices=sorted(PROVIDERS))
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="annualised risk-free rate")
    parser.add_argument(
        "--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD, dest="dividend_yield"
    )
    parser.add_argument("--out", type=Path, default=None, help="write the full table as CSV here")
    args = parser.parse_args(argv)

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

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
