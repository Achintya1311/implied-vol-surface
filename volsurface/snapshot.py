"""CLI: capture a live option chain and write it as a committed-shape fixture.

    python -m volsurface.snapshot --underlying RELIANCE --provider nse

Defaults to writing under ``fixtures/chains/<UNDERLYING>.json``. A failed
live fetch (network blocked, NSE rate limit, no expiries on yfinance) exits
non-zero with the provider's own error message rather than falling back to
silently doing nothing - the whole point of this command is the live fetch;
if it fails, the fixture stays whatever it already was.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from volsurface.chain import FIXTURE_DIR, PROVIDERS, ProviderError, capture_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", required=True, help="e.g. RELIANCE, NIFTY")
    parser.add_argument("--provider", default="nse", choices=sorted(PROVIDERS))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="defaults to fixtures/chains/<UNDERLYING>.json",
    )
    parser.add_argument(
        "--raw-capture-dir",
        type=Path,
        default=None,
        help=(
            "--provider nse only: also write each expiry's raw, unmerged "
            "NSE response under this directory, in the same shape "
            "automation/capture_chains.py's home-connection job writes "
            "under fixtures/chains/raw/ (D25)"
        ),
    )
    args = parser.parse_args(argv)

    out_path = args.out or (FIXTURE_DIR / f"{args.underlying.upper()}.json")

    try:
        snapshot = capture_snapshot(
            args.underlying,
            provider=args.provider,
            out_path=out_path,
            raw_capture_dir=args.raw_capture_dir,
        )
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{snapshot.source}: {len(snapshot.quotes)} quotes for {snapshot.underlying} "
        f"@ spot {snapshot.spot} ({snapshot.timestamp}) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
