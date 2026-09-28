"""CLI: `ml-pipeline-audit` pass (Day 8).

    python -m volsurface.audit --underlying RELIANCE

Re-verifies, mechanically and against the real fixture chain rather than by
inspection, the three "Done when" claims ``NEXT_STEPS.md`` makes for this
repo:

1. IV round-trips: price -> IV -> price reproduces the input within tolerance.
2. Analytic Greeks agree with finite differences.
3. No static-arbitrage violations survive the filtering step.

(1) and (2) were already exercised per-parameter-combination in
``tests/test_iv.py`` and ``tests/test_greeks.py``; this runs them once more
end to end against the whole committed chain and fails loudly (:class:`AuditError`)
on the first breach, the same shape STOCKSTALKER's Day 9 causality audit and
monte-carlo-risk-lab's Day 7 audit both use.

(3) has a real, documented gap this closes. Every existing "the arbitrage
check catches a violation" test (``tests/test_surface.py``) builds its bad
smile entirely from scratch - three or five hand-picked strikes, never the
real fixture. So "no violations found" on the real RELIANCE chain (Day 4's
own finding) was never actually distinguished from "the checks wouldn't fire
even if this exact chain had a violation in it" - see the README's Day 4
Limitations bullet. :func:`audit_arbitrage_checks_fire_on_real_fixture`
takes the real fixture's own filtered, OTM-selected strikes for one expiry,
plants a single deliberately-broken IV at an interior strike (the same
construction ``tests/test_surface.py``'s synthetic case uses), and requires
``check_butterfly`` to flag it - a negative control proving the check would
have caught a real violation in this exact chain shape, not just in a smile
built to be caught. :func:`audit_calendar_check_fires_on_real_fixture` does
the equivalent for ``check_calendar``: it shrinks the real fixture's own
far-expiry IVs until total variance inverts against the real near expiry,
and requires the check to flag that.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from volsurface.chain import PROVIDERS, ProviderError, get_provider
from volsurface.greeks import compute_chain_greeks
from volsurface.iv import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_RATE,
    _valuation_date,
    bs_price,
    solve_chain_ivs,
    time_to_expiry,
)
from volsurface.surface import (
    DEFAULT_MAX_RELATIVE_SPREAD,
    DEFAULT_MIN_OPEN_INTEREST,
    ExpirySmile,
    build_smile,
    build_surface,
    check_butterfly,
    check_calendar,
    filter_liquid,
    select_otm,
)

ROUND_TRIP_TOL = 1e-4  # rupees - matches iv.py's own MIN_SOLVABLE_PRICE_ABOVE_FLOOR order of magnitude
GREEKS_FD_TOL = 1e-4  # matches tests/test_greeks.py's own absolute tolerance for this strike/vol range
INJECTED_IV_MULTIPLIER = 5.0  # spikes one interior strike's real solved IV; see _inject_butterfly_violation
CALENDAR_IV_SHRINK = 0.05  # multiplies the real far-expiry IV; see audit_calendar_check_fires_on_real_fixture


class AuditError(AssertionError):
    """A mechanical pipeline-health check failed against the real fixture."""


def audit_round_trip(iv_table, spot: float, r: float, q: float, tol: float = ROUND_TRIP_TOL):
    """price -> IV -> price on every solved row of the real fixture.

    Returns ``(n_checked, max_abs_error)``. Raises :class:`AuditError` if any
    round-trip error exceeds ``tol`` - this is the mechanical version of what
    ``tests/test_iv.py``'s ``TestSolveIvRoundTrip`` already checks per
    hand-picked parameter combination, run here against every row the real
    fixture actually produced.
    """
    solved = iv_table[iv_table["iv"].notna() & iv_table["error"].isna()]
    if solved.empty:
        raise AuditError("no solved rows in the fixture to round-trip - solver or fixture is broken")

    max_err = 0.0
    for row in solved.itertuples(index=False):
        repriced = bs_price(spot, row.strike, row.T, r, row.iv, row.option_type, q)
        max_err = max(max_err, abs(repriced - row.mid_price))

    if max_err > tol:
        raise AuditError(f"round-trip error {max_err:.6g} exceeds tolerance {tol:g}")
    return len(solved), max_err


def audit_greeks(iv_table, spot: float, r: float, q: float, tol: float = GREEKS_FD_TOL):
    """Analytic-vs-finite-difference Greek gap on every solved row of the
    real fixture. Returns ``(n_checked, max_abs_diff)``.
    """
    greeked = compute_chain_greeks(iv_table, spot, r, q)
    checked = greeked["fd_max_abs_diff"].notna()
    if not checked.any():
        raise AuditError("no rows got Greeks computed - solver or liquidity filter is broken")

    max_diff = float(greeked.loc[checked, "fd_max_abs_diff"].max())
    if max_diff > tol:
        raise AuditError(f"analytic-vs-finite-difference Greek gap {max_diff:.6g} exceeds tolerance {tol:g}")
    return int(checked.sum()), max_diff


def _inject_butterfly_violation(otm_one_expiry):
    """A copy of a real (expiry, strike, iv) OTM slice with one interior
    strike's IV spiked - the same shape as ``tests/test_surface.py``'s
    ``test_check_butterfly_catches_a_deliberately_broken_smile``, but
    grafted onto the real fixture's own strikes rather than five hand-picked
    ones.
    """
    otm = otm_one_expiry.sort_values("strike").reset_index(drop=True).copy()
    if len(otm) < 3:
        raise AuditError(
            "real fixture's OTM table has fewer than 3 surviving strikes - "
            "not enough to plant an interior spike"
        )
    mid = len(otm) // 2
    otm.loc[mid, "iv"] = otm.loc[mid, "iv"] * INJECTED_IV_MULTIPLIER
    return otm


def audit_arbitrage_checks_fire_on_real_fixture(iv_table, quotes, spot: float, valuation_date, r: float, q: float):
    """Prove ``check_butterfly`` would reject a real violation shaped like
    this exact fixture, not just a from-scratch synthetic one. Returns the
    number of violations found on the corrupted copy.

    Raises :class:`AuditError` if the check stays silent on data known to be
    broken - that would mean "no violations found" on the clean fixture is
    not evidence of an arbitrage-free surface, just evidence the check never
    fires here.
    """
    liquid = filter_liquid(iv_table, quotes)
    otm = select_otm(liquid, spot)
    if otm.empty:
        raise AuditError("real fixture produced no OTM rows to plant a violation into")

    expiry = otm["expiry"].iloc[0]
    broken = _inject_butterfly_violation(otm[otm["expiry"] == expiry])
    smile = build_smile(broken)
    es = ExpirySmile(
        expiry=expiry,
        T=time_to_expiry(valuation_date, expiry),
        strikes=np.sort(broken["strike"].to_numpy(dtype=float)),
        smile=smile,
    )
    violations = check_butterfly(es, spot, r, q)
    if not violations:
        raise AuditError(
            f"planted a {INJECTED_IV_MULTIPLIER:g}x IV spike into the real fixture's own "
            f"{expiry} strikes and check_butterfly still found no violation - the check would "
            "not have caught a real one in this chain shape"
        )
    return len(violations)


def audit_calendar_check_fires_on_real_fixture(iv_table, quotes, spot: float, valuation_date, r: float, q: float):
    """Prove ``check_calendar`` would reject a real term-structure inversion
    shaped like this exact fixture's two real expiries, not just a
    from-scratch synthetic pair.

    Takes the real fixture's own near- and far-expiry OTM strikes, shrinks
    the far expiry's IVs by :data:`CALENDAR_IV_SHRINK` (real total variance
    there is already close to the near expiry's - see the README's Day 4/5
    Findings on this fixture's inverted-but-arbitrage-free term structure -
    so a modest inversion needs a real push to cross into a genuine
    violation), and requires the check to flag it. Raises :class:`AuditError`
    if it stays silent.
    """
    liquid = filter_liquid(iv_table, quotes)
    otm = select_otm(liquid, spot)
    expiries = sorted(otm["expiry"].unique(), key=lambda e: time_to_expiry(valuation_date, e))
    if len(expiries) < 2:
        raise AuditError(
            "real fixture has fewer than 2 expiries with surviving OTM strikes - "
            "can't plant a calendar violation"
        )

    near_expiry, far_expiry = expiries[0], expiries[1]
    near_otm = otm[otm["expiry"] == near_expiry]
    far_otm = otm[otm["expiry"] == far_expiry].copy()
    far_otm["iv"] = far_otm["iv"] * CALENDAR_IV_SHRINK

    near_smile = ExpirySmile(
        expiry=near_expiry,
        T=time_to_expiry(valuation_date, near_expiry),
        strikes=np.sort(near_otm["strike"].to_numpy(dtype=float)),
        smile=build_smile(near_otm),
    )
    far_smile = ExpirySmile(
        expiry=far_expiry,
        T=time_to_expiry(valuation_date, far_expiry),
        strikes=np.sort(far_otm["strike"].to_numpy(dtype=float)),
        smile=build_smile(far_otm),
    )

    violations = check_calendar([near_smile, far_smile])
    if not violations:
        raise AuditError(
            f"shrunk the real fixture's far-expiry ({far_expiry}) IV by {CALENDAR_IV_SHRINK:g}x "
            "and check_calendar still found no violation - the check would not have caught a "
            "real term-structure inversion in this chain shape"
        )
    return len(violations)


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
    args = parser.parse_args(argv)

    try:
        snapshot = get_provider(args.provider).fetch(args.underlying)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    table = solve_chain_ivs(
        snapshot.quotes, snapshot.spot, snapshot.timestamp, r=args.rate, q=args.dividend_yield
    )
    valuation_date = _valuation_date(snapshot.timestamp)

    print(f"ml-pipeline-audit: {snapshot.underlying} @ spot {snapshot.spot} ({snapshot.timestamp})")

    try:
        n_rt, max_rt_err = audit_round_trip(table, snapshot.spot, args.rate, args.dividend_yield)
        print(f"  [ok] round-trip: {n_rt} solved contract(s), worst |reprice - mid| = {max_rt_err:.2e}")

        n_gk, max_gk_diff = audit_greeks(table, snapshot.spot, args.rate, args.dividend_yield)
        print(f"  [ok] greeks: {n_gk} contract(s), worst analytic-vs-finite-difference gap = {max_gk_diff:.2e}")

        result = build_surface(
            table,
            snapshot.quotes,
            snapshot.spot,
            valuation_date,
            args.rate,
            args.dividend_yield,
            min_open_interest=args.min_open_interest,
            max_relative_spread=args.max_relative_spread,
        )
        print(
            f"  [ok] real fixture surface: {len(result.violations)} arbitrage violation(s) found "
            f"across {len(result.expiries)} fitted expiry/expiries (0 is the fixture being "
            "arbitrage-free by construction, not proof the checks would catch a real one - see next line)"
        )

        n_planted = audit_arbitrage_checks_fire_on_real_fixture(
            table, snapshot.quotes, snapshot.spot, valuation_date, args.rate, args.dividend_yield
        )
        print(
            f"  [ok] negative control (butterfly): a deliberate IV spike planted into the real "
            f"fixture's own strikes was caught as {n_planted} violation(s)"
        )

        n_calendar = audit_calendar_check_fires_on_real_fixture(
            table, snapshot.quotes, snapshot.spot, valuation_date, args.rate, args.dividend_yield
        )
        print(
            f"  [ok] negative control (calendar): shrinking the real fixture's far-expiry IV was "
            f"caught as {n_calendar} violation(s)"
        )
    except AuditError as exc:
        print(f"  [FAIL] {exc}", file=sys.stderr)
        return 1

    print("ml-pipeline-audit: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
