"""Surface construction (Day 4): filter illiquid strikes, interpolate a
smile per expiry, then check the fitted surface for static arbitrage.

Pipeline, in order:

1. :func:`filter_liquid` drops any (expiry, strike, option_type) row whose
   open interest or bid/ask spread fails a documented threshold, plus any
   row the Day-2 solver already flagged as :class:`~volsurface.iv.NoSolution`.
2. :func:`select_otm` keeps one IV per (expiry, strike): the out-of-the-money
   leg, which is the side real liquidity concentrates on (puts below spot,
   calls above). Deep ITM legs are dropped even if they individually passed
   the liquidity filter, so the smile never mixes two option types at the
   same strike.
3. :func:`build_smile` fits a shape-preserving monotone cubic (PCHIP) across
   the surviving strikes for one expiry. PCHIP is used instead of a plain
   cubic spline because it never overshoots between knots - a plain cubic
   spline can dip a smile below zero or add a wiggle the underlying quotes
   never suggested, which would itself look like arbitrage that was never
   really there.
4. :func:`check_butterfly` re-prices the *fitted* smile back into call
   prices on a fine strike grid and requires that curve to be convex - the
   condition a real vertical spread's finite price must satisfy. This is
   the check that matters for anything built on top of the surface later
   (Greeks, interpolated strikes), since that is the curve those stages
   will actually read.
5. :func:`check_calendar` requires total variance (``iv**2 * T``) at a
   shared strike to be non-decreasing from a nearer expiry to a farther
   one - the condition a real calendar spread's finite price must satisfy.

See the README's Limitations section for why the fitted-smile butterfly
check is *less* sensitive than a raw-quote check would be, and why that
matters for what "no violations found" actually proves here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from volsurface.iv import bs_price, time_to_expiry

# Arbitrary but documented, like every other threshold in this repo (see
# iv.py's BOUNDS_EPS / MIN_SOLVABLE_PRICE_ABOVE_FLOOR): NSE mid-cap chains
# routinely show single-digit-thousands open interest at the money and
# near-zero at the wings, and a 20% bid/ask spread is already wide enough
# that the "mid" is a guess, not a price. Neither is calibrated to any
# specific book's execution cost.
DEFAULT_MIN_OPEN_INTEREST = 1000
DEFAULT_MAX_RELATIVE_SPREAD = 0.20

MIN_SMILE_POINTS = 3  # PCHIP needs >= 2; 3 is the least that can show curvature
BUTTERFLY_GRID_POINTS = 200
# Absolute, not relative: call prices near a strike range like RELIANCE's
# (tens to low hundreds of rupees) make a relative tolerance meaningless at
# the wings where the price itself is near zero - the same lesson Day 2's
# no-arbitrage-floor check and Day 3's Greek tolerances already record.
BUTTERFLY_TOL = 1e-4
CALENDAR_TOL = 1e-6  # total variance is O(1e-2) to O(1e-1) here; this is float slack only


class SurfaceError(ValueError):
    """The inputs can't support a smile or an arbitrage check (too few points, bad shape)."""


def relative_spread(bid, ask):
    """``(ask - bid) / mid``, or +inf for a non-positive/degenerate mid.

    Works elementwise on a ``pandas.Series`` (the call ``filter_liquid``
    makes) or on a scalar. A non-positive mid can't happen for a real
    quote, but a synthetic or corrupted row could carry ``bid == ask == 0``;
    treating that as infinitely wide (rather than dividing by zero) keeps
    this a filter predicate, not a possible crash.
    """
    bid = pd.Series(bid) if not isinstance(bid, pd.Series) else bid
    ask = pd.Series(ask) if not isinstance(ask, pd.Series) else ask
    mid = (bid + ask) / 2.0
    safe_mid = mid.where(mid > 0, other=np.nan)
    spread = ((ask - bid) / safe_mid).fillna(float("inf"))
    spread.index = mid.index
    return spread


def filter_liquid(
    iv_table: pd.DataFrame,
    quotes: pd.DataFrame,
    min_open_interest: int = DEFAULT_MIN_OPEN_INTEREST,
    max_relative_spread: float = DEFAULT_MAX_RELATIVE_SPREAD,
) -> pd.DataFrame:
    """Join a solved IV table (from :func:`solve_chain_ivs`) to its quotes and
    keep only rows with a solved IV, sufficient open interest, and a tight
    enough bid/ask spread.

    Returns the merged frame with ``open_interest`` and ``relative_spread``
    columns attached, restricted to the rows that survive - callers that
    want to report *why* a row was dropped should diff against ``iv_table``
    rather than expect a reason column here (unlike :func:`solve_chain_ivs`,
    where a per-row reason is the point; here the reason is just "did not
    meet the threshold", the same for every dropped row).
    """
    merged = iv_table.merge(
        quotes[["expiry", "strike", "option_type", "open_interest", "bid", "ask"]],
        on=["expiry", "strike", "option_type"],
        how="left",
    )
    merged["relative_spread"] = relative_spread(merged["bid"], merged["ask"])
    solved = merged["iv"].notna()
    liquid_oi = merged["open_interest"] >= min_open_interest
    liquid_spread = merged["relative_spread"] <= max_relative_spread
    return merged[solved & liquid_oi & liquid_spread].reset_index(drop=True)


def select_otm(liquid_table: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Keep one row per (expiry, strike): the out-of-the-money leg.

    Puts below spot, calls at or above spot - the side real order flow and
    open interest concentrate on, and the convention the rest of this
    module assumes (one IV per strike, never two).
    """
    rows = []
    for (expiry, strike), group in liquid_table.groupby(["expiry", "strike"]):
        want = "PE" if strike < spot else "CE"
        leg = group[group["option_type"] == want]
        if leg.empty:
            continue
        rows.append(leg.iloc[0])
    if not rows:
        return liquid_table.iloc[0:0]
    return pd.DataFrame(rows).sort_values(["expiry", "strike"]).reset_index(drop=True)


def build_smile(otm_table_one_expiry: pd.DataFrame) -> PchipInterpolator:
    """Fit a monotone cubic (PCHIP) IV smile across strike for one expiry.

    Raises :class:`SurfaceError` if fewer than :data:`MIN_SMILE_POINTS`
    strikes survived filtering - not enough to say anything about shape,
    let alone check it for arbitrage.
    """
    strikes = otm_table_one_expiry["strike"].to_numpy(dtype=float)
    ivs = otm_table_one_expiry["iv"].to_numpy(dtype=float)
    if len(strikes) < MIN_SMILE_POINTS:
        raise SurfaceError(
            f"only {len(strikes)} strike(s) survived filtering - need at least "
            f"{MIN_SMILE_POINTS} to fit a smile"
        )
    order = np.argsort(strikes)
    return PchipInterpolator(strikes[order], ivs[order], extrapolate=False)


@dataclass
class ArbitrageViolation:
    kind: str  # "butterfly" or "calendar"
    expiry: str
    strike: float
    detail: str


@dataclass
class ExpirySmile:
    expiry: str
    T: float
    strikes: np.ndarray  # the filtered, OTM-selected strikes actually used to fit
    smile: PchipInterpolator


def check_butterfly(
    expiry_smile: ExpirySmile,
    spot: float,
    r: float,
    q: float,
    grid_points: int = BUTTERFLY_GRID_POINTS,
    tol: float = BUTTERFLY_TOL,
) -> list[ArbitrageViolation]:
    """Check the *fitted* smile for butterfly arbitrage.

    Re-prices the smile's IV back into a call price on a fine strike grid
    (never the raw quotes - this checks the surface later stages will
    actually read) and requires that curve to be convex: for evenly spaced
    grid strikes, ``C[i-1] - 2*C[i] + C[i+1] >= -tol`` at every interior
    point. A violation means the fitted smile implies a butterfly spread
    (long one wing, short two at the middle strike, long the other wing)
    with a negative price - free money if it were tradable, so evidence the
    fit (or the quotes under it) can't be right.
    """
    lo, hi = expiry_smile.strikes.min(), expiry_smile.strikes.max()
    grid = np.linspace(lo, hi, grid_points)
    ivs = expiry_smile.smile(grid)
    valid = ~np.isnan(ivs)
    grid, ivs = grid[valid], ivs[valid]
    if len(grid) < 3:
        return []

    calls = np.array(
        [bs_price(spot, K, expiry_smile.T, r, iv, "CE", q) for K, iv in zip(grid, ivs)]
    )
    curvature = calls[:-2] - 2 * calls[1:-1] + calls[2:]
    violations = []
    for K, curv in zip(grid[1:-1], curvature):
        if curv < -tol:
            violations.append(
                ArbitrageViolation(
                    kind="butterfly",
                    expiry=expiry_smile.expiry,
                    strike=float(K),
                    detail=f"call-price curvature {curv:.6f} < -{tol:g} at K={K:.2f}",
                )
            )
    return violations


def check_calendar(
    expiry_smiles: list[ExpirySmile], tol: float = CALENDAR_TOL
) -> list[ArbitrageViolation]:
    """Check total variance ``iv**2 * T`` is non-decreasing across expiries
    at every strike shared by consecutive expiries (nearest to farthest).

    A strike a farther expiry's smile can't evaluate (outside its fitted
    range) is skipped for that pair rather than treated as a violation -
    :class:`PchipInterpolator` returns NaN there by construction
    (``extrapolate=False``), and extrapolating a smile past its own data to
    manufacture a comparison would be worse than not checking that strike.
    """
    ordered = sorted(expiry_smiles, key=lambda e: e.T)
    violations = []
    for near, far in zip(ordered, ordered[1:]):
        strikes = np.union1d(near.strikes, far.strikes)
        near_iv = near.smile(strikes)
        far_iv = far.smile(strikes)
        for K, iv_n, iv_f in zip(strikes, near_iv, far_iv):
            if np.isnan(iv_n) or np.isnan(iv_f):
                continue
            w_near = iv_n**2 * near.T
            w_far = iv_f**2 * far.T
            if w_far < w_near - tol:
                violations.append(
                    ArbitrageViolation(
                        kind="calendar",
                        expiry=far.expiry,
                        strike=float(K),
                        detail=(
                            f"total variance at K={K:.2f} drops from {w_near:.6f} "
                            f"({near.expiry}, T={near.T:.4f}) to {w_far:.6f} "
                            f"({far.expiry}, T={far.T:.4f})"
                        ),
                    )
                )
    return violations


@dataclass
class SurfaceResult:
    expiries: list[ExpirySmile]
    dropped_counts: dict  # expiry -> {"filtered_out": int, "kept": int}
    violations: list[ArbitrageViolation] = field(default_factory=list)


def build_surface(
    iv_table: pd.DataFrame,
    quotes: pd.DataFrame,
    spot: float,
    valuation_date,
    r: float,
    q: float,
    min_open_interest: int = DEFAULT_MIN_OPEN_INTEREST,
    max_relative_spread: float = DEFAULT_MAX_RELATIVE_SPREAD,
) -> SurfaceResult:
    """Run the full Day-4 pipeline: filter, select OTM, fit a smile per
    expiry, then check the fitted surface for butterfly and calendar
    arbitrage.

    Expiries with fewer than :data:`MIN_SMILE_POINTS` surviving strikes are
    skipped (recorded in ``dropped_counts``, not silently absent) rather
    than raising, so one thin expiry doesn't take down the whole surface.
    """
    liquid = filter_liquid(iv_table, quotes, min_open_interest, max_relative_spread)
    otm = select_otm(liquid, spot)

    expiry_smiles: list[ExpirySmile] = []
    dropped_counts = {}
    for expiry, total_group in iv_table.groupby("expiry"):
        kept_group = otm[otm["expiry"] == expiry]
        dropped_counts[expiry] = {
            "total": len(total_group),
            "kept": len(kept_group),
        }
        if len(kept_group) < MIN_SMILE_POINTS:
            continue
        T = time_to_expiry(valuation_date, expiry)
        smile = build_smile(kept_group)
        strikes = np.sort(kept_group["strike"].to_numpy(dtype=float))
        expiry_smiles.append(ExpirySmile(expiry=expiry, T=T, strikes=strikes, smile=smile))

    violations: list[ArbitrageViolation] = []
    for es in expiry_smiles:
        violations.extend(check_butterfly(es, spot, r, q))
    violations.extend(check_calendar(expiry_smiles))

    return SurfaceResult(
        expiries=expiry_smiles, dropped_counts=dropped_counts, violations=violations
    )
