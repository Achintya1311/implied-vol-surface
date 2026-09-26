"""Analytic Greeks for the Black-Scholes-Merton model (Day 3).

Delta, gamma and theta are added here; vega is already ``iv.bs_vega`` (Day 2
needed it for Newton's step) and is reused rather than redefined, so there is
exactly one vega formula in the repo.

Every analytic formula is cross-checked against a central finite-difference
approximation of the same pricer (:func:`iv.bs_price`) rather than trusted on
its own - that is the "Done when" bar this repo set for itself
(``README`` Correctness gate). The cross-check does not exist to *find* the
answer, only to catch a sign error or a transcription mistake against a
textbook formula; it is only as good as its bump sizes, which is why they are
named constants with the reasoning attached rather than magic numbers buried
in the finite-difference functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from scipy.stats import norm

from volsurface.iv import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, bs_price, bs_vega

DAYS_PER_YEAR = 365.0  # matches iv.time_to_expiry's ACT/365 convention

# Central-difference bump sizes for the finite-difference cross-check.
#
# Delta and vega are first derivatives of a smooth pricer: their
# finite-difference error is O(h^2) truncation vs O(eps/h) roundoff, and for
# prices in the O(1-1000) range used here that trade-off bottoms out around
# h ~ 1e-4 (relative for spot and sigma). Gamma is a *second* derivative -
# the same h leaves the numerator's three price terms differing only in
# their 8th-9th significant digit, which is roundoff noise, not signal - so
# gamma gets a ten-times-larger relative bump, trading a little more
# truncation error for a lot less cancellation error. Theta bumps calendar
# time, which is always small in absolute terms (contracts here run
# 5-30 days), so its bump is relative to T rather than a fixed constant that
# would swallow a short-dated contract's own T.
#
# tests/test_greeks.py holds analytic and finite-difference Greeks to
# absolute tolerances (not relative): delta/gamma/vega all go genuinely near
# zero deep ITM/OTM, where a tiny absolute FD error is a huge relative one
# despite the cross-check working fine - the same territory Day 2's solver
# found numerically awkward for the same underlying reason.
FD_BUMP_S_REL = 1e-4
FD_BUMP_SIGMA = 1e-4
FD_BUMP_T_REL = 1e-4


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float  # per calendar day (theta_annual / 365), the convention a desk quotes


def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _validate(sigma: float, T: float) -> None:
    if sigma <= 0 or T <= 0:
        raise ValueError(f"greeks require sigma > 0 and T > 0, got sigma={sigma}, T={T}")


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> float:
    """d(price)/d(S)."""
    _validate(sigma, T)
    d1 = _d1(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    if option_type == "CE":
        return disc_q * norm.cdf(d1)
    if option_type == "PE":
        return disc_q * (norm.cdf(d1) - 1.0)
    raise ValueError(f"unknown option_type: {option_type!r}")


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = DEFAULT_DIVIDEND_YIELD) -> float:
    """d(delta)/d(S) - identical for a call and put at the same strike/T/sigma."""
    _validate(sigma, T)
    d1 = _d1(S, K, T, r, sigma, q)
    return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> float:
    """d(price)/d(t), calendar time passing (i.e. -d(price)/d(T)), per calendar day."""
    _validate(sigma, T)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    decay_term = -(S * disc_q * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type == "CE":
        theta_annual = decay_term - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    elif option_type == "PE":
        theta_annual = decay_term + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError(f"unknown option_type: {option_type!r}")
    return theta_annual / DAYS_PER_YEAR


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> Greeks:
    return Greeks(
        delta=bs_delta(S, K, T, r, sigma, option_type, q),
        gamma=bs_gamma(S, K, T, r, sigma, q),
        vega=bs_vega(S, K, T, r, sigma, q),
        theta=bs_theta(S, K, T, r, sigma, option_type, q),
    )


def fd_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> float:
    h = S * FD_BUMP_S_REL
    return (
        bs_price(S + h, K, T, r, sigma, option_type, q) - bs_price(S - h, K, T, r, sigma, option_type, q)
    ) / (2.0 * h)


def fd_gamma(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> float:
    h = S * FD_BUMP_S_REL * 10.0  # a second derivative needs a wider bump - see module docstring
    up = bs_price(S + h, K, T, r, sigma, option_type, q)
    mid = bs_price(S, K, T, r, sigma, option_type, q)
    down = bs_price(S - h, K, T, r, sigma, option_type, q)
    return (up - 2.0 * mid + down) / (h * h)


def fd_vega(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> float:
    h = FD_BUMP_SIGMA
    return (
        bs_price(S, K, T, r, sigma + h, option_type, q) - bs_price(S, K, T, r, sigma - h, option_type, q)
    ) / (2.0 * h)


def fd_theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> float:
    h = T * FD_BUMP_T_REL
    # theta is decay as calendar time *passes*, i.e. T shrinking, hence the sign flip.
    raw = -(
        bs_price(S, K, T + h, r, sigma, option_type, q) - bs_price(S, K, T - h, r, sigma, option_type, q)
    ) / (2.0 * h)
    return raw / DAYS_PER_YEAR


def fd_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> Greeks:
    return Greeks(
        delta=fd_delta(S, K, T, r, sigma, option_type, q),
        gamma=fd_gamma(S, K, T, r, sigma, option_type, q),
        vega=fd_vega(S, K, T, r, sigma, option_type, q),
        theta=fd_theta(S, K, T, r, sigma, option_type, q),
    )


GREEK_COLUMNS = ["delta", "gamma", "vega", "theta", "fd_max_abs_diff"]


def compute_chain_greeks(
    iv_table: pd.DataFrame,
    spot: float,
    r: float = DEFAULT_RATE,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> pd.DataFrame:
    """Attach analytic Greeks to every solved row of an :func:`iv.solve_chain_ivs` table.

    A row with no IV solution has no sigma to differentiate the pricer at, so
    it gets NaN Greeks rather than a fabricated number - the same explicit-gap
    convention ``solve_chain_ivs`` already uses for unsolved rows.
    ``fd_max_abs_diff`` is the largest absolute difference between an analytic
    Greek and its finite-difference cross-check on that row, carried through
    to the output so a caller (or a test) can catch a live regression, not
    just the parameter combinations covered by the unit tests.
    """
    out = iv_table.copy()
    for col in GREEK_COLUMNS:
        out[col] = float("nan")

    for idx, row in out.iterrows():
        # error is None when solve_chain_ivs sets it directly, but round-trips
        # through a DataFrame column as NaN - pd.notna() catches both shapes,
        # a plain "is not None" check silently would not.
        if pd.isna(row["iv"]) or pd.notna(row["error"]):
            continue
        S, K, T, sigma, option_type = spot, row["strike"], row["T"], row["iv"], row["option_type"]
        analytic = bs_greeks(S, K, T, r, sigma, option_type, q)
        approx = fd_greeks(S, K, T, r, sigma, option_type, q)
        out.at[idx, "delta"] = analytic.delta
        out.at[idx, "gamma"] = analytic.gamma
        out.at[idx, "vega"] = analytic.vega
        out.at[idx, "theta"] = analytic.theta
        out.at[idx, "fd_max_abs_diff"] = max(
            abs(analytic.delta - approx.delta),
            abs(analytic.gamma - approx.gamma),
            abs(analytic.vega - approx.vega),
            abs(analytic.theta - approx.theta),
        )
    return out
