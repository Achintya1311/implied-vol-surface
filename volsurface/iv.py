"""Implied volatility solver (Day 2).

Given a European option's quoted price, solves for the Black-Scholes-Merton
implied volatility: Newton-Raphson using analytic vega first (fast, usually
converges in a handful of iterations near the money), falling back to
Brent's method (bounded, derivative-free, guaranteed to find a root if one
is bracketed) when Newton's vega goes flat or the iterate wanders outside a
sane volatility range - both of which happen routinely for deep ITM/OTM
contracts.

A quote whose price sits outside the no-arbitrage bounds for *any*
volatility (a stale quote, a bad tick, an expired contract) has no
implied vol at all. That is reported as :class:`NoSolution` with the reason,
never as a silently-returned NaN - the day-1 fixture already established
that convention for parse failures, and it applies here too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# fixtures/chains/RELIANCE.json was itself Black-Scholes-priced with r=6.5%
# and no dividend adjustment (see README Findings for Day 1) - matching that
# here makes round-tripping the fixture a fair test of the solver, not a
# mismatched-assumptions test of the fixture.
DEFAULT_RATE = 0.065
DEFAULT_DIVIDEND_YIELD = 0.0

MAX_NEWTON_ITER = 50
SIGMA_TOL = 1e-8  # Newton stops once its own step is this small, in sigma units
MIN_VEGA = 1e-8  # below this a Newton step divides by noise, not signal
BRENT_LO = 1e-6
BRENT_HI = 5.0  # 500% annualised vol - generous enough to bracket any real quote
BOUNDS_EPS = 1e-9  # float slack around the no-arbitrage bound check
MIN_SOLVABLE_PRICE_ABOVE_FLOOR = 1e-6  # NSE's real tick size (0.05) sits well above this


class NoSolution(Exception):
    """No implied volatility exists for this quote.

    Distinct from the solver failing to converge: this means the price
    itself is outside what any non-negative volatility can produce (below
    intrinsic, above spot/strike, or the contract has already expired).
    """


@dataclass
class IVResult:
    iv: float
    method: str  # "newton" or "brent"
    iterations: int
    converged: bool


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> float:
    """Black-Scholes-Merton price for a European call (``CE``) or put (``PE``)."""
    if sigma <= 0 or T <= 0:
        raise ValueError(f"bs_price requires sigma > 0 and T > 0, got sigma={sigma}, T={T}")
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if option_type == "CE":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    if option_type == "PE":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    raise ValueError(f"unknown option_type: {option_type!r}")


def bs_vega(
    S: float, K: float, T: float, r: float, sigma: float, q: float = DEFAULT_DIVIDEND_YIELD
) -> float:
    """d(price)/d(sigma) - identical for calls and puts under Black-Scholes-Merton."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


def _no_arbitrage_bounds(
    S: float, K: float, T: float, r: float, option_type: str, q: float
) -> tuple[float, float]:
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if option_type == "CE":
        return max(0.0, S * disc_q - K * disc_r), S * disc_q
    return max(0.0, K * disc_r - S * disc_q), K * disc_r


def solve_iv(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = DEFAULT_DIVIDEND_YIELD,
    sigma0: float | None = None,
) -> IVResult:
    """Solve for implied volatility given a quoted European option price.

    Raises :class:`NoSolution` when the price is outside the arbitrage-free
    bounds for any sigma, or the contract has non-positive time to expiry.
    """
    if T <= 0:
        raise NoSolution(f"non-positive time to expiry ({T})")
    if S <= 0 or K <= 0:
        raise NoSolution(f"non-positive spot or strike (S={S}, K={K})")

    lo, hi = _no_arbitrage_bounds(S, K, T, r, option_type, q)
    if price < lo - BOUNDS_EPS:
        raise NoSolution(
            f"price {price:.4f} is below the no-arbitrage floor {lo:.4f} "
            f"for {option_type} K={K} T={T:.4f}y"
        )
    if price > hi + BOUNDS_EPS:
        raise NoSolution(
            f"price {price:.4f} is above the no-arbitrage ceiling {hi:.4f} "
            f"for {option_type} K={K} T={T:.4f}y"
        )
    if abs(price - lo) < MIN_SOLVABLE_PRICE_ABOVE_FLOOR:
        # A price this close to its own no-arbitrage floor underflows to the
        # same double-precision value across a wide range of sigma (a deep,
        # short-dated OTM option's price genuinely is ~0 for any sigma below
        # some point) - there is no single sigma double precision can
        # distinguish as *the* answer, so reporting one would be a
        # confident-looking guess, not a solution.
        raise NoSolution(
            f"price {price:.4f} is within {MIN_SOLVABLE_PRICE_ABOVE_FLOOR} of the "
            f"no-arbitrage floor {lo:.4f} - too close to zero for double precision "
            f"to recover a unique sigma for {option_type} K={K} T={T:.4f}y"
        )

    sigma = max(sigma0 if sigma0 is not None else 0.3, BRENT_LO)
    for i in range(1, MAX_NEWTON_ITER + 1):
        model_price = bs_price(S, K, T, r, sigma, option_type, q)
        diff = model_price - price
        vega = bs_vega(S, K, T, r, sigma, q)
        if vega < MIN_VEGA:
            break  # flat vega: deep ITM/OTM territory, Newton is unreliable here
        step = diff / vega
        # Convergence is judged on the step *in sigma*, not on the price
        # residual: deep ITM/OTM prices are so insensitive to sigma that a
        # tiny price residual can still hide a large sigma error - checking
        # the residual alone would report a confident, wrong IV instead of
        # falling through to Brent's bounded search.
        if abs(step) < SIGMA_TOL:
            return IVResult(iv=sigma - step, method="newton", iterations=i, converged=True)
        sigma -= step
        if not (BRENT_LO < sigma < BRENT_HI):
            break  # stepped outside a sane vol range - let Brent's bounds take over

    # Brent's method: bounded and derivative-free. The no-arbitrage check
    # above guarantees a root is bracketed in [BRENT_LO, BRENT_HI] for any
    # price strictly between the bounds, since bs_price(BRENT_LO) sits at
    # (essentially) the floor and bs_price(BRENT_HI) at (essentially) the
    # ceiling for realistic T.
    def objective(x: float) -> float:
        return bs_price(S, K, T, r, x, option_type, q) - price

    f_lo, f_hi = objective(BRENT_LO), objective(BRENT_HI)
    if f_lo * f_hi > 0:
        raise NoSolution(
            f"price {price:.4f} not bracketed by sigma in [{BRENT_LO}, {BRENT_HI}] "
            f"for {option_type} K={K} T={T:.4f}y (f_lo={f_lo:.6f}, f_hi={f_hi:.6f})"
        )
    root, results = brentq(objective, BRENT_LO, BRENT_HI, xtol=1e-10, full_output=True)
    return IVResult(
        iv=root, method="brent", iterations=results.iterations, converged=bool(results.converged)
    )


def time_to_expiry(valuation_date: date, expiry: str) -> float:
    """ACT/365 year fraction from ``valuation_date`` to ``expiry`` (``YYYY-MM-DD``)."""
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    return (expiry_date - valuation_date).days / 365.0


def _valuation_date(timestamp: str) -> date:
    try:
        return datetime.fromisoformat(timestamp).date()
    except ValueError:
        return datetime.now().date()


def solve_chain_ivs(
    quotes: pd.DataFrame,
    spot: float,
    timestamp: str,
    r: float = DEFAULT_RATE,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> pd.DataFrame:
    """Solve implied vol for every row of a chain's quotes frame.

    Uses the bid/ask mid as the observed price (more robust than a possibly
    stale ``last_price``); a row missing both bid and ask is itself a
    no-solution case, not a silent skip. Every row keeps its ``iv`` as NaN
    plus an explicit ``error`` reason when no solution exists, rather than
    dropping the row - a caller building a surface later decides how to
    treat the gap, but the gap is visible here.
    """
    valuation_date = _valuation_date(timestamp)
    rows = []
    for row in quotes.itertuples(index=False):
        mid = None
        if pd.notna(row.bid) and pd.notna(row.ask):
            mid = (row.bid + row.ask) / 2.0
        elif pd.notna(row.last_price):
            mid = row.last_price

        result = {
            "expiry": row.expiry,
            "strike": row.strike,
            "option_type": row.option_type,
            "mid_price": mid,
            "T": None,
            "iv": float("nan"),
            "method": None,
            "converged": False,
            "error": None,
        }
        if mid is None:
            result["error"] = "no bid/ask/last_price to solve against"
            rows.append(result)
            continue

        T = time_to_expiry(valuation_date, row.expiry)
        result["T"] = T
        try:
            solved = solve_iv(mid, spot, row.strike, T, r, row.option_type, q)
        except NoSolution as exc:
            result["error"] = str(exc)
            rows.append(result)
            continue

        result["iv"] = solved.iv
        result["method"] = solved.method
        result["converged"] = solved.converged
        rows.append(result)

    return pd.DataFrame(
        rows,
        columns=["expiry", "strike", "option_type", "mid_price", "T", "iv", "method", "converged", "error"],
    )
