"""25-delta risk-reversal skew (Day 7): the delta-selected strike the v0.6
contract's ``skew_25d`` field needs, and Day 6's history job explicitly left
out (see README Limitations - "Day 3's Greeks have not been wired into the
Day 4 surface/Day 6 history pipeline").

``skew_25d`` is defined here as ``iv(25-delta put) - iv(25-delta call)``: the
classic risk-reversal sign convention, positive on a downward equity skew
(OTM puts pricier than OTM calls) like the one this fixture was built with.

Finding "the strike where delta is +-0.25" needs the strike and its own
fitted IV at the same time - delta depends on sigma, and sigma here comes
from the fitted smile evaluated *at* that strike, not a separate lookup - so
this is a 1D root find over ``delta(K, smile(K)) - target`` bounded to the
expiry's own fitted domain (:attr:`ExpirySmile.strikes`'s min/max), never
extrapolated past the strikes Day 4 actually filtered and arbitrage-checked.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import brentq

from volsurface.greeks import bs_delta
from volsurface.surface import ExpirySmile

TARGET_ABS_DELTA = 0.25


class SkewError(ValueError):
    """No strike inside the fitted smile's own domain matches the target delta."""


def _signed_delta(
    expiry_smile: ExpirySmile, spot: float, r: float, q: float, option_type: str, K: float
) -> float:
    iv = float(expiry_smile.smile(K))
    if iv != iv:  # NaN: K outside the fitted (extrapolate=False) domain
        raise SkewError(
            f"K={K:g} is outside {expiry_smile.expiry}'s fitted domain "
            f"[{expiry_smile.strikes.min():g}, {expiry_smile.strikes.max():g}]"
        )
    return bs_delta(spot, K, expiry_smile.T, r, iv, option_type, q)


def find_delta_strike(
    expiry_smile: ExpirySmile,
    spot: float,
    r: float,
    q: float,
    option_type: str,
    target_abs_delta: float = TARGET_ABS_DELTA,
) -> float:
    """The strike inside ``expiry_smile``'s fitted domain whose delta
    (evaluated at that strike's own fitted IV) equals ``target_abs_delta`` in
    magnitude - positive for a call, negative for a put.

    Raises :class:`SkewError` if the target isn't bracketed by the delta at
    the smile's two endpoint strikes: an honest refusal, the same convention
    :func:`volsurface.iv.solve_iv` already uses for a price with no
    recoverable root, rather than returning the nearest endpoint as if it
    were the answer.
    """
    if option_type not in ("CE", "PE"):
        raise ValueError(f"unknown option_type: {option_type!r}")
    lo = float(expiry_smile.strikes.min())
    hi = float(expiry_smile.strikes.max())
    sign = 1.0 if option_type == "CE" else -1.0
    target = sign * target_abs_delta

    def objective(K: float) -> float:
        return _signed_delta(expiry_smile, spot, r, q, option_type, K) - target

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0:
        raise SkewError(
            f"{option_type} {target_abs_delta:.2f}-delta strike not bracketed by "
            f"{expiry_smile.expiry}'s fitted domain [{lo:g}, {hi:g}] "
            f"(delta there ranges {f_lo + target:.4f} to {f_hi + target:.4f})"
        )
    return brentq(objective, lo, hi, xtol=1e-6)


@dataclass
class Skew25D:
    expiry: str
    call_strike: float
    call_iv: float
    put_strike: float
    put_iv: float
    skew: float  # put_iv - call_iv; positive on a downward equity skew


def compute_skew_25d(
    expiry_smile: ExpirySmile, spot: float, r: float, q: float,
    target_abs_delta: float = TARGET_ABS_DELTA,
) -> Skew25D:
    """The 25-delta risk-reversal skew for one fitted expiry.

    Raises :class:`SkewError` (propagated from :func:`find_delta_strike`) if
    either the 25-delta call or put strike isn't found inside this expiry's
    own fitted domain - never falls back to the nearest available strike,
    since that would silently answer "the skew at whatever strike happened
    to be closest", not "the 25-delta skew".
    """
    call_K = find_delta_strike(expiry_smile, spot, r, q, "CE", target_abs_delta)
    put_K = find_delta_strike(expiry_smile, spot, r, q, "PE", target_abs_delta)
    call_iv = float(expiry_smile.smile(call_K))
    put_iv = float(expiry_smile.smile(put_K))
    return Skew25D(
        expiry=expiry_smile.expiry,
        call_strike=call_K,
        call_iv=call_iv,
        put_strike=put_K,
        put_iv=put_iv,
        skew=put_iv - call_iv,
    )
