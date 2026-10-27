"""Per-expiry forward implied by put-call parity (R25).

The solver's default prices every contract off spot with one assumed
risk-free rate and dividend yield (r = 6.5%, q = 0), which fixes the forward
at ``S * exp((r - q) * T)``. The chain itself says where the forward is: for
European options, ``C - P = exp(-r*T) * (F - K)`` at every strike, so each
strike quoted on both sides gives ``F = K + exp(r*T) * (C_mid - P_mid)``.

:func:`parity_forwards` takes the median of that estimate over the
:data:`PARITY_STRIKES` strikes closest to spot that have a two-sided quote
on both legs (the most liquid part of the chain; wing quotes are too wide
to say anything). The forward is then expressed as a per-expiry implied
dividend yield, ``q = r - ln(F / S) / T``, so the existing Black-Scholes code
prices off it unchanged. NSE index and stock options are European-style,
so parity holds in form; the estimate is only as good as the mids near
spot, which the README's Limitations state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from volsurface.iv import time_to_expiry

PARITY_STRIKES = 5


@dataclass
class ParityForward:
    expiry: str
    forward: float
    pairs: int  # strikes the median was taken over
    strikes: list[float] = field(default_factory=list)


def parity_forwards(
    quotes: pd.DataFrame,
    spot: float,
    valuation_date: date,
    r: float,
    n_strikes: int = PARITY_STRIKES,
) -> dict[str, ParityForward]:
    """Forward per expiry from put-call parity on two-sided quotes.

    An expiry with no strike quoted two-sided on both legs, or already
    expired, is left out of the result; callers fall back to the model
    forward for it and say so.
    """
    two_sided = quotes[(quotes["bid"].fillna(0) > 0) & (quotes["ask"].fillna(0) > 0)].copy()
    two_sided["mid"] = (two_sided["bid"] + two_sided["ask"]) / 2.0
    out: dict[str, ParityForward] = {}
    for expiry, group in two_sided.groupby("expiry"):
        T = time_to_expiry(valuation_date, expiry)
        if T <= 0:
            continue
        legs = group.pivot_table(index="strike", columns="option_type", values="mid", aggfunc="first")
        if "CE" not in legs or "PE" not in legs:
            continue
        legs = legs.dropna(subset=["CE", "PE"])
        if legs.empty:
            continue
        near = legs.loc[sorted(legs.index, key=lambda k: abs(k - spot))[:n_strikes]]
        estimates = near.index.to_numpy(dtype=float) + math.exp(r * T) * (near["CE"] - near["PE"]).to_numpy()
        out[expiry] = ParityForward(
            expiry=expiry,
            forward=float(np.median(estimates)),
            pairs=len(near),
            strikes=sorted(float(k) for k in near.index),
        )
    return out


def implied_dividend_yields(
    forwards: dict[str, ParityForward], spot: float, valuation_date: date, r: float
) -> dict[str, float]:
    """``q`` per expiry such that ``spot * exp((r - q) * T)`` equals the parity
    forward, for the same ``r`` the pricer is given."""
    return {
        expiry: r - math.log(f.forward / spot) / time_to_expiry(valuation_date, expiry)
        for expiry, f in forwards.items()
    }
