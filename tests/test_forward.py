"""R25: per-expiry forward from put-call parity. Offline: synthetic chains
priced from a known forward, plus the committed raw captures."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurface.forward import implied_dividend_yields, parity_forwards
from volsurface.iv import bs_price, solve_chain_ivs, time_to_expiry
from volsurface.surface import select_otm

S = 1211.2
R = 0.065
VAL = date(2026, 10, 16)
EXPIRY = "2026-11-23"


def _chain(q: float, sigma: float = 0.25, strikes=range(1100, 1330, 20), half_spread=0.25):
    T = time_to_expiry(VAL, EXPIRY)
    rows = []
    for K in strikes:
        for opt in ("CE", "PE"):
            p = bs_price(S, float(K), T, R, sigma, opt, q)
            rows.append(
                {"expiry": EXPIRY, "strike": float(K), "option_type": opt, "last_price": p,
                 "bid": p - half_spread, "ask": p + half_spread, "open_interest": 5000, "volume": 100}
            )
    return pd.DataFrame(rows)


def test_recovers_the_forward_a_chain_was_priced_from():
    q = 0.03
    T = time_to_expiry(VAL, EXPIRY)
    fwd = parity_forwards(_chain(q), S, VAL, R)
    assert fwd[EXPIRY].forward == pytest.approx(S * math.exp((R - q) * T), abs=1e-6)
    assert fwd[EXPIRY].pairs >= 3


def test_implied_dividend_yield_round_trips():
    q = 0.03
    fwd = parity_forwards(_chain(q), S, VAL, R)
    assert implied_dividend_yields(fwd, S, VAL, R)[EXPIRY] == pytest.approx(q, abs=1e-9)


def test_pairs_without_a_two_sided_quote_are_ignored():
    chain = _chain(0.03)
    near = chain["strike"] == 1200.0
    chain.loc[near & (chain["option_type"] == "PE"), ["bid", "ask"]] = 0.0
    fwd = parity_forwards(chain, S, VAL, R)
    assert 1200.0 not in fwd[EXPIRY].strikes


def test_expiry_with_no_usable_pair_is_left_out():
    chain = _chain(0.03)
    chain.loc[chain["option_type"] == "PE", ["bid", "ask"]] = 0.0
    assert parity_forwards(chain, S, VAL, R) == {}


def test_solving_with_the_parity_forward_removes_the_put_call_gap():
    # Priced with q = 3%; solved with the model's q = 0 the puts and calls at
    # one strike disagree, solved with the chain's own forward they agree.
    chain = _chain(0.03)
    fwd = parity_forwards(chain, S, VAL, R)
    qs = implied_dividend_yields(fwd, S, VAL, R)
    model = solve_chain_ivs(chain, S, "2026-10-16T15:40:00", r=R, q=0.0)
    parity = solve_chain_ivs(chain, S, "2026-10-16T15:40:00", r=R, q_by_expiry=qs)

    def gap(t):
        k = t[t["strike"] == 1200.0].set_index("option_type")["iv"]
        return abs(k["PE"] - k["CE"])

    assert gap(model) > 0.005
    assert gap(parity) < 1e-4


def test_select_otm_switches_at_the_given_forward():
    t = pd.DataFrame(
        {"expiry": [EXPIRY] * 4, "strike": [1200.0, 1200.0, 1220.0, 1220.0],
         "option_type": ["CE", "PE", "CE", "PE"], "iv": [0.2, 0.21, 0.2, 0.21]}
    )
    at_spot = select_otm(t, 1211.2)
    at_fwd = select_otm(t, 1211.2, forwards={EXPIRY: 1225.0})
    assert list(at_spot["option_type"]) == ["PE", "CE"]
    assert list(at_fwd["option_type"]) == ["PE", "PE"]


@pytest.mark.parametrize(
    "underlying,stamp",
    [("RELIANCE", "2026-10-16T2319"), ("NIFTY", "2026-10-16T2319")],
)
def test_parity_forward_is_below_spot_on_the_16_oct_captures(underlying, stamp):
    # The finding R25 rests on: on these captures the chain's own forward for
    # the nearest expiry sits below spot, while r = 6.5%, q = 0 puts it above.
    from volsurface.chain import get_provider
    from volsurface.iv import _valuation_date

    snap = get_provider("nse_raw", capture=stamp).fetch(underlying)
    val = _valuation_date(snap.timestamp)
    fwd = parity_forwards(snap.quotes, snap.spot, val, R)
    nearest = min(fwd)
    assert fwd[nearest].forward < snap.spot < snap.spot * math.exp(R * time_to_expiry(val, nearest))


@pytest.mark.parametrize(
    "underlying,stamp,forward,solved,floor,butterfly",
    [
        ("RELIANCE", "2026-10-16T2319", "model", 175, 25, 26),
        ("RELIANCE", "2026-10-16T2319", "parity", 190, 10, 15),
        ("RELIANCE", "2026-10-25T2109", "model", 180, 25, 52),
        ("RELIANCE", "2026-10-25T2109", "parity", 178, 27, 43),
        ("NIFTY", "2026-10-16T2319", "model", 508, 196, 149),
        ("NIFTY", "2026-10-16T2319", "parity", 578, 126, 143),
        ("NIFTY", "2026-10-25T2109", "model", 575, 129, 115),
        ("NIFTY", "2026-10-25T2109", "parity", 568, 136, 107),
    ],
)
def test_cli_counts_under_both_forwards(capsys, underlying, stamp, forward, solved, floor, butterfly):
    # The README's model-vs-parity table (R25). The parity forward lowers
    # every count on the 16 Oct batch but raises the floor breaks on 25 Oct;
    # both directions are pinned so neither can be quietly dropped.
    from volsurface import solve

    argv = ["--underlying", underlying, "--provider", "nse_raw", "--capture", stamp,
            "--surface", "--forward", forward]
    assert solve.main(argv) == 0
    out = capsys.readouterr().out
    assert f"\n{solved}/" in out
    assert f"  {floor} below the no-arbitrage floor" in out
    assert out.count("[butterfly]") == butterfly
    assert out.count("[calendar]") == 0


def test_greeks_are_not_combined_with_the_parity_forward(capsys):
    from volsurface import solve

    argv = ["--underlying", "RELIANCE", "--provider", "nse_raw", "--capture", "2026-10-16T2319",
            "--greeks", "--forward", "parity"]
    assert solve.main(argv) == 1
    assert "--forward parity" in capsys.readouterr().err
