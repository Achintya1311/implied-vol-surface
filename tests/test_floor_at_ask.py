"""P182: a floor break at the mid is split from one whose ask is below the floor too."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from volsurface.iv import floor_breaks_at_ask, solve_chain_ivs

SPOT, R = 100.0, 0.065


def _chain():
    # 30-day ITM calls at K=80: floor = 100 - 80*exp(-0.065*30/365) ~ 20.43.
    return pd.DataFrame(
        {
            "expiry": ["2026-11-30", "2026-11-30", "2026-11-30"],
            "strike": [80.0, 80.0, 90.0],
            "option_type": ["CE", "CE", "CE"],
            "bid": [19.0, 19.0, 10.6],
            "ask": [21.0, 20.0, 10.8],  # mid 20.0 < floor, ask above; mid 19.5, ask below; a normal quote
            "last_price": [20.0, 19.5, 10.7],
            "open_interest": [1000, 1000, 1000],
        }
    )


def test_only_the_quote_with_its_ask_below_the_floor_counts():
    quotes = _chain()
    table = solve_chain_ivs(quotes, SPOT, "2026-10-31T15:30:00", r=R, q=0.0)
    assert table["error"].fillna("").str.contains("below the no-arbitrage floor").tolist() == [True, True, False]
    assert floor_breaks_at_ask(table, quotes, SPOT, r=R, q=0.0).tolist() == [False, True, False]


@pytest.mark.parametrize(
    "underlying,stamp,mid,ask",
    [("RELIANCE", "2026-10-16T2319", 25, 7), ("NIFTY", "2026-10-16T2319", 196, 75),
     ("RELIANCE", "2026-10-25T2109", 25, 8), ("NIFTY", "2026-10-25T2109", 129, 15)],
)
def test_solve_prints_the_ask_count(capsys, underlying, stamp, mid, ask):
    from volsurface import solve

    assert solve.main(["--underlying", underlying, "--provider", "nse_raw", "--capture", stamp]) == 0
    out = capsys.readouterr().out
    assert f"{mid} below the no-arbitrage floor (" in out
    assert f"  {ask} of the {mid} with the ask below the floor too (executable; the rest break it at the mid only)" in out
