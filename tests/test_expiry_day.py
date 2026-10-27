"""P131: contracts at or past expiry on the quote date get their own line in
``solve``, not "another reason"."""

import pandas as pd

from volsurface import solve
from volsurface.chain import CHAIN_COLUMNS
from volsurface.iv import AT_OR_PAST_EXPIRY, solve_chain_ivs


def _chain(expiry):
    rows = [
        {"expiry": expiry, "strike": k, "option_type": t, "last_price": 0.0, "bid": b, "ask": b + 1.0,
         "open_interest": 100, "volume": 10}
        for k, t, b in ((1000.0, "CE", 120.0), (1000.0, "PE", 8.0))
    ]
    return pd.DataFrame(rows)[CHAIN_COLUMNS]


def test_valuation_date_on_the_expiry_gives_the_at_or_past_expiry_reason():
    table = solve_chain_ivs(_chain("2026-10-27"), 1100.0, "2026-10-27T14:25:17")
    assert table["error"].str.startswith(AT_OR_PAST_EXPIRY).all()


def test_a_later_expiry_still_solves():
    table = solve_chain_ivs(_chain("2026-11-27"), 1100.0, "2026-10-27T14:25:17")
    assert table["error"].isna().all()


def test_solve_prints_the_expiry_day_line_for_the_27_oct_batch(capsys):
    assert solve.main(["--underlying", "RELIANCE", "--provider", "nse_raw", "--capture", "2026-10-27T1430"]) == 0
    out = capsys.readouterr().out
    assert "96 at or past expiry on the quote date (T <= 0 in whole days)" in out
    assert "for another reason" not in out


def test_solve_nifty_27_oct_batch_splits_the_expiry_day_contracts(capsys):
    assert solve.main(["--underlying", "NIFTY", "--provider", "nse_raw", "--capture", "2026-10-27T1430"]) == 0
    out = capsys.readouterr().out
    assert "288 at or past expiry on the quote date" in out


def test_batches_without_expiry_day_contracts_print_no_such_line(capsys):
    assert solve.main(["--underlying", "RELIANCE", "--provider", "nse_raw", "--capture", "2026-10-16T2319"]) == 0
    assert "at or past expiry" not in capsys.readouterr().out
