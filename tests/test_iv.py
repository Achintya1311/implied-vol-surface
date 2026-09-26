"""IV solver tests. All offline: pure math, no network."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from volsurface.iv import (
    NoSolution,
    bs_price,
    bs_vega,
    solve_chain_ivs,
    solve_iv,
    time_to_expiry,
)

S = 1400.5
R = 0.065


class TestBsPrice:
    def test_call_put_parity(self):
        K, T, sigma = 1400.0, 0.25, 0.3
        call = bs_price(S, K, T, R, sigma, "CE")
        put = bs_price(S, K, T, R, sigma, "PE")
        # C - P = S - K e^{-rT} (no dividend)
        assert call - put == pytest.approx(S - K * math.exp(-R * T), abs=1e-8)

    def test_rejects_non_positive_sigma_or_t(self):
        with pytest.raises(ValueError):
            bs_price(S, 1400.0, 0.25, R, 0.0, "CE")
        with pytest.raises(ValueError):
            bs_price(S, 1400.0, 0.0, R, 0.3, "CE")

    def test_rejects_unknown_option_type(self):
        with pytest.raises(ValueError, match="unknown option_type"):
            bs_price(S, 1400.0, 0.25, R, 0.3, "XX")

    def test_deep_itm_call_approaches_intrinsic_forward_value(self):
        K, T = 100.0, 0.25
        price = bs_price(S, K, T, R, 0.3, "CE")
        intrinsic_like = S - K * math.exp(-R * T)
        assert price == pytest.approx(intrinsic_like, rel=1e-6)

    def test_deep_otm_call_is_near_zero(self):
        price = bs_price(S, 5000.0, 0.1, R, 0.3, "CE")
        assert 0 < price < 0.01


class TestSolveIvRoundTrip:
    @pytest.mark.parametrize("K", [1250.0, 1400.0, 1550.0])
    @pytest.mark.parametrize("option_type", ["CE", "PE"])
    @pytest.mark.parametrize("true_sigma", [0.15, 0.3, 0.6])
    def test_price_to_iv_to_price_round_trips(self, K, option_type, true_sigma):
        T = 30 / 365
        price = bs_price(S, K, T, R, true_sigma, option_type)
        result = solve_iv(price, S, K, T, R, option_type)
        assert result.iv == pytest.approx(true_sigma, abs=1e-4)
        recovered_price = bs_price(S, K, T, R, result.iv, option_type)
        assert recovered_price == pytest.approx(price, abs=1e-6)

    def test_deep_otm_short_dated_still_round_trips(self):
        # Mirrors the fixture: a 5-day put struck 150 points OTM, priced at
        # the minimum tick. Vega is small there but not yet numerically
        # flat, so Newton alone happens to hold - the round trip is the
        # thing that matters, whichever method gets there.
        T = 5 / 365
        result = solve_iv(0.05, S, 1250.0, T, R, "PE")
        assert result.converged
        assert bs_price(S, 1250.0, T, R, result.iv, "PE") == pytest.approx(0.05, abs=1e-6)

    def test_deep_otm_short_dated_call_still_round_trips(self):
        T = 5 / 365
        result = solve_iv(0.05, S, 1550.0, T, R, "CE")
        assert result.converged
        assert bs_price(S, 1550.0, T, R, result.iv, "CE") == pytest.approx(0.05, abs=1e-6)

    def test_newton_used_near_the_money(self):
        T = 30 / 365
        price = bs_price(S, 1400.0, T, R, 0.25, "CE")
        result = solve_iv(price, S, 1400.0, T, R, "CE")
        assert result.method == "newton"

    def test_brent_fallback_recovers_from_a_bad_starting_guess(self):
        # A poor sigma0 (near the upper bound, far from the true 15% vol)
        # makes the first Newton step overshoot out of [BRENT_LO, BRENT_HI]
        # immediately - this is what forces the fallback path, and it must
        # still land on the same answer Newton would have found from a
        # sane starting point.
        T = 30 / 365
        price = bs_price(S, 1400.0, T, R, 0.15, "CE")
        result = solve_iv(price, S, 1400.0, T, R, "CE", sigma0=4.9)
        assert result.method == "brent"
        assert result.converged
        assert result.iv == pytest.approx(0.15, abs=1e-6)


class TestDeepMoneynessNumericalFloor:
    """Deep ITM/OTM options priced far from the money underflow.

    S*N(d1) and K*e^{-rT}*N(d2) are both O(S) or O(K) while their
    difference (the option's extrinsic value) can be O(1e-6) or smaller -
    classic catastrophic cancellation. Below a certain moneyness/vol/tenor
    combination, the theoretical price is indistinguishable from its own
    no-arbitrage floor in double precision: a bracket search can't even find
    a sign change, because *every* candidate sigma prices to the same
    floating-point floor. Reporting a specific IV there would be a
    confident-looking guess, not a solution - solve_iv refuses instead.
    """

    def test_deep_itm_low_vol_has_no_recoverable_solution(self):
        T = 30 / 365
        price = bs_price(S, 1000.0, T, R, 0.15, "CE")  # 40% ITM, short-dated, low vol
        with pytest.raises(NoSolution, match="double precision"):
            solve_iv(price, S, 1000.0, T, R, "CE")

    def test_deep_otm_low_vol_has_no_recoverable_solution(self):
        T = 30 / 365
        price = bs_price(S, 2500.0, T, R, 0.3, "CE")  # 78% OTM, short-dated
        with pytest.raises(NoSolution, match="double precision"):
            solve_iv(price, S, 2500.0, T, R, "CE")


class TestSolveIvNoSolution:
    def test_expired_contract_rejected(self):
        with pytest.raises(NoSolution, match="non-positive time to expiry"):
            solve_iv(10.0, S, 1400.0, 0.0, R, "CE")

    def test_negative_time_rejected(self):
        with pytest.raises(NoSolution, match="non-positive time to expiry"):
            solve_iv(10.0, S, 1400.0, -0.01, R, "CE")

    def test_price_below_call_floor_rejected(self):
        T = 0.25
        floor = S - 1000.0 * math.exp(-R * T)
        with pytest.raises(NoSolution, match="below the no-arbitrage floor"):
            solve_iv(floor - 50.0, S, 1000.0, T, R, "CE")

    def test_price_above_call_ceiling_rejected(self):
        T = 0.25
        with pytest.raises(NoSolution, match="above the no-arbitrage ceiling"):
            solve_iv(S + 10.0, S, 1400.0, T, R, "CE")

    def test_price_above_put_ceiling_rejected(self):
        T = 0.25
        K = 1400.0
        ceiling = K * math.exp(-R * T)
        with pytest.raises(NoSolution, match="above the no-arbitrage ceiling"):
            solve_iv(ceiling + 10.0, S, K, T, R, "PE")

    def test_non_positive_strike_rejected(self):
        with pytest.raises(NoSolution, match="non-positive spot or strike"):
            solve_iv(10.0, S, 0.0, 0.25, R, "CE")


class TestBsVega:
    def test_positive_and_symmetric_atm(self):
        T = 0.25
        v_call = bs_vega(S, 1400.0, T, R, 0.3)
        assert v_call > 0
        # vega is identical for a call and put at the same strike/T/sigma
        assert v_call == bs_vega(S, 1400.0, T, R, 0.3)

    def test_flat_far_out_of_the_money(self):
        T = 5 / 365
        assert bs_vega(S, 5000.0, T, R, 0.3) < 1e-6


class TestTimeToExpiry:
    def test_act_365(self):
        from datetime import date

        assert time_to_expiry(date(2026, 9, 24), "2026-09-29") == pytest.approx(5 / 365)

    def test_past_expiry_is_negative(self):
        from datetime import date

        assert time_to_expiry(date(2026, 10, 1), "2026-09-29") < 0


class TestSolveChainIvs:
    def _quotes(self, **overrides):
        row = {
            "expiry": "2026-10-29",
            "strike": 1400.0,
            "option_type": "CE",
            "last_price": None,
            "bid": None,
            "ask": None,
            "open_interest": 100,
            "volume": 50,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_solves_from_bid_ask_mid(self):
        T = 30 / 365
        true_price = bs_price(S, 1400.0, T, R, 0.28, "CE")
        quotes = self._quotes(bid=true_price - 0.5, ask=true_price + 0.5)
        table = solve_chain_ivs(quotes, S, "2026-09-29T15:30:00")
        assert table.loc[0, "error"] is None
        assert table.loc[0, "iv"] == pytest.approx(0.28, abs=1e-3)

    def test_falls_back_to_last_price_when_no_bid_ask(self):
        T = 30 / 365
        true_price = bs_price(S, 1400.0, T, R, 0.28, "CE")
        quotes = self._quotes(last_price=true_price)
        table = solve_chain_ivs(quotes, S, "2026-09-29T15:30:00")
        assert table.loc[0, "iv"] == pytest.approx(0.28, abs=1e-3)

    def test_no_price_at_all_is_an_explicit_error_not_a_bare_nan(self):
        quotes = self._quotes()
        table = solve_chain_ivs(quotes, S, "2026-09-29T15:30:00")
        assert math.isnan(table.loc[0, "iv"])
        assert "no bid/ask/last_price" in table.loc[0, "error"]

    def test_unsolvable_row_reports_reason_not_bare_nan(self):
        quotes = self._quotes(bid=S + 10.0, ask=S + 10.0)  # violates the call ceiling
        table = solve_chain_ivs(quotes, S, "2026-09-29T15:30:00")
        assert math.isnan(table.loc[0, "iv"])
        assert "no-arbitrage ceiling" in table.loc[0, "error"]

    def test_full_fixture_chain_mostly_solves(self):
        # Every fixture row solves except two deepest-ITM near-expiry legs
        # (1250 CE and 1550 PE, both expiring 2026-09-29): their bid/ask mid,
        # rounded to the cent, sits a fraction of a paisa below the
        # theoretical no-arbitrage floor implied by r=6.5% - a real artifact
        # of quoting to two decimals at the deepest strikes, not a solver
        # bug. Both are reported with an explicit reason, not a bare NaN.
        from volsurface.chain import FixtureProvider

        snap = FixtureProvider().fetch("RELIANCE")
        table = solve_chain_ivs(snap.quotes, snap.spot, snap.timestamp)
        solved = table["error"].isna()
        assert solved.sum() == len(table) - 2
        assert (table.loc[solved, "iv"] > 0).all()
        unsolved = table.loc[~solved]
        assert set(zip(unsolved["strike"], unsolved["option_type"])) == {(1250.0, "CE"), (1550.0, "PE")}
        assert unsolved["error"].str.contains("no-arbitrage floor").all()
