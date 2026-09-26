"""Greeks tests (Day 3). All offline: pure math, no network."""

from __future__ import annotations

import math

import pytest

from volsurface.greeks import (
    Greeks,
    bs_delta,
    bs_gamma,
    bs_greeks,
    bs_theta,
    compute_chain_greeks,
    fd_delta,
    fd_gamma,
    fd_greeks,
    fd_theta,
    fd_vega,
)
from volsurface.iv import bs_price, bs_vega, solve_chain_ivs

S = 1400.5
R = 0.065

STRIKES = [1250.0, 1400.0, 1550.0]
SIGMAS = [0.15, 0.3, 0.6]
T_30D = 30 / 365


class TestAnalyticIdentities:
    """Properties that must hold from the formulas alone, independent of the FD check."""

    @pytest.mark.parametrize("K", STRIKES)
    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_put_call_delta_parity(self, K, sigma):
        # delta_call - delta_put = e^{-qT} (q=0 here, so it's exactly 1)
        call_delta = bs_delta(S, K, T_30D, R, sigma, "CE")
        put_delta = bs_delta(S, K, T_30D, R, sigma, "PE")
        assert call_delta - put_delta == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("K", STRIKES)
    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_gamma_identical_for_call_and_put(self, K, sigma):
        assert bs_gamma(S, K, T_30D, R, sigma) > 0
        # gamma has no option_type parameter - it is the same curve for both,
        # so this is really just checking it's well-defined and positive.

    def test_call_delta_bounded_zero_one(self):
        for K in [100.0, 1400.0, 5000.0]:
            d = bs_delta(S, K, T_30D, R, 0.3, "CE")
            assert 0.0 <= d <= 1.0

    def test_put_delta_bounded_minus_one_zero(self):
        for K in [100.0, 1400.0, 5000.0]:
            d = bs_delta(S, K, T_30D, R, 0.3, "PE")
            assert -1.0 <= d <= 0.0

    def test_atm_call_theta_is_negative(self):
        # Standard result: an at-the-money option (no dividends) loses value
        # as time passes - the classic "time decay" desks quote as negative.
        theta = bs_theta(S, 1400.0, T_30D, R, 0.3, "CE")
        assert theta < 0

    def test_atm_put_theta_is_negative(self):
        theta = bs_theta(S, 1400.0, T_30D, R, 0.3, "PE")
        assert theta < 0

    def test_deep_itm_call_delta_approaches_one(self):
        d = bs_delta(S, 100.0, T_30D, R, 0.3, "CE")
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_deep_otm_call_delta_approaches_zero(self):
        d = bs_delta(S, 5000.0, 0.1, R, 0.3, "CE")
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_rejects_non_positive_sigma_or_t(self):
        with pytest.raises(ValueError):
            bs_delta(S, 1400.0, T_30D, R, 0.0, "CE")
        with pytest.raises(ValueError):
            bs_gamma(S, 1400.0, 0.0, R, 0.3)

    def test_rejects_unknown_option_type(self):
        with pytest.raises(ValueError, match="unknown option_type"):
            bs_delta(S, 1400.0, T_30D, R, 0.3, "XX")
        with pytest.raises(ValueError, match="unknown option_type"):
            bs_theta(S, 1400.0, T_30D, R, 0.3, "XX")


class TestFiniteDifferenceCrossCheck:
    """The Day-3 correctness gate: analytic Greeks agree with finite differences."""

    # Absolute, not relative, tolerances throughout: delta/gamma/vega all go
    # to genuinely near-zero deep ITM/OTM (the same territory Day 2's solver
    # found numerically awkward), where a tiny absolute finite-difference
    # error is a huge *relative* one despite the cross-check working fine.
    # Thresholds below are ~10x the largest absolute diff actually observed
    # across this grid plus the fixture's own 5-day-to-expiry legs.

    @pytest.mark.parametrize("K", STRIKES)
    @pytest.mark.parametrize("option_type", ["CE", "PE"])
    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_delta_matches_finite_difference(self, K, option_type, sigma):
        analytic = bs_delta(S, K, T_30D, R, sigma, option_type)
        approx = fd_delta(S, K, T_30D, R, sigma, option_type)
        assert analytic == pytest.approx(approx, abs=1e-5)

    @pytest.mark.parametrize("K", STRIKES)
    @pytest.mark.parametrize("option_type", ["CE", "PE"])
    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_gamma_matches_finite_difference(self, K, option_type, sigma):
        analytic = bs_gamma(S, K, T_30D, R, sigma)
        approx = fd_gamma(S, K, T_30D, R, sigma, option_type)
        assert analytic == pytest.approx(approx, abs=1e-4)

    @pytest.mark.parametrize("K", STRIKES)
    @pytest.mark.parametrize("option_type", ["CE", "PE"])
    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_vega_matches_finite_difference(self, K, option_type, sigma):
        analytic = bs_vega(S, K, T_30D, R, sigma)
        approx = fd_vega(S, K, T_30D, R, sigma, option_type)
        assert analytic == pytest.approx(approx, abs=1e-4)

    @pytest.mark.parametrize("K", STRIKES)
    @pytest.mark.parametrize("option_type", ["CE", "PE"])
    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_theta_matches_finite_difference(self, K, option_type, sigma):
        analytic = bs_theta(S, K, T_30D, R, sigma, option_type)
        approx = fd_theta(S, K, T_30D, R, sigma, option_type)
        assert analytic == pytest.approx(approx, abs=1e-6)

    def test_short_dated_deep_itm_still_matches(self):
        # Mirrors the fixture's tightest case (5 days to expiry, 150 points
        # ITM/OTM) - the smallest absolute T the theta bump ever has to work
        # with.
        T = 5 / 365
        greeks = bs_greeks(S, 1250.0, T, R, 0.3, "PE")
        approx = fd_greeks(S, 1250.0, T, R, 0.3, "PE")
        assert greeks.delta == pytest.approx(approx.delta, abs=1e-5)
        assert greeks.gamma == pytest.approx(approx.gamma, abs=1e-4)
        assert greeks.vega == pytest.approx(approx.vega, abs=1e-4)
        assert greeks.theta == pytest.approx(approx.theta, abs=1e-6)


class TestComputeChainGreeks:
    def test_full_fixture_chain_cross_checks_within_tolerance(self):
        from volsurface.chain import FixtureProvider

        snap = FixtureProvider().fetch("RELIANCE")
        iv_table = solve_chain_ivs(snap.quotes, snap.spot, snap.timestamp)
        table = compute_chain_greeks(iv_table, snap.spot)

        solved = table["error"].isna()
        assert solved.sum() == len(table) - 2  # same two cent-rounding non-solves as Day 2

        solved_rows = table.loc[solved]
        assert not solved_rows["delta"].isna().any()
        assert not solved_rows["gamma"].isna().any()
        assert not solved_rows["vega"].isna().any()
        assert not solved_rows["theta"].isna().any()
        assert (solved_rows["fd_max_abs_diff"] < 1e-3).all()

    def test_unsolved_rows_get_nan_greeks_not_a_fabricated_number(self):
        from volsurface.chain import FixtureProvider

        snap = FixtureProvider().fetch("RELIANCE")
        iv_table = solve_chain_ivs(snap.quotes, snap.spot, snap.timestamp)
        table = compute_chain_greeks(iv_table, snap.spot)

        unsolved = table[~table["error"].isna()]
        assert len(unsolved) == 2
        assert unsolved["delta"].isna().all()
        assert unsolved["fd_max_abs_diff"].isna().all()
