"""Surface construction tests (Day 4). All offline: pure math plus the
committed RELIANCE fixture, no network.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurface.chain import get_provider
from volsurface.iv import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, bs_price, solve_chain_ivs
from volsurface.surface import (
    ArbitrageViolation,
    ExpirySmile,
    SurfaceError,
    build_smile,
    build_surface,
    check_butterfly,
    check_calendar,
    filter_liquid,
    relative_spread,
    select_otm,
)

SPOT = 1400.5
R = DEFAULT_RATE
Q = DEFAULT_DIVIDEND_YIELD


# ---------------------------------------------------------------------------
# relative_spread
# ---------------------------------------------------------------------------


def test_relative_spread_basic():
    assert relative_spread(pd.Series([9.0]), pd.Series([11.0])).iloc[0] == pytest.approx(0.2)


def test_relative_spread_zero_mid_is_infinite():
    assert relative_spread(pd.Series([0.0]), pd.Series([0.0])).iloc[0] == float("inf")


# ---------------------------------------------------------------------------
# filter_liquid / select_otm, against a small synthetic chain
# ---------------------------------------------------------------------------


def _synthetic_iv_table_and_quotes():
    """Three strikes: one liquid, one wide-spread, one low-open-interest."""
    quotes = pd.DataFrame(
        [
            {
                "expiry": "2026-10-29",
                "strike": 1300.0,
                "option_type": "PE",
                "open_interest": 5000,
                "bid": 9.5,
                "ask": 10.5,
            },
            {  # wide relative spread -> filtered out
                "expiry": "2026-10-29",
                "strike": 1275.0,
                "option_type": "PE",
                "open_interest": 5000,
                "bid": 0.05,
                "ask": 0.20,
            },
            {  # low open interest -> filtered out
                "expiry": "2026-10-29",
                "strike": 1500.0,
                "option_type": "CE",
                "open_interest": 10,
                "bid": 4.9,
                "ask": 5.1,
            },
        ]
    )
    iv_table = pd.DataFrame(
        [
            {"expiry": "2026-10-29", "strike": 1300.0, "option_type": "PE", "iv": 0.22, "error": None},
            {"expiry": "2026-10-29", "strike": 1275.0, "option_type": "PE", "iv": 0.30, "error": None},
            {"expiry": "2026-10-29", "strike": 1500.0, "option_type": "CE", "iv": 0.19, "error": None},
        ]
    )
    return iv_table, quotes


def test_filter_liquid_drops_wide_spread_and_low_oi():
    iv_table, quotes = _synthetic_iv_table_and_quotes()
    liquid = filter_liquid(iv_table, quotes, min_open_interest=1000, max_relative_spread=0.20)
    assert sorted(liquid["strike"].tolist()) == [1300.0]


def test_filter_liquid_drops_unsolved_rows():
    iv_table = pd.DataFrame(
        [{"expiry": "2026-10-29", "strike": 1300.0, "option_type": "PE", "iv": float("nan"), "error": "x"}]
    )
    quotes = pd.DataFrame(
        [{"expiry": "2026-10-29", "strike": 1300.0, "option_type": "PE", "open_interest": 5000, "bid": 9.5, "ask": 10.5}]
    )
    liquid = filter_liquid(iv_table, quotes)
    assert liquid.empty


def test_select_otm_picks_put_below_spot_call_at_or_above():
    liquid = pd.DataFrame(
        [
            {"expiry": "e1", "strike": 1300.0, "option_type": "PE", "iv": 0.25},
            {"expiry": "e1", "strike": 1300.0, "option_type": "CE", "iv": 0.24},  # ITM leg, should be dropped
            {"expiry": "e1", "strike": 1500.0, "option_type": "CE", "iv": 0.20},
        ]
    )
    otm = select_otm(liquid, spot=1400.0)
    assert list(zip(otm["strike"], otm["option_type"])) == [(1300.0, "PE"), (1500.0, "CE")]


def test_select_otm_empty_input_returns_empty():
    liquid = pd.DataFrame(columns=["expiry", "strike", "option_type", "iv"])
    otm = select_otm(liquid, spot=1400.0)
    assert otm.empty


# ---------------------------------------------------------------------------
# build_smile
# ---------------------------------------------------------------------------


def test_build_smile_too_few_points_raises():
    otm = pd.DataFrame({"strike": [1300.0, 1400.0], "iv": [0.25, 0.22]})
    with pytest.raises(SurfaceError):
        build_smile(otm)


def test_build_smile_interpolates_between_knots():
    otm = pd.DataFrame(
        {"strike": [1300.0, 1350.0, 1400.0, 1450.0], "iv": [0.26, 0.24, 0.22, 0.20]}
    )
    smile = build_smile(otm)
    # monotone decreasing input -> PCHIP should not overshoot outside [min, max]
    mid = smile(1375.0)
    assert 0.20 < mid < 0.24
    # exact knots round-trip
    assert smile(1300.0) == pytest.approx(0.26)
    assert smile(1450.0) == pytest.approx(0.20)


def test_build_smile_unsorted_strikes_still_works():
    otm = pd.DataFrame({"strike": [1400.0, 1300.0, 1450.0], "iv": [0.22, 0.26, 0.20]})
    smile = build_smile(otm)
    assert smile(1300.0) == pytest.approx(0.26)


# ---------------------------------------------------------------------------
# check_butterfly - synthetic cases
# ---------------------------------------------------------------------------


def _flat_smile(strikes, iv):
    return ExpirySmile(
        expiry="e1",
        T=30 / 365,
        strikes=np.array(strikes, dtype=float),
        smile=build_smile(pd.DataFrame({"strike": strikes, "iv": [iv] * len(strikes)})),
    )


def test_check_butterfly_flat_smile_is_arbitrage_free():
    es = _flat_smile([1300.0, 1350.0, 1400.0, 1450.0, 1500.0], iv=0.22)
    violations = check_butterfly(es, SPOT, R, Q)
    assert violations == []


def test_check_butterfly_catches_a_deliberately_broken_smile():
    # A sharp spike in IV at a single interior strike, steep enough to bend
    # the re-priced call curve concave at that point - a real butterfly
    # centred there would then have a negative price.
    strikes = [1200.0, 1300.0, 1400.0, 1500.0, 1600.0]
    ivs = [0.20, 0.20, 1.50, 0.20, 0.20]
    otm = pd.DataFrame({"strike": strikes, "iv": ivs})
    es = ExpirySmile(expiry="e1", T=30 / 365, strikes=np.array(strikes, dtype=float), smile=build_smile(otm))
    violations = check_butterfly(es, SPOT, R, Q)
    assert violations, "expected the spiked smile to trip the butterfly check"
    assert all(isinstance(v, ArbitrageViolation) and v.kind == "butterfly" for v in violations)


def test_check_butterfly_too_narrow_grid_returns_empty():
    # Two-point smile: build_smile itself would reject this (< MIN_SMILE_POINTS)
    # for a real pipeline, but check_butterfly should still degrade gracefully
    # rather than crash if ever handed one.
    strikes = np.array([1300.0, 1400.0])
    smile = PchipInterpolatorStub()
    es = ExpirySmile(expiry="e1", T=30 / 365, strikes=strikes, smile=smile)
    assert check_butterfly(es, SPOT, R, Q, grid_points=1) == []


class PchipInterpolatorStub:
    def __call__(self, x):
        return np.full_like(np.asarray(x, dtype=float), 0.2)


# ---------------------------------------------------------------------------
# check_calendar - synthetic cases
# ---------------------------------------------------------------------------


def test_check_calendar_increasing_total_variance_is_clean():
    near = ExpirySmile(
        expiry="near", T=5 / 365, strikes=np.array([1300.0, 1400.0, 1500.0]),
        smile=build_smile(pd.DataFrame({"strike": [1300.0, 1400.0, 1500.0], "iv": [0.24, 0.23, 0.22]})),
    )
    far = ExpirySmile(
        expiry="far", T=35 / 365, strikes=np.array([1300.0, 1400.0, 1500.0]),
        smile=build_smile(pd.DataFrame({"strike": [1300.0, 1400.0, 1500.0], "iv": [0.22, 0.21, 0.20]})),
    )
    assert check_calendar([near, far]) == []


def test_check_calendar_catches_decreasing_total_variance():
    # A high-IV near expiry and a much lower-IV far expiry, chosen so total
    # variance (iv**2 * T) actually falls despite T increasing - a real
    # inverted-and-collapsing term structure, not just an inverted IV one.
    near = ExpirySmile(
        expiry="near", T=5 / 365, strikes=np.array([1300.0, 1400.0, 1500.0]),
        smile=build_smile(pd.DataFrame({"strike": [1300.0, 1400.0, 1500.0], "iv": [0.85, 0.80, 0.75]})),
    )
    far = ExpirySmile(
        expiry="far", T=35 / 365, strikes=np.array([1300.0, 1400.0, 1500.0]),
        smile=build_smile(pd.DataFrame({"strike": [1300.0, 1400.0, 1500.0], "iv": [0.05, 0.05, 0.05]})),
    )
    violations = check_calendar([near, far])
    assert violations
    assert all(v.kind == "calendar" for v in violations)


def test_check_calendar_skips_strikes_outside_far_smiles_range():
    near = ExpirySmile(
        expiry="near", T=5 / 365, strikes=np.array([1200.0, 1300.0, 1400.0]),
        smile=build_smile(pd.DataFrame({"strike": [1200.0, 1300.0, 1400.0], "iv": [0.25, 0.24, 0.23]})),
    )
    far = ExpirySmile(
        expiry="far", T=35 / 365, strikes=np.array([1300.0, 1400.0, 1500.0]),
        smile=build_smile(pd.DataFrame({"strike": [1300.0, 1400.0, 1500.0], "iv": [0.22, 0.21, 0.20]})),
    )
    # strike 1200 is outside far's fitted range -> should be silently skipped,
    # not raise and not appear as a violation
    violations = check_calendar([near, far])
    assert violations == []


# ---------------------------------------------------------------------------
# build_surface - full pipeline against the committed RELIANCE fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def reliance_iv_and_quotes():
    snapshot = get_provider("fixture").fetch("RELIANCE")
    iv_table = solve_chain_ivs(snapshot.quotes, snapshot.spot, snapshot.timestamp, r=R, q=Q)
    return snapshot, iv_table


def test_build_surface_reliance_fixture_drops_illiquid_wings(reliance_iv_and_quotes):
    snapshot, iv_table = reliance_iv_and_quotes
    result = build_surface(
        iv_table, snapshot.quotes, snapshot.spot, date(2026, 9, 24), R, Q
    )
    by_expiry = {es.expiry: es for es in result.expiries}
    assert set(by_expiry) == {"2026-09-29", "2026-10-29"}

    near_strikes = by_expiry["2026-09-29"].strikes.tolist()
    far_strikes = by_expiry["2026-10-29"].strikes.tolist()
    # the two deep-ITM legs that never solved, and the thin/wide-spread wings,
    # do not survive filtering for the near expiry
    assert 1250.0 not in near_strikes
    assert 1550.0 not in near_strikes
    assert near_strikes == sorted(near_strikes)
    # far expiry keeps a wider core (its wings have tight spreads even at low OI)
    assert len(far_strikes) > len(near_strikes)

    assert result.dropped_counts["2026-09-29"]["kept"] < result.dropped_counts["2026-09-29"]["total"]


def test_build_surface_reliance_fixture_is_arbitrage_free(reliance_iv_and_quotes):
    """Documents an honest finding, not an assumption: this fixture's fitted
    smile has no butterfly or calendar violations. It was built from a
    single self-consistent Black-Scholes surface (see README Findings), so
    this demonstrates the checks run cleanly end to end - it is not
    evidence the checks would catch a *real* arbitrage in a live chain.
    """
    snapshot, iv_table = reliance_iv_and_quotes
    result = build_surface(
        iv_table, snapshot.quotes, snapshot.spot, date(2026, 9, 24), R, Q
    )
    assert result.violations == []


def test_build_surface_skips_expiry_with_too_few_liquid_strikes(reliance_iv_and_quotes):
    snapshot, iv_table = reliance_iv_and_quotes
    result = build_surface(
        iv_table,
        snapshot.quotes,
        snapshot.spot,
        date(2026, 9, 24),
        R,
        Q,
        min_open_interest=100_000,  # nothing in the fixture clears this
    )
    assert result.expiries == []
    assert result.violations == []
    assert all(c["kept"] == 0 for c in result.dropped_counts.values())
