"""25-delta skew tests (Day 7). All offline: pure math plus the committed
RELIANCE fixture, no network.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from volsurface.chain import get_provider
from volsurface.greeks import bs_delta
from volsurface.iv import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, solve_chain_ivs
from volsurface.skew import SkewError, compute_skew_25d, find_delta_strike
from volsurface.surface import ExpirySmile, build_surface

R = DEFAULT_RATE
Q = DEFAULT_DIVIDEND_YIELD
SPOT = 1400.5


def _flat_smile(strikes, iv, T, expiry="2026-10-29"):
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.full_like(strikes, iv)
    return ExpirySmile(
        expiry=expiry, T=T, strikes=strikes,
        smile=PchipInterpolator(strikes, ivs, extrapolate=False),
    )


@pytest.fixture()
def reliance_expiries():
    snapshot = get_provider("fixture").fetch("RELIANCE")
    iv_table = solve_chain_ivs(snapshot.quotes, snapshot.spot, snapshot.timestamp, r=R, q=Q)
    result = build_surface(iv_table, snapshot.quotes, snapshot.spot, date(2026, 9, 24), R, Q)
    return {es.expiry: es for es in result.expiries}


# ---------------------------------------------------------------------------
# find_delta_strike, against a synthetic flat-IV smile
# ---------------------------------------------------------------------------


def test_find_delta_strike_call_matches_target_delta():
    smile = _flat_smile(strikes=[1200, 1300, 1400, 1500, 1600], iv=0.25, T=30 / 365)
    K = find_delta_strike(smile, SPOT, R, Q, "CE", target_abs_delta=0.25)
    assert bs_delta(SPOT, K, smile.T, R, 0.25, "CE", Q) == pytest.approx(0.25, abs=1e-5)


def test_find_delta_strike_put_matches_target_delta():
    smile = _flat_smile(strikes=[1200, 1300, 1400, 1500, 1600], iv=0.25, T=30 / 365)
    K = find_delta_strike(smile, SPOT, R, Q, "PE", target_abs_delta=0.25)
    assert bs_delta(SPOT, K, smile.T, R, 0.25, "PE", Q) == pytest.approx(-0.25, abs=1e-5)


def test_find_delta_strike_call_put_bracket_spot():
    smile = _flat_smile(strikes=[1200, 1300, 1400, 1500, 1600], iv=0.25, T=30 / 365)
    call_K = find_delta_strike(smile, SPOT, R, Q, "CE")
    put_K = find_delta_strike(smile, SPOT, R, Q, "PE")
    assert put_K < SPOT < call_K  # 25-delta strikes sit OTM on either side of spot


def test_find_delta_strike_unbracketed_target_raises():
    # A narrow, near-the-money-only domain never reaches 5-delta.
    smile = _flat_smile(strikes=[1390, 1400, 1410], iv=0.25, T=30 / 365)
    with pytest.raises(SkewError):
        find_delta_strike(smile, SPOT, R, Q, "CE", target_abs_delta=0.05)


def test_find_delta_strike_rejects_unknown_option_type():
    smile = _flat_smile(strikes=[1200, 1300, 1400, 1500, 1600], iv=0.25, T=30 / 365)
    with pytest.raises(ValueError):
        find_delta_strike(smile, SPOT, R, Q, "XX")


# ---------------------------------------------------------------------------
# compute_skew_25d
# ---------------------------------------------------------------------------


def test_compute_skew_25d_zero_for_a_flat_smile():
    # No skew at all (flat IV across strike) should round-trip to ~0.
    smile = _flat_smile(strikes=[1200, 1300, 1400, 1500, 1600], iv=0.25, T=30 / 365)
    result = compute_skew_25d(smile, SPOT, R, Q)
    assert result.skew == pytest.approx(0.0, abs=1e-6)
    assert result.put_strike < SPOT < result.call_strike


def test_compute_skew_25d_positive_for_downward_skew():
    # IV rising as strike falls (the fixture's own documented skew shape):
    # 25-delta put should sit at a strike with higher IV than the 25-delta call.
    strikes = [1200, 1300, 1400, 1500, 1600]
    ivs = [0.34, 0.29, 0.25, 0.22, 0.20]
    smile = ExpirySmile(
        expiry="2026-10-29", T=30 / 365,
        strikes=np.array(strikes, dtype=float),
        smile=PchipInterpolator(strikes, ivs, extrapolate=False),
    )
    result = compute_skew_25d(smile, SPOT, R, Q)
    assert result.skew > 0
    assert result.put_iv > result.call_iv


def test_compute_skew_25d_against_real_fixture_far_expiry(reliance_expiries):
    # The far (30d-ish) expiry's strike range [1275, 1525] comfortably
    # brackets both 25-delta strikes for this fixture (verified by hand:
    # call delta at K=1475 is 0.25, put delta at K=1325-1350 is -0.25-ish).
    smile = reliance_expiries["2026-10-29"]
    result = compute_skew_25d(smile, 1400.5, R, Q)
    assert 1275.0 < result.put_strike < result.call_strike < 1525.0
    # Matches the fixture's documented downward equity skew (Day 1 Findings).
    assert result.skew > 0


def test_compute_skew_25d_raises_when_domain_too_narrow():
    smile = _flat_smile(strikes=[1390, 1400, 1410], iv=0.25, T=30 / 365)
    with pytest.raises(SkewError):
        compute_skew_25d(smile, SPOT, R, Q)
