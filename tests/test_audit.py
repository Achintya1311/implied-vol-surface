"""ml-pipeline-audit tests (Day 8). All offline: pure math plus the
committed RELIANCE fixture, no network.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from volsurface.audit import (
    AuditError,
    audit_arbitrage_checks_fire_on_real_fixture,
    audit_calendar_check_fires_on_real_fixture,
    audit_greeks,
    audit_round_trip,
    main,
)
from volsurface.chain import get_provider
from volsurface.iv import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, solve_chain_ivs

R = DEFAULT_RATE
Q = DEFAULT_DIVIDEND_YIELD
VALUATION_DATE = date(2026, 9, 24)


@pytest.fixture()
def reliance_iv_and_quotes():
    snapshot = get_provider("fixture").fetch("RELIANCE")
    iv_table = solve_chain_ivs(snapshot.quotes, snapshot.spot, snapshot.timestamp, r=R, q=Q)
    return snapshot, iv_table


# ---------------------------------------------------------------------------
# audit_round_trip
# ---------------------------------------------------------------------------


def test_audit_round_trip_passes_on_real_fixture(reliance_iv_and_quotes):
    snapshot, iv_table = reliance_iv_and_quotes
    n_checked, max_err = audit_round_trip(iv_table, snapshot.spot, R, Q)
    assert n_checked == int((iv_table["error"].isna() & iv_table["iv"].notna()).sum())
    assert n_checked > 0
    assert max_err < 1e-6  # far tighter than the audit's own tolerance - a real round trip, not a fluke


def test_audit_round_trip_raises_on_corrupted_mid_price():
    # A solved row whose recorded mid_price does not match what that iv
    # actually reprices to - the audit's job is to catch exactly this, not
    # just to trust the table.
    table = pd.DataFrame(
        [{"strike": 1400.0, "T": 0.25, "iv": 0.30, "option_type": "CE", "error": None, "mid_price": 9999.0}]
    )
    with pytest.raises(AuditError, match="round-trip error"):
        audit_round_trip(table, 1400.5, R, Q)


def test_audit_round_trip_raises_when_nothing_solved():
    table = pd.DataFrame(
        [{"strike": 1400.0, "T": 0.25, "iv": float("nan"), "option_type": "CE", "error": "no solution", "mid_price": 10.0}]
    )
    with pytest.raises(AuditError, match="no solved rows"):
        audit_round_trip(table, 1400.5, R, Q)


# ---------------------------------------------------------------------------
# audit_greeks
# ---------------------------------------------------------------------------


def test_audit_greeks_passes_on_real_fixture(reliance_iv_and_quotes):
    snapshot, iv_table = reliance_iv_and_quotes
    n_checked, max_diff = audit_greeks(iv_table, snapshot.spot, R, Q)
    assert n_checked > 0
    assert max_diff < 1e-4  # matches tests/test_greeks.py's own tolerance for this strike/vol range


def test_audit_greeks_raises_on_oversized_gap(monkeypatch):
    import volsurface.audit as audit_mod

    def fake_compute_chain_greeks(iv_table, spot, r, q):
        return pd.DataFrame({"fd_max_abs_diff": [1.0]})

    monkeypatch.setattr(audit_mod, "compute_chain_greeks", fake_compute_chain_greeks)
    with pytest.raises(AuditError, match="Greek gap"):
        audit_mod.audit_greeks(pd.DataFrame(), 1400.5, R, Q)


def test_audit_greeks_raises_when_nothing_checked(monkeypatch):
    import volsurface.audit as audit_mod

    def fake_compute_chain_greeks(iv_table, spot, r, q):
        return pd.DataFrame({"fd_max_abs_diff": [float("nan")]})

    monkeypatch.setattr(audit_mod, "compute_chain_greeks", fake_compute_chain_greeks)
    with pytest.raises(AuditError, match="no rows got Greeks"):
        audit_mod.audit_greeks(pd.DataFrame(), 1400.5, R, Q)


# ---------------------------------------------------------------------------
# audit_arbitrage_checks_fire_on_real_fixture - the negative control that
# closes the README's Day 4 "only ever run against an arbitrage-free chain"
# gap.
# ---------------------------------------------------------------------------


def test_negative_control_catches_a_spike_planted_in_the_real_fixture(reliance_iv_and_quotes):
    snapshot, iv_table = reliance_iv_and_quotes
    n_violations = audit_arbitrage_checks_fire_on_real_fixture(
        iv_table, snapshot.quotes, snapshot.spot, VALUATION_DATE, R, Q
    )
    assert n_violations > 0


def test_negative_control_raises_if_the_check_never_fires(monkeypatch, reliance_iv_and_quotes):
    # Proves the audit itself is a real check, not one that always passes:
    # if check_butterfly stayed silent even on data known to be broken, the
    # audit must say so rather than report success.
    import volsurface.audit as audit_mod

    snapshot, iv_table = reliance_iv_and_quotes
    monkeypatch.setattr(audit_mod, "check_butterfly", lambda *a, **k: [])
    with pytest.raises(AuditError, match="still found no violation"):
        audit_mod.audit_arbitrage_checks_fire_on_real_fixture(
            iv_table, snapshot.quotes, snapshot.spot, VALUATION_DATE, R, Q
        )


def test_negative_control_catches_a_shrunk_far_expiry_in_the_real_fixture(reliance_iv_and_quotes):
    snapshot, iv_table = reliance_iv_and_quotes
    n_violations = audit_calendar_check_fires_on_real_fixture(
        iv_table, snapshot.quotes, snapshot.spot, VALUATION_DATE, R, Q
    )
    assert n_violations > 0


def test_calendar_negative_control_raises_if_the_check_never_fires(monkeypatch, reliance_iv_and_quotes):
    import volsurface.audit as audit_mod

    snapshot, iv_table = reliance_iv_and_quotes
    monkeypatch.setattr(audit_mod, "check_calendar", lambda *a, **k: [])
    with pytest.raises(AuditError, match="still found no violation"):
        audit_mod.audit_calendar_check_fires_on_real_fixture(
            iv_table, snapshot.quotes, snapshot.spot, VALUATION_DATE, R, Q
        )


def test_inject_butterfly_violation_raises_on_too_few_strikes():
    from volsurface.audit import _inject_butterfly_violation

    otm = pd.DataFrame({"strike": [1300.0, 1400.0], "iv": [0.2, 0.22]})
    with pytest.raises(AuditError, match="fewer than 3"):
        _inject_butterfly_violation(otm)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_passes_end_to_end_against_real_fixture(capsys):
    rc = main(["--underlying", "RELIANCE", "--provider", "fixture"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "[ok] round-trip" in captured.out
    assert "[ok] greeks" in captured.out
    assert "[ok] negative control (butterfly)" in captured.out
    assert "[ok] negative control (calendar)" in captured.out
    assert "all checks passed" in captured.out


def test_cli_unknown_underlying_exits_nonzero(capsys):
    rc = main(["--underlying", "NOPE", "--provider", "fixture"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
