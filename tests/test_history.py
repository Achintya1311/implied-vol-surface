"""Daily history job tests (Day 6). All offline: pure math, the committed
RELIANCE fixture, and tmp_path for file writes - no network.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from volsurface.chain import get_provider
from volsurface.iv import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, solve_chain_ivs
from volsurface.history import (
    DailySummary,
    HistoryError,
    append_history,
    compute_daily_summary,
    load_history,
    main,
)
from volsurface.surface import SurfaceResult, build_surface

R = DEFAULT_RATE
Q = DEFAULT_DIVIDEND_YIELD


@pytest.fixture()
def reliance_result():
    snapshot = get_provider("fixture").fetch("RELIANCE")
    iv_table = solve_chain_ivs(snapshot.quotes, snapshot.spot, snapshot.timestamp, r=R, q=Q)
    result = build_surface(iv_table, snapshot.quotes, snapshot.spot, date(2026, 9, 24), R, Q)
    return snapshot, result


# ---------------------------------------------------------------------------
# compute_daily_summary
# ---------------------------------------------------------------------------


def test_compute_daily_summary_against_real_fixture(reliance_result):
    snapshot, result = reliance_result
    summary = compute_daily_summary(
        result, snapshot.spot, "reliance", snapshot.source,
        data_date=date(2026, 9, 24), run_date=date(2026, 9, 27),
    )

    assert summary.underlying == "RELIANCE"  # normalised to upper case
    assert summary.run_date == "2026-09-27"
    assert summary.data_date == "2026-09-24"
    assert summary.source == "hand-assembled"
    assert summary.n_expiries == 2  # both fitted per the Day 4/5 fixture findings
    assert summary.atm_iv_near == pytest.approx(0.235, abs=0.01)
    assert summary.atm_iv_30d == pytest.approx(0.215, abs=0.01)
    # README's Day 5 finding: the fixture's ATM term structure is inverted.
    assert summary.term_slope is not None
    assert summary.term_slope < 0
    assert summary.n_violations == 0


def test_compute_daily_summary_no_expiries_survive_still_produces_a_row():
    empty = SurfaceResult(expiries=[], dropped_counts={}, violations=[])
    summary = compute_daily_summary(
        empty, spot=1400.5, underlying="RELIANCE", source="fixture",
        data_date=date(2026, 9, 24), run_date=date(2026, 9, 27),
    )
    assert summary.n_expiries == 0
    assert summary.atm_iv_30d is None
    assert summary.atm_iv_near is None
    assert summary.term_slope is None
    assert summary.n_violations == 0


# ---------------------------------------------------------------------------
# append_history / load_history
# ---------------------------------------------------------------------------


def _summary(run_date, underlying="RELIANCE", atm_iv_near=0.2):
    return DailySummary(
        run_date=run_date, data_date="2026-09-24", underlying=underlying, source="fixture",
        spot=1400.5, n_expiries=2, atm_iv_30d=0.21, atm_iv_near=atm_iv_near,
        term_slope=-0.02, n_violations=0,
    )


def test_append_history_creates_file_with_one_row(tmp_path):
    path = tmp_path / "RELIANCE.json"
    rows = append_history(_summary("2026-09-27"), path)
    assert len(rows) == 1
    assert rows[0]["run_date"] == "2026-09-27"
    on_disk = json.loads(path.read_text())
    assert on_disk == rows


def test_append_history_two_different_days_accumulates(tmp_path):
    path = tmp_path / "RELIANCE.json"
    append_history(_summary("2026-09-26"), path)
    rows = append_history(_summary("2026-09-27"), path)
    assert [r["run_date"] for r in rows] == ["2026-09-26", "2026-09-27"]


def test_append_history_same_day_rerun_upserts_not_duplicates(tmp_path):
    path = tmp_path / "RELIANCE.json"
    append_history(_summary("2026-09-27", atm_iv_near=0.2), path)
    rows = append_history(_summary("2026-09-27", atm_iv_near=0.99), path)
    assert len(rows) == 1
    assert rows[0]["atm_iv_near"] == pytest.approx(0.99)


def test_append_history_keeps_separate_underlyings_on_same_day(tmp_path):
    path = tmp_path / "history.json"
    append_history(_summary("2026-09-27", underlying="RELIANCE"), path)
    rows = append_history(_summary("2026-09-27", underlying="TATACHEM"), path)
    assert len(rows) == 2
    assert {r["underlying"] for r in rows} == {"RELIANCE", "TATACHEM"}


def test_load_history_missing_file_is_empty_list(tmp_path):
    assert load_history(tmp_path / "nope.json") == []


def test_load_history_rejects_non_array(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(HistoryError):
        load_history(path)


def test_load_history_rejects_row_missing_required_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"underlying": "RELIANCE"}]))  # no run_date
    with pytest.raises(HistoryError):
        load_history(path)


def test_load_history_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(HistoryError):
        load_history(path)


# ---------------------------------------------------------------------------
# CLI end to end, against the committed fixture, tmp_path only
# ---------------------------------------------------------------------------


def test_cli_writes_one_row_against_fixture(tmp_path, capsys):
    out = tmp_path / "RELIANCE.json"
    rc = main(["--underlying", "RELIANCE", "--provider", "fixture", "--as-of", "2026-09-27", "--out", str(out)])
    assert rc == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 1
    assert rows[0]["run_date"] == "2026-09-27"
    assert rows[0]["data_date"] == "2026-09-24"
    assert rows[0]["underlying"] == "RELIANCE"
    captured = capsys.readouterr()
    assert "RELIANCE" in captured.out


def test_cli_rerun_same_as_of_day_is_idempotent(tmp_path):
    out = tmp_path / "RELIANCE.json"
    main(["--underlying", "RELIANCE", "--provider", "fixture", "--as-of", "2026-09-27", "--out", str(out)])
    main(["--underlying", "RELIANCE", "--provider", "fixture", "--as-of", "2026-09-27", "--out", str(out)])
    rows = json.loads(out.read_text())
    assert len(rows) == 1


def test_cli_two_different_as_of_days_builds_a_small_history(tmp_path):
    out = tmp_path / "RELIANCE.json"
    main(["--underlying", "RELIANCE", "--provider", "fixture", "--as-of", "2026-09-26", "--out", str(out)])
    main(["--underlying", "RELIANCE", "--provider", "fixture", "--as-of", "2026-09-27", "--out", str(out)])
    rows = json.loads(out.read_text())
    assert [r["run_date"] for r in rows] == ["2026-09-26", "2026-09-27"]


def test_cli_unknown_underlying_exits_nonzero(tmp_path, capsys):
    out = tmp_path / "NOPE.json"
    rc = main(["--underlying", "NOPE", "--provider", "fixture", "--as-of", "2026-09-27", "--out", str(out)])
    assert rc == 1
    assert not out.exists()
    captured = capsys.readouterr()
    assert "error:" in captured.err
