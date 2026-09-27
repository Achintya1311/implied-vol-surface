"""Event-skew CLI/query tests (Day 7). All offline: synthetic history rows
plus tmp_path for file writes, no network.
"""

from __future__ import annotations

import json

import pytest

from volsurface.event_skew import event_skew_report, main


def _row(data_date, skew_25d, run_date=None, underlying="RELIANCE"):
    return {
        "run_date": run_date or data_date,
        "data_date": data_date,
        "underlying": underlying,
        "source": "fixture",
        "spot": 1400.5,
        "n_expiries": 2,
        "atm_iv_30d": 0.21,
        "atm_iv_near": 0.24,
        "term_slope": -0.02,
        "n_violations": 0,
        "skew_25d": skew_25d,
    }


# ---------------------------------------------------------------------------
# event_skew_report
# ---------------------------------------------------------------------------


def test_report_computes_change_across_a_real_event_window():
    rows = [
        _row("2026-10-10", 0.02),
        _row("2026-10-12", 0.03),
        _row("2026-10-16", 0.07),
        _row("2026-10-18", 0.08),
    ]
    report = event_skew_report(rows, "2026-10-15", window_days=5)
    assert report.pre_rows == 2
    assert report.post_rows == 2
    assert report.pre_mean_skew == pytest.approx(0.025)
    assert report.post_mean_skew == pytest.approx(0.075)
    assert report.change == pytest.approx(0.05)
    assert "usable skew_25d" in report.note


def test_report_excludes_rows_outside_the_window():
    rows = [
        _row("2026-09-01", 0.90),  # far outside the window either side
        _row("2026-10-12", 0.03),
        _row("2026-10-18", 0.08),
        _row("2027-01-01", -0.90),
    ]
    report = event_skew_report(rows, "2026-10-15", window_days=5)
    assert report.pre_rows == 1
    assert report.post_rows == 1
    assert report.pre_mean_skew == pytest.approx(0.03)
    assert report.post_mean_skew == pytest.approx(0.08)


def test_report_excludes_rows_with_no_skew_value():
    rows = [
        _row("2026-10-12", None),
        _row("2026-10-13", 0.05),
        _row("2026-10-17", 0.09),
    ]
    report = event_skew_report(rows, "2026-10-15", window_days=5)
    assert report.pre_rows == 1
    assert report.pre_mean_skew == pytest.approx(0.05)


def test_report_single_data_date_explains_why_there_is_no_comparison():
    # This sandbox's actual state today: one fixture-derived data_date, repeated.
    rows = [_row("2026-09-24", 0.03, run_date=d) for d in ("2026-09-25", "2026-09-26", "2026-09-27")]
    report = event_skew_report(rows, "2026-10-15", window_days=5)
    assert report.change is None
    assert report.pre_mean_skew is None
    assert report.post_mean_skew is None
    assert "one data_date" in report.note


def test_report_insufficient_data_on_one_side_says_so():
    rows = [_row("2026-10-12", 0.03), _row("2026-10-13", 0.04)]  # both pre-event only
    report = event_skew_report(rows, "2026-10-15", window_days=5)
    assert report.change is None
    assert report.pre_rows == 2
    assert report.post_rows == 0
    assert "not enough" in report.note


def test_event_date_row_itself_counts_as_post():
    rows = [_row("2026-10-15", 0.05)]
    report = event_skew_report(rows, "2026-10-15", window_days=5)
    assert report.post_rows == 1
    assert report.pre_rows == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_reports_against_a_synthetic_history_file(tmp_path, capsys):
    path = tmp_path / "RELIANCE.json"
    rows = [_row("2026-10-12", 0.03), _row("2026-10-13", 0.04), _row("2026-10-18", 0.09)]
    path.write_text(json.dumps(rows))

    rc = main([
        "--underlying", "RELIANCE", "--event-date", "2026-10-15",
        "--window-days", "5", "--history-file", str(path),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "RELIANCE event=2026-10-15" in captured.out
    assert "change=" in captured.out


def test_cli_missing_history_file_exits_nonzero(tmp_path, capsys):
    rc = main([
        "--underlying", "NOPE", "--event-date", "2026-10-15",
        "--history-file", str(tmp_path / "missing.json"),
    ])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_cli_empty_history_file_exits_nonzero(tmp_path, capsys):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    rc = main(["--underlying", "RELIANCE", "--event-date", "2026-10-15", "--history-file", str(path)])
    assert rc == 1
    assert "error:" in capsys.readouterr().err
