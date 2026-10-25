"""S18: every CLI that prints numbers on a named underlying (RELIANCE, NIFTY)
opens with PLAN.md rule 1's disclaimer, not only the static README."""

import shutil
from pathlib import Path

import pytest

from volsurface import DISCLAIMER, audit, event_skew, history, snapshot, solve

REPO = Path(__file__).resolve().parents[1]


def _first_line(capsys) -> str:
    return capsys.readouterr().out.splitlines()[0]


def test_disclaimer_is_the_plan_md_line():
    assert DISCLAIMER == (
        "Educational research on historical data. Not investment advice. "
        "Not a SEBI-registered research analyst."
    )


def test_solve(capsys):
    assert solve.main(["--underlying", "RELIANCE", "--provider", "fixture"]) == 0
    assert _first_line(capsys) == DISCLAIMER


def test_history(tmp_path, capsys):
    out = tmp_path / "RELIANCE.json"
    assert history.main(["--underlying", "RELIANCE", "--provider", "fixture", "--as-of", "2026-10-25", "--out", str(out)]) == 0
    assert _first_line(capsys) == DISCLAIMER


def test_snapshot(tmp_path, capsys):
    assert snapshot.main(["--underlying", "RELIANCE", "--provider", "fixture", "--out", str(tmp_path / "chain.json")]) == 0
    assert _first_line(capsys) == DISCLAIMER


def test_audit(capsys):
    audit.main(["--underlying", "RELIANCE", "--provider", "fixture"])
    assert _first_line(capsys) == DISCLAIMER


def test_event_skew(tmp_path, capsys):
    hist = tmp_path / "RELIANCE.json"
    shutil.copy(REPO / "fixtures" / "history" / "RELIANCE.json", hist)
    event_skew.main(["--underlying", "RELIANCE", "--event-date", "2026-10-15", "--history-file", str(hist)])
    assert _first_line(capsys) == DISCLAIMER
