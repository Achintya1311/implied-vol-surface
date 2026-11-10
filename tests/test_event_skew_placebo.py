"""P183: the before/after change gets a yardstick, the same change at every other quote day."""
from __future__ import annotations

from datetime import date, timedelta

from volsurface.event_skew import event_skew_placebo, main


def _rows(skews):
    start = date(2026, 10, 1)
    return [
        {"run_date": (start + timedelta(days=i)).isoformat(), "data_date": (start + timedelta(days=i)).isoformat(),
         "underlying": "RELIANCE", "source": "nse", "skew_25d": s}
        for i, s in enumerate(skews)
    ]


def test_a_step_is_the_largest_change_of_any_date():
    # flat at 0.00, then a jump to 0.05 on day 6 (2026-10-07)
    rows = _rows([0.0] * 6 + [0.05] * 6)
    p = event_skew_placebo(rows, "2026-10-07", window_days=3)
    assert p.n_dates >= 3
    assert p.n_as_large == 0  # no other date's change is as large as the step's


def test_noise_ranks_ordinary():
    rows = _rows([0.01, -0.01] * 6)
    p = event_skew_placebo(rows, "2026-10-07", window_days=2)
    assert p.n_dates > 0
    assert p.n_as_large == p.n_dates  # every other date moves as much


def test_no_change_means_no_placebo():
    p = event_skew_placebo(_rows([0.0, 0.0]), "2026-12-01", window_days=2)
    assert p.n_dates == 0 and p.n_as_large == 0


def test_cli_prints_the_placebo_line(tmp_path, capsys):
    import json

    path = tmp_path / "h.json"
    path.write_text(json.dumps(_rows([0.0] * 6 + [0.05] * 6)))
    assert main(["--underlying", "RELIANCE", "--event-date", "2026-10-07", "--window-days", "3",
                 "--history-file", str(path)]) == 0
    out = capsys.readouterr().out
    assert "placebo:" in out and "0 show a change at least as large" in out
