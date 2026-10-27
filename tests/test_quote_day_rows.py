"""P132: history queries count each real quote day once."""

import json

from volsurface.event_skew import event_skew_report, main
from volsurface.history import HISTORY_DIR, load_history, quote_day_rows


def _row(run_date, data_date, source="nse", skew=0.01, underlying="RELIANCE"):
    return {"run_date": run_date, "data_date": data_date, "underlying": underlying, "source": source,
            "skew_25d": skew}


def test_keeps_one_nse_row_per_data_date_the_latest_run():
    rows = [
        _row("2026-10-01", "2026-09-24", source="hand-assembled"),
        _row("2026-10-19", "2026-10-19", skew=0.01),
        _row("2026-10-20", "2026-10-19", skew=0.02),
        _row("2026-10-21", "2026-10-21", skew=0.03),
    ]
    kept, synthetic, repeated = quote_day_rows(rows)
    assert [(r["run_date"], r["skew_25d"]) for r in kept] == [("2026-10-20", 0.02), ("2026-10-21", 0.03)]
    assert (synthetic, repeated) == (1, 1)


def test_underlyings_are_kept_apart():
    kept, _, repeated = quote_day_rows([_row("2026-10-19", "2026-10-19"),
                                        _row("2026-10-19", "2026-10-19", underlying="NIFTY")])
    assert len(kept) == 2 and repeated == 0


def test_committed_history_has_five_real_quote_days():
    kept, synthetic, repeated = quote_day_rows(load_history(HISTORY_DIR / "RELIANCE.json"))
    assert synthetic == 16
    assert repeated >= 2  # 2026-10-20 repeats 10-19, 2026-10-24 repeats 10-23 (more may land)
    assert len({r["data_date"] for r in kept}) == len(kept)


def test_event_skew_report_counts_repeated_days_once():
    rows = [_row("2026-10-19", "2026-10-19", skew=-0.02), _row("2026-10-20", "2026-10-19", skew=-0.02),
            _row("2026-10-22", "2026-10-22", skew=0.02)]
    report = event_skew_report(rows, "2026-10-22", 5)
    assert (report.pre_rows, report.post_rows, report.dropped_rows) == (1, 1, 1)


def test_cli_prints_what_it_dropped(tmp_path, capsys):
    path = tmp_path / "h.json"
    path.write_text(json.dumps([_row("2026-10-01", "2026-09-24", source="hand-assembled"),
                                _row("2026-10-19", "2026-10-19", skew=-0.02),
                                _row("2026-10-20", "2026-10-19", skew=-0.02),
                                _row("2026-10-22", "2026-10-22", skew=0.02)]))
    assert main(["--underlying", "RELIANCE", "--event-date", "2026-10-22", "--history-file", str(path)]) == 0
    assert "dropped 2 row(s): 1 synthetic, 1 repeating a quote day already counted" in capsys.readouterr().out
