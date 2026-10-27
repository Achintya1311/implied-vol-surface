"""P130: say where each raw batch came from, and order batches by time.

The home job writes naive IST ``captured_at`` and IST file stamps; the
Actions runner writes UTC with an offset and UTC stamps. Stamp text does
not sort two clocks into time order."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from volsurface.chain import RawCaptureProvider, batch_origin, batch_time_utc
from test_chain import _trimmed_real_v3_payload  # noqa: E402  (pytest puts tests/ on sys.path)


def _write(tmp_path, stamp, captured_at, spot):
    payload = _trimmed_real_v3_payload()
    payload["records"]["underlyingValue"] = spot
    out = tmp_path / "RELIANCE" / f"{stamp}_27-Oct-2026.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"captured_at": captured_at, "source": "https://www.nseindia.com/x", "payload": payload}))


@pytest.mark.parametrize(
    "captured_at, origin",
    [
        ("2026-10-26T17:06:16.185057+00:00", "Actions runner capture"),
        ("2026-10-27T14:30:39.624964", "home-connection capture"),
        ("2026-10-27T14:30:39+05:30", "home-connection capture"),
        ("2026-10-27T14:30:39+01:00", "raw capture"),
    ],
)
def test_batch_origin_from_captured_at(captured_at, origin):
    assert batch_origin(captured_at) == origin


def test_naive_captured_at_is_read_as_ist():
    assert batch_time_utc("2026-10-27T14:30:00", "2026-10-27T1430") == datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc)


def test_unparseable_captured_at_falls_back_to_the_stamp_as_ist():
    assert batch_time_utc("not a time", "2026-10-27T1430") == datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc)


def test_same_day_runner_batch_after_the_close_is_newer_than_a_home_batch_mid_session(tmp_path):
    # Home 14:30 IST = 09:00 UTC; runner 11:30 UTC = 17:00 IST. Text order says home is newer.
    _write(tmp_path, "2026-10-27T1430", "2026-10-27T14:30:39.624964", 1192.3)
    _write(tmp_path, "2026-10-27T1130", "2026-10-27T11:30:00+00:00", 1201.0)
    snap = RawCaptureProvider(directory=tmp_path).fetch("RELIANCE")
    assert snap.spot == pytest.approx(1201.0)
    assert snap.source.startswith("Actions runner capture (1 expiries, batch 2026-10-27T1130")


def test_pinned_home_batch_is_labelled_home(tmp_path):
    _write(tmp_path, "2026-10-27T1430", "2026-10-27T14:30:39.624964", 1192.3)
    _write(tmp_path, "2026-10-27T1130", "2026-10-27T11:30:00+00:00", 1201.0)
    snap = RawCaptureProvider(directory=tmp_path, capture="2026-10-27T1430").fetch("RELIANCE")
    assert snap.source.startswith("home-connection capture")


def test_committed_runner_batches_are_labelled_runner():
    for stamp in ("2026-10-24T0228", "2026-10-26T1706"):
        assert RawCaptureProvider(capture=stamp).fetch("RELIANCE").source.startswith("Actions runner capture")
    assert RawCaptureProvider(capture="2026-10-16T2319").fetch("RELIANCE").source.startswith("home-connection capture")
