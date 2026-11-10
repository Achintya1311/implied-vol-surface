"""P181: each history row and the v0.6 contract say which tenor their vol figures read."""
from __future__ import annotations

import json

import pytest

from volsurface import history

LIVE = history.ROOT / "fixtures" / "chains" / "live" / "RELIANCE"


def _run(tmp_path, day):
    out, contract = tmp_path / "h.json", tmp_path / "c.json"
    assert history.main(["--underlying", "RELIANCE", "--chain-file", str(LIVE / f"{day}.json"),
                         "--out", str(out), "--contract", str(contract)]) == 0
    return json.loads(out.read_text())[-1], json.loads(contract.read_text())["vol"]


def test_row_and_contract_carry_the_tenors(tmp_path):
    row, vol = _run(tmp_path, "2026-11-09")
    assert row["atm_tenor_days"] == 14  # 23 Nov 2026 expiry, 14 days after 9 Nov
    near, far = row["term_slope_tenor_days"]
    assert near == 14 and far > near
    assert vol["atm_tenor_days"] == 14
    assert vol["term_slope_tenor_days"] == [near, far]


def test_tenors_are_none_when_their_figure_is(tmp_path):
    row, vol = _run(tmp_path, "2026-10-30")
    assert row["term_slope"] is None and row["term_slope_tenor_days"] is None
    assert vol["term_slope_tenor_days"] is None
    assert row["atm_tenor_days"] == 24


@pytest.mark.parametrize("key", ["atm_tenor_days", "term_slope_tenor_days"])
def test_history_row_keys_list_the_tenors(key):
    assert key in history.HISTORY_ROW_KEYS
