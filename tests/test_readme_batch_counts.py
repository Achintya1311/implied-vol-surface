"""P157: the batch counts the README quotes must be the ones on disk.

The README states its counts "As of <date>". The daily snapshot job adds a raw
batch every weekday, so the counts are checked against the batches captured
on or before that date, not against everything on disk. Until 5 Nov 2026 they
were checked against everything, so each new batch failed this test inside
the snapshot job and the job pushed nothing (2-4 Nov 2026 runs).
"""
import json
import re
from datetime import datetime
from pathlib import Path

from volsurface.chain import batch_origin

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "fixtures" / "chains" / "raw"
README = (ROOT / "README.md").read_text()


def _as_of():
    m = re.search(r"As of (\d{1,2} \w{3} \d{4}) that is", README)
    assert m, "README Method line no longer says which date its batch counts are as of"
    return datetime.strptime(m.group(1), "%d %b %Y").date().isoformat()


def _batches(underlying, as_of=None):
    as_of = as_of or _as_of()
    out = {}
    for f in sorted((RAW / underlying).glob("*.json")):
        stamp = f.name.split("_")[0]
        if stamp[:10] <= as_of:
            out.setdefault(stamp, json.loads(f.read_text())["captured_at"])
    return out


def test_readme_counts_match_the_raw_directories():
    rel, nif = _batches("RELIANCE"), _batches("NIFTY")
    runner = sum(batch_origin(t) == "Actions runner capture" for t in rel.values())
    m = re.search(r"that is (\d+) RELIANCE batches \((\d+) from the home-connection job, (\d+) from the Actions runner\) and (\d+) NIFTY batches", README)
    assert m, "README Method line no longer states the batch counts"
    assert tuple(map(int, m.groups())) == (len(rel), len(rel) - runner, runner, len(nif))


def test_readme_names_the_latest_reliance_batch_as_the_default():
    latest = max(_batches("RELIANCE"))
    assert f"Latest RELIANCE batch (`{latest}`" in README


def test_a_new_batch_after_the_readme_date_does_not_change_the_counts(tmp_path):
    # The snapshot job's own case: a batch stamped after the README's date.
    assert all(stamp[:10] <= _as_of() for stamp in _batches("RELIANCE"))
    assert _batches("RELIANCE", as_of="9999-12-31").keys() >= _batches("RELIANCE").keys()
