"""P157: the batch counts the README quotes must be the ones on disk."""
import re
from pathlib import Path

from volsurface.chain import batch_origin
import json

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "fixtures" / "chains" / "raw"
README = (ROOT / "README.md").read_text()


def _batches(underlying):
    out = {}
    for f in sorted((RAW / underlying).glob("*.json")):
        out.setdefault(f.name.split("_")[0], json.loads(f.read_text())["captured_at"])
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
