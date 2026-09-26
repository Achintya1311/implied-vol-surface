"""Day 5 plotting tests. Unit-test the pure data-shaping helpers directly
(no filesystem, no plotting backend); only the render_* tests touch a real
plotting library, against the committed RELIANCE fixture, writing to
tmp_path so nothing here depends on or pollutes outputs/.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from volsurface.chain import get_provider
from volsurface.iv import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, solve_chain_ivs
from volsurface.plot import (
    PlotError,
    atm_iv,
    build_surface_html,
    plot_smiles,
    plot_surface_3d,
    plot_term_structure,
    render_all,
    smile_curve,
    term_structure_points,
)
from volsurface.surface import ExpirySmile, build_surface

R = DEFAULT_RATE
Q = DEFAULT_DIVIDEND_YIELD


def _smile(expiry, T, strikes, ivs):
    order = np.argsort(strikes)
    strikes = np.asarray(strikes, dtype=float)[order]
    ivs = np.asarray(ivs, dtype=float)[order]
    return ExpirySmile(
        expiry=expiry, T=T, strikes=strikes,
        smile=PchipInterpolator(strikes, ivs, extrapolate=False),
    )


# ---------------------------------------------------------------------------
# atm_iv / term_structure_points / smile_curve - pure functions, synthetic data
# ---------------------------------------------------------------------------


def test_atm_iv_inside_domain():
    es = _smile("e1", 0.1, [1300.0, 1400.0, 1500.0], [0.26, 0.22, 0.20])
    iv = atm_iv(es, 1400.0)
    assert iv == pytest.approx(0.22)


def test_atm_iv_outside_domain_is_none():
    es = _smile("e1", 0.1, [1450.0, 1475.0, 1500.0], [0.24, 0.23, 0.22])
    assert atm_iv(es, 1400.0) is None


def test_term_structure_points_sorted_by_tenor_and_skips_out_of_domain():
    far = _smile("2026-10-29", 0.2, [1300.0, 1400.0, 1500.0], [0.24, 0.21, 0.19])
    near = _smile("2026-09-29", 0.05, [1300.0, 1400.0, 1500.0], [0.28, 0.24, 0.21])
    no_atm = _smile("2026-11-29", 0.3, [1450.0, 1475.0, 1500.0], [0.20, 0.19, 0.18])

    points = term_structure_points([far, near, no_atm], spot=1400.0)

    assert [p[0] for p in points] == ["2026-09-29", "2026-10-29"]
    assert points[0][1] < points[1][1]  # sorted by T
    assert points[0][2] == pytest.approx(0.24)
    assert points[1][2] == pytest.approx(0.21)


def test_term_structure_points_empty_when_no_expiry_spans_spot():
    es = _smile("e1", 0.1, [1450.0, 1475.0, 1500.0], [0.24, 0.23, 0.22])
    assert term_structure_points([es], spot=1400.0) == []


def test_smile_curve_stays_within_own_strike_domain():
    es = _smile("e1", 0.1, [1300.0, 1400.0, 1500.0], [0.26, 0.22, 0.20])
    grid, ivs = smile_curve(es, grid_points=10)
    assert grid.min() == pytest.approx(1300.0)
    assert grid.max() == pytest.approx(1500.0)
    assert not np.isnan(ivs).any()


# ---------------------------------------------------------------------------
# render_all and friends - no fitted expiries at all
# ---------------------------------------------------------------------------


def test_plot_functions_raise_on_no_expiries(tmp_path):
    for fn, args in [
        (plot_smiles, ([], "RELIANCE", tmp_path)),
        (plot_term_structure, ([], 1400.0, "RELIANCE", tmp_path)),
        (plot_surface_3d, ([], "RELIANCE", tmp_path)),
        (build_surface_html, ([], "RELIANCE", tmp_path)),
    ]:
        with pytest.raises(PlotError):
            fn(*args)


def test_render_all_raises_on_empty_surface_result(tmp_path):
    from volsurface.surface import SurfaceResult

    empty = SurfaceResult(expiries=[], dropped_counts={})
    with pytest.raises(PlotError):
        render_all(empty, spot=1400.0, underlying="RELIANCE", output_dir=tmp_path)


# ---------------------------------------------------------------------------
# render_all against the committed RELIANCE fixture - real files, tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture()
def reliance_surface():
    snapshot = get_provider("fixture").fetch("RELIANCE")
    iv_table = solve_chain_ivs(snapshot.quotes, snapshot.spot, snapshot.timestamp, r=R, q=Q)
    result = build_surface(iv_table, snapshot.quotes, snapshot.spot, date(2026, 9, 24), R, Q)
    return snapshot, result


def test_render_all_writes_all_four_files(reliance_surface, tmp_path):
    snapshot, result = reliance_surface
    paths = render_all(result, snapshot.spot, "RELIANCE", output_dir=tmp_path)

    assert set(paths) == {"smile_png", "term_structure_png", "surface_png", "surface_html"}
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
        assert path.parent == tmp_path


def test_render_all_png_files_are_actually_png(reliance_surface, tmp_path):
    snapshot, result = reliance_surface
    paths = render_all(result, snapshot.spot, "RELIANCE", output_dir=tmp_path)
    png_signature = b"\x89PNG\r\n\x1a\n"
    for key in ("smile_png", "term_structure_png", "surface_png"):
        assert paths[key].read_bytes()[:8] == png_signature


def test_render_all_html_is_standalone_and_mentions_both_expiries(reliance_surface, tmp_path):
    snapshot, result = reliance_surface
    paths = render_all(result, snapshot.spot, "RELIANCE", output_dir=tmp_path)
    html = paths["surface_html"].read_text()

    assert "<html" in html.lower()
    # Plotly's JS is inlined (include_plotlyjs=True) as a <script> body, not
    # loaded via a <script src="..."> tag pointing at a CDN - this is what
    # makes the file openable with no network. (The library's own source
    # mentions cdn.plot.ly internally, e.g. in a config schema string, so
    # that substring alone isn't a valid check.)
    assert "<script src=" not in html
    assert "Plotly.newPlot" in html
    assert len(html) > 1_000_000  # the inlined bundle itself is well over 1MB
    for es in result.expiries:
        assert es.expiry in html


def test_render_all_term_structure_reflects_real_atm_skew(reliance_surface, tmp_path):
    """The fixture's spot (1400.5) sits inside both expiries' surviving
    strike ranges, so both should show up in the term structure - and, per
    this project's own Day 5 finding, the near-term ATM IV is *higher* than
    the far-term one on this fixture (see README Findings).
    """
    snapshot, result = reliance_surface
    points = term_structure_points(result.expiries, snapshot.spot)
    assert [p[0] for p in points] == ["2026-09-29", "2026-10-29"]
    near_iv, far_iv = points[0][2], points[1][2]
    assert near_iv > far_iv
