"""R26: what a butterfly count counts. Grid-point clusters on the fitted
smile, and butterfly checks on quoted strikes at mid and at executable
bid/ask. Offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsurface.surface import (
    ArbitrageViolation,
    butterfly_clusters,
    quoted_butterflies,
)


def _v(expiry, idx, K):
    return ArbitrageViolation(kind="butterfly", expiry=expiry, strike=K, detail="", grid_index=idx)


def test_consecutive_grid_points_form_one_cluster():
    vs = [_v("e1", 10, 1205.0), _v("e1", 11, 1206.0), _v("e1", 12, 1207.0), _v("e1", 40, 1250.0)]
    clusters = butterfly_clusters(vs, {"e1": np.array([1200.0, 1210.0, 1240.0, 1260.0])})
    assert [(c.expiry, c.points, c.k_low, c.k_high) for c in clusters] == [
        ("e1", 3, 1205.0, 1207.0),
        ("e1", 1, 1250.0, 1250.0),
    ]
    assert (clusters[0].quoted_low, clusters[0].quoted_high) == (1200.0, 1210.0)
    assert (clusters[1].quoted_low, clusters[1].quoted_high) == (1240.0, 1260.0)


def test_clusters_never_span_two_expiries():
    vs = [_v("e1", 5, 100.0), _v("e2", 6, 101.0)]
    clusters = butterfly_clusters(vs, {"e1": np.array([90.0, 110.0]), "e2": np.array([90.0, 110.0])})
    assert len(clusters) == 2


def test_calendar_violations_are_ignored():
    vs = [ArbitrageViolation(kind="calendar", expiry="e1", strike=100.0, detail="")]
    assert butterfly_clusters(vs, {"e1": np.array([90.0, 110.0])}) == []


def _quotes(prices, spread=0.5, opt="CE", strikes=(100.0, 110.0, 120.0)):
    return pd.DataFrame(
        {"expiry": "e1", "strike": list(strikes), "option_type": opt,
         "bid": [p - spread for p in prices], "ask": [p + spread for p in prices]}
    )


def test_convex_mids_are_clean():
    res = quoted_butterflies(_quotes([20.0, 12.0, 6.0]))
    assert (res.triples, res.negative_mid, res.negative_executable) == (1, 0, 0)


def test_concave_mids_inside_the_spread_are_not_tradeable():
    # 20 - 2*16.2 + 12 = -0.4 at mid; at ask/bid/ask: 20.5 - 2*15.7 + 12.5 = 1.6
    res = quoted_butterflies(_quotes([20.0, 16.2, 12.0]))
    assert (res.negative_mid, res.negative_executable) == (1, 0)


def test_concave_beyond_the_spread_is_executable():
    res = quoted_butterflies(_quotes([20.0, 18.0, 12.0], spread=0.1))
    assert (res.negative_mid, res.negative_executable) == (1, 1)
    assert res.examples[0].strikes == (100.0, 110.0, 120.0)


def test_uneven_strike_spacing_uses_weighted_butterfly():
    # K = 100, 110, 130: weights 2/3 and 1/3 on the wings. Linear prices are
    # exactly zero-cost, not negative.
    res = quoted_butterflies(_quotes([30.0, 20.0, 0.0 + 0.0001], spread=0.0, strikes=(100.0, 110.0, 130.0)))
    assert res.negative_mid == 0


def test_puts_and_calls_are_checked_separately():
    both = pd.concat([_quotes([20.0, 18.0, 12.0], spread=0.1), _quotes([6.0, 12.0, 20.0], opt="PE")])
    res = quoted_butterflies(both)
    assert res.triples == 2
    assert res.negative_mid == 1


@pytest.mark.parametrize(
    "underlying,stamp,forward,grid,clusters,triples,neg_mid,neg_exec",
    [
        ("RELIANCE", "2026-10-16T2319", "model", 26, 8, 46, 2, 0),
        ("RELIANCE", "2026-10-16T2319", "parity", 15, 6, 48, 2, 0),
        ("RELIANCE", "2026-10-25T2109", "model", 52, 11, 52, 11, 0),
        ("RELIANCE", "2026-10-25T2109", "parity", 43, 9, 51, 9, 0),
        ("NIFTY", "2026-10-16T2319", "model", 149, 52, 232, 48, 9),
        ("NIFTY", "2026-10-16T2319", "parity", 143, 49, 264, 55, 9),
        ("NIFTY", "2026-10-25T2109", "model", 115, 53, 261, 59, 2),
        ("NIFTY", "2026-10-25T2109", "parity", 107, 49, 258, 54, 2),
    ],
)
def test_cli_reports_clusters_and_quoted_checks(
    capsys, underlying, stamp, forward, grid, clusters, triples, neg_mid, neg_exec
):
    # The README's R26 table: grid points, the bends they make, and the
    # quoted-strike check at mid and at executable bid/ask.
    from volsurface import solve

    argv = ["--underlying", underlying, "--provider", "nse_raw", "--capture", stamp,
            "--surface", "--forward", forward]
    assert solve.main(argv) == 0
    out = capsys.readouterr().out
    assert f"butterfly: {grid} grid point(s) in {clusters} cluster(s) on the fitted smile" in out
    assert (
        f"{neg_mid}/{triples} negative at mid, {neg_exec}/{triples} negative at executable bid/ask" in out
    )
