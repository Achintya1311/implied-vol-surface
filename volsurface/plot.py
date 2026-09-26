"""Day 5: turn a :class:`~volsurface.surface.SurfaceResult` into pictures.

Two 2D slices (PNG) plus one 3D view (PNG and a standalone interactive HTML)
are built directly from the *same* per-expiry PCHIP smiles Day 4 already
fit and arbitrage-checked - nothing here refits anything.

The "surface" plotted is a union of independently fitted per-expiry smiles,
not a single sheet jointly interpolated across strike *and* tenor. Day 4
deliberately never interpolates across expiries (calendar arbitrage is only
checked between them, never enforced by construction), so a jointly-smoothed
sheet here would show a shape the fitting stage never actually produced -
see the README's Limitations for why that matters for what this view proves.

Only the 3D view gets an HTML export: reading an exact (strike, tenor, IV)
triple off a static 3D plot is close to impossible (occlusion, projection),
which is exactly what an interactive camera and hover tooltip fix. The 2D
slices already show every value a line plot can show, so a second,
non-interactive HTML copy of them would add nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from volsurface.surface import ExpirySmile, SurfaceResult

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

SMILE_GRID_POINTS = 60


class PlotError(ValueError):
    """The surface has nothing fitted to plot."""


def atm_iv(expiry_smile: ExpirySmile, spot: float) -> float | None:
    """The smile's own IV at ``strike == spot``, or ``None`` if ``spot``
    falls outside the strikes that actually survived filtering for this
    expiry (:class:`~scipy.interpolate.PchipInterpolator` returns NaN there,
    since Day 4 built it with ``extrapolate=False``).
    """
    iv = float(expiry_smile.smile(spot))
    return None if np.isnan(iv) else iv


def term_structure_points(
    expiries: list[ExpirySmile], spot: float
) -> list[tuple[str, float, float]]:
    """``(expiry, T, atm_iv)`` for every expiry whose fitted domain covers
    ``spot``, sorted by tenor. An expiry dropped here (rather than shown with
    a gap) means its surviving strikes never reached the money - honest
    silence, not an interpolated guess.
    """
    points = []
    for es in sorted(expiries, key=lambda e: e.T):
        iv = atm_iv(es, spot)
        if iv is not None:
            points.append((es.expiry, es.T, iv))
    return points


def smile_curve(
    expiry_smile: ExpirySmile, grid_points: int = SMILE_GRID_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """A fine strike grid spanning only this expiry's own surviving strikes,
    and the smile's IV at each point - never extrapolated past the data.
    """
    lo, hi = expiry_smile.strikes.min(), expiry_smile.strikes.max()
    grid = np.linspace(lo, hi, grid_points)
    return grid, expiry_smile.smile(grid)


def _require_expiries(expiries: list[ExpirySmile]) -> None:
    if not expiries:
        raise PlotError("no expiry has a fitted smile to plot")


def plot_smiles(
    expiries: list[ExpirySmile], underlying: str, output_dir: Path | None = None
) -> Path:
    """One IV-vs-strike line per expiry, with the actual surviving (strike,
    IV) points marked separately from the fitted curve between them.
    """
    _require_expiries(expiries)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"iv_smiles_{underlying}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    for es in sorted(expiries, key=lambda e: e.T):
        grid, ivs = smile_curve(es)
        (line,) = ax.plot(grid, ivs, label=f"{es.expiry} (T={es.T:.3f}y)")
        ax.scatter(es.strikes, es.smile(es.strikes), s=18, color=line.get_color())
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied vol")
    ax.set_title(f"{underlying} - IV smile by expiry")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_term_structure(
    expiries: list[ExpirySmile], spot: float, underlying: str, output_dir: Path | None = None
) -> Path:
    """ATM IV against tenor, across expiries. If no expiry's surviving
    strikes span the money, the chart says so explicitly rather than showing
    an empty axes with no explanation.
    """
    _require_expiries(expiries)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"iv_term_structure_{underlying}.png"

    points = term_structure_points(expiries, spot)

    fig, ax = plt.subplots(figsize=(7, 5))
    if not points:
        ax.text(
            0.5, 0.5, "no expiry's surviving strikes span the spot",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        Ts = [p[1] for p in points]
        ivs = [p[2] for p in points]
        ax.plot(Ts, ivs, marker="o")
        for expiry, T, iv in points:
            ax.annotate(expiry, (T, iv), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.set_xlabel("Time to expiry (years)")
        ax.set_ylabel(f"ATM implied vol (spot={spot:g})")
    ax.set_title(f"{underlying} - ATM term structure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_surface_3d(
    expiries: list[ExpirySmile], underlying: str, output_dir: Path | None = None
) -> Path:
    """A static 3D view: each expiry's fitted smile drawn as its own line in
    (strike, tenor, IV) space, with markers at the strikes that actually
    survived filtering.
    """
    _require_expiries(expiries)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"iv_surface_{underlying}.png"

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    for es in sorted(expiries, key=lambda e: e.T):
        grid, ivs = smile_curve(es)
        ax.plot(grid, [es.T] * len(grid), ivs, label=f"{es.expiry}")
        ax.scatter(es.strikes, [es.T] * len(es.strikes), es.smile(es.strikes), s=15)
    ax.set_xlabel("Strike")
    ax.set_ylabel("T (years)")
    ax.set_zlabel("Implied vol")
    ax.set_title(f"{underlying} - IV surface (per-expiry smiles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_surface_html(
    expiries: list[ExpirySmile], underlying: str, output_dir: Path | None = None
) -> Path:
    """The same per-expiry lines as :func:`plot_surface_3d`, as a standalone
    interactive HTML page (Plotly's JS bundled inline, so it opens with no
    network - matches this project's fixtures-first, offline-by-default
    posture).
    """
    _require_expiries(expiries)
    import plotly.graph_objects as go

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"iv_surface_{underlying}.html"

    fig = go.Figure()
    for es in sorted(expiries, key=lambda e: e.T):
        grid, ivs = smile_curve(es)
        fig.add_trace(
            go.Scatter3d(
                x=grid, y=[es.T] * len(grid), z=ivs,
                mode="lines", name=f"{es.expiry} (fitted)",
                line=dict(width=4),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=es.strikes, y=[es.T] * len(es.strikes), z=es.smile(es.strikes),
                mode="markers", name=f"{es.expiry} (surviving strikes)",
                marker=dict(size=4),
            )
        )
    fig.update_layout(
        title=f"{underlying} - IV surface (per-expiry smiles)",
        scene=dict(xaxis_title="Strike", yaxis_title="T (years)", zaxis_title="Implied vol"),
    )
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    return out_path


def render_all(
    result: SurfaceResult, spot: float, underlying: str, output_dir: Path | None = None
) -> dict[str, Path]:
    """Run every plot in this module against one :class:`SurfaceResult`.
    Raises :class:`PlotError` if no expiry survived Day 4's filtering -
    callers (the CLI) should check ``result.expiries`` first if they'd
    rather report that as a normal outcome than catch an exception.
    """
    _require_expiries(result.expiries)
    return {
        "smile_png": plot_smiles(result.expiries, underlying, output_dir),
        "term_structure_png": plot_term_structure(result.expiries, spot, underlying, output_dir),
        "surface_png": plot_surface_3d(result.expiries, underlying, output_dir),
        "surface_html": build_surface_html(result.expiries, underlying, output_dir),
    }
