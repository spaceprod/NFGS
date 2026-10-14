#!/usr/bin/env python3
"""Plot NFGS DCT B^> bottom maps for reference and robin weights 0 / 0.5 / 1.0.

For each TD configuration, write a 2-row figure:

    top:    reference | rho=0.0 | rho=0.5 | rho=1.0
    bottom: (legend)  | Erel    | Erel    | Erel

Top row shows B_z^> with horizontal (B_x^>, B_y^>) arrows.  Bottom row shows
``ΔB_z^> = B_z^{>,NFGS} - B_z^{>,ref}`` under each solution panel, titled by
the numerical ``E^{rel}_{B^>}``.  When multiple TDs are requested, also write a
stacked paper summary ``..._summary.png`` with panels (a)--(c).
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.scripts.nfgs_figure_style import (  # noqa: E402
    FIG_AXIS_LABEL_FONTSIZE,
    FIG_CBAR_LABEL_FONTSIZE,
    FIG_LEGEND_FONTSIZE,
    FIG_PANEL_LETTER_FONTSIZE,
    FIG_TICK_FONTSIZE,
    FIG_TITLE_FONTSIZE,
    font_scale_to_match_figure4,
)
from nfgs._backend.scripts import run_analytic_validation as analytic  # noqa: E402
from nfgs._backend.scripts.plot_nfgs_td_robin_metrics_vs_er import (  # noqa: E402
    EREL_KEY,
    bottom_plane_bgt_nfgs,
    make_bgt_helper_solver,
)
from nfgs._backend.scripts.run_nfgs_td_robin_weight_sweep import (  # noqa: E402
    load_field_cache,
    nfgs_field_cache_path,
    parse_weights,
    reference_field_cache_path,
    weight_tag,
)
from nfgs._backend.util.field_ops import curl_zyx_nfpt  # noqa: E402

DEFAULT_SWEEP_DIR = ROOT / "results" / "nfgs_td_robin_weight_sweep"
DEFAULT_CASE_CACHE = ROOT / "results" / "nfgs_analytic_validation_td" / "fields"
DEFAULT_WEIGHTS = [0.0, 0.5, 1.0]
# Target number of quiver samples along the long axis (journal-readable).
DEFAULT_QUIVER_STRIDE_TARGET = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--case-cache-dir", type=Path, default=DEFAULT_CASE_CACHE)
    p.add_argument("--weights", default=None, help="Default: 0.0,0.5,1.0")
    p.add_argument("--benchmarks", default="td1,td2,td3")
    p.add_argument("--init-mode", default="gs_cfcf")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "paper_writing" / "core_figures",
    )
    p.add_argument("--mean-bz-policy", default="uniform_open")
    p.add_argument(
        "--horizontal-zero-mode-policy", default="frozen_linear_harmonic"
    )
    p.add_argument("--d0-zero-mode-policy", default="project")
    p.add_argument("--source-zero-mode-policy", default="project")
    p.add_argument("--harmonic-residual-policy", default="diagnose")
    p.add_argument("--mean-mode-tolerance", type=float, default=1.0e-6)
    p.add_argument("--source-zero-mode-tolerance", type=float, default=1.0e-2)
    p.add_argument(
        "--gaussian-horizontal-operator",
        choices=("fd", "spectral"),
        default="spectral",
    )
    p.add_argument(
        "--quiver-stride-target",
        type=int,
        default=DEFAULT_QUIVER_STRIDE_TARGET,
        help="Approx. arrow count along the long axis (smaller => coarser).",
    )
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="CSV with E_rel_Bgt (default: <sweep-dir>/metrics_with_erel_bgt.csv).",
    )
    return p.parse_args()


def load_case_meta(case_cache_dir: Path, name: str) -> dict[str, Any]:
    path = case_cache_dir / f"{name}__case_meta.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing case meta {path}; run analytic TD validation once to cache it."
        )
    return json.loads(path.read_text())


def weight_panel_label(weight: float) -> str:
    if abs(weight - round(weight)) < 1.0e-12:
        return f"{int(round(weight))}"
    return f"{weight:g}"


def rho_panel_title(weight: float) -> str:
    return rf"$\rho={weight_panel_label(weight)}$"


def erel_panel_title(erel: float) -> str:
    return rf"$E^{{\mathrm{{rel}}}}_{{B^{{>}}}}={erel:.2g}$"


def load_erel_lookup(path: Path) -> dict[tuple[str, float], float]:
    """Map ``(benchmark, robin_weight)`` -> ``E_rel_Bgt``."""
    out: dict[tuple[str, float], float] = {}
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["benchmark"].lower(), round(float(row["robin_weight"]), 1))
            out[key] = float(row[EREL_KEY])
    return out


def metric_slices_from_meta(
    meta: dict[str, Any],
) -> tuple[slice, slice, slice] | None:
    raw = meta.get("metric_slices")
    if not raw:
        return None
    return tuple(slice(a, b, c) for a, b, c in raw)  # type: ignore[return-value]


def _merge_slices(a: slice, b: slice, *, bound: int) -> slice:
    """Return the union of two half-open index slices clipped to ``[0, bound)``."""
    a0 = 0 if a.start is None else int(a.start)
    a1 = bound if a.stop is None else int(a.stop)
    b0 = 0 if b.start is None else int(b.start)
    b1 = bound if b.stop is None else int(b.stop)
    return slice(max(0, min(a0, b0)), min(bound, max(a1, b1)))


def _bbox_slices_from_mask(
    mask: np.ndarray,
    *,
    margin_fraction: float,
) -> tuple[slice, slice]:
    """Bounding-box slices around ``mask`` with a relative margin."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        ny, nx = mask.shape
        return slice(None), slice(None)
    ny, nx = mask.shape
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad_y = int(margin_fraction * (y1 - y0 + 1))
    pad_x = int(margin_fraction * (x1 - x0 + 1))
    return (
        slice(max(0, y0 - pad_y), min(ny, y1 + pad_y + 1)),
        slice(max(0, x0 - pad_x), min(nx, x1 + pad_x + 1)),
    )


def strong_current_plot_slices(
    reference_fields: dict[str, np.ndarray],
    *,
    dx: float,
    dy: float,
    dz: float,
    metric_slices: tuple[slice, slice, slice] | None,
    ny: int,
    nx: int,
    amplitude_fraction: float = 0.25,
    margin_fraction: float = 0.20,
) -> tuple[slice, slice, slice] | None:
    """Expand the metric ROI so strong bottom-plane |J_z| is framed with margin.

  The photospheric metric crop can clip the bipolar imprint lobes in paper
  panels.  We keep the metric ROI as a floor and union it with the bounding
  box of ``|J_z(z=0)| >= amplitude_fraction * max|J_z|`` plus a relative
  margin.
    """
    jx, jy, jz = curl_zyx_nfpt(
        reference_fields["Bx"],
        reference_fields["By"],
        reference_fields["Bz"],
        dx=dx,
        dy=dy,
        dz=dz,
    )
    del jx, jy
    abs_jz_bottom = np.abs(np.asarray(jz[0], dtype=np.float64))
    peak = float(np.nanmax(abs_jz_bottom))
    if not np.isfinite(peak) or peak <= 0.0:
        return metric_slices

    mask = abs_jz_bottom >= amplitude_fraction * peak
    if not np.any(mask):
        return metric_slices

    strong_sy, strong_sx = _bbox_slices_from_mask(
        mask,
        margin_fraction=margin_fraction,
    )
    if metric_slices is None:
        return slice(None), strong_sy, strong_sx

    sz, metric_sy, metric_sx = metric_slices
    return (
        sz,
        _merge_slices(metric_sy, strong_sy, bound=int(ny)),
        _merge_slices(metric_sx, strong_sx, bound=int(nx)),
    )


def crop_bgt_to_metric_roi(
    field: np.ndarray,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    metric_slices: tuple[slice, slice, slice],
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Crop ``(3, ny, nx)`` B^> to the requested horizontal ROI."""
    _sz, sy, sx = metric_slices
    ny, nx = int(field.shape[1]), int(field.shape[2])
    xs = np.linspace(x_bounds[0], x_bounds[1], nx, endpoint=True)
    ys = np.linspace(y_bounds[0], y_bounds[1], ny, endpoint=True)
    cropped = np.asarray(field[:, sy, sx], dtype=np.float64)
    y0 = 0 if sy.start is None else int(sy.start)
    y1 = ny - 1 if sy.stop is None else int(sy.stop) - 1
    x0 = 0 if sx.start is None else int(sx.start)
    x1 = nx - 1 if sx.stop is None else int(sx.stop) - 1
    return cropped, (float(xs[x0]), float(xs[x1])), (float(ys[y0]), float(ys[y1]))


def bgt_plot_physical_bounds(
    meta: dict[str, Any],
    reference_fields: dict[str, np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the Figure-1 photospheric ROI in physical Mm coordinates."""
    nx, ny, _nz, dx, dy, dz = meta["dims"]
    x_bounds = (float(meta["x_bounds"][0]), float(meta["x_bounds"][1]))
    y_bounds = (float(meta["y_bounds"][0]), float(meta["y_bounds"][1]))
    metric_slices = metric_slices_from_meta(meta)
    plot_slices = strong_current_plot_slices(
        reference_fields,
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        metric_slices=metric_slices,
        ny=int(ny),
        nx=int(nx),
    )
    if plot_slices is None:
        return x_bounds, y_bounds
    _sz, sy, sx = plot_slices
    xs = np.linspace(x_bounds[0], x_bounds[1], int(nx), endpoint=True)
    ys = np.linspace(y_bounds[0], y_bounds[1], int(ny), endpoint=True)
    y0 = 0 if sy.start is None else int(sy.start)
    y1 = int(ny) - 1 if sy.stop is None else int(sy.stop) - 1
    x0 = 0 if sx.start is None else int(sx.start)
    x1 = int(nx) - 1 if sx.stop is None else int(sx.stop) - 1
    return (float(xs[x0]), float(xs[x1])), (float(ys[y0]), float(ys[y1]))


def narrow_centered_interval(
    bounds: tuple[float, float],
    *,
    scale: float,
) -> tuple[float, float]:
    """Return a center-fixed sub-interval whose width is ``scale`` times the original."""
    lo, hi = bounds
    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo) * float(scale)
    return (center - half_width, center + half_width)


def y_bounds_for_equal_aspect(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Shrink ``y_bounds`` so its Mm span matches ``x_bounds`` for equal-aspect maps."""
    x_extent = float(x_bounds[1] - x_bounds[0])
    y_extent = float(y_bounds[1] - y_bounds[0])
    if y_extent <= x_extent + 1.0e-12:
        return y_bounds
    center = 0.5 * (y_bounds[0] + y_bounds[1])
    half = 0.5 * x_extent
    return (center - half, center + half)


def _index_bounds_from_physical(
    bounds: tuple[float, float],
    grid_bounds: tuple[float, float],
    count: int,
) -> tuple[int, int]:
    coords = np.linspace(grid_bounds[0], grid_bounds[1], int(count), endpoint=True)
    i0 = int(np.argmin(np.abs(coords - bounds[0])))
    i1 = int(np.argmin(np.abs(coords - bounds[1])))
    return min(i0, i1), max(i0, i1)


def physical_plane_window(
    meta: dict[str, Any],
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[slice, slice, tuple[float, float, float, float]]:
    """Return ``(sy, sx, extent)`` for a fixed bottom-plane ``(x, y)`` ROI."""
    nx, ny, _nz, _dx, _dy, _dz = meta["dims"]
    x_grid = (float(meta["x_bounds"][0]), float(meta["x_bounds"][1]))
    y_grid = (float(meta["y_bounds"][0]), float(meta["y_bounds"][1]))
    y0, y1 = _index_bounds_from_physical(y_bounds, y_grid, int(ny))
    x0, x1 = _index_bounds_from_physical(x_bounds, x_grid, int(nx))
    extent = (
        float(x_bounds[0]),
        float(x_bounds[1]),
        float(y_bounds[0]),
        float(y_bounds[1]),
    )
    return slice(y0, y1 + 1), slice(x0, x1 + 1), extent


def bgt_plot_equal_aspect_window(
    meta: dict[str, Any],
    reference_fields: dict[str, np.ndarray],
) -> tuple[slice, slice, tuple[float, float, float, float]]:
    """Return ``(sy, sx, extent)`` for Figure-1-style x crop with matched y span."""
    nx, ny, _nz, dx, dy, dz = meta["dims"]
    x_grid = (float(meta["x_bounds"][0]), float(meta["x_bounds"][1]))
    y_grid = (float(meta["y_bounds"][0]), float(meta["y_bounds"][1]))
    plot_x, plot_y = bgt_plot_physical_bounds(meta, reference_fields)
    plot_y = y_bounds_for_equal_aspect(plot_x, plot_y)
    metric_slices = metric_slices_from_meta(meta)
    plot_slices = strong_current_plot_slices(
        reference_fields,
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        metric_slices=metric_slices,
        ny=int(ny),
        nx=int(nx),
    )
    if plot_slices is None:
        y0, y1 = _index_bounds_from_physical(plot_y, y_grid, int(ny))
        x0, x1 = _index_bounds_from_physical(plot_x, x_grid, int(nx))
        return slice(y0, y1 + 1), slice(x0, x1 + 1), (
            plot_x[0],
            plot_x[1],
            plot_y[0],
            plot_y[1],
        )
    _sz, _sy, sx = plot_slices
    y0, y1 = _index_bounds_from_physical(plot_y, y_grid, int(ny))
    x0, x1 = _index_bounds_from_physical(plot_x, x_grid, int(nx))
    return slice(y0, y1 + 1), slice(x0, x1 + 1), (
        plot_x[0],
        plot_x[1],
        plot_y[0],
        plot_y[1],
    )


def bgt_plot_twist_y0_window(
    meta: dict[str, Any] | None = None,
    reference_fields: dict[str, np.ndarray] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the fixed Figure-5 ``(xlim, zlim)`` for y=0 twist maps."""
    del meta, reference_fields
    return (-75.0, 75.0), (0.0, 150.0)


def bgt_plot_fig1_bottom_plane_window(
    meta: dict[str, Any],
    reference_fields: dict[str, np.ndarray],
) -> tuple[slice, slice, tuple[float, float, float, float]]:
    """Return ``(sy, sx, extent)`` for Figure-4 bottom-plane maps (Figure-1 ROI)."""
    plot_x, plot_y = bgt_plot_physical_bounds(meta, reference_fields)
    return physical_plane_window(meta, x_bounds=plot_x, y_bounds=plot_y)


def bgt_plot_x_window(
    meta: dict[str, Any],
    reference_fields: dict[str, np.ndarray],
) -> tuple[slice, tuple[float, float], tuple[float, float]]:
    """Return Figure-1 x crop with the full photospheric y extent."""
    sy, sx, extent = bgt_plot_equal_aspect_window(meta, reference_fields)
    return sx, (extent[0], extent[1]), (extent[2], extent[3])


# Shared matplotlib quiver visual parameters (must match the legend arrows).
# Smaller scale => longer arrows (journal-readable).
QUIVER_SCALE = 12.0
QUIVER_WIDTH = 0.0045
QUIVER_HEADWIDTH = 3.2
QUIVER_HEADLENGTH = 4.0
QUIVER_ALPHA = 0.85


def _quiver_stride(ny: int, nx: int, quiver_stride_target: int) -> int:
    target = max(4, int(quiver_stride_target))
    return max(6, int(np.ceil(max(nx, ny) / target)))


def _nice_magnitude(value: float) -> float:
    """Round ``value`` to a journal-friendly 1--2 significant figure."""
    v = float(abs(value))
    if not np.isfinite(v) or v <= 0.0:
        return 1.0e-30
    exp = int(np.floor(np.log10(v)))
    mant = v / (10.0**exp)
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if mant <= candidate * 1.05:
            return float(candidate * (10.0**exp))
    return float(10.0 ** (exp + 1))


def _format_magnitude(value: float) -> str:
    """Compact decimal / scientific label for reference-arrow captions."""
    v = float(abs(value))
    if v == 0.0:
        return "0"
    if 1.0e-3 <= v < 100.0:
        text = f"{v:.2g}"
    else:
        text = f"{v:.1e}"
    return text.replace("e-0", "e-").replace("e+0", "e+").replace("e+", "e")


def _transposed_extent(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Map ``(x, y)`` ROI bounds to imshow extent with axes swapped."""
    return (y_bounds[0], y_bounds[1], x_bounds[0], x_bounds[1])


def _draw_quiver(
    ax,
    field: np.ndarray,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    stride: int,
    scale_norm: float,
    transpose: bool = False,
) -> None:
    """Draw horizontal arrows with length ∝ |B_h| / ``scale_norm``."""
    ny, nx = field.shape[1:]
    ys = np.linspace(y_bounds[0], y_bounds[1], ny, endpoint=True)
    xs = np.linspace(x_bounds[0], x_bounds[1], nx, endpoint=True)
    if transpose:
        yy, xx = np.meshgrid(ys[::stride], xs[::stride], indexing="ij")
        ax.quiver(
            yy,
            xx,
            field[1, ::stride, ::stride] / scale_norm,
            field[0, ::stride, ::stride] / scale_norm,
            color="k",
            scale=QUIVER_SCALE,
            width=QUIVER_WIDTH,
            headwidth=QUIVER_HEADWIDTH,
            headlength=QUIVER_HEADLENGTH,
            alpha=QUIVER_ALPHA,
            pivot="mid",
        )
        return
    yy, xx = np.meshgrid(ys[::stride], xs[::stride], indexing="ij")
    ax.quiver(
        xx,
        yy,
        field[0, ::stride, ::stride] / scale_norm,
        field[1, ::stride, ::stride] / scale_norm,
        color="k",
        scale=QUIVER_SCALE,
        width=QUIVER_WIDTH,
        headwidth=QUIVER_HEADWIDTH,
        headlength=QUIVER_HEADLENGTH,
        alpha=QUIVER_ALPHA,
        pivot="mid",
    )


def _draw_scale_legend(
    ax,
    *,
    bh_ref: float,
    dbh_ref: float,
    legend_fontsize: float = FIG_LEGEND_FONTSIZE,
    caption_fontsize: float = FIG_TICK_FONTSIZE,
) -> None:
    """Two equal-length reference arrows for row-1 and row-2 quiver scales.

    Each arrow is a unit quiver vector (same on-figure length as a typical
    map arrow after ``/scale_norm`` on that row), labeled with the physical
    magnitude that length represents.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Equal visual length: U=1 matches |B_h|=bh_ref on row 1 and
    # |ΔB_h|=dbh_ref on row 2 after each row's own scale_norm.
    y_top, y_bot = 0.62, 0.32
    x0 = 0.10
    ax.quiver(
        [x0, x0],
        [y_top, y_bot],
        [1.0, 1.0],
        [0.0, 0.0],
        color="k",
        scale=QUIVER_SCALE,
        width=QUIVER_WIDTH * 1.15,
        headwidth=QUIVER_HEADWIDTH,
        headlength=QUIVER_HEADLENGTH,
        alpha=1.0,
        pivot="tail",
        clip_on=False,
        zorder=3,
    )
    tip = x0 + 1.0 / QUIVER_SCALE
    ax.text(
        min(0.96, tip + 0.05),
        y_top,
        rf"$|\mathbf{{B}}_h^{{>}}|={_format_magnitude(bh_ref)}$",
        ha="left",
        va="center",
        fontsize=legend_fontsize,
        transform=ax.transAxes,
    )
    ax.text(
        min(0.96, tip + 0.05),
        y_bot,
        rf"$|\Delta\mathbf{{B}}_h^{{>}}|={_format_magnitude(dbh_ref)}$",
        ha="left",
        va="center",
        fontsize=legend_fontsize,
        transform=ax.transAxes,
    )

def _bgt_row_scales(
    panels: list[tuple[str, np.ndarray, str | None]],
) -> tuple[
    list[np.ndarray],
    float,
    float,
    float,
    float,
]:
    """Return diffs plus B_z / quiver scales for one TD block."""
    if not panels:
        raise ValueError("panels must be nonempty")
    _ref_title, ref_field, _ref_delta = panels[0]
    sol_panels = panels[1:]
    fields = [field for _t, field, _d in panels]
    bz_abs = float(
        np.nanpercentile(np.abs(np.concatenate([f[2].ravel() for f in fields])), 99.0)
    )
    bz_abs = max(bz_abs, 1.0e-12)
    bh_scale = float(
        np.nanpercentile(
            np.concatenate([np.hypot(f[0], f[1]).ravel() for f in fields]),
            95.0,
        )
    )
    bh_scale = max(bh_scale, 1.0e-30)

    diffs = [field - ref_field for _t, field, _d in sol_panels]
    if diffs:
        dbz_abs = float(
            np.nanpercentile(np.abs(np.concatenate([d[2].ravel() for d in diffs])), 99.0)
        )
        dbz_abs = max(dbz_abs, 1.0e-12)
        dbh_scale = max(
            float(np.nanpercentile(np.hypot(d[0], d[1]), 95.0)) for d in diffs
        )
        dbh_scale = max(dbh_scale, 1.0e-30)
    else:
        dbz_abs = 1.0e-12
        dbh_scale = 1.0e-30

    bh_ref = _nice_magnitude(bh_scale)
    dbh_ref = _nice_magnitude(dbh_scale)
    return diffs, bz_abs, dbz_abs, bh_ref, dbh_ref


def _configure_map_axis_labels(
    ax,
    *,
    show_xy_axes: bool,
    x_label: str,
    y_label: str,
    label_fontsize: float,
) -> None:
    if show_xy_axes:
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        ax.tick_params(
            axis="both",
            labelsize=label_fontsize * (FIG_TICK_FONTSIZE / FIG_AXIS_LABEL_FONTSIZE),
            length=3.0,
            width=0.6,
        )
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _draw_bgt_block_on_axes(
    axes: np.ndarray,
    *,
    fig,
    panels: list[tuple[str, np.ndarray, str | None]],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    quiver_stride_target: int,
    title_fontsize: float = FIG_TITLE_FONTSIZE,
    label_fontsize: float = FIG_AXIS_LABEL_FONTSIZE,
    cbar_fontsize: float = FIG_CBAR_LABEL_FONTSIZE,
    cax_top=None,
    cax_bot=None,
    title_pad: float = 4.0,
) -> None:
    """Draw one TD B^> / ΔB^> block onto a ``(2, n_cols)`` axes array.

    If ``cax_top`` / ``cax_bot`` are given, colorbars use those axes (keeps
    data columns column-aligned).  Otherwise colorbars squeeze the row axes.
    """
    diffs, bz_abs, dbz_abs, bh_ref, dbh_ref = _bgt_row_scales(panels)
    sol_panels = panels[1:]
    extent = _transposed_extent(x_bounds, y_bounds)
    x_label = r"$y$"
    y_label = r"$x$"
    im_top = None
    im_bot = None

    for col, (title, field, _delta_title) in enumerate(panels):
        ax = axes[0, col]
        ny, nx = field.shape[1:]
        im_top = ax.imshow(
            field[2].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-bz_abs,
            vmax=bz_abs,
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        stride = _quiver_stride(ny, nx, quiver_stride_target)
        _draw_quiver(
            ax,
            field,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            stride=stride,
            scale_norm=bh_ref,
            transpose=True,
        )
        ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
        _configure_map_axis_labels(
            ax,
            show_xy_axes=(col == 0),
            x_label=x_label,
            y_label=y_label,
            label_fontsize=label_fontsize,
        )

    axes[1, 0].axis("off")

    for col, ((_top_title, _field, delta_title), diff) in enumerate(
        zip(sol_panels, diffs, strict=True), start=1
    ):
        ax = axes[1, col]
        ny, nx = diff.shape[1:]
        im_bot = ax.imshow(
            diff[2].T,
            origin="lower",
            cmap="PiYG",
            vmin=-dbz_abs,
            vmax=dbz_abs,
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        stride = _quiver_stride(ny, nx, quiver_stride_target)
        _draw_quiver(
            ax,
            diff,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            stride=stride,
            scale_norm=dbh_ref,
            transpose=True,
        )
        ax.set_title(
            delta_title if delta_title is not None else "",
            fontsize=title_fontsize,
            pad=title_pad,
        )
        _configure_map_axis_labels(
            ax,
            show_xy_axes=False,
            x_label=x_label,
            y_label=y_label,
            label_fontsize=label_fontsize,
        )

    if im_top is not None:
        if cax_top is not None:
            cbar_top = fig.colorbar(im_top, cax=cax_top)
        else:
            cbar_top = fig.colorbar(
                im_top, ax=axes[0, :].tolist(), fraction=0.035, pad=0.02
            )
        cbar_top.set_label(r"$B_z^{>}$", fontsize=cbar_fontsize)
        cbar_top.ax.tick_params(
            labelsize=cbar_fontsize * (FIG_TICK_FONTSIZE / FIG_CBAR_LABEL_FONTSIZE)
        )
    if im_bot is not None:
        if cax_bot is not None:
            cbar_bot = fig.colorbar(im_bot, cax=cax_bot)
        else:
            cbar_bot = fig.colorbar(
                im_bot, ax=axes[1, 1:].tolist(), fraction=0.035, pad=0.02
            )
        cbar_bot.set_label(r"$\Delta B_z^{>}$", fontsize=cbar_fontsize)
        cbar_bot.ax.tick_params(
            labelsize=cbar_fontsize * (FIG_TICK_FONTSIZE / FIG_CBAR_LABEL_FONTSIZE)
        )


def plot_bgt_with_diff(
    *,
    display: str,
    panels: list[tuple[str, np.ndarray, str | None]],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    output_path: Path,
    quiver_stride_target: int,
) -> None:
    """Two-row B^> figure: fields on top, NFGS−reference ΔB_z^> below.

    Each panel is ``(top_title, field, delta_title_or_None)``.  The reference
    column has ``delta_title=None`` and leaves the bottom cell as a legend.
    """
    n_cols = len(panels)
    x_extent = max(x_bounds[1] - x_bounds[0], 1.0e-30)
    y_extent = max(y_bounds[1] - y_bounds[0], 1.0e-30)
    panel_height = 3.35 * (x_extent / y_extent)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(3.35 * n_cols + 1.6, 2.0 * panel_height + 1.4),
        constrained_layout=True,
        squeeze=False,
    )
    _draw_bgt_block_on_axes(
        axes,
        fig=fig,
        panels=panels,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        quiver_stride_target=quiver_stride_target,
        title_pad=6.0,
    )
    fig.suptitle(display, fontsize=FIG_TITLE_FONTSIZE, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_bgt_summary(
    cases: list[dict[str, Any]],
    *,
    output_path: Path,
    quiver_stride_target: int,
) -> None:
    """Paper composite: stacked TD1--TD3 B^> / ΔB^> robin-weight panels.

    Layout::

        (a) TD1   2 x 4  (reference | rho | rho | rho ; legend | Erel...)
        (b) TD2   2 x 4
        (c) TD3   2 x 4

    Colorbar columns are reserved so data panels stay column-aligned, and
    row-1 / row-2 ``hspace`` is large enough that top ``x`` labels do not
    collide with bottom subtitles.
    """
    from matplotlib.gridspec import GridSpec

    if not cases:
        raise ValueError("cases must be nonempty")
    n_cols = len(cases[0]["panels"])
    for case in cases:
        if len(case["panels"]) != n_cols:
            raise ValueError("all cases must share the same panel count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_blocks = len(cases)
    # Landscape panels: height scales with the transposed x/y aspect of each block.
    block_heights = []
    for case in cases:
        x_bounds = case["x_bounds"]
        y_bounds = case["y_bounds"]
        x_extent = max(float(x_bounds[1] - x_bounds[0]), 1.0e-30)
        y_extent = max(float(y_bounds[1] - y_bounds[0]), 1.0e-30)
        block_heights.append(3.15 * (x_extent / y_extent))
    block_height = max(block_heights)
    fig_width = 3.15 * n_cols + 2.2
    fig_height = 2.0 * block_height * n_blocks + 0.8 * n_blocks
    # The summary PNG is re-embedded at \\textwidth, so scale type to Figure 4.
    font_scale = font_scale_to_match_figure4(fig_width, width_frac=1.0)
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = GridSpec(
        n_blocks,
        1,
        figure=fig,
        hspace=0.16,
        left=0.11,
        right=0.98,
        top=0.985,
        bottom=0.02,
    )
    panel_letters = ("a", "b", "c", "d", "e", "f")

    for block_i, case in enumerate(cases):
        letter = panel_letters[block_i] if block_i < len(panel_letters) else str(block_i + 1)
        display = str(case["display"])
        block = outer[block_i].subgridspec(1, 1)
        inner = block[0, 0].subgridspec(
            2,
            n_cols + 1,
            width_ratios=[1.0] * n_cols + [0.05],
            hspace=0.42,
            wspace=0.10,
        )
        axes = np.empty((2, n_cols), dtype=object)
        for i in range(2):
            for j in range(n_cols):
                axes[i, j] = fig.add_subplot(inner[i, j])
        cax_top = fig.add_subplot(inner[0, n_cols])
        cax_bot = fig.add_subplot(inner[1, n_cols])
        _draw_bgt_block_on_axes(
            axes,
            fig=fig,
            panels=case["panels"],
            x_bounds=case["x_bounds"],
            y_bounds=case["y_bounds"],
            quiver_stride_target=quiver_stride_target,
            title_fontsize=FIG_TITLE_FONTSIZE * font_scale,
            label_fontsize=FIG_AXIS_LABEL_FONTSIZE * font_scale,
            cbar_fontsize=FIG_CBAR_LABEL_FONTSIZE * font_scale,
            cax_top=cax_top,
            cax_bot=cax_bot,
            title_pad=10.0,
        )
        # Keep the case label in the left margin so it does not meet "ref."
        label_pos = axes[0, 0].get_position()
        fig.text(
            label_pos.x0 - 0.008,
            label_pos.y1,
            f"({letter}) {display}",
            ha="right",
            va="bottom",
            fontsize=FIG_PANEL_LETTER_FONTSIZE * font_scale,
            fontweight="bold",
            clip_on=False,
        )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.weights:
        weights = [float(item) for item in args.weights.split(",") if item.strip()]
    else:
        weights = list(DEFAULT_WEIGHTS)
    requested = [
        item.strip().lower() for item in args.benchmarks.split(",") if item.strip()
    ]
    device = analytic.resolve_device(args.device)
    dtype = analytic.dtype_from_name(args.dtype)
    metrics_csv = args.metrics_csv or (args.sweep_dir / "metrics_with_erel_bgt.csv")
    erel_lookup = load_erel_lookup(metrics_csv)
    print(f"device={device} dtype={dtype}", flush=True)
    print(f"weights={weights} benchmarks={requested}", flush=True)
    print(
        f"quiver_stride_target={args.quiver_stride_target} metrics_csv={metrics_csv}",
        flush=True,
    )

    written: list[Path] = []
    combined_cases: list[dict[str, Any]] = []
    for name in requested:
        meta = load_case_meta(args.case_cache_dir, name)
        nx, ny, _nz, dx, dy, dz = meta["dims"]
        display = str(meta.get("display", name.upper()))
        x_bounds = (float(meta["x_bounds"][0]), float(meta["x_bounds"][1]))
        y_bounds = (float(meta["y_bounds"][0]), float(meta["y_bounds"][1]))
        solver = make_bgt_helper_solver(
            nx=int(nx),
            ny=int(ny),
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
            device=device,
            dtype=dtype,
            args=args,
        )

        ref_path = reference_field_cache_path(args.sweep_dir, name)
        if not ref_path.is_file():
            # Fallback to analytic case-cache reference if sweep cache is absent.
            alt = args.case_cache_dir / f"{name}__reference.npz"
            if not alt.is_file():
                raise FileNotFoundError(ref_path)
            ref_path = alt
        ref_fields = load_field_cache(ref_path)
        ref_bgt = bottom_plane_bgt_nfgs(solver, ref_fields)
        # Crop every TD panel to a plot ROI that frames strong |J_z| with margin.
        plot_x_bounds = x_bounds
        plot_y_bounds = y_bounds
        metric_slices = metric_slices_from_meta(meta)
        plot_slices = strong_current_plot_slices(
            ref_fields,
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
            metric_slices=metric_slices,
            ny=int(ny),
            nx=int(nx),
        )
        if plot_slices is not None:
            ref_bgt, plot_x_bounds, plot_y_bounds = crop_bgt_to_metric_roi(
                ref_bgt,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                metric_slices=plot_slices,
            )
        print(
            f"  {name} reference "
            f"Bgt shape={ref_bgt.shape} "
            f"|Bz|_p99={np.nanpercentile(np.abs(ref_bgt[2]), 99):.4g}",
            flush=True,
        )

        panels: list[tuple[str, np.ndarray, str | None]] = [
            ("ref.", ref_bgt, None)
        ]
        for weight in weights:
            path = nfgs_field_cache_path(
                args.sweep_dir, name, args.init_mode, weight
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            fields = load_field_cache(path)
            bgt = bottom_plane_bgt_nfgs(solver, fields)
            if plot_slices is not None:
                bgt, _, _ = crop_bgt_to_metric_roi(
                    bgt,
                    x_bounds=x_bounds,
                    y_bounds=y_bounds,
                    metric_slices=plot_slices,
                )
            key = (name, round(float(weight), 1))
            if key not in erel_lookup:
                raise KeyError(f"Missing {EREL_KEY} for {key} in {metrics_csv}")
            erel = float(erel_lookup[key])
            panels.append((rho_panel_title(weight), bgt, erel_panel_title(erel)))
            print(
                f"  {name} robin_w{weight_tag(weight)} "
                f"Bgt shape={bgt.shape} "
                f"|Bz|_p99={np.nanpercentile(np.abs(bgt[2]), 99):.4g} "
                f"{EREL_KEY}={erel:.4g}",
                flush=True,
            )

        combined_cases.append(
            {
                "name": name,
                "display": display,
                "panels": panels,
                "x_bounds": plot_x_bounds,
                "y_bounds": plot_y_bounds,
            }
        )

        out_name = (
            f"figure_nfgs_td_bgt_robin_"
            f"{'_'.join('w' + weight_tag(w) for w in weights)}_{name}.png"
        )
        for out_dir in (args.sweep_dir, args.figures_dir):
            out_path = out_dir / out_name
            plot_bgt_with_diff(
                display=display,
                panels=panels,
                x_bounds=plot_x_bounds,
                y_bounds=plot_y_bounds,
                output_path=out_path,
                quiver_stride_target=args.quiver_stride_target,
            )
            written.append(out_path)
            print(f"figure={out_path}", flush=True)

        del solver
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if len(combined_cases) >= 2:
        summary_name = (
            f"figure_nfgs_td_bgt_robin_"
            f"{'_'.join('w' + weight_tag(w) for w in weights)}_summary.png"
        )
        for out_dir in (args.sweep_dir, args.figures_dir):
            out_path = out_dir / summary_name
            plot_combined_bgt_summary(
                combined_cases,
                output_path=out_path,
                quiver_stride_target=args.quiver_stride_target,
            )
            written.append(out_path)
            print(f"summary={out_path}", flush=True)

    for path in written:
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
