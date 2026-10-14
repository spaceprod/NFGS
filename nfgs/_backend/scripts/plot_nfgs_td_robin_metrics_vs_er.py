#!/usr/bin/env python3
"""Compute bottom-plane E^{rel}_{B^>} and plot robin-weight sweep metrics.

E^{rel}_{B^>} = (1/N) sum |b^> - B^>| / |b^>| on the photosphere, where b^>
and B^> are the NFGS endpoint-Neumann (DCT) Gaussian-separated above-plane
components of the analytic reference and NFGS reconstruction bottoms.
Pixels with |b^>| below a tiny floor are excluded.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import gc
import json
import sys
import time
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
    FIG_ANNOTATION_FONTSIZE,
    FIG_AXIS_LABEL_FONTSIZE,
    FIG_CBAR_LABEL_FONTSIZE,
    FIG_LEGEND_FONTSIZE,
    FIG_PANEL_LETTER_FONTSIZE,
    FIG_TICK_FONTSIZE,
    FIG_TITLE_FONTSIZE,
    font_scale_to_match_figure4,
)

from nfgs._backend.scripts import run_analytic_validation as analytic  # noqa: E402
from nfgs._backend.scripts import run_nfgs_analytic_validation as driver  # noqa: E402
from nfgs._backend.scripts.run_nfpt_fortran_symmetric_td_reconstruction import (  # noqa: E402
    paper_metric_domain,
    paper_metric_slices,
)
from nfgs._backend.scripts.run_nfgs_td_robin_weight_sweep import (  # noqa: E402
    DEFAULT_WEIGHTS,
    ensure_reference_field_cache,
    load_field_cache,
    make_run_args,
    nfgs_field_cache_path,
    parse_weights,
)
from nfgs_solver import NFGSSolverGPU  # noqa: E402
from nfgs._backend.util.metrics import compute_nf2_metrics  # noqa: E402

METRIC_KEYS = ["C_vec", "C_CS", "Eprime_n", "Eprime_m", "epsilon", "sigma_J"]
METRIC_LABELS = {
    "C_vec": r"$C_{\mathrm{vec}}$",
    "C_CS": r"$C_{\mathrm{CS}}$",
    "Eprime_n": r"$E'_n$",
    "Eprime_m": r"$E'_m$",
    "epsilon": r"$\varepsilon$",
    "sigma_J": r"$\sigma_J$",
}
# Metrics where larger values are preferred; sigma_J is minimized.
METRIC_HIGHER_IS_BETTER = {
    "C_vec",
    "C_CS",
    "Eprime_n",
    "Eprime_m",
    "epsilon",
}
# Ideal reference point on each metrics-vs-Erel panel: (E_rel, metric).
METRIC_IDEAL_POINT = {
    "C_vec": (0.0, 1.0),
    "C_CS": (0.0, 1.0),
    "Eprime_n": (0.0, 1.0),
    "Eprime_m": (0.0, 1.0),
    "epsilon": (0.0, 1.0),
    "sigma_J": (0.0, 0.0),
}
EREL_KEY = "E_rel_Bgt"
EREL_LABEL = r"$E^{\mathrm{rel}}_{B^{>}}$"
RHO_KEY = "robin_weight"
RHO_LABEL = r"$\rho$"
NFPT_RHO_ANCHOR = 1.06
LINESTYLES = {"td1": "-", "td2": "--", "td3": ":"}
MARKERS = {"td1": "o", "td2": "s", "td3": "^"}
NFPT_COLOR = "cyan"
NFPT_MARKER = "x"
NFPT_MARKER_SIZE = 90
NFPT_LINEWIDTH = 1.8
# Nz is unused by the bottom DCT split; keep a small stack for solver ctor.
_BGT_HELPER_NZ = 8


def nfpt_legend_handle() -> Any:
    """Legend proxy for the cyan NFPT marker."""
    return plt.Line2D(
        [],
        [],
        linestyle="none",
        marker=NFPT_MARKER,
        markersize=8,
        color=NFPT_COLOR,
        markeredgewidth=NFPT_LINEWIDTH,
        label="NFPT",
    )


def overlay_nfpt_metric_point(
    ax,
    *,
    x: float,
    y: float,
    label: bool = False,
) -> None:
    """Mark one NFPT metric point with a cyan ``x``."""
    ax.scatter(
        [float(x)],
        [float(y)],
        marker=NFPT_MARKER,
        s=NFPT_MARKER_SIZE,
        c=NFPT_COLOR,
        linewidths=NFPT_LINEWIDTH,
        zorder=5,
        label="NFPT" if label else None,
        clip_on=False,
    )


def load_nfpt_metrics(path: Path) -> dict[str, dict[str, Any]]:
    """Load ``nfpt_fortran_metrics.csv`` keyed by benchmark."""
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open() as stream:
        for row in csv.DictReader(stream):
            bench = str(row["benchmark"]).lower()
            parsed = dict(row)
            for name in [*METRIC_KEYS, "best_sigma", "best_iter", "n_iters", "wall_time_s", EREL_KEY]:
                if name in parsed and parsed[name] not in (None, ""):
                    parsed[name] = float(parsed[name])
            out[bench] = parsed
    return out


def write_nfpt_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "benchmark",
        "solver",
        *METRIC_KEYS,
        "best_sigma",
        "best_iter",
        "n_iters",
        "stop_reason",
        "wall_time_s",
        "init_mode",
        "metric_domain",
        EREL_KEY,
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: str(r["benchmark"])):
            writer.writerow({key: row.get(key) for key in keys})


def ideal_metric_point(metric: str) -> tuple[float, float]:
    """Return the ideal ``(E_rel, metric)`` coordinate for ``metric``."""
    if metric not in METRIC_IDEAL_POINT:
        raise KeyError(f"No ideal point defined for metric={metric}")
    return METRIC_IDEAL_POINT[metric]


def shared_metric_axis_limits(
    rows: list[dict[str, Any]],
    *,
    benchmarks: tuple[str, ...] = ("td1", "td2", "td3"),
    margin_frac: float = 0.1,
    nfpt_by_bench: dict[str, dict[str, Any]] | None = None,
    x_key: str = EREL_KEY,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Per-metric axis limits unified across TD1--TD3.

    Limits include the ideal point and use a relative margin
    ``0.1 * (max - min)``.  For higher-is-better metrics the y-axis maximum
    is fixed at 1.
    """
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for metric in METRIC_KEYS:
        xs: list[float] = []
        ys: list[float] = []
        for bench in benchmarks:
            group = [r for r in rows if r["benchmark"] == bench]
            for row in group:
                xs.append(float(row[x_key]))
                ys.append(float(row[metric]))
            if nfpt_by_bench and bench in nfpt_by_bench:
                nfpt = nfpt_by_bench[bench]
                if x_key == RHO_KEY:
                    xs.append(NFPT_RHO_ANCHOR)
                elif EREL_KEY in nfpt and nfpt[EREL_KEY] not in (None, ""):
                    xs.append(float(nfpt[EREL_KEY]))
                if metric in nfpt and nfpt[metric] not in (None, ""):
                    ys.append(float(nfpt[metric]))
        if not xs:
            continue
        x_ideal, y_ideal = ideal_metric_point(metric)
        if x_key == RHO_KEY:
            x_ideal = 1.0
        xs.append(x_ideal)
        ys.append(y_ideal)
        x0 = float(np.min(xs))
        x1 = float(np.max(xs))
        y0 = float(np.min(ys))
        y1 = float(np.max(ys))
        dx = max(x1 - x0, 1.0e-12)
        dy = max(y1 - y0, 1.0e-12)
        x_pad = margin_frac * dx
        y_pad = margin_frac * dy
        if metric in METRIC_HIGHER_IS_BETTER:
            y_top = 1.0
        else:
            y_top = y1 + y_pad
        out[metric] = {
            "xlim": (x0 - x_pad, x1 + x_pad),
            "ylim": (y0 - y_pad, y_top),
        }
    return out


def make_bgt_helper_solver(
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dz: float,
    device: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> NFGSSolverGPU:
    """Build a lightweight NFGS instance used only for DCT ``B^>`` separation."""
    return NFGSSolverGPU(
        int(nx),
        int(ny),
        int(_BGT_HELPER_NZ),
        float(dx),
        float(dy),
        float(dz),
        device=device,
        dtype=dtype,
        mean_bz_policy=args.mean_bz_policy,
        horizontal_zero_mode_policy=args.horizontal_zero_mode_policy,
        d0_zero_mode_policy=args.d0_zero_mode_policy,
        source_zero_mode_policy=args.source_zero_mode_policy,
        harmonic_residual_policy=args.harmonic_residual_policy,
        mean_mode_tolerance=args.mean_mode_tolerance,
        source_zero_mode_tolerance=args.source_zero_mode_tolerance,
        bottom_divergence_projection_weight=0.0,
        bottom_divergence_robin_weight=0.0,
        buffer_thickness=0,
        padded_buffer_zone=False,
        pad_calculation_domain=False,
        lateral_wall="paper_bc1",
        pin_potential_boundaries=False,
        gaussian_horizontal_operator=getattr(
            args, "gaussian_horizontal_operator", "spectral"
        ),
    )


def bottom_plane_bgt_nfgs(
    solver: NFGSSolverGPU,
    fields: dict[str, np.ndarray],
) -> np.ndarray:
    """Return NFGS DCT ``B^>`` on z=0 with shape ``(3, ny, nx)``."""
    solver.set_bottom_vector_boundary(
        np.asarray(fields["Bx"][0], dtype=np.float64),
        np.asarray(fields["By"][0], dtype=np.float64),
        np.asarray(fields["Bz"][0], dtype=np.float64),
        zero_net_current=False,
        smooth_passes=0,
        ijzbcor=3,
    )
    with torch.inference_mode():
        bgt = solver.gaussian_above_bottom_field()
    return np.asarray(bgt.detach().cpu(), dtype=np.float64)


def bottom_bgt_relative_error(
    solver: NFGSSolverGPU,
    ref_fields: dict[str, np.ndarray],
    sol_fields: dict[str, np.ndarray],
    *,
    floor_frac: float = 1.0e-8,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> float:
    """E^{rel}_{B^>} = mean(|b^> - B^>| / |b^>|) on z=0 (NFGS DCT split).

    When ``y_slice`` / ``x_slice`` are given, the mean is restricted to that
    photospheric ROI (typically the horizontal part of ``metric_slices``).
    """
    b = bottom_plane_bgt_nfgs(solver, ref_fields)
    B = bottom_plane_bgt_nfgs(solver, sol_fields)
    sy = slice(None) if y_slice is None else y_slice
    sx = slice(None) if x_slice is None else x_slice
    b = b[:, sy, sx]
    B = B[:, sy, sx]
    b_mag = np.linalg.norm(b, axis=0)
    floor = float(floor_frac) * float(np.max(b_mag) + 1.0e-30)
    mask = b_mag > floor
    if not np.any(mask):
        raise RuntimeError(
            "No valid bottom pixels for E^{rel}_{B^>} (|b^>| below floor)."
        )
    diff = np.linalg.norm(b - B, axis=0)
    return float(np.mean(diff[mask] / b_mag[mask]))


def fmt_sig2(value: float) -> str:
    """Format ``value`` with two significant figures."""
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    return f"{value:.2g}"


def selected_annotation_indices(
    xs: list[float],
    ys: list[float],
    *,
    metric: str | None = None,
    prefer_higher_y: bool = True,
) -> list[int]:
    """Sparse label set: x-endpoints plus one extremal y point.

    Avoids cluttering near-identical mid-curve values (e.g. ~0.98).
    """
    n = len(xs)
    if n == 0:
        return []
    chosen: set[int] = {int(np.argmin(xs)), int(np.argmax(xs))}
    if metric is not None:
        if metric in METRIC_HIGHER_IS_BETTER:
            chosen.add(int(np.argmax(ys)))
        else:
            chosen.add(int(np.argmin(ys)))
    elif prefer_higher_y:
        chosen.add(int(np.argmax(ys)))
    else:
        chosen.add(int(np.argmin(ys)))
    return sorted(chosen)


def min_max_y_indices(ys: list[float]) -> list[int]:
    """Return indices of the lowest and highest ``y`` values."""
    if not ys:
        return []
    return sorted({int(np.argmin(ys)), int(np.argmax(ys))})


def annotate_selected_points(
    ax,
    xs: list[float],
    ys: list[float],
    indices: list[int],
    *,
    fontsize: float = 7,
    offset: float = 6,
) -> None:
    """Annotate only ``indices`` with two-significant-figure labels."""
    for k, i in enumerate(indices):
        above = k % 2 == 0
        ax.annotate(
            fmt_sig2(ys[i]),
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(0, offset if above else -offset),
            ha="center",
            va="bottom" if above else "top",
            fontsize=fontsize,
            color="0.15",
        )


def load_existing_metrics(path: Path) -> dict[tuple[float, str], dict[str, Any]]:
    rows: dict[tuple[float, str], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (round(float(row["robin_weight"]), 1), row["benchmark"])
            parsed = dict(row)
            for name in METRIC_KEYS + [
                "best_sigma",
                "best_iter",
                "n_iters",
                "wall_time_s",
                EREL_KEY,
                "E_r",
            ]:
                if name in parsed and parsed[name] not in (None, ""):
                    parsed[name] = float(parsed[name])
            parsed["robin_weight"] = float(parsed["robin_weight"])
            rows[key] = parsed
    return rows


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "robin_weight",
        "benchmark",
        EREL_KEY,
        *METRIC_KEYS,
        "best_sigma",
        "best_iter",
        "n_iters",
        "stop_reason",
        "wall_time_s",
        "dtype",
        "init_mode",
        "metric_domain",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(
            rows, key=lambda r: (float(r["robin_weight"]), r["benchmark"])
        ):
            writer.writerow({key: row.get(key) for key in keys})


def plot_metrics_vs_erel(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    benchmark: str,
    nfpt_by_bench: dict[str, dict[str, Any]] | None = None,
) -> None:
    """2x3 metric panels for a single TD configuration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bench = benchmark.lower()
    group = sorted(
        [r for r in rows if r["benchmark"] == bench],
        key=lambda r: float(r[EREL_KEY]),
    )
    if not group:
        raise ValueError(f"No rows for benchmark={benchmark}")

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.08, w_pad=0.04, hspace=0.05, wspace=0.05)
    cmap = plt.cm.viridis
    weights = sorted({float(r["robin_weight"]) for r in group})
    norm = plt.Normalize(min(weights), max(weights))
    xs = [float(r[EREL_KEY]) for r in group]
    cs = [float(r["robin_weight"]) for r in group]
    label = bench.upper()
    nfpt = (nfpt_by_bench or {}).get(bench)

    for ax, metric in zip(axes.ravel(), METRIC_KEYS, strict=True):
        ys = [float(r[metric]) for r in group]
        ax.plot(
            xs,
            ys,
            linestyle=LINESTYLES[bench],
            color="0.45",
            linewidth=1.4,
            zorder=1,
        )
        ax.scatter(
            xs,
            ys,
            c=cs,
            cmap=cmap,
            norm=norm,
            marker=MARKERS[bench],
            s=52,
            edgecolors="k",
            linewidths=0.4,
            zorder=2,
        )
        if nfpt is not None and EREL_KEY in nfpt and metric in nfpt:
            overlay_nfpt_metric_point(
                ax,
                x=float(nfpt[EREL_KEY]),
                y=float(nfpt[metric]),
                label=(metric == METRIC_KEYS[0]),
            )
        annotate_selected_points(
            ax,
            xs,
            ys,
            selected_annotation_indices(xs, ys, metric=metric),
            fontsize=7,
            offset=6,
        )
        ax.set_xlabel(EREL_LABEL)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.25, linewidth=0.6)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.01)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(r"$\rho$")
    cbar.set_ticks(weights)
    if nfpt is not None:
        axes.ravel()[0].legend(
            handles=[nfpt_legend_handle()],
            loc="best",
            fontsize=8,
            framealpha=0.9,
        )

    fig.suptitle(
        rf"{label}: reference metrics vs "
        r"$E^{\mathrm{rel}}_{B^{>}}"
        r"=\frac{1}{N}\sum|\mathbf{b}^{>}-\mathbf{B}^{>}|/|\mathbf{b}^{>}|$",
        y=1.04,
        fontsize=12,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_path_for_benchmark(base: Path, benchmark: str) -> Path:
    """Insert ``_{tdN}`` before the suffix of ``base``."""
    bench = benchmark.lower()
    return base.with_name(f"{base.stem}_{bench}{base.suffix}")


def write_split_figures(
    rows: list[dict[str, Any]],
    *,
    figure_base: Path,
    also_dir: Path | None = None,
    nfpt_by_bench: dict[str, dict[str, Any]] | None = None,
) -> list[Path]:
    written: list[Path] = []
    benches = sorted({str(r["benchmark"]) for r in rows})
    for bench in benches:
        path = figure_path_for_benchmark(figure_base, bench)
        plot_metrics_vs_erel(
            rows, path, benchmark=bench, nfpt_by_bench=nfpt_by_bench
        )
        written.append(path)
        if also_dir is not None:
            alt = also_dir / path.name
            plot_metrics_vs_erel(
                rows, alt, benchmark=bench, nfpt_by_bench=nfpt_by_bench
            )
            written.append(alt)
    return written


def pareto_front_sigmaj_cvec(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Non-dominated set for minimize sigma_J, maximize C_vec.

    Point (s1, c1) dominates (s2, c2) if s1 <= s2, c1 >= c2, and at least one
    inequality is strict.  Returned points are sorted by increasing sigma_J.
    """
    front: list[tuple[float, float]] = []
    for i, (s1, c1) in enumerate(points):
        dominated = False
        for j, (s2, c2) in enumerate(points):
            if i == j:
                continue
            if (s2 <= s1 and c2 >= c1) and (s2 < s1 or c2 > c1):
                dominated = True
                break
        if not dominated:
            front.append((s1, c1))
    front.sort(key=lambda p: (p[0], -p[1]))
    return front


def plot_sigmaj_vs_cvec(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    nfpt_by_bench: dict[str, dict[str, Any]] | None = None,
) -> None:
    """1x3 panels: sigma_J (x) vs C_vec (y) for TD1--TD3 with Pareto fronts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, wspace=0.06)
    cmap = plt.cm.viridis
    all_weights = sorted({float(r["robin_weight"]) for r in rows})
    norm = plt.Normalize(min(all_weights), max(all_weights))
    pareto_handle = None

    for ax, bench in zip(axes, ("td1", "td2", "td3"), strict=True):
        # Sort by robin weight so the polyline follows the sweep path
        # (sigma_J is non-monotonic for some TD cases).
        group = sorted(
            [r for r in rows if r["benchmark"] == bench],
            key=lambda r: float(r["robin_weight"]),
        )
        if not group:
            raise ValueError(f"No rows for benchmark={bench}")
        xs = [float(r["sigma_J"]) for r in group]
        ys = [float(r["C_vec"]) for r in group]
        cs = [float(r["robin_weight"]) for r in group]
        ax.plot(
            xs,
            ys,
            linestyle=LINESTYLES[bench],
            color="0.55",
            linewidth=1.2,
            zorder=1,
            alpha=0.85,
            label="sweep path",
        )
        front = pareto_front_sigmaj_cvec(list(zip(xs, ys)))
        if front:
            fx = [p[0] for p in front]
            fy = [p[1] for p in front]
            ax.plot(
                fx,
                fy,
                color="#c0392b",
                linewidth=2.0,
                linestyle="-",
                marker=MARKERS[bench],
                markersize=7.0,
                markerfacecolor="none",
                markeredgecolor="#c0392b",
                markeredgewidth=1.35,
                zorder=1.5,
            )
            if pareto_handle is None:
                pareto_handle = plt.Line2D(
                    [],
                    [],
                    color="#c0392b",
                    linewidth=2.0,
                    linestyle="-",
                    label=r"Pareto front ($\downarrow\sigma_J$, $\uparrow C_{\mathrm{vec}}$)",
                )
        ax.scatter(
            xs,
            ys,
            c=cs,
            cmap=cmap,
            norm=norm,
            marker=MARKERS[bench],
            s=52,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )
        nfpt = (nfpt_by_bench or {}).get(bench)
        if nfpt is not None and "sigma_J" in nfpt and "C_vec" in nfpt:
            overlay_nfpt_metric_point(
                ax,
                x=float(nfpt["sigma_J"]),
                y=float(nfpt["C_vec"]),
                label=(bench == "td1"),
            )
        annotate_selected_points(
            ax,
            xs,
            ys,
            selected_annotation_indices(xs, ys, prefer_higher_y=True),
            fontsize=7,
            offset=6,
        )
        ax.set_xlabel(METRIC_LABELS["sigma_J"])
        ax.set_ylabel(METRIC_LABELS["C_vec"])
        ax.set_title(bench.upper())
        ax.grid(True, alpha=0.25, linewidth=0.6)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.01)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(r"$\rho$")
    cbar.set_ticks(all_weights)
    if pareto_handle is not None:
        handles = [pareto_handle]
        if nfpt_by_bench:
            handles.append(nfpt_legend_handle())
        axes[0].legend(
            handles=handles,
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
        )
    fig.suptitle(
        r"Reference metrics: $\sigma_J$ vs $C_{\mathrm{vec}}$ "
        r"(Pareto: minimize $\sigma_J$, maximize $C_{\mathrm{vec}}$)",
        y=1.06,
        fontsize=12,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_sigmaj_vs_cvec_figures(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    also_dir: Path | None = None,
    nfpt_by_bench: dict[str, dict[str, Any]] | None = None,
) -> list[Path]:
    name = "figure_nfgs_td_robin_weight_sigmaj_vs_cvec.png"
    written: list[Path] = []
    path = output_dir / name
    plot_sigmaj_vs_cvec(rows, path, nfpt_by_bench=nfpt_by_bench)
    written.append(path)
    if also_dir is not None:
        alt = also_dir / name
        plot_sigmaj_vs_cvec(rows, alt, nfpt_by_bench=nfpt_by_bench)
        written.append(alt)
    return written


def _draw_metrics_vs_erel_on_axes(
    axes: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    cmap,
    norm,
    annotate: bool = True,
    tick_fontsize: float = FIG_TICK_FONTSIZE,
    label_fontsize: float = FIG_AXIS_LABEL_FONTSIZE,
    axis_limits: dict[str, dict[str, tuple[float, float]]] | None = None,
    mark_optimal: bool = True,
    nfpt: dict[str, Any] | None = None,
    x_key: str = EREL_KEY,
    color_points_by_weight: bool = True,
    annotate_min_max_y_only: bool = False,
    nfpt_legend_label: bool = True,
    bottom_xlabel_labelpad: float | None = None,
) -> None:
    """Fill a 2x3 axes block for one TD benchmark (no colorbar / title)."""
    bench = benchmark.lower()
    group = sorted(
        [r for r in rows if r["benchmark"] == bench],
        key=lambda r: float(r[x_key]),
    )
    if not group:
        raise ValueError(f"No rows for benchmark={benchmark}")
    xs = [float(r[x_key]) for r in group]
    cs = [float(r[RHO_KEY]) for r in group]
    x_label = RHO_LABEL if x_key == RHO_KEY else EREL_LABEL
    flat = list(np.asarray(axes).ravel())
    for index, (ax, metric) in enumerate(zip(flat, METRIC_KEYS, strict=True)):
        ys = [float(r[metric]) for r in group]
        ax.plot(
            xs,
            ys,
            linestyle=LINESTYLES[bench],
            color="0.45",
            linewidth=1.2,
            zorder=1,
        )
        scatter_kwargs: dict[str, Any] = {
            "marker": MARKERS[bench],
            "s": 36,
            "edgecolors": "k",
            "linewidths": 0.35,
            "zorder": 2,
        }
        if color_points_by_weight:
            scatter_kwargs.update(
                {"x": xs, "y": ys, "c": cs, "cmap": cmap, "norm": norm}
            )
        else:
            scatter_kwargs.update({"x": xs, "y": ys, "color": "0.25"})
        ax.scatter(**scatter_kwargs)
        if nfpt is not None and metric in nfpt and nfpt[metric] not in (None, ""):
            nfpt_x = (
                NFPT_RHO_ANCHOR
                if x_key == RHO_KEY
                else float(nfpt[EREL_KEY])
                if EREL_KEY in nfpt and nfpt[EREL_KEY] not in (None, "")
                else None
            )
            if nfpt_x is not None:
                overlay_nfpt_metric_point(
                    ax,
                    x=nfpt_x,
                    y=float(nfpt[metric]),
                    label=nfpt_legend_label and index == 0,
                )
        if mark_optimal:
            x_ideal, y_ideal = ideal_metric_point(metric)
            if x_key == RHO_KEY:
                x_ideal = 1.0
            ax.scatter(
                [x_ideal],
                [y_ideal],
                marker="*",
                s=160,
                c="#c0392b",
                edgecolors="k",
                linewidths=0.4,
                zorder=4,
                label="ideal",
                clip_on=False,
            )
        if annotate:
            indices = (
                min_max_y_indices(ys)
                if annotate_min_max_y_only
                else selected_annotation_indices(xs, ys, metric=metric)
            )
            annotate_selected_points(
                ax,
                xs,
                ys,
                indices,
                fontsize=FIG_ANNOTATION_FONTSIZE
                * (label_fontsize / FIG_AXIS_LABEL_FONTSIZE),
                offset=5,
            )
        if index >= 3:
            xlab_kw: dict[str, Any] = {"fontsize": label_fontsize}
            if bottom_xlabel_labelpad is not None:
                xlab_kw["labelpad"] = bottom_xlabel_labelpad
            ax.set_xlabel(x_label, **xlab_kw)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=label_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        if axis_limits is not None and metric in axis_limits:
            ax.set_xlim(*axis_limits[metric]["xlim"])
            ax.set_ylim(*axis_limits[metric]["ylim"])
        else:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, y1 + 0.01)


def _draw_sigmaj_vs_cvec_on_axes(
    axes: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    cmap,
    norm,
    annotate: bool = True,
    tick_fontsize: float = FIG_TICK_FONTSIZE,
    label_fontsize: float = FIG_AXIS_LABEL_FONTSIZE,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    nfpt_by_bench: dict[str, dict[str, Any]] | None = None,
    annotate_min_max_y_only: bool = False,
    subplot_title_pad: float | None = None,
) -> Any:
    """Fill a 1x3 axes block for TD1--TD3 sigma_J vs C_vec; return Pareto handle."""
    pareto_handle = None
    flat = list(np.asarray(axes).ravel())
    for ax, bench in zip(flat, ("td1", "td2", "td3"), strict=True):
        group = sorted(
            [r for r in rows if r["benchmark"] == bench],
            key=lambda r: float(r["robin_weight"]),
        )
        if not group:
            raise ValueError(f"No rows for benchmark={bench}")
        xs = [float(r["sigma_J"]) for r in group]
        ys = [float(r["C_vec"]) for r in group]
        cs = [float(r["robin_weight"]) for r in group]
        ax.scatter(
            xs,
            ys,
            c=cs,
            cmap=cmap,
            norm=norm,
            marker=MARKERS[bench],
            s=36,
            edgecolors="k",
            linewidths=0.35,
            zorder=3,
        )
        nfpt = (nfpt_by_bench or {}).get(bench)
        if nfpt is not None and "sigma_J" in nfpt and "C_vec" in nfpt:
            overlay_nfpt_metric_point(
                ax,
                x=float(nfpt["sigma_J"]),
                y=float(nfpt["C_vec"]),
                label=False,
            )
        if annotate:
            indices = (
                min_max_y_indices(ys)
                if annotate_min_max_y_only
                else selected_annotation_indices(xs, ys, prefer_higher_y=True)
            )
            annotate_selected_points(
                ax,
                xs,
                ys,
                indices,
                fontsize=FIG_ANNOTATION_FONTSIZE
                * (label_fontsize / FIG_AXIS_LABEL_FONTSIZE),
                offset=5,
            )
        ax.set_xlabel(METRIC_LABELS["sigma_J"], fontsize=label_fontsize)
        ax.set_ylabel(METRIC_LABELS["C_vec"], fontsize=label_fontsize)
        title_kw: dict[str, Any] = {
            "fontsize": FIG_TITLE_FONTSIZE * (label_fontsize / FIG_AXIS_LABEL_FONTSIZE),
        }
        if subplot_title_pad is not None:
            title_kw["pad"] = subplot_title_pad
        ax.set_title(bench.upper(), **title_kw)
        ax.tick_params(labelsize=tick_fontsize)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, y1 + 0.01)
        tradeoff_points = list(zip(xs, ys))
        if nfpt is not None and "sigma_J" in nfpt and "C_vec" in nfpt:
            tradeoff_points.append((float(nfpt["sigma_J"]), float(nfpt["C_vec"])))
        _draw_supposed_tradeoff(ax, tradeoff_points)
    return pareto_handle


def _improving_pareto_front(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Non-dominated points that strictly improve C_vec as sigma_J grows."""
    front = pareto_front_sigmaj_cvec(points)
    kept: list[tuple[float, float]] = []
    for sigma, cvec in front:
        if kept and cvec <= kept[-1][1] + 1.0e-6:
            continue
        kept.append((sigma, cvec))
    return kept


def _hermite_sample(t: np.ndarray, p0: np.ndarray, t0: np.ndarray, p1: np.ndarray, t1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite on t in [0, 1]. Endpoint derivatives are t0 and t1."""
    tt = np.asarray(t, dtype=float)
    h00 = (1.0 + 2.0 * tt) * (1.0 - tt) ** 2
    h10 = tt * (1.0 - tt) ** 2
    h01 = tt**2 * (3.0 - 2.0 * tt)
    h11 = tt**2 * (tt - 1.0)
    x = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
    y = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]
    return x, y


def _two_knot_hermite(
    knots: list[tuple[float, float]],
    y_bottom: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two knots and tangents that track the Pareto points from above.

    The left knot is the leftmost non-dominated point and its tangent is
    steep enough that the backward ray meets the bottom edge. The right
    knot is the last improving point and its tangent is horizontal.
    """
    pts = np.asarray(knots, dtype=float)
    p0 = pts[0].copy()
    p1 = pts[-1].copy()
    span = p1 - p0
    span_len = float(np.hypot(span[0], span[1]))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for angle_deg in np.linspace(55.0, 82.0, 28):
        direction = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))])
        for scale0 in np.linspace(0.25, 2.8, 26):
            t0 = direction * (scale0 * span_len)
            if t0[1] <= 1.0e-8:
                continue
            drop_dx = (p0[1] - float(y_bottom)) * t0[0] / t0[1]
            if not (0.004 <= drop_dx <= 0.022):
                continue
            for scale1 in np.linspace(0.35, 2.4, 22):
                t1 = np.array([scale1 * abs(float(span[0])), 0.0])
                ts = np.linspace(0.0, 1.0, 360)
                x, y = _hermite_sample(ts, p0, t0, p1, t1)
                if np.any(np.diff(x) <= 1.0e-8):
                    continue
                if float(y.max()) > 1.001 or float(y.min()) < p0[1] - 1.0e-4:
                    continue
                running = pts[0, 1]
                ok = True
                excess = 0.0
                j = 0
                for px, py in zip(x, y):
                    while j + 1 < pts.shape[0] and pts[j + 1, 0] <= px:
                        j += 1
                        running = max(running, pts[j, 1])
                    floor = pts[0, 1] if px < pts[0, 0] else running
                    if py < floor - 1.0e-5:
                        ok = False
                        break
                if not ok:
                    continue
                for px, py in pts:
                    yc = float(np.interp(px, x, y))
                    if yc < py - 1.0e-5:
                        ok = False
                        break
                    excess += (yc - py) ** 2
                if not ok:
                    continue
                # Closest curve that still clears every non-dominated point.
                score = excess
                if best is None or score < best[0]:
                    best = (score, t0.copy(), t1.copy())
    if best is None:
        secant = span / max(span_len, 1.0e-8)
        best = (0.0, secant * span_len, np.array([abs(float(span[0])), 0.0]))
    return p0, best[1], p1, best[2]


def _sample_hermite_to_box(
    p0: np.ndarray,
    t0: np.ndarray,
    p1: np.ndarray,
    t1: np.ndarray,
    x_left: float,
    x_right: float,
    y_bottom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hermite segment plus tangent rays to the bottom and right edges."""
    x, y = _hermite_sample(np.linspace(0.0, 1.0, 400), p0, t0, p1, t1)
    # Backward along the left tangent to the panel frame (bottom axis first).
    y_touch = float(y_bottom)
    x_touch = float(x_left)
    back_x = np.empty(0)
    back_y = np.empty(0)
    if t0[1] > 1.0e-8 and float(p0[1]) > y_touch:
        u_candidates: list[float] = []
        u_candidates.append((float(p0[1]) - y_touch) / float(t0[1]))
        if t0[0] > 1.0e-8 and float(p0[0]) > x_touch:
            u_candidates.append((float(p0[0]) - x_touch) / float(t0[0]))
        u_hit = min(u for u in u_candidates if u > 1.0e-6)
        u = np.linspace(float(u_hit), 0.0, 48)
        back_x = p0[0] - u * t0[0]
        back_y = p0[1] - u * t0[1]
        back_y[-1] = float(p0[1])
        back_x[-1] = float(p0[0])
        back_y[0] = y_touch
        back_x[0] = float(back_x[0])
        if back_x[0] > x_touch + 1.0e-6:
            n_floor = max(3, int(24 * (back_x[0] - x_touch) / max(float(p0[0]) - x_touch, 1.0e-6)))
            floor_x = np.linspace(back_x[0], x_touch, n_floor)
            floor_y = np.full(n_floor, y_touch)
            back_x = np.concatenate([floor_x, back_x[1:]])
            back_y = np.concatenate([floor_y, back_y[1:]])
        elif back_x[0] < x_touch:
            back_x[0] = x_touch
            back_y[0] = y_touch
    # Forward along the flat right tangent to the right spine.
    if t1[0] > 1.0e-8:
        run = np.linspace(0.0, (x_right - p1[0]) / t1[0], 40)
    else:
        run = np.zeros(1)
    fwd_x = p1[0] + run * t1[0]
    fwd_y = p1[1] + run * t1[1]
    return (
        np.concatenate([back_x[::-1], x[1:], fwd_x[1:]]),
        np.concatenate([back_y[::-1], y[1:], fwd_y[1:]]),
    )


def _draw_supposed_tradeoff(ax, points: list[tuple[float, float]]) -> None:
    """Two-knot Hermite guide along the non-dominated points.

    The curve stays on or above that front, turns down from the leftmost
    point into the bottom edge, and runs flat to the right edge.
    """
    knots = _improving_pareto_front(points)
    if len(knots) < 2:
        return
    x_left, x_right = (float(v) for v in ax.get_xlim())
    y_bottom, _y_top = (float(v) for v in ax.get_ylim())
    p0, t0, p1, t1 = _two_knot_hermite(knots, y_bottom)
    sigma, cvec = _sample_hermite_to_box(p0, t0, p1, t1, x_left, x_right, y_bottom)
    ax.plot(
        sigma,
        cvec,
        color="black",
        linestyle=(0, (0.9, 1.5)),
        linewidth=1.15,
        zorder=2,
        clip_on=True,
    )


def _draw_supposed_tradeoff_lower_is_better(ax, points: list[tuple[float, float]]) -> None:
    """Guide for a minimized metric, matching the panel-(d) construction.

    The curve stays on or below the non-dominated errors. From the leftmost
    of those points it turns upward by a short step, the same role as the
    downward step on a maximized metric, and it runs flat to the right edge.
    """
    x_left, x_right = (float(v) for v in ax.get_xlim())
    _y_bottom, y_top = (float(v) for v in ax.get_ylim())
    negated = [(float(s), -float(y)) for s, y in points]
    knots = _improving_pareto_front(negated)
    if len(knots) < 2:
        return
    left_error = -float(knots[0][1])
    y_bad = min(float(y_top), left_error + 0.045)
    y_bottom_neg = -y_bad
    p0, t0, p1, t1 = _two_knot_hermite(knots, y_bottom_neg)
    sigma_h, neg_y = _hermite_sample(np.linspace(0.0, 1.0, 400), p0, t0, p1, t1)
    back_x = np.empty(0)
    back_y = np.empty(0)
    if t0[1] > 1.0e-8 and float(p0[1]) > y_bottom_neg:
        u_hit = (float(p0[1]) - y_bottom_neg) / float(t0[1])
        if t0[0] > 1.0e-8:
            u_left = (float(p0[0]) - x_left) / float(t0[0])
            if 1.0e-6 < u_left < u_hit:
                u_hit = u_left
        u = np.linspace(float(u_hit), 0.0, 36)
        back_x = p0[0] - u * t0[0]
        back_y = p0[1] - u * t0[1]
        back_x[-1] = float(p0[0])
        back_y[-1] = float(p0[1])
    if t1[0] > 1.0e-8:
        run = np.linspace(0.0, max(x_right - float(p1[0]), 0.0) / float(t1[0]), 40)
    else:
        run = np.zeros(1)
    fwd_x = p1[0] + run * t1[0]
    fwd_y = p1[1] + run * t1[1]
    sigma = np.concatenate([back_x[::-1], sigma_h[1:], fwd_x[1:]])
    neg = np.concatenate([back_y[::-1], neg_y[1:], fwd_y[1:]])
    ax.plot(
        sigma,
        -neg,
        color="black",
        linestyle=(0, (0.9, 1.5)),
        linewidth=1.15,
        zorder=2,
        clip_on=True,
    )


def _mean_distance_to_panel_optimum(
    rows: list[dict[str, Any]],
    y_key: str,
) -> tuple[list[float], list[float]]:
    """Mean TD1--TD3 distance from each rho to one panel utopia.

    Both axes are scaled to the unit interval by the min and max of the
    plotted states, so a step in sigma_J matches a step in the vertical
    score. The optimal point is the unattained corner (lowest sigma_J,
    highest vertical score).
    """

    def _y_value(row: dict[str, Any]) -> float:
        if y_key == "one_minus_erel":
            return 1.0 - float(row[EREL_KEY])
        return float(row[y_key])

    sigmas = [float(row["sigma_J"]) for row in rows]
    scores = [_y_value(row) for row in rows]
    sigma_span = max(sigmas) - min(sigmas)
    score_span = max(scores) - min(scores)
    if sigma_span <= 0.0 or score_span <= 0.0:
        raise ValueError(f"Panel {y_key} has a degenerate axis range")
    sigma_min = min(sigmas)
    score_min = min(scores)
    weights = sorted({float(row[RHO_KEY]) for row in rows})
    means: list[float] = []
    for weight in weights:
        distances: list[float] = []
        for bench in ("td1", "td2", "td3"):
            match = [
                row
                for row in rows
                if row["benchmark"] == bench
                and abs(float(row[RHO_KEY]) - weight) < 1.0e-6
            ]
            if len(match) != 1:
                raise ValueError(f"Expected one {bench} row at rho={weight:g}")
            row = match[0]
            x_norm = (float(row["sigma_J"]) - sigma_min) / sigma_span
            y_norm = (_y_value(row) - score_min) / score_span
            distances.append(float(np.hypot(x_norm, 1.0 - y_norm)))
        means.append(float(np.mean(distances)))
    return weights, means


def plot_sigmaj_tradeoff_pair(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> float:
    """Select rho on the CCI-error / sigma_J plane only.

    The shared min/max normalization covers all TD states. Correlation,
    connectivity, twist and energy are independent checks, not selectors.
    """
    from matplotlib.lines import Line2D

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights, means = _mean_distance_to_panel_optimum(rows, "one_minus_erel")
    chosen_index = int(np.argmin(means))
    chosen_rho = float(weights[chosen_index])
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(weights), max(weights))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.55), gridspec_kw={"width_ratios": [1, 1]})
    for bench in ("td1", "td2", "td3"):
        group = sorted([r for r in rows if r["benchmark"] == bench], key=lambda r: float(r[RHO_KEY]))
        axes[0].plot([r["sigma_J"] for r in group], [r[EREL_KEY] for r in group], color="0.75", lw=0.8, zorder=1)
        axes[0].scatter([r["sigma_J"] for r in group], [r[EREL_KEY] for r in group], c=[r[RHO_KEY] for r in group],
                        cmap=cmap, norm=norm, marker=MARKERS[bench], s=35, edgecolors="k", linewidths=0.35, zorder=3)
        selected=next(r for r in group if abs(float(r[RHO_KEY])-chosen_rho)<1e-6)
        axes[0].scatter([selected["sigma_J"]], [selected[EREL_KEY]], s=135, facecolors="none", edgecolors="k", lw=1.2, zorder=4)
    axes[0].set(xlabel=METRIC_LABELS["sigma_J"], ylabel=EREL_LABEL, xlim=(0,.30), ylim=(0,.35))
    axes[0].grid(alpha=.25, lw=.6)
    handles=[Line2D([],[],marker=MARKERS[b],ls="none",color=".35",label=b.upper()) for b in ("td1","td2","td3")]
    handles.append(Line2D([],[],marker="o",ls="none",markerfacecolor="none",color="k",label=rf"$\rho={chosen_rho:g}$"))
    axes[0].legend(handles=handles,loc="upper right",frameon=False,fontsize=8)
    positions=np.arange(len(weights))
    bars=axes[1].bar(positions,means,color=cmap(norm(weights)),width=.78)
    bars[chosen_index].set_edgecolor("k");bars[chosen_index].set_linewidth(1.5)
    axes[1].set_xticks(positions,[f"{w:g}" for w in weights],rotation=45)
    axes[1].set(xlabel=RHO_LABEL,ylabel="Mean normalized distance",ylim=(0,1.05))
    axes[1].grid(axis="y",alpha=.25,lw=.6)
    axes[1].annotate(rf"$\rho={chosen_rho:g}$: {means[chosen_index]:.3f}", (chosen_index,means[chosen_index]),
                     xytext=(chosen_index+1.1,means[chosen_index]+.18),fontsize=9,arrowprops={"arrowstyle":"->","lw":.8})
    for axis,letter in zip(axes,("a","b")):
        axis.text(-.13,1.04,f"({letter})",transform=axis.transAxes,fontweight="bold")
        axis.tick_params(labelsize=8)
    fig.subplots_adjust(left=.085,right=.91,bottom=.19,top=.91,wspace=.4)
    cax=fig.add_axes((.93,.19,.016,.72))
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
    bar.set_label(RHO_LABEL);bar.ax.tick_params(labelsize=8)
    fig.savefig(output_path,dpi=260,bbox_inches="tight",pad_inches=.05)
    plt.close(fig)
    summary={"criterion":"mean TD distance on E_rel_Bgt/sigma_J plane only", "normalization":"shared min/max across all plotted TD states", "selected_rho":chosen_rho,
             "mean_distances":{f"{w:.1f}":float(m) for w,m in zip(weights,means)}}
    output_path.with_suffix(".selection.json").write_text(json.dumps(summary,indent=2))
    print(f"CCI/force-free criterion: selected rho={chosen_rho:g}; mean distance={means[chosen_index]:.6f}",flush=True)
    return chosen_rho


def plot_combined_metrics_summary(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    nfpt_by_bench: dict[str, dict[str, Any]] | None = None,
    x_key: str = RHO_KEY,
) -> None:
    """Paper composite: TD1--TD3 metrics-vs-rho plus sigma_J--C_vec.

    Layout::

        (a) TD1  2x3 metrics vs rho
        (b) TD2  2x3
        (c) TD3  2x3
        (d)      1x3 sigma_J vs C_vec
        bottom horizontal robin-weight colorbar
    """
    from matplotlib.gridspec import GridSpec

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.cm.viridis
    all_weights = sorted({float(r[RHO_KEY]) for r in rows})
    norm = plt.Normalize(min(all_weights), max(all_weights))
    metric_limits = shared_metric_axis_limits(
        rows,
        margin_frac=0.1,
        nfpt_by_bench=nfpt_by_bench,
        x_key=x_key,
    )
    x_label = RHO_LABEL if x_key == RHO_KEY else EREL_LABEL
    # Included with width=\\textwidth and height=0.88\\textheight.
    fig_width, fig_height = 13.2, 18.8
    font_scale = font_scale_to_match_figure4(
        fig_width,
        fig_height,
        width_frac=1.0,
        height_frac=0.88,
    )
    tick_fontsize = FIG_TICK_FONTSIZE * font_scale
    label_fontsize = FIG_AXIS_LABEL_FONTSIZE * font_scale

    abc_row_ratio = 1.02
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = GridSpec(
        5,
        1,
        figure=fig,
        height_ratios=[
            abc_row_ratio,
            abc_row_ratio,
            abc_row_ratio,
            abc_row_ratio * (2.0 / 3.0),
            0.07,
        ],
        hspace=0.42,
        left=0.11,
        right=0.98,
        top=0.985,
        bottom=0.05,
    )
    panel_letter_x_pad = 0.022

    panel_letters = ("a", "b", "c")
    for row_i, (bench, letter) in enumerate(
        zip(("td1", "td2", "td3"), panel_letters, strict=True)
    ):
        block = outer[row_i].subgridspec(1, 1)
        inner = block[0, 0].subgridspec(2, 3, hspace=0.40, wspace=0.34)
        axes = np.empty((2, 3), dtype=object)
        for i in range(2):
            for j in range(3):
                axes[i, j] = fig.add_subplot(inner[i, j])
        _draw_metrics_vs_erel_on_axes(
            axes,
            rows,
            benchmark=bench,
            cmap=cmap,
            norm=norm,
            tick_fontsize=tick_fontsize,
            label_fontsize=label_fontsize,
            axis_limits=metric_limits,
            mark_optimal=True,
            nfpt=(nfpt_by_bench or {}).get(bench),
            x_key=x_key,
            color_points_by_weight=True,
            annotate_min_max_y_only=True,
            nfpt_legend_label=False,
            bottom_xlabel_labelpad=2.0,
        )
        panel_pos = axes[0, 0].get_position()
        fig.text(
            panel_pos.x0 - panel_letter_x_pad,
            panel_pos.y1 + 0.012,
            f"({letter}) {bench.upper()}",
            ha="right",
            va="bottom",
            fontsize=FIG_PANEL_LETTER_FONTSIZE * font_scale,
            fontweight="bold",
            transform=fig.transFigure,
        )

    block_d = outer[3].subgridspec(1, 1)
    d_stack = block_d[0, 0].subgridspec(2, 1, height_ratios=[0.14, 1.0], hspace=0.0)
    inner_d = d_stack[1, 0].subgridspec(1, 3, wspace=0.30)
    axes_d = np.array([fig.add_subplot(inner_d[0, j]) for j in range(3)], dtype=object)
    sigma_xlim = (0.0, 0.4)
    cvec_ylim = (0.95, 1.0)
    if nfpt_by_bench:
        nfpt_sigma = [
            float(row["sigma_J"])
            for row in nfpt_by_bench.values()
            if "sigma_J" in row
        ]
        nfpt_cvec = [
            float(row["C_vec"])
            for row in nfpt_by_bench.values()
            if "C_vec" in row
        ]
        if nfpt_sigma:
            sigma_xlim = (0.0, max(0.4, 1.1 * max(nfpt_sigma)))
        if nfpt_cvec:
            cvec_ylim = (min(0.95, 0.98 * min(nfpt_cvec)), 1.0)
    _draw_sigmaj_vs_cvec_on_axes(
        axes_d,
        rows,
        cmap=cmap,
        norm=norm,
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
        xlim=sigma_xlim,
        ylim=cvec_ylim,
        nfpt_by_bench=nfpt_by_bench,
        annotate_min_max_y_only=True,
        subplot_title_pad=2.0,
    )
    panel_d_pos = axes_d[0].get_position()
    fig.text(
        panel_d_pos.x0 - panel_letter_x_pad,
        panel_d_pos.y1 + 0.012,
        "(d)",
        ha="right",
        va="bottom",
        fontsize=FIG_PANEL_LETTER_FONTSIZE * font_scale,
        fontweight="bold",
        transform=fig.transFigure,
    )
    cax = fig.add_subplot(outer[4])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_label(RHO_LABEL, fontsize=FIG_CBAR_LABEL_FONTSIZE * font_scale)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=tick_fontsize)

    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_combined_metrics_summary_figures(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    also_dir: Path | None = None,
    nfpt_by_bench: dict[str, dict[str, Any]] | None = None,
) -> list[Path]:
    name = "figure_nfgs_td_robin_weight_metrics_summary.png"
    written: list[Path] = []
    path = output_dir / name
    plot_combined_metrics_summary(rows, path, nfpt_by_bench=nfpt_by_bench)
    written.append(path)
    if also_dir is not None:
        alt = also_dir / name
        plot_combined_metrics_summary(rows, alt, nfpt_by_bench=nfpt_by_bench)
        written.append(alt)
    return written


def plot_erel_vs_robin_weight(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Overlay TD1--TD3 curves: robin weight (x) vs E^{rel}_{B^>} (y)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    colors = {"td1": "#1f77b4", "td2": "#ff7f0e", "td3": "#2ca02c"}
    for bench in ("td1", "td2", "td3"):
        group = sorted(
            [r for r in rows if r["benchmark"] == bench],
            key=lambda r: float(r["robin_weight"]),
        )
        if not group:
            continue
        xs = [float(r["robin_weight"]) for r in group]
        ys = [float(r[EREL_KEY]) for r in group]
        ax.plot(
            xs,
            ys,
            linestyle=LINESTYLES[bench],
            marker=MARKERS[bench],
            color=colors[bench],
            linewidth=1.8,
            markersize=7,
            markeredgecolor="k",
            markeredgewidth=0.4,
            label=bench.upper(),
        )
    ax.set_xlabel("robin weight")
    ax.set_ylabel(EREL_LABEL)
    ax.set_title(
        r"$E^{\mathrm{rel}}_{B^{>}}"
        r"=\frac{1}{N}\sum|\mathbf{b}^{>}-\mathbf{B}^{>}|/|\mathbf{b}^{>}|$"
        " vs robin weight"
    )
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    ax.set_xlim(-0.05, 1.05)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_erel_vs_robin_figures(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    also_dir: Path | None = None,
) -> list[Path]:
    name = "figure_nfgs_td_robin_weight_vs_erel_bgt.png"
    paths = [output_dir / name]
    if also_dir is not None:
        paths.append(also_dir / name)
    written: list[Path] = []
    for path in paths:
        plot_erel_vs_robin_weight(rows, path)
        written.append(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "nfgs_td_robin_weight_sweep",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=ROOT
        / "paper_writing"
        / "core_figures"
        / "figure_nfgs_td_robin_weight_metrics_vs_erel_bgt.png",
    )
    parser.add_argument("--benchmarks", default="td1,td2,td3")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--init-mode", choices=("gs_cfcf", "cfcf"), default="gs_cfcf")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--sigma-improvement-tol", type=float, default=1.0e-5)
    parser.add_argument("--mean-bz-policy", default="uniform_open")
    parser.add_argument(
        "--horizontal-zero-mode-policy", default="frozen_linear_harmonic"
    )
    parser.add_argument("--k0-ode-smoothness", type=float, default=1.0e-3)
    parser.add_argument("--d0-zero-mode-policy", default="project")
    parser.add_argument("--source-zero-mode-policy", default="project")
    parser.add_argument("--harmonic-residual-policy", default="diagnose")
    parser.add_argument("--mean-mode-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--source-zero-mode-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--handoff-projection-policy", default="project")
    parser.add_argument("--handoff-projection-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--td-calculation-domain-mode",
        choices=("mudpack", "mudpack_legacy", "bottom_padding"),
        default="mudpack",
    )
    parser.add_argument("--td-topology-chunk-depth", type=int, default=4)
    parser.add_argument("--n-test-x", type=int, default=66)
    parser.add_argument("--n-test-y", type=int, default=66)
    parser.add_argument("--n-test-z", type=int, default=30)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--paper-metric-protocol",
        action="store_true",
        help=(
            "Use the Yi-confirmed TD protocol: central-half TD1/TD3 "
            "metrics and E_rel_Bgt, plus the TD2 flux-rope ROI."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--reuse-er-csv",
        action="store_true",
        help=f"If metrics CSV already has {EREL_KEY}, only replot.",
    )
    parser.add_argument(
        "--reuse-field-cache",
        action="store_true",
        default=True,
        help="Load NFGS fields from output-dir/fields when present (default: on).",
    )
    parser.add_argument(
        "--no-reuse-field-cache",
        action="store_true",
        help="Ignore existing field caches and re-solve.",
    )
    parser.add_argument(
        "--no-field-cache",
        action="store_true",
        help="Do not write NFGS / reference field caches.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-solve even if a field cache exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_er_path = out_dir / "metrics_with_erel_bgt.csv"
    weights = parse_weights(args.weights) if args.weights else list(DEFAULT_WEIGHTS)
    requested = [
        item.strip().lower() for item in args.benchmarks.split(",") if item.strip()
    ]
    reuse_fields = bool(args.reuse_field_cache) and not bool(args.no_reuse_field_cache)
    cache_fields = not bool(args.no_field_cache)
    force = bool(args.force)

    existing = load_existing_metrics(metrics_er_path)
    have_all = all(
        (round(w, 1), b) in existing and EREL_KEY in existing[(round(w, 1), b)]
        for w in weights
        for b in requested
    )

    if args.reuse_er_csv and have_all:
        rows = [existing[(round(w, 1), b)] for w in weights for b in requested]
        print(f"Reusing {metrics_er_path}", flush=True)
    else:
        prior = load_existing_metrics(out_dir / "metrics_all.csv")
        prior_er = load_existing_metrics(out_dir / "metrics_with_er.csv")
        device = analytic.resolve_device(args.device)
        dtype = analytic.dtype_from_name(args.dtype)
        print(
            f"device={device} dtype={dtype} max_iter={args.max_iter} "
            f"weights={weights} benchmarks={requested} "
            f"reuse_field_cache={reuse_fields} write_field_cache={cache_fields}",
            flush=True,
        )
        cases = driver.make_requested_cases(args, requested)
        if args.paper_metric_protocol:
            for case in cases.values():
                case["metric_slices"] = paper_metric_slices(case)
                case["metric_domain"] = paper_metric_domain(case)
        bgt_solvers: dict[str, NFGSSolverGPU] = {}
        for name in requested:
            nx, ny, _nz, dx, dy, dz = cases[name]["dims"]
            bgt_solvers[name] = make_bgt_helper_solver(
                nx=int(nx),
                ny=int(ny),
                dx=float(dx),
                dy=float(dy),
                dz=float(dz),
                device=device,
                dtype=dtype,
                args=args,
            )
        if cache_fields:
            (out_dir / "fields").mkdir(parents=True, exist_ok=True)
            for name in requested:
                ensure_reference_field_cache(out_dir, name, cases[name]["reference"])

        rows = []
        started = time.perf_counter()
        for wi, weight in enumerate(weights, 1):
            run_args = make_run_args(weight, args)
            print(f"[{wi}/{len(weights)}] robin_weight={weight:.1f}", flush=True)
            for name in requested:
                case = cases[name]
                cache_path = nfgs_field_cache_path(
                    out_dir, name, args.init_mode, weight
                )
                print(f"  {case['display']} ...", flush=True)
                if reuse_fields and cache_path.is_file() and not force:
                    print(f"    load {cache_path.name}", flush=True)
                    fields = load_field_cache(cache_path)
                    metrics = compute_nf2_metrics(
                        case["reference"], fields, case["metric_slices"]
                    )
                    record = {
                        "benchmark": case["benchmark"],
                        "dtype": str(dtype),
                        "init_mode": args.init_mode,
                        "metric_domain": case["metric_domain"],
                        "best_iter": None,
                        "n_iters": None,
                        "best_sigma": None,
                        "stop_reason": "reused_field_cache",
                        "wall_time_s": 0.0,
                        **{key: float(metrics[key]) for key in METRIC_KEYS},
                    }
                else:
                    record, fields = driver.run_case(
                        case,
                        device=device,
                        dtype=dtype,
                        max_iter=int(args.max_iter),
                        args=run_args,
                    )
                    if cache_fields:
                        analytic.save_field_cache(cache_path, fields)
                        print(f"    saved {cache_path.name}", flush=True)

                # TD1/TD3 use the horizontal metric ROI; TD2 keeps its
                # historical full-photosphere imprint evaluation.
                if name == "td2":
                    e_rel = bottom_bgt_relative_error(
                        bgt_solvers[name], case["reference"], fields
                    )
                else:
                    _sz, sy, sx = case["metric_slices"]
                    e_rel = bottom_bgt_relative_error(
                        bgt_solvers[name],
                        case["reference"],
                        fields,
                        y_slice=sy,
                        x_slice=sx,
                    )
                prior_row = prior.get((round(float(weight), 1), name), {})
                if not prior_row:
                    prior_row = prior_er.get((round(float(weight), 1), name), {})
                row = {
                    "robin_weight": float(weight),
                    "benchmark": name,
                    EREL_KEY: e_rel,
                    **{key: float(record[key]) for key in METRIC_KEYS},
                    "best_sigma": record["best_sigma"],
                    "best_iter": record["best_iter"],
                    "n_iters": record["n_iters"],
                    "stop_reason": record["stop_reason"],
                    "wall_time_s": record["wall_time_s"],
                    "dtype": record["dtype"],
                    "init_mode": record["init_mode"],
                    "metric_domain": record["metric_domain"],
                }
                # Prefer freshly computed metrics; fill gaps from prior CSV only.
                for key in METRIC_KEYS:
                    if row.get(key) is None and key in prior_row:
                        row[key] = float(prior_row[key])
                rows.append(row)
                print(
                    f"    {EREL_KEY}={e_rel:.6g} C_vec={row['C_vec']:.6f} "
                    f"({row['wall_time_s']:.1f}s)",
                    flush=True,
                )
                del fields
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            write_metrics(metrics_er_path, rows)
            (out_dir / "progress_erel_bgt.json").write_text(
                json.dumps(
                    {
                        "completed_weight": weight,
                        "completed_index": wi,
                        "total_weights": len(weights),
                        "elapsed_s": round(time.perf_counter() - started, 1),
                    },
                    indent=2,
                )
            )
        print(f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)

    write_metrics(metrics_er_path, rows)
    nfpt_by_bench = load_nfpt_metrics(
        out_dir / "nfpt_fortran_symmetric_metrics_with_erel_bgt.csv"
    )
    written = write_split_figures(
        rows,
        figure_base=args.figure_path,
        also_dir=out_dir,
        nfpt_by_bench=nfpt_by_bench,
    )
    written.extend(
        write_erel_vs_robin_figures(
            rows,
            output_dir=args.figure_path.parent,
            also_dir=out_dir,
        )
    )
    written.extend(
        write_sigmaj_vs_cvec_figures(
            rows,
            output_dir=args.figure_path.parent,
            also_dir=out_dir,
            nfpt_by_bench=nfpt_by_bench,
        )
    )
    written.extend(
        write_combined_metrics_summary_figures(
            rows,
            output_dir=args.figure_path.parent,
            also_dir=out_dir,
            nfpt_by_bench=nfpt_by_bench,
        )
    )
    print(f"metrics={metrics_er_path}", flush=True)
    for path in written:
        print(f"figure={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
