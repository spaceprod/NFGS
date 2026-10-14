#!/usr/bin/env python3
"""Measure FastQSL twist maps and high-twist apex locations on the TD y=0 plane."""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.ndimage import label

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.util.fastqsl_plane import compute_y0_qmap  # noqa: E402
from nfgs._backend.scripts.plot_nfgs_td_robin_bgt_compare import bgt_plot_twist_y0_window  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field-dir", type=Path,
        default=ROOT / "results" / "nfgs_paper" / "referee_revision" / "rho_full" / "fields",
    )
    parser.add_argument(
        "--nfpt-field-dir", type=Path,
        default=ROOT / "results" / "nfgs_td_robin_weight_sweep" / "fields",
    )
    parser.add_argument(
        "--omit-nfpt",
        action="store_true",
        help="Plot the reference and NFGS weights only.",
    )
    parser.add_argument(
        "--allow-legacy-nfpt-padding", action="store_true",
        help="Allow the archived asymmetric nfpt_fortran field for a diagnostic-only run.",
    )
    parser.add_argument("--nfpt-variant", default="nfpt_fortran_symmetric")
    parser.add_argument("--benchmarks", default="td1,td2,td3")
    parser.add_argument("--weights", default="0,0.5,1")
    parser.add_argument("--step-len", type=float, default=1.0 / 8.0)
    parser.add_argument("--tol", type=float, default=0.1)
    parser.add_argument("--block-x", type=int, default=128)
    parser.add_argument(
        "--apex-window-mm", type=float, default=70.0,
        help="Fixed y=0 search radius around the analytic high-twist apex.",
    )
    parser.add_argument(
        "--core-fraction", type=float, default=0.4,
        help="Absolute-Tw threshold as a fraction of the analytic peak.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "nfgs_paper" / "referee_revision" / "twist_apex",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_field(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        if "B" in data:
            b = np.asarray(data["B"])
            return {name: b[index] for index, name in enumerate(("Bx", "By", "Bz"))}
        return {name: np.asarray(data[name]) for name in ("Bx", "By", "Bz")}


def method_specs(args: argparse.Namespace, case: str) -> list[tuple[str, Path]]:
    specs = [("analytic_reference", args.field_dir / f"{case}__reference.npz")]
    if not args.omit_nfpt:
        specs.append(
            (args.nfpt_variant, args.nfpt_field_dir / f"{case}__{args.nfpt_variant}.npz")
        )
    for weight in sorted({float(value) for value in args.weights.split(",") if value.strip()}):
        tag = f"{weight:.1f}".replace(".", "p")
        specs.append(
            (
                f"nfgs_rho_{weight:.1f}",
                args.field_dir / f"{case}__nfgs__gs_cfcf__robin_w{tag}.npz",
            )
        )
    return specs


def coordinates(meta: dict[str, object], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    nx = int(meta["dims"][0])
    nz = int(meta["dims"][2])
    dx = float(meta["dims"][3])
    x0 = float(meta["x_bounds"][0])
    z0 = float(meta["z_bounds"][0])
    if shape != (nz, nx):
        raise ValueError(f"twist map shape {shape} does not match metadata {(nz, nx)}")
    return x0 + dx * np.arange(nx), z0 + dx * np.arange(nz), dx


def peak_in_window(
    twist: np.ndarray,
    xx: np.ndarray,
    zz: np.ndarray,
    reference_x: float,
    reference_z: float,
    radius_mm: float,
) -> tuple[int, int]:
    distance = np.hypot(xx - reference_x, zz - reference_z)
    valid = np.isfinite(twist) & (distance <= radius_mm)
    if not np.any(valid):
        raise RuntimeError("no finite twist samples inside the fixed apex window")
    score = np.where(valid, np.abs(twist), -np.inf)
    return tuple(int(value) for value in np.unravel_index(np.argmax(score), score.shape))


def high_twist_apex(
    twist: np.ndarray,
    xx: np.ndarray,
    zz: np.ndarray,
    *,
    threshold: float,
    reference_peak_x: float,
    reference_peak_z: float,
    max_distance_mm: float,
) -> tuple[float, float, float, int]:
    """Return the top of the thresholded core nearest the analytic core."""
    labels, n_labels = label(np.isfinite(twist) & (np.abs(twist) >= threshold))
    if n_labels == 0:
        return float("nan"), float("nan"), float("nan"), 0
    best_label = 0
    best_distance = float("inf")
    for candidate in range(1, n_labels + 1):
        iz, ix = np.where(labels == candidate)
        distance = float(np.min(np.hypot(xx[iz, ix] - reference_peak_x, zz[iz, ix] - reference_peak_z)))
        if distance < best_distance:
            best_distance = distance
            best_label = candidate
    if best_distance > max_distance_mm:
        return float("nan"), float("nan"), float("nan"), 0
    iz, ix = np.where(labels == best_label)
    top_z = float(np.max(zz[iz, ix]))
    top_x = float(np.mean(xx[iz, ix][np.isclose(zz[iz, ix], top_z)]))
    return top_x, top_z, best_distance, int(np.count_nonzero(labels == best_label))


from nfgs._backend.scripts.nfgs_figure_style import (  # noqa: E402
    FIG_AXIS_LABEL_FONTSIZE,
    FIG_CBAR_LABEL_FONTSIZE,
    FIG_ROW_LABEL_FONTSIZE,
    FIG_TICK_FONTSIZE,
    FIG_TITLE_FONTSIZE,
    font_scale_to_match_figure4,
)

METHOD_LABELS = {
    "analytic_reference": "ref.",
    "nfgs_rho_1.0": r"NFGS $\rho=1$",
    "nfgs_rho_0.5": r"NFGS $\rho=0.5$",
    "nfgs_rho_0.1": r"NFGS $\rho=0.1$",
    "nfgs_rho_0.0": r"NFGS $\rho=0$",
}
PANEL_TITLE_FONTSIZE = FIG_TITLE_FONTSIZE
ROW_LABEL_FONTSIZE = FIG_ROW_LABEL_FONTSIZE
AXIS_LABEL_FONTSIZE = FIG_AXIS_LABEL_FONTSIZE
TICK_FONTSIZE = FIG_TICK_FONTSIZE
CBAR_LABEL_FONTSIZE = FIG_CBAR_LABEL_FONTSIZE
CBAR_TICK_FONTSIZE = FIG_TICK_FONTSIZE


def _method_order(maps: dict[str, np.ndarray]) -> list[str]:
    return [key for key in METHOD_LABELS if key in maps]


def _twist_color_limit(maps: dict[str, np.ndarray], order: list[str]) -> float:
    finite = np.concatenate([np.abs(maps[key][np.isfinite(maps[key])]) for key in order])
    return float(np.percentile(finite, 99.5)) if finite.size else 1.0


def _decorate_twist_axis(
    axis,
    *,
    row: int,
    col: int,
    n_rows: int,
    case: str,
    method: str,
    show_column_titles: bool,
    font_scale: float = 1.0,
) -> None:
    if show_column_titles:
        axis.set_title(METHOD_LABELS[method], fontsize=PANEL_TITLE_FONTSIZE * font_scale)
    if col == 0:
        axis.set_ylabel(case.upper(), fontsize=ROW_LABEL_FONTSIZE * font_scale)
        axis.tick_params(axis="y", labelsize=TICK_FONTSIZE * font_scale)
        axis.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _pos: "" if np.isclose(value, 150.0) else f"{int(round(value))}"
            )
        )
        if row == n_rows - 1:
            axis.set_xlabel("x (Mm)", fontsize=AXIS_LABEL_FONTSIZE * font_scale)
            axis.tick_params(axis="x", labelsize=TICK_FONTSIZE * font_scale)
        else:
            axis.tick_params(labelbottom=False)
    else:
        axis.tick_params(labelleft=False, labelbottom=False)


def _draw_twist_panel(
    axis,
    method: str,
    twist: np.ndarray,
    row: dict[str, object],
    x: np.ndarray,
    z: np.ndarray,
    *,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    limit: float,
):
    image = axis.imshow(
        twist,
        origin="lower",
        extent=(x[0], x[-1], z[0], z[-1]),
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    if np.isfinite(float(row["core_apex_x_mm"])):
        axis.plot(
            float(row["core_apex_x_mm"]),
            float(row["core_apex_z_mm"]),
            "ko",
            ms=3.5,
        )
    axis.set_xlim(xlim)
    axis.set_ylim(zlim)
    return image


def plot_maps(
    output: Path,
    case: str,
    maps: dict[str, np.ndarray],
    rows: list[dict[str, object]],
    x: np.ndarray,
    z: np.ndarray,
    *,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
) -> None:
    order = _method_order(maps)
    limit = _twist_color_limit(maps, order)
    figure = plt.figure(figsize=(3.15 * len(order) + 0.45, 3.35))
    grid = figure.add_gridspec(
        1, len(order) + 1,
        width_ratios=[1.0] * len(order) + [0.045],
        wspace=0.10,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(order))]
    for axis in axes[1:]:
        axis.sharex(axes[0])
        axis.sharey(axes[0])
    colorbar_axis = figure.add_subplot(grid[0, -1])
    by_method = {str(row["method"]): row for row in rows}
    image = None
    for col, (axis, method) in enumerate(zip(axes, order, strict=True)):
        image = _draw_twist_panel(
            axis,
            method,
            maps[method],
            by_method[method],
            x,
            z,
            xlim=xlim,
            zlim=zlim,
            limit=limit,
        )
        _decorate_twist_axis(
            axis,
            row=0,
            col=col,
            n_rows=1,
            case=case,
            method=method,
            show_column_titles=True,
        )
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"$T_w$", fontsize=CBAR_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    figure.subplots_adjust(left=0.07, right=0.965, bottom=0.14, top=0.90)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_combined_twist_maps(
    output: Path,
    blocks: list[
        tuple[
            str,
            dict[str, np.ndarray],
            list[dict[str, object]],
            np.ndarray,
            np.ndarray,
            tuple[float, float],
            tuple[float, float],
        ]
    ],
) -> None:
    if not blocks:
        raise ValueError("blocks must be nonempty")
    order = _method_order(blocks[0][1])
    n_rows = len(blocks)
    n_cols = len(order)
    fig_width = 3.15 * n_cols + 0.65
    fig_height = 3.05 * n_rows + 0.2
    font_scale = font_scale_to_match_figure4(fig_width, width_frac=0.98)
    figure = plt.figure(figsize=(fig_width, fig_height))
    grid = figure.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.055],
        wspace=0.08,
        hspace=0.04,
    )
    image = None
    for row_index, (case, maps, case_rows, x, z, xlim, zlim) in enumerate(blocks):
        limit = _twist_color_limit(maps, order)
        by_method = {str(row["method"]): row for row in case_rows}
        row_axes = []
        for col_index, method in enumerate(order):
            axis = figure.add_subplot(grid[row_index, col_index])
            if row_axes:
                axis.sharex(row_axes[0])
                axis.sharey(row_axes[0])
            row_axes.append(axis)
            image = _draw_twist_panel(
                axis,
                method,
                maps[method],
                by_method[method],
                x,
                z,
                xlim=xlim,
                zlim=zlim,
                limit=limit,
            )
            _decorate_twist_axis(
                axis,
                row=row_index,
                col=col_index,
                n_rows=n_rows,
                case=case,
                method=method,
                show_column_titles=row_index == 0,
                font_scale=font_scale,
            )
    colorbar_axis = figure.add_subplot(grid[:, -1])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"$T_w$", fontsize=CBAR_LABEL_FONTSIZE * font_scale)
    colorbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE * font_scale)
    figure.subplots_adjust(left=0.06, right=0.955, bottom=0.06, top=0.98)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if (
        not args.omit_nfpt
        and args.nfpt_variant != "nfpt_fortran_symmetric"
        and not args.allow_legacy_nfpt_padding
    ):
        raise SystemExit(
            "Paper twist maps require nfpt_fortran_symmetric so TD3 uses "
            "mirror-symmetric x/y padding.  Pass --allow-legacy-nfpt-padding "
            "only for an explicitly legacy diagnostic."
        )
    cases = tuple(value.strip().lower() for value in args.benchmarks.split(",") if value.strip())
    if not cases or set(cases) - {"td1", "td2", "td3"}:
        raise SystemExit("--benchmarks must be a nonempty subset of td1,td2,td3")
    if not 0.0 < args.core_fraction <= 1.0:
        raise SystemExit("--core-fraction must lie in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    figures: list[str] = []
    combined_blocks: list[
        tuple[
            str,
            dict[str, np.ndarray],
            list[dict[str, object]],
            np.ndarray,
            np.ndarray,
            tuple[float, float],
            tuple[float, float],
        ]
    ] = []
    for case in cases:
        meta = json.loads((args.field_dir / f"{case}__case_meta.json").read_text())
        reference_fields = load_field(args.field_dir / f"{case}__reference.npz")
        plot_x_bounds, plot_z_bounds = bgt_plot_twist_y0_window(meta, reference_fields)
        maps: dict[str, np.ndarray] = {}
        for method, field_path in method_specs(args, case):
            if not field_path.is_file():
                raise SystemExit(f"missing field: {field_path}")
            cache_path = args.output_dir / f"{case}__{method}.npz"
            if args.resume and cache_path.is_file():
                with np.load(cache_path) as cached:
                    maps[method] = np.asarray(cached["twist"])
                continue
            print(f"FastQSL twist {case.upper()} {method}", flush=True)
            field = load_field(field_path)
            bundle = compute_y0_qmap(
                field["Bx"], field["By"], field["Bz"],
                step_len=args.step_len, tol=args.tol, block_x=args.block_x,
                b_smooth_sigma_cells=0.0, include_twist=True,
            )
            maps[method] = np.asarray(bundle["twist"])
            np.savez_compressed(cache_path, twist=maps[method], metadata=json.dumps(bundle["metadata"]))
        x, z, _dx = coordinates(meta, maps["analytic_reference"].shape)
        xx, zz = np.meshgrid(x, z, indexing="xy")
        reference = maps["analytic_reference"]
        ref_iz, ref_ix = peak_in_window(reference, xx, zz, 0.0, float(np.nanmax(z)), float("inf"))
        ref_x, ref_z = float(xx[ref_iz, ref_ix]), float(zz[ref_iz, ref_ix])
        ref_abs_tw = float(abs(reference[ref_iz, ref_ix]))
        threshold = float(args.core_fraction * ref_abs_tw)
        ref_apex_x, ref_apex_z, _ref_distance, ref_area = high_twist_apex(
            reference, xx, zz, threshold=threshold, reference_peak_x=ref_x,
            reference_peak_z=ref_z, max_distance_mm=args.apex_window_mm,
        )
        case_rows: list[dict[str, object]] = []
        for method, _field_path in method_specs(args, case):
            apex_iz, apex_ix = peak_in_window(maps[method], xx, zz, ref_x, ref_z, args.apex_window_mm)
            apex_tw = float(maps[method][apex_iz, apex_ix])
            displacement = float(np.hypot(xx[apex_iz, apex_ix] - ref_x, zz[apex_iz, apex_ix] - ref_z))
            core_apex_x, core_apex_z, core_nearest_distance, core_area = high_twist_apex(
                maps[method], xx, zz, threshold=threshold, reference_peak_x=ref_x,
                reference_peak_z=ref_z, max_distance_mm=args.apex_window_mm,
            )
            row = {
                "benchmark": case,
                "method": method,
                "reference_peak_x_mm": ref_x,
                "reference_peak_z_mm": ref_z,
                "reference_peak_Tw": float(reference[ref_iz, ref_ix]),
                "reference_core_apex_x_mm": ref_apex_x,
                "reference_core_apex_z_mm": ref_apex_z,
                "reference_core_area_pixels": ref_area,
                "peak_x_mm": float(xx[apex_iz, apex_ix]),
                "peak_z_mm": float(zz[apex_iz, apex_ix]),
                "peak_Tw": apex_tw,
                "peak_abs_Tw_error": float(abs(abs(apex_tw) - ref_abs_tw)),
                "peak_abs_Tw_relative_error": float(abs(abs(apex_tw) - ref_abs_tw) / max(ref_abs_tw, 1e-12)),
                "peak_displacement_mm": displacement,
                "core_apex_x_mm": core_apex_x,
                "core_apex_z_mm": core_apex_z,
                "core_apex_displacement_mm": float(np.hypot(core_apex_x - ref_apex_x, core_apex_z - ref_apex_z)),
                "core_apex_height_error_mm": float(abs(core_apex_z - ref_apex_z)),
                "core_nearest_distance_mm": core_nearest_distance,
                "core_area_pixels": core_area,
                "core_threshold_abs_Tw": threshold,
                "search_radius_mm": float(args.apex_window_mm),
            }
            rows.append(row)
            case_rows.append(row)
        figure_path = args.output_dir / f"figure_{case}_fastqsl_twist_apex_y0.png"
        plot_maps(
            figure_path,
            case,
            maps,
            case_rows,
            x,
            z,
            xlim=plot_x_bounds,
            zlim=plot_z_bounds,
        )
        figures.append(figure_path.name)
        combined_blocks.append(
            (case, maps, case_rows, x, z, plot_x_bounds, plot_z_bounds)
        )
    combined_path = args.output_dir / "figure_fastqsl_twist_apex_y0_combined.png"
    plot_combined_twist_maps(combined_path, combined_blocks)
    figures.append(combined_path.name)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "benchmarks": list(cases),
        "weights": args.weights,
        "plane": "physical y=0",
        "twist_definition": "FastQSL line integral of (curl B dot B)/(4 pi |B|^2)",
        "apex_definition": "top of the thresholded Tw core nearest the analytic y=0 high-twist core",
        "apex_window_mm": args.apex_window_mm,
        "core_fraction": args.core_fraction,
        "figures": figures,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
