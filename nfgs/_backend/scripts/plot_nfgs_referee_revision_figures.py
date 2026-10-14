#!/usr/bin/env python3
"""Plot stopping, formulation, resolution, and quantitative-topology results."""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.scripts.nfgs_figure_style import (  # noqa: E402
    FIG_ANNOTATION_FONTSIZE,
    FIG_AXIS_LABEL_FONTSIZE,
    FIG_CBAR_LABEL_FONTSIZE,
    FIG_LEGEND_FONTSIZE,
    FIG_ROW_LABEL_FONTSIZE,
    FIG_TICK_FONTSIZE,
    FIG_TITLE_FONTSIZE,
)
from nfgs._backend.scripts.plot_nfgs_td_robin_bgt_compare import (  # noqa: E402
    _transposed_extent,
    bgt_plot_fig1_bottom_plane_window,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sequence(history: dict, key: str) -> np.ndarray:
    return np.asarray(history.get(key, []), dtype=float)


def plot_sigma_stopping_histories(rho_dir: Path, output_dir: Path) -> None:
    records: list[dict] = []
    for rho in (0.0, 0.5, 1.0):
        tag = f"{rho:.1f}".replace(".", "p")
        path = rho_dir / f"robin_w{tag}.json"
        for record in json.loads(path.read_text()):
            records.append({**record, "rho": rho})
    for case in ("td1", "td2", "td3"):
        fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), layout="constrained")
        for record in [row for row in records if row["benchmark"] == case]:
            history = record.get("history") or {}
            if not isinstance(history, dict) or not history:
                # Archived FP32 rho_full rows may omit iteration histories.
                continue
            iterations = sequence(history, "iter")
            if iterations.size == 0:
                iterations = np.arange(1, len(sequence(history, "sigma")) + 1)
            rho = float(record["rho"])
            label = rf"$\rho={rho:g}$"
            sigma = sequence(history, "sigma")
            if sigma.size == 0:
                continue
            running_best = np.minimum.accumulate(sigma)
            excess = sigma - running_best
            panels = (
                (sigma, r"monitored $\widehat\sigma_J$"),
                (excess, r"$\widehat\sigma_J-\widehat\sigma_{J,*}$"),
                (sequence(history, "energy"), "energy"),
                (sequence(history, "divB"), r"discrete $\nabla\!\cdot\!\mathbf{B}$"),
                (
                    sequence(history, "bottom_field_imprint_relative"),
                    r"bottom imprint residual",
                ),
                (
                    np.maximum(
                        sequence(history, "alpha_clip_fraction"),
                        sequence(history, "weak_field_floor_fraction"),
                    ),
                    "regularization activation",
                ),
            )
            for axis, (values, ylabel) in zip(axes.ravel(), panels, strict=True):
                if values.size:
                    axis.plot(iterations[: values.size], values, linewidth=1.1, label=label)
                axis.set_ylabel(ylabel)
            best_iter = int(record.get("best_iter", history.get("best_iter", 0) or 0))
            if 1 <= best_iter <= sigma.size:
                axes[0, 0].scatter(
                    [best_iter], [sigma[best_iter - 1]], s=18, zorder=4
                )
        for index, axis in enumerate(axes.ravel()):
            axis.set_xlabel("iteration")
            axis.grid(alpha=0.25)
            if index not in {2}:
                axis.set_yscale("symlog", linthresh=1.0e-14)
        axes[0, 0].legend(frameon=False, fontsize=FIG_LEGEND_FONTSIZE)
        for axis in axes.ravel():
            axis.tick_params(labelsize=FIG_TICK_FONTSIZE)
            axis.xaxis.label.set_size(FIG_AXIS_LABEL_FONTSIZE)
            axis.yaxis.label.set_size(FIG_AXIS_LABEL_FONTSIZE)
        fig.suptitle(
            f"{case.upper()} heuristic current-weighted-sine stopping",
            fontsize=FIG_TITLE_FONTSIZE,
        )
        fig.savefig(
            output_dir / f"figure_{case}_sigma_stopping_histories.png", dpi=240
        )
        plt.close(fig)


def plot_factorized_bridge(path: Path, output: Path) -> None:
    rows = read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), layout="constrained")
    for axis, case in zip(axes, ("td1", "td2", "td3"), strict=True):
        group = sorted(
            (row for row in rows if row["benchmark"] == case),
            key=lambda row: row["variant_id"],
        )
        x = np.arange(len(group))
        axis.plot(x, [float(row["C_vec"]) for row in group], "o-", label=r"$C_{\rm vec}$")
        axis.plot(x, [float(row["sigma_J_common_fd2"]) for row in group], "s-", label=r"common $\sigma_J$")
        axis.set_xticks(x, [row["variant_id"] for row in group])
        axis.set_title(case.upper(), fontsize=FIG_TITLE_FONTSIZE)
        axis.tick_params(labelsize=FIG_TICK_FONTSIZE)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=FIG_LEGEND_FONTSIZE)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_gamma_scaling(path: Path, output: Path) -> None:
    rows = read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), layout="constrained")
    cmap = plt.get_cmap("viridis")
    anchors = sorted({float(row["rho_reference"]) for row in rows})
    for axis, case in zip(axes, ("td1", "td2", "td3"), strict=True):
        for index, anchor in enumerate(anchors):
            group = [
                row for row in rows
                if row["benchmark"] == case and float(row["rho_reference"]) == anchor
            ]
            group.sort(key=lambda row: float(row["requested_ratio"]))
            axis.plot(
                [float(row["E_rel_Bgt_domain"]) for row in group],
                [float(row["sigma_J_common_fd2"]) for row in group],
                "o-", color=cmap(index / max(len(anchors) - 1, 1)),
                label=rf"$\rho_{{\rm ref}}={anchor:g}$",
            )
            for row in group:
                axis.annotate(
                    f"{float(row['requested_ratio']):g}x",
                    (float(row["E_rel_Bgt_domain"]), float(row["sigma_J_common_fd2"])),
                    fontsize=FIG_ANNOTATION_FONTSIZE,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
        axis.set_title(case.upper(), fontsize=FIG_TITLE_FONTSIZE)
        axis.set_xlabel(r"$E^{\rm rel}_{B^>}$", fontsize=FIG_AXIS_LABEL_FONTSIZE)
        axis.tick_params(labelsize=FIG_TICK_FONTSIZE)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(
        r"common-domain $\sigma_J$",
        fontsize=FIG_AXIS_LABEL_FONTSIZE,
    )
    axes[-1].legend(frameon=False, fontsize=FIG_LEGEND_FONTSIZE)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_quantitative_topology_maps(
    topology_dir: Path,
    output: Path,
    *,
    field_dir: Path,
) -> None:
    methods = (
        ("analytic_reference", "ref."),
        ("nfgs_rho_1.0", r"NFGS $\rho=1$"),
        ("nfgs_rho_0.5", r"NFGS $\rho=0.5$"),
        ("nfgs_rho_0.1", r"NFGS $\rho=0.1$"),
        ("nfgs_rho_0.0", r"NFGS $\rho=0$"),
    )
    cases = ("td1", "td2", "td3")
    required = [
        topology_dir / f"{case}__{method}.npz"
        for case in cases
        for method, _label in methods
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Quantitative-topology map bundles are incomplete: "
            + ", ".join(str(path) for path in missing)
        )

    title_fontsize = FIG_TITLE_FONTSIZE
    row_label_fontsize = FIG_ROW_LABEL_FONTSIZE
    cbar_label_fontsize = FIG_CBAR_LABEL_FONTSIZE
    cbar_tick_fontsize = FIG_TICK_FONTSIZE
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d9d9d9")
    case_windows: dict[str, tuple[slice, slice, tuple[float, float, float, float]]] = {}
    row_aspects: list[float] = []
    for case in cases:
        meta = json.loads((field_dir / f"{case}__case_meta.json").read_text())
        with np.load(field_dir / f"{case}__reference.npz") as data:
            reference_fields = {key: np.asarray(data[key]) for key in ("Bx", "By", "Bz")}
        case_windows[case] = bgt_plot_fig1_bottom_plane_window(meta, reference_fields)
        extent = case_windows[case][2]
        x_span = float(extent[1] - extent[0])
        y_span = float(extent[3] - extent[2])
        row_aspects.append(x_span / max(y_span, 1.0e-30))
    col_width_in = 1.48
    n_cols = len(methods)
    n_rows = len(cases)
    plot_height_in = col_width_in * sum(row_aspects)
    fig_width = col_width_in * n_cols + 0.62
    fig_height = plot_height_in + 0.68
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.07],
        height_ratios=row_aspects,
        wspace=0.05,
        hspace=0.04,
        left=0.10,
        right=0.98,
        top=0.90,
        bottom=0.10,
    )
    colorbar_axis = fig.add_subplot(grid[:, n_cols])
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for column in range(n_cols):
            axes[row, column] = fig.add_subplot(grid[row, column])
    image = None
    for row, case in enumerate(cases):
        sy, sx, extent = case_windows[case]
        x_bounds = (extent[0], extent[1])
        y_bounds = (extent[2], extent[3])
        plot_extent = _transposed_extent(x_bounds, y_bounds)
        for column, (method, label) in enumerate(methods):
            with np.load(topology_dir / f"{case}__{method}.npz") as data:
                q = np.asarray(data["q"], dtype=float)
            logq = np.log10(np.maximum(q, 1.0))
            logq_view = np.asarray(logq[sy, sx], dtype=float)
            logq_plot = np.ma.masked_invalid(logq_view.T)
            axis = axes[row, column]
            image = axis.imshow(
                logq_plot,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
                cmap=cmap,
                vmin=0.0,
                vmax=5.0,
                extent=plot_extent,
            )
            axis.contour(
                np.where(np.isfinite(logq_view.T), logq_view.T, -np.inf),
                levels=[2.0],
                colors=["lime"],
                linewidths=1.4,
                extent=plot_extent,
                origin="lower",
            )
            if column == 0:
                axis.tick_params(
                    axis="both",
                    labelsize=cbar_tick_fontsize,
                    pad=3.0,
                )
                if row != n_rows - 1:
                    axis.tick_params(labelbottom=False)
            else:
                axis.set_xticks([])
                axis.set_yticks([])
            if row == 0:
                axis.set_title(label, fontsize=title_fontsize)
            if column == 0:
                axis.set_ylabel(
                    case.upper(),
                    fontsize=row_label_fontsize,
                    labelpad=6.0,
                )
    if image is not None:
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(r"$\log_{10} Q$", fontsize=cbar_label_fontsize, labelpad=8)
        colorbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    fig.savefig(output, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root", type=Path,
        default=ROOT / "results" / "nfgs_paper" / "referee_revision",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir or (args.campaign_root / "figures")
    output.mkdir(parents=True, exist_ok=True)
    plot_sigma_stopping_histories(args.campaign_root / "rho_full", output)
    plot_factorized_bridge(
        args.campaign_root / "factorized_ablation" / "metrics_all.csv",
        output / "figure_factorized_pt_to_nfgs_bridge.png",
    )
    plot_gamma_scaling(
        args.campaign_root / "rho_gamma_scaling" / "metrics_all.csv",
        output / "figure_rho_gamma_resolution_tradeoff.png",
    )
    topology_dir = args.campaign_root / "quantitative_topology"
    field_dir = args.campaign_root / "rho_full" / "fields"
    if (topology_dir / "metrics.csv").is_file():
        plot_quantitative_topology_maps(
            topology_dir,
            output / "figure_quantitative_topology_bottom_qmaps.png",
            field_dir=field_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
