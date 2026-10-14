#!/usr/bin/env python3
"""Compare six-face energy and relative-helicity diagnostics on TD volumes."""

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

from nfgs.diagnostics.helicity_diagnostics import relative_helicity_diagnostics
from nfgs.diagnostics.volume_diagnostics import cell_center_potential, energy_diagnostics
from nfgs._backend.scripts.nfgs_figure_style import (  # noqa: E402
    FIG_AXIS_LABEL_FONTSIZE,
    FIG_LEGEND_FONTSIZE,
    FIG_TICK_FONTSIZE,
    FIG_TITLE_FONTSIZE,
    font_scale_to_match_figure4,
)


CASES = ("td1", "td2", "td3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--nfpt-field-dir", type=Path, required=True)
    parser.add_argument(
        "--nfpt-variant", default="nfpt_fortran_symmetric",
        help="NFPT field variant; the default enforces mirror-symmetric x/y padding.",
    )
    parser.add_argument(
        "--allow-legacy-nfpt-padding", action="store_true",
        help="Allow the archived asymmetric nfpt_fortran field for a diagnostic-only run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", default="0,0.5,1")
    parser.add_argument(
        "--omit-nfpt",
        action="store_true",
        help="Plot the reference and NFGS weights only.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the figure from metrics.csv without recomputing fields.",
    )
    return parser.parse_args()


def weight_token(value: float) -> str:
    return f"w{value:.1f}".replace(".", "p")


def protocol_slices(meta: dict) -> tuple[slice, slice, slice]:
    return tuple(slice(*item) for item in meta["metric_slices"])  # type: ignore[return-value]


def load_crop(path: Path, slices: tuple[slice, slice, slice]) -> dict[str, np.ndarray]:
    field: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for key in ("Bx", "By", "Bz"):
            field[key] = np.array(data[key][slices], copy=True)
    return field


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if (
        not args.omit_nfpt
        and args.nfpt_variant != "nfpt_fortran_symmetric"
        and not args.allow_legacy_nfpt_padding
    ):
        raise SystemExit(
            "Paper helicity comparison requires nfpt_fortran_symmetric so TD3 "
            "uses mirror-symmetric x/y padding."
        )
    weights = [float(item) for item in args.weights.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with (args.output_dir / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = []
    if not args.plot_only:
        for case in CASES:
            meta = json.loads((args.field_dir / f"{case}__case_meta.json").read_text())
            spacing = float(meta["dims"][3])
            slices = protocol_slices(meta)
            specs: list[tuple[str, float | None, Path]] = [
                ("analytic_reference", None, args.field_dir / f"{case}__reference.npz"),
            ]
            if not args.omit_nfpt:
                specs.append(
                    (
                        args.nfpt_variant,
                        None,
                        args.nfpt_field_dir / f"{case}__{args.nfpt_variant}.npz",
                    )
                )
            specs.extend(
                (
                    f"nfgs_rho_{weight:.1f}", weight,
                    args.field_dir / f"{case}__nfgs__gs_cfcf__robin_{weight_token(weight)}.npz",
                )
                for weight in weights
            )
            reference_values: dict[str, float] | None = None
            for method, rho, path in specs:
                print(f"[{case}] {method}", flush=True)
                field = load_crop(path, slices)
                potential, flux_imbalance = cell_center_potential(field, spacing)
                b = np.stack((field["Bx"], field["By"], field["Bz"]), axis=0)
                energy = energy_diagnostics(
                    field, spacing, potential=potential, flux_imbalance=flux_imbalance,
                )
                helicity = relative_helicity_diagnostics(b, potential, spacing)
                values = {**energy, **helicity}
                if reference_values is None:
                    reference_values = {
                        key: float(values[key])
                        for key in (
                            "total_energy_common", "potential_energy_6face", "free_energy_6face",
                            "relative_helicity_6face", "current_carrying_helicity_fraction_abs",
                        )
                    }
                row: dict[str, object] = {
                    "benchmark": case,
                    "method": method,
                    "rho": "" if rho is None else rho,
                    "domain": "protocol",
                    **values,
                }
                for key, reference_value in reference_values.items():
                    row[f"{key}_relative_difference"] = (
                        float(values[key]) - reference_value
                    ) / max(abs(reference_value), 1.0e-30)
                rows.append(row)
                del field, potential, b
        write_csv(args.output_dir / "metrics.csv", rows)

    labels = {"analytic_reference": "reference"}
    if not args.omit_nfpt:
        labels[args.nfpt_variant] = "NFPT"
    labels.update({f"nfgs_rho_{w:.1f}": rf"$\rho={w:g}$" for w in weights})
    fig_width, fig_height = 10.5, 3.6
    font_scale = font_scale_to_match_figure4(fig_width, width_frac=0.82)
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_width, fig_height), constrained_layout=True
    )
    # Keep the previous NFGS colors; only the reference square turns gray.
    nfgs_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]
    for ax, key, title in (
        (axes[0], "free_energy_6face_relative_difference", "free-energy difference"),
        (axes[1], "current_carrying_helicity_fraction_abs", r"$|H_J|/|H_R|$"),
    ):
        series_colors = iter(nfgs_palette)
        for method in labels:
            subset = [row for row in rows if row["method"] == method]
            if not subset:
                continue
            is_reference = method == "analytic_reference"
            ax.plot(
                [row["benchmark"].upper() for row in subset],
                [float(row[key]) for row in subset],
                "s" if is_reference else "o",
                markersize=11 if is_reference else 6,
                color="gray" if is_reference else next(series_colors),
                label=labels[method],
            )
        ax.set_title(title, fontsize=FIG_TITLE_FONTSIZE * font_scale)
        ax.tick_params(labelsize=FIG_TICK_FONTSIZE * font_scale)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(
        "relative to analytic reference",
        fontsize=FIG_AXIS_LABEL_FONTSIZE * font_scale,
    )
    axes[1].set_ylabel(
        "absolute fraction",
        fontsize=FIG_AXIS_LABEL_FONTSIZE * font_scale,
    )
    axes[1].legend(
        fontsize=FIG_LEGEND_FONTSIZE * font_scale,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    fig.savefig(
        args.output_dir / "figure_six_face_energy_helicity.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "domain": "primary_protocol", "potential_method": "six_face_flh_neumann",
        "helicity_method": "common_devore_Az_zero", "weights": weights,
        "metrics": "energy, relative helicity, current-carrying helicity fraction",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
