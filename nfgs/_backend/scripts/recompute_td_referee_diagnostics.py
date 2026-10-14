#!/usr/bin/env python3
"""Recompute cache-only TD diagnostics requested during peer review.

This script never advances the nonlinear solver.  It reads the frozen TD
reference/NFGS/NFPT field caches and applies one explicit, common FD2 operator
to all methods for the diagnostics that depend on a discrete derivative.  The
outputs deliberately retain both the historical paper domains and consistent
full/fractional-core domains so that cross-model means are not needed.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.scripts.plot_nfgs_td_robin_metrics_vs_er import (  # noqa: E402
    bottom_plane_bgt_nfgs,
    make_bgt_helper_solver,
)
from nfgs.diagnostics.volume_diagnostics import (  # noqa: E402
    cell_center_potential,
    energy_diagnostics,
    helmholtz_non_solenoidal_component,
)


CASES = ("td1", "td2", "td3")
METRIC_COLUMNS = (
    "C_vec",
    "C_CS",
    "Eprime_n",
    "Eprime_m",
    "epsilon",
    "sigma_J",
    "divB_normalized",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_td_robin_weight_sweep_frozen_linear_harmonic_vertical_staggered"
            / "fields"
        ),
    )
    parser.add_argument(
        "--case-meta-dir",
        type=Path,
        default=ROOT / "results" / "nfgs_analytic_validation_td" / "fields",
    )
    parser.add_argument(
        "--nfpt-field-dir",
        type=Path,
        default=None,
        help="Optional separate directory containing the selected NFPT caches.",
    )
    parser.add_argument(
        "--release-field-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_td_robin_zero_release_frozen_linear_harmonic_vertical_staggered"
            / "fields"
        ),
        help="Cached post-release best states; absent files are skipped.",
    )
    parser.add_argument(
        "--source-metrics",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_td_robin_weight_sweep_frozen_linear_harmonic_vertical_staggered"
            / "robin_w0p0.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_td_referee_diagnostics_frozen_linear_harmonic_vertical_staggered"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=(
            ROOT
            / "paper_writing"
            / "core_figures"
            / "frozen_linear_harmonic_vertical_staggered"
        ),
    )
    parser.add_argument(
        "--weights",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="NFGS mixed-row weights to postprocess.",
    )
    parser.add_argument(
        "--horizontal-zero-mode-policy",
        choices=(
            "coupled_k0",
            "omit",
            "frozen_linear_harmonic",
            "constrained_spline_harmonic",
        ),
        default="frozen_linear_harmonic",
        help="Expected NFGS source-cache k=0 protocol.",
    )
    parser.add_argument(
        "--tradeoff-only",
        action="store_true",
        help=(
            "Compute only cached-native protocol-domain rows required by "
            "figure_nfgs_td_imprint_tradeoff.png. Do not create profile, "
            "release, budget, component, or figure artifacts."
        ),
    )
    parser.add_argument(
        "--profile-weights",
        default="0.0,0.4,1.0",
        help="Subset used for common-curl checks and height profiles.",
    )
    parser.add_argument(
        "--profile-bin-planes",
        type=int,
        default=4,
        help=(
            "Current-weighted height-bin width used in the paper profile figure. "
            "Raw one-plane values remain visible as faint curves."
        ),
    )
    parser.add_argument(
        "--allow-legacy-k0",
        action="store_true",
        help=(
            "Allow diagnostic postprocessing of pre-staggered NFGS k0 caches. "
            "The default rejects those caches as paper-facing evidence."
        ),
    )
    parser.add_argument(
        "--allow-legacy-vertical-current",
        action="store_true",
        help=(
            "Allow diagnostic-only postprocessing of collocated-centered "
            "NFGS current caches. Production output requires staggered_face."
        ),
    )
    parser.add_argument(
        "--nfpt-variant", default="nfpt_fortran_symmetric"
    )
    parser.add_argument(
        "--extended-volume-metrics",
        action="store_true",
        help=(
            "Compute common six-face free energy and an explicitly labelled "
            "Helmholtz non-solenoidal energy fraction for every cached field."
        ),
    )
    parser.add_argument(
        "--extended-volume-weights",
        default="0,0.4,1",
        help="NFGS rho subset for expensive six-face/Helmholtz volume diagnostics.",
    )
    parser.add_argument(
        "--nfpt-protocol-only",
        action="store_true",
        help=(
            "Write only the NFPT cached-native row on the primary protocol "
            "domain for TD1--TD3. This memory-bounded path skips reference "
            "diagnostic rows, NFGS weights, common-curl/profile diagnostics, "
            "release caches, component tables, and figures."
        ),
    )
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def weight_token(weight: float) -> str:
    return f"w{weight:.1f}".replace(".", "p")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_npz_crop(
    path: Path,
    keys: Iterable[str],
    slices: tuple[slice, slice, slice],
) -> dict[str, np.ndarray]:
    """Load selected NPZ components one at a time and retain only ``slices``.

    The frozen caches are compressed NPZ files, so NumPy must inflate a whole
    component before it can slice it.  Copying each requested crop immediately
    lets the full component be released before the next one is inflated.
    """
    selected: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for key in keys:
            full = np.asarray(data[key])
            selected[key] = np.array(full[slices], copy=True)
            del full
    return selected


def fd2_axis(array: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Second-order centered derivative with second-order one-sided edges."""
    values = np.asarray(array)
    moved = np.moveaxis(values, axis, 0)
    out = np.empty_like(moved)
    n = moved.shape[0]
    if n < 3:
        if n == 1:
            out[...] = 0
        else:
            out[0] = (moved[1] - moved[0]) / spacing
            out[1] = out[0]
        return np.moveaxis(out, 0, axis)
    out[1:-1] = (moved[2:] - moved[:-2]) / (2.0 * spacing)
    out[0] = (-3.0 * moved[0] + 4.0 * moved[1] - moved[2]) / (2.0 * spacing)
    out[-1] = (3.0 * moved[-1] - 4.0 * moved[-2] + moved[-3]) / (2.0 * spacing)
    return np.moveaxis(out, 0, axis)


def curl_fd2(fields: dict[str, np.ndarray], spacing: float) -> dict[str, np.ndarray]:
    bx, by, bz = fields["Bx"], fields["By"], fields["Bz"]
    return {
        "Jx": fd2_axis(bz, spacing, 1) - fd2_axis(by, spacing, 0),
        "Jy": fd2_axis(bx, spacing, 0) - fd2_axis(bz, spacing, 2),
        "Jz": fd2_axis(by, spacing, 2) - fd2_axis(bx, spacing, 1),
    }


def divergence_fd2(fields: dict[str, np.ndarray], spacing: float) -> np.ndarray:
    return (
        fd2_axis(fields["Bx"], spacing, 2)
        + fd2_axis(fields["By"], spacing, 1)
        + fd2_axis(fields["Bz"], spacing, 0)
    )


def divergence_fd2_npz_crop(
    path: Path,
    spacing: float,
    slices: tuple[slice, slice, slice],
) -> np.ndarray:
    """Evaluate the unchanged full-grid FD2 divergence, retaining one crop.

    Derivatives are formed before slicing, so ROI-edge values use precisely
    the same outside-ROI stencil points as :func:`divergence_fd2`.  Components
    are processed sequentially to bound the peak resident memory on TD3.
    """
    result: np.ndarray | None = None
    with np.load(path) as data:
        for key, axis in (("Bx", 2), ("By", 1), ("Bz", 0)):
            full = np.asarray(data[key])
            derivative = fd2_axis(full, spacing, axis)
            cropped = np.array(derivative[slices], copy=True)
            if result is None:
                result = cropped
            else:
                result += cropped
            del full, derivative, cropped
    if result is None:  # pragma: no cover - fixed nonempty component list
        raise RuntimeError(f"No magnetic-field components found in {path}")
    return result


def slice_from_triplet(item: list[int | None]) -> slice:
    return slice(item[0], item[1], item[2])


def domain_slices(meta: dict) -> dict[str, tuple[slice, slice, slice]]:
    nx, ny, nz = (int(meta["dims"][0]), int(meta["dims"][1]), int(meta["dims"][2]))
    protocol = tuple(slice_from_triplet(item) for item in meta["metric_slices"])
    def centered_half(n: int) -> slice:
        width = n // 2
        start = (n - width) // 2
        return slice(start, start + width)

    half_y = centered_half(ny)
    half_x = centered_half(nx)
    # A descriptive ROI sensitivity check: the lower 30% in z and central
    # halves in x/y.  It is never treated as a preferred physical estimator.
    core = (
        slice(0, max(1, int(round(0.30 * nz)))),
        half_y,
        half_x,
    )
    central_half = (
        slice(None),
        half_y,
        half_x,
    )
    return {
        "full": (slice(None), slice(None), slice(None)),
        "central_half": central_half,
        "protocol": protocol,  # type: ignore[dict-item]
        "fractional_core": core,
    }


def metric_shape(shape: tuple[int, int, int], slices: tuple[slice, slice, slice]) -> str:
    dims = []
    for n, selection in zip(shape, slices, strict=True):
        start, stop, step = selection.indices(n)
        dims.append(len(range(start, stop, step)))
    return "x".join(str(n) for n in dims)


def compute_metrics(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    divergence: np.ndarray,
    slices: tuple[slice, slice, slice],
    spacing: float,
) -> dict[str, float]:
    """Compute metrics plane by plane to avoid large float64 temporaries."""
    sz, sy, sx = slices
    nz = reference["Bx"].shape[0]
    z_indices = range(*sz.indices(nz))
    dot_sum = ref2_sum = sol2_sum = 0.0
    cosine_sum = error_ratio_sum = 0.0
    error_sum = refnorm_sum = 0.0
    sigma_num = sigma_den = 0.0
    div_abs_sum = bnorm_sum = 0.0
    count = 0
    for iz in z_indices:
        bref = np.stack(
            [reference[key][iz, sy, sx] for key in ("Bx", "By", "Bz")], axis=-1
        ).astype(np.float64, copy=False)
        bsol = np.stack(
            [candidate[key][iz, sy, sx] for key in ("Bx", "By", "Bz")], axis=-1
        ).astype(np.float64, copy=False)
        jsol = np.stack(
            [current[key][iz, sy, sx] for key in ("Jx", "Jy", "Jz")], axis=-1
        ).astype(np.float64, copy=False)
        bn_ref = np.linalg.norm(bref, axis=-1)
        bn_sol = np.linalg.norm(bsol, axis=-1)
        jn = np.linalg.norm(jsol, axis=-1)
        diff = np.linalg.norm(bsol - bref, axis=-1)
        dot = np.sum(bref * bsol, axis=-1)
        dot_sum += float(np.sum(dot))
        ref2_sum += float(np.sum(bref * bref))
        sol2_sum += float(np.sum(bsol * bsol))
        cosine_sum += float(np.sum(dot / (bn_ref * bn_sol + 1.0e-12)))
        error_sum += float(np.sum(diff))
        refnorm_sum += float(np.sum(bn_ref))
        error_ratio_sum += float(np.sum(diff / (bn_ref + 1.0e-12)))
        sigma_num += float(
            np.sum(np.linalg.norm(np.cross(jsol, bsol), axis=-1) / np.maximum(bn_sol, 1.0e-10))
        )
        sigma_den += float(np.sum(jn))
        div_abs_sum += float(np.sum(np.abs(divergence[iz, sy, sx], dtype=np.float64)))
        bnorm_sum += float(np.sum(bn_sol))
        count += int(bn_sol.size)
    return {
        "C_vec": dot_sum / (math.sqrt(ref2_sum * sol2_sum) + 1.0e-12),
        "C_CS": cosine_sum / max(count, 1),
        "Eprime_n": 1.0 - error_sum / (refnorm_sum + 1.0e-12),
        "Eprime_m": 1.0 - error_ratio_sum / max(count, 1),
        "epsilon": sol2_sum / (ref2_sum + 1.0e-12),
        "sigma_J": sigma_num / max(sigma_den, 1.0e-30),
        "divB_normalized": (div_abs_sum / max(count, 1))
        / (6.0 * (bnorm_sum / max(count, 1)) / spacing + 1.0e-12),
    }


def bgt_relative_error(
    reference_bgt: np.ndarray,
    candidate_bgt: np.ndarray,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
) -> float:
    reference_bgt = reference_bgt[:, y_slice, x_slice]
    candidate_bgt = candidate_bgt[:, y_slice, x_slice]
    ref_mag = np.linalg.norm(reference_bgt, axis=0)
    mask = ref_mag > 1.0e-8 * (float(np.max(ref_mag)) + 1.0e-30)
    return float(
        np.mean(np.linalg.norm(candidate_bgt - reference_bgt, axis=0)[mask] / ref_mag[mask])
    )


def profile_and_budgets(
    case: str,
    method: str,
    fields: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    spacing: float,
    slices: tuple[slice, slice, slice],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sz, sy, sx = slices
    z_start = sz.indices(fields["Bx"].shape[0])[0]
    b = np.stack(
        [fields[key][sz, sy, sx] for key in ("Bx", "By", "Bz")], axis=-1
    ).astype(np.float64, copy=False)
    j = np.stack(
        [current[key][sz, sy, sx] for key in ("Jx", "Jy", "Jz")], axis=-1
    ).astype(np.float64, copy=False)
    bn = np.linalg.norm(b, axis=-1)
    jn = np.linalg.norm(j, axis=-1)
    numerator = np.linalg.norm(np.cross(j, b), axis=-1) / np.maximum(bn, 1.0e-10)
    total_num = float(np.sum(numerator))
    total_j = float(np.sum(jn))
    profiles: list[dict[str, object]] = []
    for iz in range(b.shape[0]):
        plane_num = float(np.sum(numerator[iz]))
        plane_j = float(np.sum(jn[iz]))
        profiles.append(
            {
                "benchmark": case,
                "method": method,
                "z_index": iz,
                "z_Mm": (z_start + iz) * spacing,
                "sigma_J_plane": plane_num / max(plane_j, 1.0e-30),
                "numerator_fraction": plane_num / max(total_num, 1.0e-30),
                "current_fraction": plane_j / max(total_j, 1.0e-30),
            }
        )

    nz, ny, nx = numerator.shape
    labels = np.full((nz, ny, nx), "bulk", dtype="U12")
    labels[:, :2, :] = "lateral"
    labels[:, -2:, :] = "lateral"
    labels[:, :, :2] = "lateral"
    labels[:, :, -2:] = "lateral"
    labels[-2:, :, :] = "top"
    labels[1, :, :] = "first_row"
    labels[0, :, :] = "bottom"
    budgets: list[dict[str, object]] = []
    for region in ("bottom", "first_row", "lateral", "top", "bulk"):
        mask = labels == region
        region_num = float(np.sum(numerator[mask]))
        region_j = float(np.sum(jn[mask]))
        budgets.append(
            {
                "benchmark": case,
                "method": method,
                "region": region,
                "sigma_J_region": region_num / max(region_j, 1.0e-30),
                "numerator_fraction": region_num / max(total_num, 1.0e-30),
                "current_fraction": region_j / max(total_j, 1.0e-30),
            }
        )
    return profiles, budgets


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def component_rows(source_metrics: Path) -> list[dict[str, object]]:
    records = json.loads(source_metrics.read_text())
    output = []
    for record in records:
        diag = record["source_diagnostics"]
        coupled = diag.get("coupled_k0", {})
        top_derivative = coupled.get("top_derivative") or []
        profile_by_height = coupled.get("profile_by_height") or []
        output.append(
            {
                "benchmark": record["benchmark"],
                "bottom_reconstruction_rms": diag.get("bottom_reconstruction_rms"),
                "flux_imbalance_fraction": diag.get("flux_imbalance_fraction"),
                "Bz_open_mean": diag.get("Bz_open_mean"),
                "dct_zero_mode_Bx": diag.get("dct_zero_mode_Bx"),
                "dct_zero_mode_By": diag.get("dct_zero_mode_By"),
                "dct_zero_mode_Bz": diag.get("dct_zero_mode_Bz"),
                "Bh_mean_bottom_x": (diag.get("Bh_mean_bottom") or [None, None])[0],
                "Bh_mean_bottom_y": (diag.get("Bh_mean_bottom") or [None, None])[1],
                "bottom_horizontal_reconstruction_ratio": diag.get(
                    "bottom_horizontal_reconstruction_ratio"
                ),
                "harmonic_residual_nonconstant_rms": diag.get(
                    "harmonic_residual_nonconstant_rms"
                ),
                "source_zero_mode_ratio_max": diag.get("source_zero_mode_ratio_max"),
                "H_discretization": coupled.get("discretization"),
                "H_profile_max": coupled.get("profile_max_abs"),
                "H_ode_residual_rms": coupled.get("ode_residual_rms"),
                "H_face_ode_residual_rms": coupled.get("face_ode_residual_rms"),
                "H_face_ode_residual_relative": coupled.get(
                    "face_ode_residual_relative"
                ),
                "H_gridscale_profile_rms": coupled.get("gridscale_profile_rms"),
                "H_gridscale_profile_relative": coupled.get(
                    "gridscale_profile_relative"
                ),
                "H_top_derivative_rms": (
                    math.sqrt(
                        sum(float(value) ** 2 for value in top_derivative)
                        / len(top_derivative)
                    )
                    if top_derivative
                    else None
                ),
                "mean_field_profile_points": (
                    len(profile_by_height[0]) if profile_by_height else 0
                ),
            }
        )
    return output


def _current_weighted_height_bins(
    rows: list[dict[str, object]], bin_planes: int
) -> tuple[list[float], list[float]]:
    if bin_planes < 1:
        raise ValueError("bin_planes must be >= 1")
    ordered = sorted(rows, key=lambda row: int(row["z_index"]))
    z_out: list[float] = []
    sigma_out: list[float] = []
    # Preserve the bottom and modified first-interior planes as individual
    # points.  Aggregate only the remaining bulk using the exact current
    # weighting in sigma_J rather than smoothing the magnetic field.
    groups = [[row] for row in ordered[:2]]
    bulk = ordered[2:]
    groups.extend(
        bulk[start : start + bin_planes]
        for start in range(0, len(bulk), bin_planes)
    )
    for group in groups:
        weights = np.asarray(
            [float(row["current_fraction"]) for row in group], dtype=float
        )
        sigma = np.asarray(
            [float(row["sigma_J_plane"]) for row in group], dtype=float
        )
        z = np.asarray([float(row["z_Mm"]) for row in group], dtype=float)
        total = float(np.sum(weights))
        if total <= 1.0e-30:
            continue
        z_out.append(float(np.sum(weights * z) / total))
        sigma_out.append(float(np.sum(weights * sigma) / total))
    return z_out, sigma_out


def plot_profiles(
    rows: list[dict[str, object]], path: Path, *, bin_planes: int = 4
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25), sharey=True)
    styles = {
        "analytic_reference_fd2": ("black", "-"),
        "nfpt_fortran": ("#1f77b4", "--"),
        "nfpt_fortran_symmetric": ("#1f77b4", "--"),
        "nfgs_rho_0.0": ("#999999", ":"),
        "nfgs_rho_0.4": ("#d95f02", "-"),
        "nfgs_rho_1.0": ("#7570b3", "-."),
    }
    for ax, case in zip(axes, CASES, strict=True):
        methods = sorted({str(row["method"]) for row in rows if row["benchmark"] == case})
        for method in methods:
            group = [row for row in rows if row["benchmark"] == case and row["method"] == method]
            group.sort(key=lambda row: int(row["z_index"]))
            color, linestyle = styles.get(method, (None, "-"))
            label = method.replace("analytic_reference_fd2", "analytic FD2").replace(
                "nfpt_fortran_symmetric", "NFPT control"
            ).replace("nfpt_fortran", "NFPT control").replace(
                "nfgs_rho_", r"NFGS $\rho=$"
            )
            ax.plot(
                [float(row["z_Mm"]) for row in group],
                [float(row["sigma_J_plane"]) for row in group],
                color=color,
                linestyle=linestyle,
                linewidth=0.55,
                alpha=0.22,
            )
            z_bin, sigma_bin = _current_weighted_height_bins(group, bin_planes)
            ax.plot(
                z_bin,
                sigma_bin,
                color=color,
                linestyle=linestyle,
                linewidth=1.45,
                label=label,
            )
        ax.set_title(case.upper())
        ax.set_xlabel("Height [Mm]")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel(r"plane $\sigma_J(z)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7.5, frameon=False)
    fig.text(
        0.5,
        0.015,
        f"thin: raw planes; thick: current-weighted bins ({bin_planes} bulk planes)",
        ha="center",
        fontsize=7.0,
        color="0.35",
    )
    fig.subplots_adjust(top=0.80, bottom=0.18, wspace=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_nfpt_protocol_only(
    args: argparse.Namespace,
    bgt_args: SimpleNamespace,
) -> None:
    """Write the three primary-domain NFPT rows with bounded TD3 memory."""
    metric_rows: list[dict[str, object]] = []
    full_crop = (slice(None), slice(None), slice(None))
    for case in CASES:
        meta = json.loads((args.case_meta_dir / f"{case}__case_meta.json").read_text())
        nx, ny, _nz, dx, dy, dz = meta["dims"]
        if not (math.isclose(dx, dy) and math.isclose(dx, dz)):
            raise ValueError(f"Expected isotropic TD grid for {case}: {meta['dims']}")
        spacing = float(dx)
        protocol = domain_slices(meta)["protocol"]
        reference_path = args.field_dir / f"{case}__reference.npz"
        candidate_path = (
            (args.nfpt_field_dir or args.field_dir)
            / f"{case}__{args.nfpt_variant}.npz"
        )
        print(f"[{case}] {args.nfpt_variant}: {candidate_path.name}", flush=True)

        reference = load_npz_crop(
            reference_path, ("Bx", "By", "Bz"), protocol
        )
        candidate = load_npz_crop(
            candidate_path,
            ("Bx", "By", "Bz", "Jx", "Jy", "Jz"),
            protocol,
        )
        native_current = {key: candidate[key] for key in ("Jx", "Jy", "Jz")}
        divergence = divergence_fd2_npz_crop(candidate_path, spacing, protocol)

        # B^> is defined from the complete lower plane.  Load that plane only;
        # its horizontal-domain restrictions are applied after separation just
        # as in the default path.
        bottom = (slice(0, 1), slice(None), slice(None))
        reference_bottom = load_npz_crop(
            reference_path, ("Bx", "By", "Bz"), bottom
        )
        candidate_bottom = load_npz_crop(
            candidate_path, ("Bx", "By", "Bz"), bottom
        )
        helper = make_bgt_helper_solver(
            nx=int(nx),
            ny=int(ny),
            dx=spacing,
            dy=spacing,
            dz=spacing,
            device="cpu",
            dtype=torch.float32,
            args=bgt_args,
        )
        reference_bgt = bottom_plane_bgt_nfgs(helper, reference_bottom)
        candidate_bgt = bottom_plane_bgt_nfgs(helper, candidate_bottom)
        _sz, sy, sx = protocol
        values = compute_metrics(
            reference,
            candidate,
            native_current,
            divergence,
            full_crop,
            spacing,
        )
        metric_rows.append(
            {
                "benchmark": case,
                "method": args.nfpt_variant,
                "rho": "",
                "domain": "protocol",
                "shape_zyx": metric_shape(
                    tuple(int(value) for value in reference["Bx"].shape), full_crop
                ),
                "current_source": "cached_native",
                "E_rel_Bgt_domain": bgt_relative_error(
                    reference_bgt, candidate_bgt, y_slice=sy, x_slice=sx
                ),
                "E_rel_Bgt_full_bottom": bgt_relative_error(
                    reference_bgt, candidate_bgt
                ),
                **values,
            }
        )
        del (
            helper,
            reference,
            candidate,
            native_current,
            divergence,
            reference_bottom,
            candidate_bottom,
            reference_bgt,
            candidate_bgt,
        )
        gc.collect()

    write_csv(args.output_dir / "metrics_by_domain.csv", metric_rows)
    manifest = {
        "status": "complete",
        "algorithm": "cache-only common-FD2 referee diagnostics",
        "mode": "nfpt_protocol_only",
        "field_dir": str(args.field_dir.resolve()),
        "case_meta_dir": str(args.case_meta_dir.resolve()),
        "nfpt_field_dir": str((args.nfpt_field_dir or args.field_dir).resolve()),
        "nfpt_variant": args.nfpt_variant,
        "benchmarks": list(CASES),
        "domain": "protocol",
        "current_source": "cached_native",
        "derivative_operator": "FD2 centered interior, FD2 one-sided boundary",
        "notes": [
            "No nonlinear iteration was run.",
            "Metric definitions are identical to the corresponding default-path rows.",
            "Full-grid FD2 derivatives are evaluated before the protocol crop is retained.",
            "Only metrics_by_domain.csv is emitted in this focused mode.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    weights = parse_float_list(args.weights)
    profile_weights = set(parse_float_list(args.profile_weights))
    extended_volume_weights = set(parse_float_list(args.extended_volume_weights))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    release_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []

    bgt_args = SimpleNamespace(
        mean_bz_policy="uniform_open",
        horizontal_zero_mode_policy=args.horizontal_zero_mode_policy,
        d0_zero_mode_policy="project",
        source_zero_mode_policy="project",
        harmonic_residual_policy="diagnose",
        mean_mode_tolerance=1.0e-6,
        source_zero_mode_tolerance=1.0e-2,
        gaussian_horizontal_operator="spectral",
    )

    if args.nfpt_protocol_only:
        run_nfpt_protocol_only(args, bgt_args)
        return

    if args.profile_bin_planes < 1:
        raise ValueError("--profile-bin-planes must be >= 1")
    source_records = json.loads(args.source_metrics.read_text())
    source_policies = sorted(
        {str(record.get("horizontal_zero_mode_policy", "legacy_unspecified")) for record in source_records}
    )
    if source_policies != [args.horizontal_zero_mode_policy]:
        raise RuntimeError(
            "Source-cache zero-mode provenance does not match the requested protocol: "
            f"expected={args.horizontal_zero_mode_policy!r}, found={source_policies}"
        )
    k0_discretizations = sorted(
        {
            str(record.get("source_diagnostics", {}).get("coupled_k0", {}).get("discretization"))
            for record in source_records
            if record.get("source_diagnostics", {}).get("coupled_k0")
        }
    )
    corrected_k0 = k0_discretizations == ["staggered_face_D1_true_three_point_curvature"]
    if (
        args.horizontal_zero_mode_policy == "coupled_k0"
        and not corrected_k0
        and not args.allow_legacy_k0
    ):
        raise RuntimeError(
            "Refusing paper-facing diagnostics from legacy coupled-k0 caches. "
            "Rerun the NFGS profiles with the staggered k0 solver, or use "
            "--allow-legacy-k0 for diagnostic-only output."
        )
    vertical_current_operators = sorted(
        {str(record.get("vertical_current_operator", "legacy_unspecified"))
         for record in source_records}
    )
    if (
        vertical_current_operators != ["staggered_face"]
        and not args.allow_legacy_vertical_current
    ):
        raise RuntimeError(
            "Refusing paper-facing diagnostics from collocated or unspecified "
            "vertical-current caches. Rerun with --vertical-current-operator "
            "staggered_face, or explicitly allow diagnostic-only legacy output."
        )

    for case in CASES:
        meta = json.loads((args.case_meta_dir / f"{case}__case_meta.json").read_text())
        nx, ny, nz, dx, dy, dz = meta["dims"]
        if not (math.isclose(dx, dy) and math.isclose(dx, dz)):
            raise ValueError(f"Expected isotropic TD grid for {case}: {meta['dims']}")
        spacing = float(dx)
        all_domains = domain_slices(meta)
        domains = (
            {"protocol": all_domains["protocol"]}
            if args.tradeoff_only
            else all_domains
        )
        reference = load_npz(args.field_dir / f"{case}__reference.npz")
        reference_current = curl_fd2(reference, spacing)
        reference_div = divergence_fd2(reference, spacing)
        helper = make_bgt_helper_solver(
            nx=int(nx),
            ny=int(ny),
            dx=spacing,
            dy=spacing,
            dz=spacing,
            device="cpu",
            dtype=torch.float32,
            args=bgt_args,
        )
        reference_bgt = bottom_plane_bgt_nfgs(helper, reference)
        reference_energy: dict[str, dict[str, float | str]] = {}
        if args.extended_volume_metrics and not args.tradeoff_only:
            reference_potential, reference_flux = cell_center_potential(reference, spacing)
            reference_ns, reference_projection = helmholtz_non_solenoidal_component(
                reference, spacing
            )
            for domain_name, slices in domains.items():
                values = energy_diagnostics(
                    reference,
                    spacing,
                    slices=slices,
                    potential=reference_potential,
                    non_solenoidal=reference_ns,
                    flux_imbalance=reference_flux,
                )
                values.update(reference_projection)
                reference_energy[domain_name] = values
                energy_rows.append(
                    {
                        "benchmark": case,
                        "method": "analytic_reference",
                        "rho": "",
                        "domain": domain_name,
                        "free_energy_relative_error": 0.0,
                        **values,
                    }
                )
            del reference_potential, reference_ns
            gc.collect()

        if not args.tradeoff_only:
            for domain_name, slices in domains.items():
                values = compute_metrics(
                    reference,
                    reference,
                    reference_current,
                    reference_div,
                    slices,
                    spacing,
                )
                metric_rows.append(
                    {
                        "benchmark": case,
                        "method": "analytic_reference_fd2",
                        "rho": "",
                        "domain": domain_name,
                        "shape_zyx": metric_shape(reference["Bx"].shape, slices),
                        "current_source": "common_fd2_curl",
                        "E_rel_Bgt_domain": 0.0,
                        "E_rel_Bgt_full_bottom": 0.0,
                        **values,
                    }
                )
            profiles, budgets = profile_and_budgets(
                case,
                "analytic_reference_fd2",
                reference,
                reference_current,
                spacing,
                domains["protocol"],
            )
            profile_rows.extend(profiles)
            budget_rows.extend(budgets)

        candidate_specs: list[tuple[str, float | None, Path]] = [
            (
                args.nfpt_variant,
                None,
                (args.nfpt_field_dir or args.field_dir)
                / f"{case}__{args.nfpt_variant}.npz",
            )
        ]
        candidate_specs.extend(
            (
                f"nfgs_rho_{weight:.1f}",
                weight,
                args.field_dir
                / f"{case}__nfgs__gs_cfcf__robin_{weight_token(weight)}.npz",
            )
            for weight in weights
        )

        for method, weight, path in candidate_specs:
            print(f"[{case}] {method}: {path.name}", flush=True)
            candidate = load_npz(path)
            native_current = {key: candidate[key] for key in ("Jx", "Jy", "Jz")}
            common_current = curl_fd2(candidate, spacing)
            div = divergence_fd2(candidate, spacing)
            candidate_bgt = bottom_plane_bgt_nfgs(helper, candidate)
            erel_full = bgt_relative_error(reference_bgt, candidate_bgt)
            for domain_name, slices in domains.items():
                _sz, sy, sx = slices
                erel_domain = bgt_relative_error(
                    reference_bgt, candidate_bgt, y_slice=sy, x_slice=sx
                )
                values = compute_metrics(
                    reference, candidate, native_current, div, slices, spacing
                )
                metric_rows.append(
                    {
                        "benchmark": case,
                        "method": method,
                        "rho": "" if weight is None else weight,
                        "domain": domain_name,
                        "shape_zyx": metric_shape(reference["Bx"].shape, slices),
                        "current_source": "cached_native",
                        "E_rel_Bgt_domain": erel_domain,
                        "E_rel_Bgt_full_bottom": erel_full,
                        **values,
                    }
                )

                common_values = compute_metrics(
                    reference, candidate, common_current, div, slices, spacing
                )
                metric_rows.append(
                    {
                        "benchmark": case,
                        "method": method,
                        "rho": "" if weight is None else weight,
                        "domain": domain_name,
                        "shape_zyx": metric_shape(reference["Bx"].shape, slices),
                        "current_source": "common_fd2_curl",
                        "E_rel_Bgt_domain": erel_domain,
                        "E_rel_Bgt_full_bottom": erel_full,
                        **common_values,
                    }
                )

            selected = (
                not args.tradeoff_only
                and (weight is None or weight in profile_weights)
            )
            if selected:
                profiles, budgets = profile_and_budgets(
                    case,
                    method,
                    candidate,
                    common_current,
                    spacing,
                    domains["protocol"],
                )
                profile_rows.extend(profiles)
                budget_rows.extend(budgets)
            if (
                args.extended_volume_metrics
                and not args.tradeoff_only
                and (weight is None or weight in extended_volume_weights)
            ):
                candidate_potential, candidate_flux = cell_center_potential(
                    candidate, spacing
                )
                candidate_ns, candidate_projection = helmholtz_non_solenoidal_component(
                    candidate, spacing
                )
                for domain_name, slices in domains.items():
                    values = energy_diagnostics(
                        candidate,
                        spacing,
                        slices=slices,
                        potential=candidate_potential,
                        non_solenoidal=candidate_ns,
                        flux_imbalance=candidate_flux,
                    )
                    values.update(candidate_projection)
                    reference_free = float(
                        reference_energy[domain_name]["free_energy_6face"]
                    )
                    energy_rows.append(
                        {
                            "benchmark": case,
                            "method": method,
                            "rho": "" if weight is None else weight,
                            "domain": domain_name,
                            "free_energy_relative_error": abs(
                                float(values["free_energy_6face"]) - reference_free
                            ) / max(abs(reference_free), 1.0e-30),
                            **values,
                        }
                    )
                del candidate_potential, candidate_ns
                gc.collect()
            del candidate, native_current, common_current, div, candidate_bgt

        for weight in (() if args.tradeoff_only else weights):
            release_path = (
                args.release_field_dir
                / f"robin_{weight_token(weight)}__{case}__post_best_sigma.npz"
            )
            if not release_path.exists():
                continue
            released = load_npz(release_path)
            released_current = {
                key: released[key] for key in ("Jx", "Jy", "Jz")
            }
            released_div = divergence_fd2(released, spacing)
            released_bgt = bottom_plane_bgt_nfgs(helper, released)
            protocol = domains["protocol"]
            _sz, sy, sx = protocol
            values = compute_metrics(
                reference,
                released,
                released_current,
                released_div,
                protocol,
                spacing,
            )
            release_rows.append(
                {
                    "benchmark": case,
                    "initial_rho": weight,
                    "released_rho": 0.0,
                    "domain": "protocol",
                    "shape_zyx": metric_shape(reference["Bx"].shape, protocol),
                    "current_source": "cached_native",
                    "E_rel_Bgt_domain": bgt_relative_error(
                        reference_bgt, released_bgt, y_slice=sy, x_slice=sx
                    ),
                    "E_rel_Bgt_full_bottom": bgt_relative_error(
                        reference_bgt, released_bgt
                    ),
                    **values,
                }
            )
            del released, released_current, released_div, released_bgt

        del helper, reference, reference_current, reference_div, reference_bgt

    write_csv(args.output_dir / "metrics_by_domain.csv", metric_rows)
    write_csv(
        args.output_dir / "headline_common_fd2_protocol.csv",
        [
            row for row in metric_rows
            if row["domain"] == "protocol"
            and row["current_source"] == "common_fd2_curl"
        ],
    )
    if args.tradeoff_only:
        manifest = {
            "algorithm": "cache-only TD imprint-tradeoff diagnostics",
            "mode": "tradeoff_only",
            "field_dir": str(args.field_dir.resolve()),
            "case_meta_dir": str(args.case_meta_dir.resolve()),
            "nfpt_field_dir": str((args.nfpt_field_dir or args.field_dir).resolve()),
            "weights": weights,
            "horizontal_zero_mode_policy": args.horizontal_zero_mode_policy,
            "nfgs_vertical_current_operators": vertical_current_operators,
            "nfpt_variant": args.nfpt_variant,
            "domain": (
                "primary paper domains: central horizontal half through full z for "
                "TD1/TD3; Demcsak flux-rope-centered 66x66x30 for TD2"
            ),
            "current_source": "cached_native",
            "notes": [
                "No nonlinear iteration was run in this postprocessor.",
                "Only rows required by figure_nfgs_td_imprint_tradeoff.png were written.",
                "No other figure type was generated.",
            ],
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"Wrote {args.output_dir}", flush=True)
        return
    write_csv(args.output_dir / "release_metrics_primary.csv", release_rows)
    write_csv(args.output_dir / "height_profiles.csv", profile_rows)
    write_csv(args.output_dir / "force_free_region_budgets.csv", budget_rows)
    if args.extended_volume_metrics:
        write_csv(args.output_dir / "energy_divergence_by_domain.csv", energy_rows)
    write_csv(
        args.output_dir / "represented_component_diagnostics.csv",
        component_rows(args.source_metrics),
    )
    figure_path = args.figure_dir / "figure_nfgs_td_force_free_height_profiles.png"
    plot_profiles(profile_rows, figure_path, bin_planes=args.profile_bin_planes)
    plot_profiles(
        profile_rows,
        args.output_dir / figure_path.name,
        bin_planes=args.profile_bin_planes,
    )
    manifest = {
        "status": "complete",
        "algorithm": "cache-only common-FD2 referee diagnostics",
        "field_dir": str(args.field_dir.resolve()),
        "case_meta_dir": str(args.case_meta_dir.resolve()),
        "release_field_dir": str(args.release_field_dir.resolve()),
        "weights": weights,
        "profile_weights": sorted(profile_weights),
        "profile_bin_planes": int(args.profile_bin_planes),
        "horizontal_zero_mode_policy": args.horizontal_zero_mode_policy,
        "nfgs_k0_discretizations": k0_discretizations,
        "nfgs_vertical_current_operators": vertical_current_operators,
        "legacy_k0_allowed": bool(args.allow_legacy_k0),
        "legacy_vertical_current_allowed": bool(
            args.allow_legacy_vertical_current
        ),
        "nfpt_variant": args.nfpt_variant,
        "nfpt_field_dir": str(args.nfpt_field_dir or args.field_dir),
        "domains": {
            "full": "complete observed/reference array",
            "central_half": "full z and central Nx0/2 by Ny0/2 flux-rope-containing domain",
            "protocol": (
                "primary paper domains: central horizontal half through full z for "
                "TD1/TD3; Demcsak flux-rope-centered 66x66x30 for TD2"
            ),
            "fractional_core": "lower 30% z and central half x/y; descriptive sensitivity only",
        },
        "derivative_operator": "FD2 centered interior, FD2 one-sided boundary",
        "headline_current_source": "common_fd2_curl",
        "extended_volume_metrics": bool(args.extended_volume_metrics),
        "extended_volume_weights": sorted(extended_volume_weights),
        "notes": [
            "No nonlinear iteration was run.",
            "Candidate native-current rows use cached J; common_fd2_curl rows recompute J from cached B.",
            "Analytic-reference current is the common FD2 curl of the sampled field.",
            "E_rel_Bgt_domain uses the horizontal footprint of each listed 3D domain.",
            "E_rel_Bgt_full_bottom uses the common spectral Gaussian-separation implementation on the full lower plane.",
            "release_metrics_primary.csv recomputes cached post-release states on the frozen primary domains.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
