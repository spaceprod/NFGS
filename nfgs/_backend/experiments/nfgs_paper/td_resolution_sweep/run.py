#!/usr/bin/env python3
"""Run an NFGS TD resolution sweep on fixed *physical* evaluation volumes.

This runner is deliberately separate from the production TD reconstruction and
NFGS ablation runners.  It re-evaluates the continuous TD field on each target
lattice (rather than interpolating a stored array), keeps the full physical
box fixed, and maps the reference-resolution metric-volume cell edges onto the
new lattice.  Consequently a resolution change cannot silently expand or
shrink the scientific evaluation region in index space.

Use ``--contract-only`` for a lightweight, solver-free audit of all lattice and
metric-domain mappings.  A normal invocation performs the nonlinear solves and
writes collision-resistant, configuration-hashed artifacts.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.scripts import run_analytic_validation as analytic  # noqa: E402
from nfgs._backend.scripts import run_nfgs_analytic_validation as driver  # noqa: E402
from nfgs._backend.td import generate_td_model  # noqa: E402
from nfgs._backend.util.field_ops import curl_zyx_nfpt  # noqa: E402
from nfgs._backend.scripts.plot_nfgs_td_robin_metrics_vs_er import (  # noqa: E402
    bottom_bgt_relative_error,
    make_bgt_helper_solver,
)
from nfgs.diagnostics.common_fd2 import sigma_j as common_fd2_sigma_j  # noqa: E402


TD_NAMES = ("td1", "td2", "td3")
METRIC_KEYS = tuple(analytic.METRIC_KEYS)
DEFAULT_RATIOS = (0.5, 0.75, 1.0)

# These are the frozen reference-lattice analysis domains.  TD1/TD3 are the
# central Nx0/2 x Ny0/2 x Nz volumes.  TD2 adopts the 66 x 66 x 30 grid-point
# count of the Demcsak et al. restricted analysis protocol and centres it on
# the |Jz| centroid (iy, ix)=(124, 74); it does not copy their physical box.
REFERENCE_SLICES_ZYX: dict[str, tuple[tuple[int, int], ...]] = {
    "td1": ((0, 100), (63, 188), (38, 113)),
    "td2": ((0, 30), (91, 157), (41, 107)),
    "td3": ((0, 200), (88, 263), (64, 192)),
}
REFERENCE_SHAPES_ZYX = {
    "td1": (100, 250, 150),
    "td2": (100, 250, 150),
    "td3": (200, 350, 256),
}
TD2_REFERENCE_CENTER_YX = (124, 74)

# Fixed across every resolution, precision, and benchmark in this runner.
FIXED_NUMERICS: dict[str, Any] = {
    "init_mode": "gs_cfcf",
    "calculation_domain_mode": "mudpack",
    "bottom_divergence_robin_weight": 0.4,
    "horizontal_zero_mode_policy": "frozen_linear_harmonic",
    "k0_ode_smoothness": 1.0e-3,
    "poloidal_update_mode": "direct",
    "gaussian_horizontal_operator": "spectral",
    "vertical_current_operator": "staggered_face",
    "mean_bz_policy": "uniform_open",
    "d0_zero_mode_policy": "project",
    "source_zero_mode_policy": "project",
    "harmonic_residual_policy": "diagnose",
    "bottom_divergence_projection_weight": 0.0,
    "handoff_projection_policy": "project",
    "handoff_projection_tolerance": 1.0e-6,
    "mean_mode_tolerance": 1.0e-6,
    "source_zero_mode_tolerance": 1.0e-2,
    "mu": 1.0,
    "smooth_passes": 0,
    "zero_net_current": False,
    "ijzbcor": 3,
    "record_iteration_diagnostics": True,
}


def _csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_choices(value: str, allowed: Iterable[str], label: str) -> list[str]:
    allowed_set = set(allowed)
    items = list(dict.fromkeys(item.lower() for item in _csv_items(value)))
    invalid = [item for item in items if item not in allowed_set]
    if not items or invalid:
        raise argparse.ArgumentTypeError(
            f"{label} must be a nonempty comma-separated subset of "
            f"{sorted(allowed_set)}; invalid={invalid}"
        )
    return items


def _parse_ratios(value: str) -> list[float]:
    try:
        ratios = sorted({float(item) for item in _csv_items(value)})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratios must be numeric") from exc
    if not ratios or any((not math.isfinite(item)) or item <= 0.0 for item in ratios):
        raise argparse.ArgumentTypeError("ratios must be finite and > 0")
    return ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_paper"
            / "frozen_linear_harmonic_vertical_staggered"
            / "td_resolution_sweep"
        ),
    )
    parser.add_argument("--benchmarks", default="td1,td2,td3")
    parser.add_argument(
        "--ratios",
        default=",".join(str(item) for item in DEFAULT_RATIOS),
        help="Comma-separated target interval-density ratios (default: 0.5,0.75,1.0).",
    )
    parser.add_argument(
        "--dtypes",
        default="float32",
        help="Comma-separated subset of float32,float64.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--sigma-improvement-tol", type=float, default=1.0e-5)
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--stop-mode", choices=("sigma_best",), default="sigma_best")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--horizontal-zero-mode-policy",
        choices=(
            "frozen_linear_harmonic",
            "constrained_spline_harmonic",
            "coupled_k0",
            "omit",
        ),
        default=FIXED_NUMERICS["horizontal_zero_mode_policy"],
        help="Horizontal k=0 protocol; production default is frozen_linear_harmonic.",
    )
    parser.add_argument(
        "--td-calculation-domain-mode",
        choices=("mudpack", "mudpack_legacy", "bottom_padding"),
        default=FIXED_NUMERICS["calculation_domain_mode"],
        help="Symmetric MUDPACK placement is the production default.",
    )
    parser.add_argument(
        "--reference-chunk-depth",
        type=int,
        default=4,
        help="Number of z planes per continuous-TD generator chunk.",
    )
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Write and validate lattice/ROI mappings without generating fields or solving.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-field-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.benchmarks = _parse_choices(args.benchmarks, TD_NAMES, "benchmarks")
    args.ratios = _parse_ratios(args.ratios)
    args.dtypes = _parse_choices(
        args.dtypes, ("float32", "float64"), "dtypes"
    )
    if args.max_iter < 1:
        parser.error("--max-iter must be >= 1")
    if args.tol <= 0.0:
        parser.error("--tol must be > 0")
    if args.patience < 0:
        parser.error("--patience must be >= 0")
    if args.sigma_improvement_tol < 0.0:
        parser.error("--sigma-improvement-tol must be >= 0")
    if not 0.0 <= args.rho <= 1.0:
        parser.error("--rho must be in [0,1]")
    if args.reference_chunk_depth < 1:
        parser.error("--reference-chunk-depth must be >= 1")
    return args


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            analytic.json_safe(value),
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(
        analytic.json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ratio_tag(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def _target_count(reference_count: int, ratio: float) -> int:
    """Scale interval density, resolving exact half ties upward."""
    intervals = max(2, int(math.floor((reference_count - 1) * ratio + 0.5)))
    return intervals + 1


def reference_axes(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact coordinates used by the reference-resolution TD suite."""
    nz, ny, nx = REFERENCE_SHAPES_ZYX[name]
    spacing = 360.0 / float(nx - 1)
    x = np.linspace(-180.0, 180.0, nx, dtype=np.float64)
    y = spacing * (np.arange(ny, dtype=np.float64) - 0.5 * float(ny - 1))
    z = spacing * np.arange(nz, dtype=np.float64)
    return x, y, z


def target_axes(
    name: str, ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the full physical box while changing interval density."""
    x_ref, y_ref, z_ref = reference_axes(name)
    nx = _target_count(x_ref.size, ratio)
    ny = _target_count(y_ref.size, ratio)
    nz = _target_count(z_ref.size, ratio)
    return (
        np.linspace(x_ref[0], x_ref[-1], nx, dtype=np.float64),
        np.linspace(y_ref[0], y_ref[-1], ny, dtype=np.float64),
        np.linspace(z_ref[0], z_ref[-1], nz, dtype=np.float64),
    )


def _slice(start_stop: tuple[int, int]) -> slice:
    return slice(int(start_stop[0]), int(start_stop[1]))


def _spacing(axis: np.ndarray) -> float:
    if axis.size < 2:
        raise ValueError("each target axis must contain at least two samples")
    return float(axis[1] - axis[0])


def _slice_edge_bounds(axis: np.ndarray, item: slice) -> tuple[float, float]:
    start = int(item.start or 0)
    stop = int(item.stop if item.stop is not None else axis.size)
    if not (0 <= start < stop <= axis.size):
        raise ValueError(f"invalid slice {item} for axis of length {axis.size}")
    step = _spacing(axis)
    return float(axis[start] - 0.5 * step), float(axis[stop - 1] + 0.5 * step)


def _map_edge_bounds(axis: np.ndarray, bounds: tuple[float, float]) -> slice:
    """Select contiguous target centres inside a frozen reference cell-edge box."""
    lo, hi = bounds
    scale = max(1.0, abs(lo), abs(hi), float(np.max(np.abs(axis))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    indices = np.flatnonzero((axis >= lo - tolerance) & (axis < hi - tolerance))
    if indices.size == 0:
        centre = 0.5 * (lo + hi)
        nearest = int(np.argmin(np.abs(axis - centre)))
        return slice(nearest, nearest + 1)
    if not np.all(np.diff(indices) == 1):
        raise RuntimeError("physical ROI mapped to a non-contiguous target index set")
    return slice(int(indices[0]), int(indices[-1]) + 1)


def _axis_extent_record(axis: np.ndarray, item: slice) -> dict[str, Any]:
    start = int(item.start or 0)
    stop = int(item.stop if item.stop is not None else axis.size)
    edge_lo, edge_hi = _slice_edge_bounds(axis, item)
    return {
        "slice": [start, stop, None],
        "n_samples": stop - start,
        "centre_bounds_mm": [float(axis[start]), float(axis[stop - 1])],
        "cell_edge_bounds_mm": [edge_lo, edge_hi],
        "cell_edge_extent_mm": edge_hi - edge_lo,
    }


def build_domain_mapping(name: str, ratio: float) -> dict[str, Any]:
    """Describe and validate one fixed-physics ROI mapping."""
    x_ref, y_ref, z_ref = reference_axes(name)
    x, y, z = target_axes(name, ratio)
    reference_slices = tuple(_slice(item) for item in REFERENCE_SLICES_ZYX[name])
    reference_axes_zyx = (z_ref, y_ref, x_ref)
    target_axes_zyx = (z, y, x)
    frozen_bounds = tuple(
        _slice_edge_bounds(axis, item)
        for axis, item in zip(reference_axes_zyx, reference_slices)
    )
    target_slices = tuple(
        _map_edge_bounds(axis, bounds)
        for axis, bounds in zip(target_axes_zyx, frozen_bounds)
    )

    axis_names = ("z", "y", "x")
    requested: dict[str, Any] = {}
    realized: dict[str, Any] = {}
    for axis_name, ref_axis, target_axis, ref_slice, target_slice, bounds in zip(
        axis_names,
        reference_axes_zyx,
        target_axes_zyx,
        reference_slices,
        target_slices,
        frozen_bounds,
    ):
        requested[axis_name] = {
            **_axis_extent_record(ref_axis, ref_slice),
            "frozen_cell_edge_bounds_mm": [float(bounds[0]), float(bounds[1])],
        }
        target_record = _axis_extent_record(target_axis, target_slice)
        target_record["edge_mismatch_mm"] = [
            float(target_record["cell_edge_bounds_mm"][0] - bounds[0]),
            float(target_record["cell_edge_bounds_mm"][1] - bounds[1]),
        ]
        realized[axis_name] = target_record

    base_nz, base_ny, base_nx = REFERENCE_SHAPES_ZYX[name]
    result: dict[str, Any] = {
        "benchmark": name,
        "requested_ratio": float(ratio),
        "ratio_definition": "target/reference interval density on a fixed full box",
        "reference_grid_xyz": [base_nx, base_ny, base_nz],
        "target_grid_xyz": [int(x.size), int(y.size), int(z.size)],
        "realized_interval_density_ratio_xyz": [
            float((x.size - 1) / (base_nx - 1)),
            float((y.size - 1) / (base_ny - 1)),
            float((z.size - 1) / (base_nz - 1)),
        ],
        "full_box_bounds_mm": {
            "x": [float(x[0]), float(x[-1])],
            "y": [float(y[0]), float(y[-1])],
            "z": [float(z[0]), float(z[-1])],
        },
        "spacing_xyz_mm": [_spacing(x), _spacing(y), _spacing(z)],
        "reference_metric_slices_zyx": [
            [item.start, item.stop, item.step] for item in reference_slices
        ],
        "target_metric_slices_zyx": [
            [item.start, item.stop, item.step] for item in target_slices
        ],
        "target_metric_shape_zyx": [
            int(item.stop) - int(item.start or 0) for item in target_slices
        ],
        "requested_reference_roi": requested,
        "realized_target_roi": realized,
        "reference_sampling": {
            "method": (
                "continuous TD99 analytic field re-evaluated at each target "
                "coordinate; no array interpolation"
            ),
            "coordinate_rule": (
                "full reference x/y/z endpoint bounds retained; sample intervals "
                "scaled by requested ratio and rounded to nearest, ties upward"
            ),
        },
    }

    if ratio == 1.0:
        expected = [list(item) + [None] for item in REFERENCE_SLICES_ZYX[name]]
        if result["target_grid_xyz"] != [base_nx, base_ny, base_nz]:
            raise AssertionError(f"{name}: ratio=1 did not recover the reference grid")
        if result["target_metric_slices_zyx"] != expected:
            raise AssertionError(
                f"{name}: ratio=1 did not recover frozen ROI: "
                f"{result['target_metric_slices_zyx']} != {expected}"
            )
        expected_shape = [stop - start for start, stop in REFERENCE_SLICES_ZYX[name]]
        if result["target_metric_shape_zyx"] != expected_shape:
            raise AssertionError(f"{name}: ratio=1 metric shape contract failed")
    return result


def _evaluate_td_reference(
    name: str,
    mapping: dict[str, Any],
    *,
    chunk_depth: int,
) -> dict[str, np.ndarray]:
    """Evaluate the continuous TD field directly on the mapped target lattice."""
    ratio = float(mapping["requested_ratio"])
    x, y, z = target_axes(name, ratio)
    config = analytic.TD_TOPOLOGY[name]["config"]
    # Keep one common high-precision analytic reference for both nonlinear
    # dtypes.  The solver converts the boundary to its requested dtype, while
    # paper metrics remain referenced to the same float64 sampled field.
    bx = np.empty((z.size, y.size, x.size), dtype=np.float64)
    by = np.empty_like(bx)
    bz = np.empty_like(bx)
    for k0 in range(0, z.size, int(chunk_depth)):
        k1 = min(z.size, k0 + int(chunk_depth))
        xx, yy, zz = np.meshgrid(x, y, z[k0:k1], indexing="ij")
        values = generate_td_model(
            xx,
            yy,
            zz,
            R=config.R,
            a=config.a,
            L=config.L,
            d=config.d,
            q=config.q,
            # The archived TD99 implementation uses the opposite internal sign.
            I_0=-config.I0,
            l_i=config.l_i,
            p_Bt_ratio=config.p_Bt_ratio,
            elliptic_invalid="nan",
        )
        bx[k0:k1] = np.transpose(values[0], (2, 1, 0)).astype(np.float64, copy=False)
        by[k0:k1] = np.transpose(values[1], (2, 1, 0)).astype(np.float64, copy=False)
        bz[k0:k1] = np.transpose(values[2], (2, 1, 0)).astype(np.float64, copy=False)
    for component, values in (("Bx", bx), ("By", by), ("Bz", bz)):
        if not np.all(np.isfinite(values)):
            count = int(values.size - np.count_nonzero(np.isfinite(values)))
            raise FloatingPointError(f"{name} {component}: {count} non-finite samples")
    return {"Bx": bx, "By": by, "Bz": bz}


def _mapping_slices(mapping: dict[str, Any]) -> tuple[slice, slice, slice]:
    return tuple(
        slice(int(item[0]), int(item[1]), item[2])
        for item in mapping["target_metric_slices_zyx"]
    )  # type: ignore[return-value]


def _build_case(
    name: str,
    mapping: dict[str, Any],
    *,
    chunk_depth: int,
    calculation_domain_mode: str,
) -> dict[str, Any]:
    reference = _evaluate_td_reference(name, mapping, chunk_depth=chunk_depth)
    x, y, z = target_axes(name, float(mapping["requested_ratio"]))
    dx, dy, dz = _spacing(x), _spacing(y), _spacing(z)
    _jx, _jy, jz = curl_zyx_nfpt(
        reference["Bx"], reference["By"], reference["Bz"], dx=dx, dy=dy, dz=dz
    )
    metric_slices = _mapping_slices(mapping)
    shape_zyx = tuple(
        int(item.stop) - int(item.start or 0) for item in metric_slices
    )
    if name == "td2":
        domain = (
            f"fixed physical ROI from the Demcsak-count 66x66x30 reference mapping to "
            f"{shape_zyx[2]}x{shape_zyx[1]}x{shape_zyx[0]} target samples"
        )
    else:
        domain = (
            f"fixed physical reference central Nx0/2 x Ny0/2 x Nz ROI mapped "
            f"to {shape_zyx[2]}x{shape_zyx[1]}x{shape_zyx[0]} target samples"
        )
    centre_y = 0.5 * sum(
        mapping["requested_reference_roi"]["y"]["centre_bounds_mm"]
    )
    centre_x = 0.5 * sum(
        mapping["requested_reference_roi"]["x"]["centre_bounds_mm"]
    )
    return {
        "benchmark": name,
        "display": name.upper(),
        "dims": (int(x.size), int(y.size), int(z.size), dx, dy, dz),
        "boundary": {
            "Bx": reference["Bx"][0].astype(np.float64, copy=False),
            "By": reference["By"][0].astype(np.float64, copy=False),
            "Bz": reference["Bz"][0].astype(np.float64, copy=False),
        },
        "reference": reference,
        "Jz_obs": np.asarray(jz[0], dtype=np.float64),
        "metric_slices": metric_slices,
        "metric_shape": shape_zyx,
        "metric_domain": domain,
        "solver_kwargs": {
            "lateral_wall": "paper_bc1",
            "pin_potential_boundaries": False,
            "buffer_thickness": 0,
            "padded_buffer_zone": True,
            "pad_calculation_domain": True,
            "calculation_domain_mode": calculation_domain_mode,
        },
        "mu": 1.0,
        "x_bounds": (float(x[0]), float(x[-1])),
        "y_bounds": (float(y[0]), float(y[-1])),
        "z_bounds": (float(z[0]), float(z[-1])),
        "flux_rope_iy": int(np.argmin(np.abs(y - centre_y))),
        "flux_rope_ix": int(np.argmin(np.abs(x - centre_x))),
        "geometry": {
            "model": name.upper(),
            "shape_zyx": [int(z.size), int(y.size), int(x.size)],
            "spacing_xyz_mm": [dx, dy, dz],
            "resolution_mapping": mapping,
        },
    }


def _driver_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        mean_bz_policy=FIXED_NUMERICS["mean_bz_policy"],
        horizontal_zero_mode_policy=args.horizontal_zero_mode_policy,
        k0_ode_smoothness=FIXED_NUMERICS["k0_ode_smoothness"],
        d0_zero_mode_policy=FIXED_NUMERICS["d0_zero_mode_policy"],
        source_zero_mode_policy=FIXED_NUMERICS["source_zero_mode_policy"],
        harmonic_residual_policy=FIXED_NUMERICS["harmonic_residual_policy"],
        mean_mode_tolerance=FIXED_NUMERICS["mean_mode_tolerance"],
        source_zero_mode_tolerance=FIXED_NUMERICS["source_zero_mode_tolerance"],
        bottom_divergence_projection_weight=0.0,
        bottom_divergence_robin_weight=float(args.rho),
        poloidal_update_mode="direct",
        record_iteration_diagnostics=True,
        handoff_projection_policy="project",
        handoff_projection_tolerance=1.0e-6,
        gaussian_horizontal_operator="spectral",
        vertical_current_operator="staggered_face",
        source_partition_mode="gaussian_separated",
        alpha_clip_scale=1.0,
        weak_field_floor_b2=1.0e-20,
        stop_mode=args.stop_mode,
        tol=float(args.tol),
        patience=int(args.patience),
        sigma_improvement_tol=float(args.sigma_improvement_tol),
        verbose=bool(args.verbose),
        init_mode="gs_cfcf",
    )


def _set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _history_csv(path: Path, history: dict[str, Any]) -> None:
    """Write every per-iteration scalar sequence, retaining full JSON separately."""
    n_steps = len(history.get("sigma", []))
    sequences: dict[str, list[Any]] = {}
    for key, value in history.items():
        if not isinstance(value, (list, tuple)) or len(value) != n_steps:
            continue
        if all(item is None or isinstance(item, (bool, int, float, np.generic)) for item in value):
            sequences[key] = list(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        fieldnames = ("iteration", *sorted(sequences))
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index in range(n_steps):
            writer.writerow(
                {"iteration": index + 1, **{key: values[index] for key, values in sequences.items()}}
            )


def _save_fields(path: Path, fields: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{key: np.asarray(value) for key, value in fields.items()},
    )


def _hardware(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "resolved_device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        prop = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda_device_index": int(index),
                "cuda_device_name": prop.name,
                "cuda_capability": [int(prop.major), int(prop.minor)],
                "cuda_total_memory": int(prop.total_memory),
            }
        )
    return result


def _run_config(
    *,
    args: argparse.Namespace,
    mapping: dict[str, Any],
    dtype_name: str,
    device: str,
    code_sha256: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": mapping["benchmark"],
        "requested_ratio": mapping["requested_ratio"],
        "target_grid_xyz": mapping["target_grid_xyz"],
        "target_metric_slices_zyx": mapping["target_metric_slices_zyx"],
        "target_metric_shape_zyx": mapping["target_metric_shape_zyx"],
        "requested_reference_roi": mapping["requested_reference_roi"],
        "realized_target_roi": mapping["realized_target_roi"],
        "reference_sampling": mapping["reference_sampling"],
        "analytic_reference_dtype": "float64",
        "field_cache_dtype": "solver_native",
        "dtype": dtype_name,
        "device": device,
        "seed": int(args.seed),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "patience": int(args.patience),
        "sigma_improvement_tol": float(args.sigma_improvement_tol),
        "stop_mode": args.stop_mode,
        "fixed_numerics": {
            **FIXED_NUMERICS,
            "bottom_divergence_robin_weight": float(args.rho),
            "horizontal_zero_mode_policy": args.horizontal_zero_mode_policy,
            "calculation_domain_mode": args.td_calculation_domain_mode,
        },
        "code_sha256": code_sha256,
        "toolchain": _hardware(device),
    }


def _run_id(config: dict[str, Any], config_hash: str) -> str:
    return "__".join(
        (
            str(config["benchmark"]),
            f"r{_ratio_tag(config['requested_ratio'])}",
            f"rho{_ratio_tag(config['fixed_numerics']['bottom_divergence_robin_weight'])}",
            "x".join(str(item) for item in config["target_grid_xyz"]),
            str(config["dtype"]),
            config_hash[:12],
        )
    )


METRICS_COLUMNS = (
    "run_id",
    "config_hash",
    "benchmark",
    "requested_ratio",
    "realized_interval_density_ratio_xyz",
    "grid_xyz",
    "calculation_grid_xyz",
    "spacing_xyz_mm",
    "metric_shape_zyx",
    "metric_slices_zyx",
    "requested_roi_cell_edge_bounds_zyx_mm",
    "realized_roi_cell_edge_bounds_zyx_mm",
    "dtype",
    "resolved_device",
    "init_mode",
    "calculation_domain_mode",
    "rho",
    "horizontal_zero_mode_policy",
    "poloidal_update_mode",
    "vertical_current_operator",
    "max_iter",
    "tol",
    "patience",
    "sigma_improvement_tol",
    "best_iter",
    "n_iters",
    "best_sigma",
    "stop_reason",
    "stop_mode",
    "selected_state",
    "wall_time_s",
    "E_rel_Bgt_domain",
    "sigma_J_common_fd2",
    *METRIC_KEYS,
    "cache_status",
)


def _metrics_row(
    *,
    run_id: str,
    config_hash: str,
    config: dict[str, Any],
    mapping: dict[str, Any],
    record: dict[str, Any],
    cache_status: str,
) -> dict[str, Any]:
    requested_bounds = [
        mapping["requested_reference_roi"][axis]["cell_edge_bounds_mm"]
        for axis in ("z", "y", "x")
    ]
    realized_bounds = [
        mapping["realized_target_roi"][axis]["cell_edge_bounds_mm"]
        for axis in ("z", "y", "x")
    ]
    return {
        "run_id": run_id,
        "config_hash": config_hash,
        "benchmark": record["benchmark"],
        "requested_ratio": mapping["requested_ratio"],
        "realized_interval_density_ratio_xyz": json.dumps(
            mapping["realized_interval_density_ratio_xyz"]
        ),
        "grid_xyz": json.dumps(record["grid_xyz"]),
        "calculation_grid_xyz": json.dumps(record["calculation_grid_xyz"]),
        "spacing_xyz_mm": json.dumps(mapping["spacing_xyz_mm"]),
        "metric_shape_zyx": json.dumps(record["metric_shape_zyx"]),
        "metric_slices_zyx": json.dumps(mapping["target_metric_slices_zyx"]),
        "requested_roi_cell_edge_bounds_zyx_mm": json.dumps(requested_bounds),
        "realized_roi_cell_edge_bounds_zyx_mm": json.dumps(realized_bounds),
        "dtype": config["dtype"],
        "resolved_device": config["device"],
        "init_mode": "gs_cfcf",
        "calculation_domain_mode": record["calculation_domain_mode"],
        "rho": config["fixed_numerics"]["bottom_divergence_robin_weight"],
        "horizontal_zero_mode_policy": config["fixed_numerics"][
            "horizontal_zero_mode_policy"
        ],
        "poloidal_update_mode": "direct",
        "vertical_current_operator": "staggered_face",
        "max_iter": config["max_iter"],
        "tol": config["tol"],
        "patience": config["patience"],
        "sigma_improvement_tol": config["sigma_improvement_tol"],
        "best_iter": record["best_iter"],
        "n_iters": record["n_iters"],
        "best_sigma": record["best_sigma"],
        "stop_reason": record["stop_reason"],
        "stop_mode": record.get("stop_mode", config["stop_mode"]),
        "selected_state": record.get("selected_state"),
        "wall_time_s": record["wall_time_s"],
        "E_rel_Bgt_domain": record["E_rel_Bgt_domain"],
        "sigma_J_common_fd2": record["sigma_J_common_fd2"],
        **{key: record[key] for key in METRIC_KEYS},
        "cache_status": cache_status,
    }


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=METRICS_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _mapping_csv(path: Path, mappings: list[dict[str, Any]]) -> None:
    fields = (
        "benchmark",
        "requested_ratio",
        "reference_grid_xyz",
        "target_grid_xyz",
        "realized_interval_density_ratio_xyz",
        "spacing_xyz_mm",
        "reference_metric_slices_zyx",
        "target_metric_slices_zyx",
        "target_metric_shape_zyx",
        "requested_roi_cell_edge_bounds_zyx_mm",
        "realized_roi_cell_edge_bounds_zyx_mm",
        "edge_mismatch_zyx_mm",
    )
    rows = []
    for mapping in mappings:
        rows.append(
            {
                **{key: mapping[key] for key in fields if key in mapping},
                "requested_roi_cell_edge_bounds_zyx_mm": [
                    mapping["requested_reference_roi"][axis]["cell_edge_bounds_mm"]
                    for axis in ("z", "y", "x")
                ],
                "realized_roi_cell_edge_bounds_zyx_mm": [
                    mapping["realized_target_roi"][axis]["cell_edge_bounds_mm"]
                    for axis in ("z", "y", "x")
                ],
                "edge_mismatch_zyx_mm": [
                    mapping["realized_target_roi"][axis]["edge_mismatch_mm"]
                    for axis in ("z", "y", "x")
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    code_paths = {
        "runner": Path(__file__).resolve(),
        "analytic_helper": ROOT / "nfgs" / "_backend" / "scripts" / "run_analytic_validation.py",
        "nfgs_solver": ROOT / "nfgs_solver.py",
        "nfpt_stable_solver": ROOT / "nfgs_solver.py",
        "validation_driver": ROOT / "nfgs" / "_backend" / "scripts" / "run_nfgs_analytic_validation.py",
        "field_ops": ROOT / "nfgs" / "_backend" / "util" / "field_ops.py",
        "metrics": ROOT / "nfgs" / "_backend" / "util" / "metrics.py",
        "td_generator": ROOT / "nfgs/_backend/td.py",
    }
    code_sha256 = {name: _sha256_file(path) for name, path in code_paths.items()}
    mappings = [
        build_domain_mapping(name, ratio)
        for name in args.benchmarks
        for ratio in args.ratios
    ]
    _mapping_csv(out_dir / "domain_mapping.csv", mappings)
    _write_json(
        out_dir / "domain_mapping.json",
        {
            "schema_version": 1,
            "statement": (
                "All ratios use the same reference-resolution physical metric "
                "cell-edge bounds; target slices are their discrete-grid mapping."
            ),
            "td2_reference_contract": {
                "shape_xyz": [66, 66, 30],
                "centre_yx": list(TD2_REFERENCE_CENTER_YX),
                "slices_zyx": [list(item) for item in REFERENCE_SLICES_ZYX["td2"]],
            },
            "code_paths": {
                name: str(path.resolve()) for name, path in code_paths.items()
            },
            "code_sha256": code_sha256,
            "mappings": mappings,
        },
    )

    if args.contract_only:
        progress = {
            "status": "contract_only_complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "nonlinear_solves_run": 0,
            "reference_fields_generated": 0,
            "n_domain_mappings_validated": len(mappings),
            "requested_device_not_resolved": args.device,
            "code_sha256": code_sha256,
            "note": "This is not a completed TD resolution sweep.",
        }
        _write_json(out_dir / "progress.json", progress)
        print(
            f"contract-only PASS: {len(mappings)} fixed-physics mappings; "
            "nonlinear solves=0",
            flush=True,
        )
        return 0

    resolved_device = analytic.resolve_device(args.device)
    records_dir = out_dir / "records"
    histories_dir = out_dir / "histories"
    fields_dir = out_dir / "fields"
    records_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_field_cache:
        fields_dir.mkdir(parents=True, exist_ok=True)

    total = len(mappings) * len(args.dtypes)
    progress: dict[str, Any] = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "total_runs": total,
        "completed_runs": 0,
        "completed_run_ids": [],
        "current_run_id": None,
    }
    _write_json(out_dir / "progress.json", progress)
    driver_args = _driver_args(args)
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    try:
        for mapping in mappings:
            name = str(mapping["benchmark"])
            ratio = float(mapping["requested_ratio"])
            print(
                f"building continuous reference {name} ratio={ratio:g} "
                f"grid={mapping['target_grid_xyz']}",
                flush=True,
            )
            case = _build_case(
                name,
                mapping,
                chunk_depth=int(args.reference_chunk_depth),
                calculation_domain_mode=args.td_calculation_domain_mode,
            )
            for dtype_name in args.dtypes:
                config = _run_config(
                    args=args,
                    mapping=mapping,
                    dtype_name=dtype_name,
                    device=resolved_device,
                    code_sha256=code_sha256,
                )
                config_hash = _canonical_hash(config)
                run_id = _run_id(config, config_hash)
                record_path = records_dir / f"{run_id}.json"
                history_json_path = histories_dir / f"{run_id}.json"
                history_csv_path = histories_dir / f"{run_id}.csv"
                field_path = fields_dir / f"{run_id}.npz"
                required_cache = (
                    record_path.is_file()
                    and history_json_path.is_file()
                    and history_csv_path.is_file()
                    and (args.no_field_cache or field_path.is_file())
                )
                progress["current_run_id"] = run_id
                _write_json(out_dir / "progress.json", progress)
                if required_cache and not args.force:
                    cached = json.loads(record_path.read_text())
                    row = dict(cached["metrics_row"])
                    row["cache_status"] = "reused"
                    rows.append(row)
                    manifest.append(cached["artifact_manifest"])
                    print(f"reuse {run_id}", flush=True)
                else:
                    _set_seed(int(args.seed), resolved_device)
                    print(f"solve {run_id}", flush=True)
                    record, fields = driver.run_case(
                        case,
                        device=resolved_device,
                        dtype=analytic.dtype_from_name(dtype_name),
                        max_iter=int(args.max_iter),
                        args=driver_args,
                    )
                    nx, ny, _nz, dx, dy, dz = case["dims"]
                    bgt_args = SimpleNamespace(
                        mean_bz_policy="uniform_open",
                        horizontal_zero_mode_policy="frozen_linear_harmonic",
                        d0_zero_mode_policy="project",
                        source_zero_mode_policy="project",
                        harmonic_residual_policy="diagnose",
                        mean_mode_tolerance=1.0e-6,
                        source_zero_mode_tolerance=1.0e-2,
                        gaussian_horizontal_operator="spectral",
                    )
                    bgt_helper = make_bgt_helper_solver(
                        nx=int(nx), ny=int(ny), dx=float(dx), dy=float(dy), dz=float(dz),
                        device=resolved_device, dtype=analytic.dtype_from_name(dtype_name),
                        args=bgt_args,
                    )
                    _sz, sy, sx = case["metric_slices"]
                    record["E_rel_Bgt_domain"] = bottom_bgt_relative_error(
                        bgt_helper, case["reference"], fields, y_slice=sy, x_slice=sx
                    )
                    record["sigma_J_common_fd2"] = common_fd2_sigma_j(
                        fields, float(dx), case["metric_slices"]
                    )
                    del bgt_helper
                    history = dict(record.pop("history"))
                    # Preserve the nonlinear stopping selection in the compact
                    # record as well as the separate full-history artifact.  The
                    # campaign gate intentionally audits records without loading
                    # multi-megabyte histories.
                    record["selected_state"] = history.get("selected_state")
                    _write_json(history_json_path, history)
                    _history_csv(history_csv_path, history)
                    if not args.no_field_cache:
                        _save_fields(field_path, fields)
                    row = _metrics_row(
                        run_id=run_id,
                        config_hash=config_hash,
                        config=config,
                        mapping=mapping,
                        record=record,
                        cache_status="computed",
                    )
                    artifact_manifest = {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "record": str(record_path.relative_to(out_dir)),
                        "history_json": str(history_json_path.relative_to(out_dir)),
                        "history_csv": str(history_csv_path.relative_to(out_dir)),
                        "fields": (
                            None
                            if args.no_field_cache
                            else str(field_path.relative_to(out_dir))
                        ),
                    }
                    payload = {
                        "config": config,
                        "mapping": mapping,
                        "record": record,
                        "metrics_row": row,
                        "artifact_manifest": artifact_manifest,
                    }
                    _write_json(record_path, payload)
                    rows.append(row)
                    manifest.append(artifact_manifest)
                    del fields

                progress["completed_runs"] = int(progress["completed_runs"]) + 1
                progress["completed_run_ids"].append(run_id)
                progress["current_run_id"] = None
                _write_json(out_dir / "progress.json", progress)
                _write_metrics(out_dir / "metrics.csv", rows)
                _write_json(out_dir / "run_manifest.json", {"runs": manifest})
                gc.collect()
                if resolved_device.startswith("cuda"):
                    torch.cuda.empty_cache()
            del case
            gc.collect()

        progress.update(
            {
                "status": "complete",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "current_run_id": None,
            }
        )
        _write_json(out_dir / "progress.json", progress)
        print(f"complete: {len(rows)} runs; metrics={out_dir / 'metrics.csv'}", flush=True)
        return 0
    except BaseException as exc:
        progress.update(
            {
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(out_dir / "progress.json", progress)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
