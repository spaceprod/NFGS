#!/usr/bin/env python3
"""Run the publication Low--Lou C2 and TD analytic validation suite.

This is the single entry point for the analytic-field metrics and the first
two manuscript figures.  It runs ``nfpt_stable`` with potential-field (PF),
curl-free-current-field (CFCF), and Auto-RBSL initial fields on both analytic
benchmarks.  The Low--Lou setup matches Jarolim et al. (2023) C2: an
80 x 80 x 72 computational grid, potential side/top boundaries, and the
central 64^3 evaluation cube.

Figure 1 compares the original Fortran NFPT result and ``nfpt_stable`` on the
same TD field, CFCF initialization, and 66 x 66 x 30 evaluation volume.
Figure 2 compares reference and stable TD field lines in top and side views.
The magenta and blue current-axis lines use explicit y seed coordinates; the
opposite-polarity seed is always mirrored about the physical y=0 axis.

With ``--td-topology-qsl``, the same entry point also evaluates the
TD1--TD3 analytic fields with vendored FastQSL on the y=0 plane.  It stores
unclipped Q maps, Q-map comparison metrics, and the three-panel topology plot.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import gc
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfpt_analytic_matplotlib")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/nfpt_analytic_mesa")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nfgs_solver import StableNFPTSolverGPU  # noqa: E402
from nfgs._backend.util.lnl_io import build_lnl_jarolim_c2_reference_and_obs  # noqa: E402
from nfgs._backend.util.metrics import compute_nf2_metrics  # noqa: E402
from nfgs._backend.util.td2_io import TD2Grid, build_td2_reference_and_obs, load_b_czyx  # noqa: E402
from nfgs._backend.util.fastqsl_plane import DEFAULT_FASTQSL_ROOT, compute_y0_qmap  # noqa: E402
from nfgs.diagnostics.topology_metrics import (  # noqa: E402
    fastqsl_bundle_metrics,
    q_map_metrics,
)
from nfgs._backend.td import TDModelConfig, generate_td_model  # noqa: E402

INIT_MODES = ("potential", "curl_free_current", "auto_rbsl")
INIT_LABELS = {
    "potential": "PF",
    "curl_free_current": "CFCF",
    "auto_rbsl": "Auto-RBSL",
}
JAROLIM_C2 = {
    "C_vec": 1.00,
    "C_CS": 0.93,
    "Eprime_n": 0.84,
    "Eprime_m": 0.68,
    "epsilon": 1.07,
    "sigma_J": 0.0351,
}
METRIC_KEYS = (
    "C_vec",
    "C_CS",
    "Eprime_n",
    "Eprime_m",
    "epsilon",
    "sigma_J",
)
SOLVER_PUBLIC_NAME = "Fast NFPT"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def default_source_root() -> Path:
    candidates = (PROJECT_ROOT, PROJECT_ROOT.parents[1])
    for candidate in candidates:
        if (candidate / "data" / "td2" / "bcor").is_file():
            return candidate
    return PROJECT_ROOT


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return name


def metric_shape(slices: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    return tuple(int(s.stop) - int(s.start or 0) for s in slices)  # type: ignore[return-value]


def make_cases(
    source_root: Path,
    calculation_domain_mode: str = "bottom_padding",
) -> dict[str, dict[str, Any]]:
    lnl = build_lnl_jarolim_c2_reference_and_obs()
    lnl_slices = lnl["metric_slices"]
    lnl_case = {
        "benchmark": "low_lou_c2",
        "display": "Low--Lou C2",
        "dims": (80, 80, 72, lnl["dx"], lnl["dy"], lnl["dz"]),
        "boundary": {"Bx": lnl["Bx_obs"], "By": lnl["By_obs"], "Bz": lnl["Bz_obs"]},
        "reference": {"Bx": lnl["Bx_ref"], "By": lnl["By_ref"], "Bz": lnl["Bz_ref"]},
        "Jz_obs": lnl["Jz_obs"],
        "metric_slices": lnl_slices,
        "metric_shape": metric_shape(lnl_slices),
        "metric_domain": "Jarolim et al. C2 central 64^3 cube",
        "solver_kwargs": {
            "lateral_wall": "legacy_potential_pin",
            "pin_potential_boundaries": True,
            "buffer_thickness": 0,
            "padded_buffer_zone": True,
            "pad_calculation_domain": True,
            "calculation_domain_mode": calculation_domain_mode,
        },
        "mu": 1.0,
        "jarolim_c2": lnl["jarolim_c2"],
    }

    grid = TD2Grid()
    td = build_td2_reference_and_obs(source_root / "data" / "td2" / "bcor", grid)
    td_slices = td["metric_slices"]
    td_case = {
        "benchmark": "td",
        "display": "Titov--Demoulin",
        "dims": (grid.Nx, grid.Ny, grid.Nz, td["dx"], td["dy"], td["dz"]),
        "boundary": {"Bx": td["Bx_obs"], "By": td["By_obs"], "Bz": td["Bz_obs"]},
        "reference": {"Bx": td["Bx_ref"], "By": td["By_ref"], "Bz": td["Bz_ref"]},
        "Jz_obs": td["Jz_obs"],
        "metric_slices": td_slices,
        "metric_shape": metric_shape(td_slices),
        "metric_domain": "flux-rope-centered 66 x 66 x 30 cube",
        "solver_kwargs": {
            "lateral_wall": "paper_bc1",
            "pin_potential_boundaries": False,
            "buffer_thickness": 0,
            "padded_buffer_zone": True,
            "pad_calculation_domain": True,
            "calculation_domain_mode": calculation_domain_mode,
        },
        "mu": 1.0,
        "x_bounds": td["x_bounds"],
        "y_bounds": td["y_bounds"],
        "z_bounds": td["z_bounds"],
        "flux_rope_iy": td["flux_rope_iy"],
        "flux_rope_ix": td["flux_rope_ix"],
    }
    return {"low_lou_c2": lnl_case, "td": td_case}


def fields_from_solver(solver: StableNFPTSolverGPU) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in solver.fields_to_numpy().items()}


def save_field_cache(path: Path, fields: dict[str, np.ndarray]) -> None:
    """Write a compressed field cache atomically on the destination volume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(
                stream,
                **{
                    name: np.asarray(value, dtype=np.float32)
                    for name, value in fields.items()
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def run_stable_case(
    case: dict[str, Any],
    init_mode: str,
    *,
    device: str,
    dtype: torch.dtype,
    max_iter: int,
    tol: float,
    patience: int,
    sigma_improvement_tol: float,
    auto_rbsl_axis_n_pts: int,
    auto_rbsl_chunk_size: int,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    nx, ny, nz, dx, dy, dz = case["dims"]
    solver = StableNFPTSolverGPU(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        device=device,
        dtype=dtype,
        **case["solver_kwargs"],
    )
    boundary = case["boundary"]
    solver.set_bottom_vector_boundary(
        boundary["Bx"],
        boundary["By"],
        boundary["Bz"],
        zero_net_current=False,
        smooth_passes=0,
        ijzbcor=3,
    )

    t0 = time.perf_counter()
    with torch.inference_mode():
        history = solver.iterate(
            max_iter=int(max_iter),
            tol=float(tol),
            mu=float(case["mu"]),
            patience=int(patience),
            sigma_improvement_tol=float(sigma_improvement_tol),
            auto_rbsl_improvement_grace=20,
            auto_rbsl_include_residual_current=True,
            verbose=bool(verbose),
            polarity_mask="both",
            init_mode=init_mode,
            axis_n_pts=int(auto_rbsl_axis_n_pts),
            chunk_size=int(auto_rbsl_chunk_size),
            work_dtype=torch.float32,
            adaptive_mu=False,
        )
    wall_time = time.perf_counter() - t0
    fields = fields_from_solver(solver)
    metrics = compute_nf2_metrics(case["reference"], fields, case["metric_slices"])
    sz, sy, sx = case["metric_slices"]
    j_roi = np.stack([fields["Jx"], fields["Jy"], fields["Jz"]], axis=0)[:, sz, sy, sx]
    current_norm_sum = float(np.sqrt(np.sum(j_roi * j_roi, axis=0)).sum())
    sigma_j_defined = current_norm_sum > 1e-12
    if not sigma_j_defined:
        metrics["sigma_J"] = float("nan")
    init_diagnostics = history.get("init_diagnostics") or {}
    record = {
        "benchmark": case["benchmark"],
        "benchmark_display": case["display"],
        "solver": SOLVER_PUBLIC_NAME,
        "init_mode": init_mode,
        "init_label": INIT_LABELS[init_mode],
        "grid_xyz": [int(nx), int(ny), int(nz)],
        "calculation_grid_xyz": [int(solver.Nx), int(solver.Ny), int(solver.Nz)],
        "calculation_domain_padded": bool(solver._auto_padded_calculation_domain),
        "calculation_domain_mode": solver._calculation_domain_mode,
        "calculation_pad_xy": [int(v) for v in solver._calculation_pad_xy],
        "metric_shape_zyx": list(case["metric_shape"]),
        "metric_domain": case["metric_domain"],
        "boundary_profile": case["solver_kwargs"]["lateral_wall"],
        "best_iter": int(history["best_iter"]),
        "n_iters": len(history["sigma"]),
        "best_sigma": float(history["best_sigma"]),
        "stop_reason": history["stop_reason"],
        "wall_time_s": float(wall_time),
        "mfr_count": int(init_diagnostics.get("mfr_count", 0)),
        "sigma_J_defined": bool(sigma_j_defined),
        **{key: float(metrics[key]) for key in METRIC_KEYS},
        "history": history,
    }
    del solver
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return record, fields


def fortran_td_paths(source_root: Path) -> tuple[Path, Path, Path]:
    tag = "td2_Nx150_Ny250_Nz100_iter3000"
    run_dir = source_root / "results" / "gpu_stable_initializers" / "fortran_runs" / tag
    return (
        run_dir / "ext" / "b" / f"b_{tag}",
        run_dir / "ext" / "j" / f"j_{tag}",
        run_dir / "anal" / "cwsin",
    )


def load_fortran_td_record(
    case: dict[str, Any], source_root: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    b_path, j_path, history_path = fortran_td_paths(source_root)
    if not b_path.is_file() or not j_path.is_file():
        return None
    nx, ny, nz, _dx, _dy, _dz = case["dims"]
    bx, by, bz = load_b_czyx(str(b_path), nx, ny, nz)
    jx, jy, jz = load_b_czyx(str(j_path), nx, ny, nz)
    fields = {"Bx": bx, "By": by, "Bz": bz, "Jx": jx, "Jy": jy, "Jz": jz}
    metrics = compute_nf2_metrics(case["reference"], fields, case["metric_slices"])
    history = np.loadtxt(history_path, ndmin=2) if history_path.is_file() else np.empty((0, 2))
    record = {
        "benchmark": "td",
        "benchmark_display": "Titov--Demoulin",
        "solver": "nfpt_fortran",
        "init_mode": "curl_free_current",
        "init_label": "CFCF",
        "grid_xyz": [int(nx), int(ny), int(nz)],
        "metric_shape_zyx": list(case["metric_shape"]),
        "metric_domain": case["metric_domain"],
        "boundary_profile": "original Fortran NFPT",
        "best_iter": int(history[-1, 0]) if history.size else 0,
        "n_iters": int(history.shape[0]),
        "best_sigma": float(np.nanmin(history[:, 1])) if history.size else float("nan"),
        "stop_reason": "precomputed original Fortran run",
        "wall_time_s": float("nan"),
        **{key: float(metrics[key]) for key in METRIC_KEYS},
        "history": {
            "iter": history[:, 0].astype(int).tolist() if history.size else [],
            "sigma": history[:, 1].astype(float).tolist() if history.size else [],
        },
        "source_b": str(b_path),
        "source_j": str(j_path),
    }
    return record, fields


def published_precision_better(metric: str, value: float) -> bool:
    if not math.isfinite(float(value)):
        return False
    reference = JAROLIM_C2[metric]
    value_published = round(float(value), 2 if metric != "sigma_J" else 4)
    if metric in ("C_vec", "C_CS", "Eprime_n", "Eprime_m"):
        return value_published >= reference
    if metric == "epsilon":
        return abs(value_published - 1.0) <= abs(reference - 1.0)
    return value_published <= reference


def jarolim_comparison(record: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for key in METRIC_KEYS:
        value = float(record[key])
        reference = float(JAROLIM_C2[key])
        if key == "epsilon":
            gain = abs(reference - 1.0) - abs(value - 1.0)
        elif key in ("sigma_J",):
            gain = reference - value
        else:
            gain = value - reference
        comparison[key] = {
            "stable": value,
            "jarolim_c2": reference,
            "improvement": gain,
            "equal_or_better_at_published_precision": published_precision_better(key, value),
        }
    comparison["all_equal_or_better_at_published_precision"] = all(
        comparison[key]["equal_or_better_at_published_precision"] for key in METRIC_KEYS
    )
    return comparison


def write_metrics_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "benchmark",
        "benchmark_display",
        "solver",
        "init_mode",
        "init_label",
        "grid_xyz",
        "calculation_grid_xyz",
        "calculation_domain_padded",
        "calculation_domain_mode",
        "calculation_pad_xy",
        "metric_shape_zyx",
        "metric_domain",
        "boundary_profile",
        "best_iter",
        "n_iters",
        "best_sigma",
        "stop_reason",
        "wall_time_s",
        "mfr_count",
        "sigma_J_defined",
        *METRIC_KEYS,
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(record[key]) if isinstance(record.get(key), list) else record.get(key) for key in fields})


def write_history_csv(path: Path, record: dict[str, Any]) -> None:
    history = record["history"]
    sigma = list(history.get("sigma", []))
    n_rows = len(sigma)
    iterations = list(history.get("iter", range(1, n_rows + 1)))
    preferred = ["sigma", "divB", "energy", "mu", "rollback"]
    dynamic = sorted(
        key
        for key, value in history.items()
        if key not in preferred
        and isinstance(value, (list, tuple))
        and len(value) == n_rows
    )
    columns = preferred + dynamic
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("iteration", *columns))
        for index, iteration in enumerate(iterations):
            writer.writerow(
                [iteration]
                + [
                    history.get(key, [float("nan")] * n_rows)[index]
                    for key in columns
                ]
            )


def plot_figure1(
    fortran_record: dict[str, Any],
    stable_record: dict[str, Any],
    output_path: Path,
) -> None:
    labels = ("Fortran NFPT", SOLVER_PUBLIC_NAME)
    colors = ("#7f8c8d", "#1f77b4")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))

    accuracy_keys = ("C_vec", "C_CS", "Eprime_n", "Eprime_m")
    accuracy_labels = (r"$C_{\rm vec}$", r"$C_{\rm CS}$", r"$E'_n$", r"$E'_m$")
    x = np.arange(len(accuracy_keys), dtype=float)
    width = 0.36
    for offset, record, label, color in zip((-0.5, 0.5), (fortran_record, stable_record), labels, colors):
        axes[0].bar(x + offset * width, [record[key] for key in accuracy_keys], width, label=label, color=color)
    axes[0].set_xticks(x, accuracy_labels)
    axes[0].set_ylim(0.0, 1.04)
    axes[0].set_ylabel("Larger is better")
    axes[0].set_title("(a) TD field agreement")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    error_keys = ("epsilon", "sigma_J")
    error_labels = (r"$|\epsilon-1|$", r"$\sigma_J$")
    for offset, record, color in zip((-0.5, 0.5), (fortran_record, stable_record), colors):
        values = (abs(float(record["epsilon"]) - 1.0), float(record["sigma_J"]))
        axes[1].bar(np.arange(2) + offset * width, values, width, color=color)
    axes[1].set_xticks(np.arange(2), error_labels)
    axes[1].set_ylabel("Smaller is better")
    axes[1].set_title("(b) Energy and force-free errors")
    axes[1].grid(axis="y", alpha=0.25)

    f_hist = fortran_record["history"]
    s_hist = stable_record["history"]
    f_iter = np.asarray(f_hist.get("iter", range(len(f_hist.get("sigma", [])))), dtype=float)
    s_sigma = np.asarray(s_hist.get("sigma", []), dtype=float)
    s_iter = np.arange(1, len(s_sigma) + 1, dtype=float)
    axes[2].plot(f_iter, f_hist.get("sigma", []), color=colors[0], lw=2.0, label=labels[0])
    axes[2].plot(s_iter, s_sigma, color=colors[1], lw=2.0, label=labels[1])
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel(r"$\sigma_J$ monitored by each solver")
    axes[2].set_title("(c) Convergence history")
    axes[2].grid(which="both", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=9)

    fig.suptitle(f"Original Fortran NFPT versus {SOLVER_PUBLIC_NAME} on the TD benchmark", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sunpy_cmap():
    """Use the neutral paper magnetogram palette in legacy TD views too."""
    return "gray"


def _vector_grid(pv, vector_fields: dict[str, np.ndarray], dx: float, dy: float, dz: float, max_z: int):
    first = next(iter(vector_fields.values()))
    _ncomp, nz, ny, nx = first.shape
    viz_nz = min(int(nz), int(max_z))
    grid = pv.ImageData(dimensions=(nx, ny, viz_nz), spacing=(dx, dy, dz), origin=(0.0, 0.0, 0.0))
    for name, field in vector_fields.items():
        arr = np.asarray(field)[:, :viz_nz]
        vectors = np.moveaxis(arr, 0, -1).reshape(-1, 3)
        grid.point_data[name] = np.ascontiguousarray(vectors, dtype=np.float32)
    return grid


def _bottom_plane(pv, values: np.ndarray, dx: float, dy: float):
    ny, nx = values.shape
    plane = pv.ImageData(dimensions=(nx, ny, 1), spacing=(dx, dy, 1.0), origin=(0.0, 0.0, 0.0))
    plane.point_data["Bz"] = np.ascontiguousarray(values.reshape(-1), dtype=np.float32)
    return plane


def _trace_lines(pv, grid, vector_name: str, seeds: np.ndarray, *, max_steps: int = 3500):
    source = pv.PolyData(np.ascontiguousarray(seeds, dtype=np.float32))
    max_length = max(
        grid.bounds[1] - grid.bounds[0],
        grid.bounds[3] - grid.bounds[2],
        grid.bounds[5] - grid.bounds[4],
    ) * 2.4
    return grid.streamlines_from_source(
        source,
        vectors=vector_name,
        integrator_type=45,
        integration_direction="both",
        initial_step_length=0.35,
        min_step_length=0.02,
        max_step_length=1.25,
        max_steps=int(max_steps),
        terminal_speed=1e-12,
        max_length=float(max_length),
        compute_vorticity=False,
    )


def _manual_axis_pair(cx: float, cy: float, y_seed: float, dz: float) -> tuple[np.ndarray, np.ndarray]:
    z = 0.5 * float(dz)
    positive = np.asarray([[cx, float(y_seed), z]], dtype=np.float32)
    negative = np.asarray([[cx, 2.0 * cy - float(y_seed), z]], dtype=np.float32)
    return positive, negative


def _bald_patch_seeds(cx: float, cy: float, dx: float, dy: float, dz: float, n: int = 50) -> np.ndarray:
    radius = (6.5 / 3.0) * min(float(dx), float(dy))
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return np.asarray(
        [[cx + radius * math.cos(t), cy + radius * math.sin(t), 0.5 * dz] for t in theta],
        dtype=np.float32,
    )


def plot_figure2(
    case: dict[str, Any],
    solution_fields: dict[str, np.ndarray],
    *,
    init_mode: str,
    reference_axis_y: float,
    solution_axis_y: float,
    output_path: Path,
) -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    pv.OFF_SCREEN = True
    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"
    pv.global_theme.font.size = 10

    _nx, _ny, nz, dx, dy, dz = case["dims"]
    max_z_planes = max(2, int(math.ceil(nz / 3.0)))
    ref_vec = np.stack([case["reference"]["Bx"], case["reference"]["By"], case["reference"]["Bz"]])
    sol_vec = np.stack([solution_fields["Bx"], solution_fields["By"], solution_fields["Bz"]])
    vector_fields = {"B_ref": ref_vec, "B_stable": sol_vec}
    grid = _vector_grid(pv, vector_fields, dx, dy, dz, max_z_planes)
    bottom = _bottom_plane(pv, np.asarray(case["boundary"]["Bz"], dtype=np.float32), dx, dy)

    cx = -float(case["x_bounds"][0])
    cy = -float(case["y_bounds"][0])
    bald = _bald_patch_seeds(cx, cy, dx, dy, dz)
    ref_pos, ref_neg = _manual_axis_pair(cx, cy, reference_axis_y, dz)
    sol_pos, sol_neg = _manual_axis_pair(cx, cy, solution_axis_y, dz)
    panel_seeds = {
        "B_ref": (ref_pos, ref_neg),
        "B_stable": (sol_pos, sol_neg),
    }

    all_seed_points = np.vstack((bald, ref_pos, ref_neg, sol_pos, sol_neg))
    focus_bounds = (
        float(all_seed_points[:, 0].min() - 28.0),
        float(all_seed_points[:, 0].max() + 28.0),
        float(all_seed_points[:, 1].min() - 28.0),
        float(all_seed_points[:, 1].max() + 28.0),
        0.0,
        float(case["metric_slices"][0].stop * dz),
    )

    plotter = pv.Plotter(shape=(2, 2), window_size=(1400, 980), border=False, off_screen=True)
    plotter.set_background("white", all_renderers=True)
    bz_abs = float(np.nanpercentile(np.abs(case["boundary"]["Bz"]), 99.0)) or 1.0
    cmap = _sunpy_cmap()
    line_radius = 0.48 * min(dx, dy, dz)
    axis_radius = 1.35 * min(dx, dy, dz)
    panel_titles = {
        "B_ref": "Reference TD field",
        "B_stable": f"{SOLVER_PUBLIC_NAME} ({INIT_LABELS[init_mode]})",
    }
    panel_names = ("B_ref", "B_stable")
    views = ("top", "side")
    letters = (("(a)", "(b)"), ("(c)", "(d)"))

    for row, view in enumerate(views):
        for col, name in enumerate(panel_names):
            plotter.subplot(row, col)
            plotter.add_mesh(bottom, scalars="Bz", cmap=cmap, clim=(-bz_abs, bz_abs), opacity=0.47, show_scalar_bar=False)
            red_lines = _trace_lines(pv, grid, name, bald)
            pos_lines = _trace_lines(pv, grid, name, panel_seeds[name][0])
            neg_lines = _trace_lines(pv, grid, name, panel_seeds[name][1])
            if red_lines.n_points:
                plotter.add_mesh(red_lines.tube(radius=line_radius, n_sides=10), color="#ef4444", smooth_shading=True)
            if pos_lines.n_points:
                plotter.add_mesh(pos_lines.tube(radius=axis_radius, n_sides=12), color="#d946ef", smooth_shading=True)
            if neg_lines.n_points:
                plotter.add_mesh(neg_lines.tube(radius=axis_radius, n_sides=12), color="#2563eb", smooth_shading=True)
            label = f"{letters[row][col]} {panel_titles[name]} — {view} view"
            plotter.add_text(label, position="upper_left", font_size=11, color="black")
            if view == "top":
                plotter.view_xy()
                plotter.camera.roll = float(plotter.camera.roll) + 90.0
            else:
                plotter.view_yz()
            plotter.reset_camera(bounds=focus_bounds)
            plotter.camera.zoom(1.125)

    plotter.link_views([0, 1])
    plotter.link_views([2, 3])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(interactive=False, screenshot=str(output_path), auto_close=True)


def write_latex_rows(path: Path, records: list[dict[str, Any]]) -> None:
    stable = [record for record in records if record["solver"] == SOLVER_PUBLIC_NAME]
    lines = [
        "% Generated by scripts/run_analytic_validation.py; do not edit by hand.",
    ]
    for record in stable:
        sigma_text = f"{record['sigma_J']:.4f}" if math.isfinite(float(record["sigma_J"])) else r"\nodata"
        lines.append(
            f"{record['benchmark_display']} & {record['init_label']} & "
            f"{record['C_vec']:.3f} & {record['C_CS']:.3f} & "
            f"{record['Eprime_n']:.3f} & {record['Eprime_m']:.3f} & "
            f"{record['epsilon']:.3f} & {sigma_text} \\\\"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# TD Table 1.  The paper reports grid *points*, while the
# NFPT/FastQSL arrays use the corresponding number of cells along each axis.
TD_TOPOLOGY = {
    "td1": {"config": TDModelConfig(R=110.0, a=35.0, L=100.0, d=50.0, q=100.0, I0=-13.0, Nt=2.577), "nodes": (151, 251, 101)},
    "td2": {"config": TDModelConfig(R=100.0, a=35.0, L=60.0, d=60.0, q=80.0, I0=-3.0, Nt=8.8), "nodes": (151, 251, 101)},
    "td3": {"config": TDModelConfig(R=110.0, a=32.5, L=50.0, d=50.0, q=100.0, I0=-7.0, Nt=5.0), "nodes": (257, 351, 201)},
}


def td_topology_field(name: str, *, chunk_depth: int = 4) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate one published TD configuration on its compatible lattice."""
    spec = TD_TOPOLOGY[name]
    if int(chunk_depth) < 1:
        raise ValueError("td_topology_chunk_depth must be >= 1")
    config: TDModelConfig = spec["config"]
    nx, ny, nz = (int(value) - 1 for value in spec["nodes"])
    h = 360.0 / float(nx - 1)
    x = np.linspace(-180.0, 180.0, nx, dtype=np.float64)
    y = h * (np.arange(ny, dtype=np.float64) - 0.5 * float(ny - 1))
    z = h * np.arange(nz, dtype=np.float64)
    bx = np.empty((nz, ny, nx), dtype=np.float32)
    by = np.empty_like(bx)
    bz = np.empty_like(bx)
    for k0 in range(0, nz, int(chunk_depth)):
        k1 = min(nz, k0 + int(chunk_depth))
        xx, yy, zz = np.meshgrid(x, y, z[k0:k1], indexing="ij")
        # The archived TD99 port uses the opposite internal I0 sign to Table 1.
        values = generate_td_model(
            xx, yy, zz, R=config.R, a=config.a, L=config.L, d=config.d,
            q=config.q, I_0=-config.I0, l_i=config.l_i,
            p_Bt_ratio=config.p_Bt_ratio, elliptic_invalid="nan",
        )
        bx[k0:k1] = np.transpose(values[0], (2, 1, 0)).astype(np.float32, copy=False)
        by[k0:k1] = np.transpose(values[1], (2, 1, 0)).astype(np.float32, copy=False)
        bz[k0:k1] = np.transpose(values[2], (2, 1, 0)).astype(np.float32, copy=False)
    return {"Bx": bx, "By": by, "Bz": bz}, {
        "model": name.upper(), "parameters": {key: float(value) for key, value in config.__dict__.items()}, "shape_zyx": [nz, ny, nx],
        "spacing_mm": h, "x_bounds_mm": [float(x[0]), float(x[-1])],
        "y_bounds_mm": [float(y[0]), float(y[-1])], "z_bounds_mm": [float(z[0]), float(z[-1])],
    }


def load_topology_candidate(path: Path) -> dict[str, np.ndarray]:
    """Load a candidate field from B=(3,z,y,x) or Bx/By/Bz NPZ arrays."""
    with np.load(path) as data:
        if "B" in data:
            values = np.asarray(data["B"])
            if values.ndim != 4 or values.shape[0] != 3:
                raise ValueError(f"{path}: B must have shape (3,nz,ny,nx)")
            return {key: np.asarray(values[index], dtype=np.float32) for index, key in enumerate(("Bx", "By", "Bz"))}
        if all(key in data for key in ("Bx", "By", "Bz")):
            return {key: np.asarray(data[key], dtype=np.float32) for key in ("Bx", "By", "Bz")}
    raise KeyError(f"{path}: expected B or Bx/By/Bz arrays")


def q_topology_metrics(reference: np.ndarray, candidate: np.ndarray, *, high_logq: float) -> dict[str, float | int]:
    return q_map_metrics(reference, candidate, high_logq=high_logq)


def plot_td_topology_qmaps(
    qmaps: dict[str, np.ndarray], metadata: dict[str, dict[str, Any]], output_path: Path, *, high_logq: float
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 4.35), sharey=True, constrained_layout=True)
    for axis, name in zip(axes, ("td1", "td2", "td3")):
        q = qmaps[name]
        logq = np.log10(np.maximum(q, 1.0))
        bounds = metadata[name]
        image = axis.imshow(
            logq, origin="lower", aspect="auto", cmap="inferno", vmin=0.0, vmax=max(high_logq, float(np.nanpercentile(logq, 99.5))),
            extent=(bounds["x_bounds_mm"][0], bounds["x_bounds_mm"][1], bounds["z_bounds_mm"][0], bounds["z_bounds_mm"][1]),
        )
        axis.set_title(f"{name.upper()} analytic — $y=0$")
        axis.set_xlabel("x (Mm)")
        axis.set_xlim(-100.0, 100.0)
        axis.set_ylim(0.0, min(150.0, bounds["z_bounds_mm"][1]))
        if name == "td3":
            axis.contour(logq, levels=[high_logq], colors="cyan", linewidths=0.65, origin="lower", extent=(bounds["x_bounds_mm"][0], bounds["x_bounds_mm"][1], bounds["z_bounds_mm"][0], bounds["z_bounds_mm"][1]))
    axes[0].set_ylabel("z (Mm)")
    colorbar = fig.colorbar(image, ax=axes, pad=0.015)
    colorbar.set_label(r"$\log_{10} Q$ (FastQSL)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_td_topology_qsl(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the GPU FastQSL TD1--TD3 experiment and persist paper-facing data."""
    requested = [item.strip().lower() for item in args.td_topology_models.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(TD_TOPOLOGY))
    if unknown:
        raise ValueError(f"Unknown TD topology models: {unknown}")
    candidate_dir = args.td_topology_candidate_dir.resolve() if args.td_topology_candidate_dir else None
    out_dir = args.td_topology_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    qmaps: dict[str, np.ndarray] = {}
    geometry: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for name in requested:
        print(f"FastQSL topology: {name.upper()} analytic reference", flush=True)
        fields, info = td_topology_field(name, chunk_depth=args.td_topology_chunk_depth)
        reference = compute_y0_qmap(
            fields["Bx"], fields["By"], fields["Bz"], fastqsl_root=args.fastqsl_root,
            step_len=args.fastqsl_step_len, tol=args.fastqsl_tol, block_x=args.fastqsl_block_x,
        )
        q_reference = reference["q"]
        qmaps[name] = q_reference
        geometry[name] = info
        np.savez_compressed(out_dir / f"{name}_analytic_fastqsl_y0.npz", q=q_reference, flag_start=reference["flag_start"], flag_end=reference["flag_end"], start_xyz=reference["start_xyz"], end_xyz=reference["end_xyz"], line_len=reference["line_len"], metadata=np.asarray(json.dumps(json_safe({**info, **reference["metadata"]}), indent=2)))
        spacing = float(info["spacing_mm"])
        record: dict[str, Any] = {"model": name.upper(), "field": "analytic_reference", **info, **reference["metadata"], **fastqsl_bundle_metrics(reference, reference, high_logq=args.td_topology_high_logq, spacing_xyz=(spacing, spacing, spacing))}
        records.append(record)
        if candidate_dir is not None:
            candidate_path = candidate_dir / f"{name}.npz"
            if not candidate_path.is_file():
                print(f"  candidate skipped (not found): {candidate_path}", flush=True)
                continue
            candidate = load_topology_candidate(candidate_path)
            candidate_q = compute_y0_qmap(candidate["Bx"], candidate["By"], candidate["Bz"], fastqsl_root=args.fastqsl_root, step_len=args.fastqsl_step_len, tol=args.fastqsl_tol, block_x=args.fastqsl_block_x)
            if candidate_q["q"].shape != q_reference.shape:
                raise ValueError(f"{candidate_path}: Q shape {candidate_q['q'].shape} does not match {q_reference.shape}")
            candidate_record = {"model": name.upper(), "field": str(candidate_path), **info, **candidate_q["metadata"], **fastqsl_bundle_metrics(reference, candidate_q, high_logq=args.td_topology_high_logq, spacing_xyz=(spacing, spacing, spacing))}
            records.append(candidate_record)
            np.savez_compressed(out_dir / f"{name}_candidate_fastqsl_y0.npz", q=candidate_q["q"], flag_start=candidate_q["flag_start"], flag_end=candidate_q["flag_end"], start_xyz=candidate_q["start_xyz"], end_xyz=candidate_q["end_xyz"], line_len=candidate_q["line_len"], metadata=np.asarray(json.dumps(json_safe(candidate_record), indent=2)))
    (out_dir / "metrics.json").write_text(json.dumps(json_safe(records), indent=2) + "\n")
    with (out_dir / "metrics.csv").open("w", newline="") as stream:
        keys = ("model", "field", "valid_pixels", "logq_mae", "logq_rmse", "logq_pearson", "high_q_dice", "high_q_jaccard", "endpoint_valid_seeds", "endpoint_pair_rmse", "endpoint_pair_mae", "endpoint_pair_p95", "endpoint_boundary_pair_match_fraction", "elapsed_s", "plane_y_index")
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in keys} for record in records)
    if set(requested) == set(TD_TOPOLOGY):
        plot_td_topology_qmaps(qmaps, geometry, args.figures_dir / "figure_td_fastqsl_qmaps.png", high_logq=args.td_topology_high_logq)
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=default_source_root(), help="Checkout containing data/td2/bcor and precomputed Fortran output.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper_writing" / "analytic_validation")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "paper_writing" / "core_figures")
    parser.add_argument("--field-cache-dir", type=Path, default=PROJECT_ROOT / "results" / "analytic_validation" / "fields")
    parser.add_argument("--benchmarks", default="low_lou_c2,td", help="Comma-separated subset of low_lou_c2,td.")
    parser.add_argument("--init-modes", default=",".join(INIT_MODES), help="Comma-separated subset of potential,curl_free_current,auto_rbsl.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--calculation-domain-mode",
        choices=("mudpack", "mudpack_legacy", "bottom_padding"),
        default="bottom_padding",
        help=(
            "Calculation-grid rule. bottom_padding (the publication default) "
            "uses b=(Nx+Ny)//6 on both x/y sides and "
            "Nz=min(Nx_calc,Ny_calc). mudpack uses equal left/right MUDPACK "
            "pads; mudpack_legacy matches Fortran odd-remainder placement."
        ),
    )
    parser.add_argument("--max-iter-lnl", type=int, default=1000)
    parser.add_argument("--max-iter-td", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--sigma-improvement-tol", type=float, default=1e-5)
    parser.add_argument("--auto-rbsl-axis-n-pts", type=int, default=160)
    parser.add_argument("--auto-rbsl-chunk-size", type=int, default=4)
    parser.add_argument("--figure2-init", choices=INIT_MODES, default="curl_free_current")
    parser.add_argument("--td-reference-axis-y", type=float, default=180.0, help="Manual PyVista y seed for the magenta reference TD current axis.")
    parser.add_argument("--td-solution-axis-y", type=float, default=311.4790344238281, help="Manual PyVista y seed for the magenta stable-solution current axis.")
    parser.add_argument("--td-topology-qsl", action="store_true", help="Run the GPU FastQSL topological-consistency experiment for TD1--TD3.")
    parser.add_argument("--topology-only", action="store_true", help="Run only --td-topology-qsl; skip the Low--Lou/NFPT metric suite.")
    parser.add_argument("--td-topology-models", default="td1,td2,td3", help="Comma-separated TD models for FastQSL: td1,td2,td3.")
    parser.add_argument("--td-topology-output-dir", type=Path, default=PROJECT_ROOT / "paper_writing" / "analytic_validation" / "td_topology_fastqsl")
    parser.add_argument("--td-topology-candidate-dir", type=Path, default=None, help="Optional directory with td1.npz, td2.npz, td3.npz candidate fields for Q-map comparison.")
    parser.add_argument("--td-topology-chunk-depth", type=int, default=4, help="Analytic TD z planes evaluated per CPU chunk before GPU Q tracing.")
    parser.add_argument("--td-topology-high-logq", type=float, default=3.0, help="high-Q threshold in log10(Q) for the Dice topological-overlap metric.")
    parser.add_argument("--fastqsl-root", type=Path, default=DEFAULT_FASTQSL_ROOT)
    parser.add_argument("--fastqsl-step-len", type=float, default=1.0 / 8.0)
    parser.add_argument("--fastqsl-tol", type=float, default=0.1)
    parser.add_argument("--fastqsl-block-x", type=int, default=128)
    parser.add_argument("--no-field-cache", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.topology_only:
        if not args.td_topology_qsl:
            raise SystemExit("--topology-only requires --td-topology-qsl")
        records = run_td_topology_qsl(args)
        print(f"FastQSL topology metrics={args.td_topology_output_dir / 'metrics.csv'}")
        print(f"FastQSL topology cases={len(records)}")
        return 0
    source_root = args.source_root.resolve()
    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    selected_benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    selected_init_modes = [item.strip() for item in args.init_modes.split(",") if item.strip()]
    unknown_benchmarks = sorted(set(selected_benchmarks) - {"low_lou_c2", "td"})
    unknown_init_modes = sorted(set(selected_init_modes) - set(INIT_MODES))
    if unknown_benchmarks or unknown_init_modes:
        raise SystemExit(f"Unknown benchmarks={unknown_benchmarks}, init_modes={unknown_init_modes}")

    print(f"source_root={source_root}")
    print(f"device={device} dtype={dtype}")
    cases = make_cases(source_root, calculation_domain_mode=args.calculation_domain_mode)
    records: list[dict[str, Any]] = []
    solutions: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for benchmark in selected_benchmarks:
        case = cases[benchmark]
        max_iter = args.max_iter_lnl if benchmark == "low_lou_c2" else args.max_iter_td
        for init_mode in selected_init_modes:
            print(f"Running {case['display']} | {SOLVER_PUBLIC_NAME} | {INIT_LABELS[init_mode]}", flush=True)
            record, fields = run_stable_case(
                case,
                init_mode,
                device=device,
                dtype=dtype,
                max_iter=max_iter,
                tol=args.tol,
                patience=args.patience,
                sigma_improvement_tol=args.sigma_improvement_tol,
                auto_rbsl_axis_n_pts=args.auto_rbsl_axis_n_pts,
                auto_rbsl_chunk_size=args.auto_rbsl_chunk_size,
                verbose=args.verbose,
            )
            records.append(record)
            solutions[(benchmark, init_mode)] = fields
            print(
                f"  C_vec={record['C_vec']:.6f} C_CS={record['C_CS']:.6f} "
                f"E'_n={record['Eprime_n']:.6f} E'_m={record['Eprime_m']:.6f} "
                f"epsilon={record['epsilon']:.6f} sigma_J={record['sigma_J']:.6f} "
                f"iter={record['best_iter']} wall={record['wall_time_s']:.2f}s",
                flush=True,
            )
            history_name = f"{benchmark}__fast_nfpt__{init_mode}.csv"
            write_history_csv(args.output_dir / "histories" / history_name, record)
            if not args.no_field_cache:
                save_field_cache(args.field_cache_dir / f"{benchmark}__fast_nfpt__{init_mode}.npz", fields)

    fortran_bundle = load_fortran_td_record(cases["td"], source_root) if "td" in selected_benchmarks else None
    if fortran_bundle is not None:
        fortran_record, fortran_fields = fortran_bundle
        records.append(fortran_record)
        solutions[("td", "nfpt_fortran")] = fortran_fields
        write_history_csv(args.output_dir / "histories" / "td__nfpt_fortran__curl_free_current.csv", fortran_record)
        print(
            f"Loaded original Fortran TD result: C_vec={fortran_record['C_vec']:.6f} "
            f"sigma_J={fortran_record['sigma_J']:.6f}"
        )
    else:
        print("Original Fortran TD output was not found; Figure 1 will be skipped.")

    lnl_records = [record for record in records if record["benchmark"] == "low_lou_c2" and record["solver"] == SOLVER_PUBLIC_NAME]
    comparison = {record["init_mode"]: jarolim_comparison(record) for record in lnl_records}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(args.output_dir / "metrics.csv", records)
    (args.output_dir / "metrics.json").write_text(json.dumps(json_safe(records), indent=2))
    (args.output_dir / "jarolim_c2_comparison.json").write_text(json.dumps(json_safe(comparison), indent=2))
    write_latex_rows(args.output_dir / "analytic_table_rows.tex", records)

    if not args.skip_figures:
        stable_cfcf = next(
            (record for record in records if record["benchmark"] == "td" and record["solver"] == SOLVER_PUBLIC_NAME and record["init_mode"] == "curl_free_current"),
            None,
        )
        if fortran_bundle is not None and stable_cfcf is not None:
            plot_figure1(
                fortran_bundle[0],
                stable_cfcf,
                args.figures_dir / "figure1_fortran_vs_fast_nfpt.png",
            )
        figure2_fields = solutions.get(("td", args.figure2_init))
        if figure2_fields is not None:
            plot_figure2(
                cases["td"],
                figure2_fields,
                init_mode=args.figure2_init,
                reference_axis_y=args.td_reference_axis_y,
                solution_axis_y=args.td_solution_axis_y,
                output_path=args.figures_dir / "figure2_td_reference_vs_fast_nfpt.png",
            )

    if args.td_topology_qsl:
        topology_records = run_td_topology_qsl(args)
        print(f"FastQSL topology cases={len(topology_records)}")

    all_better = [mode for mode, result in comparison.items() if result["all_equal_or_better_at_published_precision"]]
    print(f"Jarolim C2 equal-or-better across all six published metrics: {all_better}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
