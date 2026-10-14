#!/usr/bin/env python3
"""Run NFGS on TD1--TD3 analytic references and build the unified figure.

For each Titov--Demoulin configuration the script:
  1. builds the analytic reference on the TD lattice,
  2. reconstructs the field with NFGS from the bottom vector boundary,
  3. evaluates Schrijver-style metrics (TD1/TD3: central Nx0/2 x Ny0/2 x Nz;
     TD2: flux-rope-centered 66 x 66 x 30 ROI adapting the Demcsak et al.
     grid-point count), and
  4. for TD3 also compares FastQSL ``y=0`` Q maps.

Visualisation knobs and rendering live in:
  - ``util/td_viz_config.py`` (``TD_VIZ``)
  - ``util/td_viz.py`` (``plot_unified_td_figure``)

For fast visualisation iteration (seeds / camera / Q colormap), use::

  python scripts/run_nfgs_analytic_validation.py --figures-only --reuse-field-cache

This skips solving and metrics, loads cached analytic references + NFGS
solutions + TD3 Q maps, and only regenerates the unified figure.
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
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfpt_analytic_matplotlib")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/nfpt_analytic_mesa")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.scripts import run_analytic_validation as analytic  # noqa: E402
from nfgs_solver import NFGSSolverGPU  # noqa: E402
from nfgs._backend.util.fastqsl_plane import (  # noqa: E402
    DEFAULT_FASTQSL_ROOT,
    Y0_PLANE_COORDINATE_CONTRACT,
    compute_y0_qmap,
    y0_qmap_cache_matches,
)
from nfgs._backend.util.field_ops import curl_zyx_nfpt  # noqa: E402
from nfgs._backend.util.metrics import (  # noqa: E402
    compute_nf2_metrics,
    flux_rope_center_from_bottom_jz,
    test_domain_slices_zyx,
)
from nfgs._backend.util.td_viz import clear_viz_caches, plot_unified_td_figure  # noqa: E402

TD_NAMES = ("td1", "td2", "td3")
LEGACY_NAMES = ("low_lou_c2", "td")
ALL_BENCHMARKS = TD_NAMES + LEGACY_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_paper"
            / "frozen_linear_harmonic_vertical_staggered"
            / "reference_configuration"
        ),
    )
    parser.add_argument(
        "--field-cache-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_paper"
            / "frozen_linear_harmonic_vertical_staggered"
            / "reference_configuration"
            / "fields"
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=(
            ROOT
            / "paper_writing"
            / "core_figures"
            / "frozen_linear_harmonic_vertical_staggered"
        ),
    )
    parser.add_argument(
        "--benchmarks",
        default="td1,td2,td3",
        help="Comma-separated subset of td1,td2,td3,low_lou_c2,td.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--init-mode",
        choices=("gs_cfcf", "cfcf"),
        default="gs_cfcf",
        help="NFGS initializer (default: native Gaussian-separated CFCF).",
    )
    parser.add_argument(
        "--lnl-calculation-domain-mode",
        choices=("mudpack", "mudpack_legacy", "bottom_padding"),
        default="mudpack",
        help=(
            "Calculation-domain rule for Low--Lou C2 (default: mudpack). "
            "mudpack uses equal left/right MUDPACK pads; mudpack_legacy matches "
            "Fortran odd-remainder placement."
        ),
    )
    parser.add_argument(
        "--td-calculation-domain-mode",
        choices=("mudpack", "mudpack_legacy", "bottom_padding"),
        default="mudpack",
        help=(
            "Calculation-domain rule for TD models (default: mudpack). "
            "mudpack uses equal left/right MUDPACK pads; mudpack_legacy matches "
            "Fortran odd-remainder placement."
        ),
    )
    parser.add_argument("--max-iter-lnl", type=int, default=1000)
    parser.add_argument("--max-iter-td", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--sigma-improvement-tol", type=float, default=1.0e-5)
    parser.add_argument(
        "--stop-mode",
        choices=("sigma_best", "fixed_point"),
        default="sigma_best",
    )
    parser.add_argument("--fixed-point-update-tol", type=float, default=None)
    parser.add_argument("--fixed-point-pde-tol", type=float, default=1.0e-8)
    parser.add_argument("--fixed-point-observable-tol", type=float, default=None)
    parser.add_argument("--fixed-point-patience", type=int, default=10)
    parser.add_argument(
        "--source-partition-mode",
        choices=("gaussian_separated", "legacy_single_field"),
        default="gaussian_separated",
    )
    parser.add_argument("--alpha-clip-scale", type=float, default=1.0)
    parser.add_argument("--weak-field-floor-b2", type=float, default=1.0e-20)
    parser.add_argument(
        "--mean-bz-policy",
        choices=("uniform_open", "require_balanced"),
        default="uniform_open",
    )
    parser.add_argument(
        "--horizontal-zero-mode-policy",
        choices=(
            "omit",
            "frozen_linear_harmonic",
            "constrained_spline_harmonic",
            "coupled_k0",
        ),
        default="frozen_linear_harmonic",
    )
    parser.add_argument("--k0-spline-basis-size", type=int, default=8)
    parser.add_argument("--k0-ode-smoothness", type=float, default=1.0e-3)
    parser.add_argument(
        "--d0-zero-mode-policy",
        choices=("error", "warn_project", "project"),
        default="project",
        help="Project the incompatible finite-patch D0 constant mode and record it.",
    )
    parser.add_argument(
        "--source-zero-mode-policy",
        choices=("error", "warn_project", "project"),
        default="project",
        help="Project incompatible nonlinear source zero modes and retain diagnostics.",
    )
    parser.add_argument(
        "--harmonic-residual-policy",
        choices=("diagnose", "warn", "error"),
        default="diagnose",
        help="How to handle the measured finite-domain horizontal harmonic residual.",
    )
    parser.add_argument(
        "--gaussian-horizontal-operator",
        choices=("fd", "spectral"),
        default="spectral",
        help=(
            "Horizontal div/grad used in Gaussian separation: finite difference "
            "or endpoint-Neumann DCT-consistent spectral operators (default)."
        ),
    )
    parser.add_argument(
        "--vertical-current-operator",
        choices=("staggered_face", "collocated_centered"),
        default="staggered_face",
        help=(
            "Vertical current/force-free placement. The production default "
            "uses face differences; collocated_centered is the legacy ablation."
        ),
    )
    parser.add_argument("--mean-mode-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--source-zero-mode-tolerance", type=float, default=1.0e-2)
    parser.add_argument(
        "--bottom-divergence-projection-weight",
        type=float,
        default=0.0,
        help=(
            "0..1 relaxed post-solve projection toward the processed bottom "
            "horizontal-divergence constraint (default: disabled)."
        ),
    )
    parser.add_argument(
        "--bottom-divergence-robin-weight",
        type=float,
        default=0.0,
        help=(
            "0..1 blend between the first-interior poloidal PDE row and the "
            "bottom derivative condition (default: disabled; mutually "
            "exclusive with projection)."
        ),
    )
    parser.add_argument(
        "--poloidal-update-mode",
        choices=("direct", "two_stage"),
        default="direct",
        help=(
            "Solve Phi_c directly (production default) or solve Bc_z first "
            "and invert the horizontal Laplacian (conditioning ablation)."
        ),
    )
    parser.add_argument(
        "--record-iteration-diagnostics",
        action="store_true",
        help="Record scalar-update, stationary-PDE, imprint, and limiter histories.",
    )
    parser.add_argument(
        "--handoff-projection-policy", choices=("project", "error"), default="project"
    )
    parser.add_argument("--handoff-projection-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--td-topology-chunk-depth",
        type=int,
        default=4,
        help="Analytic TD z planes evaluated per CPU chunk.",
    )
    parser.add_argument("--td-topology-high-logq", type=float, default=3.0)
    parser.add_argument("--fastqsl-root", type=Path, default=DEFAULT_FASTQSL_ROOT)
    parser.add_argument("--fastqsl-step-len", type=float, default=1.0 / 8.0)
    parser.add_argument("--fastqsl-tol", type=float, default=0.1)
    parser.add_argument("--fastqsl-block-x", type=int, default=128)
    parser.add_argument(
        "--qmap-b-smooth-sigma-cells",
        type=float,
        default=0.0,
        help=(
            "Common FFT Gaussian sigma (grid cells) applied to both reference "
            "and reconstruction only while computing Q (default 0: raw B). "
            "Field caches and vector metrics remain raw."
        ),
    )
    parser.add_argument(
        "--n-test-x",
        type=int,
        default=66,
        help="TD2 metric ROI width in x (flux-rope-centered, Demcsak-count ROI).",
    )
    parser.add_argument(
        "--n-test-y",
        type=int,
        default=66,
        help="TD2 metric ROI width in y (flux-rope-centered, Demcsak-count ROI).",
    )
    parser.add_argument(
        "--n-test-z",
        type=int,
        default=30,
        help="TD2 metric ROI height in z (flux-rope-centered, Demcsak-count ROI).",
    )
    parser.add_argument(
        "--reuse-field-cache",
        action="store_true",
        help="Reuse cached NFGS fields when present instead of re-solving.",
    )
    parser.add_argument("--no-field-cache", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--figures-only",
        "--viz-only",
        action="store_true",
        dest="figures_only",
        help=(
            "Skip solving and metrics; load cached analytic references, NFGS "
            "solutions, and TD3 Q maps, then regenerate the unified figure only. "
            "Use this to iterate on seed / camera / colormap changes quickly."
        ),
    )
    parser.add_argument(
        "--reuse-viz-seeds",
        action="store_true",
        help=(
            "With --figures-only, reuse disk-cached closed seed groups from the "
            "previous viz run (fast camera/colormap iteration). Omit this flag "
            "when changing the seed strategy."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _axis_footpoint_y_phys(params: dict[str, float]) -> float:
    """Approximate positive magnetic-axis footpoint y from the TD torus geometry."""
    r = float(params["R"])
    d = float(params["d"])
    return float(math.sqrt(max(r * r - d * d, 0.0)))


def make_td_case(
    name: str,
    *,
    calculation_domain_mode: str,
    chunk_depth: int,
    n_test_x: int,
    n_test_y: int,
    n_test_z: int,
) -> dict[str, Any]:
    """Build an NFGS case dict for one TD1--TD3 analytic reference.

    TD1/TD3 metrics use the central Nx0/2 x Ny0/2 x Nz volume (domain centre).
    TD2 metrics use a flux-rope-centered 66 x 66 x 30 ROI.  This adopts the
    grid-point count of the restricted analysis volume reported by Demcsak
    et al. (2020), not that paper's 158.4 x 158.4 x 102 Mm physical box.
    """
    fields, info = analytic.td_topology_field(name, chunk_depth=chunk_depth)
    bx = np.asarray(fields["Bx"], dtype=np.float64)
    by = np.asarray(fields["By"], dtype=np.float64)
    bz = np.asarray(fields["Bz"], dtype=np.float64)
    nz, ny, nx = bx.shape
    dx = dy = dz = float(info["spacing_mm"])
    x_bounds = (float(info["x_bounds_mm"][0]), float(info["x_bounds_mm"][1]))
    y_bounds = (float(info["y_bounds_mm"][0]), float(info["y_bounds_mm"][1]))
    z_bounds = (float(info["z_bounds_mm"][0]), float(info["z_bounds_mm"][1]))
    _jx, _jy, jz = curl_zyx_nfpt(bx, by, bz, dx=dx, dy=dy, dz=dz)
    jz_obs = np.asarray(jz[0], dtype=np.float64)
    iy_c, ix_c = flux_rope_center_from_bottom_jz(jz_obs)
    if str(name) == "td2":
        metric_slices = test_domain_slices_zyx(
            nz, ny, nx, iy_c, ix_c, n_test_x, n_test_y, n_test_z
        )
        metric_domain = (
            f"flux-rope-centered {n_test_x} x {n_test_y} x {n_test_z} cube "
            "(Demcsak et al. grid-count protocol; physical extent not copied)"
        )
    else:
        # Central half-domain: Nx0/2 x Ny0/2 horizontally, full Nz.
        n_half_x = max(1, int(nx) // 2)
        n_half_y = max(1, int(ny) // 2)
        metric_slices = test_domain_slices_zyx(
            nz, ny, nx, ny // 2, nx // 2, n_half_x, n_half_y, nz
        )
        metric_domain = (
            f"central {n_half_x} x {n_half_y} x {nz} "
            f"(Nx0/2 x Ny0/2 x Nz; domain centre)"
        )
    axis_y_phys = _axis_footpoint_y_phys(info["parameters"])
    return {
        "benchmark": name,
        "display": f"{name.upper()}",
        "dims": (nx, ny, nz, dx, dy, dz),
        "boundary": {
            "Bx": bx[0].astype(np.float64, copy=False),
            "By": by[0].astype(np.float64, copy=False),
            "Bz": bz[0].astype(np.float64, copy=False),
        },
        "reference": {"Bx": bx, "By": by, "Bz": bz},
        "Jz_obs": jz_obs,
        "metric_slices": metric_slices,
        "metric_shape": analytic.metric_shape(metric_slices),
        "metric_domain": metric_domain,
        "solver_kwargs": {
            "lateral_wall": "paper_bc1",
            "pin_potential_boundaries": False,
            "buffer_thickness": 0,
            "padded_buffer_zone": True,
            "pad_calculation_domain": True,
            "calculation_domain_mode": calculation_domain_mode,
        },
        "mu": 1.0,
        "x_bounds": x_bounds,
        "y_bounds": y_bounds,
        "z_bounds": z_bounds,
        "flux_rope_iy": iy_c,
        "flux_rope_ix": ix_c,
        "geometry": info,
        "axis_y_phys": axis_y_phys,
    }


def make_requested_cases(
    args: argparse.Namespace, requested: list[str]
) -> dict[str, dict[str, Any]]:
    source_root = args.source_root.resolve()
    cases: dict[str, dict[str, Any]] = {}
    need_legacy = any(name in requested for name in LEGACY_NAMES)
    if need_legacy:
        legacy = analytic.make_cases(
            source_root, calculation_domain_mode=args.lnl_calculation_domain_mode
        )
        if "low_lou_c2" in requested:
            cases["low_lou_c2"] = legacy["low_lou_c2"]
        if "td" in requested:
            cases["td"] = analytic.make_cases(
                source_root, calculation_domain_mode=args.td_calculation_domain_mode
            )["td"]
            # Corner-origin figure2 seed expressed in the physical frame.
            cases["td"]["axis_y_phys"] = 180.0 + float(cases["td"]["y_bounds"][0])
    for name in TD_NAMES:
        if name not in requested:
            continue
        cases[name] = make_td_case(
            name,
            calculation_domain_mode=args.td_calculation_domain_mode,
            chunk_depth=args.td_topology_chunk_depth,
            n_test_x=args.n_test_x,
            n_test_y=args.n_test_y,
            n_test_z=args.n_test_z,
        )
    return cases


def run_case(
    case: dict,
    *,
    device: str,
    dtype: torch.dtype,
    max_iter: int,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, np.ndarray]]:
    nx, ny, nz, dx, dy, dz = case["dims"]
    solver = NFGSSolverGPU(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        device=device,
        dtype=dtype,
        source_partition_mode=getattr(
            args, "source_partition_mode", "gaussian_separated"
        ),
        alpha_clip_scale=getattr(args, "alpha_clip_scale", 1.0),
        weak_field_floor_b2=getattr(args, "weak_field_floor_b2", 1.0e-20),
        mean_bz_policy=args.mean_bz_policy,
        horizontal_zero_mode_policy=args.horizontal_zero_mode_policy,
        k0_ode_smoothness=args.k0_ode_smoothness,
        coupled_k0_discretization=getattr(
            args,
            "coupled_k0_discretization",
            "staggered_face_D1_true_three_point_curvature",
        ),
        k0_update_mu=getattr(args, "k0_update_mu", None),
        k0_spline_basis_size=getattr(args, "k0_spline_basis_size", 8),
        k0_current_target_placement=getattr(
            args, "k0_current_target_placement", None
        ),
        bottom_parallel_current_mode=getattr(
            args, "bottom_parallel_current_mode", "observed_full"
        ),
        d0_zero_mode_policy=args.d0_zero_mode_policy,
        source_zero_mode_policy=args.source_zero_mode_policy,
        harmonic_residual_policy=args.harmonic_residual_policy,
        mean_mode_tolerance=args.mean_mode_tolerance,
        source_zero_mode_tolerance=args.source_zero_mode_tolerance,
        bottom_divergence_projection_weight=(args.bottom_divergence_projection_weight),
        bottom_divergence_robin_weight=args.bottom_divergence_robin_weight,
        poloidal_update_mode=getattr(args, "poloidal_update_mode", "direct"),
        record_iteration_diagnostics=getattr(
            args, "record_iteration_diagnostics", False
        ),
        handoff_projection_policy=args.handoff_projection_policy,
        handoff_projection_tolerance=args.handoff_projection_tolerance,
        gaussian_horizontal_operator=getattr(
            args, "gaussian_horizontal_operator", "spectral"
        ),
        vertical_current_operator=getattr(
            args, "vertical_current_operator", "staggered_face"
        ),
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
    mu = float(getattr(args, "mu", case.get("mu", 1.0)))
    per_step_callback = getattr(args, "per_step_callback", None)
    bind_solver_callback = getattr(args, "bind_solver_callback", None)
    if bind_solver_callback is not None:
        per_step_callback = bind_solver_callback(solver)
    t0 = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    with torch.inference_mode():
        history = solver.iterate(
            max_iter=max_iter,
            tol=args.tol,
            mu=mu,
            patience=args.patience,
            sigma_improvement_tol=args.sigma_improvement_tol,
            verbose=args.verbose,
            polarity_mask="both",
            init_mode=args.init_mode,
            per_step_callback=per_step_callback,
            stop_mode=getattr(args, "stop_mode", "sigma_best"),
            fixed_point_update_tol=getattr(args, "fixed_point_update_tol", None),
            fixed_point_pde_tol=getattr(args, "fixed_point_pde_tol", 1.0e-8),
            fixed_point_observable_tol=getattr(
                args, "fixed_point_observable_tol", None
            ),
            fixed_point_patience=getattr(args, "fixed_point_patience", 10),
        )
    wall_s = time.perf_counter() - t0
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(torch.device(device)))
        if device.startswith("cuda")
        else None
    )
    fields = {key: np.asarray(value) for key, value in solver.fields_to_numpy().items()}
    metrics = compute_nf2_metrics(case["reference"], fields, case["metric_slices"])
    source = solver.compute_source_diagnostics()
    record = {
        "benchmark": case["benchmark"],
        "benchmark_display": case["display"],
        "solver": "NFGS",
        "source_partition_mode": solver.source_partition_mode,
        "init_mode": args.init_mode,
        "dtype": str(dtype),
        "resolved_device": device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda")
            else "CPU"
        ),
        "grid_xyz": [nx, ny, nz],
        "calculation_grid_xyz": [solver.Nx, solver.Ny, solver.Nz],
        "calculation_domain_mode": solver._calculation_domain_mode,
        "mean_bz_policy": solver.mean_bz_policy,
        "horizontal_zero_mode_policy": solver.horizontal_zero_mode_policy,
        "coupled_k0_discretization": solver.coupled_k0_discretization,
        "k0_update_mu": solver.k0_update_mu,
        "k0_spline_basis_size": solver.k0_spline_basis_size,
        "mu": mu,
        "d0_zero_mode_policy": solver.d0_zero_mode_policy,
        "source_zero_mode_policy": solver.source_zero_mode_policy,
        "harmonic_residual_policy": solver.harmonic_residual_policy,
        "gaussian_horizontal_operator": solver.gaussian_horizontal_operator,
        "vertical_current_operator": solver.vertical_current_operator,
        "bottom_divergence_projection_weight": (
            solver.bottom_divergence_projection_weight
        ),
        "bottom_divergence_robin_weight": solver.bottom_divergence_robin_weight,
        "poloidal_update_mode": solver.poloidal_update_mode,
        "record_iteration_diagnostics": solver.record_iteration_diagnostics,
        "metric_shape_zyx": list(case["metric_shape"]),
        "metric_domain": case["metric_domain"],
        "best_iter": int(history["best_iter"]),
        "n_iters": len(history["sigma"]),
        "best_sigma": float(history["best_sigma"]),
        "stop_reason": history["stop_reason"],
        "stop_mode": history["stop_mode"],
        "selected_state": history["selected_state"],
        "fixed_point_converged": bool(history["fixed_point_converged"]),
        "wall_time_s": wall_s,
        "iterations_per_s": len(history["sigma"]) / max(wall_s, 1.0e-30),
        "peak_memory_bytes": peak_memory_bytes,
        "source_diagnostics": source,
        **{key: float(metrics[key]) for key in analytic.METRIC_KEYS},
        "history": history,
    }
    del solver
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return record, fields


def compute_td3_q_bundle(
    case: dict[str, Any],
    solution_fields: dict[str, np.ndarray],
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    geometry = case["geometry"]
    ref = case["reference"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "td3_analytic_fastqsl_y0.npz"
    sol_path = out_dir / "td3_nfgs_fastqsl_y0.npz"
    smooth_sigma = float(getattr(args, "qmap_b_smooth_sigma_cells", 0.0))

    def load_q_and_metadata(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        with np.load(path) as data:
            q = np.asarray(data["q"])
            raw = data["metadata"] if "metadata" in data.files else None
        metadata: dict[str, Any] = {}
        if raw is not None:
            try:
                metadata = json.loads(str(raw.item() if hasattr(raw, "item") else raw))
            except Exception:
                metadata = {}
        return q, metadata

    if args.reuse_field_cache and ref_path.is_file() and sol_path.is_file():
        q_ref_cached, ref_meta = load_q_and_metadata(ref_path)
        q_sol_cached, sol_meta = load_q_and_metadata(sol_path)
        if (
            y0_qmap_cache_matches(
                ref_meta,
                ny=int(np.asarray(ref["Bx"]).shape[1]),
                b_smooth_sigma_cells=smooth_sigma,
            )
            and y0_qmap_cache_matches(
                sol_meta,
                ny=int(np.asarray(solution_fields["Bx"]).shape[1]),
                b_smooth_sigma_cells=smooth_sigma,
            )
        ):
            print("Loading cached TD3 FastQSL Q maps", flush=True)
            q_metrics = analytic.q_topology_metrics(
                q_ref_cached, q_sol_cached, high_logq=args.td_topology_high_logq
            )
            return {
                "q_ref": q_ref_cached,
                "q_sol": q_sol_cached,
                "geometry": geometry,
                "q_metrics": q_metrics,
                "reference_metadata": ref_meta,
                "solution_metadata": sol_meta,
            }
        print(
            "Ignoring stale/asymmetric TD3 Q cache "
            f"(need {Y0_PLANE_COORDINATE_CONTRACT} and common "
            f"qmap_b_smooth_sigma_cells={smooth_sigma:g}); recomputing",
            flush=True,
        )

    print(
        "FastQSL topology: TD3 analytic reference "
        f"(common qmap_b_smooth_sigma_cells={smooth_sigma:g})",
        flush=True,
    )
    reference_q = compute_y0_qmap(
        ref["Bx"],
        ref["By"],
        ref["Bz"],
        fastqsl_root=args.fastqsl_root,
        step_len=args.fastqsl_step_len,
        tol=args.fastqsl_tol,
        block_x=args.fastqsl_block_x,
        b_smooth_sigma_cells=smooth_sigma,
    )
    # Free GPU pressure before the second large TD3 trace.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    print(
        f"FastQSL topology: TD3 NFGS solution "
        f"(common qmap_b_smooth_sigma_cells={smooth_sigma:g})",
        flush=True,
    )
    solution_q = compute_y0_qmap(
        solution_fields["Bx"],
        solution_fields["By"],
        solution_fields["Bz"],
        fastqsl_root=args.fastqsl_root,
        step_len=args.fastqsl_step_len,
        tol=args.fastqsl_tol,
        block_x=args.fastqsl_block_x,
        b_smooth_sigma_cells=smooth_sigma,
    )
    q_ref = reference_q["q"]
    q_sol = solution_q["q"]
    q_metrics = analytic.q_topology_metrics(
        q_ref, q_sol, high_logq=args.td_topology_high_logq
    )
    np.savez_compressed(
        ref_path,
        q=q_ref,
        flag_start=reference_q["flag_start"],
        flag_end=reference_q["flag_end"],
        line_len=reference_q["line_len"],
        metadata=np.asarray(
            json.dumps(
                analytic.json_safe({**geometry, **reference_q["metadata"]}), indent=2
            )
        ),
    )
    np.savez_compressed(
        sol_path,
        q=q_sol,
        flag_start=solution_q["flag_start"],
        flag_end=solution_q["flag_end"],
        line_len=solution_q["line_len"],
        metadata=np.asarray(
            json.dumps(
                analytic.json_safe({**geometry, **solution_q["metadata"], **q_metrics}),
                indent=2,
            )
        ),
    )
    return {
        "q_ref": q_ref,
        "q_sol": q_sol,
        "geometry": geometry,
        "q_metrics": q_metrics,
        "reference_metadata": reference_q["metadata"],
        "solution_metadata": solution_q["metadata"],
    }


def _save_viz_seed_groups(path: Path, groups: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for idx, group in enumerate(groups):
        meta.append(
            {
                "label": str(group["label"]),
                "color": str(group["color"]),
                "radius": float(group["radius"]),
            }
        )
        arrays[f"seeds_{idx}"] = np.asarray(group["seeds"], dtype=np.float32)
    arrays["meta_json"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def _load_viz_seed_groups(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        groups: list[dict[str, Any]] = []
        for idx, item in enumerate(meta):
            groups.append(
                {
                    "label": item["label"],
                    "color": item["color"],
                    "radius": float(item["radius"]),
                    "seeds": np.asarray(data[f"seeds_{idx}"], dtype=np.float32),
                }
            )
    return groups


def _get_or_build_seed_groups(
    *,
    tag: str,
    builder,
    viz_seed_dir: Path | None,
    reuse_viz_seeds: bool,
) -> list[dict[str, Any]]:
    path = None if viz_seed_dir is None else viz_seed_dir / f"{tag}.npz"
    if reuse_viz_seeds and path is not None:
        cached = _load_viz_seed_groups(path)
        if cached is not None:
            print(f"  reused viz seeds {path.name}", flush=True)
            return cached
    groups = builder()
    if path is not None:
        _save_viz_seed_groups(path, groups)
        print(f"  saved viz seeds {path.name}", flush=True)
    return groups


def _field_cache_path(args: argparse.Namespace, name: str) -> Path:
    return args.field_cache_dir / f"{name}__nfgs__{args.init_mode}.npz"


def _reference_cache_path(args: argparse.Namespace, name: str) -> Path:
    return args.field_cache_dir / f"{name}__reference.npz"


def _case_meta_path(args: argparse.Namespace, name: str) -> Path:
    return args.field_cache_dir / f"{name}__case_meta.json"


def _load_cached_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _serialize_slices(slices: tuple[slice, slice, slice]) -> list[list[int | None]]:
    out: list[list[int | None]] = []
    for item in slices:
        out.append([item.start, item.stop, item.step])
    return out


def _deserialize_slices(raw: list[list[int | None]]) -> tuple[slice, slice, slice]:
    return tuple(slice(a, b, c) for a, b, c in raw)  # type: ignore[return-value]


def save_td_case_cache(args: argparse.Namespace, case: dict[str, Any]) -> None:
    """Persist analytic reference + lightweight case metadata for --figures-only."""
    name = str(case["benchmark"])
    if name not in TD_NAMES:
        return
    args.field_cache_dir.mkdir(parents=True, exist_ok=True)
    ref = case["reference"]
    analytic.save_field_cache(
        _reference_cache_path(args, name),
        {"Bx": ref["Bx"], "By": ref["By"], "Bz": ref["Bz"]},
    )
    meta = {
        "benchmark": name,
        "display": case["display"],
        "dims": list(case["dims"]),
        "x_bounds": list(case["x_bounds"]),
        "y_bounds": list(case["y_bounds"]),
        "z_bounds": list(case["z_bounds"]),
        "metric_slices": _serialize_slices(case["metric_slices"]),
        "metric_shape": list(case["metric_shape"]),
        "metric_domain": case["metric_domain"],
        "flux_rope_iy": int(case["flux_rope_iy"]),
        "flux_rope_ix": int(case["flux_rope_ix"]),
        "axis_y_phys": float(case["axis_y_phys"]),
        "geometry": case.get("geometry", {}),
        "mu": float(case.get("mu", 1.0)),
        "solver_kwargs": case.get("solver_kwargs", {}),
    }
    _case_meta_path(args, name).write_text(json.dumps(analytic.json_safe(meta), indent=2))


def load_td_case_cache(args: argparse.Namespace, name: str) -> dict[str, Any] | None:
    """Load a TD case previously saved by ``save_td_case_cache``."""
    ref_path = _reference_cache_path(args, name)
    meta_path = _case_meta_path(args, name)
    if not ref_path.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    ref = _load_cached_fields(ref_path)
    bx = np.asarray(ref["Bx"])
    by = np.asarray(ref["By"])
    bz = np.asarray(ref["Bz"])
    return {
        "benchmark": meta["benchmark"],
        "display": meta["display"],
        "dims": tuple(meta["dims"]),
        "boundary": {
            "Bx": bx[0].astype(np.float64, copy=False),
            "By": by[0].astype(np.float64, copy=False),
            "Bz": bz[0].astype(np.float64, copy=False),
        },
        "reference": {"Bx": bx, "By": by, "Bz": bz},
        "Jz_obs": None,
        "metric_slices": _deserialize_slices(meta["metric_slices"]),
        "metric_shape": tuple(meta["metric_shape"]),
        "metric_domain": meta["metric_domain"],
        "solver_kwargs": meta.get("solver_kwargs", {}),
        "mu": float(meta.get("mu", 1.0)),
        "x_bounds": tuple(meta["x_bounds"]),
        "y_bounds": tuple(meta["y_bounds"]),
        "z_bounds": tuple(meta["z_bounds"]),
        "flux_rope_iy": int(meta["flux_rope_iy"]),
        "flux_rope_ix": int(meta["flux_rope_ix"]),
        "geometry": meta.get("geometry", {}),
        "axis_y_phys": float(meta["axis_y_phys"]),
    }


def load_td3_q_bundle_cached(
    case: dict[str, Any],
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    """Load TD3 Q maps from disk without recomputing FastQSL."""
    ref_path = out_dir / "td3_analytic_fastqsl_y0.npz"
    sol_path = out_dir / "td3_nfgs_fastqsl_y0.npz"
    if not ref_path.is_file() or not sol_path.is_file():
        raise FileNotFoundError(
            f"Missing cached TD3 Q maps under {out_dir}. "
            "Run a full validation once with TD3 before --figures-only."
        )
    print("Loading cached TD3 FastQSL Q maps (figures-only)", flush=True)
    with np.load(ref_path) as data:
        q_ref = np.asarray(data["q"])
    with np.load(sol_path) as data:
        q_sol = np.asarray(data["q"])
    return {
        "q_ref": q_ref,
        "q_sol": q_sol,
        "geometry": case["geometry"],
        "q_metrics": {},
        "reference_metadata": {},
        "solution_metadata": {},
    }


def run_figures_only(args: argparse.Namespace) -> int:
    """Regenerate the unified figure from cached fields / Q maps only."""
    needed = list(TD_NAMES)
    missing_sol = [
        name for name in needed if not _field_cache_path(args, name).is_file()
    ]
    if missing_sol:
        raise SystemExit(
            "figures-only requires cached NFGS solutions for "
            f"{needed}; missing={missing_sol}"
        )

    print("figures-only: loading cached analytic references + NFGS solutions", flush=True)
    t0 = time.perf_counter()
    cases: dict[str, dict[str, Any]] = {}
    solutions: dict[str, dict[str, np.ndarray]] = {}
    for name in needed:
        case = load_td_case_cache(args, name)
        if case is None:
            print(f"  building+caching analytic reference for {name} ...", flush=True)
            case = make_td_case(
                name,
                calculation_domain_mode=args.td_calculation_domain_mode,
                chunk_depth=args.td_topology_chunk_depth,
                n_test_x=args.n_test_x,
                n_test_y=args.n_test_y,
                n_test_z=args.n_test_z,
            )
            save_td_case_cache(args, case)
        else:
            print(f"  loaded cached reference {name}", flush=True)
        cases[name] = case
        sol_path = _field_cache_path(args, name)
        solutions[name] = _load_cached_fields(sol_path)
        print(f"  loaded NFGS solution {sol_path.name}", flush=True)

    q_bundle = load_td3_q_bundle_cached(
        cases["td3"],
        args=args,
        out_dir=args.output_dir / "fastqsl",
    )
    # Clear in-process seed caches so viz edits always take effect unless
    # the user explicitly asked to reuse disk-cached viz seeds.
    clear_viz_caches()

    figure_path = (
        args.figures_dir / "figure_nfgs_td_reference_configuration_comparison.png"
    )
    viz_seed_dir = args.output_dir / "viz_seeds"
    print(f"Writing unified figure -> {figure_path}", flush=True)
    if args.reuse_viz_seeds:
        print(f"  reuse-viz-seeds from {viz_seed_dir}", flush=True)
    plot_unified_td_figure(
        cases=cases,
        solutions=solutions,
        q_bundle=q_bundle,
        high_logq=args.td_topology_high_logq,
        output_path=figure_path,
        viz_seed_dir=viz_seed_dir,
        reuse_viz_seeds=bool(args.reuse_viz_seeds),
    )
    elapsed = time.perf_counter() - t0
    print(f"figure={figure_path}", flush=True)
    print(f"figures-only done in {elapsed:.1f}s", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.figures_only:
        args.reuse_field_cache = True
        return run_figures_only(args)

    device = analytic.resolve_device(args.device)
    dtype = analytic.dtype_from_name(args.dtype)
    requested = [item.strip().lower() for item in args.benchmarks.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(ALL_BENCHMARKS))
    if unknown:
        raise SystemExit(f"Unknown benchmarks: {unknown}")

    print(f"device={device} dtype={dtype} init_mode={args.init_mode}", flush=True)
    cases = make_requested_cases(args, requested)
    # Cache analytic references for later --figures-only iterations.
    for name, case in cases.items():
        if name in TD_NAMES and not args.no_field_cache:
            save_td_case_cache(args, case)
    records: list[dict] = []
    solutions: dict[str, dict[str, np.ndarray]] = {}

    for name in requested:
        case = cases[name]
        max_iter = args.max_iter_lnl if name == "low_lou_c2" else args.max_iter_td
        cache_path = _field_cache_path(args, name)
        if args.reuse_field_cache and cache_path.is_file():
            print(f"Loading cached {case['display']} | NFGS | {args.init_mode}", flush=True)
            fields = _load_cached_fields(cache_path)
            metrics = compute_nf2_metrics(case["reference"], fields, case["metric_slices"])
            record = {
                "benchmark": case["benchmark"],
                "benchmark_display": case["display"],
                "solver": "NFGS",
                "init_mode": args.init_mode,
                "dtype": str(dtype),
                "grid_xyz": list(case["dims"][:3]),
                "calculation_grid_xyz": None,
                "calculation_domain_mode": args.td_calculation_domain_mode
                if name != "low_lou_c2"
                else args.lnl_calculation_domain_mode,
                "mean_bz_policy": args.mean_bz_policy,
                "horizontal_zero_mode_policy": args.horizontal_zero_mode_policy,
                "d0_zero_mode_policy": args.d0_zero_mode_policy,
                "source_zero_mode_policy": args.source_zero_mode_policy,
                "harmonic_residual_policy": args.harmonic_residual_policy,
                "vertical_current_operator": args.vertical_current_operator,
                "bottom_divergence_projection_weight": (
                    args.bottom_divergence_projection_weight
                ),
                "bottom_divergence_robin_weight": args.bottom_divergence_robin_weight,
                "poloidal_update_mode": args.poloidal_update_mode,
                "record_iteration_diagnostics": args.record_iteration_diagnostics,
                "metric_shape_zyx": list(case["metric_shape"]),
                "metric_domain": case["metric_domain"],
                "best_iter": None,
                "n_iters": None,
                "best_sigma": None,
                "stop_reason": "reused_field_cache",
                "wall_time_s": 0.0,
                "source_diagnostics": {},
                **{key: float(metrics[key]) for key in analytic.METRIC_KEYS},
                "history": {},
            }
        else:
            print(f"Running {case['display']} | NFGS | {args.init_mode}", flush=True)
            record, fields = run_case(
                case, device=device, dtype=dtype, max_iter=max_iter, args=args
            )
            if not args.no_field_cache:
                analytic.save_field_cache(cache_path, fields)
        records.append(record)
        solutions[name] = fields
        print(
            f"  C_vec={record['C_vec']:.6f} sigma_J={record['sigma_J']:.6f} "
            f"best_sigma={record.get('best_sigma')}",
            flush=True,
        )
        history_path = args.output_dir / "histories" / f"{name}__nfgs__{args.init_mode}.csv"
        if record.get("history"):
            analytic.write_history_csv(history_path, record)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_metrics() -> None:
        metric_fields = [
            "benchmark",
            "benchmark_display",
            "solver",
            "init_mode",
            "dtype",
            "grid_xyz",
            "calculation_grid_xyz",
            "calculation_domain_mode",
            "mean_bz_policy",
            "horizontal_zero_mode_policy",
            "d0_zero_mode_policy",
            "source_zero_mode_policy",
            "harmonic_residual_policy",
            "vertical_current_operator",
            "bottom_divergence_projection_weight",
            "bottom_divergence_robin_weight",
            "poloidal_update_mode",
            "record_iteration_diagnostics",
            "metric_shape_zyx",
            "metric_domain",
            "best_iter",
            "n_iters",
            "best_sigma",
            "stop_reason",
            "wall_time_s",
            *analytic.METRIC_KEYS,
        ]
        with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=metric_fields)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        key: json.dumps(record[key])
                        if isinstance(record.get(key), list)
                        else record.get(key)
                        for key in metric_fields
                    }
                )
        (args.output_dir / "metrics.json").write_text(
            json.dumps(analytic.json_safe(records), indent=2, allow_nan=True)
        )

    # Persist field metrics before optional FastQSL/figure stages.
    _write_metrics()

    q_bundle = None
    if "td3" in requested and "td3" in solutions:
        q_bundle = compute_td3_q_bundle(
            cases["td3"],
            solutions["td3"],
            args=args,
            out_dir=args.output_dir / "fastqsl",
        )
        for record in records:
            if record["benchmark"] == "td3":
                record["q_topology"] = q_bundle["q_metrics"]
                break
        print(
            "  TD3 Q metrics: "
            f"logq_mae={q_bundle['q_metrics']['logq_mae']:.4f} "
            f"logq_rmse={q_bundle['q_metrics']['logq_rmse']:.4f} "
            f"high_q_dice={q_bundle['q_metrics']['high_q_dice']:.4f}",
            flush=True,
        )
        _write_metrics()

    if not args.skip_figures and set(TD_NAMES).issubset(solutions):
        if q_bundle is None:
            raise SystemExit(
                "Unified TD figure requires TD3 FastQSL Q maps; "
                "include td3 in --benchmarks."
            )
        figure_path = (
            args.figures_dir / "figure_nfgs_td_reference_configuration_comparison.png"
        )
        print(f"Writing unified figure -> {figure_path}", flush=True)
        plot_unified_td_figure(
            cases=cases,
            solutions=solutions,
            q_bundle=q_bundle,
            high_logq=args.td_topology_high_logq,
            output_path=figure_path,
        )
        print(f"figure={figure_path}", flush=True)

    print(f"metrics={args.output_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
