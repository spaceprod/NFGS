#!/usr/bin/env python3
"""Run the mixed-Robin NFGS Low--Lou C2 and TD validation campaign.

The mixed-Robin weight is deliberately required.  The development sweep gave
different knees for analytic and observational objectives, so silently using
zero or guessing a publication value would make the result non-reproducible.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.experiments.nfgs_paper.analytic_nlfff import benchmarks as analytic  # noqa: E402
from nfgs_solver import NFGSSolverGPU  # noqa: E402
from nfgs._backend.util.metrics import compute_nf2_metrics  # noqa: E402
from nfgs.diagnostics.common_fd2 import sigma_j as common_fd2_sigma_j  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--field-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--benchmarks",
        default="low_lou_c2,low_lou_ll1,low_lou_ll2",
        help="Comma-separated Low--Lou family states and optional legacy td case.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--init-mode",
        choices=("gs_cfcf", "cfcf", "soft_cfcf"),
        default="gs_cfcf",
        help="NFGS initializer, including the boundary-exact soft Yi solve.",
    )
    parser.add_argument("--soft-cfcf-epsilon", type=float, default=None)
    parser.add_argument("--soft-cfcf-mode-batch-size", type=int, default=1024)
    parser.add_argument(
        "--lnl-calculation-domain-mode",
        choices=("mudpack", "mudpack_legacy", "bottom_padding"),
        default="mudpack",
        help=(
            "Calculation-domain rule for Low--Lou C2 (default: mudpack). "
            "mudpack: equal left/right pads; mudpack_legacy: Fortran placement."
        ),
    )
    parser.add_argument(
        "--td-calculation-domain-mode",
        choices=("mudpack", "mudpack_legacy", "bottom_padding"),
        default="mudpack",
        help=(
            "Calculation-domain rule for TD (default: mudpack). "
            "mudpack: equal left/right pads; mudpack_legacy: Fortran placement."
        ),
    )
    parser.add_argument("--max-iter-lnl", type=int, default=1000)
    parser.add_argument("--max-iter-td", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--sigma-improvement-tol", type=float, default=1.0e-5)
    parser.add_argument("--stop-mode", choices=("sigma_best",), default="sigma_best")
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
            "frozen_linear_harmonic",
            "constrained_spline_harmonic",
            "omit",
            "coupled_k0",
        ),
        default="frozen_linear_harmonic",
    )
    parser.add_argument("--k0-ode-smoothness", type=float, default=1.0e-3)
    parser.add_argument(
        "--poloidal-update-mode",
        choices=("direct", "two_stage"),
        default="direct",
    )
    parser.add_argument(
        "--vertical-current-operator",
        choices=("staggered_face", "collocated_centered"),
        default="staggered_face",
    )
    parser.add_argument(
        "--record-iteration-diagnostics",
        action="store_true",
        help="Record scalar/PDE/imprint/limiter histories for precision studies.",
    )
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
    parser.add_argument("--mean-mode-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--source-zero-mode-tolerance", type=float, default=1.0e-2)
    parser.add_argument(
        "--mixed-robin-weight",
        type=float,
        required=True,
        help=(
            "Required rho in (0, 1]: blend between the first-interior "
            "poloidal PDE row and the bottom derivative condition."
        ),
    )
    parser.add_argument(
        "--handoff-projection-policy", choices=("project", "error"), default="project"
    )
    parser.add_argument("--handoff-projection-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--no-field-cache", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace artifacts in an existing exact output directory.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_case_arrays(case: dict) -> str:
    """Fingerprint the exact common reference and bottom vector boundary."""
    digest = hashlib.sha256()
    for section in ("reference", "boundary"):
        for key in ("Bx", "By", "Bz"):
            array = np.ascontiguousarray(case[section][key])
            digest.update(section.encode("utf-8"))
            digest.update(key.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    encoded = json.dumps(
        analytic.json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _backup_path(path: Path) -> Path:
    """Return a unique, recoverable sibling path for an explicitly replaced artifact."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    return candidate


def _prepare_output_targets(
    *,
    output_dir: Path,
    field_cache_dir: Path,
    requested: list[str],
    init_mode: str,
    no_field_cache: bool,
    force: bool,
) -> dict[str, str]:
    """Enforce clean outputs; ``--force`` preserves replaced data as siblings."""
    unsafe_roots = {Path("/").resolve(), Path.home().resolve(), ROOT.resolve()}
    if output_dir.resolve() in unsafe_roots:
        raise RuntimeError(f"Refusing unsafe --output-dir target: {output_dir}")

    backups: dict[str, str] = {}
    if output_dir.exists() and any(output_dir.iterdir()):
        manifest_path = output_dir / "manifest.json"
        prior_hash = None
        if manifest_path.is_file():
            prior_hash = json.loads(manifest_path.read_text()).get(
                "configuration_sha256"
            )
        if not force:
            raise RuntimeError(
                f"Refusing to overwrite nonempty output directory {output_dir}; "
                f"prior configuration={prior_hash!r}. Choose a new --output-dir "
                "or pass --force explicitly."
            )
        backup = _backup_path(output_dir)
        output_dir.rename(backup)
        backups["output_dir"] = str(backup)

    output_dir.mkdir(parents=True, exist_ok=True)
    if no_field_cache or _is_within(field_cache_dir, output_dir):
        return backups

    for name in requested:
        target = field_cache_dir / f"{name}__nfgs__{init_mode}.npz"
        if not target.exists():
            continue
        if not force:
            raise RuntimeError(
                f"Refusing to overwrite external field cache {target}; choose a "
                "new --field-cache-dir or pass --force explicitly."
            )
        backup = _backup_path(target)
        target.rename(backup)
        backups[f"external_field_cache:{name}"] = str(backup)
    return backups


def slice_contract(items: tuple[slice, slice, slice]) -> list[list[int | None]]:
    return [[item.start, item.stop, item.step] for item in items]


def run_case(case: dict, *, device: str, dtype: torch.dtype, max_iter: int, args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    nx, ny, nz, dx, dy, dz = case["dims"]
    solver = NFGSSolverGPU(
        nx, ny, nz, dx, dy, dz,
        device=device,
        dtype=dtype,
        source_partition_mode="gaussian_separated",
        alpha_clip_scale=args.alpha_clip_scale,
        weak_field_floor_b2=args.weak_field_floor_b2,
        mean_bz_policy=args.mean_bz_policy,
        horizontal_zero_mode_policy=args.horizontal_zero_mode_policy,
        k0_ode_smoothness=args.k0_ode_smoothness,
        d0_zero_mode_policy=args.d0_zero_mode_policy,
        source_zero_mode_policy=args.source_zero_mode_policy,
        harmonic_residual_policy=args.harmonic_residual_policy,
        mean_mode_tolerance=args.mean_mode_tolerance,
        source_zero_mode_tolerance=args.source_zero_mode_tolerance,
        bottom_divergence_projection_weight=0.0,
        bottom_divergence_robin_weight=args.mixed_robin_weight,
        poloidal_update_mode=args.poloidal_update_mode,
        record_iteration_diagnostics=args.record_iteration_diagnostics,
        vertical_current_operator=args.vertical_current_operator,
        soft_cfcf_epsilon=args.soft_cfcf_epsilon,
        soft_cfcf_mode_batch_size=args.soft_cfcf_mode_batch_size,
        handoff_projection_policy=args.handoff_projection_policy,
        handoff_projection_tolerance=args.handoff_projection_tolerance,
        **case["solver_kwargs"],
    )
    boundary = case["boundary"]
    solver.set_bottom_vector_boundary(boundary["Bx"], boundary["By"], boundary["Bz"], zero_net_current=False, smooth_passes=0, ijzbcor=3)
    t0 = time.perf_counter()
    with torch.inference_mode():
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        history = solver.iterate(
            max_iter=max_iter, tol=args.tol, mu=float(case["mu"]), patience=args.patience,
            sigma_improvement_tol=args.sigma_improvement_tol, verbose=args.verbose,
            polarity_mask="both", init_mode=args.init_mode,
            stop_mode=args.stop_mode,
        )
    wall_s = time.perf_counter() - t0
    fields = {key: np.asarray(value) for key, value in solver.fields_to_numpy().items()}
    metrics = compute_nf2_metrics(case["reference"], fields, case["metric_slices"])
    sigma_common = common_fd2_sigma_j(
        fields, (float(dx), float(dy), float(dz)), case["metric_slices"]
    )
    source = solver.compute_source_diagnostics()
    record = {
        "benchmark": case["benchmark"], "benchmark_display": case["display"], "solver": "NFGS",
        "init_mode": args.init_mode, "device": device, "dtype": str(dtype), "grid_xyz": [nx, ny, nz],
        "source_partition_mode": solver.source_partition_mode,
        "calculation_grid_xyz": [solver.Nx, solver.Ny, solver.Nz],
        "calculation_domain_mode": solver._calculation_domain_mode,
        "mean_bz_policy": solver.mean_bz_policy,
        "horizontal_zero_mode_policy": solver.horizontal_zero_mode_policy,
        "d0_zero_mode_policy": solver.d0_zero_mode_policy,
        "source_zero_mode_policy": solver.source_zero_mode_policy,
        "harmonic_residual_policy": solver.harmonic_residual_policy,
        "mixed_robin_weight": solver.bottom_divergence_robin_weight,
        "poloidal_update_mode": solver.poloidal_update_mode,
        "vertical_current_operator": solver.vertical_current_operator,
        "record_iteration_diagnostics": solver.record_iteration_diagnostics,
        "soft_cfcf_epsilon": solver.soft_cfcf_epsilon,
        "max_iter": int(max_iter), "tol": float(args.tol),
        "patience": int(args.patience),
        "sigma_improvement_tol": float(args.sigma_improvement_tol),
        "metric_shape_zyx": list(case["metric_shape"]),
        "metric_slices_zyx": slice_contract(case["metric_slices"]),
        "metric_domain": case["metric_domain"],
        "analytic_reference_dtype": str(np.asarray(case["reference"]["Bx"]).dtype),
        "best_iter": int(history["best_iter"]), "n_iters": len(history["sigma"]),
        "best_sigma": float(history["best_sigma"]), "stop_reason": history["stop_reason"],
        "stop_mode": history["stop_mode"],
        "selected_state": history["selected_state"],
        "wall_time_s": wall_s,
        "iterations_per_s": len(history["sigma"]) / max(wall_s, 1.0e-30),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(device)))
            if device.startswith("cuda") else None
        ),
        "source_diagnostics": source,
        **{key: float(metrics[key]) for key in analytic.METRIC_KEYS},
        "sigma_J_common_fd2": sigma_common,
        "history": history,
    }
    del solver
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return record, fields


def main() -> int:
    args = parse_args()
    if not 0.0 < args.mixed_robin_weight <= 1.0:
        raise SystemExit("--mixed-robin-weight must be in (0, 1]")
    weight_tag = f"{args.mixed_robin_weight:.6g}".replace(".", "p")
    if args.output_dir is None:
        args.output_dir = (
            ROOT
            / "results"
            / "nfgs_paper"
            / "analytic"
            / f"{args.dtype}_mixed_robin_w{weight_tag}"
        )
    if args.field_cache_dir is None:
        args.field_cache_dir = args.output_dir / "fields"
    device = analytic.resolve_device(args.device)
    dtype = analytic.dtype_from_name(args.dtype)
    requested = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    low_lou_names = {"low_lou_c2", "low_lou_ll1", "low_lou_ll2"}
    unknown = sorted(set(requested) - (low_lou_names | {"td"}))
    if unknown:
        raise SystemExit(f"Unknown benchmarks: {unknown}")
    if not requested:
        raise SystemExit("At least one benchmark is required.")
    requested = list(dict.fromkeys(requested))
    source_root = args.source_root.resolve()
    cases: dict[str, dict] = {}
    for name in requested:
        if name in low_lou_names:
            cases[name] = analytic.make_low_lou_case(
                args.lnl_calculation_domain_mode, name
            )
    if "td" in requested:
        cases["td"] = analytic.make_td_case(
            source_root, args.td_calculation_domain_mode
        )

    reference_sha256 = {
        name: sha256_case_arrays(cases[name]) for name in requested
    }
    code_paths = {
        "runner": Path(__file__).resolve(),
        "benchmark_builder": Path(analytic.__file__).resolve(),
        "nfgs_solver": ROOT / "nfgs_solver.py",
        "nfpt_stable_solver": ROOT / "nfgs_solver.py",
        "nfpt_stable_base": ROOT / "nfgs_solver.py",
        "field_ops": ROOT / "nfgs" / "_backend" / "util" / "field_ops.py",
        "metrics": ROOT / "nfgs" / "_backend" / "util" / "metrics.py",
        "nfpt_grid_compat": ROOT / "nfgs" / "_backend" / "util" / "nfpt_fortran_compat.py",
    }
    if low_lou_names.intersection(requested):
        code_paths["low_lou_generator"] = ROOT / "nfgs" / "_backend" / "util" / "lnl_io.py"
    if "td" in requested:
        code_paths["td_loader"] = ROOT / "nfgs" / "_backend" / "util" / "td2_io.py"
    code_sha256 = {name: sha256_file(path) for name, path in code_paths.items()}
    input_paths: dict[str, Path] = {}
    if "td" in requested:
        input_paths["td_bcor"] = source_root / "data" / "analytic" / "td2" / "bcor"
    input_sha256 = {name: sha256_file(path) for name, path in input_paths.items()}

    configuration = {
        "schema_version": 1,
        "benchmarks": requested,
        "resolved_device": device,
        "solver_dtype": args.dtype,
        "init_mode": args.init_mode,
        "soft_cfcf_epsilon": args.soft_cfcf_epsilon,
        "soft_cfcf_mode_batch_size": args.soft_cfcf_mode_batch_size,
        "calculation_domain_modes": {
            name: (
                args.lnl_calculation_domain_mode
                if name in low_lou_names
                else args.td_calculation_domain_mode
            )
            for name in requested
        },
        "max_iter": {
            name: args.max_iter_lnl if name in low_lou_names else args.max_iter_td
            for name in requested
        },
        "tol": args.tol,
        "patience": args.patience,
        "sigma_improvement_tol": args.sigma_improvement_tol,
        "stop_mode": args.stop_mode,
        "alpha_clip_scale": args.alpha_clip_scale,
        "weak_field_floor_b2": args.weak_field_floor_b2,
        "mean_bz_policy": args.mean_bz_policy,
        "horizontal_zero_mode_policy": args.horizontal_zero_mode_policy,
        "k0_ode_smoothness": args.k0_ode_smoothness,
        "poloidal_update_mode": args.poloidal_update_mode,
        "vertical_current_operator": args.vertical_current_operator,
        "record_iteration_diagnostics": args.record_iteration_diagnostics,
        "d0_zero_mode_policy": args.d0_zero_mode_policy,
        "source_zero_mode_policy": args.source_zero_mode_policy,
        "harmonic_residual_policy": args.harmonic_residual_policy,
        "mean_mode_tolerance": args.mean_mode_tolerance,
        "source_zero_mode_tolerance": args.source_zero_mode_tolerance,
        "mixed_robin_weight": args.mixed_robin_weight,
        "bottom_divergence_projection_weight": 0.0,
        "handoff_projection_policy": args.handoff_projection_policy,
        "handoff_projection_tolerance": args.handoff_projection_tolerance,
        "boundary_preparation": {
            "zero_net_current": False,
            "smooth_passes": 0,
            "ijzbcor": 3,
            "polarity_mask": "both",
        },
        "case_contracts": {
            name: {
                "grid_xyz": list(cases[name]["dims"][:3]),
                "spacing_xyz": list(cases[name]["dims"][3:]),
                "metric_shape_zyx": list(cases[name]["metric_shape"]),
                "metric_slices_zyx": slice_contract(cases[name]["metric_slices"]),
                "metric_domain": cases[name]["metric_domain"],
                "analytic_reference_dtype": str(
                    np.asarray(cases[name]["reference"]["Bx"]).dtype
                ),
                "solver_kwargs": analytic.json_safe(cases[name]["solver_kwargs"]),
                "mu": float(cases[name]["mu"]),
                "benchmark_parameters": analytic.json_safe(
                    cases[name]["benchmark_parameters"]
                ),
                "reference_and_boundary_sha256": reference_sha256[name],
            }
            for name in requested
        },
        "code_sha256": code_sha256,
        "input_sha256": input_sha256,
    }
    configuration_sha256 = canonical_hash(configuration)
    args.output_dir = args.output_dir.resolve()
    args.field_cache_dir = args.field_cache_dir.resolve()
    manifest_path = args.output_dir / "manifest.json"
    replaced_artifact_backups = _prepare_output_targets(
        output_dir=args.output_dir,
        field_cache_dir=args.field_cache_dir,
        requested=requested,
        init_mode=args.init_mode,
        no_field_cache=args.no_field_cache,
        force=args.force,
    )

    records: list[dict] = []
    output_paths: dict[str, Path] = {}
    for name in requested:
        max_iter = args.max_iter_lnl if name in low_lou_names else args.max_iter_td
        print(f"Running {cases[name]['display']} | NFGS | {args.init_mode}", flush=True)
        record, fields = run_case(cases[name], device=device, dtype=dtype, max_iter=max_iter, args=args)
        record["configuration_sha256"] = configuration_sha256
        record["reference_and_boundary_sha256"] = reference_sha256[name]
        records.append(record)
        print(f"  C_vec={record['C_vec']:.6f} sigma_J={record['sigma_J']:.6f} best_sigma={record['best_sigma']:.6e}", flush=True)
        history_path = args.output_dir / "histories" / f"{name}__nfgs__{args.init_mode}.csv"
        analytic.write_history_csv(history_path, record)
        output_paths[f"{name}_history_csv"] = history_path
        if not args.no_field_cache:
            field_path = args.field_cache_dir / f"{name}__nfgs__{args.init_mode}.npz"
            analytic.save_field_cache(field_path, fields)
            output_paths[f"{name}_fields"] = field_path
    csv_fields = ["benchmark", "benchmark_display", "solver", "source_partition_mode", "init_mode", "device", "dtype", "grid_xyz", "calculation_grid_xyz", "calculation_domain_mode", "mean_bz_policy", "horizontal_zero_mode_policy", "d0_zero_mode_policy", "source_zero_mode_policy", "harmonic_residual_policy", "mixed_robin_weight", "poloidal_update_mode", "vertical_current_operator", "record_iteration_diagnostics", "soft_cfcf_epsilon", "max_iter", "tol", "patience", "sigma_improvement_tol", "stop_mode", "selected_state", "metric_shape_zyx", "metric_slices_zyx", "metric_domain", "analytic_reference_dtype", "configuration_sha256", "reference_and_boundary_sha256", "best_iter", "n_iters", "best_sigma", "stop_reason", "wall_time_s", "iterations_per_s", "peak_memory_bytes", *analytic.METRIC_KEYS, "sigma_J_common_fd2"]
    metrics_csv_path = args.output_dir / "metrics.csv"
    metrics_json_path = args.output_dir / "metrics.json"
    with metrics_csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields); writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(record[key]) if isinstance(record.get(key), list) else record.get(key) for key in csv_fields})
    metrics_json_path.write_text(
        json.dumps(analytic.json_safe(records), indent=2, allow_nan=True) + "\n"
    )
    output_paths.update({"metrics_csv": metrics_csv_path, "metrics_json": metrics_json_path})

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }
    manifest = {
        "experiment": "independent analytic NLFFF validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "code_paths": {name: str(path) for name, path in code_paths.items()},
        "code_sha256": code_sha256,
        "input_paths": {name: str(path.resolve()) for name, path in input_paths.items()},
        "input_sha256": input_sha256,
        "reference_and_boundary_sha256": reference_sha256,
        "runtime": runtime,
        "replaced_artifact_backups": replaced_artifact_backups,
        "diagnostic_scopes": {
            "sigma": "Fortran-CWsin diagnostic window on the calculation grid",
            "divB_and_energy": "complete calculation grid",
            "iteration_PDE_updates": "attempted pre-rollback candidate interior",
            "final_NF2_metrics": "the exact per-case metric_slices_zyx contract",
        },
        "smoke_warning": (
            "A max_iter value of 1 validates the execution path only and must not "
            "be used as nonlinear accuracy evidence."
        ),
        "outputs": {name: str(path.resolve()) for name, path in output_paths.items()},
        "output_sha256": {name: sha256_file(path) for name, path in output_paths.items()},
    }
    manifest_path.write_text(
        json.dumps(analytic.json_safe(manifest), indent=2, allow_nan=True) + "\n"
    )
    print(f"metrics={metrics_csv_path}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
