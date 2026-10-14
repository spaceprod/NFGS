#!/usr/bin/env python3
"""Sweep mixed-Robin weights on TD1--TD3 reference metrics only.

No FastQSL / topology and no figure generation.  For each robin weight in
``[0.0, 0.1, ..., 1.0]`` (override with ``--weights``), run NFGS with
``max_iter`` and record Schrijver-style reference metrics.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.scripts import run_analytic_validation as analytic  # noqa: E402
from nfgs._backend.scripts import run_nfgs_analytic_validation as driver  # noqa: E402
from nfgs._backend.experiments.nfgs_paper.analytic_nlfff.benchmarks import write_history_csv  # noqa: E402
from nfgs.diagnostics.common_fd2 import sigma_j as common_fd2_sigma_j  # noqa: E402


METRIC_KEYS = list(analytic.METRIC_KEYS)
DEFAULT_WEIGHTS = [round(i / 10.0, 1) for i in range(11)]
K0_DISCRETIZATION = "staggered_face_D1_true_three_point_curvature"
SOLVER_SOURCE = ROOT / "nfgs_solver.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_match_current_solver(
    records: list[dict[str, Any]],
    *,
    solver_sha256: str,
    horizontal_zero_mode_policy: str,
    expected_run_config: dict[str, Any] | None = None,
) -> bool:
    if not records:
        return False
    for record in records:
        if record.get("solver_sha256") != solver_sha256:
            return False
        if record.get("horizontal_zero_mode_policy") != horizontal_zero_mode_policy:
            return False
        if "sigma_J_common_fd2" not in record:
            return False
        if "device_name" not in record:
            return False
        if (
            expected_run_config is not None
            and record.get("run_config") != expected_run_config
        ):
            return False
        if horizontal_zero_mode_policy == "coupled_k0":
            discretization = (
                record.get("source_diagnostics", {})
                .get("coupled_k0", {})
                .get("discretization")
            )
            expected = K0_DISCRETIZATION
            if expected_run_config is not None:
                expected = str(
                    expected_run_config.get(
                        "coupled_k0_discretization", K0_DISCRETIZATION
                    )
                )
            if discretization != expected:
                return False
    return True


def weight_tag(weight: float) -> str:
    """Filesystem-safe robin weight label (0.7 -> '0p7')."""
    return f"{float(weight):.1f}".replace(".", "p")


def weight_json_path(out_dir: Path, weight: float) -> Path:
    return out_dir / f"robin_w{weight_tag(weight)}.json"


def field_cache_dir(out_dir: Path) -> Path:
    return out_dir / "fields"


def nfgs_field_cache_path(
    out_dir: Path, benchmark: str, init_mode: str, weight: float
) -> Path:
    return (
        field_cache_dir(out_dir)
        / f"{benchmark}__nfgs__{init_mode}__robin_w{weight_tag(weight)}.npz"
    )


def reference_field_cache_path(out_dir: Path, benchmark: str) -> Path:
    return field_cache_dir(out_dir) / f"{benchmark}__reference.npz"


def ensure_reference_field_cache(
    out_dir: Path, benchmark: str, reference: dict[str, np.ndarray]
) -> Path:
    """Compatibility helper used by existing cache postprocessors."""
    path = reference_field_cache_path(out_dir, benchmark)
    if not path.is_file():
        analytic.save_field_cache(path, reference)
    return path


def load_field_cache(path: Path) -> dict[str, np.ndarray]:
    """Compatibility helper for read-only postprocessors.

    The rho runner itself no longer trusts a bare NPZ for restart provenance;
    callers that explicitly postprocess a selected cache may still use this
    loader.
    """
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def parse_weights(value: str | None) -> list[float]:
    if value is None:
        return list(DEFAULT_WEIGHTS)
    weights = sorted({round(float(item.strip()), 12) for item in value.split(",")})
    if not weights or any(w < 0.0 or w > 1.0 for w in weights):
        raise argparse.ArgumentTypeError("weights must be a nonempty subset of [0, 1]")
    return weights


def make_run_args(robin_weight: float, base: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        mean_bz_policy=base.mean_bz_policy,
        horizontal_zero_mode_policy=base.horizontal_zero_mode_policy,
        k0_ode_smoothness=base.k0_ode_smoothness,
        coupled_k0_discretization=getattr(
            base,
            "coupled_k0_discretization",
            K0_DISCRETIZATION,
        ),
        k0_update_mu=getattr(base, "k0_update_mu", None),
        k0_spline_basis_size=getattr(base, "k0_spline_basis_size", 8),
        k0_current_target_placement=getattr(
            base, "k0_current_target_placement", None
        ),
        bottom_parallel_current_mode=getattr(
            base, "bottom_parallel_current_mode", "observed_full"
        ),
        mu=float(getattr(base, "mu", 1.0)),
        d0_zero_mode_policy=base.d0_zero_mode_policy,
        source_zero_mode_policy=base.source_zero_mode_policy,
        harmonic_residual_policy=base.harmonic_residual_policy,
        mean_mode_tolerance=base.mean_mode_tolerance,
        source_zero_mode_tolerance=base.source_zero_mode_tolerance,
        bottom_divergence_projection_weight=0.0,
        bottom_divergence_robin_weight=float(robin_weight),
        poloidal_update_mode=getattr(base, "poloidal_update_mode", "direct"),
        record_iteration_diagnostics=bool(
            getattr(base, "record_iteration_diagnostics", False)
        ),
        handoff_projection_policy=getattr(
            base, "handoff_projection_policy", "project"
        ),
        handoff_projection_tolerance=float(
            getattr(base, "handoff_projection_tolerance", 1.0e-6)
        ),
        gaussian_horizontal_operator=getattr(
            base, "gaussian_horizontal_operator", "spectral"
        ),
        vertical_current_operator=getattr(
            base, "vertical_current_operator", "staggered_face"
        ),
        source_partition_mode=getattr(
            base, "source_partition_mode", "gaussian_separated"
        ),
        alpha_clip_scale=float(getattr(base, "alpha_clip_scale", 1.0)),
        weak_field_floor_b2=float(getattr(base, "weak_field_floor_b2", 1.0e-20)),
        stop_mode=getattr(base, "stop_mode", "sigma_best"),
        fixed_point_update_tol=getattr(base, "fixed_point_update_tol", None),
        fixed_point_pde_tol=float(getattr(base, "fixed_point_pde_tol", 1.0e-8)),
        fixed_point_observable_tol=getattr(
            base, "fixed_point_observable_tol", None
        ),
        fixed_point_patience=int(getattr(base, "fixed_point_patience", 10)),
        tol=base.tol,
        patience=base.patience,
        sigma_improvement_tol=base.sigma_improvement_tol,
        verbose=bool(getattr(base, "verbose", False)),
        init_mode=base.init_mode,
        per_step_callback=getattr(base, "per_step_callback", None),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "robin_weight",
        "benchmark",
        "C_vec",
        "C_CS",
        "Eprime_n",
        "Eprime_m",
        "epsilon",
        "sigma_J",
        "best_sigma",
        "best_iter",
        "n_iters",
        "stop_reason",
        "stop_mode",
        "fixed_point_converged",
        "wall_time_s",
        "iterations_per_s",
        "peak_memory_bytes",
        "dtype",
        "init_mode",
        "metric_domain",
        "poloidal_update_mode",
        "vertical_current_operator",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank weights by mean C_vec (primary) then mean Eprime_n."""
    by_w: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_w.setdefault(float(row["robin_weight"]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for weight, group in sorted(by_w.items()):
        aggregates.append(
            {
                "robin_weight": weight,
                "mean_C_vec": float(np.mean([r["C_vec"] for r in group])),
                "mean_C_CS": float(np.mean([r["C_CS"] for r in group])),
                "mean_Eprime_n": float(np.mean([r["Eprime_n"] for r in group])),
                "mean_sigma_J": float(np.mean([r["sigma_J"] for r in group])),
                "td3_C_vec": float(
                    next(r["C_vec"] for r in group if r["benchmark"] == "td3")
                )
                if any(r["benchmark"] == "td3" for r in group)
                else None,
                "per_benchmark": {
                    r["benchmark"]: {
                        "C_vec": r["C_vec"],
                        "Eprime_n": r["Eprime_n"],
                        "sigma_J": r["sigma_J"],
                        "best_sigma": r["best_sigma"],
                        "n_iters": r["n_iters"],
                        "stop_reason": r["stop_reason"],
                    }
                    for r in group
                },
            }
        )

    best_mean = max(aggregates, key=lambda a: (a["mean_C_vec"], a["mean_Eprime_n"]))
    with_td3 = [a for a in aggregates if a["td3_C_vec"] is not None]
    best_td3 = (
        max(with_td3, key=lambda a: (a["td3_C_vec"], a["mean_C_vec"]))
        if with_td3
        else None
    )
    return {
        "primary_objective": "maximize mean C_vec across requested benchmarks",
        "best_by_mean_C_vec": best_mean,
        "best_by_td3_C_vec": best_td3,
        "aggregates": aggregates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_td_robin_weight_sweep_frozen_linear_harmonic_vertical_staggered"
        ),
    )
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--benchmarks", default="td1,td2,td3")
    parser.add_argument("--weights", default=None, help="Comma-separated in [0,1].")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--init-mode", choices=("gs_cfcf", "cfcf"), default="gs_cfcf")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--sigma-improvement-tol", type=float, default=1.0e-5)
    parser.add_argument(
        "--stop-mode", choices=("sigma_best", "fixed_point"), default="sigma_best"
    )
    parser.add_argument("--fixed-point-update-tol", type=float, default=None)
    parser.add_argument("--fixed-point-pde-tol", type=float, default=1.0e-8)
    parser.add_argument("--fixed-point-observable-tol", type=float, default=None)
    parser.add_argument("--fixed-point-patience", type=int, default=10)
    parser.add_argument("--alpha-clip-scale", type=float, default=1.0)
    parser.add_argument("--weak-field-floor-b2", type=float, default=1.0e-20)
    parser.add_argument("--mu", type=float, default=1.0, help="Scalar under-relaxation.")
    parser.add_argument(
        "--k0-update-mu",
        type=float,
        default=None,
        help="Optional dedicated under-relaxation for H_k0 (defaults to --mu).",
    )
    parser.add_argument(
        "--k0-spline-basis-size",
        type=int,
        default=8,
        help="Number of cubic B-spline basis functions for the constrained closure.",
    )
    parser.add_argument("--mean-bz-policy", default="uniform_open")
    parser.add_argument(
        "--horizontal-zero-mode-policy",
        default="frozen_linear_harmonic",
        help="Horizontal k=0 closure (production default: frozen_linear_harmonic).",
    )
    parser.add_argument("--k0-ode-smoothness", type=float, default=1.0e-3)
    parser.add_argument(
        "--coupled-k0-discretization",
        choices=(
            "staggered_face_D1_true_three_point_curvature",
            "legacy_plane_centred",
        ),
        default="staggered_face_D1_true_three_point_curvature",
        help=(
            "Production corrected face k_h=0 operator, or diagnosis-only "
            "legacy plane-centred operator."
        ),
    )
    parser.add_argument(
        "--checkpoint-iters",
        default="",
        help="Comma-separated iteration indices at which to save field checkpoints.",
    )
    parser.add_argument(
        "--k0-current-target-placement",
        choices=("face", "cell"),
        default=None,
        help=(
            "Where to form the H_k0 nonlinear current target. Default: cell."
        ),
    )
    parser.add_argument(
        "--bottom-parallel-current-mode",
        choices=("observed_full", "face_parallel_exact_jz"),
        default="observed_full",
        help=(
            "Bottom source-row construction for staggered runs: overwrite with "
            "full observed current, or keep face-parallel Jx/Jy and exact Jz."
        ),
    )
    parser.add_argument("--d0-zero-mode-policy", default="project")
    parser.add_argument("--source-zero-mode-policy", default="project")
    parser.add_argument(
        "--poloidal-update-mode",
        choices=("direct", "two_stage"),
        default="direct",
    )
    parser.add_argument("--record-iteration-diagnostics", action="store_true")
    parser.add_argument("--harmonic-residual-policy", default="diagnose")
    parser.add_argument(
        "--gaussian-horizontal-operator",
        choices=("fd", "spectral"),
        default="spectral",
        help="Gaussian-separation horizontal div/grad: fd or spectral (default).",
    )
    parser.add_argument(
        "--vertical-current-operator",
        choices=("staggered_face", "collocated_centered"),
        default="staggered_face",
        help="Production face-current operator or legacy collocated ablation.",
    )
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if per-weight JSON / field caches already exist.",
    )
    parser.add_argument(
        "--no-field-cache",
        action="store_true",
        help="Do not write NFGS / reference field caches under output-dir/fields/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = parse_weights(args.weights)
    requested = [item.strip().lower() for item in args.benchmarks.split(",") if item.strip()]
    device = analytic.resolve_device(args.device)
    dtype = analytic.dtype_from_name(args.dtype)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    cache_fields = not bool(args.no_field_cache)
    solver_sha256 = sha256_file(SOLVER_SOURCE)
    checkpoint_iters = sorted(
        {
            int(item.strip())
            for item in str(args.checkpoint_iters).split(",")
            if item.strip()
        }
    )
    if any(item < 1 for item in checkpoint_iters):
        raise SystemExit("--checkpoint-iters entries must be >= 1")
    run_config = {
        "resolved_device": device,
        "dtype": args.dtype,
        "init_mode": args.init_mode,
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "patience": int(args.patience),
        "sigma_improvement_tol": float(args.sigma_improvement_tol),
        "mu": float(args.mu),
        "k0_update_mu": (
            None if args.k0_update_mu is None else float(args.k0_update_mu)
        ),
        "k0_spline_basis_size": int(args.k0_spline_basis_size),
        "td_calculation_domain_mode": args.td_calculation_domain_mode,
        "horizontal_zero_mode_policy": args.horizontal_zero_mode_policy,
        "k0_ode_smoothness": float(args.k0_ode_smoothness),
        "coupled_k0_discretization": args.coupled_k0_discretization,
        "k0_current_target_placement": args.k0_current_target_placement,
        "bottom_parallel_current_mode": args.bottom_parallel_current_mode,
        "gaussian_horizontal_operator": args.gaussian_horizontal_operator,
        "vertical_current_operator": args.vertical_current_operator,
        "poloidal_update_mode": args.poloidal_update_mode,
        "record_iteration_diagnostics": bool(args.record_iteration_diagnostics),
        "stop_mode": args.stop_mode,
        "fixed_point_update_tol": args.fixed_point_update_tol,
        "fixed_point_pde_tol": float(args.fixed_point_pde_tol),
        "fixed_point_observable_tol": args.fixed_point_observable_tol,
        "fixed_point_patience": int(args.fixed_point_patience),
        "alpha_clip_scale": float(args.alpha_clip_scale),
        "weak_field_floor_b2": float(args.weak_field_floor_b2),
        "mean_bz_policy": args.mean_bz_policy,
        "d0_zero_mode_policy": args.d0_zero_mode_policy,
        "source_zero_mode_policy": args.source_zero_mode_policy,
        "harmonic_residual_policy": args.harmonic_residual_policy,
        "mean_mode_tolerance": float(args.mean_mode_tolerance),
        "source_zero_mode_tolerance": float(args.source_zero_mode_tolerance),
        "handoff_projection_policy": args.handoff_projection_policy,
        "handoff_projection_tolerance": float(
            args.handoff_projection_tolerance
        ),
        "checkpoint_iters": checkpoint_iters,
    }
    if cache_fields:
        field_cache_dir(out_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_root = out_dir / "checkpoints"
    if checkpoint_iters:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    history_root = out_dir / "histories"
    if args.record_iteration_diagnostics:
        history_root.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} dtype={dtype} init={args.init_mode} "
        f"max_iter={args.max_iter} weights={weights} benchmarks={requested}",
        flush=True,
    )
    print("Building analytic reference cases (no topology stage).", flush=True)
    cases = driver.make_requested_cases(args, requested)
    if cache_fields:
        cache_args = SimpleNamespace(field_cache_dir=field_cache_dir(out_dir))
        for name in requested:
            driver.save_td_case_cache(cache_args, cases[name])

    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for wi, weight in enumerate(weights, 1):
        weight_path = weight_json_path(out_dir, weight)
        if weight_path.exists() and not args.force:
            # Still require field caches if caching is on; otherwise re-run.
            fields_ok = True
            if cache_fields:
                fields_ok = all(
                    nfgs_field_cache_path(
                        out_dir, name, args.init_mode, weight
                    ).is_file()
                    for name in requested
                )
            if fields_ok:
                records = json.loads(weight_path.read_text())
                if records_match_current_solver(
                    records,
                    solver_sha256=solver_sha256,
                    horizontal_zero_mode_policy=args.horizontal_zero_mode_policy,
                    expected_run_config=run_config,
                ):
                    print(f"[{wi}/{len(weights)}] reuse {weight_path.name}", flush=True)
                    all_rows.extend(records)
                    continue
                print(
                    f"[{wi}/{len(weights)}] stale solver/k0 provenance; "
                    f"re-running robin_weight={weight:.1f}",
                    flush=True,
                )
            else:
                print(
                    f"[{wi}/{len(weights)}] metrics exist but field cache missing; "
                    f"re-running robin_weight={weight:.1f}",
                    flush=True,
                )

        print(f"[{wi}/{len(weights)}] robin_weight={weight:.1f}", flush=True)
        run_args = make_run_args(weight, args)
        records: list[dict[str, Any]] = []
        for name in requested:
            case = cases[name]
            cache_path = nfgs_field_cache_path(out_dir, name, args.init_mode, weight)
            # A field NPZ has no self-contained solver/config fingerprint.  If
            # the paired JSON record is absent or stale, recompute instead of
            # silently attaching current metadata to a pre-fix field cache.
            print(f"  Running {case['display']} ...", flush=True)

            def _make_checkpoint_binder(
                benchmark: str = name, robin: float = weight
            ):
                wanted = set(checkpoint_iters)

                def _bind(solver_obj):
                    def _callback(n: int, sigma: float, B, J) -> None:
                        if int(n) not in wanted:
                            return
                        tag = weight_tag(robin)
                        stem = f"{benchmark}__iter{int(n):04d}__robin_w{tag}"
                        field_path = checkpoint_root / f"{stem}.npz"
                        meta_path = checkpoint_root / f"{stem}.json"
                        h_k0 = getattr(solver_obj, "_H_k0", None)
                        phi = getattr(solver_obj, "_Phi_c", None)
                        az = getattr(solver_obj, "_Az", None)
                        payload_arrays = {
                            "Bx": B[0].detach().cpu().numpy(),
                            "By": B[1].detach().cpu().numpy(),
                            "Bz": B[2].detach().cpu().numpy(),
                            "Jx": J[0].detach().cpu().numpy(),
                            "Jy": J[1].detach().cpu().numpy(),
                            "Jz": J[2].detach().cpu().numpy(),
                        }
                        if h_k0 is not None:
                            payload_arrays["H_k0"] = h_k0.detach().cpu().numpy()
                        if phi is not None:
                            payload_arrays["Phi_c"] = phi.detach().cpu().numpy()
                        if az is not None:
                            payload_arrays["Az"] = az.detach().cpu().numpy()
                        np.savez_compressed(field_path, **payload_arrays)
                        k0_diag = dict(getattr(solver_obj, "_k0_diagnostics", {}) or {})
                        meta = {
                            "benchmark": benchmark,
                            "robin_weight": float(robin),
                            "iteration": int(n),
                            "sigma": float(sigma),
                            "mean_profile_max_abs": float(
                                abs(payload_arrays["H_k0"]).max()
                            )
                            if "H_k0" in payload_arrays
                            else None,
                            "coupled_k0": {
                                key: k0_diag[key]
                                for key in (
                                    "discretization",
                                    "ode_residual_relative",
                                    "face_ode_residual_relative",
                                    "profile_max_abs",
                                    "desired_mean_current_rms",
                                )
                                if key in k0_diag
                            },
                            "run_config": run_config,
                        }
                        meta_path.write_text(
                            json.dumps(analytic.json_safe(meta), indent=2) + "\n"
                        )
                        print(
                            f"    checkpoint {field_path.name} sigma={sigma:.6e}",
                            flush=True,
                        )

                    return _callback

                return _bind

            if checkpoint_iters:
                run_args.bind_solver_callback = _make_checkpoint_binder()
            else:
                run_args.bind_solver_callback = None
            run_args.per_step_callback = None

            record, fields = driver.run_case(
                case,
                device=device,
                dtype=dtype,
                max_iter=int(args.max_iter),
                args=run_args,
            )
            record["sigma_J_common_fd2"] = common_fd2_sigma_j(
                fields, float(case["dims"][3]), case["metric_slices"]
            )
            if args.record_iteration_diagnostics:
                write_history_csv(
                    history_root
                    / f"{name}__{args.dtype}__robin_w{weight_tag(weight)}.csv",
                    record,
                )
            if cache_fields:
                analytic.save_field_cache(cache_path, fields)
                print(f"    saved {cache_path.name}", flush=True)
            slim = {
                "robin_weight": float(weight),
                "benchmark": record["benchmark"],
                "benchmark_display": record["benchmark_display"],
                "dtype": record["dtype"],
                "resolved_device": record["resolved_device"],
                "device_name": record["device_name"],
                "init_mode": record["init_mode"],
                "metric_domain": record["metric_domain"],
                "horizontal_zero_mode_policy": record[
                    "horizontal_zero_mode_policy"
                ],
                "k0_ode_smoothness": float(args.k0_ode_smoothness),
                "coupled_k0_discretization": record.get(
                    "coupled_k0_discretization", args.coupled_k0_discretization
                ),
                "vertical_current_operator": record["vertical_current_operator"],
                "mu": float(record.get("mu", args.mu)),
                "k0_update_mu": record.get("k0_update_mu", args.k0_update_mu),
                "k0_spline_basis_size": int(args.k0_spline_basis_size),
                "solver_sha256": solver_sha256,
                "run_config": run_config,
                "source_diagnostics": record["source_diagnostics"],
                "best_iter": record["best_iter"],
                "n_iters": record["n_iters"],
                "best_sigma": record["best_sigma"],
                "stop_reason": record["stop_reason"],
                "stop_mode": record.get("stop_mode", args.stop_mode),
                "selected_state": record.get("selected_state")
                or record.get("history", {}).get("selected_state"),
                "fixed_point_converged": record.get(
                    "fixed_point_converged", False
                ),
                "wall_time_s": record["wall_time_s"],
                "iterations_per_s": record.get("iterations_per_s"),
                "peak_memory_bytes": record.get("peak_memory_bytes"),
                "history": record["history"],
                "sigma_J_common_fd2": record["sigma_J_common_fd2"],
                **{key: record[key] for key in METRIC_KEYS},
            }
            records.append(slim)
            print(
                f"    C_vec={slim['C_vec']:.6f} Eprime_n={slim['Eprime_n']:.6f} "
                f"sigma_J={slim['sigma_J']:.6f} n_iters={slim['n_iters']} "
                f"({slim['wall_time_s']:.1f}s)",
                flush=True,
            )
            del fields
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        weight_path.write_text(json.dumps(analytic.json_safe(records), indent=2))
        all_rows.extend(records)
        progress_path.write_text(
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
        write_csv(out_dir / "metrics_all.csv", all_rows)

    summary = summarize(all_rows)
    summary["solver_provenance"] = {
        "source": str(SOLVER_SOURCE.resolve()),
        "sha256": solver_sha256,
        "coupled_k0_discretization": args.coupled_k0_discretization,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(analytic.json_safe(summary), indent=2, allow_nan=True)
    )
    write_csv(out_dir / "metrics_all.csv", all_rows)

    best = summary["best_by_mean_C_vec"]
    best_td3 = summary["best_by_td3_C_vec"]
    print("=" * 60, flush=True)
    print(
        f"BEST mean C_vec: robin_weight={best['robin_weight']:.1f} "
        f"mean_C_vec={best['mean_C_vec']:.6f} mean_Eprime_n={best['mean_Eprime_n']:.6f}",
        flush=True,
    )
    if best_td3 is not None:
        print(
            f"BEST TD3 C_vec: robin_weight={best_td3['robin_weight']:.1f} "
            f"td3_C_vec={best_td3['td3_C_vec']:.6f}",
            flush=True,
        )
    print(f"metrics={out_dir / 'metrics_all.csv'}", flush=True)
    print(f"summary={out_dir / 'summary.json'}", flush=True)
    print(f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
