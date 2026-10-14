#!/usr/bin/env python3
"""Run the same-code factorized PT-to-NFGS bridge on TD1--TD3.

F0, the untouched original Fortran NFPT, is deliberately external to this
runner and remains a descriptive anchor.  Adjacent F1--F7 rows change one
declared numerical factor, so only those adjacent contrasts are eligible for
causal attribution.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.experiments.nfgs_paper.analytic_nlfff.benchmarks import (  # noqa: E402
    save_field_cache,
    write_history_csv,
)
from nfgs._backend.scripts import run_analytic_validation as analytic  # noqa: E402
from nfgs._backend.scripts import run_nfgs_analytic_validation as driver  # noqa: E402
from nfgs.diagnostics.common_fd2 import sigma_j as common_fd2_sigma_j  # noqa: E402


@dataclass(frozen=True)
class FactorPoint:
    variant_id: str
    changed_factor: str
    source_partition_mode: str
    init_mode: str
    poloidal_update_mode: str
    rho: float
    horizontal_zero_mode_policy: str
    vertical_current_operator: str


FACTOR_CHAIN = (
    FactorPoint(
        "F1", "same-code single-field PT bridge", "legacy_single_field",
        "cfcf", "two_stage", 0.0, "omit", "collocated_centered",
    ),
    FactorPoint(
        "F2", "direct scalar solve", "legacy_single_field",
        "cfcf", "direct", 0.0, "omit", "collocated_centered",
    ),
    FactorPoint(
        "F3", "Gaussian source partition", "gaussian_separated",
        "cfcf", "direct", 0.0, "omit", "collocated_centered",
    ),
    FactorPoint(
        "F4", "GS-CFCF initializer", "gaussian_separated",
        "gs_cfcf", "direct", 0.0, "omit", "collocated_centered",
    ),
    FactorPoint(
        "F5", "mixed first-interior row", "gaussian_separated",
        "gs_cfcf", "direct", 0.4, "omit", "collocated_centered",
    ),
    FactorPoint(
        "F6", "frozen harmonic closure", "gaussian_separated",
        "gs_cfcf", "direct", 0.4, "frozen_linear_harmonic",
        "collocated_centered",
    ),
    FactorPoint(
        "F7", "vertical-face current", "gaussian_separated",
        "gs_cfcf", "direct", 0.4, "frozen_linear_harmonic", "staggered_face",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "nfgs_paper" / "referee_revision" / "factorized_ablation",
    )
    parser.add_argument("--benchmarks", default="td1,td2,td3")
    parser.add_argument("--variants", default=",".join(p.variant_id for p in FACTOR_CHAIN))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--sigma-improvement-tol", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-clip-scale", type=float, default=1.0)
    parser.add_argument("--weak-field-floor-b2", type=float, default=1.0e-20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-field-cache", action="store_true")
    parser.add_argument("--list-matrix", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.benchmarks = tuple(dict.fromkeys(x.strip().lower() for x in args.benchmarks.split(",") if x.strip()))
    args.variants = tuple(dict.fromkeys(x.strip().upper() for x in args.variants.split(",") if x.strip()))
    if not args.benchmarks or set(args.benchmarks) - {"td1", "td2", "td3"}:
        parser.error("--benchmarks must be a nonempty subset of td1,td2,td3")
    known = {point.variant_id for point in FACTOR_CHAIN}
    if not args.variants or set(args.variants) - known:
        parser.error(f"--variants must be a nonempty subset of {sorted(known)}")
    if (
        args.max_iter < 1
        or args.tol <= 0.0
        or args.patience < 1
        or args.sigma_improvement_tol < 0.0
    ):
        parser.error("invalid heuristic stopping controls")
    return args


def json_safe(value: Any) -> Any:
    return analytic.json_safe(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def matrix(args: argparse.Namespace) -> list[FactorPoint]:
    selected = set(args.variants)
    return [point for point in FACTOR_CHAIN if point.variant_id in selected]


def run_args(point: FactorPoint, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        mean_bz_policy="uniform_open",
        horizontal_zero_mode_policy=point.horizontal_zero_mode_policy,
        k0_ode_smoothness=1.0e-3,
        coupled_k0_discretization="staggered_face_D1_true_three_point_curvature",
        k0_update_mu=None,
        k0_spline_basis_size=8,
        k0_current_target_placement=None,
        bottom_parallel_current_mode="observed_full",
        d0_zero_mode_policy="project",
        source_zero_mode_policy="project",
        harmonic_residual_policy="diagnose",
        mean_mode_tolerance=1.0e-6,
        source_zero_mode_tolerance=1.0e-2,
        bottom_divergence_projection_weight=0.0,
        bottom_divergence_robin_weight=point.rho,
        poloidal_update_mode=point.poloidal_update_mode,
        record_iteration_diagnostics=True,
        handoff_projection_policy="project",
        handoff_projection_tolerance=1.0e-6,
        gaussian_horizontal_operator="spectral",
        vertical_current_operator=point.vertical_current_operator,
        source_partition_mode=point.source_partition_mode,
        alpha_clip_scale=args.alpha_clip_scale,
        weak_field_floor_b2=args.weak_field_floor_b2,
        stop_mode="sigma_best",
        tol=args.tol,
        patience=args.patience,
        sigma_improvement_tol=args.sigma_improvement_tol,
        mu=1.0,
        verbose=args.verbose,
        init_mode=point.init_mode,
    )


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "run_id", "config_hash", "variant_id", "changed_factor", "benchmark",
        "source_partition_mode", "init_mode", "poloidal_update_mode", "rho",
        "horizontal_zero_mode_policy", "vertical_current_operator", "dtype",
        "stop_mode", "stop_reason", "selected_state", "best_iter", "n_iters",
        "best_sigma", "wall_time_s", "iterations_per_s", "peak_memory_bytes",
        "C_vec", "C_CS", "Eprime_n", "Eprime_m", "epsilon", "sigma_J",
        "sigma_J_common_fd2",
        "cache_status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    points = matrix(args)
    if args.list_matrix:
        print(json.dumps([asdict(point) for point in points], indent=2))
        return 0

    device = analytic.resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    code_paths = {
        "runner": Path(__file__).resolve(),
        "solver": ROOT / "nfgs_solver.py",
        "base_solver": ROOT / "nfgs_solver.py",
        "driver": ROOT / "nfgs" / "_backend" / "scripts" / "run_nfgs_analytic_validation.py",
    }
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    out = args.output_dir.resolve()
    records_dir, histories_dir, fields_dir = out / "records", out / "histories", out / "fields"
    for path in (records_dir, histories_dir):
        path.mkdir(parents=True, exist_ok=True)
    if not args.no_field_cache:
        fields_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        name: driver.make_td_case(name, calculation_domain_mode="mudpack", chunk_depth=4, n_test_x=66, n_test_y=66, n_test_z=30)
        for name in args.benchmarks
    }
    total = len(points) * len(cases)
    progress = {"status": "running", "total_runs": total, "completed_runs": 0, "completed_run_ids": []}
    write_json(out / "progress.json", progress)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    index = 0
    try:
        for point in points:
            for benchmark, case in cases.items():
                index += 1
                config = {
                    "schema_version": 1,
                    "variant": asdict(point),
                    "benchmark": benchmark,
                    "dtype": args.dtype,
                    "device": device,
                    "max_iter": args.max_iter,
                    "stopping": {
                        "mode": "sigma_best",
                        "sigma_tol": args.tol,
                        "patience": args.patience,
                        "sigma_improvement_tol": args.sigma_improvement_tol,
                        "selected_state": "best_sigma",
                    },
                    "alpha_clip_scale": args.alpha_clip_scale,
                    "weak_field_floor_b2": args.weak_field_floor_b2,
                    "calculation_domain_mode": "mudpack",
                    "seed": args.seed,
                    "code_sha256": code_hashes,
                }
                config_hash = canonical_hash(config)
                run_id = f"{point.variant_id.lower()}__{benchmark}__{args.dtype}__{config_hash[:12]}"
                record_path = records_dir / f"{run_id}.json"
                history_path = histories_dir / f"{run_id}.csv"
                field_path = fields_dir / f"{run_id}.npz"
                cache_ok = record_path.is_file() and not args.force and (args.no_field_cache or field_path.is_file())
                if cache_ok:
                    payload = json.loads(record_path.read_text())
                    cache_ok = payload.get("config_hash") == config_hash
                if cache_ok:
                    record = payload["result"]
                    cache_status = "reused_exact_config"
                else:
                    random.seed(args.seed)
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    if device.startswith("cuda"):
                        torch.cuda.manual_seed_all(args.seed)
                    record, fields = driver.run_case(
                        case, device=device, dtype=dtype, max_iter=args.max_iter,
                        args=run_args(point, args),
                    )
                    record["sigma_J_common_fd2"] = common_fd2_sigma_j(
                        fields, float(case["dims"][3]), case["metric_slices"]
                    )
                    payload = {"schema_version": 1, "run_id": run_id, "config_hash": config_hash, "config": config, "result": record}
                    write_json(record_path, payload)
                    if not args.no_field_cache:
                        save_field_cache(field_path, fields)
                    del fields
                    cache_status = "computed"
                write_history_csv(history_path, record)
                row = {
                    "run_id": run_id, "config_hash": config_hash,
                    "variant_id": point.variant_id, "changed_factor": point.changed_factor,
                    "benchmark": benchmark, "rho": point.rho, "dtype": args.dtype,
                    "cache_status": cache_status, **record,
                }
                rows.append(row)
                write_metrics(out / "metrics_all.csv", rows)
                runs.append({"run_id": run_id, "record": str(record_path), "history": str(history_path), "field": None if args.no_field_cache else str(field_path), "cache_status": cache_status})
                progress.update({"completed_runs": index, "completed_run_ids": [item["run_id"] for item in runs], "current_run_id": None})
                write_json(out / "progress.json", progress)
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    except BaseException as exc:
        progress.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_json(out / "progress.json", progress)
        raise

    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "finished_utc": finished,
        "interpretation": (
            "Only adjacent F1-F7 contrasts are one-factor causal comparisons; "
            "F0 is an external untouched-Fortran anchor. Nonlinear rows use the "
            "registered heuristic sigma monitor and restore the lowest-sigma state."
        ),
        "factor_chain": [asdict(point) for point in points],
        "code_sha256": code_hashes,
        "runs": runs,
    }
    write_json(out / "manifest.json", manifest)
    progress.update({"status": "complete", "finished_utc": finished})
    write_json(out / "progress.json", progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
