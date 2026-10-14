#!/usr/bin/env python3
"""Run an NFPT Fortran executable on TD1--TD3 bottoms and save metrics.

Writes field caches, per-benchmark JSON, and a combined CSV under the robin
weight sweep directory so topology / metric figures can include an NFPT column.
The selected executable and archived source are hashed in every record.  A
release executable with altered output/MUDPACK switches must not be described
as the unmodified archived source merely because it implements NFPT.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.scripts import run_analytic_validation as analytic  # noqa: E402
from nfgs._backend.scripts import run_nfgs_analytic_validation as driver  # noqa: E402
from nfgs._backend.scripts.run_nfgs_td_robin_weight_sweep import (  # noqa: E402
    field_cache_dir,
    load_field_cache,
)
from nfgs._backend.util.metrics import compute_nf2_metrics  # noqa: E402
from nfgs._backend.util.td2_io import load_b_czyx  # noqa: E402

DEFAULT_SWEEP_DIR = ROOT / "results" / "nfgs_td_robin_weight_sweep"
# Prefer the directly compiled archived source.  A release build is retained
# only as a compatibility fallback and must not be labeled untouched-original.
_ORIGINAL_BUILD_EXE = (
    ROOT
    / "results"
    / "nfpt_original_archived_mudpack_legacy"
    / "build"
    / "nfpt_original_gfortran"
)
_RELEASE_EXE = (
    ROOT
    / ".worktrees"
    / "xflare-harp-preonset"
    / ".cache"
    / "nfpt_fortran_release"
    / "nfpt_release_gfortran"
)
_DEBUG_EXE = (
    ROOT
    / "archive"
    / "review_before_deletion"
    / "legacy_solvers"
    / "nfpt_fortran"
    / "nfpt"
    / "nfpt_debug_gfortran"
)
if _ORIGINAL_BUILD_EXE.is_file():
    DEFAULT_FORTRAN_EXE = _ORIGINAL_BUILD_EXE
elif _DEBUG_EXE.is_file():
    DEFAULT_FORTRAN_EXE = _DEBUG_EXE
else:
    DEFAULT_FORTRAN_EXE = _RELEASE_EXE
DEFAULT_CASE_CACHE = ROOT / "results" / "nfgs_analytic_validation_td" / "fields"
DEFAULT_FORTRAN_SOURCE = (
    ROOT
    / "archive"
    / "review_before_deletion"
    / "legacy_solvers"
    / "nfpt_fortran"
    / "nfpt"
    / "nfpt_1.0.2.f90"
)
METRIC_KEYS = list(analytic.METRIC_KEYS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--fortran-exe", type=Path, default=DEFAULT_FORTRAN_EXE)
    p.add_argument("--fortran-source", type=Path, default=DEFAULT_FORTRAN_SOURCE)
    p.add_argument(
        "--debug-output-sink-library",
        type=Path,
        default=None,
        help=(
            "Optional LD_PRELOAD library that redirects only output/debug_* "
            "volume files. The archived Fortran source and numerical calls "
            "remain unchanged; the library and its hash are recorded."
        ),
    )
    p.add_argument("--fortran-run-root", type=Path, default=None)
    p.add_argument("--case-cache-dir", type=Path, default=DEFAULT_CASE_CACHE)
    p.add_argument("--benchmarks", default="td1,td2,td3")
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--td-calculation-domain-mode",
        choices=("mudpack", "mudpack_legacy", "bottom_padding"),
        default="mudpack_legacy",
        help=(
            "Case-cache/grid-placement contract recorded with the untouched "
            "Fortran run (default: native mudpack_legacy placement)."
        ),
    )
    p.add_argument("--td-topology-chunk-depth", type=int, default=4)
    p.add_argument("--n-test-x", type=int, default=66)
    p.add_argument("--n-test-y", type=int, default=66)
    p.add_argument("--n-test-z", type=int, default=30)
    p.add_argument("--source-root", type=Path, default=ROOT)
    return p.parse_args()


def fortran_field_cache_path(out_dir: Path, benchmark: str) -> Path:
    return field_cache_dir(out_dir) / f"{benchmark}__nfpt_fortran.npz"


def case_tag(benchmark: str, dims: tuple[Any, ...], max_iter: int) -> str:
    nx, ny, nz, *_ = dims
    return f"{benchmark}_Nx{int(nx)}_Ny{int(ny)}_Nz{int(nz)}_iter{int(max_iter)}"


def ensure_run_tree(run_dir: Path) -> None:
    for rel in (
        "input/codex",
        "output",
        "ext/b",
        "ext/j",
        "ext/az",
        "ext/phi",
        "anal/logs",
    ):
        (run_dir / rel).mkdir(parents=True, exist_ok=True)


def write_bottom_input(path: Path, case: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for name in ("Bx", "By", "Bz"):
            arr = np.ascontiguousarray(case["boundary"][name], dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"{name} must be 2D, got {arr.shape}")
            arr.tofile(stream)


def write_par(path: Path, *, input_ref: str, dims: tuple[Any, ...], max_iter: int) -> None:
    nx, ny, nz, *_ = dims
    path.write_text(
        "\n".join(
            [
                f"'{input_ref}' # input file",
                f"{int(nx)}   # x",
                f"{int(ny)}   # y",
                f"{int(nz)}   # z",
                f"{int(max_iter)}   # max iter",
                "2     # input data dim",
                "",
            ]
        )
    )


def run_fortran_cmd(
    cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None
) -> float:
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Fortran command failed (exit={proc.returncode}): {' '.join(cmd)}; "
            f"see {log_path}"
        )
    return float(wall)


def parse_elapsed(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    pattern = re.compile(r"Elapsed Time\s*:\s*([0-9.Ee+-]+)\s+seconds")
    last = None
    for line in log_path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            last = float(match.group(1))
    return last


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(analytic.json_safe(value), indent=2) + "\n")
    os.replace(temporary, path)


def parse_calculation_grid(log_path: Path) -> list[int] | None:
    if not log_path.is_file():
        return None
    pattern = re.compile(r"^\s*n1,n2,n3\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
    for line in log_path.read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            return [int(match.group(i)) for i in range(1, 4)]
    return None


def read_history(run_dir: Path) -> dict[str, list[Any]]:
    history: dict[str, list[Any]] = {}
    path = run_dir / "anal" / "cwsin"
    if not path.is_file() or path.stat().st_size == 0:
        return history
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.size == 0 or arr.shape[1] < 2:
        return history
    history["iter"] = [int(v) for v in arr[:, 0]]
    history["sigma"] = [float(v) for v in arr[:, 1]]
    return history


def load_cases(args: argparse.Namespace, requested: list[str]) -> dict[str, dict[str, Any]]:
    proxy = SimpleNamespace(
        field_cache_dir=args.case_cache_dir,
        td_calculation_domain_mode=args.td_calculation_domain_mode,
        td_topology_chunk_depth=args.td_topology_chunk_depth,
        n_test_x=args.n_test_x,
        n_test_y=args.n_test_y,
        n_test_z=args.n_test_z,
        source_root=args.source_root,
        lnl_calculation_domain_mode="mudpack",
    )
    cases: dict[str, dict[str, Any]] = {}
    for name in requested:
        case = driver.load_td_case_cache(proxy, name)
        if case is None:
            print(f"  building analytic case {name} ...", flush=True)
            case = driver.make_td_case(
                name,
                calculation_domain_mode=args.td_calculation_domain_mode,
                chunk_depth=args.td_topology_chunk_depth,
                n_test_x=args.n_test_x,
                n_test_y=args.n_test_y,
                n_test_z=args.n_test_z,
            )
            args.case_cache_dir.mkdir(parents=True, exist_ok=True)
            driver.save_td_case_cache(proxy, case)
        else:
            print(f"  loaded case cache {name}", flush=True)
        # load_td_case_cache may omit boundary; rebuild from reference bottom.
        if "boundary" not in case:
            ref = case["reference"]
            case["boundary"] = {
                "Bx": np.asarray(ref["Bx"][0], dtype=np.float64),
                "By": np.asarray(ref["By"][0], dtype=np.float64),
                "Bz": np.asarray(ref["Bz"][0], dtype=np.float64),
            }
        cases[name] = case
    return cases


def metrics_record(
    *,
    benchmark: str,
    case: dict[str, Any],
    fields: dict[str, np.ndarray],
    history: dict[str, list[Any]],
    wall_time_s: float,
    stop_reason: str,
    run_dir: Path,
    b_path: Path,
    j_path: Path,
) -> dict[str, Any]:
    metrics = compute_nf2_metrics(case["reference"], fields, case["metric_slices"])
    iters = history.get("iter", [])
    sigmas = history.get("sigma", [])
    best_iter = int(iters[int(np.nanargmin(sigmas))]) if sigmas else 0
    best_sigma = float(np.nanmin(sigmas)) if sigmas else float("nan")
    return {
        "benchmark": benchmark,
        "benchmark_display": case.get("display", benchmark.upper()),
        "solver": "nfpt_fortran",
        "init_mode": "fortran_run",
        "metric_domain": case["metric_domain"],
        "best_iter": best_iter,
        "n_iters": int(len(iters)),
        "best_sigma": best_sigma,
        "stop_reason": stop_reason,
        "wall_time_s": float(wall_time_s),
        **{key: float(metrics[key]) for key in METRIC_KEYS},
        "history": {
            "iter": iters,
            "sigma": sigmas,
        },
        "run_dir": str(run_dir),
        "source_b": str(b_path),
        "source_j": str(j_path),
        "grid_xyz": [int(case["dims"][0]), int(case["dims"][1]), int(case["dims"][2])],
    }


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        "benchmark",
        "solver",
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
        "wall_time_s",
        "init_mode",
        "metric_domain",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def run_one(
    *,
    case: dict[str, Any],
    benchmark: str,
    args: argparse.Namespace,
    run_root: Path,
    exe: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tag = case_tag(benchmark, case["dims"], args.max_iter)
    run_dir = run_root / tag
    input_path = run_dir / "input" / "codex" / tag
    b_path = run_dir / "ext" / "b" / f"b_{tag}"
    j_path = run_dir / "ext" / "j" / f"j_{tag}"
    cache_path = fortran_field_cache_path(args.sweep_dir, benchmark)
    metrics_path = args.sweep_dir / f"nfpt_fortran_{benchmark}.json"
    exe_sha256 = sha256_file(exe)
    source_path = args.fortran_source.resolve()
    source_sha256 = sha256_file(source_path) if source_path.is_file() else None
    sink_path = (
        args.debug_output_sink_library.resolve()
        if args.debug_output_sink_library is not None
        else None
    )
    sink_sha256 = sha256_file(sink_path) if sink_path is not None else None

    if (
        not args.force
        and cache_path.is_file()
        and metrics_path.is_file()
        and b_path.is_file()
        and j_path.is_file()
    ):
        record = json.loads(metrics_path.read_text())
        if (
            record.get("executable_sha256") == exe_sha256
            and record.get("source_sha256") == source_sha256
            and record.get("debug_output_sink_sha256") == sink_sha256
            and record.get("calculation_domain_mode")
            == args.td_calculation_domain_mode
        ):
            # Scientific arrays remain immutable, but descriptive domain
            # metadata may be clarified after source review.  Refresh those
            # fields from the current case contract instead of perpetuating a
            # stale cached label (for example, the TD2 Demcsak attribution).
            refreshed_metadata = {
                "metric_domain": case["metric_domain"],
                "grid_xyz": [
                    int(case["dims"][0]),
                    int(case["dims"][1]),
                    int(case["dims"][2]),
                ],
            }
            if any(record.get(key) != value for key, value in refreshed_metadata.items()):
                record.update(refreshed_metadata)
                metrics_path.write_text(
                    json.dumps(analytic.json_safe(record), indent=2) + "\n"
                )
            print(f"[cache] {benchmark} <- {cache_path.name}", flush=True)
            fields = load_field_cache(cache_path)
            return record, fields
        print(f"[stale cache] {benchmark}: provenance mismatch; rerunning", flush=True)

    ensure_run_tree(run_dir)
    write_bottom_input(input_path, case)
    write_par(
        run_dir / "par.in",
        input_ref=f"input/codex/{tag}",
        dims=case["dims"],
        max_iter=args.max_iter,
    )

    print(
        f"[run] {benchmark} | nfpt_fortran | max_iter={args.max_iter} cwd={run_dir}",
        flush=True,
    )
    run_env = None
    if sink_path is not None:
        run_env = os.environ.copy()
        existing_preload = run_env.get("LD_PRELOAD")
        run_env["LD_PRELOAD"] = (
            f"{sink_path}:{existing_preload}" if existing_preload else str(sink_path)
        )
    wall = run_fortran_cmd(
        [str(exe.resolve()), "1", "codex", tag],
        cwd=run_dir,
        log_path=run_dir / "fortran_solve.log",
        env=run_env,
    )
    wall += run_fortran_cmd(
        [str(exe.resolve()), "11", "codex", tag],
        cwd=run_dir,
        log_path=run_dir / "fortran_extract.log",
        env=run_env,
    )
    # Prefer Fortran-reported wall-clock when present (solve + extract).
    reported = [
        v
        for v in (
            parse_elapsed(run_dir / "fortran_solve.log"),
            parse_elapsed(run_dir / "fortran_extract.log"),
        )
        if v is not None
    ]
    if reported:
        wall = float(sum(reported))

    if not b_path.is_file() or not j_path.is_file():
        raise FileNotFoundError(f"Missing Fortran outputs: {b_path}, {j_path}")

    nx, ny, nz, *_ = case["dims"]
    bx, by, bz = load_b_czyx(str(b_path), int(nx), int(ny), int(nz))
    jx, jy, jz = load_b_czyx(str(j_path), int(nx), int(ny), int(nz))
    fields = {"Bx": bx, "By": by, "Bz": bz, "Jx": jx, "Jy": jy, "Jz": jz}
    history = read_history(run_dir)
    record = metrics_record(
        benchmark=benchmark,
        case=case,
        fields=fields,
        history=history,
        wall_time_s=wall,
        stop_reason="fortran_completed",
        run_dir=run_dir,
        b_path=b_path,
        j_path=j_path,
    )
    record.update(
        {
            "fortran_source": str(source_path),
            "source_sha256": source_sha256,
            "fortran_executable": str(exe.resolve()),
            "executable_sha256": exe_sha256,
            "debug_output_sink_library": (
                str(sink_path) if sink_path is not None else None
            ),
            "debug_output_sink_sha256": sink_sha256,
            "debug_output_sink_scope": (
                "output/debug_* file opens only" if sink_path is not None else None
            ),
            "calculation_domain_mode": args.td_calculation_domain_mode,
            "calculation_grid_xyz": parse_calculation_grid(
                run_dir / "fortran_solve.log"
            ),
        }
    )

    field_cache_dir(args.sweep_dir).mkdir(parents=True, exist_ok=True)
    analytic.save_field_cache(cache_path, fields)
    metrics_path.write_text(json.dumps(analytic.json_safe(record), indent=2) + "\n")
    print(
        f"  {benchmark}: C_vec={record['C_vec']:.6f} sigma_J={record['sigma_J']:.6f} "
        f"n_iters={record['n_iters']} wall={record['wall_time_s']:.1f}s",
        flush=True,
    )
    return record, fields


def main() -> int:
    args = parse_args()
    requested = [
        item.strip().lower()
        for item in args.benchmarks.split(",")
        if item.strip()
    ]
    exe = args.fortran_exe
    if not exe.is_file():
        raise FileNotFoundError(f"Fortran executable not found: {exe}")
    if not os.access(exe, os.X_OK):
        exe.chmod(exe.stat().st_mode | 0o100)
    if args.debug_output_sink_library is not None:
        args.debug_output_sink_library = args.debug_output_sink_library.resolve()
        if not args.debug_output_sink_library.is_file():
            raise FileNotFoundError(
                "Debug-output sink library not found: "
                f"{args.debug_output_sink_library}"
            )

    run_root = args.fortran_run_root or (args.sweep_dir / "fortran_runs")
    run_root.mkdir(parents=True, exist_ok=True)
    args.sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"benchmarks={requested} exe={exe}", flush=True)
    cases = load_cases(args, requested)

    progress_path = args.sweep_dir / "progress.json"
    progress: dict[str, Any] = {
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": requested,
        "total_runs": len(requested),
        "completed_runs": 0,
        "completed_benchmarks": [],
        "current_benchmark": None,
        "fortran_executable": str(exe.resolve()),
        "executable_sha256": sha256_file(exe),
        "fortran_source": str(args.fortran_source.resolve()),
        "source_sha256": (
            sha256_file(args.fortran_source)
            if args.fortran_source.is_file()
            else None
        ),
        "debug_output_sink_library": (
            str(args.debug_output_sink_library)
            if args.debug_output_sink_library is not None
            else None
        ),
        "debug_output_sink_sha256": (
            sha256_file(args.debug_output_sink_library)
            if args.debug_output_sink_library is not None
            else None
        ),
        "debug_output_sink_scope": (
            "output/debug_* file opens only"
            if args.debug_output_sink_library is not None
            else None
        ),
        "calculation_domain_mode": args.td_calculation_domain_mode,
    }
    write_json_atomic(progress_path, progress)

    rows: list[dict[str, Any]] = []
    try:
        for name in requested:
            progress["current_benchmark"] = name
            write_json_atomic(progress_path, progress)
            record, _fields = run_one(
                case=cases[name],
                benchmark=name,
                args=args,
                run_root=run_root,
                exe=exe,
            )
            rows.append(record)
            progress["completed_benchmarks"].append(name)
            progress["completed_runs"] = len(progress["completed_benchmarks"])
            write_json_atomic(progress_path, progress)
    except Exception as exc:
        progress.update(
            {
                "status": "failed",
                "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json_atomic(progress_path, progress)
        raise

    csv_path = args.sweep_dir / "nfpt_fortran_metrics.csv"
    json_path = args.sweep_dir / "nfpt_fortran_metrics.json"
    write_metrics_csv(csv_path, rows)
    json_path.write_text(json.dumps(analytic.json_safe(rows), indent=2) + "\n")
    progress.update(
        {
            "status": "complete",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "current_benchmark": None,
            "metrics_csv": str(csv_path.resolve()),
            "metrics_json": str(json_path.resolve()),
        }
    )
    write_json_atomic(progress_path, progress)
    print(f"metrics_csv={csv_path}", flush=True)
    print(f"metrics_json={json_path}", flush=True)

    code_paths: dict[str, Path] = {
        "runner": Path(__file__).resolve(),
        "analytic_validation": Path(analytic.__file__).resolve(),
        "nfgs_validation_driver": Path(driver.__file__).resolve(),
        "robin_sweep_io": ROOT / "nfgs" / "_backend" / "scripts" / "run_nfgs_td_robin_weight_sweep.py",
        "metrics": ROOT / "nfgs" / "_backend" / "util" / "metrics.py",
        "td_loader": ROOT / "nfgs" / "_backend" / "util" / "td2_io.py",
    }
    if args.debug_output_sink_library is not None:
        code_paths["debug_output_sink_source"] = (
            ROOT / "nfgs" / "_backend" / "scripts" / "nfpt_debug_output_sink.c"
        )

    input_paths: dict[str, Path] = {}
    output_paths: dict[str, Path] = {
        "metrics_csv": csv_path,
        "metrics_json": json_path,
        "progress": progress_path,
    }
    for row in rows:
        benchmark = str(row["benchmark"])
        input_paths[f"{benchmark}_case_meta"] = (
            args.case_cache_dir / f"{benchmark}__case_meta.json"
        )
        input_paths[f"{benchmark}_reference"] = (
            args.case_cache_dir / f"{benchmark}__reference.npz"
        )
        run_dir = Path(row["run_dir"])
        tag = case_tag(benchmark, tuple(row["grid_xyz"]), args.max_iter)
        input_paths[f"{benchmark}_bottom_vector"] = (
            run_dir / "input" / "codex" / tag
        )
        input_paths[f"{benchmark}_par"] = run_dir / "par.in"
        output_paths[f"{benchmark}_raw_b"] = Path(row["source_b"])
        output_paths[f"{benchmark}_raw_j"] = Path(row["source_j"])
        output_paths[f"{benchmark}_fields"] = fortran_field_cache_path(
            args.sweep_dir, benchmark
        )
        output_paths[f"{benchmark}_record"] = (
            args.sweep_dir / f"nfpt_fortran_{benchmark}.json"
        )

    build_paths = {
        "build_flags": exe.resolve().parent / "build_flags.txt",
        "compiler_version": exe.resolve().parent / "compiler_version.txt",
        "build_hashes": exe.resolve().parent / "build_sha256.txt",
        "debug_sink_hashes": (
            exe.resolve().parent / "nfpt_debug_output_sink_sha256.txt"
        ),
    }
    input_paths.update(
        {name: path for name, path in build_paths.items() if path.is_file()}
    )
    missing_inputs = [name for name, path in input_paths.items() if not path.is_file()]
    missing_outputs = [name for name, path in output_paths.items() if not path.is_file()]
    if missing_inputs or missing_outputs:
        raise FileNotFoundError(
            f"Cannot seal manifest; missing inputs={missing_inputs}, "
            f"outputs={missing_outputs}"
        )
    manifest = {
        "experiment": "untouched archived NFPT native-grid TD control",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, *sys.argv],
        "benchmarks": requested,
        "max_iter": args.max_iter,
        "calculation_domain_mode": args.td_calculation_domain_mode,
        "evaluation_domains": {
            str(row["benchmark"]): row["metric_domain"] for row in rows
        },
        "fortran_source": str(args.fortran_source.resolve()),
        "source_sha256": sha256_file(args.fortran_source.resolve()),
        "fortran_executable": str(exe.resolve()),
        "executable_sha256": sha256_file(exe.resolve()),
        "debug_output_sink_library": (
            str(args.debug_output_sink_library)
            if args.debug_output_sink_library is not None
            else None
        ),
        "debug_output_sink_sha256": (
            sha256_file(args.debug_output_sink_library)
            if args.debug_output_sink_library is not None
            else None
        ),
        "debug_output_sink_scope": (
            "output/debug_* file opens only"
            if args.debug_output_sink_library is not None
            else None
        ),
        "code_paths": {name: str(path) for name, path in code_paths.items()},
        "code_sha256": {
            name: sha256_file(path) for name, path in code_paths.items()
        },
        "input_paths": {name: str(path.resolve()) for name, path in input_paths.items()},
        "input_sha256": {
            name: sha256_file(path) for name, path in input_paths.items()
        },
        "output_paths": {
            name: str(path.resolve()) for name, path in output_paths.items()
        },
        "output_sha256": {
            name: sha256_file(path) for name, path in output_paths.items()
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "notes": [
            "The archived Fortran source and numerical calls are unchanged.",
            "The optional preload library redirects only output/debug_* opens.",
            "NFGS-only ablation, not NFPT, aligns calculation-grid placement.",
        ],
    }
    manifest_path = args.sweep_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
