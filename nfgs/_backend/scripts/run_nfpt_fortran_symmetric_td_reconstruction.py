#!/usr/bin/env python3
"""Run unmodified NFPT Fortran with symmetric MUDPACK-compatible TD padding."""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.scripts import run_analytic_validation as analytic  # noqa: E402
from nfgs._backend.scripts import run_nfgs_analytic_validation as driver  # noqa: E402
from nfgs._backend.scripts.run_nfpt_fortran_td_reconstruction import (  # noqa: E402
    DEFAULT_FORTRAN_EXE,
    DEFAULT_SWEEP_DIR,
    case_tag,
    ensure_run_tree,
    parse_elapsed,
    read_history,
    run_fortran_cmd,
    write_bottom_input,
    write_par,
)
from nfgs._backend.scripts.run_nfpt_fortran_symmetric_td3_reconstruction import resample_xy  # noqa: E402
from nfgs._backend.scripts.run_nfgs_td_robin_weight_sweep import (  # noqa: E402
    field_cache_dir,
    load_field_cache,
)
from nfgs._backend.util.metrics import compute_nf2_metrics  # noqa: E402
from nfgs._backend.util.nfpt_fortran_compat import nfpt_fortran_grid  # noqa: E402
from nfgs._backend.util.td2_io import load_b_czyx  # noqa: E402

SYMMETRIC_INPUT_XY = {"td1": (149, 249), "td2": (149, 249), "td3": (255, 349)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--fortran-exe", type=Path, default=DEFAULT_FORTRAN_EXE)
    parser.add_argument("--benchmarks", default="td1,td2,td3")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def paper_metric_slices(case: dict[str, Any]) -> tuple[slice, slice, slice]:
    """Return the TD benchmark protocol confirmed by Yi."""
    return case["metric_slices"]


def paper_metric_domain(case: dict[str, Any]) -> str:
    return str(case["metric_domain"])


def load_case(name: str) -> dict[str, Any]:
    proxy = SimpleNamespace(
        field_cache_dir=ROOT / "results" / "nfgs_analytic_validation_td" / "fields",
        td_calculation_domain_mode="mudpack",
        td_topology_chunk_depth=4,
        n_test_x=66,
        n_test_y=66,
        n_test_z=30,
        source_root=ROOT,
        lnl_calculation_domain_mode="mudpack",
    )
    case = driver.load_td_case_cache(proxy, name)
    if case is None:
        case = driver.make_td_case(
            name,
            calculation_domain_mode="mudpack",
            chunk_depth=4,
            n_test_x=66,
            n_test_y=66,
            n_test_z=30,
        )
    if "boundary" not in case:
        case["boundary"] = {
            field: np.asarray(case["reference"][field][0], dtype=np.float64)
            for field in ("Bx", "By", "Bz")
        }
    return case


def executable_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_one(
    *,
    name: str,
    case: dict[str, Any],
    sweep_dir: Path,
    exe: Path,
    max_iter: int,
    force: bool,
) -> dict[str, Any]:
    nx_in, ny_in = SYMMETRIC_INPUT_XY[name]
    nx_ref, ny_ref, nz, dx, dy, dz = case["dims"]
    grid = nfpt_fortran_grid(nx_in, ny_in, int(nz))
    if (grid.nx - nx_in) % 2 or (grid.ny - ny_in) % 2:
        raise RuntimeError(f"Asymmetric Fortran grid selected for {name}: {grid}")

    cache_path = field_cache_dir(sweep_dir) / f"{name}__nfpt_fortran_symmetric.npz"
    record_path = sweep_dir / f"nfpt_fortran_symmetric_{name}.json"
    if cache_path.is_file() and not force:
        fields = load_field_cache(cache_path)
        history: dict[str, list[Any]] = {}
        wall = float("nan")
        run_dir = None
        if record_path.is_file():
            prior = json.loads(record_path.read_text())
            history = prior.get("history", {})
            wall = float(prior.get("wall_time_s", wall))
            run_dir = prior.get("run_dir")
    else:
        tag = case_tag(f"{name}_symmetric", (nx_in, ny_in, nz, dx, dy, dz), max_iter)
        run_dir_path = sweep_dir / "fortran_runs_symmetric" / tag
        input_path = run_dir_path / "input" / "codex" / tag
        b_path = run_dir_path / "ext" / "b" / f"b_{tag}"
        j_path = run_dir_path / "ext" / "j" / f"j_{tag}"
        ensure_run_tree(run_dir_path)
        fortran_case = {
            "boundary": {
                key: resample_xy(value, nx_out=nx_in, ny_out=ny_in)
                for key, value in case["boundary"].items()
            },
            "dims": (nx_in, ny_in, nz, dx, dy, dz),
        }
        write_bottom_input(input_path, fortran_case)
        write_par(
            run_dir_path / "par.in",
            input_ref=f"input/codex/{tag}",
            dims=fortran_case["dims"],
            max_iter=max_iter,
        )
        print(
            f"[run] {name} | NFPT Fortran | input={nx_in}x{ny_in}x{nz} "
            f"| padded={grid.nx}x{grid.ny}x{grid.nz}",
            flush=True,
        )
        wall = run_fortran_cmd(
            [str(exe.resolve()), "1", "codex", tag],
            cwd=run_dir_path,
            log_path=run_dir_path / "fortran_solve.log",
        )
        wall += run_fortran_cmd(
            [str(exe.resolve()), "11", "codex", tag],
            cwd=run_dir_path,
            log_path=run_dir_path / "fortran_extract.log",
        )
        reported = [
            value
            for value in (
                parse_elapsed(run_dir_path / "fortran_solve.log"),
                parse_elapsed(run_dir_path / "fortran_extract.log"),
            )
            if value is not None
        ]
        if reported:
            wall = float(sum(reported))
        bx, by, bz = load_b_czyx(str(b_path), nx_in, ny_in, int(nz))
        jx, jy, jz = load_b_czyx(str(j_path), nx_in, ny_in, int(nz))
        fields = {
            key: resample_xy(value, nx_out=int(nx_ref), ny_out=int(ny_ref))
            for key, value in {
                "Bx": bx, "By": by, "Bz": bz, "Jx": jx, "Jy": jy, "Jz": jz,
            }.items()
        }
        analytic.save_field_cache(cache_path, fields)
        history = read_history(run_dir_path)
        run_dir = str(run_dir_path)

    slices = paper_metric_slices(case)
    metrics = compute_nf2_metrics(case["reference"], fields, slices)
    sigmas = history.get("sigma", [])
    iters = history.get("iter", [])
    record = {
        "benchmark": name,
        "benchmark_display": case["display"],
        "solver": "nfpt_fortran_symmetric",
        "implementation": "unmodified NFPT Fortran executable",
        "fortran_executable": str(exe),
        "fortran_executable_sha256": executable_sha256(exe),
        "input_grid_xyz": [nx_in, ny_in, int(nz)],
        "calculation_grid_xyz": [grid.nx, grid.ny, grid.nz],
        "calculation_pad_xy": [grid.bfx, grid.bfy],
        "metric_grid_xyz": [int(nx_ref), int(ny_ref), int(nz)],
        "padding_description": "symmetric x/y padding after input resampling",
        "metric_domain": paper_metric_domain(case),
        "wall_time_s": wall,
        "n_iters": len(iters),
        "best_iter": int(iters[int(np.nanargmin(sigmas))]) if sigmas else 0,
        "best_sigma": float(np.nanmin(sigmas)) if sigmas else float("nan"),
        "stop_reason": "fortran_completed",
        **{key: float(metrics[key]) for key in analytic.METRIC_KEYS},
        "history": history,
        "run_dir": run_dir,
    }
    record_path.write_text(json.dumps(analytic.json_safe(record), indent=2) + "\n")
    return record


def main() -> int:
    args = parse_args()
    requested = [name.strip().lower() for name in args.benchmarks.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(SYMMETRIC_INPUT_XY))
    if unknown:
        raise ValueError(f"Unsupported benchmarks: {unknown}")
    exe = args.fortran_exe
    if not exe.is_file():
        raise FileNotFoundError(exe)
    if not os.access(exe, os.X_OK):
        exe.chmod(exe.stat().st_mode | 0o100)
    rows = [
        run_one(
            name=name,
            case=load_case(name),
            sweep_dir=args.sweep_dir,
            exe=exe,
            max_iter=args.max_iter,
            force=args.force,
        )
        for name in requested
    ]
    csv_path = args.sweep_dir / "nfpt_fortran_symmetric_metrics.csv"
    keys = ["benchmark", *analytic.METRIC_KEYS, "wall_time_s", "metric_domain"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"metrics_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
