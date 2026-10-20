#!/usr/bin/env python3
"""Run the unmodified NFPT Fortran solver with symmetric TD3 x/y padding.

The legacy MUDPACK executable can only solve compatible odd node counts.  Its
native TD3 input (256 x 350 x 200) selects 257 x 385 x 257 nodes and therefore
leaves one padding cell on the positive x/y sides.  This wrapper preserves the
physical TD3 extent by resampling the photospheric input to 255 x 349 x 200.
The unmodified executable then selects 257 x 385 x 257 nodes with exactly
1/1 x padding and 18/18 y padding.  Reconstructed fields are resampled back
to the original 256 x 350 x 200 analysis lattice.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
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
from nfgs._backend.scripts.run_nfgs_td_robin_weight_sweep import field_cache_dir  # noqa: E402
from nfgs._backend.util.metrics import compute_reference_metrics  # noqa: E402
from nfgs._backend.util.nfpt_fortran_compat import nfpt_fortran_grid  # noqa: E402
from nfgs._backend.util.td2_io import load_b_czyx  # noqa: E402

INPUT_NX = 255
INPUT_NY = 349


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--fortran-exe", type=Path, default=DEFAULT_FORTRAN_EXE)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resample_uniform_axis(
    values: np.ndarray, *, axis: int, n_out: int
) -> np.ndarray:
    """Linearly resample a uniformly spaced axis while preserving endpoints."""
    n_in = values.shape[axis]
    if n_in == n_out:
        return np.asarray(values, dtype=np.float64).copy()
    src = np.arange(n_in, dtype=np.float64)
    dst = np.linspace(0.0, float(n_in - 1), int(n_out), dtype=np.float64)
    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    flat = moved.reshape(-1, n_in)
    out = np.empty((flat.shape[0], n_out), dtype=np.float64)
    for index, row in enumerate(flat):
        out[index] = np.interp(dst, src, row)
    return np.moveaxis(out.reshape(*moved.shape[:-1], n_out), -1, axis)


def resample_xy(
    values: np.ndarray, *, nx_out: int, ny_out: int
) -> np.ndarray:
    """Resample arrays ordered (..., y, x) over an unchanged physical extent."""
    return resample_uniform_axis(
        resample_uniform_axis(values, axis=-1, n_out=nx_out),
        axis=-2,
        n_out=ny_out,
    )


def load_td3_case() -> dict[str, Any]:
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
    case = driver.load_td_case_cache(proxy, "td3")
    if case is None:
        case = driver.make_td_case(
            "td3",
            calculation_domain_mode="mudpack",
            chunk_depth=4,
            n_test_x=66,
            n_test_y=66,
            n_test_z=30,
        )
    if "boundary" not in case:
        case["boundary"] = {
            name: np.asarray(case["reference"][name][0], dtype=np.float64)
            for name in ("Bx", "By", "Bz")
        }
    return case


def main() -> int:
    args = parse_args()
    exe = args.fortran_exe
    if not exe.is_file():
        raise FileNotFoundError(exe)
    if not os.access(exe, os.X_OK):
        exe.chmod(exe.stat().st_mode | 0o100)

    cache_path = field_cache_dir(args.sweep_dir) / "td3__nfpt_fortran_symmetric.npz"
    record_path = args.sweep_dir / "nfpt_fortran_symmetric_td3.json"
    if cache_path.is_file() and record_path.is_file() and not args.force:
        print(f"[cache] TD3 <- {cache_path}")
        return 0

    grid = nfpt_fortran_grid(INPUT_NX, INPUT_NY, 200)
    if (grid.nx, grid.ny, grid.nz, grid.bfx, grid.bfy) != (257, 385, 257, 1, 18):
        raise RuntimeError(f"Unexpected symmetric Fortran grid: {grid}")

    case = load_td3_case()
    _nx, _ny, nz, dx, dy, dz = case["dims"]
    fortran_case = {
        "boundary": {
            name: resample_xy(values, nx_out=INPUT_NX, ny_out=INPUT_NY)
            for name, values in case["boundary"].items()
        },
        "dims": (INPUT_NX, INPUT_NY, nz, dx, dy, dz),
    }
    tag = case_tag("td3_symmetric", fortran_case["dims"], args.max_iter)
    run_dir = args.sweep_dir / "fortran_runs_symmetric" / tag
    input_path = run_dir / "input" / "codex" / tag
    b_path = run_dir / "ext" / "b" / f"b_{tag}"
    j_path = run_dir / "ext" / "j" / f"j_{tag}"
    ensure_run_tree(run_dir)
    write_bottom_input(input_path, fortran_case)
    write_par(
        run_dir / "par.in",
        input_ref=f"input/codex/{tag}",
        dims=fortran_case["dims"],
        max_iter=args.max_iter,
    )

    print(
        f"[run] TD3 | NFPT Fortran | input={INPUT_NX}x{INPUT_NY}x{nz} "
        f"| padded={grid.nx}x{grid.ny}x{grid.nz}",
        flush=True,
    )
    wall = run_fortran_cmd(
        [str(exe.resolve()), "1", "codex", tag],
        cwd=run_dir,
        log_path=run_dir / "fortran_solve.log",
    )
    wall += run_fortran_cmd(
        [str(exe.resolve()), "11", "codex", tag],
        cwd=run_dir,
        log_path=run_dir / "fortran_extract.log",
    )
    reported = [
        elapsed
        for elapsed in (
            parse_elapsed(run_dir / "fortran_solve.log"),
            parse_elapsed(run_dir / "fortran_extract.log"),
        )
        if elapsed is not None
    ]
    if reported:
        wall = float(sum(reported))

    bx, by, bz = load_b_czyx(str(b_path), INPUT_NX, INPUT_NY, nz)
    jx, jy, jz = load_b_czyx(str(j_path), INPUT_NX, INPUT_NY, nz)
    fields = {
        name: resample_xy(values, nx_out=_nx, ny_out=_ny)
        for name, values in {
            "Bx": bx,
            "By": by,
            "Bz": bz,
            "Jx": jx,
            "Jy": jy,
            "Jz": jz,
        }.items()
    }
    metrics = compute_reference_metrics(case["reference"], fields, case["metric_slices"])
    history = read_history(run_dir)
    sigmas = history.get("sigma", [])
    iters = history.get("iter", [])
    record = {
        "benchmark": "td3",
        "benchmark_display": "TD3",
        "solver": "nfpt_fortran_symmetric",
        "implementation": "unmodified NFPT Fortran executable",
        "input_grid_xyz": [INPUT_NX, INPUT_NY, int(nz)],
        "calculation_grid_xyz": [grid.nx, grid.ny, grid.nz],
        "calculation_pad_xy": [grid.bfx, grid.bfy],
        "metric_grid_xyz": [int(_nx), int(_ny), int(nz)],
        "padding_description": "symmetric x/y padding after input resampling",
        "wall_time_s": wall,
        "n_iters": len(iters),
        "best_iter": int(iters[int(np.nanargmin(sigmas))]) if sigmas else 0,
        "best_sigma": float(np.nanmin(sigmas)) if sigmas else float("nan"),
        "stop_reason": "fortran_completed",
        "metric_domain": case["metric_domain"],
        **{key: float(metrics[key]) for key in analytic.METRIC_KEYS},
        "history": history,
        "run_dir": str(run_dir),
    }
    analytic.save_field_cache(cache_path, fields)
    record_path.write_text(json.dumps(analytic.json_safe(record), indent=2) + "\n")
    print(
        f"TD3 NFPT Fortran symmetric: C_vec={record['C_vec']:.6f} "
        f"sigma_J={record['sigma_J']:.6f} wall={wall:.1f}s"
    )
    print(f"field_cache={cache_path}")
    print(f"record={record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
