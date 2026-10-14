"""FastQSL-backed squashing-factor maps on Cartesian cross-sections.

The vendored FastQSL kernel traces all field lines on the GPU.  This module
keeps the coordinate contract used by NFPT: field components are ``(z,y,x)``
and the TD vertical diagnostic plane is the physical ``y=0`` plane.

Vendored TraceBline* kernels allocate per-thread scratch on the stack (not
device ``new``), so a full-resolution ``y=0`` plane can launch safely without
exhausting the default CUDA malloc heap.

NFGS solution Q maps optionally apply a light FFT Gaussian low-pass to B
*only for tracing* (see ``DEFAULT_QMAP_B_SMOOTH_SIGMA_CELLS``).  Stored NLFFF
fields and vector metrics remain on the raw solution.
"""

from __future__ import annotations

from nfgs.paths import project_root

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = project_root()
DEFAULT_FASTQSL_ROOT = PROJECT_ROOT / "third_party" / "FastQSL"
# Keep concurrent TraceAllBline launches within the original FastQSL guidance
# (~8192 threads) even after the stack-local fix.
DEFAULT_TRACE_CHUNK_SEEDS = 8192

# Q-map-only pre-smooth for NFGS (not analytic) B fields.
# Grid-scale / DCT high-k content in reconstructed B is amplified by FastQSL's
# neighbour footpoint Jacobian into checkerboard mottling on y=0 Q.  A light
# spectral Gaussian (sigma=0.75 cells) suppresses that artifact for topology
# figures without changing cached NLFFF fields or reference-vector metrics.
# See results/nfgs_td_robin_weight_sweep/q_spectral_filter_diag/.
DEFAULT_QMAP_B_SMOOTH_SIGMA_CELLS = 0.75

# Cache/provenance contract for the TD ``y=0`` plane.  FastQSL accepts
# floating-point seed coordinates, so an even number of y nodes must use the
# half-index between the two central nodes rather than choosing one side.
Y0_PLANE_COORDINATE_CONTRACT = "physical_midplane_fractional_index_v2"


def physical_midplane_grid_index(n_nodes: int) -> float:
    """Return the grid coordinate of the physical midpoint of a node axis."""
    n = int(n_nodes)
    if n < 1:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    return 0.5 * float(n - 1)


def y0_qmap_cache_matches(
    metadata: dict[str, Any],
    *,
    ny: int,
    b_smooth_sigma_cells: float,
) -> bool:
    """Whether a cached Q map follows the current physical-midplane contract."""
    if int(metadata.get("downsample", 1) or 1) != 1:
        return False
    if metadata.get("plane_coordinate_contract") != Y0_PLANE_COORDINATE_CONTRACT:
        return False
    try:
        plane_y = float(metadata["plane_y_index"])
        cached_sigma = float(metadata.get("qmap_b_smooth_sigma_cells", -1.0))
    except (KeyError, TypeError, ValueError):
        return False
    return (
        abs(plane_y - physical_midplane_grid_index(ny)) < 1.0e-12
        and abs(cached_sigma - float(b_smooth_sigma_cells)) < 1.0e-12
    )


def spectral_gaussian_lowpass(
    field: np.ndarray,
    *,
    sigma_cells: float,
) -> np.ndarray:
    """Isotropic Gaussian low-pass in Fourier space (sigma in grid cells)."""
    arr = np.asarray(field, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("field must be finite for spectral filtering")
    sigma = float(sigma_cells)
    if sigma <= 0.0:
        return arr.astype(np.float32, copy=True)
    freqs = [np.fft.fftfreq(n) for n in arr.shape]
    grids = np.meshgrid(*freqs, indexing="ij")
    k2 = sum(g.astype(np.float64) ** 2 for g in grids)
    transfer = np.exp(-0.5 * ((2.0 * np.pi * sigma) ** 2) * k2)
    filt = np.fft.ifftn(np.fft.fftn(arr) * transfer).real
    return filt.astype(np.float32)


def smooth_b_components_for_qmap(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    *,
    sigma_cells: float = DEFAULT_QMAP_B_SMOOTH_SIGMA_CELLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Bx/By/Bz smoothed for FastQSL Q only; callers keep raw fields."""
    return (
        spectral_gaussian_lowpass(bx, sigma_cells=sigma_cells),
        spectral_gaussian_lowpass(by, sigma_cells=sigma_cells),
        spectral_gaussian_lowpass(bz, sigma_cells=sigma_cells),
    )


def filter_vector_field(
    fields: dict[str, np.ndarray],
    *,
    sigma_cells: float = DEFAULT_QMAP_B_SMOOTH_SIGMA_CELLS,
) -> dict[str, np.ndarray]:
    """Smooth Bx/By/Bz for Q diagnostics; does not copy other keys."""
    bx, by, bz = smooth_b_components_for_qmap(
        fields["Bx"], fields["By"], fields["Bz"], sigma_cells=sigma_cells
    )
    return {"Bx": bx, "By": by, "Bz": bz}


def vector_rel_l1(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """Mean relative L1 change of vector B between two field dicts."""
    num = 0.0
    den = 0.0
    for key in ("Bx", "By", "Bz"):
        aa = np.asarray(a[key], dtype=np.float64)
        bb = np.asarray(b[key], dtype=np.float64)
        num += float(np.mean(np.abs(aa - bb)))
        den += float(np.mean(np.abs(aa)))
    return num / max(den, 1e-30)


def _configure_cuda_toolkit() -> None:
    """Make a system CUDA toolkit visible to CuPy's runtime compiler."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        candidates = [
            Path("/usr/local/cuda/bin/nvcc"),
            *sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True),
        ]
        nvcc = next((str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)
    if nvcc is None:
        raise RuntimeError(
            "FastQSL needs a CUDA Toolkit with nvcc. Install CUDA in the GPU environment "
            "and make nvcc available on PATH before running --td-topology-qsl."
        )
    cuda_root = Path(nvcc).resolve().parents[1]
    os.environ.setdefault("CUDA_PATH", str(cuda_root))
    os.environ.setdefault("CUDA_HOME", str(cuda_root))
    os.environ["PATH"] = f"{cuda_root / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_root / 'lib64'}{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}"


def _fastqsl_imports(fastqsl_root: Path):
    _configure_cuda_toolkit()
    root = fastqsl_root.resolve()
    if not (root / "fastqslpy" / "FastQSL.py").is_file():
        raise FileNotFoundError(f"FastQSL source is missing at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import cupy

    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise RuntimeError(
            "FastQSL could not initialize CUDA. Run this experiment in a GPU-enabled "
            "environment with a compatible NVIDIA driver, CUDA Toolkit, and CuPy build."
        ) from exc
    if device_count < 1:
        raise RuntimeError("FastQSL requires a CUDA GPU, but no CUDA device is visible.")
    from fastqslpy import FastQSL, kernels

    return cupy, FastQSL, kernels


def _ensure_device_malloc_heap(cupy: Any, *, bytes_limit: int = 64 << 20) -> int:
    """Raise the CUDA device malloc heap before any TraceBline launch."""
    runtime = cupy.cuda.runtime
    try:
        current = int(runtime.deviceGetLimit(runtime.cudaLimitMallocHeapSize))
    except Exception:
        return 0
    if current < int(bytes_limit):
        runtime.deviceSetLimit(runtime.cudaLimitMallocHeapSize, int(bytes_limit))
        current = int(runtime.deviceGetLimit(runtime.cudaLimitMallocHeapSize))
    return current


def _launch(kernel: Any, args: tuple[Any, ...], n_points: int, block_x: int) -> tuple[tuple[int, int, int], tuple[int, int]]:
    block = (int(block_x), 1, 1)
    grid = (max(1, int(np.ceil(n_points / block[0]))), 1)
    kernel(grid, block, args)
    return block, grid


def compute_y0_qmap(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    *,
    fastqsl_root: Path = DEFAULT_FASTQSL_ROOT,
    y_index: float | None = None,
    step_len: float = 1.0 / 8.0,
    tol: float = 0.1,
    block_x: int = 128,
    trace_chunk_seeds: int = DEFAULT_TRACE_CHUNK_SEEDS,
    b_smooth_sigma_cells: float = 0.0,
    include_twist: bool = False,
) -> dict[str, Any]:
    """Compute FastQSL's ``QCalcPlane`` map on the ``y=0`` x-z plane.

    The result has shape ``(nz,nx)``.  The plane is seeded at every x-z grid
    point, then both directions are traced to the numerical-domain boundary by
    FastQSL.  Invalid boundary transitions remain NaN rather than being
    interpolated or percentile-clipped.

    Tracing is done in full resolution; when ``trace_chunk_seeds`` is positive
    the plane is processed in batches so each CUDA launch stays within a safe
    concurrent-thread budget.  ``QCalcPlane`` still runs once on the full plane.

    If ``b_smooth_sigma_cells > 0``, Bx/By/Bz are FFT-Gaussian-smoothed *only
    for this Q calculation* (NFGS topology diagnostics).  Analytic references
    should pass ``0``.  The input arrays are never modified in-place.
    """
    # Optional Q-only pre-smooth (see DEFAULT_QMAP_B_SMOOTH_SIGMA_CELLS).
    smooth_sigma = float(b_smooth_sigma_cells)
    if smooth_sigma > 0.0:
        bx, by, bz = smooth_b_components_for_qmap(
            bx, by, bz, sigma_cells=smooth_sigma
        )

    bx = np.ascontiguousarray(bx, dtype=np.float32)
    by = np.ascontiguousarray(by, dtype=np.float32)
    bz = np.ascontiguousarray(bz, dtype=np.float32)
    if bx.ndim != 3 or bx.shape != by.shape or bx.shape != bz.shape:
        raise ValueError("bx, by, and bz must have the same (nz,ny,nx) shape")
    nz, ny, nx = bx.shape
    iy = physical_midplane_grid_index(ny) if y_index is None else float(y_index)
    if not 0.0 <= iy <= float(ny - 1):
        raise ValueError(f"y_index={iy} is outside [0,{ny - 1}]")

    cupy, FastQSL, kernels = _fastqsl_imports(fastqsl_root)
    heap_bytes = _ensure_device_malloc_heap(cupy)
    started = time.perf_counter()
    trace = kernels.compileTraceBlineAdaptive()
    bx_gpu, by_gpu, bz_gpu = (cupy.asarray(value) for value in (bx, by, bz))
    bshape = cupy.asarray(np.asarray([nx, ny, nz], dtype=np.int32))
    x, z = cupy.meshgrid(cupy.arange(nx, dtype=cupy.float32), cupy.arange(nz, dtype=cupy.float32), indexing="xy")
    x_inp = x.ravel()
    z_inp = z.ravel()
    y_inp = cupy.full(x_inp.shape, iy, dtype=cupy.float32)
    plane_shape = x.shape
    n_total = int(x_inp.size)
    arrays = [cupy.zeros(x_inp.shape, dtype=cupy.float32) for _ in range(15)]
    (
        x_start, y_start, z_start, x_end, y_end, z_end,
        bx_start, by_start, bz_start, bx_end, by_end, bz_end,
        bx_inp, by_inp, bz_inp,
    ) = arrays
    b_flag, flag_start, flag_end = [cupy.zeros(x_inp.shape, dtype=cupy.int32) for _ in range(3)]
    line_len = cupy.zeros(x_inp.shape, dtype=cupy.float64)
    normal_y = cupy.asarray([0.0, 1.0, 0.0], dtype=cupy.float32)
    dummy = cupy.zeros((1, 1), dtype=cupy.float32)
    if include_twist:
        # The FastQSL tracer accumulates (J dot B)/(4 pi |B|^2) dl along both
        # halves of every field line.  The TD grids are isotropic, so using
        # grid-coordinate derivatives and path lengths leaves T_w dimensionless.
        # Arrays follow the repository's (z, y, x) order.  FastQSL traces the
        # resulting current along the line; this centered grid curl supplies
        # the current values for that line integral.
        d_bz_dy = cupy.gradient(bz_gpu, axis=1, edge_order=2)
        d_by_dz = cupy.gradient(by_gpu, axis=0, edge_order=2)
        d_bx_dz = cupy.gradient(bx_gpu, axis=0, edge_order=2)
        d_bz_dx = cupy.gradient(bz_gpu, axis=2, edge_order=2)
        d_by_dx = cupy.gradient(by_gpu, axis=2, edge_order=2)
        d_bx_dy = cupy.gradient(bx_gpu, axis=1, edge_order=2)
        cur_bx = d_bz_dy - d_by_dz
        cur_by = d_bx_dz - d_bz_dx
        cur_bz = d_by_dx - d_bx_dy
        twist = cupy.zeros(x_inp.shape, dtype=cupy.float64)
        cur_flag = cupy.asarray([True], dtype=cupy.bool_)
    else:
        cur_bx = cur_by = cur_bz = dummy
        twist = dummy
        cur_flag = cupy.asarray([False], dtype=cupy.bool_)
    step_arr = cupy.asarray([step_len], dtype=cupy.float32)
    tol_arr = cupy.asarray([tol], dtype=cupy.float32)

    chunk = int(trace_chunk_seeds) if trace_chunk_seeds and int(trace_chunk_seeds) > 0 else n_total
    chunk = max(1, min(chunk, n_total))
    last_block: tuple[int, int, int] = (int(block_x), 1, 1)
    last_grid: tuple[int, int] = (1, 1)
    n_chunks = 0
    for start in range(0, n_total, chunk):
        stop = min(n_total, start + chunk)
        n_chunk = stop - start
        sl = slice(start, stop)
        n_trace = cupy.asarray([n_chunk], dtype=cupy.ulonglong)
        last_block, last_grid = _launch(
            trace,
            (
                bx_gpu, by_gpu, bz_gpu, bshape, cur_bx, cur_by, cur_bz, twist[sl] if include_twist else twist,
                cur_flag,
                x_inp[sl], y_inp[sl], z_inp[sl], normal_y,
                x_start[sl], y_start[sl], z_start[sl], flag_start[sl],
                x_end[sl], y_end[sl], z_end[sl], flag_end[sl],
                bx_inp[sl], by_inp[sl], bz_inp[sl], b_flag[sl],
                bx_start[sl], by_start[sl], bz_start[sl],
                bx_end[sl], by_end[sl], bz_end[sl],
                step_arr, n_trace, line_len[sl], tol_arr,
            ),
            n_chunk,
            block_x,
        )
        cupy.cuda.stream.get_current_stream().synchronize()
        n_chunks += 1

    result = FastQSL.ResReshape(
        plane_shape, x_end, y_end, z_end, flag_end, x_start, y_start, z_start,
        flag_start, bx_start, by_start, bz_start, bx_end, by_end, bz_end,
        bx_inp, by_inp, bz_inp, b_flag,
    )
    q_inner = FastQSL.QCalcPlane(*result[:-1], result[-1], cupy.float32(1.0), normal_y)
    cupy.cuda.stream.get_current_stream().synchronize()
    q = cupy.full(plane_shape, cupy.nan, dtype=cupy.float32)
    q[1:-1, 1:-1] = q_inner
    bundle = {
        "q": q.get(),
        "flag_start": result[3].get().astype(np.int32),
        "flag_end": result[7].get().astype(np.int32),
        "start_xyz": np.stack(
            [result[0].get(), result[1].get(), result[2].get()], axis=-1
        ).astype(np.float32),
        "end_xyz": np.stack(
            [result[4].get(), result[5].get(), result[6].get()], axis=-1
        ).astype(np.float32),
        "line_len": line_len.get().reshape(plane_shape).astype(np.float32),
        "metadata": {
            "algorithm": "FastQSL QCalcPlane on y=0",
            "shape_zyx": [int(nz), int(ny), int(nx)],
            "plane_y_index": float(iy),
            "plane_coordinate_contract": Y0_PLANE_COORDINATE_CONTRACT,
            "plane_is_fractional": bool(not float(iy).is_integer()),
            "plane_shape_zx": [int(nz), int(nx)],
            "step_len_grid": float(step_len),
            "tolerance": float(tol),
            "block": list(last_block),
            "grid": list(last_grid),
            "trace_chunk_seeds": int(chunk),
            "trace_n_chunks": int(n_chunks),
            "device_malloc_heap_bytes": int(heap_bytes),
            "downsample": 1,
            # 0 => raw B; >0 => Q-only FFT Gaussian pre-smooth (NFGS diagnostics).
            "qmap_b_smooth_sigma_cells": float(smooth_sigma),
            "elapsed_s": time.perf_counter() - started,
            "twist_included": bool(include_twist),
        },
    }
    if include_twist:
        bundle["twist"] = twist.get().reshape(plane_shape).astype(np.float64)
    return bundle


def compute_bottom_qmap(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    *,
    fastqsl_root: Path = DEFAULT_FASTQSL_ROOT,
    z_index: float | None = None,
    step_len: float = 1.0 / 8.0,
    tol: float = 0.1,
    block_x: int = 128,
    trace_chunk_seeds: int = DEFAULT_TRACE_CHUNK_SEEDS,
    b_smooth_sigma_cells: float = 0.0,
) -> dict[str, Any]:
    """Compute FastQSL ``QCalcPlane`` on the photosphere (bottom ``x–y`` plane).

    The result has shape ``(ny, nx)``.  Seeds sit a half-cell above the bottom
    face by default (``z=0.5``) so TraceBline starts inside the volume; the
    plane normal is ``+z``.  Invalid boundary transitions remain NaN.
    """
    smooth_sigma = float(b_smooth_sigma_cells)
    if smooth_sigma > 0.0:
        bx, by, bz = smooth_b_components_for_qmap(
            bx, by, bz, sigma_cells=smooth_sigma
        )

    bx = np.ascontiguousarray(bx, dtype=np.float32)
    by = np.ascontiguousarray(by, dtype=np.float32)
    bz = np.ascontiguousarray(bz, dtype=np.float32)
    if bx.ndim != 3 or bx.shape != by.shape or bx.shape != bz.shape:
        raise ValueError("bx, by, and bz must have the same (nz,ny,nx) shape")
    nz, ny, nx = bx.shape
    # Half-cell above z=0 keeps seeds inside the box for TraceBline.
    z_seed = 0.5 if z_index is None else float(z_index)
    if not 0.0 <= z_seed <= float(nz - 1):
        raise ValueError(f"z_index={z_seed} is outside [0,{nz - 1}]")

    cupy, FastQSL, kernels = _fastqsl_imports(fastqsl_root)
    heap_bytes = _ensure_device_malloc_heap(cupy)
    started = time.perf_counter()
    trace = kernels.compileTraceBlineAdaptive()
    bx_gpu, by_gpu, bz_gpu = (cupy.asarray(value) for value in (bx, by, bz))
    bshape = cupy.asarray(np.asarray([nx, ny, nz], dtype=np.int32))
    x, y = cupy.meshgrid(
        cupy.arange(nx, dtype=cupy.float32),
        cupy.arange(ny, dtype=cupy.float32),
        indexing="xy",
    )
    x_inp = x.ravel()
    y_inp = y.ravel()
    z_inp = cupy.full(x_inp.shape, z_seed, dtype=cupy.float32)
    plane_shape = x.shape
    n_total = int(x_inp.size)
    arrays = [cupy.zeros(x_inp.shape, dtype=cupy.float32) for _ in range(15)]
    (
        x_start, y_start, z_start, x_end, y_end, z_end,
        bx_start, by_start, bz_start, bx_end, by_end, bz_end,
        bx_inp, by_inp, bz_inp,
    ) = arrays
    b_flag, flag_start, flag_end = [
        cupy.zeros(x_inp.shape, dtype=cupy.int32) for _ in range(3)
    ]
    line_len = cupy.zeros(x_inp.shape, dtype=cupy.float64)
    normal_z = cupy.asarray([0.0, 0.0, 1.0], dtype=cupy.float32)
    dummy = cupy.zeros((1, 1), dtype=cupy.float32)
    step_arr = cupy.asarray([step_len], dtype=cupy.float32)
    tol_arr = cupy.asarray([tol], dtype=cupy.float32)
    cur_flag = cupy.asarray([False], dtype=cupy.bool_)

    chunk = int(trace_chunk_seeds) if trace_chunk_seeds and int(trace_chunk_seeds) > 0 else n_total
    chunk = max(1, min(chunk, n_total))
    last_block: tuple[int, int, int] = (int(block_x), 1, 1)
    last_grid: tuple[int, int] = (1, 1)
    n_chunks = 0
    for start in range(0, n_total, chunk):
        stop = min(n_total, start + chunk)
        n_chunk = stop - start
        sl = slice(start, stop)
        n_trace = cupy.asarray([n_chunk], dtype=cupy.ulonglong)
        last_block, last_grid = _launch(
            trace,
            (
                bx_gpu, by_gpu, bz_gpu, bshape, dummy, dummy, dummy, dummy,
                cur_flag,
                x_inp[sl], y_inp[sl], z_inp[sl], normal_z,
                x_start[sl], y_start[sl], z_start[sl], flag_start[sl],
                x_end[sl], y_end[sl], z_end[sl], flag_end[sl],
                bx_inp[sl], by_inp[sl], bz_inp[sl], b_flag[sl],
                bx_start[sl], by_start[sl], bz_start[sl],
                bx_end[sl], by_end[sl], bz_end[sl],
                step_arr, n_trace, line_len[sl], tol_arr,
            ),
            n_chunk,
            block_x,
        )
        cupy.cuda.stream.get_current_stream().synchronize()
        n_chunks += 1

    result = FastQSL.ResReshape(
        plane_shape, x_end, y_end, z_end, flag_end, x_start, y_start, z_start,
        flag_start, bx_start, by_start, bz_start, bx_end, by_end, bz_end,
        bx_inp, by_inp, bz_inp, b_flag,
    )
    q_inner = FastQSL.QCalcPlane(
        *result[:-1], result[-1], cupy.float32(1.0), normal_z
    )
    cupy.cuda.stream.get_current_stream().synchronize()
    q = cupy.full(plane_shape, cupy.nan, dtype=cupy.float32)
    q[1:-1, 1:-1] = q_inner
    return {
        "q": q.get(),
        "flag_start": result[3].get().astype(np.int32),
        "flag_end": result[7].get().astype(np.int32),
        "start_xyz": np.stack(
            [result[0].get(), result[1].get(), result[2].get()], axis=-1
        ).astype(np.float32),
        "end_xyz": np.stack(
            [result[4].get(), result[5].get(), result[6].get()], axis=-1
        ).astype(np.float32),
        "line_len": line_len.get().reshape(plane_shape).astype(np.float32),
        "metadata": {
            "algorithm": "FastQSL QCalcPlane on photosphere (bottom)",
            "shape_zyx": [int(nz), int(ny), int(nx)],
            "plane_z_index": float(z_seed),
            "plane_shape_yx": [int(ny), int(nx)],
            "step_len_grid": float(step_len),
            "tolerance": float(tol),
            "block": list(last_block),
            "grid": list(last_grid),
            "trace_chunk_seeds": int(chunk),
            "trace_n_chunks": int(n_chunks),
            "device_malloc_heap_bytes": int(heap_bytes),
            "downsample": 1,
            "qmap_b_smooth_sigma_cells": float(smooth_sigma),
            "elapsed_s": time.perf_counter() - started,
        },
    }
