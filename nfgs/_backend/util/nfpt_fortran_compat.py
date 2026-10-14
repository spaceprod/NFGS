"""Compatibility helpers for the legacy NFPT Fortran grid and bottom Jz."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class NFPTFortranGrid:
    """Observed-grid metadata implied by ``nfpt_1.0.2.f90``."""

    nxo: int
    nyo: int
    nzo: int
    nx: int
    ny: int
    nz: int
    bfx: int
    bfy: int
    bfz: int
    dx: float
    dy: float
    dz: float
    x_bounds: Tuple[float, float]
    y_bounds: Tuple[float, float]
    z_bounds: Tuple[float, float]
    padded_x_bounds: Tuple[float, float]
    padded_y_bounds: Tuple[float, float]
    padded_z_bounds: Tuple[float, float]


def _mudpack_axis_size(n_obs: int) -> int:
    """Return the padded MUDPACK-friendly size selected by ``mudcof``."""
    n_obs = int(n_obs)
    if n_obs < 2:
        return max(1, n_obs)
    n_real = float(n_obs)
    exp2 = math.ceil(math.log((n_real - 1.0) / 2.0) / math.log(2.0)) + 1
    exp3 = math.ceil(math.log((n_real - 1.0) / 3.0) / math.log(2.0)) + 1
    n2 = int(2 * (2 ** (exp2 - 1)) + 1)
    n3 = int(3 * (2 ** (exp3 - 1)) + 1)
    return min(n2, n3)


def _even_xy_pad_size(n_calc: int, n_obs: int) -> int:
    """Grow ``n_calc`` by one cell when needed so left/right pads match.

    Legacy Fortran uses ``bfx = (nx - nxo) // 2``, which leaves any odd
    remainder on the high-index side only (e.g. rem=1 → left=0, right=1).
    For mirror-symmetric analytic benchmarks, bump the calculation size so
    both sides receive the same pad count.
    """
    n_calc = int(n_calc)
    n_obs = int(n_obs)
    if n_calc < n_obs:
        raise ValueError(
            f"calculation size {n_calc} is smaller than observed size {n_obs}"
        )
    if (n_calc - n_obs) % 2 == 1:
        return n_calc + 1
    return n_calc


def nfpt_fortran_grid(
    nxo: int,
    nyo: int,
    nzo: int,
    *,
    symmetric_xy: bool = False,
) -> NFPTFortranGrid:
    """Reproduce the Fortran ``mudcof``/``initial`` coordinate spacing.

    The Fortran code selects MUDPACK-friendly ``nx/ny/nz``, pads x/y with
    ``bfx=(nx-nxo)//2`` (odd remainder on the high-index side only), and pads
    z on top only.  It then derives one isotropic grid spacing from the padded
    x domain.

    Parameters
    ----------
    symmetric_xy :
        If True, grow ``nx``/``ny`` by one cell whenever ``(n - n_obs)`` is
        odd so that left and right (or bottom and top in y) pads are equal.
        z padding remains Fortran-like (top only).  Default False preserves
        bit-for-bit legacy NFPT padding.
    """
    nxo, nyo, nzo = int(nxo), int(nyo), int(nzo)
    nx = _mudpack_axis_size(nxo)
    ny = _mudpack_axis_size(nyo)
    nz = _mudpack_axis_size(nzo)
    if symmetric_xy:
        nx = _even_xy_pad_size(nx, nxo)
        ny = _even_xy_pad_size(ny, nyo)
    bfx = (nx - nxo) // 2
    bfy = (ny - nyo) // 2
    bfz = nz - nzo
    if symmetric_xy:
        if (nx - nxo) - bfx != bfx or (ny - nyo) - bfy != bfy:
            raise RuntimeError(
                "symmetric_xy padding failed to equalize left/right pads: "
                f"nx={nx}, nxo={nxo}, bfx={bfx}; ny={ny}, nyo={nyo}, bfy={bfy}"
            )

    if nx <= 1:
        h = 1.0
        lx = 0.0
    else:
        lx = float(nxo - 1) * (1.0 + float(2 * bfx) / float(nxo - 1))
        h = 2.0 * lx / float(nx - 1)
    ly = lx * float(ny - 1) / float(nx - 1) if nx > 1 else 0.0
    lz = 2.0 * lx * float(nz - 1) / float(nx - 1) if nx > 1 else 0.0

    padded_x = (-lx, lx)
    padded_y = (-ly, ly)
    padded_z = (0.0, lz)
    x0 = padded_x[0] + float(bfx) * h
    y0 = padded_y[0] + float(bfy) * h
    return NFPTFortranGrid(
        nxo=nxo,
        nyo=nyo,
        nzo=nzo,
        nx=nx,
        ny=ny,
        nz=nz,
        bfx=bfx,
        bfy=bfy,
        bfz=bfz,
        dx=h,
        dy=h,
        dz=h,
        x_bounds=(x0, x0 + float(max(nxo - 1, 0)) * h),
        y_bounds=(y0, y0 + float(max(nyo - 1, 0)) * h),
        z_bounds=(0.0, float(max(nzo - 1, 0)) * h),
        padded_x_bounds=padded_x,
        padded_y_bounds=padded_y,
        padded_z_bounds=padded_z,
    )


def nfpt_fortran_spacing(nxo: int, nyo: int, nzo: int) -> Tuple[float, float, float]:
    """Return ``(dx, dy, dz)`` implied by the legacy Fortran code."""
    g = nfpt_fortran_grid(nxo, nyo, nzo)
    return g.dx, g.dy, g.dz


def bottom_jz_fortran_readin_botbal(
    Bx: np.ndarray,
    By: np.ndarray,
    *,
    dx: float,
    dy: float,
    ijzbcor: int = 3,
) -> np.ndarray:
    """Bottom ``Jz`` matching Fortran ``readin`` plus ``botbal``.

    ``Bx`` and ``By`` are bottom maps in ``(Ny, Nx)`` order. The returned array
    keeps only physical nodes, with physical edges zeroed as in Fortran.
    """
    bx = np.asarray(Bx, dtype=np.float64)
    by = np.asarray(By, dtype=np.float64)
    if bx.ndim != 2 or by.ndim != 2 or bx.shape != by.shape:
        raise ValueError(f"Bx/By must be matching 2D arrays; got {bx.shape}, {by.shape}")

    nyo, nxo = bx.shape
    grid = nfpt_fortran_grid(nxo, nyo, 2)
    bxb = np.zeros((grid.ny, grid.nx), dtype=np.float64)
    byb = np.zeros_like(bxb)
    y1 = grid.bfy + nyo
    x1 = grid.bfx + nxo
    bxb[grid.bfy : y1, grid.bfx : x1] = bx
    byb[grid.bfy : y1, grid.bfx : x1] = by

    jzb = np.zeros_like(bxb, dtype=np.float64)
    if grid.nx >= 3 and grid.ny >= 3:
        jzb[1:-1, 1:-1] = (
            (byb[1:-1, 2:] - byb[1:-1, :-2]) / (2.0 * float(dx))
            - (bxb[2:, 1:-1] - bxb[:-2, 1:-1]) / (2.0 * float(dy))
        )

    mode = int(ijzbcor)
    if mode >= 2 and grid.nx >= 3 and grid.ny >= 3:
        jzp = np.where(jzb >= 0.0, jzb, 0.0)
        jzn = np.where(jzb < 0.0, jzb, 0.0)

        def cell_sum(a: np.ndarray) -> float:
            return float(
                0.25
                * (
                    np.sum(a[1:-1, 2:])
                    + np.sum(a[2:, 1:-1])
                    + np.sum(a[1:-1, 1:-1])
                    + np.sum(a[2:, 2:])
                )
            )

        jzsum = cell_sum(jzb)
        jzpsum = cell_sum(jzp)
        jznsum = cell_sum(jzn)
        if mode == 2:
            jzflcor = jzpsum - 0.5 * jzsum
        elif mode == 3:
            jzflcor = max(jzpsum, abs(jznsum))
        elif mode == 4:
            jzflcor = min(jzpsum, abs(jznsum))
        else:
            jzflcor = None

        if jzflcor is not None:
            if abs(jzpsum) > 0.0:
                jzb = np.where(jzb >= 0.0, (jzflcor / jzpsum) * jzb, jzb)
            if abs(jznsum) > 0.0:
                jzb = np.where(jzb < 0.0, (jzflcor / abs(jznsum)) * jzb, jzb)

    if grid.nx > 0:
        jzb[:, 0] = 0.0
        jzb[:, -1] = 0.0
    if grid.ny > 0:
        jzb[0, :] = 0.0
        jzb[-1, :] = 0.0
    return jzb[grid.bfy : y1, grid.bfx : x1].copy()
