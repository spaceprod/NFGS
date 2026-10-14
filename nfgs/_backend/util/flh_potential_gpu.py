"""GPU-capable potential-field solver matching ``flhtools.BField.computePotentialField``.

The solver here mirrors the Neumann potential-field construction in
``flhtools/flhcart.py`` for arrays stored in the project-standard ``(z, y, x)``
order.  It uses all six normal-field boundaries, not only the bottom ``Bz``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import scipy.fftpack as fp
import torch


ArrayOrder = Literal["zyx"]


@dataclass(frozen=True)
class FLHPotentialField:
    """Face-centered potential field in ``(z, y, x)`` order.

    Shapes follow the staggered layout used by ``flhtools`` after transposing
    from its internal ``(x, y, z)`` storage:

    - ``Bpx``: ``(nz + 1, ny + 1, nx)``
    - ``Bpy``: ``(nz + 1, ny, nx + 1)``
    - ``Bpz``: ``(nz, ny + 1, nx + 1)``
    """

    Bpx: np.ndarray
    Bpy: np.ndarray
    Bpz: np.ndarray
    flux_imbalance: float


def compute_potential_field_flh_numpy(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
    order: ArrayOrder = "zyx",
) -> FLHPotentialField:
    """Reference NumPy/SciPy implementation of the FLHTOOLS potential solver."""
    if order != "zyx":
        raise ValueError('Only order="zyx" is supported')
    bx_xyz, by_xyz, bz_xyz = _as_xyz_numpy(Bx, By, Bz)
    bpx, bpy, bpz, imbalance = _potential_field_xyz_numpy(
        bx_xyz, by_xyz, bz_xyz, dx=float(dx), dy=float(dy), dz=float(dz)
    )
    return FLHPotentialField(
        Bpx=np.transpose(bpx, (2, 1, 0)),
        Bpy=np.transpose(bpy, (2, 1, 0)),
        Bpz=np.transpose(bpz, (2, 1, 0)),
        flux_imbalance=float(imbalance),
    )


def compute_potential_field_flh_torch(
    Bx: np.ndarray | torch.Tensor,
    By: np.ndarray | torch.Tensor,
    Bz: np.ndarray | torch.Tensor,
    *,
    dx: float,
    dy: float,
    dz: float,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    order: ArrayOrder = "zyx",
    return_numpy: bool = True,
) -> FLHPotentialField | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch implementation of the FLHTOOLS potential solver.

    Use ``device="cuda"`` for GPU acceleration.  When ``device`` is omitted,
    CUDA is selected if available, otherwise CPU is used.  ``return_numpy=True``
    returns the same dataclass as the NumPy reference implementation; set it to
    ``False`` to keep tensors on the selected device.
    """
    if order != "zyx":
        raise ValueError('Only order="zyx" is supported')
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    bx_xyz, by_xyz, bz_xyz = _as_xyz_torch(Bx, By, Bz, device=device, dtype=dtype)
    bpx, bpy, bpz, imbalance = _potential_field_xyz_torch(
        bx_xyz, by_xyz, bz_xyz, dx=float(dx), dy=float(dy), dz=float(dz)
    )
    bpx = bpx.permute(2, 1, 0).contiguous()
    bpy = bpy.permute(2, 1, 0).contiguous()
    bpz = bpz.permute(2, 1, 0).contiguous()
    if not return_numpy:
        return bpx, bpy, bpz, imbalance
    return FLHPotentialField(
        Bpx=bpx.detach().cpu().numpy(),
        Bpy=bpy.detach().cpu().numpy(),
        Bpz=bpz.detach().cpu().numpy(),
        flux_imbalance=float(imbalance.detach().cpu().item()),
    )


def _as_xyz_numpy(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bx = np.asarray(Bx, dtype=np.float64)
    by = np.asarray(By, dtype=np.float64)
    bz = np.asarray(Bz, dtype=np.float64)
    if bx.shape != by.shape or bx.shape != bz.shape or bx.ndim != 3:
        raise ValueError("Bx, By, and Bz must have identical 3D shapes")
    return np.transpose(bx, (2, 1, 0)), np.transpose(by, (2, 1, 0)), np.transpose(bz, (2, 1, 0))


def _as_xyz_torch(
    Bx: np.ndarray | torch.Tensor,
    By: np.ndarray | torch.Tensor,
    Bz: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bx = torch.as_tensor(Bx, device=device, dtype=dtype)
    by = torch.as_tensor(By, device=device, dtype=dtype)
    bz = torch.as_tensor(Bz, device=device, dtype=dtype)
    if bx.shape != by.shape or bx.shape != bz.shape or bx.ndim != 3:
        raise ValueError("Bx, By, and Bz must have identical 3D shapes")
    return bx.permute(2, 1, 0).contiguous(), by.permute(2, 1, 0).contiguous(), bz.permute(2, 1, 0).contiguous()


def _boundary_normal_faces_xyz_np(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bx0 = 0.25 * (bx[0, :-1, :-1] + bx[0, 1:, :-1] + bx[0, :-1, 1:] + bx[0, 1:, 1:])
    bx1 = 0.25 * (bx[-1, :-1, :-1] + bx[-1, 1:, :-1] + bx[-1, :-1, 1:] + bx[-1, 1:, 1:])
    by0 = 0.25 * (by[:-1, 0, :-1] + by[1:, 0, :-1] + by[:-1, 0, 1:] + by[1:, 0, 1:])
    by1 = 0.25 * (by[:-1, -1, :-1] + by[1:, -1, :-1] + by[:-1, -1, 1:] + by[1:, -1, 1:])
    bz0 = 0.25 * (bz[:-1, :-1, 0] + bz[1:, :-1, 0] + bz[:-1, 1:, 0] + bz[1:, 1:, 0])
    bz1 = 0.25 * (bz[:-1, :-1, -1] + bz[1:, :-1, -1] + bz[:-1, 1:, -1] + bz[1:, 1:, -1])
    return bx0, bx1, by0, by1, bz0, bz1


def _potential_field_xyz_numpy(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nx, ny, nz = bx.shape
    bx0, bx1, by0, by1, bz0, bz1 = _boundary_normal_faces_xyz_np(bx, by, bz)
    bx0, bx1, by0, by1, bz0, bz1, imbalance = _remove_flux_imbalance_np(
        bx0, bx1, by0, by1, bz0, bz1, dx=dx, dy=dy, dz=dz
    )

    u = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.float64)
    u[0, :, :] = bx0 / dx
    u[-1, :, :] -= bx1 / dx
    u[:, 0, :] += by0 / dy
    u[:, -1, :] -= by1 / dy
    u[:, :, 0] += bz0 / dz
    u[:, :, -1] -= bz1 / dz

    u = fp.dct(u, axis=0, type=2, norm="ortho")
    u = fp.dct(u, axis=1, type=2, norm="ortho")
    u = fp.dct(u, axis=2, type=2, norm="ortho")

    m, n, p = np.mgrid[0 : nx - 1, 0 : ny - 1, 0 : nz - 1]
    m = (2.0 * np.cos(m * np.pi / (nx - 1)) - 2.0) / dx**2
    n = (2.0 * np.cos(n * np.pi / (ny - 1)) - 2.0) / dy**2
    p = (2.0 * np.cos(p * np.pi / (nz - 1)) - 2.0) / dz**2
    u = u / (m + n + p + 1.0e-16)

    u = fp.idct(u, axis=2, type=2, norm="ortho")
    u = fp.idct(u, axis=1, type=2, norm="ortho")
    u = fp.idct(u, axis=0, type=2, norm="ortho")

    ub = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.float64)
    ub[1:-1, 1:-1, 1:-1] = u
    ub[0, 1:-1, 1:-1] = ub[1, 1:-1, 1:-1] - bx0 * dx
    ub[-1, 1:-1, 1:-1] = ub[-2, 1:-1, 1:-1] + bx1 * dx
    ub[1:-1, 0, 1:-1] = ub[1:-1, 1, 1:-1] - by0 * dy
    ub[1:-1, -1, 1:-1] = ub[1:-1, -2, 1:-1] + by1 * dy
    ub[1:-1, 1:-1, 0] = ub[1:-1, 1:-1, 1] - bz0 * dz
    ub[1:-1, 1:-1, -1] = ub[1:-1, 1:-1, -2] + bz1 * dz

    bpx = np.zeros((nx, ny + 1, nz + 1), dtype=np.float64)
    bpy = np.zeros((nx + 1, ny, nz + 1), dtype=np.float64)
    bpz = np.zeros((nx + 1, ny + 1, nz), dtype=np.float64)
    bpx[:, 1:-1, 1:-1] = (ub[1:, 1:-1, 1:-1] - ub[:-1, 1:-1, 1:-1]) / dx
    bpy[1:-1, :, 1:-1] = (ub[1:-1, 1:, 1:-1] - ub[1:-1, :-1, 1:-1]) / dy
    bpz[1:-1, 1:-1, :] = (ub[1:-1, 1:-1, 1:] - ub[1:-1, 1:-1, :-1]) / dz
    return (*_ghost_b_xyz(bpx, bpy, bpz, dx=dx, dy=dy, dz=dz), float(imbalance))


def _remove_flux_imbalance_np(
    bx0: np.ndarray,
    bx1: np.ndarray,
    by0: np.ndarray,
    by1: np.ndarray,
    bz0: np.ndarray,
    bz1: np.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    nxm1, nym1 = bz0.shape
    nzm1 = bx0.shape[1]
    imbalance = (
        np.sum(bx0) * dy * dz
        - np.sum(bx1) * dy * dz
        + np.sum(by0) * dx * dz
        - np.sum(by1) * dx * dz
        + np.sum(bz0) * dx * dy
        - np.sum(bz1) * dx * dy
    )
    surfarea = 2.0 * nym1 * nzm1 * dy * dz + 2.0 * nxm1 * nzm1 * dx * dz + 2.0 * nxm1 * nym1 * dx * dy
    corr = imbalance / surfarea
    return bx0 - corr, bx1 + corr, by0 - corr, by1 + corr, bz0 - corr, bz1 + corr, float(imbalance)


def _potential_field_xyz_torch(
    bx: torch.Tensor,
    by: torch.Tensor,
    bz: torch.Tensor,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nx, ny, nz = bx.shape
    bx0, bx1, by0, by1, bz0, bz1 = _boundary_normal_faces_xyz_torch(bx, by, bz)
    bx0, bx1, by0, by1, bz0, bz1, imbalance = _remove_flux_imbalance_torch(
        bx0, bx1, by0, by1, bz0, bz1, dx=dx, dy=dy, dz=dz
    )

    u = torch.zeros((nx - 1, ny - 1, nz - 1), device=bx.device, dtype=bx.dtype)
    u[0, :, :] = bx0 / dx
    u[-1, :, :] -= bx1 / dx
    u[:, 0, :] += by0 / dy
    u[:, -1, :] -= by1 / dy
    u[:, :, 0] += bz0 / dz
    u[:, :, -1] -= bz1 / dz

    u = _dct2_ortho(u, 0)
    u = _dct2_ortho(u, 1)
    u = _dct2_ortho(u, 2)

    kx = torch.arange(nx - 1, device=bx.device, dtype=bx.dtype).reshape(-1, 1, 1)
    ky = torch.arange(ny - 1, device=bx.device, dtype=bx.dtype).reshape(1, -1, 1)
    kz = torch.arange(nz - 1, device=bx.device, dtype=bx.dtype).reshape(1, 1, -1)
    denom = (
        (2.0 * torch.cos(kx * torch.pi / (nx - 1)) - 2.0) / dx**2
        + (2.0 * torch.cos(ky * torch.pi / (ny - 1)) - 2.0) / dy**2
        + (2.0 * torch.cos(kz * torch.pi / (nz - 1)) - 2.0) / dz**2
        + 1.0e-16
    )
    u = u / denom

    u = _idct2_ortho(u, 2)
    u = _idct2_ortho(u, 1)
    u = _idct2_ortho(u, 0)

    ub = torch.zeros((nx + 1, ny + 1, nz + 1), device=bx.device, dtype=bx.dtype)
    ub[1:-1, 1:-1, 1:-1] = u
    ub[0, 1:-1, 1:-1] = ub[1, 1:-1, 1:-1] - bx0 * dx
    ub[-1, 1:-1, 1:-1] = ub[-2, 1:-1, 1:-1] + bx1 * dx
    ub[1:-1, 0, 1:-1] = ub[1:-1, 1, 1:-1] - by0 * dy
    ub[1:-1, -1, 1:-1] = ub[1:-1, -2, 1:-1] + by1 * dy
    ub[1:-1, 1:-1, 0] = ub[1:-1, 1:-1, 1] - bz0 * dz
    ub[1:-1, 1:-1, -1] = ub[1:-1, 1:-1, -2] + bz1 * dz

    bpx = torch.zeros((nx, ny + 1, nz + 1), device=bx.device, dtype=bx.dtype)
    bpy = torch.zeros((nx + 1, ny, nz + 1), device=bx.device, dtype=bx.dtype)
    bpz = torch.zeros((nx + 1, ny + 1, nz), device=bx.device, dtype=bx.dtype)
    bpx[:, 1:-1, 1:-1] = (ub[1:, 1:-1, 1:-1] - ub[:-1, 1:-1, 1:-1]) / dx
    bpy[1:-1, :, 1:-1] = (ub[1:-1, 1:, 1:-1] - ub[1:-1, :-1, 1:-1]) / dy
    bpz[1:-1, 1:-1, :] = (ub[1:-1, 1:-1, 1:] - ub[1:-1, 1:-1, :-1]) / dz
    return (*_ghost_b_xyz(bpx, bpy, bpz, dx=dx, dy=dy, dz=dz), imbalance)


def _boundary_normal_faces_xyz_torch(
    bx: torch.Tensor,
    by: torch.Tensor,
    bz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bx0 = 0.25 * (bx[0, :-1, :-1] + bx[0, 1:, :-1] + bx[0, :-1, 1:] + bx[0, 1:, 1:])
    bx1 = 0.25 * (bx[-1, :-1, :-1] + bx[-1, 1:, :-1] + bx[-1, :-1, 1:] + bx[-1, 1:, 1:])
    by0 = 0.25 * (by[:-1, 0, :-1] + by[1:, 0, :-1] + by[:-1, 0, 1:] + by[1:, 0, 1:])
    by1 = 0.25 * (by[:-1, -1, :-1] + by[1:, -1, :-1] + by[:-1, -1, 1:] + by[1:, -1, 1:])
    bz0 = 0.25 * (bz[:-1, :-1, 0] + bz[1:, :-1, 0] + bz[:-1, 1:, 0] + bz[1:, 1:, 0])
    bz1 = 0.25 * (bz[:-1, :-1, -1] + bz[1:, :-1, -1] + bz[:-1, 1:, -1] + bz[1:, 1:, -1])
    return bx0, bx1, by0, by1, bz0, bz1


def _remove_flux_imbalance_torch(
    bx0: torch.Tensor,
    bx1: torch.Tensor,
    by0: torch.Tensor,
    by1: torch.Tensor,
    bz0: torch.Tensor,
    bz1: torch.Tensor,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nxm1, nym1 = bz0.shape
    nzm1 = bx0.shape[1]
    imbalance = (
        torch.sum(bx0) * dy * dz
        - torch.sum(bx1) * dy * dz
        + torch.sum(by0) * dx * dz
        - torch.sum(by1) * dx * dz
        + torch.sum(bz0) * dx * dy
        - torch.sum(bz1) * dx * dy
    )
    surfarea = 2.0 * nym1 * nzm1 * dy * dz + 2.0 * nxm1 * nzm1 * dx * dz + 2.0 * nxm1 * nym1 * dx * dy
    corr = imbalance / surfarea
    return bx0 - corr, bx1 + corr, by0 - corr, by1 + corr, bz0 - corr, bz1 + corr, imbalance


@lru_cache(maxsize=64)
def _dct2_ortho_matrix_cpu(n: int, dtype_name: str) -> torch.Tensor:
    dtype = getattr(torch, dtype_name)
    k = torch.arange(n, dtype=dtype).reshape(n, 1)
    j = torch.arange(n, dtype=dtype).reshape(1, n)
    mat = torch.cos(torch.pi * k * (2.0 * j + 1.0) / (2.0 * n))
    mat[0, :] *= (1.0 / n) ** 0.5
    if n > 1:
        mat[1:, :] *= (2.0 / n) ** 0.5
    return mat


def _dct2_ortho_matrix(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    dtype_name = str(dtype).replace("torch.", "")
    return _dct2_ortho_matrix_cpu(n, dtype_name).to(device=device)


def _dct2_ortho(x: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim if dim >= 0 else x.ndim + dim
    n = x.shape[dim]
    mat = _dct2_ortho_matrix(n, device=x.device, dtype=x.dtype)
    moved = torch.movedim(x, dim, 0)
    y = mat @ moved.reshape(n, -1)
    return torch.movedim(y.reshape_as(moved), 0, dim)


def _idct2_ortho(x: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim if dim >= 0 else x.ndim + dim
    n = x.shape[dim]
    mat = _dct2_ortho_matrix(n, device=x.device, dtype=x.dtype).T
    moved = torch.movedim(x, dim, 0)
    y = mat @ moved.reshape(n, -1)
    return torch.movedim(y.reshape_as(moved), 0, dim)


def _ghost_b_xyz(bx, by, bz, *, dx: float, dy: float, dz: float):
    """Fill face-centered ghost values with the ``flhcart.ghostB(type="jxn")`` rules."""
    bz[1:-1, 0, 1:-1] = bz[1:-1, 1, 1:-1] - dy * (by[1:-1, 0, 2:-1] - by[1:-1, 0, 1:-2]) / dz
    bz[1:-1, -1, 1:-1] = bz[1:-1, -2, 1:-1] + dy * (by[1:-1, -1, 2:-1] - by[1:-1, -1, 1:-2]) / dz
    by[1:-1, 1:-1, 0] = by[1:-1, 1:-1, 1] - dz * (bz[1:-1, 2:-1, 0] - bz[1:-1, 1:-2, 0]) / dy
    by[1:-1, 1:-1, -1] = by[1:-1, 1:-1, -2] + dz * (bz[1:-1, 2:-1, -1] - bz[1:-1, 1:-2, -1]) / dy
    bz[0, 1:-1, 1:-1] = bz[1, 1:-1, 1:-1] - dx * (bx[0, 1:-1, 2:-1] - bx[0, 1:-1, 1:-2]) / dz
    bz[-1, 1:-1, 1:-1] = bz[-2, 1:-1, 1:-1] + dx * (bx[-1, 1:-1, 2:-1] - bx[-1, 1:-1, 1:-2]) / dz
    bx[1:-1, 1:-1, 0] = bx[1:-1, 1:-1, 1] - dz * (bz[2:-1, 1:-1, 0] - bz[1:-2, 1:-1, 0]) / dx
    bx[1:-1, 1:-1, -1] = bx[1:-1, 1:-1, -2] + dz * (bz[2:-1, 1:-1, -1] - bz[1:-2, 1:-1, -1]) / dx
    bx[:, 0, 1:-1] = bx[:, 1, 1:-1] - dy * (by[1:, 0, 1:-1] - by[:-1, 0, 1:-1]) / dx
    bx[:, -1, 1:-1] = bx[:, -2, 1:-1] + dy * (by[1:, -1, 1:-1] - by[:-1, -1, 1:-1]) / dx
    by[0, :, 1:-1] = by[1, :, 1:-1] - dx * (bx[0, 1:, 1:-1] - bx[0, :-1, 1:-1]) / dy
    by[-1, :, 1:-1] = by[-2, :, 1:-1] + dx * (bx[-1, 1:, 1:-1] - bx[-1, :-1, 1:-1]) / dy

    bx[0, 0, 1:-1] = 2.0 * bx[0, 1, 1:-1] - bx[0, 2, 1:-1]
    bx[-1, 0, 1:-1] = 2.0 * bx[-1, 1, 1:-1] - bx[-1, 2, 1:-1]
    bx[0, -1, 1:-1] = 2.0 * bx[0, -2, 1:-1] - bx[0, -3, 1:-1]
    bx[-1, -1, 1:-1] = 2.0 * bx[-1, -2, 1:-1] - bx[-1, -3, 1:-1]
    bx[:, :, 0] = 2.0 * bx[:, :, 1] - bx[:, :, 2]
    bx[:, :, -1] = 2.0 * bx[:, :, -2] - bx[:, :, -3]

    by[0, 0, 1:-1] = 2.0 * by[1, 0, 1:-1] - by[2, 0, 1:-1]
    by[0, -1, 1:-1] = 2.0 * by[1, -1, 1:-1] - by[2, -1, 1:-1]
    by[-1, 0, 1:-1] = 2.0 * by[-2, 0, 1:-1] - by[-3, 0, 1:-1]
    by[-1, -1, 1:-1] = 2.0 * by[-2, -1, 1:-1] - by[-3, -1, 1:-1]
    by[:, :, 0] = 2.0 * by[:, :, 1] - by[:, :, 2]
    by[:, :, -1] = 2.0 * by[:, :, -2] - by[:, :, -3]

    bz[0, 0, 1:-1] = 2.0 * bz[1, 0, 1:-1] - bz[2, 0, 1:-1]
    bz[0, -1, 1:-1] = 2.0 * bz[1, -1, 1:-1] - bz[2, -1, 1:-1]
    bz[-1, 0, 1:-1] = 2.0 * bz[-2, 0, 1:-1] - bz[-3, 0, 1:-1]
    bz[-1, -1, 1:-1] = 2.0 * bz[-2, -1, 1:-1] - bz[-3, -1, 1:-1]
    bz[:, :, 0] = 2.0 * bz[:, :, 1] - bz[:, :, 2]
    bz[:, :, -1] = 2.0 * bz[:, :, -2] - bz[:, :, -3]
    return bx, by, bz
