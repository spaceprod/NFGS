"""Array operations shared by notebooks and benchmark scripts."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def to_numpy(a):
    """Return a CPU numpy array for numpy arrays or torch-like tensors."""
    detach = getattr(a, "detach", None)
    if detach is not None:
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _d_dx_centered_np(f: np.ndarray, dx: float) -> np.ndarray:
    """Centered d/dx on the last axis; matches ``NFPTSolverGPU._d_dx_centered``."""
    out = np.zeros_like(f)
    nx = f.shape[-1]
    if nx < 2:
        return out
    if nx < 5:
        if nx >= 3:
            out[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2.0 * dx)
            out[..., 0] = (-3.0 * f[..., 0] + 4.0 * f[..., 1] - f[..., 2]) / (2.0 * dx)
            out[..., -1] = (3.0 * f[..., -1] - 4.0 * f[..., -2] + f[..., -3]) / (2.0 * dx)
        else:
            out[..., 0] = (f[..., 1] - f[..., 0]) / dx
            out[..., -1] = (f[..., -1] - f[..., -2]) / dx
        return out
    out[..., 2:-2] = (
        -f[..., 4:] + 8.0 * f[..., 3:-1] - 8.0 * f[..., 1:-3] + f[..., :-4]
    ) / (12.0 * dx)
    out[..., 1] = (f[..., 2] - f[..., 0]) / (2.0 * dx)
    out[..., -2] = (f[..., -1] - f[..., -3]) / (2.0 * dx)
    out[..., 0] = (-3.0 * f[..., 0] + 4.0 * f[..., 1] - f[..., 2]) / (2.0 * dx)
    out[..., -1] = (3.0 * f[..., -1] - 4.0 * f[..., -2] + f[..., -3]) / (2.0 * dx)
    return out


def _d_dy_centered_np(f: np.ndarray, dy: float) -> np.ndarray:
    """Centered d/dy on the penultimate axis (2D ``(y,x)`` or 3D ``(z,y,x)``)."""
    f = np.asarray(f)
    if f.ndim < 2:
        return np.zeros_like(f)
    f_y = np.moveaxis(f, -2, 0)
    out_y = _d_dz_centered_np(f_y, dy)
    return np.moveaxis(out_y, 0, -2)


def _d_dz_centered_np(f: np.ndarray, dz: float) -> np.ndarray:
    """Centered d/dz on axis 0; matches ``NFPTSolverGPU._d_dz_centered``."""
    out = np.zeros_like(f)
    nz = f.shape[0]
    if nz < 2:
        return out
    if nz < 5:
        if nz >= 3:
            out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dz)
            out[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dz)
            out[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * dz)
        else:
            out[0] = (f[1] - f[0]) / dz
            out[-1] = (f[-1] - f[-2]) / dz
        return out
    out[2:-2] = (-f[4:] + 8.0 * f[3:-1] - 8.0 * f[1:-3] + f[:-4]) / (12.0 * dz)
    out[1] = (f[2] - f[0]) / (2.0 * dz)
    out[-2] = (f[-1] - f[-3]) / (2.0 * dz)
    out[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dz)
    out[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * dz)
    return out


def jz_bottom_from_b_horizontal_nfpt(
    Bx: np.ndarray,
    By: np.ndarray,
    *,
    dx: float,
    dy: float,
    curl_scale: float = 1.0,
) -> np.ndarray:
    """Photosphere Jz = (curl B)_z from horizontal B only (NFPT FD stencils)."""
    dBy_dx = _d_dx_centered_np(By, dx)
    dBx_dy = _d_dy_centered_np(Bx, dy)
    return (dBy_dx - dBx_dy) * float(curl_scale)


def curl_zyx_nfpt(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curl of B with NFPTSolverGPU-compatible FD stencils on (z, y, x) arrays."""
    dBy_dx = _d_dx_centered_np(By, dx)
    dBx_dy = _d_dy_centered_np(Bx, dy)
    dBz_dy = _d_dy_centered_np(Bz, dy)
    dBy_dz = _d_dz_centered_np(By, dz)
    dBx_dz = _d_dz_centered_np(Bx, dz)
    dBz_dx = _d_dx_centered_np(Bz, dx)
    Jx = dBz_dy - dBy_dz
    Jy = dBx_dz - dBz_dx
    Jz = dBy_dx - dBx_dy
    return Jx, Jy, Jz


def curl_zyx(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curl of B on a regular grid in (z, y, x) array order (numpy ``gradient``)."""
    return curl_zyx_nfpt(Bx, By, Bz, dx=dx, dy=dy, dz=dz)


def fields_dict_from_bj(B, J) -> Dict[str, np.ndarray]:
    """Full-grid B,J tensors (component, Z, Y, X) to a numpy field dict."""
    Bn = to_numpy(B)
    Jn = to_numpy(J)
    return {
        "Bx": Bn[0],
        "By": Bn[1],
        "Bz": Bn[2],
        "Jx": Jn[0],
        "Jy": Jn[1],
        "Jz": Jn[2],
    }


def finite_stats(name: str, arr) -> None:
    """Print compact finite-value stats for notebooks."""
    a = np.asarray(to_numpy(arr))
    print(
        f"{name:>14s}: shape={a.shape}, min={np.nanmin(a): .3e}, "
        f"max={np.nanmax(a): .3e}, mean={np.nanmean(a): .3e}, std={np.nanstd(a): .3e}"
    )
