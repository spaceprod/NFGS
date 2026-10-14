"""Binary field loaders and TD2 benchmark grid helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np

from .field_ops import curl_zyx_nfpt
from .metrics import flux_rope_center_from_bottom_jz, test_domain_slices_zyx
from .nfpt_fortran_compat import bottom_jz_fortran_readin_botbal, nfpt_fortran_grid


def load_td_fortran_bcor(
    path: str,
    nx: int,
    ny: int,
    nz: int,
    *,
    bin_layout: str = "nx_ny_nz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load float64 Bx, By, Bz blocks in a Fortran-style per-component layout."""
    raw = np.fromfile(path, dtype=np.float64)
    npt = nx * ny * nz
    need = 3 * npt
    if raw.size != need:
        raise ValueError(
            f"bcor size mismatch: got {raw.size} floats, expected {need} (= 3 * {npt})"
        )
    if bin_layout == "nx_ny_nz":
        shape_f = (nx, ny, nz)
        perm = None
    elif bin_layout == "nx_nz_ny":
        shape_f = (nx, nz, ny)
        perm = (0, 2, 1)
    else:
        raise ValueError('bin_layout must be "nx_ny_nz" or "nx_nz_ny"')

    def _one_block(start: int) -> np.ndarray:
        arr = raw[start : start + npt].reshape(shape_f, order="F")
        return np.transpose(arr, perm) if perm is not None else arr

    bx = _one_block(0)
    by = _one_block(npt)
    bz = _one_block(2 * npt)
    return bx, by, bz


def load_b_czyx(path: str, nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Bx, By, Bz as (Nz, Ny, Nx) float64 from a C-order component blob."""
    raw = np.fromfile(path, dtype=np.float64)
    npt = nx * ny * nz
    need = 3 * npt
    if raw.size != need:
        raise ValueError(f"Expected {need} floats, got {raw.size}")
    vol = raw.reshape((3, nz, ny, nx), order="C")
    return vol[0], vol[1], vol[2]


@dataclass
class TD2Grid:
    Nx: int = 150
    Ny: int = 250
    Nz: int = 100
    N_TEST_X: int = 66
    N_TEST_Y: int = 66
    N_TEST_Z: int = 30
    L_ch: float = 100.0
    spacing_mode: Literal["fortran", "td_model", "yi_td2"] = "fortran"
    ijzbcor: int = 3

    def spacing(
        self,
    ) -> Tuple[
        float,
        float,
        float,
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]:
        if self.spacing_mode == "fortran":
            g = nfpt_fortran_grid(self.Nx, self.Ny, self.Nz)
            return g.dx, g.dy, g.dz, g.x_bounds, g.y_bounds, g.z_bounds
        if self.spacing_mode == "yi_td2":
            canonical_shape = (150, 250, 100)
            if (self.Nx, self.Ny, self.Nz) != canonical_shape:
                raise ValueError(
                    "Yi TD2 reference coordinates are defined for the frozen "
                    f"150x250x100 artifact, got {self.Nx}x{self.Ny}x{self.Nz}."
                )
            h = 360.0 / float(self.Nx - 1)
            x_bounds = (-180.0, 180.0)
            y_max = h * float(self.Ny - 1) / 2.0
            y_bounds = (-y_max, y_max)
            z_bounds = (0.0, h * float(self.Nz - 1))
            return h, h, h, x_bounds, y_bounds, z_bounds
        if self.spacing_mode != "td_model":
            raise ValueError(f"Unsupported TD2 spacing_mode {self.spacing_mode!r}")
        x_bounds = (-self.L_ch * 1.2, self.L_ch * 1.2)
        dx = (x_bounds[1] - x_bounds[0]) / float(self.Nx - 1)
        dy = 1.5
        dz = 1.1
        y_max = (dy * (self.Ny - 1)) / 2.0
        y_bounds = (-y_max, y_max)
        z_max = dz * (self.Nz - 1)
        z_bounds = (0.0, z_max)
        return dx, dy, dz, x_bounds, y_bounds, z_bounds


def build_td2_reference_and_obs(bcor_path: Path, grid: TD2Grid) -> Dict[str, Any]:
    """Load TD2 bcor reference, compute J, bottom observations, and metric ROI."""
    Bx_xyz, By_xyz, Bz_xyz = load_td_fortran_bcor(str(bcor_path), grid.Nx, grid.Ny, grid.Nz)
    dx, dy, dz, x_bounds, y_bounds, z_bounds = grid.spacing()
    Bx_ref = np.transpose(Bx_xyz, (2, 1, 0)).astype(np.float64, copy=False)
    By_ref = np.transpose(By_xyz, (2, 1, 0)).astype(np.float64, copy=False)
    Bz_ref = np.transpose(Bz_xyz, (2, 1, 0)).astype(np.float64, copy=False)
    Jx_ref, Jy_ref, Jz_ref = curl_zyx_nfpt(Bx_ref, By_ref, Bz_ref, dx=dx, dy=dy, dz=dz)
    Bx_obs = Bx_ref[0].astype(np.float64, copy=False)
    By_obs = By_ref[0].astype(np.float64, copy=False)
    Bz_obs = Bz_ref[0].astype(np.float64, copy=False)
    if grid.spacing_mode == "fortran":
        Jz_obs = bottom_jz_fortran_readin_botbal(
            Bx_obs,
            By_obs,
            dx=dx,
            dy=dy,
            ijzbcor=grid.ijzbcor,
        )
    else:
        Jz_obs = Jz_ref[0].astype(np.float64, copy=False)
    iy_c, ix_c = flux_rope_center_from_bottom_jz(Jz_obs)
    sz, sy, sx = test_domain_slices_zyx(
        grid.Nz,
        grid.Ny,
        grid.Nx,
        iy_c,
        ix_c,
        grid.N_TEST_X,
        grid.N_TEST_Y,
        grid.N_TEST_Z,
    )
    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "x_bounds": x_bounds,
        "y_bounds": y_bounds,
        "z_bounds": z_bounds,
        "Bx_ref": Bx_ref,
        "By_ref": By_ref,
        "Bz_ref": Bz_ref,
        "Jx_ref": Jx_ref,
        "Jy_ref": Jy_ref,
        "Jz_ref": Jz_ref,
        "Bx_obs": Bx_obs,
        "By_obs": By_obs,
        "Bz_obs": Bz_obs,
        "Jz_obs": Jz_obs,
        "flux_rope_iy": iy_c,
        "flux_rope_ix": ix_c,
        "metric_slices": (sz, sy, sx),
    }
