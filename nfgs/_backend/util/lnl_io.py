"""Low & Lou analytic NLFFF benchmark helpers."""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import numpy as np
from scipy.integrate import solve_bvp

from .field_ops import curl_zyx_nfpt
from .metrics import metric_roi_slices_zyx
from .nfpt_fortran_compat import bottom_jz_fortran_readin_botbal, nfpt_fortran_grid

BUFFER_THICKNESS = (128 + 128) // 6
JAROLIM_C2_RESOLUTION = (80, 80, 72)
JAROLIM_C2_EVAL_SIZE = 64
JAROLIM_C2_BOUNDS = (-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
LOW_LOU_FAMILY = {
    "low_lou_c2": {"n": 1, "m": 1, "l": 0.3, "psi": float(np.pi * 0.15)},
    "low_lou_ll1": {"n": 1, "m": 1, "l": 0.3, "psi": float(np.pi / 4.0)},
    "low_lou_ll2": {"n": 3, "m": 1, "l": 0.3, "psi": float(4.0 * np.pi / 5.0)},
}


def lnl_buffer_thickness(resolution: Tuple[int, int, int]) -> int:
    nx, ny, _nz = resolution
    return max(0, (int(nx) + int(ny)) // 6)


def jarolim_c2_metric_slices_zyx(
    resolution: Tuple[int, int, int] = JAROLIM_C2_RESOLUTION,
    eval_size: int = JAROLIM_C2_EVAL_SIZE,
) -> Tuple[slice, slice, slice]:
    """Return the central cube used for Jarolim C2.

    Jarolim et al. (2023) construct C2 on an 80 x 80 x 72 grid. Their
    publication evaluation code crops eight cells from both horizontal
    boundaries and retains the bottom 64 vertical planes.
    """
    nx, ny, nz = (int(v) for v in resolution)
    n = int(eval_size)
    if n <= 0 or n > min(nx, ny, nz):
        raise ValueError(
            f"eval_size={n} must fit inside resolution={resolution}."
        )
    x0 = (nx - n) // 2
    y0 = (ny - n) // 2
    return slice(0, n), slice(y0, y0 + n), slice(x0, x0 + n)


def solve_low_lou_profile(n: int = 1, m: int = 1):
    """Solve the Low & Lou angular ODE and return interpolant plus eigenvalue."""

    def rhs(x, y, p):
        a2 = p[0]
        d2p = -(
            n * (n + 1) * y[0]
            + a2 * (1 + n) / n * y[0] * np.abs(y[0]) ** (2 / n)
        ) / (1 - x**2 + 1e-6)
        return [y[1], d2p]

    def boundary(pa, pb, _p):
        return np.array([pa[0], pb[0], pa[1] - 10.0])

    mu = np.linspace(-1, 1, num=256)
    init = (
        np.cos(mu * (m + 1) * np.pi / 2)
        if m % 2 == 0
        else np.sin(mu * (m + 1) * np.pi / 2)
    )
    initial = np.stack([init, 10.0 * np.ones_like(init)])

    def shooting(a2_init):
        sol = solve_bvp(
            rhs,
            boundary,
            x=mu,
            y=initial,
            p=[float(a2_init)],
            verbose=0,
            tol=1e-6,
        )
        return sol if sol.success else None

    candidates = [shooting(v) for v in np.linspace(0, 10, 100, dtype=np.float32)]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        raise RuntimeError("Failed to find valid Low & Lou BVP solutions.")

    # Reuse the converged BVP object. Retrying at a rounded eigenvalue can
    # fail for the n=3 family even though solve_bvp already returned a valid
    # solution at the unrounded value.
    final_sol = max(candidates, key=lambda candidate: float(candidate.p[0]))
    return final_sol.sol, float(final_sol.p[0])


def get_analytic_b_field(
    n: int = 1,
    m: int = 1,
    l: float = 0.3,
    psi: float = np.pi * 0.15,
    resolution: Tuple[int, int, int] = (128, 128, 128),
    bounds: Tuple[float, float, float, float, float, float] = (-1, 1, -1, 1, 0, 2),
    dtype: Any = np.float32,
) -> np.ndarray:
    """Generate analytic Low & Lou field as ``(Nx, Ny, Nz, 3)``."""
    field_dtype = np.dtype(dtype)
    if field_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Low--Lou field dtype must be float32 or float64, got {field_dtype}")
    sol_p, a2 = solve_low_lou_profile(n=n, m=m)
    nx, ny, nz = resolution
    x = np.linspace(bounds[0], bounds[1], nx, dtype=field_dtype)
    y = np.linspace(bounds[2], bounds[3], ny, dtype=field_dtype)
    z = np.linspace(bounds[4], bounds[5], nz, dtype=field_dtype)
    coords = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)

    xx, yy, zz = coords[..., 0], coords[..., 1], coords[..., 2]
    X = xx * np.cos(psi) - (zz + l) * np.sin(psi)
    Y = yy
    Z = xx * np.sin(psi) + (zz + l) * np.cos(psi)

    xy = X**2 + Y**2
    r = np.sqrt(xy + Z**2 + 1e-12)
    theta = np.arctan2(np.sqrt(xy), Z)
    phi = np.arctan2(Y, X)
    mu = np.cos(theta)

    P, dP_dmu = sol_p(mu)
    A = P / (r**n)
    dA_dtheta = -np.sin(theta) / (r**n) * dP_dmu
    dA_dr = P * (-n * r ** (-n - 1))
    Q = np.sqrt(a2) * A * np.abs(A) ** (1 / n)

    Br = dA_dtheta / (r**2 * np.sin(theta) + 1e-12)
    Btheta = -dA_dr / (r * np.sin(theta) + 1e-12)
    Bphi = Q / (r * np.sin(theta) + 1e-12)

    BX = (
        Br * np.sin(theta) * np.cos(phi)
        + Btheta * np.cos(theta) * np.cos(phi)
        - Bphi * np.sin(phi)
    )
    BY = (
        Br * np.sin(theta) * np.sin(phi)
        + Btheta * np.cos(theta) * np.sin(phi)
        + Bphi * np.cos(phi)
    )
    BZ = Br * np.cos(theta) - Btheta * np.sin(theta)

    Bx = BX * np.cos(psi) + BZ * np.sin(psi)
    By = BY
    Bz = -BX * np.sin(psi) + BZ * np.cos(psi)
    return np.real(np.stack([Bx, By, Bz], axis=-1)).astype(field_dtype, copy=False)


def build_lnl_reference_and_obs(
    *,
    resolution: Tuple[int, int, int] = (128, 128, 128),
    bounds: Tuple[float, float, float, float, float, float] = (-1, 1, -1, 1, 0, 2),
    pad: int | None = None,
    spacing_mode: Literal["bounds", "fortran"] = "bounds",
    ijzbcor: int = 3,
    reference_dtype: Any = np.float32,
    n: int = 1,
    m: int = 1,
    l: float = 0.3,
    psi: float = np.pi * 0.15,
) -> Dict[str, Any]:
    """Build Low & Lou reference fields, vector bottom observations, and ROI."""
    nx, ny, nz = resolution
    if pad is None:
        pad = lnl_buffer_thickness(resolution)
    if spacing_mode == "fortran":
        g = nfpt_fortran_grid(nx, ny, nz)
        dx, dy, dz = g.dx, g.dy, g.dz
        grid_bounds = (*g.x_bounds, *g.y_bounds, *g.z_bounds)
    elif spacing_mode == "bounds":
        dx = (bounds[1] - bounds[0]) / (nx - 1)
        dy = (bounds[3] - bounds[2]) / (ny - 1)
        dz = (bounds[5] - bounds[4]) / (nz - 1)
        grid_bounds = bounds
    else:
        raise ValueError(f"Unsupported spacing_mode {spacing_mode!r}")
    b_xyz = get_analytic_b_field(
        n=n,
        m=m,
        l=l,
        psi=psi,
        resolution=resolution,
        bounds=bounds,
        dtype=reference_dtype,
    )
    bx_ref = np.transpose(b_xyz[..., 0], (2, 1, 0)).astype(np.float64, copy=False)
    by_ref = np.transpose(b_xyz[..., 1], (2, 1, 0)).astype(np.float64, copy=False)
    bz_ref = np.transpose(b_xyz[..., 2], (2, 1, 0)).astype(np.float64, copy=False)
    jx_ref, jy_ref, jz_ref = curl_zyx_nfpt(bx_ref, by_ref, bz_ref, dx=dx, dy=dy, dz=dz)
    bx_obs = bx_ref[0].astype(np.float64, copy=False)
    by_obs = by_ref[0].astype(np.float64, copy=False)
    bz_obs = bz_ref[0].astype(np.float64, copy=False)
    if spacing_mode == "fortran":
        jz_obs = bottom_jz_fortran_readin_botbal(
            bx_obs,
            by_obs,
            dx=dx,
            dy=dy,
            ijzbcor=ijzbcor,
        )
    else:
        jz_obs = jz_ref[0].astype(np.float64, copy=False)
    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "bounds": grid_bounds,
        "spacing_mode": spacing_mode,
        "Bx_ref": bx_ref,
        "By_ref": by_ref,
        "Bz_ref": bz_ref,
        "Jx_ref": jx_ref,
        "Jy_ref": jy_ref,
        "Jz_ref": jz_ref,
        "Bx_obs": bx_obs,
        "By_obs": by_obs,
        "Bz_obs": bz_obs,
        "Jz_obs": jz_obs,
        "metric_slices": metric_roi_slices_zyx(nz, ny, nx, pad=pad),
        "buffer_thickness": int(pad),
        "low_lou": {
            "n": int(n),
            "m": int(m),
            "l": float(l),
            "psi": float(psi),
        },
    }


def build_lnl_jarolim_c2_reference_and_obs() -> Dict[str, Any]:
    """Build the Low--Lou C2 configuration used by Jarolim et al. (2023)."""
    ref = build_lnl_reference_and_obs(
        resolution=JAROLIM_C2_RESOLUTION,
        bounds=JAROLIM_C2_BOUNDS,
        pad=0,
        spacing_mode="bounds",
        reference_dtype=np.float64,
    )
    ref["metric_slices"] = jarolim_c2_metric_slices_zyx()
    ref["jarolim_c2"] = {
        "n": 1,
        "m": 1,
        "l": 0.3,
        "psi": float(np.pi * 0.15),
        "resolution_xyz": JAROLIM_C2_RESOLUTION,
        "evaluation_shape_zyx": (64, 64, 64),
        "analytic_reference_dtype": "float64",
        "boundary_condition": "bottom analytic vector field; potential side and top boundaries",
    }
    return ref


def build_lnl_family_reference_and_obs(case_name: str) -> Dict[str, Any]:
    """Build one frozen member of the paper Low--Lou benchmark family."""
    key = str(case_name).lower().replace("-", "_")
    if key not in LOW_LOU_FAMILY:
        raise ValueError(
            f"Unknown Low--Lou case {case_name!r}; choose from {sorted(LOW_LOU_FAMILY)}"
        )
    if key == "low_lou_c2":
        return build_lnl_jarolim_c2_reference_and_obs()
    parameters = LOW_LOU_FAMILY[key]
    ref = build_lnl_reference_and_obs(
        resolution=JAROLIM_C2_RESOLUTION,
        bounds=JAROLIM_C2_BOUNDS,
        pad=0,
        spacing_mode="bounds",
        reference_dtype=np.float64,
        **parameters,
    )
    ref["metric_slices"] = jarolim_c2_metric_slices_zyx()
    ref["benchmark_family"] = {
        "case_name": key,
        **parameters,
        "resolution_xyz": JAROLIM_C2_RESOLUTION,
        "evaluation_shape_zyx": (64, 64, 64),
        "analytic_reference_dtype": "float64",
        "boundary_condition": (
            "bottom analytic vector field; potential side and top boundaries"
        ),
    }
    return ref
