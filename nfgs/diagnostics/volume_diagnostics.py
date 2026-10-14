"""Common six-face energy and Helmholtz divergence diagnostics.

The potential field follows the repository's FLHTOOLS-compatible six-face
Neumann construction.  The divergence-energy quantity is explicitly named a
Helmholtz diagnostic; it is not presented as the full Valori et al. cross-term
decomposition.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from nfgs._backend.util.flh_potential_gpu import compute_potential_field_flh_numpy


def cell_center_potential(field: dict[str, np.ndarray], spacing: float) -> tuple[np.ndarray, float]:
    """Return the common six-face potential at the input field's nodes."""
    potential = compute_potential_field_flh_numpy(
        field["Bx"], field["By"], field["Bz"],
        dx=float(spacing), dy=float(spacing), dz=float(spacing),
    )
    bpx = np.asarray(potential.Bpx, dtype=np.float64)
    bpy = np.asarray(potential.Bpy, dtype=np.float64)
    bpz = np.asarray(potential.Bpz, dtype=np.float64)
    bx = 0.25 * (
        bpx[:-1, :-1, :] + bpx[1:, :-1, :]
        + bpx[:-1, 1:, :] + bpx[1:, 1:, :]
    )
    by = 0.25 * (
        bpy[:-1, :, :-1] + bpy[1:, :, :-1]
        + bpy[:-1, :, 1:] + bpy[1:, :, 1:]
    )
    bz = 0.25 * (
        bpz[:, :-1, :-1] + bpz[:, 1:, :-1]
        + bpz[:, :-1, 1:] + bpz[:, 1:, 1:]
    )
    return np.stack((bx, by, bz), axis=0), float(potential.flux_imbalance)


def _fd2_axis(values: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    result = np.empty_like(moved)
    n = moved.shape[0]
    if n < 3:
        result[...] = 0.0 if n == 1 else (moved[-1] - moved[0]) / spacing
        return np.moveaxis(result, 0, axis)
    result[1:-1] = (moved[2:] - moved[:-2]) / (2.0 * spacing)
    result[0] = (-3.0 * moved[0] + 4.0 * moved[1] - moved[2]) / (2.0 * spacing)
    result[-1] = (3.0 * moved[-1] - 4.0 * moved[-2] + moved[-3]) / (2.0 * spacing)
    return np.moveaxis(result, 0, axis)


def helmholtz_non_solenoidal_component(
    field: dict[str, np.ndarray], spacing: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Return grad(psi) from a cell-centred Neumann Poisson projection.

    The mean source is projected out to satisfy Neumann compatibility and is
    reported.  DCT-II eigenvalues match the second-difference Neumann operator.
    """
    bx = np.asarray(field["Bx"], dtype=np.float64)
    by = np.asarray(field["By"], dtype=np.float64)
    bz = np.asarray(field["Bz"], dtype=np.float64)
    divergence = (
        _fd2_axis(bx, spacing, 2)
        + _fd2_axis(by, spacing, 1)
        + _fd2_axis(bz, spacing, 0)
    )
    mean_source = float(np.mean(divergence))
    source = divergence - mean_source
    nz, ny, nx = source.shape
    lam_z = (2.0 * np.cos(np.pi * np.arange(nz) / nz) - 2.0) / spacing**2
    lam_y = (2.0 * np.cos(np.pi * np.arange(ny) / ny) - 2.0) / spacing**2
    lam_x = (2.0 * np.cos(np.pi * np.arange(nx) / nx) - 2.0) / spacing**2
    denominator = lam_z[:, None, None] + lam_y[None, :, None] + lam_x[None, None, :]
    source_hat = dctn(source, type=2, norm="ortho")
    psi_hat = np.zeros_like(source_hat)
    mask = denominator != 0.0
    psi_hat[mask] = source_hat[mask] / denominator[mask]
    psi = idctn(psi_hat, type=2, norm="ortho")
    gradient = np.stack(
        (
            _fd2_axis(psi, spacing, 2),
            _fd2_axis(psi, spacing, 1),
            _fd2_axis(psi, spacing, 0),
        ),
        axis=0,
    )
    return gradient, {
        "helmholtz_projected_divergence_mean": mean_source,
        "helmholtz_source_rms": float(np.sqrt(np.mean(source**2))),
    }


def energy_diagnostics(
    field: dict[str, np.ndarray],
    spacing: float,
    *,
    slices: tuple[slice, slice, slice] = (slice(None), slice(None), slice(None)),
    potential: np.ndarray | None = None,
    non_solenoidal: np.ndarray | None = None,
    flux_imbalance: float | None = None,
) -> dict[str, float | str]:
    """Compute common total/free and Helmholtz non-solenoidal energies."""
    if potential is None:
        potential, flux_value = cell_center_potential(field, spacing)
        flux_imbalance = flux_value
    projection_meta: dict[str, float] = {}
    if non_solenoidal is None:
        non_solenoidal, projection_meta = helmholtz_non_solenoidal_component(field, spacing)
    sz, sy, sx = slices
    b = np.stack(
        [np.asarray(field[key][sz, sy, sx], dtype=np.float64) for key in ("Bx", "By", "Bz")],
        axis=0,
    )
    bp = np.asarray(potential[:, sz, sy, sx], dtype=np.float64)
    bns = np.asarray(non_solenoidal[:, sz, sy, sx], dtype=np.float64)
    dv = float(spacing) ** 3
    total = 0.5 * float(np.sum(b * b)) * dv
    potential_energy = 0.5 * float(np.sum(bp * bp)) * dv
    non_solenoidal_energy = 0.5 * float(np.sum(bns * bns)) * dv
    return {
        "total_energy_common": total,
        "potential_energy_6face": potential_energy,
        "free_energy_6face": total - potential_energy,
        "free_energy_over_potential": (total - potential_energy) / max(potential_energy, 1.0e-30),
        "helmholtz_non_solenoidal_energy": non_solenoidal_energy,
        "helmholtz_non_solenoidal_energy_fraction": non_solenoidal_energy / max(total, 1.0e-30),
        "potential_flux_imbalance": float(flux_imbalance or 0.0),
        "potential_method": "six_face_flh_neumann",
        "divergence_energy_method": "cell_centered_neumann_helmholtz_dct2",
        **projection_meta,
    }
