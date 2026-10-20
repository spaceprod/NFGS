"""Metric and region-of-interest helpers shared by NFPT notebooks."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def flux_rope_center_from_bottom_jz(Jz_bottom: np.ndarray) -> Tuple[int, int]:
    """Return (iy, ix) from the |Jz| centroid on the bottom plane."""
    w = np.abs(np.asarray(Jz_bottom, dtype=np.float64))
    s = float(w.sum()) + 1e-30
    nj, ni = w.shape
    jj, ii = np.meshgrid(
        np.arange(nj, dtype=np.float64), np.arange(ni, dtype=np.float64), indexing="ij"
    )
    iy = int(np.clip(np.round(np.sum(w * jj) / s), 0, nj - 1))
    ix = int(np.clip(np.round(np.sum(w * ii) / s), 0, ni - 1))
    return iy, ix


def metric_roi_slices_zyx(nz: int, ny: int, nx: int, pad: int) -> Tuple[slice, slice, slice]:
    """Interior (z, y, x) slices excluding a pad-cell border."""
    pad = int(pad)
    if pad <= 0:
        return slice(None), slice(None), slice(None)
    nz, ny, nx = int(nz), int(ny), int(nx)
    z_s = slice(0, nz - pad) if nz > pad else slice(None)
    y_s = slice(pad, ny - pad) if ny > 2 * pad else slice(None)
    x_s = slice(pad, nx - pad) if nx > 2 * pad else slice(None)
    return z_s, y_s, x_s


def test_domain_slices_zyx(
    nz: int,
    ny: int,
    nx: int,
    iy_c: int,
    ix_c: int,
    n_test_x: int,
    n_test_y: int,
    n_test_z: int,
) -> Tuple[slice, slice, slice]:
    """Restricted analysis volume centered on a photospheric feature."""
    nz, ny, nx = int(nz), int(ny), int(nx)
    n_test_x = min(int(n_test_x), nx)
    n_test_y = min(int(n_test_y), ny)
    n_test_z = min(int(n_test_z), nz)
    z_s = slice(0, n_test_z)
    y0 = int(np.clip(iy_c - n_test_y // 2, 0, ny - n_test_y))
    x0 = int(np.clip(ix_c - n_test_x // 2, 0, nx - n_test_x))
    return z_s, slice(y0, y0 + n_test_y), slice(x0, x0 + n_test_x)


def vector_correlation(pred: np.ndarray, ref: np.ndarray) -> float:
    """Global vector correlation between two vector arrays."""
    num = float(np.sum(pred * ref))
    den = float(np.sqrt(np.sum(pred**2) * np.sum(ref**2)) + 1e-12)
    return num / den


def roi_cwsin(
    B_vec: np.ndarray,
    J_vec: np.ndarray,
    *,
    b2min: float = 1e-10,
) -> float:
    """
    Current-weighted sine on a pre-sliced vector volume.

    Matches ``eval_nfpt_TD2.ipynb``: sum(|J x B| / |B|) / sum(|J|) on the
    restricted test domain (66 x 66 x 30 flux-rope-centered ROI), without
    the Fortran interior edge exclusions used by ``fortran_cwsin``.
    """
    B = np.asarray(B_vec, dtype=np.float64)
    J = np.asarray(J_vec, dtype=np.float64)
    if B.shape != J.shape or B.ndim != 4 or B.shape[0] != 3:
        raise ValueError(
            f"B_vec/J_vec must both have shape (3, Nz, Ny, Nx); got {B.shape}, {J.shape}"
        )
    bnorm = np.sqrt(np.sum(B * B, axis=0))
    jnorm = np.sqrt(np.sum(J * J, axis=0))
    jxb = np.cross(np.moveaxis(J, 0, -1), np.moveaxis(B, 0, -1), axis=-1)
    numerator = np.sum(np.linalg.norm(jxb, axis=-1) / np.maximum(bnorm, float(b2min)))
    denominator = np.sum(jnorm)
    return float(numerator / max(float(denominator), 1e-30))


def fortran_cwsin(
    B_vec: np.ndarray,
    J_vec: np.ndarray,
    *,
    b2min: float = 1e-20,
) -> float:
    """
    Current-weighted sine using the same definition as NFPT Fortran ``CWsin``.

    ``B_vec`` and ``J_vec`` are component-first arrays ``(3, Nz, Ny, Nx)`` on the
    observed/cropped domain. The convergence-check volume follows
    ``nfpt_1.0.2.f90``: z starts at plane 1 and excludes the upper quarter;
    x/y exclude one eighth of the observed domain, with the same endpoint
    convention used by the Fortran loop bounds.
    """
    B = np.asarray(B_vec, dtype=np.float64)
    J = np.asarray(J_vec, dtype=np.float64)
    if B.shape != J.shape or B.ndim != 4 or B.shape[0] != 3:
        raise ValueError(f"B_vec/J_vec must both have shape (3, Nz, Ny, Nx); got {B.shape}, {J.shape}")

    _, nz, ny, nx = B.shape
    abfx = nx // 8
    abfy = ny // 8
    abfz = nz // 4
    sz = slice(1, max(1, nz - abfz))
    sy = slice(abfy, max(abfy + 1, ny - abfy + 1))
    sx = slice(abfx, max(abfx + 1, nx - abfx + 1))

    b = B[:, sz, sy, sx]
    j = J[:, sz, sy, sx]
    bnorm = np.sqrt(np.sum(b * b, axis=0))
    jnorm = np.sqrt(np.sum(j * j, axis=0))
    jxb = np.cross(np.moveaxis(j, 0, -1), np.moveaxis(b, 0, -1), axis=-1)
    numerator = np.sum(np.linalg.norm(jxb, axis=-1) / np.maximum(bnorm, np.sqrt(float(b2min))))
    denominator = np.sum(jnorm)
    return float(numerator / max(float(denominator), 1e-30))


def compute_reference_metrics(
    ref_fields: Dict[str, np.ndarray],
    fields: Dict[str, np.ndarray],
    metric_slices: Tuple[slice, slice, slice],
) -> Dict[str, float]:
    """Reference-field agreement and force-freeness metrics on a zyx ROI.

    All reported scalars, including ``sigma_J``, are evaluated on
    ``metric_slices`` (the flux-rope-centered test domain used by
    ``eval_nfpt_TD2.ipynb``).
    """
    sz, sy, sx = metric_slices
    Bref_vec = np.stack(
        [ref_fields["Bx"], ref_fields["By"], ref_fields["Bz"]], axis=0
    )[:, sz, sy, sx]
    Bsol_vec = np.stack([fields["Bx"], fields["By"], fields["Bz"]], axis=0)[:, sz, sy, sx]

    if not np.all(np.isfinite(Bref_vec)) or not np.all(np.isfinite(Bsol_vec)):
        raise RuntimeError("Non-finite values in B fields for metrics.")

    B_i = Bref_vec
    b_i = Bsol_vec
    Cvec = vector_correlation(b_i, B_i)

    Bi_norm = np.sqrt(np.sum(B_i**2, axis=0))
    bi_norm = np.sqrt(np.sum(b_i**2, axis=0))
    dot = np.sum(B_i * b_i, axis=0)
    cos_theta = dot / (Bi_norm * bi_norm + 1e-12)
    CCS = float(np.mean(cos_theta))

    diff_norm = np.sqrt(np.sum((b_i - B_i) ** 2, axis=0))
    En = float(np.sum(diff_norm) / (np.sum(Bi_norm) + 1e-12))
    Em = float(np.mean(diff_norm / (Bi_norm + 1e-12)))
    epsilon = float(np.sum(b_i**2) / (np.sum(B_i**2) + 1e-12))

    J_vec = np.stack([fields["Jx"], fields["Jy"], fields["Jz"]], axis=0)
    J_roi = J_vec[:, sz, sy, sx]
    if not np.all(np.isfinite(J_roi)):
        raise RuntimeError("Non-finite values in J for sigma_J.")
    sigma_J = roi_cwsin(b_i, J_roi)

    return {
        "C_vec": Cvec,
        "C_CS": CCS,
        "En": En,
        "Em": Em,
        "Eprime_n": 1.0 - En,
        "Eprime_m": 1.0 - Em,
        "epsilon": epsilon,
        "sigma_J": sigma_J,
    }
