"""Six-face-potential relative-helicity diagnostics on a finite Cartesian box.

The reference potential is reconstructed from the normal field on all six
faces.  Vector potentials for the field and its potential counterpart are
then built in one common DeVore gauge (``A_z=0``).  This makes the reported
relative helicity and current-carrying helicity internally comparable while
keeping the gauge and its discrete curl residual explicit in the output.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from nfgs.diagnostics.volume_diagnostics import _fd2_axis


def _devore_vector_potential(field: np.ndarray, spacing: float) -> tuple[np.ndarray, dict[str, float]]:
    """Construct a DeVore-gauge vector potential for a ``(3,z,y,x)`` field."""
    bx, by, bz = (np.asarray(component, dtype=np.float64) for component in field)
    nz, ny, nx = bz.shape
    base = bz[0]
    source_mean = float(np.mean(base))
    source = base - source_mean
    lam_y = (2.0 * np.cos(np.pi * np.arange(ny) / ny) - 2.0) / spacing**2
    lam_x = (2.0 * np.cos(np.pi * np.arange(nx) / nx) - 2.0) / spacing**2
    denominator = lam_y[:, None] + lam_x[None, :]
    source_hat = dctn(source, type=2, norm="ortho")
    phi_hat = np.zeros_like(source_hat)
    mask = denominator != 0.0
    phi_hat[mask] = source_hat[mask] / denominator[mask]
    phi = idctn(phi_hat, type=2, norm="ortho")
    ax0 = -_fd2_axis(phi, spacing, axis=0)
    ay0 = _fd2_axis(phi, spacing, axis=1)

    ax = np.empty_like(bx)
    ay = np.empty_like(by)
    ax[0] = ax0
    ay[0] = ay0
    if nz > 1:
        ax[1:] = ax0 + np.cumsum(0.5 * spacing * (by[:-1] + by[1:]), axis=0)
        ay[1:] = ay0 - np.cumsum(0.5 * spacing * (bx[:-1] + bx[1:]), axis=0)
    az = np.zeros_like(bz)
    vector_potential = np.stack((ax, ay, az), axis=0)
    curl = np.stack(
        (
            -_fd2_axis(ay, spacing, axis=0),
            _fd2_axis(ax, spacing, axis=0),
            _fd2_axis(ay, spacing, axis=2) - _fd2_axis(ax, spacing, axis=1),
        ),
        axis=0,
    )
    mismatch = curl - field
    return vector_potential, {
        "devore_bottom_bz_mean_removed": source_mean,
        "devore_curl_relative_rms": float(
            np.sqrt(np.mean(mismatch * mismatch))
            / max(np.sqrt(np.mean(field * field)), 1.0e-30)
        ),
    }


def relative_helicity_diagnostics(
    field: np.ndarray, potential: np.ndarray, spacing: float
) -> dict[str, float | str]:
    """Return relative and current-carrying helicity for a common six-face potential.

    ``field`` and ``potential`` must be collocated arrays in ``(3,z,y,x)``
    order.  The current-carrying fraction is reported both signed and absolute;
    the latter is the commonly used eruptivity-oriented ratio.
    """
    b = np.asarray(field, dtype=np.float64)
    bp = np.asarray(potential, dtype=np.float64)
    if b.shape != bp.shape or b.ndim != 4 or b.shape[0] != 3:
        raise ValueError("field and potential must have matching (3,z,y,x) shapes")
    a, a_meta = _devore_vector_potential(b, spacing)
    ap, ap_meta = _devore_vector_potential(bp, spacing)
    bc = b - bp
    dv = float(spacing) ** 3
    h_rel = float(np.sum((a + ap) * bc)) * dv
    h_current = float(np.sum((a - ap) * bc)) * dv
    normal_mismatch = np.concatenate(
        (
            bc[0, :, :, 0].ravel(), bc[0, :, :, -1].ravel(),
            bc[1, :, 0, :].ravel(), bc[1, :, -1, :].ravel(),
            bc[2, 0, :, :].ravel(), bc[2, -1, :, :].ravel(),
        )
    )
    field_rms = max(float(np.sqrt(np.mean(b * b))), 1.0e-30)
    return {
        "relative_helicity_6face": h_rel,
        "current_carrying_helicity_6face": h_current,
        "current_carrying_helicity_fraction_signed": h_current / max(abs(h_rel), 1.0e-30),
        "current_carrying_helicity_fraction_abs": abs(h_current) / max(abs(h_rel), 1.0e-30),
        "six_face_normal_mismatch_relative_rms": float(
            np.sqrt(np.mean(normal_mismatch * normal_mismatch)) / field_rms
        ),
        "helicity_gauge": "common_devore_Az_zero",
        **{f"field_{key}": value for key, value in a_meta.items()},
        **{f"potential_{key}": value for key, value in ap_meta.items()},
    }
