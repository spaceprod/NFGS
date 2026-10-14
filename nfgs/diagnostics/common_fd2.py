"""Method-independent second-order B-to-J and divergence diagnostics."""

from __future__ import annotations

import numpy as np

Spacing = float | tuple[float, float, float]


def _spacing_xyz(spacing: Spacing) -> tuple[float, float, float]:
    """Accept an isotropic scalar or physical spacings ordered x, y, z."""
    if np.isscalar(spacing):
        values = (float(spacing),) * 3
    else:
        values = tuple(float(value) for value in spacing)
    if len(values) != 3 or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("spacing must be positive: a scalar or (dx, dy, dz)")
    return values


def derivative(values: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    array = np.asarray(values)
    moved = np.moveaxis(array, axis, 0)
    out = np.empty_like(moved)
    n = moved.shape[0]
    if n < 3:
        if n == 1:
            out[...] = 0
        else:
            out[0] = (moved[1] - moved[0]) / spacing
            out[1] = out[0]
        return np.moveaxis(out, 0, axis)
    out[1:-1] = (moved[2:] - moved[:-2]) / (2.0 * spacing)
    out[0] = (-3.0 * moved[0] + 4.0 * moved[1] - moved[2]) / (2.0 * spacing)
    out[-1] = (3.0 * moved[-1] - 4.0 * moved[-2] + moved[-3]) / (2.0 * spacing)
    return np.moveaxis(out, 0, axis)


def curl(fields: dict[str, np.ndarray], spacing: Spacing) -> dict[str, np.ndarray]:
    """Curl of z-y-x arrays using scalar spacing or (dx, dy, dz)."""
    dx, dy, dz = _spacing_xyz(spacing)
    bx, by, bz = fields["Bx"], fields["By"], fields["Bz"]
    return {
        "Jx": derivative(bz, dy, 1) - derivative(by, dz, 0),
        "Jy": derivative(bx, dz, 0) - derivative(bz, dx, 2),
        "Jz": derivative(by, dx, 2) - derivative(bx, dy, 1),
    }


def divergence(fields: dict[str, np.ndarray], spacing: Spacing) -> np.ndarray:
    """Divergence using each axis's physical spacing on z-y-x arrays."""
    dx, dy, dz = _spacing_xyz(spacing)
    return (
        derivative(fields["Bx"], dx, 2)
        + derivative(fields["By"], dy, 1)
        + derivative(fields["Bz"], dz, 0)
    )


def sigma_j(
    fields: dict[str, np.ndarray],
    spacing: Spacing,
    slices: tuple[slice, slice, slice],
) -> float:
    current = curl(fields, spacing)
    sz, sy, sx = slices
    b = np.stack(
        [fields[key][sz, sy, sx] for key in ("Bx", "By", "Bz")], axis=-1
    ).astype(np.float64)
    j = np.stack(
        [current[key][sz, sy, sx] for key in ("Jx", "Jy", "Jz")], axis=-1
    ).astype(np.float64)
    bn = np.linalg.norm(b, axis=-1)
    numerator = np.sum(
        np.linalg.norm(np.cross(j, b), axis=-1) / np.maximum(bn, 1.0e-10)
    )
    denominator = np.sum(np.linalg.norm(j, axis=-1))
    return float(numerator / max(float(denominator), 1.0e-30))
