"""TD2 field-line seed helpers for paper-style PyVista figures."""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np
from scipy import ndimage

from .field_ops import to_numpy

CentroidMode = Literal["jz_weighted", "jz_blob_geometric", "bz_blob_geometric"]
CurrentAxisSelection = Literal["jz_blob_geometric", "origin_symmetric_trace"]

# Bald-patch circle radius in grid-cell units (1/3 of the previous 6.5 px scale).
BALD_PATCH_RADIUS_CELLS = 6.5 / 3.0


def _physical_origin_pyvista_xy(x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]) -> Tuple[float, float]:
    """Map physical photosphere origin (0, 0) to PyVista seed coordinates."""
    return (-float(x_bounds[0]), -float(y_bounds[0]))


def _index_to_pyvista_xyz(iy: float, ix: float, *, dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    z = 0.5 * float(dz)
    return (float(ix * dx), float(iy * dy), z)


def _bottom_jz_region_center_ij(bottom_jz: np.ndarray, *, sign: int) -> tuple[float, float]:
    """Return the Jz-weighted centroid (iy, ix) for Jz>0 or Jz<0 bottom regions."""
    jz = np.asarray(to_numpy(bottom_jz), dtype=np.float64)
    if sign > 0:
        weight = np.where(jz > 0.0, jz, 0.0)
    elif sign < 0:
        weight = np.where(jz < 0.0, -jz, 0.0)
    else:
        raise ValueError("sign must be +1 or -1")
    total = float(weight.sum())
    if total <= 0.0:
        flat_idx = int(np.argmax(jz if sign > 0 else -jz))
        iy, ix = np.unravel_index(flat_idx, jz.shape)
        return float(iy), float(ix)
    iy_grid, ix_grid = np.indices(jz.shape, dtype=np.float64)
    iy = float((weight * iy_grid).sum() / total)
    ix = float((weight * ix_grid).sum() / total)
    return iy, ix


def _signed_blob_mask(
    bottom_field: np.ndarray,
    *,
    sign: int,
    fraction_of_peak: float | None = 0.35,
) -> np.ndarray:
    """Boolean mask for the largest signed blob on the bottom plane."""
    field = np.asarray(to_numpy(bottom_field), dtype=np.float64)
    if sign > 0:
        mask = field > 0.0
        if fraction_of_peak is not None and fraction_of_peak > 0.0:
            peak = float(field.max())
            if peak <= 0.0:
                return mask
            mask = mask & (field >= float(fraction_of_peak) * peak)
    elif sign < 0:
        mask = field < 0.0
        if fraction_of_peak is not None and fraction_of_peak > 0.0:
            peak = float(-field.min())
            if peak <= 0.0:
                return mask
            mask = mask & (field <= -float(fraction_of_peak) * peak)
    else:
        raise ValueError("sign must be +1 or -1")
    if not np.any(mask):
        return mask

    labeled, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
    label = 1 + int(np.argmax(sizes))
    return labeled == label


def _bottom_signed_blob_geometric_center_ij(
    bottom_field: np.ndarray,
    *,
    sign: int,
    fraction_of_peak: float | None = 0.35,
) -> tuple[float, float]:
    """Return the unweighted geometric center of the largest signed blob."""
    blob = _signed_blob_mask(bottom_field, sign=sign, fraction_of_peak=fraction_of_peak)
    field = np.asarray(to_numpy(bottom_field), dtype=np.float64)
    if not np.any(blob):
        return _bottom_jz_region_center_ij(field, sign=sign)

    iy_grid, ix_grid = np.indices(field.shape, dtype=np.float64)
    iy = float((blob * iy_grid).sum() / blob.sum())
    ix = float((blob * ix_grid).sum() / blob.sum())
    return iy, ix


def _mirror_xy_pv(x: float, y: float, *, cx: float, cy: float) -> tuple[float, float]:
    return (2.0 * cx - float(x), 2.0 * cy - float(y))


def _as_seed_points(seed_points) -> np.ndarray:
    arr = np.asarray(to_numpy(seed_points), dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected seed points with shape (N, 3), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def mirror_current_axis_seed(
    pos_seed: np.ndarray,
    *,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> np.ndarray:
    """Return the origin-symmetric partner of a magenta current-axis seed."""
    cx, cy = _physical_origin_pyvista_xy(x_bounds, y_bounds)
    arr = _as_seed_points(pos_seed)
    if arr.shape[0] != 1:
        raise ValueError("Expected one magenta seed point with shape (1, 3)")
    x, y, z = (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2]))
    x_m, y_m = _mirror_xy_pv(x, y, cx=cx, cy=cy)
    return np.asarray([[x_m, y_m, z]], dtype=np.float32)


def snap_seed_to_physical_x0_line(
    point,
    *,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    dx: float,
    dy: float,
    dz: float,
    ny: int,
) -> np.ndarray:
    """Snap a picked point to physical x=0 on the bottom plane (z = 0.5 * dz)."""
    cx, _ = _physical_origin_pyvista_xy(x_bounds, y_bounds)
    z = 0.5 * float(dz)
    y = float(np.asarray(point, dtype=float).reshape(3)[1])
    y = float(np.clip(y, 0.0, float(ny - 1) * float(dy)))
    return np.asarray([[cx, y, z]], dtype=np.float32)


def _invert_points_about_origin(points: np.ndarray, *, cx: float, cy: float) -> np.ndarray:
    """Invert 3D points through the physical photosphere origin."""
    out = np.asarray(points, dtype=np.float64).copy()
    if out.size == 0:
        return out.reshape(0, 3)
    out[:, 0] = 2.0 * cx - out[:, 0]
    out[:, 1] = 2.0 * cy - out[:, 1]
    out[:, 2] = -out[:, 2]
    return out


def _line_origin_symmetry_mismatch(pos_pts: np.ndarray, neg_pts: np.ndarray, *, cx: float, cy: float) -> float:
    """Mean bidirectional point-to-polyline distance after origin inversion."""
    pos_pts = np.asarray(pos_pts, dtype=np.float64)
    neg_pts = np.asarray(neg_pts, dtype=np.float64)
    if pos_pts.shape[0] < 2 or neg_pts.shape[0] < 2:
        return float("inf")
    pos_inv = _invert_points_about_origin(pos_pts, cx=cx, cy=cy)
    from scipy.spatial import cKDTree

    tree_neg = cKDTree(neg_pts)
    tree_pos_inv = cKDTree(pos_inv)
    d_pos_to_neg, _ = tree_neg.query(pos_inv)
    d_neg_to_pos, _ = tree_pos_inv.query(neg_pts)
    return float(d_pos_to_neg.mean() + d_neg_to_pos.mean())


def _symmetric_blob_seed_candidates(
    pos_blob: np.ndarray,
    neg_blob: np.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
    cx: float,
    cy: float,
    sample_stride: int = 4,
    max_candidates: int = 160,
) -> np.ndarray:
    """Return origin-symmetric seed pairs (pos, neg) with z = 0.5 * dz."""
    z = 0.5 * float(dz)
    iy_idx, ix_idx = np.nonzero(pos_blob)
    if sample_stride > 1:
        iy_idx = iy_idx[::sample_stride]
        ix_idx = ix_idx[::sample_stride]
    ny, nx = pos_blob.shape
    points: list[tuple[float, float, float]] = []
    for iy, ix in zip(iy_idx, ix_idx):
        x = float(ix * dx)
        y = float(iy * dy)
        x_m, y_m = _mirror_xy_pv(x, y, cx=cx, cy=cy)
        ix_m = int(round(x_m / dx))
        iy_m = int(round(y_m / dy))
        if ix_m < 0 or ix_m >= nx or iy_m < 0 or iy_m >= ny:
            continue
        if not neg_blob[iy_m, ix_m]:
            continue
        points.append((x, y, z))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape[0] > max_candidates:
        step = max(1, int(np.ceil(arr.shape[0] / max_candidates)))
        arr = arr[::step]
    return arr


def find_origin_symmetric_current_axis_seeds(
    case: dict,
    vector_field: np.ndarray,
    *,
    blob_fraction_of_peak: float = 0.35,
    max_z_planes: int | None = None,
    sample_stride: int = 4,
    max_candidates: int = 160,
    max_steps: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick magenta/blue seeds whose traced lines best match under origin inversion."""
    import pyvista as pv

    from .pyvista_viz import trace_lines, vector_grid_from_zyx

    _, _, _, dx, dy, dz = case["dims"]
    x_bounds = tuple(case["x_bounds"])
    y_bounds = tuple(case["y_bounds"])
    cx, cy = _physical_origin_pyvista_xy(x_bounds, y_bounds)
    bottom_jz = case["Jz_obs"]

    pos_blob = _signed_blob_mask(bottom_jz, sign=1, fraction_of_peak=blob_fraction_of_peak)
    neg_blob = _signed_blob_mask(bottom_jz, sign=-1, fraction_of_peak=blob_fraction_of_peak)
    candidates = _symmetric_blob_seed_candidates(
        pos_blob,
        neg_blob,
        dx=dx,
        dy=dy,
        dz=dz,
        cx=cx,
        cy=cy,
        sample_stride=sample_stride,
        max_candidates=max_candidates,
    )
    if candidates.size == 0:
        (iy_pos, ix_pos), (iy_neg, ix_neg) = _symmetric_current_axis_centers_ij(
            bottom_jz,
            case["boundary"]["Bz"],
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            dx=dx,
            dy=dy,
            centroid_mode="jz_blob_geometric",
            blob_fraction_of_peak=blob_fraction_of_peak,
        )
        return (
            _current_axis_seed_from_ij(iy_pos, ix_pos, dx=dx, dy=dy, dz=dz),
            _current_axis_seed_from_ij(iy_neg, ix_neg, dx=dx, dy=dy, dz=dz),
        )

    pv.OFF_SCREEN = True
    grid = vector_grid_from_zyx(
        pv,
        {"B_seed": vector_field},
        dx=dx,
        dy=dy,
        dz=dz,
        max_z_planes=max_z_planes,
    )
    best_score = float("inf")
    best_pos = candidates[0:1]
    best_neg = np.asarray(
        [[*_mirror_xy_pv(best_pos[0, 0], best_pos[0, 1], cx=cx, cy=cy), best_pos[0, 2]]],
        dtype=np.float32,
    )
    for seed in candidates:
        pos_seed = seed.reshape(1, 3)
        neg_xy = _mirror_xy_pv(float(seed[0]), float(seed[1]), cx=cx, cy=cy)
        neg_seed = np.asarray([[neg_xy[0], neg_xy[1], float(seed[2])]], dtype=np.float32)
        pos_line = trace_lines(pv, grid, "B_seed", pos_seed, max_steps=max_steps)
        neg_line = trace_lines(pv, grid, "B_seed", neg_seed, max_steps=max_steps)
        score = _line_origin_symmetry_mismatch(
            np.asarray(pos_line.points),
            np.asarray(neg_line.points),
            cx=cx,
            cy=cy,
        )
        if score < best_score:
            best_score = score
            best_pos = pos_seed.copy()
            best_neg = neg_seed.copy()

    return best_pos.astype(np.float32), best_neg.astype(np.float32)


def _td2_physical_origin_ij(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    *,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    """Grid indices (iy, ix) for the TD2 photosphere origin (0, 0)."""
    iy0 = -float(y_bounds[0]) / float(dy)
    ix0 = -float(x_bounds[0]) / float(dx)
    return iy0, ix0


def _symmetric_current_axis_centers_ij(
    bottom_jz: np.ndarray,
    bottom_bz: np.ndarray | None,
    *,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    dx: float,
    dy: float,
    centroid_mode: CentroidMode = "jz_blob_geometric",
    blob_fraction_of_peak: float = 0.35,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return origin-symmetric (iy, ix) centers for positive and negative blobs."""
    iy0, ix0 = _td2_physical_origin_ij(x_bounds, y_bounds, dx=dx, dy=dy)
    iy_pos, ix_pos = _current_axis_center_ij(
        bottom_jz,
        bottom_bz,
        sign=1,
        centroid_mode=centroid_mode,
        blob_fraction_of_peak=blob_fraction_of_peak,
    )
    iy_neg, ix_neg = _current_axis_center_ij(
        bottom_jz,
        bottom_bz,
        sign=-1,
        centroid_mode=centroid_mode,
        blob_fraction_of_peak=blob_fraction_of_peak,
    )
    d_iy = 0.5 * (abs(iy_pos - iy0) + abs(iy_neg - iy0))
    d_ix = 0.5 * (abs(ix_pos - ix0) + abs(ix_neg - ix0))
    sign_iy = 1.0 if iy_pos >= iy0 else -1.0
    sign_ix = 1.0 if ix_pos >= ix0 else -1.0
    return (
        (iy0 + sign_iy * d_iy, ix0 + sign_ix * d_ix),
        (iy0 - sign_iy * d_iy, ix0 - sign_ix * d_ix),
    )


def _current_axis_center_ij(
    bottom_jz: np.ndarray,
    bottom_bz: np.ndarray | None,
    *,
    sign: int,
    centroid_mode: CentroidMode = "jz_blob_geometric",
    blob_fraction_of_peak: float = 0.35,
) -> tuple[float, float]:
    if centroid_mode == "jz_weighted":
        return _bottom_jz_region_center_ij(bottom_jz, sign=sign)
    if centroid_mode == "jz_blob_geometric":
        return _bottom_signed_blob_geometric_center_ij(
            bottom_jz,
            sign=sign,
            fraction_of_peak=blob_fraction_of_peak,
        )
    if centroid_mode == "bz_blob_geometric":
        if bottom_bz is None:
            raise ValueError("bottom_bz is required for centroid_mode='bz_blob_geometric'")
        return _bottom_signed_blob_geometric_center_ij(
            bottom_bz,
            sign=sign,
            fraction_of_peak=None,
        )
    raise ValueError(f"Unsupported centroid_mode {centroid_mode!r}")


def bald_patch_seed_points(
    *,
    dx: float,
    dy: float,
    dz: float,
    n: int = 50,
    radius: float | None = None,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    z0: float | None = None,
) -> np.ndarray:
    """Seed red field lines on a small circle around physical (0, 0) on the bottom plane."""
    z = 0.5 * float(dz) if z0 is None else float(z0)
    cx, cy, _ = (float(center[0]), float(center[1]), float(center[2]))
    r = BALD_PATCH_RADIUS_CELLS * float(min(dx, dy)) if radius is None else float(radius)
    n = max(1, int(n))
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    points = [
        (cx + r * np.cos(t), cy + r * np.sin(t), z)
        for t in theta
    ]
    return np.asarray(points, dtype=np.float32)


def _current_axis_seed_from_ij(
    iy: float,
    ix: float,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    x, y, z = _index_to_pyvista_xyz(iy, ix, dx=dx, dy=dy, dz=dz)
    return np.asarray([[x, y, z]], dtype=np.float32)


def current_axis_seed_points(
    bottom_jz: np.ndarray,
    *,
    dx: float,
    dy: float,
    dz: float,
    polarity: str,
    bottom_bz: np.ndarray | None = None,
    centroid_mode: CentroidMode = "jz_blob_geometric",
    blob_fraction_of_peak: float = 0.35,
) -> np.ndarray:
    """Return one seed at the center of a signed bottom-plane current blob."""
    if polarity == "positive":
        sign = 1
    elif polarity == "negative":
        sign = -1
    else:
        raise ValueError(f"polarity must be 'positive' or 'negative', got {polarity!r}")
    iy, ix = _current_axis_center_ij(
        bottom_jz,
        bottom_bz,
        sign=sign,
        centroid_mode=centroid_mode,
        blob_fraction_of_peak=blob_fraction_of_peak,
    )
    x, y, z = _index_to_pyvista_xyz(iy, ix, dx=dx, dy=dy, dz=dz)
    return np.asarray([[x, y, z]], dtype=np.float32)


def td2_field_line_focus_bounds(
    case: dict,
    seed_bundle: Dict[str, np.ndarray],
    *,
    pad_xy: float = 28.0,
    z_max: float | None = None,
) -> Tuple[float, float, float, float, float, float]:
    """Return PyVista bounds for a tight TD2 flux-rope close-up around seed points."""
    _, _, _, dx, dy, dz = case["dims"]
    sz, _, _ = case["metric_slices"]
    seeds = np.vstack(
        [
            seed_bundle["bald_patch"],
            seed_bundle["pos_current_axis"],
            seed_bundle["neg_current_axis"],
        ]
    )
    xmin = float(seeds[:, 0].min() - pad_xy)
    xmax = float(seeds[:, 0].max() + pad_xy)
    ymin = float(seeds[:, 1].min() - pad_xy)
    ymax = float(seeds[:, 1].max() + pad_xy)
    zmin = 0.0
    zmax = float(sz.stop * dz) if z_max is None else float(z_max)
    return xmin, xmax, ymin, ymax, zmin, zmax


def compute_td2_field_line_seed_bundle(
    case: dict,
    *,
    n_bald: int = 50,
    bald_radius: float | None = None,
    vector_field: np.ndarray | None = None,
    max_z_planes: int | None = None,
    current_axis_selection: CurrentAxisSelection = "origin_symmetric_trace",
    current_axis_centroid_mode: CentroidMode = "jz_blob_geometric",
    current_axis_blob_fraction_of_peak: float = 0.35,
    manual_pos_current_axis: np.ndarray | None = None,
    symmetrize_current_axis: bool = True,
    trace_sample_stride: int = 4,
    trace_max_candidates: int = 160,
) -> Dict[str, np.ndarray]:
    """Return bald-patch and current-axis seed bundles for TD2 PyVista panels."""
    _, _, _, dx, dy, dz = case["dims"]
    x_bounds = tuple(case["x_bounds"])
    y_bounds = tuple(case["y_bounds"])
    bottom_jz = case["Jz_obs"]
    bottom_bz = case["boundary"]["Bz"]
    cx, cy = _physical_origin_pyvista_xy(x_bounds, y_bounds)

    bald = bald_patch_seed_points(
        dx=dx,
        dy=dy,
        dz=dz,
        n=n_bald,
        radius=bald_radius,
        center=(cx, cy, 0.0),
    )
    if manual_pos_current_axis is not None:
        pos_axis = _as_seed_points(manual_pos_current_axis)
        if pos_axis.shape[0] != 1:
            raise ValueError("manual_pos_current_axis must contain exactly one seed point")
        neg_axis = mirror_current_axis_seed(
            pos_axis,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
    elif current_axis_selection == "origin_symmetric_trace":
        if vector_field is None:
            raise ValueError("vector_field is required for current_axis_selection='origin_symmetric_trace'")
        pos_axis, neg_axis = find_origin_symmetric_current_axis_seeds(
            case,
            vector_field,
            blob_fraction_of_peak=current_axis_blob_fraction_of_peak,
            max_z_planes=max_z_planes,
            sample_stride=trace_sample_stride,
            max_candidates=trace_max_candidates,
        )
    elif symmetrize_current_axis:
        (iy_pos, ix_pos), (iy_neg, ix_neg) = _symmetric_current_axis_centers_ij(
            bottom_jz,
            bottom_bz,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            dx=dx,
            dy=dy,
            centroid_mode=current_axis_centroid_mode,
            blob_fraction_of_peak=current_axis_blob_fraction_of_peak,
        )
        pos_axis = _current_axis_seed_from_ij(iy_pos, ix_pos, dx=dx, dy=dy, dz=dz)
        neg_axis = _current_axis_seed_from_ij(iy_neg, ix_neg, dx=dx, dy=dy, dz=dz)
    else:
        axis_kwargs = dict(
            dx=dx,
            dy=dy,
            dz=dz,
            bottom_bz=bottom_bz,
            centroid_mode=current_axis_centroid_mode,
            blob_fraction_of_peak=current_axis_blob_fraction_of_peak,
        )
        pos_axis = current_axis_seed_points(bottom_jz, polarity="positive", **axis_kwargs)
        neg_axis = current_axis_seed_points(bottom_jz, polarity="negative", **axis_kwargs)
    return {
        "bald_patch": bald,
        "pos_current_axis": pos_axis,
        "neg_current_axis": neg_axis,
    }
