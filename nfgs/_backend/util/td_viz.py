"""TD1--TD3 PyVista / Matplotlib visualisation helpers.

Used by ``scripts/run_nfgs_analytic_validation.py`` to build the unified
reference-vs-NFGS figure. Tune appearance via ``util.td_viz_config.TD_VIZ``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import contextlib
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfpt_analytic_matplotlib")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/nfpt_analytic_mesa")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib

# Allow interactive pickers to keep a GUI backend (set NFPT_INTERACTIVE_MPL=1).
if not os.environ.get("NFPT_INTERACTIVE_MPL"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg
from matplotlib.gridspec import GridSpec

from nfgs._backend.scripts import run_analytic_validation as analytic
from nfgs._backend.util.fastqsl_plane import (
    DEFAULT_QMAP_B_SMOOTH_SIGMA_CELLS,
    Y0_PLANE_COORDINATE_CONTRACT,
    compute_bottom_qmap,
    compute_y0_qmap,
    y0_qmap_cache_matches,
)
from nfgs._backend.util.field_ops import curl_zyx_nfpt
from nfgs._backend.util.td_viz_config import TD_VIZ, TdVizConfig

__all__ = [
    "TdVizConfig",
    "TD_VIZ",
    "clear_viz_caches",
    "compute_td3_footpoint_q_bundle",
    "plot_unified_td_figure",
    "plot_td_four_method_topology_figure",
    "plot_td3_footpoint_qmaps",
    "show_interactive_proj_preview",
    "show_interactive_field_preview",
    "symmetric_xz_box_from_edge_points",
]


def clear_viz_caches() -> None:
    """Drop in-process seed / field-sample caches (figures-only re-runs)."""
    _SEED_GROUP_CACHE.clear()
    _TD3_SEED_CACHE.clear()
    _B_STACK_CACHE.clear()


def _sunpy_cmap():
    """Paper topology figures use a neutral gray photospheric magnetogram."""
    return "gray"


def _grid_origin(case: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(case["x_bounds"][0]),
        float(case["y_bounds"][0]),
        float(case["z_bounds"][0]),
    )


def _vector_grid(pv, vector_fields: dict[str, np.ndarray], case: dict[str, Any], max_z: int):
    _nx, _ny, nz, dx, dy, dz = case["dims"]
    first = next(iter(vector_fields.values()))
    _ncomp, field_nz, ny, nx = first.shape
    viz_nz = min(int(field_nz), int(max_z), int(nz))
    ox, oy, oz = _grid_origin(case)
    grid = pv.ImageData(
        dimensions=(nx, ny, viz_nz),
        spacing=(dx, dy, dz),
        origin=(ox, oy, oz),
    )
    for name, field in vector_fields.items():
        arr = np.asarray(field)[:, :viz_nz]
        vectors = np.moveaxis(arr, 0, -1).reshape(-1, 3)
        grid.point_data[name] = np.ascontiguousarray(vectors, dtype=np.float32)
    return grid


def _bottom_plane(pv, values: np.ndarray, case: dict[str, Any]):
    _nx, _ny, _nz, dx, dy, _dz = case["dims"]
    ny, nx = values.shape
    ox, oy, oz = _grid_origin(case)
    plane = pv.ImageData(
        dimensions=(nx, ny, 1),
        spacing=(dx, dy, 1.0),
        origin=(ox, oy, oz),
    )
    plane.point_data["Bz"] = np.ascontiguousarray(values.reshape(-1), dtype=np.float32)
    return plane


def _trace_lines(pv, grid, vector_name: str, seeds: np.ndarray, *, max_steps: int | None = None):
    source = pv.PolyData(np.ascontiguousarray(seeds, dtype=np.float32))
    steps = int(max_steps if max_steps is not None else TD_VIZ.streamline_max_steps)
    max_length = (
        max(
            grid.bounds[1] - grid.bounds[0],
            grid.bounds[3] - grid.bounds[2],
            grid.bounds[5] - grid.bounds[4],
        )
        * float(TD_VIZ.streamline_max_length_factor)
    )
    return grid.streamlines_from_source(
        source,
        vectors=vector_name,
        integrator_type=45,
        integration_direction="both",
        initial_step_length=0.35,
        min_step_length=0.02,
        max_step_length=1.25,
        max_steps=steps,
        terminal_speed=1e-12,
        max_length=float(max_length),
        compute_vorticity=False,
    )


def _integrate_closed_polyline(
    fields: dict[str, np.ndarray],
    seed: np.ndarray,
    case: dict[str, Any],
    *,
    max_steps: int = 2500,
    step_frac: float = 0.7,
) -> np.ndarray | None:
    """RK trace both ways; return a closed photosphere-to-photosphere polyline."""
    nx, ny, nz, dx, dy, dz = case["dims"]
    x0, y0, z0 = _grid_origin(case)
    x1 = x0 + (nx - 1) * dx
    y1 = y0 + (ny - 1) * dy
    z1 = z0 + (nz - 1) * dz
    ds = step_frac * min(dx, dy, dz)
    z_floor = z0 + 0.25 * dz

    def _one_way(direction: float) -> tuple[list[np.ndarray], str]:
        p = np.asarray(seed, dtype=np.float64).copy()
        path = [p.copy()]
        stagnant = 0
        last = p.copy()
        for _ in range(max_steps):
            b = _sample_b_at(fields, p, case)
            bn = float(np.linalg.norm(b))
            if bn < 1e-14:
                return path, "null"
            p = p + direction * ds * (b / bn)
            path.append(p.copy())
            if p[2] <= z_floor:
                p[2] = z_floor
                path[-1] = p.copy()
                if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                    return path, "bottom"
                return path, "bottom_out"
            if p[0] <= x0 or p[0] >= x1 or p[1] <= y0 or p[1] >= y1 or p[2] >= z1:
                return path, "side_or_top"
            if float(np.linalg.norm(p - last)) < 1e-4 * ds:
                stagnant += 1
                if stagnant >= 8:
                    return path, "stagnant"
            else:
                stagnant = 0
            last = p.copy()
        return path, "max_steps"

    fwd, st_f = _one_way(+1.0)
    bwd, st_b = _one_way(-1.0)
    if st_f != "bottom" or st_b != "bottom":
        return None
    # bwd is from seed toward one footpoint; reverse it then append fwd[1:].
    pts = np.asarray(list(reversed(bwd)) + fwd[1:], dtype=np.float32)
    return pts if pts.shape[0] >= 3 else None


def _drop_escaped_streamlines(
    pv,
    lines,
    case: dict[str, Any],
    *,
    z_vis_top: float | None = None,
):
    """Remove streamlines that leave through the top or lateral domain faces."""
    if lines is None or int(getattr(lines, "n_points", 0)) == 0:
        return lines
    nx, ny, nz, dx, dy, dz = case["dims"]
    x0, y0, z0 = _grid_origin(case)
    x1 = x0 + (nx - 1) * dx
    y1 = y0 + (ny - 1) * dy
    z1 = z0 + (nz - 1) * dz
    z_cut = float(z_vis_top) if z_vis_top is not None else (z0 + 0.92 * (z1 - z0))
    margin = 1.5 * float(min(dx, dy, dz))
    points = np.asarray(lines.points, dtype=np.float64)
    raw = np.asarray(lines.lines, dtype=np.int64)
    out_pts: list[np.ndarray] = []
    out_lines: list[int] = []
    i = 0
    while i < raw.size:
        n = int(raw[i])
        ids = raw[i + 1 : i + 1 + n]
        i = i + 1 + n
        if n < 2:
            continue
        pts = points[ids]
        if float(np.max(pts[:, 2])) >= z_cut:
            continue
        if (
            float(np.min(pts[:, 0])) <= x0 + margin
            or float(np.max(pts[:, 0])) >= x1 - margin
            or float(np.min(pts[:, 1])) <= y0 + margin
            or float(np.max(pts[:, 1])) >= y1 - margin
        ):
            continue
        start = sum(len(p) for p in out_pts)
        out_pts.append(pts.astype(np.float32))
        out_lines.append(len(pts))
        out_lines.extend(range(start, start + len(pts)))
    if not out_pts:
        return pv.PolyData()
    return pv.PolyData(np.vstack(out_pts), lines=np.asarray(out_lines, dtype=np.int64))


def _ring_seeds(
    center_xy: tuple[float, float],
    radius: float,
    z: float,
    *,
    n: int = 24,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    cx, cy = center_xy
    return np.asarray(
        [[cx + radius * math.cos(t), cy + radius * math.sin(t), z] for t in theta],
        dtype=np.float32,
    )


def _multi_ring_seeds(
    center_xy: tuple[float, float],
    radii: list[float],
    z: float,
    *,
    n_per_ring: int = 20,
) -> np.ndarray:
    chunks = [_ring_seeds(center_xy, float(radius), z, n=n_per_ring) for radius in radii if radius > 0.0]
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(chunks).astype(np.float32)


_B_STACK_CACHE: dict[tuple[int, tuple[int, ...], str], np.ndarray] = {}


def _stacked_b(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Cached (3, Nz, Ny, Nx) view for fast field sampling during seed filters.

    Key includes array shape/dtype so a recycled ``id(Bx)`` after GC cannot
    return a stacked cube from a previous (smaller) benchmark lattice.
    """
    bx = np.asarray(fields["Bx"])
    key = (id(bx), tuple(int(v) for v in bx.shape), str(bx.dtype))
    cached = _B_STACK_CACHE.get(key)
    if cached is not None and cached.shape[1:] == bx.shape:
        return cached
    stacked = np.stack(
        [
            bx,
            np.asarray(fields["By"]),
            np.asarray(fields["Bz"]),
        ],
        axis=0,
    )
    _B_STACK_CACHE[key] = stacked
    return stacked


def _sample_b_at(
    fields: dict[str, np.ndarray],
    point: np.ndarray,
    case: dict[str, Any],
) -> np.ndarray:
    """Nearest-neighbour sample of B (fast enough for closed-seed filters)."""
    nx, ny, nz, dx, dy, dz = case["dims"]
    x0, y0, z0 = _grid_origin(case)
    fx = (float(point[0]) - x0) / float(dx)
    fy = (float(point[1]) - y0) / float(dy)
    fz = (float(point[2]) - z0) / float(dz)
    if not (0.0 <= fx <= nx - 1 and 0.0 <= fy <= ny - 1 and 0.0 <= fz <= nz - 1):
        return np.zeros(3, dtype=np.float64)
    i = int(min(max(int(round(fx)), 0), nx - 1))
    j = int(min(max(int(round(fy)), 0), ny - 1))
    k = int(min(max(int(round(fz)), 0), nz - 1))
    return np.asarray(_stacked_b(fields)[:, k, j, i], dtype=np.float64)


def _trace_footpoints(
    fields: dict[str, np.ndarray],
    seed: np.ndarray,
    case: dict[str, Any],
    *,
    max_steps: int = 8000,
    step_frac: float = 0.35,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Trace both ways from ``seed``; return bottom footpoints and exit status.

    Status is ``closed`` when both ends hit the photosphere inside the lateral
    domain, otherwise ``open`` (left through a side/top boundary).
    """
    nx, ny, nz, dx, dy, dz = case["dims"]
    x0, y0, z0 = _grid_origin(case)
    x1 = x0 + (nx - 1) * dx
    y1 = y0 + (ny - 1) * dy
    z1 = z0 + (nz - 1) * dz
    ds = step_frac * min(dx, dy, dz)
    z_floor = z0 + 0.25 * dz

    def _one_way(direction: float) -> tuple[np.ndarray | None, str]:
        p = np.asarray(seed, dtype=np.float64).copy()
        last = p.copy()
        stagnant = 0
        for _ in range(max_steps):
            b = _sample_b_at(fields, p, case)
            bn = float(np.linalg.norm(b))
            if bn < 1e-14:
                return None, "null"
            p = p + direction * ds * (b / bn)
            if p[2] <= z_floor:
                p[2] = z_floor
                if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                    return p.copy(), "bottom"
                return p.copy(), "bottom_out"
            if p[0] <= x0 or p[0] >= x1 or p[1] <= y0 or p[1] >= y1 or p[2] >= z1:
                return p.copy(), "side_or_top"
            if float(np.linalg.norm(p - last)) < 1e-4 * ds:
                stagnant += 1
                if stagnant >= 8:
                    return p.copy(), "stagnant"
            else:
                stagnant = 0
            last = p.copy()
        return p.copy(), "max_steps"

    fp_fwd, st_fwd = _one_way(+1.0)
    fp_bwd, st_bwd = _one_way(-1.0)
    if st_fwd == "bottom" and st_bwd == "bottom" and fp_fwd is not None and fp_bwd is not None:
        return fp_fwd, fp_bwd, "closed"
    return fp_fwd, fp_bwd, "open"


def _find_symmetric_axis_seeds(
    case: dict[str, Any],
    fields: dict[str, np.ndarray],
    *,
    y_guess: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pick x=0,z~0 seeds whose conjugate footpoint is most centre-symmetric.

    ``y_guess`` defaults to ``|case['axis_y_phys']|`` when present, otherwise 80 Mm.
    Callers should pass an explicit guess rather than mutating ``case`` in place.
    """
    _nx, _ny, _nz, dx, dy, dz = case["dims"]
    z = 0.5 * float(dz)
    if y_guess is None:
        y_guess = abs(float(case.get("axis_y_phys", 80.0)))
    y_guess = abs(float(y_guess))
    # Coarse then refine along the x=0 photospheric cut near the geometric footpoint.
    y_coarse = np.linspace(0.45 * y_guess, 1.25 * y_guess, 21)
    best_y = y_guess
    best_score = float("inf")
    best_pair: tuple[np.ndarray, np.ndarray] | None = None

    def _score_y(y: float) -> tuple[float, tuple[np.ndarray, np.ndarray] | None]:
        seed = np.asarray([0.0, float(y), z], dtype=np.float64)
        bz = float(_sample_b_at(fields, seed, case)[2])
        if bz <= 0.0:
            return float("inf"), None
        fp_a, fp_b, status = _trace_footpoints(fields, seed, case, max_steps=4000)
        if status != "closed" or fp_a is None or fp_b is None:
            return float("inf"), None
        d_a = abs(float(fp_a[1]) - float(y))
        d_b = abs(float(fp_b[1]) - float(y))
        conj = fp_b if d_a <= d_b else fp_a
        # Centre symmetry about (0,0): conjugate should lie near (0, -y).
        score = abs(float(conj[0]) - 0.0) + abs(float(conj[1]) + float(y))
        seed_pos = np.asarray([[0.0, float(y), z]], dtype=np.float32)
        seed_neg = np.asarray([[float(conj[0]), float(conj[1]), z]], dtype=np.float32)
        return score, (seed_pos, seed_neg)

    for y in y_coarse:
        score, pair = _score_y(float(y))
        if pair is not None and score < best_score:
            best_score = score
            best_y = float(y)
            best_pair = pair
    if best_pair is not None:
        # Local refinement around the coarse minimum.
        y_fine = np.linspace(best_y - 0.08 * y_guess, best_y + 0.08 * y_guess, 11)
        for y in y_fine:
            score, pair = _score_y(float(y))
            if pair is not None and score < best_score:
                best_score = score
                best_y = float(y)
                best_pair = pair
    if best_pair is None:
        pos = np.asarray([[0.0, y_guess, z]], dtype=np.float32)
        neg = np.asarray([[0.0, -y_guess, z]], dtype=np.float32)
        return pos, neg, y_guess
    return best_pair[0], best_pair[1], best_y


_SEED_GROUP_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}


def _field_line_seed_groups(
    case: dict[str, Any],
    *,
    fields: dict[str, np.ndarray] | None = None,
    q: np.ndarray | None = None,
    filter_fields: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """TD-figure seed policy.

    TD1: wide flux-rope footpoint rings (r/r0=1..5, r0=7 Mm) + centre-symmetric axis.
    TD2: bald-patch seeds + axis (same axis footpoints as the analytic model).
    TD3: Fig.4 R1/R2/R3 strong-flux outer-contour seeds on y=0 (closed only).

    For TD1/TD2, seed locations are always taken from the analytic reference field
    (same footpoints for ref and NFGS). For TD3, closed-line filtering uses
    ``filter_fields`` (the field being visualised) so lines stay inside the domain.
    """
    name = str(case["benchmark"])
    chooser = case["reference"]  # analytic footpoints for TD1/TD2
    closer = filter_fields if filter_fields is not None else (
        fields if fields is not None else chooser
    )
    cache_key = (
        name,
        id(chooser),
        id(closer) if name == "td3" else 0,
        None if q is None else (id(q), float(np.nanmax(q))),
        # Manual TD1/TD2 axis |y| and seed counts change groups without field ids.
        (
            None
            if TD_VIZ.td1_axis_y_mm is None
            else float(TD_VIZ.td1_axis_y_mm)
        ),
        (
            None
            if TD_VIZ.td2_axis_y_mm is None
            else float(TD_VIZ.td2_axis_y_mm)
        ),
        int(TD_VIZ.td1_rope_seeds_per_polarity),
        int(TD_VIZ.td2_axis_ring_seeds),
        int(TD_VIZ.td3_r1_seeds_per_polarity),
        int(TD_VIZ.td3_r2_n_seeds),
        int(TD_VIZ.td3_r3_nx),
        int(TD_VIZ.td3_r3_nz),
    )
    cached = _SEED_GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _nx, _ny, _nz, dx, dy, dz = case["dims"]
    z = 0.5 * float(dz)
    cell = min(dx, dy, dz)
    thin = TD_VIZ.tube_radius_thin * cell
    thick = TD_VIZ.tube_radius_thick * cell

    if name == "td3":
        geometry = case["geometry"]
        if q is None:
            raise ValueError("TD3 seed groups require a y=0 Q map")
        groups = _td3_fig4_seed_groups(case, q, geometry, fields=closer)[0]
        _SEED_GROUP_CACHE[cache_key] = groups
        return groups

    if name == "td2":
        manual_y = TD_VIZ.td2_axis_y_mm
        if manual_y is not None and float(manual_y) > 0.0:
            y_abs = abs(float(manual_y))
            pos_c = (0.0, y_abs)
            neg_c = (0.0, -y_abs)
        else:
            axis_pos, axis_neg, _axis_y = _find_symmetric_axis_seeds(
                case, chooser, y_guess=80.0
            )
            pos_c = (float(axis_pos[0, 0]), float(axis_pos[0, 1]))
            neg_c = (float(axis_neg[0, 0]), float(axis_neg[0, 1]))
        # Fig.3: bald-patch flux-rope seeds + 10 axis seeds on a narrow ring.
        bald = analytic._bald_patch_seeds(0.0, 0.0, dx, dy, dz)
        axis_pos_ring = _ring_seeds(
            pos_c, TD_VIZ.td2_axis_ring_radius_mm, z, n=TD_VIZ.td2_axis_ring_seeds
        )
        axis_neg_ring = _ring_seeds(
            neg_c, TD_VIZ.td2_axis_ring_radius_mm, z, n=TD_VIZ.td2_axis_ring_seeds
        )
        groups = [
            {"seeds": bald, "color": TD_VIZ.color_td2_bald, "radius": thin, "label": "bald"},
            {
                "seeds": axis_pos_ring,
                "color": TD_VIZ.color_axis_pos,
                "radius": thick,
                "label": "axis_pos",
            },
            {
                "seeds": axis_neg_ring,
                "color": TD_VIZ.color_axis_neg,
                "radius": thick,
                "label": "axis_neg",
            },
        ]
        _SEED_GROUP_CACHE[cache_key] = groups
        return groups

    # TD1 (Fig.1): axis in blue/magenta; outer rope boundary only (10 seeds / polarity).
    manual_y = TD_VIZ.td1_axis_y_mm
    if manual_y is not None and float(manual_y) > 0.0:
        y_abs = abs(float(manual_y))
        pos_c = (0.0, y_abs)
        neg_c = (0.0, -y_abs)
        axis_pos = np.asarray([[0.0, y_abs, z]], dtype=np.float32)
        axis_neg = np.asarray([[0.0, -y_abs, z]], dtype=np.float32)
    else:
        axis_pos, axis_neg, _axis_y = _find_symmetric_axis_seeds(
            case, chooser, y_guess=80.0
        )
        # Do not write axis_y back onto the shared case dict (pollutes later builds).
        pos_c = (float(axis_pos[0, 0]), float(axis_pos[0, 1]))
        neg_c = (float(axis_neg[0, 0]), float(axis_neg[0, 1]))

    params = case.get("geometry", {}).get("parameters", {})
    rope_a = float(params.get("a", 35.0))
    outer_r = min(
        TD_VIZ.td1_rope_r0_mm * TD_VIZ.td1_rope_outer_ratio,
        TD_VIZ.td1_rope_a_scale * rope_a,
    )
    pos_ring = _filter_closed_seed_array(
        _ring_seeds(
            pos_c, outer_r, z, n=TD_VIZ.td1_rope_seeds_per_polarity
        ),
        chooser,
        case,
    )
    neg_ring = _filter_closed_seed_array(
        _ring_seeds(
            neg_c, outer_r, z, n=TD_VIZ.td1_rope_seeds_per_polarity
        ),
        chooser,
        case,
    )
    groups = [
        {
            "seeds": pos_ring,
            "color": TD_VIZ.color_rope_pos,
            "radius": thin,
            "label": "rope_pos",
        },
        {
            "seeds": neg_ring,
            "color": TD_VIZ.color_rope_neg,
            "radius": thin,
            "label": "rope_neg",
        },
        {
            "seeds": axis_pos,
            "color": TD_VIZ.color_axis_pos,
            "radius": thick,
            "label": "axis_pos",
        },
        {
            "seeds": axis_neg,
            "color": TD_VIZ.color_axis_neg,
            "radius": thick,
            "label": "axis_neg",
        },
    ]
    _SEED_GROUP_CACHE[cache_key] = groups
    return groups


def _filter_closed_seed_array(
    seeds: np.ndarray,
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
) -> np.ndarray:
    """Keep photosphere seeds whose field lines stay closed in the box."""
    kept: list[np.ndarray] = []
    for seed in np.asarray(seeds, dtype=np.float64):
        _a, _b, status = _trace_footpoints(
            fields, seed, case, max_steps=1200, step_frac=1.0
        )
        if status == "closed":
            kept.append(seed.astype(np.float32))
    if not kept:
        return np.asarray(seeds, dtype=np.float32)
    return np.vstack(kept)


def _photospheric_centroid(case: dict[str, Any]) -> tuple[float, float]:
    """|Bz|-weighted bipolar centre on the photosphere (Mm)."""
    bz = np.asarray(case["boundary"]["Bz"], dtype=np.float64)
    _nx, _ny, _nz, dx, dy, _dz = case["dims"]
    x0, y0, _z0 = _grid_origin(case)
    w = np.abs(bz)
    if not np.any(np.isfinite(w)) or float(np.nansum(w)) <= 0.0:
        return (0.0, 0.0)
    jj, ii = np.indices(bz.shape)
    wsum = float(np.nansum(w))
    cx = float(x0 + np.nansum(w * ii) / wsum * dx)
    cy = float(y0 + np.nansum(w * jj) / wsum * dy)
    return (cx, cy)


def _crop_params_for_case(case: dict[str, Any]) -> dict[str, float | None]:
    """Return independent top/side crop knobs for the active benchmark."""
    name = str(case["benchmark"])
    cfg = TD_VIZ
    if name == "td2":
        return {
            "top_cx": cfg.top_center_x_td2,
            "top_cy": cfg.top_center_y_td2,
            "top_x_pad": cfg.top_x_pad_td2,
            "top_y_pad": cfg.top_y_pad_td2,
            "side_cy": cfg.side_center_y_td2,
            "side_y_pad": cfg.side_y_pad_td2,
            "side_z_min": cfg.side_z_min_td2,
            "side_z_max": cfg.side_z_max_td2,
        }
    if name == "td3":
        return {
            "top_cx": cfg.top_center_x_td3,
            "top_cy": cfg.top_center_y_td3,
            "top_x_pad": cfg.top_x_pad_td3,
            "top_y_pad": cfg.top_y_pad_td3,
            "side_cy": cfg.side_center_y_td3,
            "side_y_pad": cfg.side_y_pad_td3,
            "side_z_min": cfg.side_z_min_td3,
            "side_z_max": cfg.side_z_max_td3,
        }
    return {
        "top_cx": cfg.top_center_x_td1,
        "top_cy": cfg.top_center_y_td1,
        "top_x_pad": cfg.top_x_pad_td1,
        "top_y_pad": cfg.top_y_pad_td1,
        "side_cy": cfg.side_center_y_td1,
        "side_y_pad": cfg.side_y_pad_td1,
        "side_z_min": cfg.side_z_min_td1,
        "side_z_max": cfg.side_z_max_td1,
    }


def _metric_xy_bounds(
    case: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    """Physical ``(x0, x1, y0, y1)`` of the horizontal metric ROI, if present."""
    slices = case.get("metric_slices")
    if not slices:
        return None
    _sz, sy, sx = slices
    nx, ny, _nz, _dx, _dy, _dz = case["dims"]
    nx, ny = int(nx), int(ny)
    xs = np.linspace(
        float(case["x_bounds"][0]), float(case["x_bounds"][1]), nx, endpoint=True
    )
    ys = np.linspace(
        float(case["y_bounds"][0]), float(case["y_bounds"][1]), ny, endpoint=True
    )
    y0 = 0 if sy.start is None else int(sy.start)
    y1 = ny - 1 if sy.stop is None else int(sy.stop) - 1
    x0 = 0 if sx.start is None else int(sx.start)
    x1 = nx - 1 if sx.stop is None else int(sx.stop) - 1
    return float(xs[x0]), float(xs[x1]), float(ys[y0]), float(ys[y1])


def _top_focus_bounds(case: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """Top-view crop only: ``(x0, x1, y0, y1, z0, z1)`` from ``top_*`` knobs.

    Independent of ``side_*``: changing ``side_y_pad`` / ``side_z_*`` never
    affects this window (and vice versa).
    """
    params = _crop_params_for_case(case)
    auto_cx, auto_cy = _photospheric_centroid(case)
    cx = float(auto_cx if params["top_cx"] is None else params["top_cx"])
    cy = float(auto_cy if params["top_cy"] is None else params["top_cy"])
    x_pad = float(params["top_x_pad"])
    y_pad = float(params["top_y_pad"])
    z0 = float(case["z_bounds"][0])
    # Dummy z span for reset_camera; top view ignores height framing.
    z1 = z0 + max(1.0, 0.25 * (2.0 * max(x_pad, y_pad)))
    return (cx - x_pad, cx + x_pad, cy - y_pad, cy + y_pad, z0, z1)


def _side_focus_bounds(case: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """Side-view crop only: ``(x0, x1, y0, y1, z0, z1)`` from ``side_*`` knobs.

    Independent of ``top_*``: uses ``side_center_y`` / ``side_y_pad`` /
    ``side_z_min`` / ``side_z_max`` only.
    """
    params = _crop_params_for_case(case)
    _auto_cx, auto_cy = _photospheric_centroid(case)
    cy = float(auto_cy if params["side_cy"] is None else params["side_cy"])
    y_pad = float(params["side_y_pad"])
    z0 = max(float(params["side_z_min"]), float(case["z_bounds"][0]))
    z1 = float(params["side_z_max"])
    # x span is unused by the YZ side camera but kept for a valid bounds tuple.
    x_half = max(y_pad, 0.5 * (z1 - z0))
    return (-x_half, x_half, cy - y_pad, cy + y_pad, z0, z1)


def _visual_focus_bounds(
    case: dict[str, Any],
    seed_groups: list[dict[str, Any]] | None = None,
    *,
    view: str = "top",
) -> tuple[float, float, float, float, float, float]:
    """Dispatch to independent top/side crop helpers."""
    del seed_groups
    if view == "side":
        return _side_focus_bounds(case)
    return _top_focus_bounds(case)


def _unified_panel_window_size() -> tuple[int, int]:
    """Single (width, height) shared by every top/side field panel."""
    h = max(64, int(TD_VIZ.panel_height_px))
    w = max(64, int(round(h * float(TD_VIZ.panel_aspect))))
    return (w, h)


def _set_top_view_camera(
    plotter,
    focus: tuple[float, ...],
    *,
    window_size: tuple[int, int],
) -> None:
    """Lock top-view framing to the configured x/y crop (parallel projection).

    Orientation matches the original TD panels: ``view_xy`` then +90° roll
    (world-y horizontal on the page). Only ``parallel_scale`` / focal point are
    adjusted so identical ``top_*`` pads yield the same Mm/pixel for TD1/TD2.
    """
    x0, x1, y0, y1, z0, _z1 = (float(v) for v in focus)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    x_span = max(x1 - x0, 1.0)
    y_span = max(y1 - y0, 1.0)
    aspect = float(window_size[0]) / float(max(window_size[1], 1))

    plotter.enable_parallel_projection()
    plotter.view_xy()
    plotter.camera.roll = float(plotter.camera.roll) + 90.0

    # Keep the rolled camera basis; only recentre on the crop.
    fp = np.asarray(plotter.camera.focal_point, dtype=np.float64)
    pos = np.asarray(plotter.camera.position, dtype=np.float64)
    view_dir = pos - fp
    norm = float(np.linalg.norm(view_dir))
    if norm < 1.0e-12:
        view_dir = np.asarray([0.0, 0.0, 1.0])
        norm = 1.0
    view_dir = view_dir / norm
    new_fp = np.asarray([cx, cy, z0], dtype=np.float64)
    new_pos = new_fp + view_dir * (max(x_span, y_span) * 3.0)
    plotter.camera.focal_point = (float(new_fp[0]), float(new_fp[1]), float(new_fp[2]))
    plotter.camera.position = (float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))

    # After +90° roll: image horizontal ↔ world y, image vertical ↔ world x.
    # parallel_scale is half the viewport height in world units.
    half_h = max(0.5 * x_span, 0.5 * y_span / max(aspect, 1e-6))
    plotter.camera.parallel_scale = float(half_h)
    try:
        plotter.camera.clipping_range = (0.1, float(np.linalg.norm(new_pos - new_fp)) * 20.0)
    except Exception:
        pass


def _set_side_view_camera(
    plotter,
    focus: tuple[float, ...],
    *,
    window_size: tuple[int, int],
    elevate: bool = True,
) -> None:
    """Elevated YZ side view with the photosphere floor flush to the panel bottom."""
    y0, y1 = float(focus[2]), float(focus[3])
    z0, z1 = float(focus[4]), float(focus[5])
    pad_z_top = TD_VIZ.side_camera_z_pad_fraction * max(z1 - z0, 1.0)
    y_lo, y_hi = y0, y1
    z_lo, z_hi = z0, z1 + pad_z_top
    cy = 0.5 * (y_lo + y_hi)
    y_span = max(y_hi - y_lo, 1.0)
    z_span = max(z_hi - z_lo, 1.0)
    aspect = float(window_size[0]) / float(max(window_size[1], 1))
    elev_deg = TD_VIZ.side_camera_elev_deg
    elev = math.radians(elev_deg)
    eye_x = max(y_span, z_span) * 2.8
    plotter.enable_parallel_projection()
    # Bias the look-at point downward so the floor sits near the image bottom.
    floor_bias = float(np.clip(TD_VIZ.side_floor_bottom_bias, 0.5, 1.0))
    look_z = z_lo + (1.0 - floor_bias) * z_span + 0.08 * z_span
    if elevate:
        plotter.camera_position = [
            (
                eye_x,
                cy,
                look_z + math.tan(elev) * 0.35 * y_span,
            ),
            (0.0, cy, look_z),
            (0.0, 0.0, 1.0),
        ]
    else:
        plotter.camera_position = [
            (eye_x, cy, look_z),
            (0.0, cy, look_z),
            (0.0, 0.0, 1.0),
        ]
    # Fit z into the upper portion of the frame; y foreshortening still fits width.
    y_proj = y_span * math.cos(elev)
    half_h = max(0.52 * z_span / max(floor_bias, 0.5), 0.52 * y_proj / max(aspect, 1e-6))
    plotter.camera.parallel_scale = float(half_h)
    try:
        plotter.camera.clipping_range = (0.1, float(eye_x) * 20.0)
    except Exception:
        pass


def _closed_polylines_polydata(
    pv,
    fields: dict[str, np.ndarray],
    seeds: np.ndarray,
    case: dict[str, Any],
):
    """Build a PolyData of photosphere-to-photosphere polylines (conjugated feet)."""
    out_pts: list[np.ndarray] = []
    out_lines: list[int] = []
    for seed in np.asarray(seeds, dtype=np.float64):
        poly = _integrate_closed_polyline(
            fields,
            seed,
            case,
            max_steps=int(TD_VIZ.streamline_max_steps),
            step_frac=0.55,
        )
        if poly is None or poly.shape[0] < 3:
            continue
        start = sum(len(p) for p in out_pts)
        out_pts.append(np.asarray(poly, dtype=np.float32))
        out_lines.append(len(poly))
        out_lines.extend(range(start, start + len(poly)))
    if not out_pts:
        return pv.PolyData()
    return pv.PolyData(np.vstack(out_pts), lines=np.asarray(out_lines, dtype=np.int64))


def _export_plotter_html(plotter: Any, html_path: Path | None) -> None:
    """Write an interactive HTML scene for ``plotter`` (requires trame stack)."""
    if html_path is None:
        return
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotter.export_html(str(html_path))
        print(f"  html -> {html_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  html export skipped ({html_path.name}): {exc}", flush=True)


def _render_field_view(
    *,
    case: dict[str, Any],
    vector_name: str,
    vector_fields: dict[str, np.ndarray],
    view: str,
    title: str,
    output_path: Path,
    window_size: tuple[int, int] | None = None,
    q: np.ndarray | None = None,
    seed_groups: list[dict[str, Any]] | None = None,
    html_path: Path | None = None,
) -> None:
    import pyvista as pv

    pv.OFF_SCREEN = True
    pv.global_theme.background = TD_VIZ.pv_background
    pv.global_theme.font.color = "black"
    # Avoid stale offscreen buffers / recycled array-id cache entries when
    # many Plotters are created back-to-back (TD1→TD3, side→top).
    try:
        pv.close_all()
    except Exception:
        pass
    _B_STACK_CACHE.clear()

    _nx, _ny, nz, dx, dy, dz = case["dims"]
    focus = _visual_focus_bounds(case, view=view)
    side_focus = _side_focus_bounds(case)
    if TD_VIZ.streamline_use_full_z:
        max_z_planes = int(nz)
    else:
        z0 = float(case["z_bounds"][0])
        z_need = max(float(focus[5]), float(side_focus[5]))
        max_z_planes = min(
            int(nz),
            max(2, int(math.ceil((z_need - z0) / float(dz))) + 2),
        )
    grid = _vector_grid(pv, vector_fields, case, max_z_planes)
    bottom = _bottom_plane(pv, np.asarray(case["boundary"]["Bz"], dtype=np.float32), case)
    # Clip photosphere to the view crop so identical pads share one floor frame.
    fx0, fx1, fy0, fy1, fz0, _fz1 = (float(v) for v in focus)
    if view == "top":
        bottom = bottom.clip_box(
            [fx0, fx1, fy0, fy1, fz0 - float(dz), fz0 + float(dz)],
            invert=False,
        )
    trace_fields = {
        "Bx": np.asarray(vector_fields[vector_name][0]),
        "By": np.asarray(vector_fields[vector_name][1]),
        "Bz": np.asarray(vector_fields[vector_name][2]),
    }
    if seed_groups is not None:
        groups = seed_groups
    elif str(case["benchmark"]) == "td3":
        # Filter closed lines on the visualised field so streams stay in-domain.
        groups = _field_line_seed_groups(
            case, fields=trace_fields, q=q, filter_fields=trace_fields
        )
    else:
        # Same analytic footpoints for reference and numerical panels.
        groups = _field_line_seed_groups(case, fields=case["reference"], q=q)
    nonempty = [group for group in groups if len(group["seeds"]) > 0]
    if not nonempty:
        raise RuntimeError(f"No field-line seeds available for {case['benchmark']}")
    if window_size is None:
        window_size = _unified_panel_window_size()
    bz_abs = float(np.nanpercentile(np.abs(case["boundary"]["Bz"]), 99.0)) or 1.0
    drop_escaped = str(case["benchmark"]) == "td3"

    plotter = pv.Plotter(window_size=list(window_size), border=False, off_screen=True)
    plotter.set_background(TD_VIZ.pv_background)
    plotter.add_mesh(
        bottom,
        scalars="Bz",
        cmap=_sunpy_cmap(),
        clim=(-bz_abs, bz_abs),
        opacity=(
            TD_VIZ.bottom_opacity_side
            if view == "side"
            else TD_VIZ.bottom_opacity_top
        ),
        show_scalar_bar=False,
    )
    for group in nonempty:
        # Prefer closed foot-to-foot RK traces so arches are not truncated mid-volume.
        lines = _closed_polylines_polydata(pv, trace_fields, group["seeds"], case)
        if int(getattr(lines, "n_points", 0)) == 0:
            lines = _trace_lines(pv, grid, vector_name, group["seeds"])
            if drop_escaped:
                lines = _drop_escaped_streamlines(
                    pv,
                    lines,
                    case,
                    z_vis_top=float(side_focus[5]) + 2.0 * float(dz),
                )
        if lines.n_points:
            plotter.add_mesh(
                lines.tube(radius=float(group["radius"]), n_sides=12),
                color=group["color"],
                smooth_shading=True,
            )
    # Subtitle is overlaid after letterbox so all panels share one baseline.
    if view == "top":
        _set_top_view_camera(plotter, focus, window_size=window_size)
    else:
        _set_side_view_camera(
            plotter,
            focus,
            window_size=window_size,
            elevate=True,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit render + screenshot: Plotter.show(screenshot=...) can reuse a
    # stale framebuffer after a previous side-view Plotter on some Mesa builds,
    # which made TD3 NFGS top identical to TD3 ref top (subtitle aside).
    try:
        plotter.render()
        plotter.screenshot(str(output_path), window_size=list(window_size))
        _export_plotter_html(plotter, html_path)
    finally:
        plotter.close()
        try:
            pv.close_all()
        except Exception:
            pass
    if view == "top":
        # Do NOT trim left/right: quiet magnetogram ≈ background would otherwise
        # crop TD1 tighter than TD2 even when top_* pads match.
        _trim_panel_whitespace(
            output_path,
            trim_top=False,
            trim_bottom=False,
            trim_left=False,
            trim_right=False,
            margin_px=TD_VIZ.trim_margin_px,
            background_rgb=_color_to_rgb01(TD_VIZ.pv_background),
            tolerance=TD_VIZ.bg_trim_tolerance,
        )
    else:
        _trim_panel_whitespace(
            output_path,
            trim_top=bool(TD_VIZ.side_trim_top),
            trim_bottom=bool(TD_VIZ.side_trim_bottom),
            trim_left=False,
            trim_right=False,
            margin_px=TD_VIZ.trim_margin_px,
            background_rgb=_color_to_rgb01(TD_VIZ.pv_background),
            tolerance=TD_VIZ.bg_trim_tolerance,
        )
    # Re-pad / letterbox to the shared panel window so all tiles share one AR.
    _enforce_panel_window_size(output_path, window_size)
    _overlay_panel_subtitle(output_path, title)


def _color_to_rgb01(value: str) -> tuple[float, float, float]:
    """Parse '#rrggbb' or a matplotlib colour name to RGB in [0, 1]."""
    text = str(value).strip()
    if text.startswith("#") and len(text.lstrip("#")) == 6:
        raw = text.lstrip("#")
        return tuple(int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    try:
        from matplotlib.colors import to_rgb

        rgb = to_rgb(text)
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    except Exception:
        return (0.69, 0.69, 0.69)


def _hex_to_rgb01(value: str) -> tuple[float, float, float]:
    return _color_to_rgb01(value)


def _enforce_panel_window_size(
    image_path: Path,
    window_size: tuple[int, int],
) -> None:
    """Letterbox a panel into the shared (W, H) canvas with the PyVista background."""
    from matplotlib import pyplot as _plt

    rgba = np.asarray(mpimg.imread(str(image_path)))
    if rgba.ndim != 3:
        return
    target_w, target_h = int(window_size[0]), int(window_size[1])
    src_h, src_w = int(rgba.shape[0]), int(rgba.shape[1])
    if src_w == target_w and src_h == target_h:
        return
    scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    try:
        from PIL import Image

        mode = "RGBA" if rgba.shape[2] == 4 else "RGB"
        img = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), mode=mode)
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        resized = np.asarray(img).astype(np.float32) / 255.0
    except Exception:
        yy = (np.linspace(0, src_h - 1, new_h)).astype(int)
        xx = (np.linspace(0, src_w - 1, new_w)).astype(int)
        resized = rgba[yy][:, xx]

    bg = np.asarray(_color_to_rgb01(TD_VIZ.pv_background), dtype=np.float32)
    canvas = np.ones((target_h, target_w, resized.shape[2]), dtype=np.float32)
    canvas[..., 0] = bg[0]
    canvas[..., 1] = bg[1]
    canvas[..., 2] = bg[2]
    if canvas.shape[2] == 4:
        canvas[..., 3] = 1.0
    x0 = (target_w - new_w) // 2
    # Side panels: pin content to the bottom so the magnetogram floor stays flush.
    is_side = "_side" in image_path.name
    y0 = (target_h - new_h) if is_side else (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w, : resized.shape[2]] = resized
    _plt.imsave(str(image_path), np.clip(canvas, 0.0, 1.0))


def _overlay_panel_subtitle(image_path: Path, title: str) -> None:
    """Draw subtitle at a fixed top-left pixel offset after the panel is finalised.

    All panels share the same pad so top/side/Q/proj titles sit on one baseline.
    """
    from PIL import Image, ImageDraw, ImageFont

    text = str(title).strip()
    if not text:
        return
    img = Image.open(str(image_path)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font_size = max(10, int(TD_VIZ.subtitle_fontsize_px))
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    ox = int(TD_VIZ.subtitle_pad_x_px)
    oy = int(TD_VIZ.subtitle_pad_y_px)
    # Soft halo so the title stays readable on both light and dark content.
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((ox + dx, oy + dy), text, font=font, fill=(255, 255, 255, 220))
    draw.text((ox, oy), text, font=font, fill=(0, 0, 0, 255))
    img.save(str(image_path))


def _trim_panel_whitespace(
    image_path: Path,
    *,
    trim_top: bool,
    trim_bottom: bool,
    trim_left: bool = True,
    trim_right: bool = True,
    margin_px: int = 4,
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tolerance: float = 0.035,
) -> None:
    """Crop near-background margins so the projected floor sits on the frame edge."""
    if not (trim_top or trim_bottom or trim_left or trim_right):
        return
    rgba = np.asarray(mpimg.imread(str(image_path)))
    if rgba.ndim != 3:
        return
    rgb = rgba[..., :3]
    bg = np.asarray(background_rgb, dtype=np.float64)
    content = np.linalg.norm(rgb - bg, axis=2) > float(tolerance)
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return
    top = int(rows[0]) if trim_top else 0
    bottom = (
        min(int(rows[-1]) + int(margin_px) + 1, rgba.shape[0])
        if trim_bottom
        else rgba.shape[0]
    )
    left = max(int(cols[0]) - int(margin_px), 0) if trim_left else 0
    right = (
        min(int(cols[-1]) + int(margin_px) + 1, rgba.shape[1])
        if trim_right
        else rgba.shape[1]
    )
    cropped = rgba[top:bottom, left:right]
    from matplotlib import pyplot as _plt

    _plt.imsave(str(image_path), np.clip(cropped, 0.0, 1.0))


def _locate_hft(q: np.ndarray, geometry: dict[str, Any]) -> tuple[float, float]:
    """Locate the TD3 HFT/QS near the base X-point on the y=0 Q map."""
    logq = np.log10(np.maximum(np.asarray(q, dtype=np.float64), 1.0))
    nz, nx = logq.shape
    x0, x1 = geometry["x_bounds_mm"]
    z0, z1 = geometry["z_bounds_mm"]
    x = np.linspace(x0, x1, nx)
    z = np.linspace(z0, z1, nz)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    # Prefer the low-altitude X-point rather than bright ring segments higher up.
    mask = (np.abs(xx) <= 25.0) & (zz >= 2.0) & (zz <= 45.0) & np.isfinite(logq)
    if not np.any(mask):
        mask = np.isfinite(logq)
    masked = np.where(mask, logq, -np.inf)
    iz, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return float(x[ix]), float(z[iz])


def _hft_box_bounds(
    x_hft: float,
    z_hft: float,
    *,
    q: np.ndarray | None = None,
    geometry: dict[str, Any] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Fig. 4 boxes: R1 above QS, R2 on QS, R3 below QS (x0,x1,z0,z1).

    Default ROIs use the HFT-centred prism::

        x in [x_hft ± half_width],
        z in [z_hft + z0_offset, z_hft + z1_offset].

    Absolute overrides ``td3_r{1,2,3}_box_abs`` replace the corresponding box
    when set (interactive picker / YAML).  ``q`` / ``geometry`` are accepted for
    call-site compatibility and ignored.
    """
    del q, geometry
    from nfgs._backend.util.td_viz_config import _coerce_abs_box

    boxes = {
        "R1": (
            x_hft - TD_VIZ.td3_r1_half_width,
            x_hft + TD_VIZ.td3_r1_half_width,
            z_hft + TD_VIZ.td3_r1_z0_offset,
            z_hft + TD_VIZ.td3_r1_z1_offset,
        ),
        "R2": (
            x_hft - TD_VIZ.td3_r2_half_width,
            x_hft + TD_VIZ.td3_r2_half_width,
            z_hft + TD_VIZ.td3_r2_z0_offset,
            z_hft + TD_VIZ.td3_r2_z1_offset,
        ),
        "R3": (
            x_hft - TD_VIZ.td3_r3_half_width,
            x_hft + TD_VIZ.td3_r3_half_width,
            z_hft + TD_VIZ.td3_r3_z0_offset,
            z_hft + TD_VIZ.td3_r3_z1_offset,
        ),
    }
    for name, attr in (
        ("R1", "td3_r1_box_abs"),
        ("R2", "td3_r2_box_abs"),
        ("R3", "td3_r3_box_abs"),
    ):
        abs_box = _coerce_abs_box(getattr(TD_VIZ, attr, None))
        if abs_box is not None:
            boxes[name] = abs_box
    return boxes


def _td3_jz_flux_rope_axis_centers(
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
    *,
    blob_fraction_of_peak: float = 0.35,
    border_fraction: float = 0.15,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Locate ± flux-rope foot centres from signed bottom-Jz blobs.

    Edge current sheets on the lateral domain boundary are masked out
    (``border_fraction`` of nx/ny) so the largest remaining ±Jz clumps
    correspond to the flux-rope feet rather than wall artefacts.  Centres are
    then forced into origin-symmetric positions.
    """
    from nfgs._backend.util.td2_viz import _signed_blob_mask

    nx, ny, _nz, dx, dy, dz = case["dims"]
    _jx, _jy, jz = curl_zyx_nfpt(
        fields["Bx"], fields["By"], fields["Bz"], dx=dx, dy=dy, dz=dz
    )
    jz_bottom = np.asarray(jz[0], dtype=np.float64).copy()
    margin_y = max(1, int(round(float(border_fraction) * ny)))
    margin_x = max(1, int(round(float(border_fraction) * nx)))
    jz_bottom[:margin_y, :] = 0.0
    jz_bottom[-margin_y:, :] = 0.0
    jz_bottom[:, :margin_x] = 0.0
    jz_bottom[:, -margin_x:] = 0.0

    def _blob_center(sign: int) -> tuple[float, float] | None:
        blob = _signed_blob_mask(
            jz_bottom, sign=sign, fraction_of_peak=blob_fraction_of_peak
        )
        if not np.any(blob):
            return None
        iy_grid, ix_grid = np.indices(jz_bottom.shape, dtype=np.float64)
        total = float(blob.sum())
        iy = float((blob * iy_grid).sum() / total)
        ix = float((blob * ix_grid).sum() / total)
        return iy, ix

    pos = _blob_center(+1)
    neg = _blob_center(-1)
    x0, y0, _z0 = _grid_origin(case)
    if pos is None or neg is None:
        # Fallback: conjugate footpoint search on x=0.
        axis_pos, axis_neg, _axis_y = _find_symmetric_axis_seeds(case, fields)
        return (
            (float(axis_pos[0, 0]), float(axis_pos[0, 1])),
            (float(axis_neg[0, 0]), float(axis_neg[0, 1])),
        )

    iy_pos, ix_pos = pos
    iy_neg, ix_neg = neg
    # Origin-symmetrise the pair (average magnitudes, keep signs).
    ix_mag = 0.5 * (abs(ix_pos - (nx - 1) / 2.0) + abs(ix_neg - (nx - 1) / 2.0))
    iy_mag = 0.5 * (abs(iy_pos - (ny - 1) / 2.0) + abs(iy_neg - (ny - 1) / 2.0))
    # Map back so +Jz foot stays in the half-plane of the raw + blob.
    cy = 0.5 * (ny - 1)
    cx = 0.5 * (nx - 1)
    if iy_pos >= cy:
        iy_pos_s, iy_neg_s = cy + iy_mag, cy - iy_mag
    else:
        iy_pos_s, iy_neg_s = cy - iy_mag, cy + iy_mag
    if ix_pos >= cx:
        ix_pos_s, ix_neg_s = cx + ix_mag, cx - ix_mag
    else:
        ix_pos_s, ix_neg_s = cx - ix_mag, cx + ix_mag

    pos_c = (float(x0 + ix_pos_s * dx), float(y0 + iy_pos_s * dy))
    neg_c = (float(x0 + ix_neg_s * dx), float(y0 + iy_neg_s * dy))
    return pos_c, neg_c


def _td3_r1_axis_centers(
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Resolve ± R1 foot centres: manual y, Jz blobs, or conjugate search.

    Priority
    --------
    1. ``td3_r1_axis_y_mm`` > 0 → ``(0, +y)`` and ``(0, -y)`` (Bz-map pick)
    2. ``td3_r1_axis_from_jz`` → ± bottom-Jz blob centres
    3. otherwise → conjugate footpoint search near y≈80 Mm
    """
    manual_y = TD_VIZ.td3_r1_axis_y_mm
    if manual_y is not None and float(manual_y) > 0.0:
        y = abs(float(manual_y))
        return (0.0, y), (0.0, -y)
    if bool(TD_VIZ.td3_r1_axis_from_jz):
        return _td3_jz_flux_rope_axis_centers(fields, case)
    axis_pos, axis_neg, _axis_y = _find_symmetric_axis_seeds(
        case, fields, y_guess=80.0
    )
    return (
        (float(axis_pos[0, 0]), float(axis_pos[0, 1])),
        (float(axis_neg[0, 0]), float(axis_neg[0, 1])),
    )


def _bbox_xz_from_polylines(
    polylines: list[np.ndarray],
    *,
    pad_mm: float = 0.0,
    symmetrize_x: bool = True,
    y_tol_mm: float | None = 5.0,
) -> tuple[float, float, float, float] | None:
    """Axis-aligned x–z box touching field-line samples (optionally near y=0).

    When ``y_tol_mm`` is set, only samples with ``|y| <= y_tol_mm`` contribute.
    This matches Fig.4 R1/R2/R3 rectangles drawn on the y=0 Q map.
    """
    xs: list[float] = []
    zs: list[float] = []
    for poly in polylines:
        pts = np.asarray(poly, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
            continue
        if y_tol_mm is not None:
            mask = np.abs(pts[:, 1]) <= float(y_tol_mm)
            if not np.any(mask):
                continue
            pts = pts[mask]
        xs.extend(pts[:, 0].tolist())
        zs.extend(pts[:, 2].tolist())
    if not xs or not zs:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    z_arr = np.asarray(zs, dtype=np.float64)
    pad = float(pad_mm)
    z0 = float(z_arr.min()) - pad
    z1 = float(z_arr.max()) + pad
    if symmetrize_x:
        half = float(np.max(np.abs(x_arr))) + pad
        return (-half, half, z0, z1)
    return (
        float(x_arr.min()) - pad,
        float(x_arr.max()) + pad,
        z0,
        z1,
    )


def symmetric_xz_box_from_edge_points(
    x: float,
    z_a: float,
    z_b: float,
) -> tuple[float, float, float, float]:
    """Build an x=0-symmetric rectangle from one vertical edge (x, z_a)–(x, z_b)."""
    half = abs(float(x))
    z0 = min(float(z_a), float(z_b))
    z1 = max(float(z_a), float(z_b))
    return (-half, half, z0, z1)



def _box_plane_seeds(
    bounds: tuple[float, float, float, float],
    *,
    n_per_side: int | None = None,
    n_x: int | None = None,
    n_z: int | None = None,
) -> np.ndarray:
    """Uniform Cartesian seeds filling a y=0 ROI rectangle."""
    xmin, xmax, zmin, zmax = bounds
    if n_x is None or n_z is None:
        n = max(2, int(n_per_side if n_per_side is not None else 6))
        n_x = n if n_x is None else int(n_x)
        n_z = n if n_z is None else int(n_z)
    n_x = max(1, int(n_x))
    n_z = max(1, int(n_z))
    xs = np.linspace(xmin, xmax, n_x)
    zs = np.linspace(zmin, zmax, n_z)
    xx, zz = np.meshgrid(xs, zs, indexing="xy")
    return np.column_stack(
        [xx.ravel(), np.zeros(xx.size), zz.ravel()]
    ).astype(np.float32)


def _y0_plane_index(case: dict[str, Any]) -> int:
    _nx, ny, _nz, _dx, dy, _dz = case["dims"]
    _x0, y0, _z0 = _grid_origin(case)
    jy = int(round((0.0 - y0) / float(dy)))
    return int(max(0, min(ny - 1, jy)))


def _y0_bmag(
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (|B| on y=0 as nz×nx, xs_mm, zs_mm)."""
    nx, _ny, nz, dx, _dy, dz = case["dims"]
    x0, _y0, z0 = _grid_origin(case)
    jy = _y0_plane_index(case)
    stacked = _stacked_b(fields)
    bmag = np.linalg.norm(np.asarray(stacked[:, :, jy, :], dtype=np.float64), axis=0)
    xs = x0 + np.arange(nx, dtype=np.float64) * float(dx)
    zs = z0 + np.arange(nz, dtype=np.float64) * float(dz)
    return bmag, xs, zs


def _closed_isocontours_xz(
    field_zx: np.ndarray,
    xs: np.ndarray,
    zs: np.ndarray,
    level: float,
    *,
    min_points: int,
    min_area_mm2: float,
) -> list[np.ndarray]:
    """Closed iso-contours of a z×x field as polylines ``(N, 2)`` in (x, z) Mm."""
    from skimage import measure

    contours: list[np.ndarray] = []
    for contour in measure.find_contours(np.asarray(field_zx, dtype=np.float64), float(level)):
        if contour.shape[0] < int(min_points):
            continue
        if float(np.hypot(contour[0, 0] - contour[-1, 0], contour[0, 1] - contour[-1, 1])) > 4.0:
            continue
        cz = np.interp(contour[:, 0], np.arange(len(zs), dtype=np.float64), zs)
        cx = np.interp(contour[:, 1], np.arange(len(xs), dtype=np.float64), xs)
        area = 0.5 * abs(
            float(np.dot(cx, np.roll(cz, 1)) - np.dot(cz, np.roll(cx, 1)))
        )
        if area < float(min_area_mm2):
            continue
        poly = np.column_stack([cx, cz]).astype(np.float64)
        # Ensure geometric closure for arc-length resampling.
        if float(np.hypot(*(poly[0] - poly[-1]))) > 1.0e-6:
            poly = np.vstack([poly, poly[0]])
        contours.append(poly)
    contours.sort(
        key=lambda poly: -0.5
        * abs(
            float(
                np.dot(poly[:, 0], np.roll(poly[:, 1], 1))
                - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))
            )
        )
    )
    return contours


def _clip_polyline_to_box(
    poly_xz: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    margin_mm: float,
) -> np.ndarray | None:
    """Longest contiguous arc of ``poly_xz`` that lies inside the ROI box."""
    xmin, xmax, zmin, zmax = bounds
    m = (
        (poly_xz[:, 0] >= xmin - margin_mm)
        & (poly_xz[:, 0] <= xmax + margin_mm)
        & (poly_xz[:, 1] >= zmin - margin_mm)
        & (poly_xz[:, 1] <= zmax + margin_mm)
    )
    if int(m.sum()) < 5:
        return None
    poly = np.asarray(poly_xz, dtype=np.float64)
    mask = np.asarray(m, dtype=bool)
    # Rotate a wrap-around arc so the kept run is contiguous.
    if mask[0] and mask[-1] and (not bool(mask.all())):
        gaps = np.where(~mask)[0]
        if gaps.size:
            rot = int(gaps[0] + 1)
            mask = np.roll(mask, -rot)
            poly = np.roll(poly, -rot, axis=0)
    idx = np.where(mask)[0]
    if idx.size < 5:
        return None
    breaks = np.where(np.diff(idx) > 1)[0]
    if breaks.size == 0:
        return poly[idx]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, idx.size - 1]
    lengths = ends - starts + 1
    k = int(np.argmax(lengths))
    return poly[idx[starts[k] : ends[k] + 1]]


def _equal_arc_samples_xz(
    poly_xz: np.ndarray,
    n: int,
    *,
    closed: bool,
) -> np.ndarray:
    """Resample a polyline to ``n`` equal-arc-length samples in (x, z)."""
    poly = np.asarray(poly_xz, dtype=np.float64)
    if poly.shape[0] < 2 or n <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    if closed and float(np.hypot(*(poly[0] - poly[-1]))) > 1.0e-6:
        poly = np.vstack([poly, poly[0]])
    seg = np.sqrt(np.sum(np.diff(poly, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0.0:
        return np.repeat(poly[:1], n, axis=0)
    # Closed loops: omit the duplicated endpoint so samples stay unique.
    t = np.linspace(0.0, total, int(n), endpoint=not closed)
    return np.column_stack(
        [np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])]
    )


def _polyline_enters_xz_box(
    poly: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> bool:
    """True if any polyline sample lies inside the x–z box (y unrestricted)."""
    xmin, xmax, zmin, zmax = bounds
    pts = np.asarray(poly, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return False
    return bool(
        np.any(
            (pts[:, 0] >= xmin)
            & (pts[:, 0] <= xmax)
            & (pts[:, 2] >= zmin)
            & (pts[:, 2] <= zmax)
        )
    )


def _ring_all_closed_enter_r1(
    center_xy: tuple[float, float],
    radius: float,
    z: float,
    *,
    n: int,
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
    r1_bounds: tuple[float, float, float, float],
    max_steps: int = 2500,
) -> bool:
    """True if every ring seed yields a closed line that intersects R1."""
    if float(radius) <= 0.0 or int(n) <= 0:
        return False
    for seed in _ring_seeds(center_xy, float(radius), float(z), n=int(n)):
        poly = _integrate_closed_polyline(
            fields, seed, case, max_steps=max_steps, step_frac=0.7
        )
        if poly is None or not _polyline_enters_xz_box(poly, r1_bounds):
            return False
    return True


def _max_r1_footpoint_ring_radius(
    center_xy: tuple[float, float],
    z: float,
    *,
    n: int,
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
    r1_bounds: tuple[float, float, float, float],
) -> float:
    """Largest ring radius where all seeds stay closed and pass through R1."""
    cfg = TD_VIZ
    r_min = float(cfg.td3_r1_radius_min_mm)
    r_max = float(cfg.td3_r1_radius_max_mm)
    n_coarse = max(4, int(cfg.td3_r1_radius_n_coarse))
    candidates = np.linspace(r_min, r_max, n_coarse)
    best = 0.0
    for r in candidates:
        if _ring_all_closed_enter_r1(
            center_xy, float(r), z, n=n, fields=fields, case=case, r1_bounds=r1_bounds
        ):
            best = float(r)
        elif best > 0.0:
            break
    if best <= 0.0:
        # Fallback: try the minimum radius alone.
        if _ring_all_closed_enter_r1(
            center_xy, r_min, z, n=n, fields=fields, case=case, r1_bounds=r1_bounds
        ):
            return r_min
        return r_min
    # Binary refine between best and the first failing neighbour.
    lo = best
    hi = min(r_max, best + (r_max - r_min) / max(n_coarse - 1, 1))
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if _ring_all_closed_enter_r1(
            center_xy, mid, z, n=n, fields=fields, case=case, r1_bounds=r1_bounds
        ):
            lo = mid
        else:
            hi = mid
    return float(lo)


def _td3_r1_footpoint_ring_seeds(
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
    r1_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float, float, float] | None]:
    """Sample R1 like TD1 outer rope: rings at ± flux-rope footpoints.

    Default: largest common radius such that every seed on both rings produces
    a closed field line that enters the R1 prism.

    Interactive / YAML mode (``td3_r1_ring_radius_mm`` set):
      - centre on manual ``td3_r1_axis_y_mm``, ± bottom-Jz blobs, or
        conjugate-search axes (see ``_td3_r1_axis_centers``)
      - use the fixed radius
      - auto-fit an x=0-symmetric box that touches the traced field-line region
    """
    _nx, _ny, _nz, _dx, _dy, dz = case["dims"]
    z = 0.5 * float(dz)
    n = int(TD_VIZ.td3_r1_seeds_per_polarity)
    fixed_radius = TD_VIZ.td3_r1_ring_radius_mm

    pos_c, neg_c = _td3_r1_axis_centers(fields, case)

    if fixed_radius is not None and float(fixed_radius) > 0.0:
        radius = float(fixed_radius)
    else:
        r_pos = _max_r1_footpoint_ring_radius(
            pos_c, z, n=n, fields=fields, case=case, r1_bounds=r1_bounds
        )
        r_neg = _max_r1_footpoint_ring_radius(
            neg_c, z, n=n, fields=fields, case=case, r1_bounds=r1_bounds
        )
        radius = float(min(r_pos, r_neg))

    pos_ring = _filter_closed_seed_array(
        _ring_seeds(pos_c, radius, z, n=n), fields, case
    )
    neg_ring = _filter_closed_seed_array(
        _ring_seeds(neg_c, radius, z, n=n), fields, case
    )

    auto_box: tuple[float, float, float, float] | None = None
    # Fixed-radius / Jz-axis mode: keep closed seeds and fit the R1 box to lines.
    if fixed_radius is not None and float(fixed_radius) > 0.0:
        polys: list[np.ndarray] = []
        kept_pos: list[np.ndarray] = []
        kept_neg: list[np.ndarray] = []
        for seed in np.asarray(pos_ring, dtype=np.float64):
            poly = _integrate_closed_polyline(
                fields, seed, case, max_steps=2500, step_frac=0.7
            )
            if poly is None:
                continue
            kept_pos.append(seed.astype(np.float32))
            polys.append(poly)
        for seed in np.asarray(neg_ring, dtype=np.float64):
            poly = _integrate_closed_polyline(
                fields, seed, case, max_steps=2500, step_frac=0.7
            )
            if poly is None:
                continue
            kept_neg.append(seed.astype(np.float32))
            polys.append(poly)
        auto_box = _bbox_xz_from_polylines(
            polys,
            pad_mm=float(TD_VIZ.td3_r1_bbox_pad_mm),
            symmetrize_x=True,
            y_tol_mm=float(TD_VIZ.td3_r1_bbox_y_tol_mm),
        )
        pos_out = (
            np.vstack(kept_pos)
            if kept_pos
            else np.zeros((0, 3), dtype=np.float32)
        )
        neg_out = (
            np.vstack(kept_neg)
            if kept_neg
            else np.zeros((0, 3), dtype=np.float32)
        )
        return pos_out, neg_out, radius, auto_box

    # Legacy: keep only lines that enter the provided R1 prism.
    def _keep_enter(seeds: np.ndarray) -> np.ndarray:
        kept: list[np.ndarray] = []
        for seed in np.asarray(seeds, dtype=np.float64):
            poly = _integrate_closed_polyline(
                fields, seed, case, max_steps=2500, step_frac=0.7
            )
            if poly is not None and _polyline_enters_xz_box(poly, r1_bounds):
                kept.append(seed.astype(np.float32))
        if not kept:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(kept)

    return _keep_enter(pos_ring), _keep_enter(neg_ring), radius, None


def _strong_flux_outer_seeds(
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
    bounds: tuple[float, float, float, float],
    *,
    n_seeds: int,
) -> np.ndarray:
    """Equal-arc seeds on the outer |B| contour of the strong-flux lobe in a ROI.

    Scans closed |B| iso-contours on y=0 and keeps the one with the strongest
    overlap with ``bounds`` (R2--R3 HFT-associated lobe). Falls back to a
    sparse Cartesian grid if no usable contour is found.
    """
    bmag, xs, zs = _y0_bmag(fields, case)
    bmax = float(np.nanmax(bmag))
    if not np.isfinite(bmax) or bmax <= 0.0:
        n_side = max(3, int(round(math.sqrt(max(n_seeds, 1)))))
        return _box_plane_seeds(bounds, n_per_side=n_side)

    cfg = TD_VIZ
    fracs = np.linspace(
        cfg.td3_flux_level_frac_min,
        cfg.td3_flux_level_frac_max,
        max(3, int(cfg.td3_flux_level_n)),
    )
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for frac in fracs:
        for poly in _closed_isocontours_xz(
            bmag,
            xs,
            zs,
            float(frac) * bmax,
            min_points=cfg.td3_flux_min_contour_points,
            min_area_mm2=cfg.td3_flux_min_contour_area_mm2,
        ):
            arc = _clip_polyline_to_box(
                poly, bounds, margin_mm=cfg.td3_flux_contour_margin_mm
            )
            if arc is None or arc.shape[0] < int(cfg.td3_flux_min_arc_points):
                continue
            # Prefer longer ROI overlap, then larger parent contour (outer lobe).
            area = 0.5 * abs(
                float(
                    np.dot(poly[:, 0], np.roll(poly[:, 1], 1))
                    - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))
                )
            )
            score = float(arc.shape[0]) + 1.0e-3 * area
            if best is None or score > best[0]:
                best = (score, area, poly, arc)

    if best is None:
        n_side = max(3, int(round(math.sqrt(max(n_seeds, 1)))))
        return _box_plane_seeds(bounds, n_per_side=n_side)

    _score, _area, poly, arc = best
    # If the clipped arc is almost the full loop, sample the closed outer boundary.
    poly_len = float(np.sum(np.sqrt(np.sum(np.diff(poly, axis=0) ** 2, axis=1))))
    arc_len = float(np.sum(np.sqrt(np.sum(np.diff(arc, axis=0) ** 2, axis=1))))
    use_closed = bool(poly_len > 0.0 and arc_len >= 0.85 * poly_len)
    samples = _equal_arc_samples_xz(
        poly if use_closed else arc,
        int(n_seeds),
        closed=use_closed,
    )
    if samples.shape[0] == 0:
        n_side = max(3, int(round(math.sqrt(max(n_seeds, 1)))))
        return _box_plane_seeds(bounds, n_per_side=n_side)
    return np.column_stack(
        [
            samples[:, 0],
            np.zeros(samples.shape[0], dtype=np.float64),
            samples[:, 1],
        ]
    ).astype(np.float32)


def _filter_closed_box_seeds(
    seeds: np.ndarray,
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Keep only seeds whose field lines hit the photosphere at both ends."""
    kept: list[np.ndarray] = []
    foots: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in np.asarray(seeds, dtype=np.float64):
        fp_a, fp_b, status = _trace_footpoints(
            fields, seed, case, max_steps=700, step_frac=1.25
        )
        if status == "closed" and fp_a is not None and fp_b is not None:
            kept.append(seed.astype(np.float32))
            foots.append((fp_a, fp_b))
    if not kept:
        return np.zeros((0, 3), dtype=np.float32), []
    return np.vstack(kept), foots


def _filter_r3_foots_inside_rope_span(
    seeds: np.ndarray,
    foots: list[tuple[np.ndarray, np.ndarray]],
    rope_y_abs: float,
) -> np.ndarray:
    """Keep R3 seeds whose both photospheric feet have |y| < rope foot |y|.

    This selects short connections inside the flux-rope footpoint span rather
    than long arches that land at or beyond the rope feet.
    """
    y_lim = abs(float(rope_y_abs))
    if y_lim <= 0.0 or len(foots) == 0:
        return np.asarray(seeds, dtype=np.float32)
    kept: list[np.ndarray] = []
    for seed, (fp_a, fp_b) in zip(np.asarray(seeds, dtype=np.float64), foots, strict=True):
        ya = abs(float(fp_a[1]))
        yb = abs(float(fp_b[1]))
        if ya < y_lim and yb < y_lim:
            kept.append(seed.astype(np.float32))
    if not kept:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(kept)


def _bottom_abs_jz(
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
) -> tuple[np.ndarray, float, float, float, float, int, int]:
    """Return bottom |Jz|(y,x) plus grid origin/spacing for footpoint sampling."""
    nx, ny, _nz, dx, dy, dz = case["dims"]
    _jx, _jy, jz = curl_zyx_nfpt(
        fields["Bx"], fields["By"], fields["Bz"], dx=dx, dy=dy, dz=dz
    )
    jz0 = np.abs(np.asarray(jz[0], dtype=np.float64))
    x0, y0, _z0 = _grid_origin(case)
    return jz0, float(x0), float(y0), float(dx), float(dy), int(nx), int(ny)


def _bottom_strong_jz_distance_mm(
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
    *,
    percentile: float,
) -> tuple[np.ndarray, float, float, float, float, float, int, int]:
    """Distance (Mm) from each bottom pixel to the nearest strong-|Jz| pixel.

    Strong pixels satisfy |Jz| >= the given percentile of the bottom layer.
    Returns (dist_mm[y,x], jz_threshold, x0, y0, dx, dy, nx, ny).
    """
    from scipy.ndimage import distance_transform_edt

    jz0, x0, y0, dx, dy, nx, ny = _bottom_abs_jz(fields, case)
    thr = float(np.percentile(jz0, percentile))
    strong = jz0 >= thr
    if not np.any(strong):
        dist = np.full(jz0.shape, np.inf, dtype=np.float64)
    else:
        # sampling=(dy, dx) so EDT returns physical Mm on the (y, x) plane.
        dist = distance_transform_edt(~strong, sampling=(dy, dx)).astype(np.float64)
    return dist, thr, x0, y0, dx, dy, nx, ny


def _sample_bottom_scalar(
    field_yx: np.ndarray,
    foot: np.ndarray,
    *,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> float:
    i = int(np.clip(round((float(foot[0]) - x0) / dx), 0, nx - 1))
    j = int(np.clip(round((float(foot[1]) - y0) / dy), 0, ny - 1))
    return float(field_yx[j, i])


def _filter_r2_foots_on_strong_jz(
    seeds: np.ndarray,
    foots: list[tuple[np.ndarray, np.ndarray]],
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
    *,
    percentile: float | None = None,
    max_distance_mm: float | None = None,
    require_both: bool | None = None,
) -> tuple[np.ndarray, float]:
    """Keep R2 seeds whose photospheric feet are near strong bottom |Jz|.

    Strong |Jz| = bottom pixels at/above ``percentile``. A foot passes if its
    planar distance to the nearest strong pixel is strictly less than
    ``max_distance_mm``. By default both feet must pass.
    Returns (kept_seeds, absolute |Jz| threshold used for the strong mask).
    """
    pct = float(
        TD_VIZ.td3_r2_jz_percentile if percentile is None else percentile
    )
    dmax = float(
        TD_VIZ.td3_r2_jz_max_distance_mm
        if max_distance_mm is None
        else max_distance_mm
    )
    both = bool(
        TD_VIZ.td3_r2_jz_require_both_feet
        if require_both is None
        else require_both
    )
    if len(foots) == 0:
        return np.asarray(seeds, dtype=np.float32), 0.0

    dist, thr, x0, y0, dx, dy, nx, ny = _bottom_strong_jz_distance_mm(
        fields, case, percentile=pct
    )
    kept: list[np.ndarray] = []
    for seed, (fp_a, fp_b) in zip(np.asarray(seeds, dtype=np.float64), foots, strict=True):
        da = _sample_bottom_scalar(
            dist, fp_a, x0=x0, y0=y0, dx=dx, dy=dy, nx=nx, ny=ny
        )
        db = _sample_bottom_scalar(
            dist, fp_b, x0=x0, y0=y0, dx=dx, dy=dy, nx=nx, ny=ny
        )
        ok_a = da < dmax
        ok_b = db < dmax
        ok = (ok_a and ok_b) if both else (ok_a or ok_b)
        if ok:
            kept.append(seed.astype(np.float32))
    if not kept:
        return np.zeros((0, 3), dtype=np.float32), thr
    return np.vstack(kept), thr


def _split_r1_by_polarity(
    seeds: np.ndarray,
    foots: list[tuple[np.ndarray, np.ndarray]],
    fields: dict[str, np.ndarray],
    case: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Fig.4: colour R1 lines by the nearer photospheric polarity."""
    pos: list[np.ndarray] = []
    neg: list[np.ndarray] = []
    for seed, (fp_a, fp_b) in zip(np.asarray(seeds, dtype=np.float64), foots, strict=True):
        # Nearer footpoint along Euclidean distance from the y=0 seed.
        d_a = float(np.linalg.norm(fp_a - seed))
        d_b = float(np.linalg.norm(fp_b - seed))
        near = fp_a if d_a <= d_b else fp_b
        bz_near = float(_sample_b_at(fields, near, case)[2])
        if bz_near >= 0.0:
            pos.append(seed.astype(np.float32))
        else:
            neg.append(seed.astype(np.float32))
    pos_arr = np.vstack(pos) if pos else np.zeros((0, 3), dtype=np.float32)
    neg_arr = np.vstack(neg) if neg else np.zeros((0, 3), dtype=np.float32)
    return pos_arr, neg_arr


_TD3_SEED_CACHE: dict[
    tuple[Any, ...],
    tuple[list[dict[str, Any]], dict[str, tuple[float, float, float, float]], float],
] = {}


def _td3_fig4_seed_groups(
    case: dict[str, Any],
    q: np.ndarray,
    geometry: dict[str, Any],
    *,
    fields: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float, float, float]], float]:
    """Fig.4 seed policy: R1 footpoint rings; R2 contour; R3 box grid.

    R1 matches the TD1 outer-rope style: rings around ± flux-rope footpoints
    at the largest radius where every seed stays closed and enters the R1
    prism. R2 samples the HFT-associated |B| lobe, keeping only closed lines
    whose photospheric feet lie within ``td3_r2_jz_max_distance_mm`` of a
    strong bottom-|Jz| patch. R3 fills the HFT-relative R3 prism (same box
    rule as R1/R2) with a uniform y=0 Cartesian grid, keeping only closed
    lines whose photospheric feet both satisfy |y| < rope-foot |y|.
    """
    ref_fields = fields if fields is not None else case["reference"]
    cache_key = (id(case), id(ref_fields), id(q), float(np.nanmax(q)))
    cached = _TD3_SEED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    x_hft, z_hft = _locate_hft(q, geometry)
    boxes = _hft_box_bounds(x_hft, z_hft, q=q, geometry=geometry)
    _nx, _ny, _nz, dx, dy, dz = case["dims"]
    thin = TD_VIZ.tube_radius_td3_thin * min(dx, dy, dz)

    # Analytic: always build R1/R2/R3 on this field.  NFGS: rebuild when
    # ``td3_nfgs_rebuild_seeds`` (per-figure YAML) is on; otherwise reuse
    # analytic seed positions and only re-check closed lines.
    analytic = case["reference"]
    using_analytic = ref_fields is analytic or (
        ref_fields.get("Bx") is analytic.get("Bx")
    )
    rebuild_nfgs = bool(TD_VIZ.td3_nfgs_rebuild_seeds)
    print(
        f"    TD3 filtering closed seeds (HFT~{x_hft:.1f},{z_hft:.1f}) ...",
        flush=True,
    )
    if using_analytic or rebuild_nfgs:
        if not using_analytic:
            print("    TD3 NFGS: rebuilding R1/R2/R3 from solution field", flush=True)
        r1_pos, r1_neg, r1_radius, r1_auto_box = _td3_r1_footpoint_ring_seeds(
            ref_fields, case, boxes["R1"]
        )
        if r1_auto_box is not None:
            boxes["R1"] = r1_auto_box
        axis_mode = (
            "manual-y"
            if TD_VIZ.td3_r1_axis_y_mm is not None
            and float(TD_VIZ.td3_r1_axis_y_mm) > 0.0
            else ("jz-axis" if TD_VIZ.td3_r1_axis_from_jz else "sym-search")
        )
        print(
            f"    TD3 R1 footpoint ring radius={r1_radius:.2f} Mm "
            f"(n={TD_VIZ.td3_r1_seeds_per_polarity}/polarity"
            f", {axis_mode})",
            flush=True,
        )
        # Rope foot |y| used to keep R3 arches inside the rope span.
        # Prefer the same R1 axis |y| (manual or Jz) so R3 filtering matches
        # the rings; otherwise start from the canonical 80 Mm guess — never
        # reuse a prior solution's ``axis_y_phys`` left on the shared case.
        manual_y = TD_VIZ.td3_r1_axis_y_mm
        if manual_y is not None and float(manual_y) > 0.0:
            rope_y = abs(float(manual_y))
        elif bool(TD_VIZ.td3_r1_axis_from_jz):
            pos_c, neg_c = _td3_jz_flux_rope_axis_centers(ref_fields, case)
            rope_y = 0.5 * (abs(float(pos_c[1])) + abs(float(neg_c[1])))
        else:
            _axis_pos, _axis_neg, rope_y = _find_symmetric_axis_seeds(
                case, ref_fields, y_guess=80.0
            )
        r2_raw = _strong_flux_outer_seeds(
            ref_fields, case, boxes["R2"], n_seeds=TD_VIZ.td3_r2_n_seeds
        )
        r3_raw = _box_plane_seeds(
            boxes["R3"],
            n_x=TD_VIZ.td3_r3_nx,
            n_z=TD_VIZ.td3_r3_nz,
        )
        r2_closed, r2_foots = _filter_closed_box_seeds(r2_raw, ref_fields, case)
        n_r2_closed = len(r2_closed)
        r2_closed, jz_thr = _filter_r2_foots_on_strong_jz(
            r2_closed, r2_foots, ref_fields, case
        )
        print(
            f"    TD3 R2 feet near strong |Jz| "
            f"(p{TD_VIZ.td3_r2_jz_percentile:.0f} thr={jz_thr:.3e}, "
            f"d<{TD_VIZ.td3_r2_jz_max_distance_mm:.1f} Mm, "
            f"both={TD_VIZ.td3_r2_jz_require_both_feet}): "
            f"{len(r2_closed)}/{n_r2_closed}",
            flush=True,
        )
        r3_closed, r3_foots = _filter_closed_box_seeds(r3_raw, ref_fields, case)
        n_r3_closed = len(r3_closed)
        r3_closed = _filter_r3_foots_inside_rope_span(r3_closed, r3_foots, rope_y)
        print(
            f"    TD3 R3 foot |y| < rope |y|={abs(float(rope_y)):.1f} Mm: "
            f"{len(r3_closed)}/{n_r3_closed}",
            flush=True,
        )
    else:
        # Prefer already-built analytic groups (same R1 footpoint rings as ref).
        analytic_groups = None
        for key, cached_val in _TD3_SEED_CACHE.items():
            if key[0] == id(case) and key[1] == id(analytic):
                analytic_groups = cached_val[0]
                break
        if analytic_groups is None:
            analytic_groups, _, _ = _td3_fig4_seed_groups(
                case, q, geometry, fields=analytic
            )
        # Reuse analytic seed positions; keep those that stay closed in solution.
        r1_pos_raw = next(
            (g["seeds"] for g in analytic_groups if g["label"] == "R1_pos"),
            np.zeros((0, 3), dtype=np.float32),
        )
        r1_neg_raw = next(
            (g["seeds"] for g in analytic_groups if g["label"] == "R1_neg"),
            np.zeros((0, 3), dtype=np.float32),
        )
        r2_raw = next(
            (g["seeds"] for g in analytic_groups if g["label"] == "R2"),
            np.zeros((0, 3), dtype=np.float32),
        )
        r3_raw = next(
            (g["seeds"] for g in analytic_groups if g["label"] == "R3"),
            np.zeros((0, 3), dtype=np.float32),
        )
        # R1 matches TD1: same analytic footpoint-ring seeds for ref and NFGS.
        r1_pos = np.asarray(r1_pos_raw, dtype=np.float32)
        r1_neg = np.asarray(r1_neg_raw, dtype=np.float32)
        # Analytic R2 already passed the strong-|Jz| foot cut; only re-check closed.
        r2_closed, _ = _filter_closed_box_seeds(r2_raw, ref_fields, case)
        # Analytic R3 already passed the rope-span cut; only re-check closed.
        r3_closed, _ = _filter_closed_box_seeds(r3_raw, ref_fields, case)
    print(
        f"    TD3 closed seeds: R1+={len(r1_pos)} R1-={len(r1_neg)} "
        f"R2={len(r2_closed)} R3={len(r3_closed)}",
        flush=True,
    )

    groups = [
        {
            "seeds": r1_pos,
            "color": TD_VIZ.color_rope_pos,
            "radius": thin,
            "label": "R1_pos",
        },
        {
            "seeds": r1_neg,
            "color": TD_VIZ.color_rope_neg,
            "radius": thin,
            "label": "R1_neg",
        },
        {"seeds": r2_closed, "color": TD_VIZ.color_td3_r2, "radius": thin, "label": "R2"},
        {"seeds": r3_closed, "color": TD_VIZ.color_td3_r3, "radius": thin, "label": "R3"},
    ]
    result = (groups, boxes, z_hft)
    _TD3_SEED_CACHE[cache_key] = result
    return result


def _yi_q_cmap():
    """Fig.4 (a1) Q colormap: white -> pink/magenta -> blue."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "yi_q_white_pink_blue",
        [
            (0.00, "#ffffff"),
            (0.20, "#f7d4e4"),
            (0.45, "#e879b0"),
            (0.70, "#6b7fe0"),
            (1.00, "#1e3a8a"),
        ],
    )


def _interpolate_nan_q(
    q: np.ndarray,
    *,
    max_passes: int = 2,
    min_neighbors: int = 5,
    kernel_size: int = 3,
) -> np.ndarray:
    """Fill tiny NaN holes on the 2D y=0 Q plane for display only.

    Uses only in-plane ``(z, x)`` neighbours (local mean).  A NaN cell is filled
    only when at least ``min_neighbors`` finite samples lie in its
    ``kernel_size`` window.  Large FastQSL NaN patches from side-wall / flag
    discontinuities therefore stay NaN and render as background, instead of
    being nearest-filled into artificial low-Q.  Cached Q for HFT / seeds is
    never modified.
    """
    from scipy.ndimage import uniform_filter

    arr = np.asarray(q, dtype=np.float64)
    out = arr.copy()
    if not np.any(~np.isfinite(out)):
        return out
    if not np.any(np.isfinite(out)):
        return out

    size = int(kernel_size)
    if size < 3 or size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    scale = float(size * size)
    for _ in range(int(max_passes)):
        bad = ~np.isfinite(out)
        if not np.any(bad):
            break
        finite = np.isfinite(out)
        mask = finite.astype(np.float64)
        vals = np.where(finite, out, 0.0)
        count = uniform_filter(mask, size=size, mode="constant", cval=0.0) * scale
        total = uniform_filter(vals, size=size, mode="constant", cval=0.0) * scale
        local_mean = total / np.maximum(count, 1.0)
        fill = bad & (count >= float(min_neighbors))
        if not np.any(fill):
            break
        out[fill] = local_mean[fill]
    return out


def _prepare_q_for_plot(q: np.ndarray, *, vmax: float) -> np.ndarray:
    """Locally interpolate small NaNs then clip; large holes stay NaN."""
    return np.clip(_interpolate_nan_q(q), 0.0, float(vmax))


def _q_display_vmax(q: np.ndarray, *, high_logq: float) -> float:
    """Linear-Q colour ceiling matching Fig.4 (~0--100)."""
    del q, high_logq
    return float(TD_VIZ.qmap_colorbar_vmax)


def _q_plane_mesh(pv, q: np.ndarray, geometry: dict[str, Any], *, vmax: float):
    q_plot = _prepare_q_for_plot(q, vmax=vmax)
    nz, nx = q_plot.shape
    x0, x1 = geometry["x_bounds_mm"]
    z0, z1 = geometry["z_bounds_mm"]
    dx = (x1 - x0) / max(nx - 1, 1)
    dz = (z1 - z0) / max(nz - 1, 1)
    plane = pv.ImageData(
        dimensions=(nx, 1, nz),
        spacing=(dx, 1.0, dz),
        origin=(x0, 0.0, z0),
    )
    plane.point_data["Q"] = np.ascontiguousarray(q_plot.ravel(), dtype=np.float32)
    return plane


def _qmap_display_window_mm() -> tuple[float, float, float, float]:
    """x/z limits drawn by the TD3 Q-map panel, in Mm."""
    x0 = float(TD_VIZ.qmap_x_min)
    x1 = float(TD_VIZ.qmap_x_max)
    x_half = max(abs(x0), abs(x1))
    return (-x_half, x_half, float(TD_VIZ.qmap_z_min), float(TD_VIZ.qmap_z_max))


def _q_plane_mesh_roi(
    pv,
    q: np.ndarray,
    geometry: dict[str, Any],
    *,
    vmax: float,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
):
    """y=0 Q plane restricted to samples inside the requested x/z window."""
    q_full = _prepare_q_for_plot(q, vmax=vmax)
    nz, nx = q_full.shape
    gx0, gx1 = geometry["x_bounds_mm"]
    gz0, gz1 = geometry["z_bounds_mm"]
    xs = np.linspace(gx0, gx1, nx)
    zs = np.linspace(gz0, gz1, nz)
    # Inclusive window, no extra cell outside the Q-map limits.
    ix0 = int(np.searchsorted(xs, x_min, side="left"))
    ix1 = int(np.searchsorted(xs, x_max, side="right"))
    iz0 = int(np.searchsorted(zs, z_min, side="left"))
    iz1 = int(np.searchsorted(zs, z_max, side="right"))
    ix0 = max(0, min(ix0, nx - 1))
    ix1 = max(ix0 + 1, min(ix1, nx))
    iz0 = max(0, min(iz0, nz - 1))
    iz1 = max(iz0 + 1, min(iz1, nz))
    q_plot = q_full[iz0:iz1, ix0:ix1]
    sub_nz, sub_nx = q_plot.shape
    x0 = float(xs[ix0])
    z0 = float(zs[iz0])
    dx = float(xs[1] - xs[0]) if nx > 1 else 1.0
    dz = float(zs[1] - zs[0]) if nz > 1 else 1.0
    plane = pv.ImageData(
        dimensions=(sub_nx, 1, sub_nz),
        spacing=(dx, 1.0, dz),
        origin=(x0, 0.0, z0),
    )
    plane.point_data["Q"] = np.ascontiguousarray(q_plot.ravel(), dtype=np.float32)
    return plane


def _qmap_roi_box_lines(
    pv,
    boxes: dict[str, tuple[float, float, float, float]],
) -> list[tuple[Any, str]]:
    """Polyline meshes for R1/R2/R3 rectangles on the y=0 Q plane."""
    colors = {
        "R1": TD_VIZ.qmap_box_r1,
        "R2": TD_VIZ.qmap_box_r2,
        "R3": TD_VIZ.qmap_box_r3,
    }
    out: list[tuple[Any, str]] = []
    for name, (xmin, xmax, zmin, zmax) in boxes.items():
        corners = np.asarray(
            [
                [xmin, 0.0, zmin],
                [xmax, 0.0, zmin],
                [xmax, 0.0, zmax],
                [xmin, 0.0, zmax],
                [xmin, 0.0, zmin],
            ],
            dtype=np.float32,
        )
        out.append((pv.lines_from_points(corners), colors.get(name, "white")))
    return out


def _export_qmap_interactive_html(
    q: np.ndarray,
    geometry: dict[str, Any],
    *,
    high_logq: float,
    html_path: Path,
    boxes: dict[str, tuple[float, float, float, float]] | None = None,
    title: str = "Q map",
) -> None:
    """Build a lightweight PyVista Q-plane scene and export interactive HTML."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    vmax = _q_display_vmax(q, high_logq=high_logq)
    q_plane = _q_plane_mesh(pv, q, geometry, vmax=vmax)
    window_size = _unified_panel_window_size()
    plotter = pv.Plotter(window_size=list(window_size), border=False, off_screen=True)
    plotter.set_background(TD_VIZ.pv_background)
    plotter.add_text(str(title), font_size=12, color="black", position="upper_left")
    plotter.add_mesh(
        q_plane,
        scalars="Q",
        cmap=_yi_q_cmap(),
        clim=(0.0, float(vmax)),
        opacity=float(TD_VIZ.proj_q_plane_opacity),
        nan_color=str(TD_VIZ.qmap_nan_color),
        show_scalar_bar=True,
    )
    colors = {
        "R1": TD_VIZ.qmap_box_r1,
        "R2": TD_VIZ.qmap_box_r2,
        "R3": TD_VIZ.qmap_box_r3,
    }
    if boxes:
        for name, (xmin, xmax, zmin, zmax) in boxes.items():
            corners = np.asarray(
                [
                    [xmin, 0.0, zmin],
                    [xmax, 0.0, zmin],
                    [xmax, 0.0, zmax],
                    [xmin, 0.0, zmax],
                    [xmin, 0.0, zmin],
                ],
                dtype=np.float64,
            )
            plotter.add_mesh(
                pv.lines_from_points(corners, close=False),
                color=colors.get(name, "black"),
                line_width=3,
                name=f"box_{name}",
            )
    # Face the y=0 Q plane.
    plotter.camera_position = [
        (0.0, max(80.0, float(TD_VIZ.qmap_x_max)), 0.5 * float(TD_VIZ.qmap_z_max)),
        (0.0, 0.0, 0.5 * float(TD_VIZ.qmap_z_max)),
        (0.0, 0.0, 1.0),
    ]
    try:
        plotter.render()
        _export_plotter_html(plotter, html_path)
    finally:
        plotter.close()
        try:
            pv.close_all()
        except Exception:
            pass


def _render_qmap_panel(
    q: np.ndarray,
    geometry: dict[str, Any],
    *,
    title: str,
    high_logq: float,
    output_path: Path,
    boxes: dict[str, tuple[float, float, float, float]] | None = None,
    html_path: Path | None = None,
    font_scale: float = 1.0,
) -> None:
    """Fig.4 (a1): Q map fills the panel; vertical colourbar is overlaid on-map."""
    from matplotlib import pyplot as _plt

    window_size = _unified_panel_window_size()
    vmax = _q_display_vmax(q, high_logq=high_logq)
    q_plot = _prepare_q_for_plot(q, vmax=vmax)
    x0 = float(TD_VIZ.qmap_x_min)
    x1 = float(TD_VIZ.qmap_x_max)
    z0 = float(TD_VIZ.qmap_z_min)
    z1 = float(TD_VIZ.qmap_z_max)
    extent = (
        geometry["x_bounds_mm"][0],
        geometry["x_bounds_mm"][1],
        geometry["z_bounds_mm"][0],
        geometry["z_bounds_mm"][1],
    )

    target_w, target_h = window_size
    dpi = 160.0
    bg = TD_VIZ.pv_background
    fig = _plt.figure(figsize=(target_w / dpi, target_h / dpi), dpi=dpi, facecolor=bg)
    # Full-bleed map: no outer grey gutter horizontally.
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(bg)
    q_cmap = _yi_q_cmap()
    nan_color = str(TD_VIZ.qmap_nan_color)
    q_cmap.set_bad(nan_color)
    image = ax.imshow(
        np.ma.masked_invalid(q_plot),
        origin="lower",
        aspect="auto",
        cmap=q_cmap,
        vmin=0.0,
        vmax=vmax,
        extent=extent,
    )
    if boxes:
        colors = {
            "R1": TD_VIZ.qmap_box_r1,
            "R2": TD_VIZ.qmap_box_r2,
            "R3": TD_VIZ.qmap_box_r3,
        }
        label_fs = float(TD_VIZ.qmap_region_label_fontsize)
        label_dx = float(TD_VIZ.qmap_region_label_offset_mm)
        for name, (xmin, xmax, zmin, zmax) in boxes.items():
            color = colors.get(name, "black")
            ax.add_patch(
                _plt.Rectangle(
                    (xmin, zmin),
                    xmax - xmin,
                    zmax - zmin,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.8,
                )
            )
            ax.text(
                xmin - label_dx,
                0.5 * (zmin + zmax),
                name,
                color=color,
                fontsize=label_fs,
                fontweight="bold",
                ha="right",
                va="center",
                clip_on=True,
            )
    # Symmetric x window so the right horn is not clipped harder than the left.
    x_half = max(abs(x0), abs(x1))
    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(z0, z1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colourbar fully inside the panel; ticks/label on the LEFT so the right
    # edge is not clipped by the figure boundary.
    cax = ax.inset_axes([0.88, 0.14, 0.045, 0.72])
    cbar = fig.colorbar(image, cax=cax, orientation="vertical")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label(r"$Q$", fontsize=11 * float(font_scale), labelpad=2)
    cbar.set_ticks([0.0, 0.5 * float(vmax), float(vmax)])
    cbar.ax.tick_params(labelsize=9 * float(font_scale), colors="black", pad=1)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=bg, bbox_inches=None, pad_inches=0.0)
    _plt.close(fig)
    _enforce_panel_window_size(output_path, window_size)
    _overlay_panel_subtitle(output_path, title)
    if html_path is not None:
        _export_qmap_interactive_html(
            q,
            geometry,
            high_logq=high_logq,
            html_path=html_path,
            boxes=boxes,
            title=title,
        )


def _bz_footpoint_levels(bz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive / negative |Bz| iso-levels for photosphere footpoint contours."""
    abs_bz = np.abs(np.asarray(bz, dtype=np.float64))
    finite = abs_bz[np.isfinite(abs_bz)]
    if finite.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    peak = float(np.nanpercentile(finite, 98.0)) or 1.0
    n_lev = max(1, int(TD_VIZ.footpoint_bz_n_levels))
    f0 = float(TD_VIZ.footpoint_bz_level_frac_min)
    f1 = float(TD_VIZ.footpoint_bz_level_frac_max)
    fracs = np.linspace(f0, f1, n_lev, dtype=np.float64)
    pos = peak * fracs
    neg = -peak * fracs[::-1]  # increasing: most negative → least negative
    return pos, neg


def _render_qmap_footpoint_panel(
    q: np.ndarray,
    bz: np.ndarray,
    case: dict[str, Any],
    *,
    title: str,
    output_path: Path,
    panel_label: str | None = None,
) -> None:
    """Photosphere log-Q grayscale map with dashed Bz / PIL footpoint contours.

    Matches the classic TD footpoint-Q style: black→white ``LogNorm`` Q,
    red/blue dashed polarity contours, white dashed PIL (``Bz=0``).
    """
    from matplotlib import colors as mcolors
    from matplotlib import pyplot as _plt
    from matplotlib.ticker import LogLocator, LogFormatterMathtext

    q_arr = np.asarray(q, dtype=np.float64)
    bz_arr = np.asarray(bz, dtype=np.float64)
    if q_arr.shape != bz_arr.shape:
        raise ValueError(
            f"Q and Bz shapes must match for footpoint map; "
            f"got Q{q_arr.shape} vs Bz{bz_arr.shape}"
        )

    q_plot = _interpolate_nan_q(q_arr)
    vmin = max(float(TD_VIZ.footpoint_q_vmin), 1.0e-6)
    vmax = max(float(TD_VIZ.footpoint_q_vmax), vmin * 10.0)
    valid = np.isfinite(q_plot) & (q_plot > 0.0)
    q_disp = np.full(q_plot.shape, np.nan, dtype=np.float64)
    q_disp[valid] = np.clip(q_plot[valid], vmin, vmax)

    # Prefer the metric evaluation ROI when available; else viz top-focus crop.
    metric_xy = _metric_xy_bounds(case)
    if metric_xy is not None:
        x0, x1, y0, y1 = metric_xy
    else:
        x0, x1, y0, y1, _z0, _z1 = _top_focus_bounds(case)
    gx0 = float(case["x_bounds"][0])
    gx1 = float(case["x_bounds"][1])
    gy0 = float(case["y_bounds"][0])
    gy1 = float(case["y_bounds"][1])
    extent = (gx0, gx1, gy0, gy1)

    window_size = _unified_panel_window_size()
    target_w, target_h = window_size
    dpi = 160.0
    bg = "#1a1a1a"
    fig = _plt.figure(figsize=(target_w / dpi, target_h / dpi), dpi=dpi, facecolor=bg)
    ax = fig.add_axes([0.12, 0.10, 0.72, 0.82])
    ax.set_facecolor("black")
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    image = ax.imshow(
        np.ma.masked_invalid(q_disp),
        origin="lower",
        aspect="auto",
        cmap="gray",
        norm=norm,
        extent=extent,
        interpolation="nearest",
    )

    xs = np.linspace(gx0, gx1, bz_arr.shape[1])
    ys = np.linspace(gy0, gy1, bz_arr.shape[0])
    pos_levels, neg_levels = _bz_footpoint_levels(bz_arr)
    lw = float(TD_VIZ.footpoint_contour_linewidth)
    if pos_levels.size:
        ax.contour(
            xs,
            ys,
            bz_arr,
            levels=pos_levels,
            colors=str(TD_VIZ.footpoint_pos_color),
            linewidths=lw,
            linestyles="--",
        )
    if neg_levels.size:
        ax.contour(
            xs,
            ys,
            bz_arr,
            levels=neg_levels,
            colors=str(TD_VIZ.footpoint_neg_color),
            linewidths=lw,
            linestyles="--",
        )
    # Polarity inversion line (PIL).
    ax.contour(
        xs,
        ys,
        bz_arr,
        levels=[0.0],
        colors=str(TD_VIZ.footpoint_pil_color),
        linewidths=float(TD_VIZ.footpoint_pil_linewidth),
        linestyles="--",
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=9,
        colors="white",
        length=4,
    )
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)
    if panel_label:
        ax.text(
            0.03,
            0.04,
            panel_label,
            transform=ax.transAxes,
            color="white",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    cax = fig.add_axes([0.86, 0.18, 0.035, 0.66])
    cbar = fig.colorbar(image, cax=cax, orientation="vertical")
    cbar.set_label(r"$Q$", fontsize=12, color="white", labelpad=4)
    cbar.ax.yaxis.set_major_locator(LogLocator(base=10.0))
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cbar.ax.tick_params(labelsize=9, colors="white")
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.8)
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_color("white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=bg, bbox_inches=None, pad_inches=0.0)
    _plt.close(fig)
    _enforce_panel_window_size(output_path, window_size)
    _overlay_panel_subtitle(output_path, title)


def compute_td3_footpoint_q_bundle(
    case: dict[str, Any],
    solution_fields: dict[str, np.ndarray],
    *,
    args: Any,
    out_dir: Path,
    sol_cache_name: str = "td3_nfgs_fastqsl_bottom.npz",
) -> dict[str, Any]:
    """Load or compute TD3 photosphere (footpoint) FastQSL Q for ref and NFGS."""
    import gc

    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    ref = case["reference"]
    ref_path = out_dir / "td3_analytic_fastqsl_bottom.npz"
    sol_path = out_dir / str(sol_cache_name)
    smooth_sigma = float(
        getattr(
            args,
            "qmap_b_smooth_sigma_cells",
            DEFAULT_QMAP_B_SMOOTH_SIGMA_CELLS,
        )
    )
    chunk = int(getattr(args, "fastqsl_trace_chunk_seeds", 8192) or 0)

    def _load_meta(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        try:
            text = str(raw.item() if hasattr(raw, "item") else raw)
            return json.loads(text)
        except Exception:
            return {}

    reuse = bool(getattr(args, "reuse_field_cache", True))
    q_ref: np.ndarray | None = None
    ref_meta: dict[str, Any] = {}
    if reuse and ref_path.is_file():
        with np.load(ref_path) as data:
            q_ref = np.asarray(data["q"])
            ref_meta = _load_meta(data["metadata"] if "metadata" in data.files else None)
        print("Loading cached TD3 analytic photosphere FastQSL Q", flush=True)

    if reuse and sol_path.is_file() and q_ref is not None:
        with np.load(sol_path) as data:
            q_sol = np.asarray(data["q"])
            sol_meta = _load_meta(data["metadata"] if "metadata" in data.files else None)
        cached_sigma = float(sol_meta.get("qmap_b_smooth_sigma_cells", -1.0))
        if abs(cached_sigma - smooth_sigma) < 1.0e-12:
            print("Loading cached TD3 NFGS photosphere FastQSL Q", flush=True)
            return {
                "q_ref": q_ref,
                "q_sol": q_sol,
                "bz_ref": np.asarray(ref["Bz"][0], dtype=np.float64),
                "bz_sol": np.asarray(solution_fields["Bz"][0], dtype=np.float64),
                "reference_metadata": ref_meta,
                "solution_metadata": sol_meta,
            }
        print(
            "Ignoring stale TD3 photosphere NFGS Q cache "
            f"(need qmap_b_smooth_sigma_cells={smooth_sigma:g}); recomputing",
            flush=True,
        )

    if q_ref is None:
        print("FastQSL footpoint Q: TD3 analytic reference", flush=True)
        reference_q = compute_bottom_qmap(
            ref["Bx"],
            ref["By"],
            ref["Bz"],
            fastqsl_root=args.fastqsl_root,
            step_len=args.fastqsl_step_len,
            tol=args.fastqsl_tol,
            block_x=args.fastqsl_block_x,
            trace_chunk_seeds=chunk,
            b_smooth_sigma_cells=0.0,
        )
        q_ref = reference_q["q"]
        ref_meta = reference_q["metadata"]
        np.savez_compressed(
            ref_path,
            q=q_ref,
            flag_start=reference_q["flag_start"],
            flag_end=reference_q["flag_end"],
            line_len=reference_q["line_len"],
            metadata=np.asarray(json.dumps(ref_meta, indent=2)),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    print(
        f"FastQSL footpoint Q: TD3 solution "
        f"(qmap_b_smooth_sigma_cells={smooth_sigma:g})",
        flush=True,
    )
    solution_q = compute_bottom_qmap(
        solution_fields["Bx"],
        solution_fields["By"],
        solution_fields["Bz"],
        fastqsl_root=args.fastqsl_root,
        step_len=args.fastqsl_step_len,
        tol=args.fastqsl_tol,
        block_x=args.fastqsl_block_x,
        trace_chunk_seeds=chunk,
        b_smooth_sigma_cells=smooth_sigma,
    )
    q_sol = solution_q["q"]
    sol_meta = {
        **solution_q["metadata"],
        "qmap_b_smooth_sigma_cells": smooth_sigma,
    }
    np.savez_compressed(
        sol_path,
        q=q_sol,
        flag_start=solution_q["flag_start"],
        flag_end=solution_q["flag_end"],
        line_len=solution_q["line_len"],
        metadata=np.asarray(json.dumps(sol_meta, indent=2)),
    )
    return {
        "q_ref": q_ref,
        "q_sol": q_sol,
        "bz_ref": np.asarray(ref["Bz"][0], dtype=np.float64),
        "bz_sol": np.asarray(solution_fields["Bz"][0], dtype=np.float64),
        "reference_metadata": ref_meta,
        "solution_metadata": sol_meta,
    }


def plot_td3_footpoint_qmaps(
    *,
    case: dict[str, Any],
    footpoint_bundle: dict[str, Any],
    output_path: Path,
    sol_panel_title: str = "TD3 NFGS",
) -> None:
    """Write a two-panel TD3 photosphere Q + Bz footpoint figure."""
    with tempfile.TemporaryDirectory(prefix="nfgs_td_fp_") as tmp:
        tmp_dir = Path(tmp)
        ref_path = tmp_dir / "td3_ref_footpoint.png"
        sol_path = tmp_dir / "td3_sol_footpoint.png"
        print("  panel TD3 ref. footpoint Q ...", flush=True)
        _render_qmap_footpoint_panel(
            footpoint_bundle["q_ref"],
            footpoint_bundle["bz_ref"],
            case,
            title="",
            output_path=ref_path,
            panel_label="(a)",
        )
        print(f"  panel {sol_panel_title} footpoint Q ...", flush=True)
        _render_qmap_footpoint_panel(
            footpoint_bundle["q_sol"],
            footpoint_bundle["bz_sol"],
            case,
            title="",
            output_path=sol_path,
            panel_label="(b)",
        )
        fig = plt.figure(figsize=(10.5, 5.2), dpi=180)
        grid = GridSpec(1, 2, figure=fig, wspace=0.08, left=0.02, right=0.98, top=0.92, bottom=0.04)
        for col, path, label in (
            (0, ref_path, "TD3 analytic"),
            (1, sol_path, str(sol_panel_title)),
        ):
            axis = fig.add_subplot(grid[0, col])
            axis.imshow(mpimg.imread(path), aspect="auto")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(label, fontsize=12, pad=4)
            for spine in axis.spines.values():
                spine.set_visible(False)
        fig.suptitle(
            "Photosphere Q-map with magnetic footpoint contours",
            fontsize=13,
            y=0.99,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


def _assemble_proj_plotter(
    *,
    case: dict[str, Any],
    fields: dict[str, np.ndarray],
    q: np.ndarray,
    geometry: dict[str, Any],
    high_logq: float,
    groups: list[dict[str, Any]] | None = None,
    boxes: dict[str, tuple[float, float, float, float]] | None = None,
    z_hft: float | None = None,
    off_screen: bool = True,
    window_size: tuple[int, int] | None = None,
    title: str | None = None,
    crop_q_to_qmap_window: bool = False,
):
    """Build a Fig.4-style projected PyVista plotter (bottom + Q plane + tubes).

    When ``groups`` is None, seeds/boxes are rebuilt via ``_td3_fig4_seed_groups``.
    """
    import math

    import pyvista as pv

    if groups is None or boxes is None or z_hft is None:
        groups, boxes, z_hft = _td3_fig4_seed_groups(
            case, q, geometry, fields=fields
        )
    assert groups is not None and boxes is not None and z_hft is not None

    _nx, _ny, nz, dx, dy, dz = case["dims"]
    if TD_VIZ.streamline_use_full_z:
        max_z_planes = int(nz)
    else:
        max_z_planes = max(2, int(math.ceil(nz / 2.0)))
    vector_fields = {
        "B": np.stack([fields["Bx"], fields["By"], fields["Bz"]], axis=0)
    }
    grid = _vector_grid(pv, vector_fields, case, max_z_planes)
    bottom = _bottom_plane(pv, np.asarray(case["boundary"]["Bz"], dtype=np.float32), case)
    vmax = _q_display_vmax(q, high_logq=high_logq)
    if crop_q_to_qmap_window:
        x_min, x_max, z_min, z_max = _qmap_display_window_mm()
        q_plane = _q_plane_mesh_roi(
            pv,
            q,
            geometry,
            vmax=vmax,
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
        )
    else:
        q_plane = _q_plane_mesh(pv, q, geometry, vmax=vmax)
    bz_abs = float(np.nanpercentile(np.abs(case["boundary"]["Bz"]), 99.0)) or 1.0

    size = list(window_size if window_size is not None else _unified_panel_window_size())
    plotter = pv.Plotter(window_size=size, border=False, off_screen=bool(off_screen))
    plotter.set_background(TD_VIZ.pv_background)
    if title:
        plotter.add_text(str(title), font_size=12, color="black", position="upper_left")
    plotter.add_mesh(
        bottom,
        scalars="Bz",
        cmap=_sunpy_cmap(),
        clim=(-bz_abs, bz_abs),
        opacity=float(TD_VIZ.proj_bottom_opacity),
        show_scalar_bar=False,
    )
    plotter.add_mesh(
        q_plane,
        scalars="Q",
        cmap=_yi_q_cmap(),
        clim=(0.0, float(vmax)),
        opacity=float(TD_VIZ.proj_q_plane_opacity),
        nan_color=str(TD_VIZ.qmap_nan_color),
        show_scalar_bar=False,
    )
    line_alpha = float(TD_VIZ.proj_fieldline_opacity)
    for group in groups:
        if len(group["seeds"]) == 0:
            continue
        lines = _closed_polylines_polydata(pv, fields, group["seeds"], case)
        if int(getattr(lines, "n_points", 0)) == 0:
            lines = _trace_lines(pv, grid, "B", group["seeds"])
            lines = _drop_escaped_streamlines(
                pv,
                lines,
                case,
                z_vis_top=float(TD_VIZ.side_z_max_td3) + 2.0 * float(dz),
            )
        if lines.n_points:
            plotter.add_mesh(
                lines.tube(radius=float(group["radius"]), n_sides=8),
                color=group["color"],
                smooth_shading=True,
                opacity=line_alpha,
            )
    cam = TD_VIZ.proj_camera_position
    foc = TD_VIZ.proj_camera_focal
    up = TD_VIZ.proj_camera_up
    focal_z = float(foc[2])
    if TD_VIZ.proj_camera_raise_focal_to_hft:
        focal_z = max(focal_z, float(z_hft))
    plotter.camera_position = [
        (float(cam[0]), float(cam[1]), float(cam[2])),
        (float(foc[0]), float(foc[1]), focal_z),
        (float(up[0]), float(up[1]), float(up[2])),
    ]
    plotter.camera.zoom(float(TD_VIZ.proj_camera_zoom))
    return plotter, boxes, float(z_hft)


def show_interactive_proj_preview(
    *,
    case: dict[str, Any],
    fields: dict[str, np.ndarray],
    q: np.ndarray,
    geometry: dict[str, Any],
    high_logq: float,
    groups: list[dict[str, Any]],
    boxes: dict[str, tuple[float, float, float, float]],
    z_hft: float,
    title: str,
    previous_plotter: Any | None = None,
    window_size: tuple[int, int] = (1100, 780),
):
    """Open / refresh an on-screen proj-view preview for the interactive picker.

    Returns the live ``pv.Plotter`` (caller should close it when done).
    """
    import os

    import pyvista as pv

    # Force an interactive (on-screen) backend for this preview window.
    os.environ["PYVISTA_OFF_SCREEN"] = "false"
    pv.OFF_SCREEN = False

    if previous_plotter is not None:
        try:
            previous_plotter.close()
        except Exception:
            pass
        # Do not pv.close_all() — a sibling TD2 preview window may be open.

    plotter, _boxes, _z = _assemble_proj_plotter(
        case=case,
        fields=fields,
        q=q,
        geometry=geometry,
        high_logq=high_logq,
        groups=groups,
        boxes=boxes,
        z_hft=z_hft,
        off_screen=False,
        window_size=window_size,
        title=title,
    )
    # Non-blocking so the matplotlib Q-map picker keeps its event loop.
    plotter.show(
        title=title,
        interactive_update=True,
        auto_close=False,
    )
    return plotter


def show_interactive_field_preview(
    *,
    case: dict[str, Any],
    fields: dict[str, np.ndarray],
    seed_groups: list[dict[str, Any]],
    title: str,
    view: str = "side",
    previous_plotter: Any | None = None,
    window_size: tuple[int, int] = (900, 700),
):
    """Open / refresh an on-screen top/side field-line preview (e.g. TD2).

    Traces ``seed_groups`` on ``fields`` with the same camera crop as the
    unified figure panels.  Returns the live ``pv.Plotter``.
    """
    import os

    import pyvista as pv

    os.environ["PYVISTA_OFF_SCREEN"] = "false"
    pv.OFF_SCREEN = False
    pv.global_theme.background = TD_VIZ.pv_background
    pv.global_theme.font.color = "black"

    if previous_plotter is not None:
        try:
            previous_plotter.close()
        except Exception:
            pass

    _B_STACK_CACHE.clear()
    _nx, _ny, nz, dx, dy, dz = case["dims"]
    focus = _visual_focus_bounds(case, view=view)
    side_focus = _side_focus_bounds(case)
    if TD_VIZ.streamline_use_full_z:
        max_z_planes = int(nz)
    else:
        z0 = float(case["z_bounds"][0])
        z_need = max(float(focus[5]), float(side_focus[5]))
        max_z_planes = min(
            int(nz),
            max(2, int(math.ceil((z_need - z0) / float(dz))) + 2),
        )

    vector_fields = {
        "B": np.stack(
            [fields["Bx"], fields["By"], fields["Bz"]],
            axis=0,
        )
    }
    grid = _vector_grid(pv, vector_fields, case, max_z_planes)
    bottom = _bottom_plane(
        pv, np.asarray(case["boundary"]["Bz"], dtype=np.float32), case
    )
    fx0, fx1, fy0, fy1, fz0, _fz1 = (float(v) for v in focus)
    if view == "top":
        bottom = bottom.clip_box(
            [fx0, fx1, fy0, fy1, fz0 - float(dz), fz0 + float(dz)],
            invert=False,
        )

    nonempty = [g for g in seed_groups if len(g["seeds"]) > 0]
    if not nonempty:
        raise RuntimeError(f"No field-line seeds for {case.get('benchmark')} preview")

    bz_abs = float(np.nanpercentile(np.abs(case["boundary"]["Bz"]), 99.0)) or 1.0
    plotter = pv.Plotter(
        window_size=list(window_size), border=False, off_screen=False
    )
    plotter.set_background(TD_VIZ.pv_background)
    if title:
        plotter.add_text(
            str(title), font_size=12, color="black", position="upper_left"
        )
    plotter.add_mesh(
        bottom,
        scalars="Bz",
        cmap=_sunpy_cmap(),
        clim=(-bz_abs, bz_abs),
        opacity=(
            TD_VIZ.bottom_opacity_side
            if view == "side"
            else TD_VIZ.bottom_opacity_top
        ),
        show_scalar_bar=False,
    )
    for group in nonempty:
        lines = _closed_polylines_polydata(pv, fields, group["seeds"], case)
        if int(getattr(lines, "n_points", 0)) == 0:
            lines = _trace_lines(pv, grid, "B", group["seeds"])
        if lines.n_points:
            plotter.add_mesh(
                lines.tube(radius=float(group["radius"]), n_sides=12),
                color=group["color"],
                smooth_shading=True,
            )
    if view == "top":
        _set_top_view_camera(plotter, focus, window_size=window_size)
    else:
        _set_side_view_camera(
            plotter, focus, window_size=window_size, elevate=True
        )
    plotter.show(
        title=title,
        interactive_update=True,
        auto_close=False,
    )
    return plotter


def _render_proj_with_qmap(
    *,
    case: dict[str, Any],
    fields: dict[str, np.ndarray],
    q: np.ndarray,
    geometry: dict[str, Any],
    title: str,
    high_logq: float,
    output_path: Path,
    html_path: Path | None = None,
    crop_q_to_qmap_window: bool = False,
) -> dict[str, tuple[float, float, float, float]]:
    import pyvista as pv

    pv.OFF_SCREEN = True
    try:
        pv.close_all()
    except Exception:
        pass
    _B_STACK_CACHE.clear()
    window_size = _unified_panel_window_size()
    plotter, boxes, _z_hft = _assemble_proj_plotter(
        case=case,
        fields=fields,
        q=q,
        geometry=geometry,
        high_logq=high_logq,
        off_screen=True,
        window_size=window_size,
        crop_q_to_qmap_window=crop_q_to_qmap_window,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotter.render()
        plotter.screenshot(str(output_path), window_size=list(window_size))
        _export_plotter_html(plotter, html_path)
    finally:
        plotter.close()
        try:
            pv.close_all()
        except Exception:
            pass
    _trim_panel_whitespace(
        output_path,
        trim_top=False,
        trim_bottom=True,
        margin_px=TD_VIZ.trim_margin_px,
        background_rgb=_color_to_rgb01(TD_VIZ.pv_background),
        tolerance=TD_VIZ.bg_trim_tolerance,
    )
    _enforce_panel_window_size(output_path, window_size)
    _overlay_panel_subtitle(output_path, title)
    return boxes


def compute_td3_q_bundle(
    case: dict[str, Any],
    solution_fields: dict[str, np.ndarray],
    *,
    args: Any,
    out_dir: Path,
) -> dict[str, Any]:
    import gc

    import torch

    geometry = case["geometry"]
    ref = case["reference"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "td3_analytic_fastqsl_y0.npz"
    sol_path = out_dir / "td3_nfgs_fastqsl_y0.npz"
    smooth_sigma = float(
        getattr(
            args,
            "qmap_b_smooth_sigma_cells",
            DEFAULT_QMAP_B_SMOOTH_SIGMA_CELLS,
        )
    )

    if args.reuse_field_cache and ref_path.is_file() and sol_path.is_file():
        with np.load(sol_path) as data:
            sol_meta_raw = data["metadata"] if "metadata" in data.files else None
            q_sol_cached = np.asarray(data["q"])
        sol_meta: dict[str, Any] = {}
        if sol_meta_raw is not None:
            try:
                sol_meta = json.loads(
                    str(
                        sol_meta_raw.item()
                        if hasattr(sol_meta_raw, "item")
                        else sol_meta_raw
                    )
                )
            except Exception:
                sol_meta = {}
        with np.load(ref_path) as data:
            q_ref_cached = np.asarray(data["q"])
            ref_meta_raw = data["metadata"] if "metadata" in data.files else None
        ref_meta: dict[str, Any] = {}
        if ref_meta_raw is not None:
            try:
                ref_meta = json.loads(
                    str(
                        ref_meta_raw.item()
                        if hasattr(ref_meta_raw, "item")
                        else ref_meta_raw
                    )
                )
            except Exception:
                ref_meta = {}
        if (
            y0_qmap_cache_matches(
                ref_meta,
                ny=int(np.asarray(ref["Bx"]).shape[1]),
                b_smooth_sigma_cells=0.0,
            )
            and y0_qmap_cache_matches(
                sol_meta,
                ny=int(np.asarray(solution_fields["Bx"]).shape[1]),
                b_smooth_sigma_cells=smooth_sigma,
            )
        ):
            print("Loading cached TD3 FastQSL Q maps", flush=True)
            q_ref = q_ref_cached
            q_sol = q_sol_cached
            q_metrics = analytic.q_topology_metrics(
                q_ref, q_sol, high_logq=args.td_topology_high_logq
            )
            return {
                "q_ref": q_ref,
                "q_sol": q_sol,
                "geometry": geometry,
                "q_metrics": q_metrics,
                "reference_metadata": {},
                "solution_metadata": sol_meta,
            }
        print(
            "Ignoring stale TD3 NFGS Q cache "
            f"(need {Y0_PLANE_COORDINATE_CONTRACT} and "
            f"qmap_b_smooth_sigma_cells={smooth_sigma:g}); recomputing",
            flush=True,
        )

    print("FastQSL topology: TD3 analytic reference", flush=True)
    # Analytic B is left raw (no Q-only smooth).
    reference_q = compute_y0_qmap(
        ref["Bx"],
        ref["By"],
        ref["Bz"],
        fastqsl_root=args.fastqsl_root,
        step_len=args.fastqsl_step_len,
        tol=args.fastqsl_tol,
        block_x=args.fastqsl_block_x,
        b_smooth_sigma_cells=0.0,
    )
    # Free GPU pressure before the second large TD3 trace.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    # Q-only FFT Gaussian on NFGS B; raw solution fields / metrics stay unchanged.
    print(
        f"FastQSL topology: TD3 NFGS solution "
        f"(qmap_b_smooth_sigma_cells={smooth_sigma:g})",
        flush=True,
    )
    solution_q = compute_y0_qmap(
        solution_fields["Bx"],
        solution_fields["By"],
        solution_fields["Bz"],
        fastqsl_root=args.fastqsl_root,
        step_len=args.fastqsl_step_len,
        tol=args.fastqsl_tol,
        block_x=args.fastqsl_block_x,
        b_smooth_sigma_cells=smooth_sigma,
    )
    q_ref = reference_q["q"]
    q_sol = solution_q["q"]
    q_metrics = analytic.q_topology_metrics(
        q_ref, q_sol, high_logq=args.td_topology_high_logq
    )
    np.savez_compressed(
        ref_path,
        q=q_ref,
        flag_start=reference_q["flag_start"],
        flag_end=reference_q["flag_end"],
        line_len=reference_q["line_len"],
        metadata=np.asarray(
            json.dumps(
                analytic.json_safe({**geometry, **reference_q["metadata"]}), indent=2
            )
        ),
    )
    np.savez_compressed(
        sol_path,
        q=q_sol,
        flag_start=solution_q["flag_start"],
        flag_end=solution_q["flag_end"],
        line_len=solution_q["line_len"],
        metadata=np.asarray(
            json.dumps(
                analytic.json_safe({**geometry, **solution_q["metadata"], **q_metrics}),
                indent=2,
            )
        ),
    )
    return {
        "q_ref": q_ref,
        "q_sol": q_sol,
        "geometry": geometry,
        "q_metrics": q_metrics,
        "reference_metadata": reference_q["metadata"],
        "solution_metadata": solution_q["metadata"],
    }


def _save_viz_seed_groups(path: Path, groups: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for idx, group in enumerate(groups):
        meta.append(
            {
                "label": str(group["label"]),
                "color": str(group["color"]),
                "radius": float(group["radius"]),
            }
        )
        arrays[f"seeds_{idx}"] = np.asarray(group["seeds"], dtype=np.float32)
    arrays["meta_json"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def _load_viz_seed_groups(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        groups: list[dict[str, Any]] = []
        for idx, item in enumerate(meta):
            groups.append(
                {
                    "label": item["label"],
                    "color": item["color"],
                    "radius": float(item["radius"]),
                    "seeds": np.asarray(data[f"seeds_{idx}"], dtype=np.float32),
                }
            )
    return groups


def _get_or_build_seed_groups(
    *,
    tag: str,
    builder,
    viz_seed_dir: Path | None,
    reuse_viz_seeds: bool,
) -> list[dict[str, Any]]:
    path = None if viz_seed_dir is None else viz_seed_dir / f"{tag}.npz"
    if reuse_viz_seeds and path is not None:
        cached = _load_viz_seed_groups(path)
        if cached is not None:
            print(f"  reused viz seeds {path.name}", flush=True)
            return cached
    groups = builder()
    if path is not None:
        _save_viz_seed_groups(path, groups)
        print(f"  saved viz seeds {path.name}", flush=True)
    return groups


def _get_or_build_td3_seed_bundle(
    *,
    tag: str,
    case: dict[str, Any],
    q: np.ndarray,
    geometry: dict[str, Any],
    fields: dict[str, np.ndarray],
    viz_seed_dir: Path | None,
    reuse_viz_seeds: bool,
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float, float, float]], float]:
    """Load or rebuild TD3 R1/R2/R3 groups together with the matching y=0 boxes."""
    path = None if viz_seed_dir is None else viz_seed_dir / f"{tag}.npz"
    if reuse_viz_seeds and path is not None:
        cached = _load_viz_seed_groups(path)
        if cached is not None:
            print(f"  reused viz seeds {path.name}", flush=True)
            _x_hft, z_hft = _locate_hft(q, geometry)
            # Boxes follow the *current* TD_VIZ (defaults for ref, overrides for NFGS).
            boxes = _hft_box_bounds(_x_hft, z_hft, q=q, geometry=geometry)
            return cached, boxes, float(z_hft)

    groups, boxes, z_hft = _td3_fig4_seed_groups(case, q, geometry, fields=fields)
    if path is not None:
        _save_viz_seed_groups(path, groups)
        print(f"  saved viz seeds {path.name}", flush=True)
    return groups, boxes, float(z_hft)


def plot_unified_td_figure(
    *,
    cases: dict[str, dict[str, Any]],
    solutions: dict[str, dict[str, np.ndarray]],
    q_bundle: dict[str, Any] | None,
    high_logq: float,
    output_path: Path,
    viz_seed_dir: Path | None = None,
    reuse_viz_seeds: bool = False,
    viz_overrides: dict[str, Any] | None = None,
    html_dir: Path | None = None,
) -> None:
    """Compose the 4x4 NFPT-style comparison figure for TD1--TD3.

    ``viz_overrides`` apply to **TD1/TD2 axis** picks and **TD3 NFGS** R1/R2/R3
    seeds/boxes (from ``configs/td_topology_nfgs_r123.yaml`` / interactive wizard).
    Analytic TD3 reference panels keep the default ``TdVizConfig`` R123 settings.

    When ``html_dir`` is set, each distinct 3D scene is exported once as
    interactive HTML (no separate top/side files — the viewer can rotate).
    """
    from nfgs._backend.util.td_viz_config import apply_td_viz_overrides

    needed = ("td1", "td2", "td3")
    missing = [name for name in needed if name not in cases or name not in solutions]
    if missing:
        raise ValueError(
            "Unified TD figure requires td1, td2, and td3 solutions; "
            f"missing={missing}"
        )
    if q_bundle is None:
        raise ValueError("Unified TD figure requires TD3 FastQSL Q maps.")

    overrides = dict(viz_overrides or {})
    html_root = Path(html_dir) if html_dir is not None else None
    if html_root is not None:
        html_root.mkdir(parents=True, exist_ok=True)
        print(f"  interactive HTML dir -> {html_root}", flush=True)

    def _html(name: str) -> Path | None:
        if html_root is None:
            return None
        return html_root / f"{name}.html"

    with tempfile.TemporaryDirectory(prefix="nfgs_td_fig_") as tmp:
        tmp_dir = Path(tmp)
        panel_paths: dict[tuple[int, int], Path] = {}

        for row, name in enumerate(("td1", "td2")):
            case = cases[name]
            ref_vec = np.stack(
                [
                    case["reference"]["Bx"],
                    case["reference"]["By"],
                    case["reference"]["Bz"],
                ],
                axis=0,
            )
            sol_vec = np.stack(
                [
                    solutions[name]["Bx"],
                    solutions[name]["By"],
                    solutions[name]["Bz"],
                ],
                axis=0,
            )
            vector_fields = {"B_ref": ref_vec, "B_nfgs": sol_vec}
            # Analytic footpoints once per benchmark (shared by ref / NFGS panels).
            # TD1/TD2 may use a per-figure manual axis |y| from the interactive picker.
            seed_ctx = (
                apply_td_viz_overrides(overrides)
                if name in ("td1", "td2")
                else contextlib.nullcontext()
            )
            with seed_ctx:
                seed_groups = _get_or_build_seed_groups(
                    tag=f"{name}_ref",
                    builder=lambda c=case: _field_line_seed_groups(
                        c, fields=c["reference"]
                    ),
                    viz_seed_dir=viz_seed_dir,
                    reuse_viz_seeds=reuse_viz_seeds,
                )
            specs = (
                (0, "B_ref", "top", f"{name.upper()} ref. top"),
                (1, "B_ref", "side", f"{name.upper()} ref. side"),
                (2, "B_nfgs", "top", f"{name.upper()} NFGS top"),
                (3, "B_nfgs", "side", f"{name.upper()} NFGS side"),
            )
            for col, vector_name, view, title in specs:
                path = tmp_dir / f"{name}_{vector_name}_{view}.png"
                print(f"  panel {name} {vector_name} {view} ...", flush=True)
                # One interactive HTML per field (not per top/side); export on top.
                html_path = (
                    _html(f"{name}_{vector_name}") if view == "top" else None
                )
                _render_field_view(
                    case=case,
                    vector_name=vector_name,
                    vector_fields=vector_fields,
                    view=view,
                    title=title,
                    output_path=path,
                    seed_groups=seed_groups,
                    html_path=html_path,
                )
                panel_paths[(row, col)] = path

        case3 = cases["td3"]
        geometry = q_bundle["geometry"]

        # Reference TD3: default TdVizConfig R1/R2/R3 (ignore NFGS YAML overrides).
        print("  TD3 reference R123: default config", flush=True)
        td3_ref_groups, boxes_ref, z_hft_ref = _get_or_build_td3_seed_bundle(
            tag="td3_ref",
            case=case3,
            q=q_bundle["q_ref"],
            geometry=geometry,
            fields=case3["reference"],
            viz_seed_dir=viz_seed_dir,
            reuse_viz_seeds=reuse_viz_seeds,
        )

        # NFGS TD3: per-figure interactive / YAML R123 overrides only.
        print(
            f"  TD3 NFGS R123 overrides={sorted(overrides.keys())}",
            flush=True,
        )
        with apply_td_viz_overrides(overrides):
            td3_sol_groups, boxes_sol, z_hft_sol = _get_or_build_td3_seed_bundle(
                tag="td3_nfgs",
                case=case3,
                q=q_bundle["q_sol"],
                geometry=geometry,
                fields=solutions["td3"],
                viz_seed_dir=viz_seed_dir,
                reuse_viz_seeds=reuse_viz_seeds,
            )

        # Populate in-memory TD3 cache used by proj rendering.
        _TD3_SEED_CACHE[
            (
                id(case3),
                id(case3["reference"]),
                id(q_bundle["q_ref"]),
                float(np.nanmax(q_bundle["q_ref"])),
            )
        ] = (td3_ref_groups, boxes_ref, float(z_hft_ref))
        _TD3_SEED_CACHE[
            (
                id(case3),
                id(solutions["td3"]),
                id(q_bundle["q_sol"]),
                float(np.nanmax(q_bundle["q_sol"])),
            )
        ] = (td3_sol_groups, boxes_sol, float(z_hft_sol))

        _render_proj_with_qmap(
            case=case3,
            fields=case3["reference"],
            q=q_bundle["q_ref"],
            geometry=geometry,
            title="TD3 ref. proj. view with Q",
            high_logq=high_logq,
            output_path=tmp_dir / "td3_ref_proj.png",
            html_path=_html("td3_ref_proj"),
        )
        _render_proj_with_qmap(
            case=case3,
            fields=solutions["td3"],
            q=q_bundle["q_sol"],
            geometry=geometry,
            title="TD3 NFGS proj. view with Q",
            high_logq=high_logq,
            output_path=tmp_dir / "td3_nfgs_proj.png",
            html_path=_html("td3_nfgs_proj"),
        )
        _render_qmap_panel(
            q_bundle["q_ref"],
            geometry,
            title="TD3 ref. Q map (y=0)",
            high_logq=high_logq,
            output_path=tmp_dir / "td3_ref_q.png",
            boxes=boxes_ref,
            html_path=_html("td3_ref_q"),
        )
        _render_qmap_panel(
            q_bundle["q_sol"],
            geometry,
            title="TD3 NFGS Q map (y=0)",
            high_logq=high_logq,
            output_path=tmp_dir / "td3_nfgs_q.png",
            boxes=boxes_sol,
            html_path=_html("td3_nfgs_q"),
        )
        panel_paths[(2, 0)] = tmp_dir / "td3_ref_q.png"
        panel_paths[(2, 1)] = tmp_dir / "td3_ref_proj.png"
        panel_paths[(3, 0)] = tmp_dir / "td3_nfgs_q.png"
        panel_paths[(3, 1)] = tmp_dir / "td3_nfgs_proj.png"

        ref_vec3 = np.stack(
            [
                case3["reference"]["Bx"],
                case3["reference"]["By"],
                case3["reference"]["Bz"],
            ],
            axis=0,
        )
        sol_vec3 = np.stack(
            [
                solutions["td3"]["Bx"],
                solutions["td3"]["By"],
                solutions["td3"]["Bz"],
            ],
            axis=0,
        )
        vector_fields3 = {"B_ref": ref_vec3, "B_nfgs": sol_vec3}
        td3_seed_cache = {"B_ref": td3_ref_groups, "B_nfgs": td3_sol_groups}
        # Render both top views before any side view so a stale side→top
        # offscreen buffer cannot copy TD3 ref top into TD3 NFGS top.
        for col, vector_name, view, title, row, q_key in (
            (2, "B_ref", "top", "TD3 ref. top", 2, "q_ref"),
            (2, "B_nfgs", "top", "TD3 NFGS top", 3, "q_sol"),
            (3, "B_ref", "side", "TD3 ref. side", 2, "q_ref"),
            (3, "B_nfgs", "side", "TD3 NFGS side", 3, "q_sol"),
        ):
            path = tmp_dir / f"td3_{vector_name}_{view}_r{row}.png"
            print(f"  panel td3 {vector_name} {view} ...", flush=True)
            html_path = (
                _html(f"td3_{vector_name}") if view == "top" else None
            )
            _render_field_view(
                case=case3,
                vector_name=vector_name,
                vector_fields=vector_fields3,
                view=view,
                title=title,
                output_path=path,
                q=q_bundle[q_key],
                seed_groups=td3_seed_cache[vector_name],
                html_path=html_path,
            )
            panel_paths[(row, col)] = path

        row_labels = ("TD1", "TD2", "TD3 ref", "TD3 NFGS")
        # Equal-width tiles, zero gaps: every field panel shares one aspect ratio.
        fig = plt.figure(figsize=(16.0, 14.0), dpi=180)
        grid = GridSpec(
            4,
            5,
            figure=fig,
            width_ratios=[0.07, 1.0, 1.0, 1.0, 1.0],
            wspace=float(TD_VIZ.figure_wspace),
            hspace=float(TD_VIZ.figure_hspace),
            left=0.035,
            right=0.995,
            top=0.965,
            bottom=0.01,
        )
        for row in range(4):
            label_ax = fig.add_subplot(grid[row, 0])
            label_ax.axis("off")
            label_ax.text(
                0.5,
                0.5,
                row_labels[row],
                rotation=90,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
            )
            for col in range(4):
                axis = fig.add_subplot(grid[row, col + 1])
                # Fill the cell: all tiles already share one baked-in panel AR.
                axis.imshow(mpimg.imread(panel_paths[(row, col)]), aspect="auto")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_anchor("C")
                for spine in axis.spines.values():
                    spine.set_visible(False)
        fig.suptitle(
            "NFGS versus analytic Titov--Demoulin references (TD1--TD3)",
            fontsize=14,
            y=0.995,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


def _vertical_content_span(rgb: np.ndarray, margin: int = 6) -> tuple[int, int]:
    """Row range that contains field lines or the magnetogram, not empty sky."""
    from PIL import Image, ImageFilter

    src = np.asarray(rgb[..., :3], dtype=np.float32)
    if float(src.max()) > 1.5:
        src = src / 255.0
    height = int(src.shape[0])
    image = Image.fromarray((np.clip(src, 0.0, 1.0) * 255.0).astype(np.uint8))
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=16))).astype(np.float32)
    blurred = blurred / 255.0
    row = np.abs(src - blurred).max(axis=2).mean(axis=1)
    if height > 8:
        row[:4] = 0.0
        row[-4:] = 0.0
    idx = np.flatnonzero(row > 0.02)
    if idx.size == 0:
        return 0, height - 1
    y0 = max(0, int(idx[0]) - int(margin))
    y1 = min(height - 1, int(idx[-1]) + int(margin))
    return y0, y1


def _crop_row_pairs_to_content(
    panel_paths: dict[tuple[int, int], Path],
    *,
    n_rows: int = 6,
    n_cols: int = 4,
) -> tuple[dict[tuple[int, int], np.ndarray], list[int]]:
    """Crop each view row to the shared vertical content span of its columns."""
    crops: dict[tuple[int, int], np.ndarray] = {}
    heights: list[int] = []
    for row in range(n_rows):
        images: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        for col in range(n_cols):
            rgb = np.asarray(mpimg.imread(panel_paths[(row, col)]))
            images.append(rgb)
            spans.append(_vertical_content_span(rgb))
        y0 = min(span[0] for span in spans)
        y1 = max(span[1] for span in spans)
        for col, rgb in enumerate(images):
            crops[(row, col)] = rgb[y0 : y1 + 1]
        heights.append(int(y1 - y0 + 1))
    return crops, heights


def plot_td_four_method_topology_figure(
    *,
    cases: dict[str, dict[str, Any]],
    columns: list[dict[str, Any]],
    high_logq: float,
    output_path: Path,
    viz_seed_dir: Path | None = None,
    reuse_viz_seeds: bool = False,
    status_lines: list[str] | None = None,
    html_dir: Path | None = None,
    shared_seeds: bool = False,
    font_scale: float = 1.0,
) -> None:
    """Compose a 6x4 method-comparison topology figure.

    Layout (rows x columns)::

                      ref | NFGS ρ=1 | NFGS ρ=0.5 | NFGS ρ=0
        TD1 top view
        TD1 side view
        TD2 top view
        TD2 side view
        TD3 top view
        TD3 side view

    Each entry in ``columns`` must provide::

        label: str
        fields: {td1, td2, td3} field dicts
        q_td3: y=0 Q map for TD3
        viz_overrides: optional dict applied to that column's seeds

    With ``shared_seeds=True``, all columns trace the exact physical seed
    arrays constructed once from the reference field/Q map.  No per-method
    closed-line filtering or seed rebuilding is then applied.
    """
    from nfgs._backend.util.td_viz_config import apply_td_viz_overrides

    needed = ("td1", "td2", "td3")
    n_cols = len(columns)
    if n_cols < 1:
        raise ValueError("Expected at least one method column")
    for col in columns:
        missing = [n for n in needed if n not in col.get("fields", {})]
        if missing:
            raise ValueError(f"Column {col.get('label')!r} missing fields: {missing}")
        if col.get("q_td3") is None:
            raise ValueError(f"Column {col.get('label')!r} missing q_td3")

    html_root = Path(html_dir) if html_dir is not None else None
    if html_root is not None:
        html_root.mkdir(parents=True, exist_ok=True)

    def _html(name: str) -> Path | None:
        if html_root is None:
            return None
        return html_root / f"{name}.html"

    row_labels = (
        "TD1 top view",
        "TD1 side view",
        "TD2 top view",
        "TD2 side view",
        "TD3 top view",
        "TD3 side view",
    )

    with tempfile.TemporaryDirectory(prefix="nfgs_td4_") as tmp:
        tmp_dir = Path(tmp)
        panel_paths: dict[tuple[int, int], Path] = {}

        case3 = cases["td3"]
        geometry = case3.get("geometry") or {}
        if not geometry:
            # Fall back to any column geometry metadata if present.
            geometry = columns[0].get("geometry") or {}

        shared_axis_groups: dict[str, list[dict[str, Any]]] = {}
        shared_td3_bundle = None
        if shared_seeds:
            reference_fields = columns[0]["fields"]
            for name in ("td1", "td2"):
                shared_axis_groups[name] = _get_or_build_seed_groups(
                    tag=f"{name}_shared",
                    builder=lambda n=name: _field_line_seed_groups(
                        cases[n], fields=cases[n]["reference"]
                    ),
                    viz_seed_dir=viz_seed_dir,
                    reuse_viz_seeds=reuse_viz_seeds,
                )
            shared_td3_bundle = _get_or_build_td3_seed_bundle(
                tag="td3_shared",
                case=case3,
                q=np.asarray(columns[0]["q_td3"], dtype=np.float64),
                geometry=geometry,
                fields=reference_fields["td3"],
                viz_seed_dir=viz_seed_dir,
                reuse_viz_seeds=reuse_viz_seeds,
            )

        for col_i, col in enumerate(columns):
            label = str(col["label"])
            overrides = dict(col.get("viz_overrides") or {})
            fields_td = col["fields"]
            q_td3 = np.asarray(col["q_td3"], dtype=np.float64)
            print(f"  column[{col_i}] {label}", flush=True)

            # TD1 / TD2 top & side — per-column axis picks from YAML / wizard.
            for name, row_top, row_side in (
                ("td1", 0, 1),
                ("td2", 2, 3),
            ):
                case = cases[name]
                vec = np.stack(
                    [
                        fields_td[name]["Bx"],
                        fields_td[name]["By"],
                        fields_td[name]["Bz"],
                    ],
                    axis=0,
                )
                vector_fields = {"B": vec}
                if shared_seeds:
                    seed_groups = shared_axis_groups[name]
                else:
                    seed_ctx = (
                        apply_td_viz_overrides(overrides)
                        if overrides
                        else contextlib.nullcontext()
                    )
                    with seed_ctx:
                        seed_groups = _get_or_build_seed_groups(
                            tag=f"{name}_c{col_i}",
                            builder=lambda c=case: _field_line_seed_groups(
                                c, fields=c["reference"]
                            ),
                            viz_seed_dir=viz_seed_dir,
                            reuse_viz_seeds=reuse_viz_seeds,
                        )
                for view, row in (("top", row_top), ("side", row_side)):
                    path = tmp_dir / f"c{col_i}_{name}_{view}.png"
                    print(f"    panel {name} {view} ...", flush=True)
                    _render_field_view(
                        case=case,
                        vector_name="B",
                        vector_fields=vector_fields,
                        view=view,
                        title="",
                        output_path=path,
                        seed_groups=seed_groups,
                        html_path=_html(f"c{col_i}_{name}") if view == "top" else None,
                    )
                    panel_paths[(row, col_i)] = path

            # TD3 seeds / Q / proj for this method column.
            if shared_seeds:
                if shared_td3_bundle is None:
                    raise RuntimeError("Shared TD3 seed bundle was not initialized")
                td3_groups, boxes, z_hft = shared_td3_bundle
            else:
                seed_ctx = (
                    apply_td_viz_overrides(overrides)
                    if overrides
                    else contextlib.nullcontext()
                )
                with seed_ctx:
                    td3_groups, boxes, z_hft = _get_or_build_td3_seed_bundle(
                        tag=f"td3_c{col_i}",
                        case=case3,
                        q=q_td3,
                        geometry=geometry,
                        fields=fields_td["td3"],
                        viz_seed_dir=viz_seed_dir,
                        reuse_viz_seeds=reuse_viz_seeds,
                    )
            _TD3_SEED_CACHE[
                (
                    id(case3),
                    id(fields_td["td3"]),
                    id(q_td3),
                    float(np.nanmax(q_td3)),
                )
            ] = (td3_groups, boxes, float(z_hft))

            vec3 = np.stack(
                [
                    fields_td["td3"]["Bx"],
                    fields_td["td3"]["By"],
                    fields_td["td3"]["Bz"],
                ],
                axis=0,
            )
            for view, row in (("top", 4), ("side", 5)):
                path = tmp_dir / f"c{col_i}_td3_{view}.png"
                print(f"    panel td3 {view} ...", flush=True)
                _render_field_view(
                    case=case3,
                    vector_name="B",
                    vector_fields={"B": vec3},
                    view=view,
                    title="",
                    output_path=path,
                    seed_groups=td3_groups,
                    html_path=_html(f"c{col_i}_td3") if view == "top" else None,
                )
                panel_paths[(row, col_i)] = path

        from nfgs._backend.scripts.nfgs_figure_style import (
            FIG_ROW_LABEL_FONTSIZE,
            FIG_TITLE_FONTSIZE,
        )

        title_fontsize = FIG_TITLE_FONTSIZE * float(font_scale)
        row_fontsize = FIG_ROW_LABEL_FONTSIZE * float(font_scale)
        # Keep each camera frame intact. Flush top/side pairs remove only the
        # gutter between those two rows; the white gap stays between cases.
        panels: dict[tuple[int, int], np.ndarray] = {}
        row_heights: list[int] = []
        for row in range(6):
            for col_i in range(n_cols):
                panels[(row, col_i)] = np.asarray(
                    mpimg.imread(panel_paths[(row, col_i)])
                )
            row_heights.append(int(panels[(row, 0)].shape[0]))
        pair_heights = [
            row_heights[0] + row_heights[1],
            row_heights[2] + row_heights[3],
            row_heights[4] + row_heights[5],
        ]
        # Keep the per-panel size of the original four-column canvas.
        fig_w = 18.0 * (0.12 + n_cols) / (0.12 + 4.0)
        font_scale = float(font_scale) * (fig_w / 18.0)
        title_fontsize = FIG_TITLE_FONTSIZE * float(font_scale)
        row_fontsize = FIG_ROW_LABEL_FONTSIZE * float(font_scale)
        left, right = 0.045, 0.995
        top = 0.975
        bottom = 0.04 if status_lines else 0.012
        outer_wspace, outer_hspace = 0.012, 0.055
        label_w = 0.12
        panel_span = float(n_cols)
        label_to_panels = label_w / (label_w + panel_span)
        width_frac = (right - left) / (1.0 + outer_wspace * 0.5)
        panel_w_in = fig_w * width_frac * (1.0 - label_to_panels) / n_cols
        panel_w_px = float(panels[(0, 0)].shape[1])
        block_in = [panel_w_in * (h / panel_w_px) for h in pair_heights]
        n_cases = 3
        axes_frac = (top - bottom) / (1.0 + outer_hspace * (n_cases - 1) / n_cases)
        fig_h = sum(block_in) / axes_frac
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
        outer = GridSpec(
            3,
            2,
            figure=fig,
            width_ratios=[label_w, panel_span],
            height_ratios=pair_heights,
            wspace=outer_wspace,
            hspace=outer_hspace,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        for case_i in range(3):
            label_ax = fig.add_subplot(outer[case_i, 0])
            label_ax.axis("off")
            label_ax.text(
                0.5,
                0.5,
                f"TD{case_i + 1}",
                rotation=90,
                ha="center",
                va="center",
                fontsize=row_fontsize,
                fontweight="bold",
            )
            panel_grid = outer[case_i, 1].subgridspec(
                2,
                n_cols,
                height_ratios=[
                    row_heights[case_i * 2],
                    row_heights[case_i * 2 + 1],
                ],
                hspace=0.0,
                wspace=float(TD_VIZ.figure_wspace),
            )
            for sub in range(2):
                row = case_i * 2 + sub
                for col_i in range(n_cols):
                    axis = fig.add_subplot(panel_grid[sub, col_i])
                    axis.imshow(panels[(row, col_i)], aspect="auto")
                    axis.set_xticks([])
                    axis.set_yticks([])
                    axis.set_anchor("C")
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                    if row == 0:
                        axis.set_title(
                            str(columns[col_i]["label"]),
                            fontsize=title_fontsize,
                            pad=8,
                        )

        if status_lines:
            fig.text(
                0.04,
                0.005,
                "\n".join(status_lines),
                fontsize=7.5,
                family="monospace",
                va="bottom",
                ha="left",
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
