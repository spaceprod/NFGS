"""TD1--TD3 figure visualisation knobs.

Edit ``TD_VIZ`` (or construct a custom ``TdVizConfig``) to tune seeds,
crops, colours, and cameras without changing experiment orchestration.

Per-figure NFGS / NFPT R1/R2/R3 overrides live in
``configs/td_topology_nfgs_r123.yaml`` (see ``load_nfgs_r123_config`` /
``nfgs_r123_overrides_for_figure``). Figure tags include ``robin_w0pX`` and
``nfpt_fortran_symmetric``. After changing TD3 seed parameters, clear
``viz_seeds/td3_*.npz`` (or omit ``--reuse-viz-seeds``) so seeds are rebuilt.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


# Knobs that control TD3 R1/R2/R3 ROI boxes and seed selection (Fig. 4).
TD3_R123_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "td3_nfgs_rebuild_seeds",
        "td3_r1_half_width",
        "td3_r1_z0_offset",
        "td3_r1_z1_offset",
        "td3_r2_half_width",
        "td3_r2_z0_offset",
        "td3_r2_z1_offset",
        "td3_r3_half_width",
        "td3_r3_z0_offset",
        "td3_r3_z1_offset",
        "td3_r1_seeds_per_polarity",
        "td1_rope_seeds_per_polarity",
        "td2_axis_ring_seeds",
        "td3_r1_radius_min_mm",
        "td3_r1_radius_max_mm",
        "td3_r1_radius_n_coarse",
        "td3_r1_ring_radius_mm",
        "td3_r1_axis_from_jz",
        "td3_r1_axis_y_mm",
        "td1_axis_y_mm",
        "td2_axis_y_mm",
        "td3_r1_bbox_pad_mm",
        "td3_r1_bbox_y_tol_mm",
        "td3_r1_box_abs",
        "td3_r2_box_abs",
        "td3_r3_box_abs",
        "td3_r2_n_seeds",
        "td3_r2_jz_percentile",
        "td3_r2_jz_max_distance_mm",
        "td3_r2_jz_require_both_feet",
        "td3_r3_nx",
        "td3_r3_nz",
        "td3_flux_level_frac_min",
        "td3_flux_level_frac_max",
        "td3_flux_level_n",
        "td3_flux_contour_margin_mm",
        "td3_flux_min_contour_points",
        "td3_flux_min_contour_area_mm2",
        "td3_flux_min_arc_points",
        "color_td3_r2",
        "color_td3_r3",
        "qmap_box_r1",
        "qmap_box_r2",
        "qmap_box_r3",
        "qmap_nan_color",
    }
)

# Optional absolute y=0 boxes stored as YAML lists [x0, x1, z0, z1].
_TD3_ABS_BOX_FIELDS: frozenset[str] = frozenset(
    {"td3_r1_box_abs", "td3_r2_box_abs", "td3_r3_box_abs"}
)


@dataclass
class TdVizConfig:
    """Central knobs for TD1--TD3 figure rendering (seeds, crop, colours, camera)."""

    # Panel background, window, and figure tiling
    pv_background: str = "#b0b0b0"
    # All field panels share one pixel window (unified aspect ratio).
    panel_height_px: int = 480
    panel_aspect: float = 1.15  # width / height for top AND side screenshots
    figure_wspace: float = 0.0
    figure_hspace: float = 0.0
    trim_margin_px: int = 2
    bg_trim_tolerance: float = 0.035
    # Side view: crop empty bg below the magnetogram so the floor sits on the frame edge.
    side_trim_bottom: bool = True
    side_trim_top: bool = False

    # Shared field-line colours (TD1 flux rope == TD3 R1 polarity split)
    color_axis_pos: str = "#2563eb"  # blue
    color_axis_neg: str = "#d946ef"  # magenta
    color_rope_pos: str = "#fb923c"  # orange
    color_rope_neg: str = "#38bdf8"  # skyblue
    color_td2_bald: str = "#ef4444"
    color_td3_r2: str = "palegreen"
    color_td3_r3: str = "#fb923c"

    # Q-map ROI box outline colours (Fig. 4 a1)
    qmap_box_r1: str = "black"
    qmap_box_r2: str = "#86efac"
    qmap_box_r3: str = "#fb923c"
    # FastQSL NaN / invalid Q cells (matplotlib set_bad + PyVista nan_color)
    qmap_nan_color: str = "white"

    # Tube radii as fractions of min(dx, dy, dz)
    tube_radius_thin: float = 0.48
    tube_radius_thick: float = 1.35
    tube_radius_td3_thin: float = 0.75

    # =========================================================================
    # Seed selection — TD1 (Fig. 1 outer flux rope)
    # =========================================================================
    # Seeds sit on two photospheric rings centred on the ± flux-rope footpoints.
    # Final kept count ≈ 2 * td1_rope_seeds_per_polarity (closed lines only).
    td1_rope_r0_mm: float = 5.0  # analytic rope minor-radius scale (Mm)
    td1_rope_outer_ratio: float = 5.0  # ring radius ≈ r0 * outer_ratio
    td1_rope_seeds_per_polarity: int = 10  # seeds per ± footpoint ring
    td1_rope_a_scale: float = 0.95  # shrink factor applied to ring radius
    # Manual photospheric axis |y| (Mm) on x=0.  When set (>0), centres are
    # (0, +y) and (0, -y), matching the interactive Bz-map pick for TD1.
    td1_axis_y_mm: float | None = None

    # =========================================================================
    # Seed selection — TD2 (Fig. 3 bald-patch / axis bundle)
    # =========================================================================
    # Small ring around the coronal axis seed; all ring seeds are traced.
    td2_axis_ring_radius_mm: float = 2.5
    td2_axis_ring_seeds: int = 10
    # Manual photospheric axis |y| (Mm) on x=0.  When set (>0), centres are
    # (0, +y) and (0, -y), matching the interactive Bz-map pick for TD2.
    td2_axis_y_mm: float | None = None

    # --- Explicit view crops (Mm); top and side are fully independent ---
    # Top view: centre-based half-extents  [cx ± top_x_pad, cy ± top_y_pad].
    # Default centre is the origin (0,0); set explicitly to override.
    top_center_x_td1: float | None = 0.0
    top_center_y_td1: float | None = 0.0
    top_y_pad_td1: float = 150.0
    top_x_pad_td1: float = top_y_pad_td1 / 1.15
    top_center_x_td2: float | None = 0.0
    top_center_y_td2: float | None = 0.0
    top_y_pad_td2: float = 115.0
    top_x_pad_td2: float = top_y_pad_td2 / 1.15
    top_center_x_td3: float | None = 0.0
    top_center_y_td3: float | None = 0.0
    top_y_pad_td3: float = 150.0
    top_x_pad_td3: float = top_y_pad_td3 / 1.15

    # Side view: independent centre + y half-extent; z from z_min to z_max.
    # Does NOT inherit top_y_pad / top centre.
    side_center_y_td1: float | None = None
    side_y_pad_td1: float = 150.0
    side_z_min_td1: float = 0.0
    side_z_max_td1: float = 150.0
    side_center_y_td2: float | None = None
    side_y_pad_td2: float = 150.0
    side_z_min_td2: float = 0.0
    side_z_max_td2: float = 150.0
    side_center_y_td3: float | None = None
    side_y_pad_td3: float = 150.0
    side_z_min_td3: float = 0.0
    side_z_max_td3: float = 150.0

    # Panel subtitle: drawn LAST at a fixed top-left pixel offset (all panels aligned).
    subtitle_pad_x_px: int = 12
    subtitle_pad_y_px: int = 10
    subtitle_fontsize_px: int = 24

    # TD3 Q-map panel (Fig.4 a1): magnified x–z crop + right vertical colourbar
    qmap_x_min: float = -55.0
    qmap_x_max: float = 55.0
    qmap_z_min: float = 0.0
    qmap_z_max: float = 95.0
    qmap_colorbar_vmax: float = 100.0
    qmap_region_label_fontsize: float = 16.0
    qmap_region_label_offset_mm: float = 12.0  # left of box edge (4x former 3 Mm)
    qmap_left_label_pad_mm: float = 48.0  # extra x room for R1/R2/R3 text

    # Photosphere footpoint Q map (log-Q grayscale + Bz polarity contours)
    footpoint_q_vmin: float = 1.0
    footpoint_q_vmax: float = 1.0e3
    footpoint_bz_n_levels: int = 5
    footpoint_bz_level_frac_min: float = 0.15
    footpoint_bz_level_frac_max: float = 0.85
    footpoint_contour_linewidth: float = 0.9
    footpoint_pos_color: str = "#dc2626"  # red dashed Bz > 0
    footpoint_neg_color: str = "#2563eb"  # blue dashed Bz < 0
    footpoint_pil_color: str = "#ffffff"  # white dashed Bz = 0
    footpoint_pil_linewidth: float = 1.2

    # TD3 projected view (Fig.4 a2): camera + opacities
    proj_camera_position: tuple[float, float, float] = (-250, +190.0, 60.0)
    proj_camera_focal: tuple[float, float, float] = (30.0, 0.0, 40.0)
    proj_camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    proj_camera_zoom: float = 0.8
    proj_camera_raise_focal_to_hft: bool = True
    proj_fieldline_opacity: float = 1.0
    proj_q_plane_opacity: float = 0.75
    proj_bottom_opacity: float = 1.0

    # Streamline / tube integration (full volume so arches reach conjugated feet)
    streamline_max_steps: int = 8000
    streamline_max_length_factor: float = 4.0
    streamline_use_full_z: bool = True

    # =========================================================================
    # Seed selection — TD3 (Fig. 4 R1 / R2 / R3)
    # =========================================================================
    # When True, NFGS panels rebuild R1/R2/R3 on the solution field + Q using
    # the knobs below (needed for per-figure YAML overrides).  When False,
    # NFGS reuses analytic seed positions and only re-checks closed lines.
    td3_nfgs_rebuild_seeds: bool = True

    # --- ROI prisms relative to the hyperbolic flux tube (HFT) [Mm] ---
    # Same rule for R1/R2/R3:
    #   x in [x_hft ± half_width],  z in [z_hft + z0_offset, z_hft + z1_offset]
    td3_r1_half_width: float = 21.0
    td3_r1_z0_offset: float = 32.0
    td3_r1_z1_offset: float = 55.0
    td3_r2_half_width: float = 18.0
    td3_r2_z0_offset: float = 5.0
    td3_r2_z1_offset: float = 30.0
    td3_r3_half_width: float = 16.0
    # Below QS / HFT (photosphere toward R2 bottom); tune per figure in YAML.
    td3_r3_z0_offset: float = -20.0
    td3_r3_z1_offset: float = 5.0

    # --- R1 seeds: footpoint rings ---
    td3_r1_seeds_per_polarity: int = 10
    td3_r1_radius_min_mm: float = 1.0
    td3_r1_radius_max_mm: float = 35.0
    td3_r1_radius_n_coarse: int = 18
    # When set (>0), use this fixed photospheric ring radius instead of the
    # largest-closed auto search.  Interactive picker writes this knob.
    td3_r1_ring_radius_mm: float | None = None
    # When True, centre rings on ± bottom-|Jz| blob axes (else symmetry search).
    td3_r1_axis_from_jz: bool = False
    # Manual photospheric axis |y| (Mm) on x=0.  When set (>0), centres are
    # (0, +y) and (0, -y) and override Jz / conjugate-search axis picking.
    # Interactive Bz-map clicks write this knob.
    td3_r1_axis_y_mm: float | None = None
    # Padding (Mm) when auto-fitting the R1 box to traced field-line extents.
    td3_r1_bbox_pad_mm: float = 1.0
    # Only |y| <= this (Mm) samples contribute to the auto R1 box (y=0 Q map).
    td3_r1_bbox_y_tol_mm: float = 5.0
    # Absolute y=0 boxes [x0, x1, z0, z1] (Mm).  When set, override HFT offsets.
    td3_r1_box_abs: tuple[float, float, float, float] | None = None
    td3_r2_box_abs: tuple[float, float, float, float] | None = None
    td3_r3_box_abs: tuple[float, float, float, float] | None = None

    # --- R2 seeds ---
    td3_r2_n_seeds: int = 50
    td3_r2_jz_percentile: float = 95.0
    td3_r2_jz_max_distance_mm: float = 20.0
    td3_r2_jz_require_both_feet: bool = True

    # --- R3 seeds ---
    td3_r3_nx: int = 50
    td3_r3_nz: int = 20

    # --- R2 contour picker ---
    td3_flux_level_frac_min: float = 0.35
    td3_flux_level_frac_max: float = 0.90
    td3_flux_level_n: int = 28
    td3_flux_contour_margin_mm: float = 3.0
    td3_flux_min_contour_points: int = 24
    td3_flux_min_contour_area_mm2: float = 80.0
    td3_flux_min_arc_points: int = 8

    # Side-view camera
    side_camera_elev_deg: float = 24.0
    side_camera_z_pad_fraction: float = 0.02
    side_floor_bottom_bias: float = 0.92

    # Bottom magnetogram opacity
    bottom_opacity_top: float = 1.0
    bottom_opacity_side: float = 1.0


# Edit this instance to tune the unified TD figure without hunting through helpers.
TD_VIZ = TdVizConfig()

DEFAULT_NFGS_R123_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "td_topology_nfgs_r123.yaml"
)


def _coerce_abs_box(value: Any) -> tuple[float, float, float, float] | None:
    """Parse ``[x0, x1, z0, z1]`` (or null) into a sorted absolute box."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        x0, x1, z0, z1 = (float(v) for v in value)
        return (min(x0, x1), max(x0, x1), min(z0, z1), max(z0, z1))
    raise ValueError(f"Absolute box must be null or [x0, x1, z0, z1]; got {value!r}")


def _coerce_override_value(name: str, value: Any) -> Any:
    """Cast YAML values to the types expected by ``TdVizConfig``."""
    if name in _TD3_ABS_BOX_FIELDS:
        return _coerce_abs_box(value)
    if name in {"td3_r1_ring_radius_mm", "td3_r1_axis_y_mm", "td1_axis_y_mm", "td2_axis_y_mm"}:
        if value is None:
            return None
        return float(value)
    field_types = {f.name: f.type for f in fields(TdVizConfig)}
    hint = field_types.get(name)
    if hint is bool or hint == "bool":
        return bool(value)
    if hint is int or hint == "int":
        return int(value)
    if hint is float or hint == "float":
        return float(value)
    return value


def load_nfgs_r123_config(path: Path | None = None) -> dict[str, Any]:
    """Load the per-figure NFGS R1/R2/R3 YAML config."""
    cfg_path = Path(path) if path is not None else DEFAULT_NFGS_R123_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"NFGS R1/R2/R3 config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {cfg_path}")
    return data


def nfgs_r123_overrides_for_figure(
    figure_tag: str,
    *,
    config: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Merge ``defaults`` with ``figures[figure_tag]`` for R1/R2/R3 knobs.

    ``figure_tag`` matches the topology figure stem suffix, e.g. ``robin_w0p5``.
    Only keys listed in ``TD3_R123_FIELD_NAMES`` are applied.
    """
    data = dict(config) if config is not None else load_nfgs_r123_config(config_path)
    defaults = data.get("defaults") or {}
    figures = data.get("figures") or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("'defaults' must be a mapping")
    if not isinstance(figures, Mapping):
        raise ValueError("'figures' must be a mapping")
    per = figures.get(figure_tag) or figures.get(str(figure_tag)) or {}
    if per is None:
        per = {}
    if not isinstance(per, Mapping):
        raise ValueError(f"figures[{figure_tag!r}] must be a mapping")

    merged: dict[str, Any] = {}
    for source in (defaults, per):
        for key, value in source.items():
            name = str(key)
            if name not in TD3_R123_FIELD_NAMES:
                continue
            merged[name] = _coerce_override_value(name, value)
    return merged


@contextmanager
def apply_td_viz_overrides(overrides: Mapping[str, Any]) -> Iterator[TdVizConfig]:
    """Temporarily set fields on the global ``TD_VIZ`` instance."""
    if not overrides:
        yield TD_VIZ
        return
    unknown = [k for k in overrides if not hasattr(TD_VIZ, k)]
    if unknown:
        raise ValueError(f"Unknown TdVizConfig fields: {unknown}")
    saved = {k: getattr(TD_VIZ, k) for k in overrides}
    try:
        for key, value in overrides.items():
            setattr(TD_VIZ, key, value)
        yield TD_VIZ
    finally:
        for key, value in saved.items():
            setattr(TD_VIZ, key, value)


def td_viz_with_overrides(overrides: Mapping[str, Any]) -> TdVizConfig:
    """Return a copy of ``TD_VIZ`` with ``overrides`` applied (does not mutate)."""
    clean = {
        k: _coerce_override_value(k, v)
        for k, v in overrides.items()
        if hasattr(TD_VIZ, k)
    }
    return replace(TD_VIZ, **clean)


def upsert_nfgs_r123_figure_overrides(
    figure_tag: str,
    overrides: Mapping[str, Any],
    *,
    config_path: Path | None = None,
) -> Path:
    """Merge ``overrides`` into ``figures[figure_tag]`` and rewrite the YAML."""
    cfg_path = Path(config_path) if config_path is not None else DEFAULT_NFGS_R123_CONFIG
    data = load_nfgs_r123_config(cfg_path) if cfg_path.is_file() else {"defaults": {}, "figures": {}}
    if "defaults" not in data or not isinstance(data.get("defaults"), dict):
        data["defaults"] = data.get("defaults") or {}
    figures = data.setdefault("figures", {})
    if not isinstance(figures, dict):
        raise ValueError("'figures' must be a mapping")
    per = dict(figures.get(figure_tag) or {})
    for key, value in overrides.items():
        name = str(key)
        if name not in TD3_R123_FIELD_NAMES:
            continue
        coerced = _coerce_override_value(name, value)
        if coerced is None:
            per[name] = None
        elif isinstance(coerced, tuple):
            per[name] = [float(v) for v in coerced]
        else:
            per[name] = coerced
    figures[figure_tag] = per
    data["figures"] = figures
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return cfg_path


def interactive_r123_overrides_dict(
    *,
    r1_radius_mm: float,
    r1_box: tuple[float, float, float, float] | None,
    r2_box: tuple[float, float, float, float] | None,
    r3_box: tuple[float, float, float, float] | None,
    r1_axis_y_mm: float | None = None,
    td1_axis_y_mm: float | None = None,
    td2_axis_y_mm: float | None = None,
    td1_rope_seeds_per_polarity: int = 10,
    td2_axis_ring_seeds: int = 10,
    td3_r1_seeds_per_polarity: int = 10,
    td3_r2_n_seeds: int = 50,
    td3_r3_nx: int = 50,
    td3_r3_nz: int = 20,
) -> dict[str, Any]:
    """Build the NFGS override dict written by the interactive seed wizard."""
    manual_axis = (
        float(r1_axis_y_mm)
        if r1_axis_y_mm is not None and float(r1_axis_y_mm) > 0.0
        else None
    )
    td1_axis = (
        float(td1_axis_y_mm)
        if td1_axis_y_mm is not None and float(td1_axis_y_mm) > 0.0
        else None
    )
    td2_axis = (
        float(td2_axis_y_mm)
        if td2_axis_y_mm is not None and float(td2_axis_y_mm) > 0.0
        else None
    )
    out: dict[str, Any] = {
        "td3_nfgs_rebuild_seeds": True,
        # Manual Bz pick wins; otherwise fall back to ± bottom-Jz blob axes.
        "td3_r1_axis_from_jz": manual_axis is None,
        "td3_r1_ring_radius_mm": float(r1_radius_mm),
        "td3_r1_axis_y_mm": manual_axis,
        "td1_axis_y_mm": td1_axis,
        "td2_axis_y_mm": td2_axis,
        "td1_rope_seeds_per_polarity": max(1, int(td1_rope_seeds_per_polarity)),
        "td2_axis_ring_seeds": max(1, int(td2_axis_ring_seeds)),
        "td3_r1_seeds_per_polarity": max(1, int(td3_r1_seeds_per_polarity)),
        "td3_r2_n_seeds": max(1, int(td3_r2_n_seeds)),
        "td3_r3_nx": max(1, int(td3_r3_nx)),
        "td3_r3_nz": max(1, int(td3_r3_nz)),
    }
    if r1_box is not None:
        out["td3_r1_box_abs"] = tuple(float(v) for v in r1_box)
    if r2_box is not None:
        out["td3_r2_box_abs"] = tuple(float(v) for v in r2_box)
    if r3_box is not None:
        out["td3_r3_box_abs"] = tuple(float(v) for v in r3_box)
    return out
