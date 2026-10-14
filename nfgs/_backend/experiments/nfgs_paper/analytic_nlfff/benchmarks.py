"""Shared analytic benchmark construction for the NFGS experiment suite."""

from __future__ import annotations

from nfgs.paths import project_root

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from nfgs._backend.util.lnl_io import LOW_LOU_FAMILY, build_lnl_family_reference_and_obs
from nfgs._backend.util.td2_io import TD2Grid, build_td2_reference_and_obs


ROOT = project_root()
METRIC_KEYS = (
    "C_vec",
    "C_CS",
    "Eprime_n",
    "Eprime_m",
    "epsilon",
    "sigma_J",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return name


def metric_shape(slices: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    return tuple(int(item.stop) - int(item.start or 0) for item in slices)  # type: ignore[return-value]


def make_low_lou_case(
    calculation_domain_mode: str = "mudpack",
    case_name: str = "low_lou_c2",
) -> dict[str, Any]:
    """Build one frozen Low--Lou family state."""
    lnl = build_lnl_family_reference_and_obs(case_name)
    lnl_slices = lnl["metric_slices"]
    key = str(case_name).lower().replace("-", "_")
    display = {
        "low_lou_c2": "Low--Lou C2",
        "low_lou_ll1": "Low--Lou LL1",
        "low_lou_ll2": "Low--Lou LL2",
    }[key]
    return {
        "benchmark": key,
        "display": display,
        "dims": (80, 80, 72, lnl["dx"], lnl["dy"], lnl["dz"]),
        "boundary": {
            "Bx": lnl["Bx_obs"],
            "By": lnl["By_obs"],
            "Bz": lnl["Bz_obs"],
        },
        "reference": {
            "Bx": lnl["Bx_ref"],
            "By": lnl["By_ref"],
            "Bz": lnl["Bz_ref"],
        },
        "Jz_obs": lnl["Jz_obs"],
        "metric_slices": lnl_slices,
        "metric_shape": metric_shape(lnl_slices),
        "metric_domain": (
            f"{display}: central 64 x 64 horizontal footprint and bottom "
            "64 vertical planes"
        ),
        "solver_kwargs": {
            "lateral_wall": "legacy_potential_pin",
            "pin_potential_boundaries": True,
            "buffer_thickness": 0,
            "padded_buffer_zone": True,
            "pad_calculation_domain": True,
            "calculation_domain_mode": calculation_domain_mode,
        },
        "mu": 1.0,
        "benchmark_parameters": {
            "low_lou": lnl.get("jarolim_c2", lnl.get("benchmark_family")),
            "bounds_xyz": list(lnl["bounds"]),
            "spacing_mode": lnl["spacing_mode"],
        },
    }


def make_td_case(
    source_root: Path = ROOT,
    calculation_domain_mode: str = "mudpack",
) -> dict[str, Any]:
    """Build only the legacy TD artifact case."""
    grid = TD2Grid()
    td_path = source_root / "data" / "analytic" / "td2" / "bcor"
    td = build_td2_reference_and_obs(td_path, grid)
    td_slices = td["metric_slices"]
    return {
        "benchmark": "td",
        "display": "Titov--Demoulin",
        "dims": (grid.Nx, grid.Ny, grid.Nz, td["dx"], td["dy"], td["dz"]),
        "boundary": {
            "Bx": td["Bx_obs"],
            "By": td["By_obs"],
            "Bz": td["Bz_obs"],
        },
        "reference": {
            "Bx": td["Bx_ref"],
            "By": td["By_ref"],
            "Bz": td["Bz_ref"],
        },
        "Jz_obs": td["Jz_obs"],
        "metric_slices": td_slices,
        "metric_shape": metric_shape(td_slices),
        "metric_domain": (
            "flux-rope-centered 66 x 66 x 30 cube "
            "(Demcsak et al. grid-count protocol)"
        ),
        "solver_kwargs": {
            "lateral_wall": "paper_bc1",
            "pin_potential_boundaries": False,
            "buffer_thickness": 0,
            "padded_buffer_zone": True,
            "pad_calculation_domain": True,
            "calculation_domain_mode": calculation_domain_mode,
        },
        "mu": 1.0,
        "benchmark_parameters": {
            "bounds_xyz": {
                "x": list(td["x_bounds"]),
                "y": list(td["y_bounds"]),
                "z": list(td["z_bounds"]),
            },
            "flux_rope_center_indices_yx": [
                int(td["flux_rope_iy"]),
                int(td["flux_rope_ix"]),
            ],
            "evaluation_protocol": (
                "flux-rope-centered 66 x 66 x 30 grid-count protocol"
            ),
        },
        "x_bounds": td["x_bounds"],
        "y_bounds": td["y_bounds"],
        "z_bounds": td["z_bounds"],
        "flux_rope_iy": td["flux_rope_iy"],
        "flux_rope_ix": td["flux_rope_ix"],
    }


def make_cases(
    source_root: Path = ROOT,
    calculation_domain_mode: str = "mudpack",
    requested: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper that lazily constructs only requested cases."""
    names = ("low_lou_c2", "td") if requested is None else tuple(dict.fromkeys(requested))
    allowed = set(LOW_LOU_FAMILY) | {"td"}
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"unknown analytic benchmarks: {unknown}")
    cases: dict[str, dict[str, Any]] = {}
    for name in names:
        if name in LOW_LOU_FAMILY:
            cases[name] = make_low_lou_case(calculation_domain_mode, name)
    if "td" in names:
        cases["td"] = make_td_case(source_root, calculation_domain_mode)
    return cases


def save_field_cache(path: Path, fields: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{name: np.asarray(value) for name, value in fields.items()},
    )


def write_history_csv(path: Path, record: dict[str, Any]) -> None:
    history = record["history"]
    sigma = list(history.get("sigma", []))
    n_rows = len(sigma)
    iterations = list(history.get("iter", range(1, n_rows + 1)))
    preferred = ["sigma", "divB", "energy", "mu", "rollback"]
    dynamic = sorted(
        key
        for key, value in history.items()
        if key not in preferred
        and isinstance(value, (list, tuple))
        and len(value) == n_rows
    )
    columns = preferred + dynamic
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("iteration", *columns))
        for index, iteration in enumerate(iterations):
            writer.writerow(
                [iteration]
                + [
                    history.get(key, [float("nan")] * n_rows)[index]
                    for key in columns
                ]
            )
