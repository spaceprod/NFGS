#!/usr/bin/env python3
"""Extended CPU manufactured-solution checks for NFGS scalar reconstruction.

The runner deliberately exercises the production modal routines rather than a
standalone reference matrix:

* ``_solve_modal_robin_from_hat`` for compatible and incompatible mixed rows;
* ``_solve_phi_c_robin`` and ``_solve_az_from_jz_poisson_source`` for an
  end-to-end poloidal/toroidal manufactured field; and
* ``_reconstruct_pt_state_from_scalars`` for all three components of B and J.

Horizontal manufactured modes use the solver's discrete endpoint-Neumann or
endpoint-Dirichlet eigenvalues.  In particular, every poloidal vertical decay
uses ``k_h=sqrt(-lambda_h)`` from the actual DCT-I operator.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs_solver import NFGSSolverGPU  # noqa: E402


MIXED_COMPONENTS = (
    (1, 1, 1.0, 0.10),
    (2, 1, -0.25, -0.04),
)
PHI_COMPONENTS = (
    (1, 1, 0.70, 0.08),
    (2, 1, -0.20, -0.035),
)
AZ_COMPONENTS = (
    (1, 1, 0.45, 0.35),
    (2, 1, -0.18, -0.20),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--grids", default="17,33,65")
    parser.add_argument("--mixed-rhos", default="0,0.4,1")
    parser.add_argument(
        "--mixed-gammas",
        default="1",
        help="gamma values for rho(h)=gamma*h/(1+gamma*h); empty disables",
    )
    parser.add_argument("--incompatible-amplitude", type=float, default=0.25)
    parser.add_argument("--complete-rho", type=float, default=0.4)
    parser.add_argument("--skip-mixed", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def _parse_ints(value: str, *, minimum: int) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] < minimum:
        raise ValueError(f"values must be integers >= {minimum}")
    return values


def _parse_floats(value: str, *, minimum: float | None = None) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if minimum is not None and values and values[0] < minimum:
        raise ValueError(f"values must be >= {minimum}")
    return values


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _resolve_dtypes(value: str) -> list[torch.dtype]:
    mapping = {"float32": torch.float32, "float64": torch.float64}
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(mapping))
    if unknown:
        raise ValueError(f"unsupported dtypes: {unknown}")
    if not names:
        raise ValueError("at least one dtype is required")
    return [mapping[name] for name in names]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value * value)).item())


def _relative(error: torch.Tensor, reference: torch.Tensor) -> float:
    return _rms(error) / max(_rms(reference), 1.0e-30)


def _component_relative(error: torch.Tensor, reference: torch.Tensor) -> list[float]:
    return [
        _rms(error[index]) / max(_rms(reference[index]), 1.0e-30)
        for index in range(error.shape[0])
    ]


def _make_solver(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    bottom_robin_weight: float = 0.0,
) -> NFGSSolverGPU:
    h = 1.0 / (n - 1)
    return NFGSSolverGPU(
        n,
        n,
        n,
        h,
        h,
        h,
        device=device,
        dtype=dtype,
        horizontal_zero_mode_policy="omit",
        source_zero_mode_policy="project",
        bottom_divergence_robin_weight=bottom_robin_weight,
        buffer_thickness=0,
        padded_buffer_zone=False,
        pad_calculation_domain=False,
    )


def _coordinates(
    solver: NFGSSolverGPU,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.arange(solver.Nx, device=solver.device, dtype=solver.dtype) * float(
        solver.dx
    )
    y = torch.arange(solver.Ny, device=solver.device, dtype=solver.dtype) * float(
        solver.dy
    )
    z = torch.arange(solver.Nz, device=solver.device, dtype=solver.dtype) * float(
        solver.dz
    )
    return x, y, z


def _cosine_mode(
    solver: NFGSSolverGPU, mx: int, my: int
) -> torch.Tensor:
    x, y, _ = _coordinates(solver)
    lx = float((solver.Nx - 1) * solver.dx)
    ly = float((solver.Ny - 1) * solver.dy)
    return torch.cos(math.pi * my * y / ly)[:, None] * torch.cos(
        math.pi * mx * x / lx
    )[None, :]


def _top_robin_residual(
    solver: NFGSSolverGPU, phi: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    phi_hat = solver._dct2(phi)
    derivative_hat = (
        3.0 * phi_hat[-1] - 4.0 * phi_hat[-2] + phi_hat[-3]
    ) / (2.0 * float(solver.dz))
    kh = torch.sqrt(torch.clamp(-solver._endpoint_neumann_lam_xy(), min=0.0))
    residual_hat = derivative_hat + kh * phi_hat[-1] - solver._dct2(target)
    residual_hat[0, 0] = 0.0
    return solver._idct2(residual_hat)


def _bottom_derivative(solver: NFGSSolverGPU, field: torch.Tensor) -> torch.Tensor:
    return (-3.0 * field[0] + 4.0 * field[1] - field[2]) / (
        2.0 * float(solver.dz)
    )


def _mixed_manufactured_fields(solver: NFGSSolverGPU) -> dict[str, object]:
    """Semidiscrete poloidal field used to probe the first mixed row."""
    shape = (solver.Nz, solver.Ny, solver.Nx)
    phi = torch.zeros(shape, device=solver.device, dtype=solver.dtype)
    source = torch.zeros_like(phi)
    bottom = torch.zeros((solver.Ny, solver.Nx), device=solver.device, dtype=solver.dtype)
    top = torch.zeros_like(bottom)
    _, _, z = _coordinates(solver)
    height = float((solver.Nz - 1) * solver.dz)
    lam_xy = solver._endpoint_neumann_lam_xy()
    components: list[dict[str, float]] = []

    for mx, my, amplitude, a_value in MIXED_COMPONENTS:
        lam = float(lam_xy[my, mx].item())
        kh = math.sqrt(max(-lam, 0.0))
        mode = _cosine_mode(solver, mx, my)
        f = torch.exp(-kh * z) + a_value * z * z
        src_z = 2.0 * a_value - kh * kh * a_value * z * z
        top_value = 2.0 * a_value * height + kh * a_value * height * height
        phi += amplitude * f[:, None, None] * mode[None]
        source += amplitude * src_z[:, None, None] * mode[None]
        bottom += amplitude * mode
        top += amplitude * top_value * mode
        components.append(
            {
                "mx": mx,
                "my": my,
                "amplitude": amplitude,
                "a": a_value,
                "lambda_h": lam,
                "k_h": kh,
            }
        )

    return {
        "phi": phi,
        "source": source,
        "bottom": bottom,
        "top": top,
        # This target is exactly compatible with the sampled reference under
        # the same one-sided second-order derivative used by the mixed row.
        "q_compatible": _bottom_derivative(solver, phi),
        "components": components,
    }


def run_mixed_case(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rho: float,
    weight_scheme: str,
    gamma: float | None,
    target_kind: str,
    incompatible_amplitude: float,
) -> tuple[dict[str, object], list[float]]:
    solver = _make_solver(n, device=device, dtype=dtype)
    fields = _mixed_manufactured_fields(solver)
    phi_exact = fields["phi"]
    source = fields["source"]
    bottom = fields["bottom"]
    top = fields["top"]
    q_compatible = fields["q_compatible"]
    assert isinstance(phi_exact, torch.Tensor)
    assert isinstance(source, torch.Tensor)
    assert isinstance(bottom, torch.Tensor)
    assert isinstance(top, torch.Tensor)
    assert isinstance(q_compatible, torch.Tensor)

    q_target = q_compatible.clone()
    perturbation = torch.zeros_like(q_target)
    if target_kind == "incompatible":
        perturbation = incompatible_amplitude * _cosine_mode(solver, 1, 1)
        q_target += perturbation
    elif target_kind != "compatible":
        raise ValueError(f"unknown target kind: {target_kind}")

    start = time.perf_counter()
    with torch.inference_mode():
        phi_num = solver._solve_modal_robin_from_hat(
            solver._dct2(source),
            solver._dct2(bottom),
            top_robin_hat=solver._dct2(top),
            bottom_derivative_hat=(solver._dct2(q_target) if rho > 0.0 else None),
            bottom_robin_weight=rho,
        )
    wall_time = time.perf_counter() - start

    h = float(solver.dz)
    q_num = _bottom_derivative(solver, phi_num)
    lap = solver._lap_xy(phi_num)
    vertical = (phi_num[2:] - 2.0 * phi_num[1:-1] + phi_num[:-2]) / (
        h * h
    )
    pde = vertical + lap[1:-1] - source[1:-1]
    mixed_algebraic = (1.0 - rho) * pde[0] - rho * (q_num - q_target) / h
    mixed_scale = (1.0 - rho) * (
        _rms(vertical[0]) + _rms(lap[1]) + _rms(source[1])
    ) + rho * (_rms(q_num) + _rms(q_target)) / h
    error = phi_num - phi_exact
    profile = torch.sqrt(torch.mean(error * error, dim=(-2, -1)))
    scale = max(_rms(phi_exact), 1.0e-30)
    row: dict[str, object] = {
        "experiment": "mixed_row",
        "dtype": _dtype_name(dtype),
        "device": str(device),
        "n": n,
        "h": h,
        "weight_scheme": weight_scheme,
        "weight_parameter": (
            f"gamma:{gamma:.12g}" if gamma is not None else f"rho:{rho:.12g}"
        ),
        "gamma": gamma,
        "rho": rho,
        "target_kind": target_kind,
        "incompatible_amplitude": incompatible_amplitude,
        "incompatible_perturbation_rms": _rms(perturbation),
        "E_phi": _rms(error),
        "E_phi_relative": _rms(error) / scale,
        "E_phi_first_unknown": _rms(error[1]),
        "E_phi_near_bottom": _rms(error[1:3]),
        "E_phi_bulk": _rms(error[2:]) if n > 3 else _rms(error[1:]),
        "R_bottom_dirichlet": _rms(phi_num[0] - bottom),
        "R_bottom_derivative_target": _rms(q_num - q_target),
        "R_bottom_derivative_compatible": _rms(q_num - q_compatible),
        "R_first_pde": _rms(pde[0]),
        "R_bulk_pde": _rms(pde[1:]) if pde.shape[0] > 1 else 0.0,
        "R_mixed_algebraic": _rms(mixed_algebraic),
        "R_mixed_algebraic_relative": _rms(mixed_algebraic)
        / max(mixed_scale, 1.0e-30),
        "R_top_robin": _rms(_top_robin_residual(solver, phi_num, top)),
        "max_profile_error_z": int(torch.argmax(profile).item()),
        "wall_time_s": wall_time,
        "order_phi": None,
        "order_phi_bulk": None,
        "components": fields["components"],
    }
    return row, [float(value.item()) for value in profile]


def _complete_manufactured_fields(solver: NFGSSolverGPU) -> dict[str, object]:
    """Build semidiscrete Phi/Az sources and analytic B/J references."""
    shape = (solver.Nz, solver.Ny, solver.Nx)
    phi = torch.zeros(shape, device=solver.device, dtype=solver.dtype)
    source_phi = torch.zeros_like(phi)
    source_bcz = torch.zeros_like(phi)
    az = torch.zeros_like(phi)
    rhs_az = torch.zeros_like(phi)
    source_jz = torch.zeros_like(phi)
    b_reference = torch.zeros((3, *shape), device=solver.device, dtype=solver.dtype)
    j_reference = torch.zeros_like(b_reference)
    x, y, z = _coordinates(solver)
    lx = float((solver.Nx - 1) * solver.dx)
    ly = float((solver.Ny - 1) * solver.dy)
    height = float((solver.Nz - 1) * solver.dz)
    lam_neumann = solver._endpoint_neumann_lam_xy()
    lam_dirichlet = solver._dirichlet_lam_xy_interior()
    phi_metadata: list[dict[str, float]] = []
    az_metadata: list[dict[str, float]] = []

    for mx, my, amplitude, a_value in PHI_COMPONENTS:
        kx = math.pi * mx / lx
        ky = math.pi * my / ly
        cos_x = torch.cos(kx * x)
        sin_x = torch.sin(kx * x)
        cos_y = torch.cos(ky * y)
        sin_y = torch.sin(ky * y)
        coscos = cos_y[:, None] * cos_x[None, :]
        sincos = cos_y[:, None] * sin_x[None, :]
        cossin = sin_y[:, None] * cos_x[None, :]
        lam = float(lam_neumann[my, mx].item())
        kh = math.sqrt(max(-lam, 0.0))
        distance = height - z
        f = torch.exp(-kh * z) + a_value * distance**4
        f_prime = -kh * torch.exp(-kh * z) - 4.0 * a_value * distance**3
        f_second = kh * kh * torch.exp(-kh * z) + 12.0 * a_value * distance**2
        scalar_source = f_second + lam * f

        phi += amplitude * f[:, None, None] * coscos[None]
        source_phi += amplitude * scalar_source[:, None, None] * coscos[None]
        source_bcz += amplitude * lam * scalar_source[:, None, None] * coscos[None]
        b_reference[0] += amplitude * kx * f_prime[:, None, None] * sincos[None]
        b_reference[1] += amplitude * ky * f_prime[:, None, None] * cossin[None]
        b_reference[2] += amplitude * lam * f[:, None, None] * coscos[None]
        j_reference[0] += -amplitude * ky * scalar_source[:, None, None] * cossin[None]
        j_reference[1] += amplitude * kx * scalar_source[:, None, None] * sincos[None]
        phi_metadata.append(
            {
                "mx": mx,
                "my": my,
                "amplitude": amplitude,
                "quartic_amplitude": a_value,
                "lambda_h": lam,
                "k_h": kh,
                "k_h_H": kh * height,
            }
        )

    scaled_z = z / height
    for mx, my, amplitude, c_value in AZ_COMPONENTS:
        kx = math.pi * mx / lx
        ky = math.pi * my / ly
        cos_x = torch.cos(kx * x)
        sin_x = torch.sin(kx * x)
        cos_y = torch.cos(ky * y)
        sin_y = torch.sin(ky * y)
        sinsin = sin_y[:, None] * sin_x[None, :]
        sincos = cos_y[:, None] * sin_x[None, :]
        cossin = sin_y[:, None] * cos_x[None, :]
        lam = float(lam_dirichlet[my - 1, mx - 1].item())
        exp_cz = torch.exp(c_value * scaled_z)
        g = (1.0 - scaled_z) * exp_cz
        g_prime = exp_cz * (c_value * (1.0 - scaled_z) - 1.0) / height
        g_second = (
            exp_cz
            * (c_value * c_value * (1.0 - scaled_z) - 2.0 * c_value)
            / (height * height)
        )
        scalar_rhs = g_second + lam * g

        az += amplitude * g[:, None, None] * sinsin[None]
        rhs_az += amplitude * scalar_rhs[:, None, None] * sinsin[None]
        source_jz += -amplitude * lam * scalar_rhs[:, None, None] * sinsin[None]
        b_reference[0] += amplitude * ky * g[:, None, None] * sincos[None]
        b_reference[1] += -amplitude * kx * g[:, None, None] * cossin[None]
        j_reference[0] += amplitude * kx * g_prime[:, None, None] * cossin[None]
        j_reference[1] += amplitude * ky * g_prime[:, None, None] * sincos[None]
        j_reference[2] += -amplitude * lam * g[:, None, None] * sinsin[None]
        az_metadata.append(
            {
                "mx": mx,
                "my": my,
                "amplitude": amplitude,
                "vertical_exponential_c": c_value,
                "lambda_h": lam,
                "k_h": math.sqrt(max(-lam, 0.0)),
            }
        )

    return {
        "phi": phi,
        "source_phi": source_phi,
        "source_bcz": source_bcz,
        "az": az,
        "rhs_az": rhs_az,
        "source_jz": source_jz,
        "b_reference": b_reference,
        "j_reference": j_reference,
        "phi_components": phi_metadata,
        "az_components": az_metadata,
    }


def _curl(solver: NFGSSolverGPU, field: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            solver._d_dy_centered(field[2]) - solver._d_dz_centered(field[1]),
            solver._d_dz_centered(field[0]) - solver._d_dx_centered(field[2]),
            solver._d_dx_centered(field[1]) - solver._d_dy_centered(field[0]),
        ]
    )


def _divergence(solver: NFGSSolverGPU, field: torch.Tensor) -> torch.Tensor:
    return (
        solver._d_dx_centered(field[0])
        + solver._d_dy_centered(field[1])
        + solver._d_dz_centered(field[2])
    )


def run_complete_case(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rho: float,
) -> dict[str, object]:
    solver = _make_solver(
        n, device=device, dtype=dtype, bottom_robin_weight=rho
    )
    fields = _complete_manufactured_fields(solver)
    phi_exact = fields["phi"]
    az_exact = fields["az"]
    source_phi = fields["source_phi"]
    source_bcz = fields["source_bcz"]
    rhs_az = fields["rhs_az"]
    source_jz = fields["source_jz"]
    b_reference = fields["b_reference"]
    j_reference = fields["j_reference"]
    for value in (
        phi_exact,
        az_exact,
        source_phi,
        source_bcz,
        rhs_az,
        source_jz,
        b_reference,
        j_reference,
    ):
        assert isinstance(value, torch.Tensor)

    # Exercise the actual production wrappers.  The manufactured bottom maps
    # are installed directly because running Gaussian separation would change
    # the prescribed scalar problem before this isolated operator test.
    solver._bottom_dphi_c_target = _bottom_derivative(solver, phi_exact)
    solver.Az_bottom = az_exact[0].clone()
    start = time.perf_counter()
    with torch.inference_mode():
        phi_num = solver._solve_phi_c_robin(source_bcz, phi_exact[0])
        az_num = solver._solve_az_from_jz_poisson_source(source_jz)
        bcz_num = solver._lap_xy(phi_num)
        solver.Phi_pot = torch.zeros_like(phi_num)
        solver.B_pot = torch.zeros(
            (3, solver.Nz, solver.Ny, solver.Nx),
            device=solver.device,
            dtype=solver.dtype,
        )
        solver.Bz_obs = bcz_num[0].clone()
        solver.Jz_obs = -solver._lap_xy(az_num)[0]
        b_num, j_num, _ = solver._reconstruct_pt_state_from_scalars(
            phi_num, az_num, bcz_num
        )
        bcz_exact = solver._lap_xy(phi_exact)
        solver.Bz_obs = bcz_exact[0].clone()
        solver.Jz_obs = -solver._lap_xy(az_exact)[0]
        b_discrete, j_discrete, _ = solver._reconstruct_pt_state_from_scalars(
            phi_exact, az_exact, bcz_exact
        )
    wall_time = time.perf_counter() - start

    h = float(solver.dz)
    phi_pde = (phi_num[2:] - 2.0 * phi_num[1:-1] + phi_num[:-2]) / (h * h)
    phi_pde += solver._lap_xy(phi_num)[1:-1] - source_phi[1:-1]
    az_pde = (az_num[2:] - 2.0 * az_num[1:-1] + az_num[:-2]) / (h * h)
    az_pde += solver._lap_xy(az_num)[1:-1] - rhs_az[1:-1]
    zero_top = torch.zeros(
        (solver.Ny, solver.Nx), device=solver.device, dtype=solver.dtype
    )

    core = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
    quarter = max(1, n // 4)
    central = (
        slice(None),
        slice(quarter, n - quarter),
        slice(quarter, n - quarter),
        slice(quarter, n - quarter),
    )
    b_error = b_num - b_reference
    j_error = j_num - j_reference
    b_core_error = b_error[core]
    j_core_error = j_error[core]
    b_core_reference = b_reference[core]
    j_core_reference = j_reference[core]
    b_central_error = b_error[central]
    j_central_error = j_error[central]
    b_central_reference = b_reference[central]
    j_central_reference = j_reference[central]
    div_b = _divergence(solver, b_num)
    curl_residual = _curl(solver, b_num) - j_num
    b_component = _component_relative(b_error, b_reference)
    j_component = _component_relative(j_error, j_reference)
    b_core_component = _component_relative(b_core_error, b_core_reference)
    j_core_component = _component_relative(j_core_error, j_core_reference)
    b_central_component = _component_relative(
        b_central_error, b_central_reference
    )
    j_central_component = _component_relative(
        j_central_error, j_central_reference
    )

    return {
        "experiment": "complete_phi_az_b_j",
        "dtype": _dtype_name(dtype),
        "device": str(device),
        "n": n,
        "h": h,
        "rho": rho,
        "E_phi": _rms(phi_num - phi_exact),
        "E_phi_relative": _relative(phi_num - phi_exact, phi_exact),
        "E_Az": _rms(az_num - az_exact),
        "E_Az_relative": _relative(az_num - az_exact, az_exact),
        "E_B_relative_full": _relative(b_error, b_reference),
        "E_B_relative_core": _relative(b_core_error, b_core_reference),
        "E_B_relative_central": _relative(
            b_central_error, b_central_reference
        ),
        "E_J_relative_full": _relative(j_error, j_reference),
        "E_J_relative_core": _relative(j_core_error, j_core_reference),
        "E_J_relative_central": _relative(
            j_central_error, j_central_reference
        ),
        "E_Bx_relative_full": b_component[0],
        "E_By_relative_full": b_component[1],
        "E_Bz_relative_full": b_component[2],
        "E_Jx_relative_full": j_component[0],
        "E_Jy_relative_full": j_component[1],
        "E_Jz_relative_full": j_component[2],
        "E_Bx_relative_core": b_core_component[0],
        "E_By_relative_core": b_core_component[1],
        "E_Bz_relative_core": b_core_component[2],
        "E_Jx_relative_core": j_core_component[0],
        "E_Jy_relative_core": j_core_component[1],
        "E_Jz_relative_core": j_core_component[2],
        "E_Bx_relative_central": b_central_component[0],
        "E_By_relative_central": b_central_component[1],
        "E_Bz_relative_central": b_central_component[2],
        "E_Jx_relative_central": j_central_component[0],
        "E_Jy_relative_central": j_central_component[1],
        "E_Jz_relative_central": j_central_component[2],
        "E_B_discrete_path_relative": _relative(b_num - b_discrete, b_discrete),
        "E_J_discrete_path_relative": _relative(j_num - j_discrete, j_discrete),
        "R_phi_bottom": _rms(phi_num[0] - phi_exact[0]),
        "R_phi_bottom_derivative": _rms(
            _bottom_derivative(solver, phi_num) - solver._bottom_dphi_c_target
        ),
        "R_phi_top_robin": _rms(_top_robin_residual(solver, phi_num, zero_top)),
        "R_phi_pde": _rms(phi_pde),
        "R_Az_bottom": _rms(az_num[0] - az_exact[0]),
        "R_Az_top": _rms(az_num[-1]),
        "R_Az_lateral": max(
            _rms(az_num[:, :, 0]),
            _rms(az_num[:, :, -1]),
            _rms(az_num[:, 0, :]),
            _rms(az_num[:, -1, :]),
        ),
        "R_Az_pde": _rms(az_pde[:, 1:-1, 1:-1]),
        "R_Az_pde_all_xy": _rms(az_pde),
        "div_B_rms_full": _rms(div_b),
        "div_B_rms_core": _rms(div_b[1:-1, 1:-1, 1:-1]),
        "div_B_rms_central": _rms(
            div_b[quarter : n - quarter, quarter : n - quarter, quarter : n - quarter]
        ),
        "curl_B_minus_J_relative_full": _relative(curl_residual, j_num),
        "curl_B_minus_J_relative_core": _relative(
            curl_residual[:, 1:-1, 1:-1, 1:-1],
            j_num[:, 1:-1, 1:-1, 1:-1],
        ),
        "curl_B_minus_J_relative_central": _relative(
            curl_residual[central], j_num[central]
        ),
        "wall_time_s": wall_time,
        "order_phi": None,
        "order_Az": None,
        "order_B_core": None,
        "order_J_core": None,
        "order_B_central": None,
        "order_J_central": None,
        "phi_components": fields["phi_components"],
        "az_components": fields["az_components"],
    }


def _add_orders(
    rows: list[dict[str, object]],
    *,
    group_keys: Sequence[str],
    error_order_pairs: Sequence[tuple[str, str]],
) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: float(row["h"]), reverse=True)
        for coarse, fine in zip(group, group[1:]):
            h_ratio = float(coarse["h"]) / float(fine["h"])
            for error_key, order_key in error_order_pairs:
                coarse_error = float(coarse[error_key])
                fine_error = float(fine[error_key])
                if coarse_error > 0.0 and fine_error > 0.0:
                    fine[order_key] = math.log(coarse_error / fine_error) / math.log(
                        h_ratio
                    )


def _write_csv(path: Path, rows: list[dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finest(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(
            row.get(name)
            for name in (
                "experiment",
                "dtype",
                "weight_scheme",
                "weight_parameter",
                "target_kind",
            )
        )
        if key not in best or float(row["h"]) < float(best[key]["h"]):
            best[key] = row
    return list(best.values())


def main() -> int:
    args = parse_args()
    try:
        grids = _parse_ints(args.grids, minimum=5)
        dtypes = _resolve_dtypes(args.dtypes)
        device = _resolve_device(args.device)
        rhos = _parse_floats(args.mixed_rhos, minimum=0.0)
        gammas = _parse_floats(args.mixed_gammas, minimum=0.0)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if len(grids) < 3:
        raise SystemExit("--grids must contain at least three refinement levels")
    if any(rho > 1.0 for rho in rhos):
        raise SystemExit("--mixed-rhos values must be <= 1")
    if not 0.0 <= args.complete_rho <= 1.0:
        raise SystemExit("--complete-rho must be between 0 and 1")
    if args.incompatible_amplitude <= 0.0:
        raise SystemExit("--incompatible-amplitude must be positive")

    output_dir = args.output_dir or (
        ROOT / "results" / "nfgs_paper" / "extended_manufactured"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed_rows: list[dict[str, object]] = []
    mixed_profiles: list[dict[str, object]] = []
    if not args.skip_mixed:
        fixed_configs = [("fixed", rho, None) for rho in rhos]
        for dtype in dtypes:
            for n in grids:
                h = 1.0 / (n - 1)
                scaled_configs = [
                    ("gamma_scaled", gamma * h / (1.0 + gamma * h), gamma)
                    for gamma in gammas
                ]
                for scheme, rho, gamma in fixed_configs + scaled_configs:
                    target_kinds = ("compatible",) if rho == 0.0 else (
                        "compatible",
                        "incompatible",
                    )
                    for target_kind in target_kinds:
                        row, profile = run_mixed_case(
                            n,
                            device=device,
                            dtype=dtype,
                            rho=rho,
                            weight_scheme=scheme,
                            gamma=gamma,
                            target_kind=target_kind,
                            incompatible_amplitude=args.incompatible_amplitude,
                        )
                        mixed_rows.append(row)
                        mixed_profiles.append(
                            {
                                "dtype": row["dtype"],
                                "n": n,
                                "h": row["h"],
                                "weight_scheme": scheme,
                                "gamma": gamma,
                                "rho": rho,
                                "target_kind": target_kind,
                                "z_over_H": [index / (n - 1) for index in range(n)],
                                "error_rms_by_z": profile,
                            }
                        )
                        print(
                            f"[mixed] dtype={row['dtype']} n={n} scheme={scheme} "
                            f"rho={rho:.6g} target={target_kind} "
                            f"E_phi={float(row['E_phi']):.6e} "
                            f"R_mix={float(row['R_mixed_algebraic']):.6e}",
                            flush=True,
                        )
        _add_orders(
            mixed_rows,
            group_keys=("dtype", "weight_scheme", "weight_parameter", "target_kind"),
            error_order_pairs=(
                ("E_phi", "order_phi"),
                ("E_phi_bulk", "order_phi_bulk"),
            ),
        )

    complete_rows: list[dict[str, object]] = []
    if not args.skip_complete:
        for dtype in dtypes:
            for n in grids:
                row = run_complete_case(
                    n,
                    device=device,
                    dtype=dtype,
                    rho=args.complete_rho,
                )
                complete_rows.append(row)
                print(
                    f"[complete] dtype={row['dtype']} n={n} "
                    f"E_phi={float(row['E_phi_relative']):.6e} "
                    f"E_Az={float(row['E_Az_relative']):.6e} "
                    f"E_B_core={float(row['E_B_relative_core']):.6e} "
                    f"E_J_core={float(row['E_J_relative_core']):.6e}",
                    flush=True,
                )
        _add_orders(
            complete_rows,
            group_keys=("dtype", "rho"),
            error_order_pairs=(
                ("E_phi", "order_phi"),
                ("E_Az", "order_Az"),
                ("E_B_relative_core", "order_B_core"),
                ("E_J_relative_core", "order_J_core"),
                ("E_B_relative_central", "order_B_central"),
                ("E_J_relative_central", "order_J_central"),
            ),
        )

    mixed_fields = [
        "experiment", "dtype", "device", "n", "h", "weight_scheme",
        "weight_parameter", "gamma", "rho", "target_kind", "incompatible_amplitude",
        "incompatible_perturbation_rms", "E_phi", "E_phi_relative", "order_phi",
        "E_phi_first_unknown", "E_phi_near_bottom", "E_phi_bulk",
        "order_phi_bulk", "R_bottom_dirichlet", "R_bottom_derivative_target",
        "R_bottom_derivative_compatible", "R_first_pde", "R_bulk_pde",
        "R_mixed_algebraic", "R_mixed_algebraic_relative", "R_top_robin",
        "max_profile_error_z", "wall_time_s",
    ]
    complete_fields = [
        "experiment", "dtype", "device", "n", "h", "rho", "E_phi",
        "E_phi_relative", "order_phi", "E_Az", "E_Az_relative", "order_Az",
        "E_B_relative_full", "E_B_relative_core", "order_B_core",
        "E_B_relative_central", "order_B_central", "E_J_relative_full",
        "E_J_relative_core", "order_J_core", "E_J_relative_central",
        "order_J_central",
        "E_Bx_relative_full", "E_By_relative_full", "E_Bz_relative_full",
        "E_Jx_relative_full", "E_Jy_relative_full", "E_Jz_relative_full",
        "E_Bx_relative_core", "E_By_relative_core", "E_Bz_relative_core",
        "E_Jx_relative_core", "E_Jy_relative_core", "E_Jz_relative_core",
        "E_Bx_relative_central", "E_By_relative_central",
        "E_Bz_relative_central", "E_Jx_relative_central",
        "E_Jy_relative_central", "E_Jz_relative_central",
        "E_B_discrete_path_relative", "E_J_discrete_path_relative",
        "R_phi_bottom", "R_phi_bottom_derivative", "R_phi_top_robin", "R_phi_pde",
        "R_Az_bottom", "R_Az_top", "R_Az_lateral", "R_Az_pde",
        "R_Az_pde_all_xy", "div_B_rms_full", "div_B_rms_core",
        "div_B_rms_central", "curl_B_minus_J_relative_full",
        "curl_B_minus_J_relative_core", "curl_B_minus_J_relative_central",
        "wall_time_s",
    ]
    if mixed_rows:
        _write_csv(output_dir / "mixed_row.csv", mixed_rows, mixed_fields)
        (output_dir / "mixed_profiles.json").write_text(
            json.dumps(mixed_profiles, indent=2) + "\n"
        )
    if complete_rows:
        _write_csv(output_dir / "complete_fields.csv", complete_rows, complete_fields)

    summary = {
        "experiment": "extended NFGS manufactured validation",
        "timestamp_unix": time.time(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "configuration": {
            "grids": grids,
            "dtypes": [_dtype_name(dtype) for dtype in dtypes],
            "mixed_fixed_rhos": rhos,
            "mixed_gammas": gammas,
            "mixed_gamma_definition": "rho(h)=gamma*h/(1+gamma*h)",
            "incompatible_amplitude": args.incompatible_amplitude,
            "complete_rho": args.complete_rho,
            "horizontal_wavenumber": (
                "k_h=sqrt(-lambda_h) from the actual endpoint DCT-I/DST-I spectra"
            ),
            "compatible_derivative_target": (
                "one-sided second-order derivative of the sampled exact Phi"
            ),
            "complete_reference": (
                "semidiscrete horizontal second derivatives with analytic first "
                "derivatives; core excludes one cell and central uses the middle "
                "half of every coordinate"
            ),
        },
        "finest_mixed": _finest(mixed_rows),
        "finest_complete": _finest(complete_rows),
        "mixed_row": mixed_rows,
        "complete_fields": complete_rows,
        "interpretation_guards": [
            "The incompatible derivative target is intentionally inconsistent with "
            "the exact bottom Dirichlet trace and is not expected to converge to the "
            "same manufactured Phi.",
            "Full-domain J includes the production endpoint second-z-derivative "
            "closure; convergence claims should use the explicitly reported core.",
            "This is an isolated linear scalar/reconstruction verification, not a "
            "nonlinear NLFFF convergence experiment.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"results={output_dir}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
