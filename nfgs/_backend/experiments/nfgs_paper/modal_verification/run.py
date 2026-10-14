#!/usr/bin/env python3
"""Run manufactured-solution checks for the NFGS modal Robin solver."""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs_solver import NFGSSolverGPU  # noqa: E402


CASE_COMPONENTS = {
    "homogeneous": ((1, 0, 1.0, 0.0),),
    "forced": ((1, 0, 1.0, 0.20),),
    "multimode": ((1, 1, 1.0, 0.12), (2, 1, 0.30, -0.05)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--grids", default="17,33,65")
    parser.add_argument("--longwave-nx", default="17,33,65,129")
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--skip-long-wave", action="store_true")
    parser.add_argument("--skip-zero-mode", action="store_true")
    parser.add_argument(
        "--zero-mode-policy",
        choices=(
            "frozen_linear_harmonic",
            "constrained_spline_harmonic",
            "coupled_k0",
            "omit",
        ),
        default="frozen_linear_harmonic",
        help=(
            "Physical horizontal zero-mode closure tested when the zero-mode "
            "check is enabled. The production default verifies the constant, "
            "current-free frozen lift."
        ),
    )
    return parser.parse_args()


def _parse_ints(value: str, *, minimum: int) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or result[0] < minimum:
        raise ValueError(f"values must be integers >= {minimum}")
    return result


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _resolve_dtypes(value: str) -> list[torch.dtype]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    mapping = {"float32": torch.float32, "float64": torch.float64}
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_solver(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    horizontal_zero_mode_policy: str = "omit",
    k0_ode_smoothness: float = 0.0,
) -> NFGSSolverGPU:
    return NFGSSolverGPU(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        device=device,
        dtype=dtype,
        horizontal_zero_mode_policy=horizontal_zero_mode_policy,
        k0_ode_smoothness=k0_ode_smoothness,
        buffer_thickness=0,
        padded_buffer_zone=False,
        pad_calculation_domain=False,
    )


def _cosine_mode(solver: NFGSSolverGPU, mx: int, my: int) -> torch.Tensor:
    x = torch.arange(solver.Nx, device=solver.device, dtype=solver.dtype)
    y = torch.arange(solver.Ny, device=solver.device, dtype=solver.dtype)
    x *= float(solver.dx)
    y *= float(solver.dy)
    lx = max((solver.Nx - 1) * float(solver.dx), float(solver.dx))
    ly = max((solver.Ny - 1) * float(solver.dy), float(solver.dy))
    mode_x = torch.cos(math.pi * mx * x / lx)
    mode_y = torch.cos(math.pi * my * y / ly)
    return mode_y[:, None] * mode_x[None, :]


def manufactured_fields(
    solver: NFGSSolverGPU,
    components: Iterable[tuple[int, int, float, float]],
) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    """Build semidiscrete modal MMS fields on the solver's DCT-I basis.

    Each component is ``(m_x, m_y, amplitude, a)`` and uses
    ``f(z)=exp(-k_h z)+a z^2`` with ``k_h=sqrt(-lambda_h)`` from the actual
    endpoint-Neumann finite-difference spectrum.  This makes the horizontal
    modal operator exact and isolates the second-order z/Robin discretization.
    """
    shape = (solver.Nz, solver.Ny, solver.Nx)
    phi = torch.zeros(shape, device=solver.device, dtype=solver.dtype)
    source = torch.zeros_like(phi)
    bz = torch.zeros_like(phi)
    bottom = torch.zeros(
        (solver.Ny, solver.Nx), device=solver.device, dtype=solver.dtype
    )
    top_robin = torch.zeros_like(bottom)
    z = torch.arange(solver.Nz, device=solver.device, dtype=solver.dtype)
    z *= float(solver.dz)
    height = float((solver.Nz - 1) * solver.dz)
    lam_xy = solver._endpoint_neumann_lam_xy()
    metadata: list[dict[str, float]] = []

    for mx, my, amplitude, a_value in components:
        lam = float(lam_xy[my, mx].item())
        kh = math.sqrt(max(-lam, 0.0))
        mode = _cosine_mode(solver, mx, my)
        a = torch.as_tensor(a_value, device=solver.device, dtype=solver.dtype)
        f = torch.exp(-kh * z) + a * z * z
        src_z = 2.0 * a - kh * kh * a * z * z
        g_top = 2.0 * a_value * height + kh * a_value * height * height
        scale = torch.as_tensor(amplitude, device=solver.device, dtype=solver.dtype)
        phi += scale * f[:, None, None] * mode[None, :, :]
        source += scale * src_z[:, None, None] * mode[None, :, :]
        bz += scale * lam * f[:, None, None] * mode[None, :, :]
        bottom += scale * mode
        top_robin += scale * g_top * mode
        metadata.append(
            {
                "mx": float(mx),
                "my": float(my),
                "amplitude": float(amplitude),
                "a": float(a_value),
                "lambda_h": lam,
                "k_h": kh,
                "k_h_H": kh * height,
            }
        )
    return {
        "phi": phi,
        "source": source,
        "bz": bz,
        "bottom": bottom,
        "top_robin": top_robin,
        "components": metadata,
    }


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


def run_manufactured_case(
    case: str,
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    timing_repeats: int = 1,
) -> dict:
    if case not in CASE_COMPONENTS:
        raise ValueError(f"unknown manufactured case: {case}")
    spacing = 1.0 / (n - 1)
    solver = make_solver(
        n, n, n, spacing, spacing, spacing, device=device, dtype=dtype
    )
    fields = manufactured_fields(solver, CASE_COMPONENTS[case])
    source = fields["source"]
    bottom = fields["bottom"]
    top_robin = fields["top_robin"]

    if timing_repeats < 1:
        raise ValueError("timing_repeats must be >= 1")
    source_hat = solver._dct2(source)
    bottom_hat = solver._dct2(bottom)
    top_robin_hat = solver._dct2(top_robin)
    timings: list[float] = []
    phi_num = None
    # One untimed call removes FFT-plan and allocator startup from the table.
    with torch.inference_mode():
        phi_num = solver._solve_modal_robin_from_hat(
            source_hat, bottom_hat, top_robin_hat=top_robin_hat
        )
        for _ in range(timing_repeats):
            _sync(device)
            start = time.perf_counter()
            phi_num = solver._solve_modal_robin_from_hat(
                source_hat, bottom_hat, top_robin_hat=top_robin_hat
            )
            _sync(device)
            timings.append(time.perf_counter() - start)
    assert phi_num is not None
    wall_time = statistics.median(timings)

    phi_exact = fields["phi"]
    bz_num = solver._lap_xy(phi_num)
    bz_exact = fields["bz"]
    phi_error = _rms(phi_num - phi_exact)
    bz_error = _rms(bz_num - bz_exact)
    phi_scale = max(_rms(phi_exact), 1.0e-30)
    bz_scale = max(_rms(bz_exact), 1.0e-30)
    pde = (
        (phi_num[2:] - 2.0 * phi_num[1:-1] + phi_num[:-2])
        / float(solver.dz) ** 2
        + solver._lap_xy(phi_num)[1:-1]
        - source[1:-1]
    )
    return {
        "case": case,
        "dtype": _dtype_name(dtype),
        "device": str(device),
        "nx": n,
        "ny": n,
        "nz": n,
        "h": spacing,
        "mode_count": len(CASE_COMPONENTS[case]),
        "E_phi": phi_error,
        "E_phi_relative": phi_error / phi_scale,
        "E_Bz": bz_error,
        "E_Bz_relative": bz_error / bz_scale,
        "R_D": _rms(phi_num[0] - bottom),
        "R_R": _rms(_top_robin_residual(solver, phi_num, top_robin)),
        "PDE_residual": _rms(pde),
        "wall_time_s": wall_time,
        "timing_repeats": timing_repeats,
        "order_phi": None,
        "order_Bz": None,
        "components": fields["components"],
    }


def add_observed_orders(rows: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["case"], row["dtype"]), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda item: item["h"], reverse=True)
        for coarse, fine in zip(group, group[1:]):
            h_ratio = coarse["h"] / fine["h"]
            if coarse["E_phi"] > 0.0 and fine["E_phi"] > 0.0:
                fine["order_phi"] = math.log(
                    coarse["E_phi"] / fine["E_phi"]
                ) / math.log(h_ratio)
            if coarse["E_Bz"] > 0.0 and fine["E_Bz"] > 0.0:
                fine["order_Bz"] = math.log(
                    coarse["E_Bz"] / fine["E_Bz"]
                ) / math.log(h_ratio)


def run_long_wavelength_case(
    nx: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    perturbation_amplitude: float = 1.0e-7,
) -> dict:
    ny = 9
    nz = 65
    solver = make_solver(
        nx, ny, nz, 1.0, 1.0, 1.0 / (nz - 1), device=device, dtype=dtype
    )
    fields = manufactured_fields(solver, ((1, 0, 1.0, 0.20),))
    lam = float(solver._endpoint_neumann_lam_xy()[0, 1].item())
    kh = math.sqrt(-lam)
    source = fields["source"]
    bottom = fields["bottom"]
    top_robin = fields["top_robin"]

    with torch.inference_mode():
        phi_direct = solver._solve_modal_robin_from_hat(
            solver._dct2(source),
            solver._dct2(bottom),
            top_robin_hat=solver._dct2(top_robin),
        )
        bz_source = lam * source
        bz_bottom = lam * bottom
        bz_top_robin = lam * top_robin
        bz_num = solver._solve_modal_robin_from_hat(
            solver._dct2(bz_source),
            solver._dct2(bz_bottom),
            top_robin_hat=solver._dct2(bz_top_robin),
        )
        lam_xy = solver._endpoint_neumann_lam_xy()
        lam_safe = lam_xy.clone()
        lam_safe[0, 0] = 1.0
        phi_via_bz_hat = solver._dct2(bz_num) / lam_safe[None, :, :]
        phi_via_bz_hat[:, 0, 0] = 0.0
        phi_via_bz = solver._idct2(phi_via_bz_hat)

        mode = _cosine_mode(solver, 1, 0)
        bz_perturbation = perturbation_amplitude * mode[None, :, :]
        perturbed_hat = (
            solver._dct2(bz_num + bz_perturbation) / lam_safe[None, :, :]
        )
        perturbed_hat[:, 0, 0] = 0.0
        phi_perturbed = solver._idct2(perturbed_hat)

    perturbation_rms = _rms(bz_perturbation.expand_as(bz_num))
    propagated_rms = _rms(phi_perturbed - phi_via_bz)
    exact = fields["phi"]
    return {
        "dtype": _dtype_name(dtype),
        "device": str(device),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "domain_length_x": float(nx - 1),
        "lambda_h": lam,
        "k_h": kh,
        "k_h_H": kh * float((nz - 1) * solver.dz),
        "E_phi_direct": _rms(phi_direct - exact),
        "E_phi_via_Bz": _rms(phi_via_bz - exact),
        "direct_vs_via_Bz": _rms(phi_direct - phi_via_bz),
        "Bz_perturbation_rms": perturbation_rms,
        "propagated_phi_perturbation_rms": propagated_rms,
        "measured_inverse_amplification": propagated_rms / perturbation_rms,
        "predicted_inverse_amplification": 1.0 / abs(lam),
    }


def _scalar_zero_mode_projection_max_abs(
    solver: NFGSSolverGPU,
) -> float:
    """Verify that the scalar modal Robin solve still projects its gauge k=0."""
    n = solver.Nz
    source_hat = torch.zeros(
        (n, solver.Ny, solver.Nx), device=solver.device, dtype=solver.dtype
    )
    bottom_hat = torch.zeros(
        (solver.Ny, solver.Nx), device=solver.device, dtype=solver.dtype
    )
    top_hat = torch.zeros_like(bottom_hat)
    source_hat[:, 0, 0] = 1.0
    bottom_hat[0, 0] = 1.0
    top_hat[0, 0] = 1.0
    projected = solver._solve_modal_robin_from_hat(
        source_hat, bottom_hat, top_robin_hat=top_hat
    )
    return float(projected.abs().max().item())


def run_coupled_zero_mode_case(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    spacing = 1.0 / (n - 1)
    solver = make_solver(
        n,
        n,
        n,
        spacing,
        spacing,
        spacing,
        device=device,
        dtype=dtype,
        horizontal_zero_mode_policy="coupled_k0",
        k0_ode_smoothness=0.0,
    )
    shape = (n, n)
    hx0 = 0.70
    hy0 = -0.40
    solver.set_bottom_vector_boundary(
        torch.full(shape, hx0, device=device, dtype=dtype),
        torch.full(shape, hy0, device=device, dtype=dtype),
        torch.ones(shape, device=device, dtype=dtype),
        smooth_passes=0,
    )
    z = torch.arange(n, device=device, dtype=dtype) * spacing
    hx_exact = hx0 - 0.20 * z + 0.15 * z * z
    hy_exact = hy0 + 0.30 * z - 0.10 * z * z
    qx = -0.20 + 0.30 * z
    qy = 0.30 - 0.20 * z
    with torch.inference_mode():
        hx_num = solver._solve_coupled_k0_profile(
            qx, torch.tensor(hx0, device=device, dtype=dtype)
        )
        hy_num = solver._solve_coupled_k0_profile(
            qy, torch.tensor(hy0, device=device, dtype=dtype)
        )
        profile = torch.stack((hx_num, hy_num))
        derivative = solver._k0_center_derivative(profile)
        face_derivative = (profile[:, 1:] - profile[:, :-1]) / spacing

        scalar_k0_max = _scalar_zero_mode_projection_max_abs(solver)

    desired = torch.stack((qx, qy))
    desired_face = 0.5 * (desired[:, :-1] + desired[:, 1:])
    return {
        "zero_mode_policy": "coupled_k0",
        "dtype": _dtype_name(dtype),
        "device": str(device),
        "nx": n,
        "ny": n,
        "nz": n,
        "h": spacing,
        "E_Hx": _rms(hx_num - hx_exact),
        "E_Hy": _rms(hy_num - hy_exact),
        "bottom_residual": max(
            abs(float(hx_num[0].item()) - hx0),
            abs(float(hy_num[0].item()) - hy0),
        ),
        "derivative_residual": _rms(derivative - desired),
        "face_derivative_residual": _rms(face_derivative - desired_face),
        "horizontal_current_max_abs": float(
            solver._k0_field_and_current(profile)[1].abs().max().item()
        ),
        "scalar_modal_k0_projected_max_abs": scalar_k0_max,
        "k0_ode_smoothness": 0.0,
    }


def run_frozen_zero_mode_case(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Recover the production constant horizontal lift and its zero current."""
    spacing = 1.0 / (n - 1)
    solver = make_solver(
        n,
        n,
        n,
        spacing,
        spacing,
        spacing,
        device=device,
        dtype=dtype,
        horizontal_zero_mode_policy="frozen_linear_harmonic",
    )
    shape = (n, n)
    hx0 = 0.70
    hy0 = -0.40
    solver.set_bottom_vector_boundary(
        torch.full(shape, hx0, device=device, dtype=dtype),
        torch.full(shape, hy0, device=device, dtype=dtype),
        torch.ones(shape, device=device, dtype=dtype),
        smooth_passes=0,
    )
    profile = solver._H_k0
    assert profile is not None
    target = solver._H_k0_bottom_target
    assert target is not None
    exact = target[:, None].expand(-1, n)
    with torch.inference_mode():
        derivative = solver._k0_center_derivative(profile)
        face_derivative = (profile[:, 1:] - profile[:, :-1]) / spacing
        _b_harm, j_harm = solver._k0_field_and_current()
        scalar_k0_max = _scalar_zero_mode_projection_max_abs(solver)
    return {
        "zero_mode_policy": "frozen_linear_harmonic",
        "dtype": _dtype_name(dtype),
        "device": str(device),
        "nx": n,
        "ny": n,
        "nz": n,
        "h": spacing,
        "input_Hx": hx0,
        "input_Hy": hy0,
        "target_Hx": float(target[0].item()),
        "target_Hy": float(target[1].item()),
        "E_Hx": _rms(profile[0] - exact[0]),
        "E_Hy": _rms(profile[1] - exact[1]),
        "bottom_residual": float((profile[:, 0] - target).abs().max().item()),
        "derivative_residual": _rms(derivative),
        "face_derivative_residual": _rms(face_derivative),
        "horizontal_current_max_abs": float(j_harm.abs().max().item()),
        "scalar_modal_k0_projected_max_abs": scalar_k0_max,
        "k0_ode_smoothness": 0.0,
    }


def run_zero_mode_case(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    policy: str = "frozen_linear_harmonic",
) -> dict:
    if policy == "frozen_linear_harmonic":
        return run_frozen_zero_mode_case(n, device=device, dtype=dtype)
    if policy == "coupled_k0":
        return run_coupled_zero_mode_case(n, device=device, dtype=dtype)
    raise ValueError(f"No modal manufactured zero-mode check for policy={policy!r}")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _verification_summary(
    convergence: list[dict], longwave: list[dict], zero_mode: list[dict]
) -> dict:
    finest: dict[tuple[str, str], dict] = {}
    for row in convergence:
        key = (row["dtype"], row["case"])
        if key not in finest or row["h"] < finest[key]["h"]:
            finest[key] = row
    float64_fine = [
        row for (dtype, _case), row in finest.items() if dtype == "float64"
    ]
    longwave_mismatch = [
        abs(
            row["measured_inverse_amplification"]
            - row["predicted_inverse_amplification"]
        )
        / row["predicted_inverse_amplification"]
        for row in longwave
    ]
    return {
        "float64_second_order_pass": bool(float64_fine)
        and all(
            row["order_phi"] is not None
            and row["order_phi"] >= 1.9
            and row["order_Bz"] is not None
            and row["order_Bz"] >= 1.9
            for row in float64_fine
        ),
        "float64_finest_min_order_phi": min(
            (row["order_phi"] for row in float64_fine), default=None
        ),
        "float64_finest_min_order_Bz": min(
            (row["order_Bz"] for row in float64_fine), default=None
        ),
        "float64_max_R_D": max(
            (row["R_D"] for row in convergence if row["dtype"] == "float64"),
            default=None,
        ),
        "float64_max_R_R": max(
            (row["R_R"] for row in convergence if row["dtype"] == "float64"),
            default=None,
        ),
        "float32_max_R_D": max(
            (row["R_D"] for row in convergence if row["dtype"] == "float32"),
            default=None,
        ),
        "float32_max_R_R": max(
            (row["R_R"] for row in convergence if row["dtype"] == "float32"),
            default=None,
        ),
        "longwave_max_relative_amplification_mismatch": max(
            longwave_mismatch, default=None
        ),
        "zero_mode_float64_max_profile_error": max(
            (
                max(row["E_Hx"], row["E_Hy"])
                for row in zero_mode
                if row["dtype"] == "float64"
            ),
            default=None,
        ),
        "zero_mode_float32_max_profile_error": max(
            (
                max(row["E_Hx"], row["E_Hy"])
                for row in zero_mode
                if row["dtype"] == "float32"
            ),
            default=None,
        ),
        "zero_mode_float64_max_face_derivative_residual": max(
            (
                row["face_derivative_residual"]
                for row in zero_mode
                if row["dtype"] == "float64"
            ),
            default=None,
        ),
        "zero_mode_float32_max_face_derivative_residual": max(
            (
                row["face_derivative_residual"]
                for row in zero_mode
                if row["dtype"] == "float32"
            ),
            default=None,
        ),
        "zero_mode_max_horizontal_current_abs": max(
            (row["horizontal_current_max_abs"] for row in zero_mode),
            default=None,
        ),
        "scalar_modal_k0_projected_max_abs": max(
            (row["scalar_modal_k0_projected_max_abs"] for row in zero_mode),
            default=None,
        ),
    }


def main() -> int:
    args = parse_args()
    try:
        grids = _parse_ints(args.grids, minimum=3)
        longwave_nx = _parse_ints(args.longwave_nx, minimum=3)
        dtypes = _resolve_dtypes(args.dtypes)
        device = _resolve_device(args.device)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if len(grids) < 3:
        raise SystemExit("--grids must contain at least three refinement levels")
    if args.timing_repeats < 1:
        raise SystemExit("--timing-repeats must be >= 1")
    output_dir = args.output_dir or (
        ROOT
        / "results"
        / "nfgs_paper"
        / "frozen_linear_harmonic_vertical_staggered"
        / "modal"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    convergence: list[dict] = []
    for dtype in dtypes:
        for case in CASE_COMPONENTS:
            for n in grids:
                row = run_manufactured_case(
                    case,
                    n,
                    device=device,
                    dtype=dtype,
                    timing_repeats=args.timing_repeats,
                )
                convergence.append(row)
                print(
                    f"[modal] dtype={row['dtype']} case={case} n={n} "
                    f"E_phi={row['E_phi']:.6e} R_R={row['R_R']:.6e}",
                    flush=True,
                )
    add_observed_orders(convergence)

    longwave: list[dict] = []
    if not args.skip_long_wave:
        for dtype in dtypes:
            for nx in longwave_nx:
                row = run_long_wavelength_case(nx, device=device, dtype=dtype)
                longwave.append(row)
                print(
                    f"[long-wave] dtype={row['dtype']} nx={nx} "
                    f"kH={row['k_h_H']:.3e} amplification="
                    f"{row['measured_inverse_amplification']:.6e}",
                    flush=True,
                )

    zero_mode: list[dict] = []
    if not args.skip_zero_mode:
        for dtype in dtypes:
            for n in grids:
                row = run_zero_mode_case(
                    n,
                    device=device,
                    dtype=dtype,
                    policy=args.zero_mode_policy,
                )
                zero_mode.append(row)
                print(
                    f"[zero-mode] policy={row['zero_mode_policy']} "
                    f"dtype={row['dtype']} n={n} "
                    f"E_Hx={row['E_Hx']:.6e} E_Hy={row['E_Hy']:.6e}",
                    flush=True,
                )

    _write_csv(
        output_dir / "convergence.csv",
        convergence,
        [
            "case", "dtype", "device", "nx", "ny", "nz", "h", "mode_count",
            "E_phi", "E_phi_relative", "order_phi", "E_Bz", "E_Bz_relative",
            "order_Bz", "R_D", "R_R", "PDE_residual", "wall_time_s",
            "timing_repeats",
        ],
    )
    if longwave:
        _write_csv(
            output_dir / "long_wavelength.csv",
            longwave,
            [
                "dtype", "device", "nx", "ny", "nz", "domain_length_x",
                "lambda_h", "k_h", "k_h_H", "E_phi_direct", "E_phi_via_Bz",
                "direct_vs_via_Bz", "Bz_perturbation_rms",
                "propagated_phi_perturbation_rms", "measured_inverse_amplification",
                "predicted_inverse_amplification",
            ],
        )
    if zero_mode:
        _write_csv(
            output_dir / "zero_mode.csv",
            zero_mode,
            [
                "zero_mode_policy", "dtype", "device", "nx", "ny", "nz", "h",
                "input_Hx", "input_Hy", "target_Hx", "target_Hy", "E_Hx", "E_Hy",
                "bottom_residual", "derivative_residual",
                "face_derivative_residual",
                "horizontal_current_max_abs",
                "scalar_modal_k0_projected_max_abs", "k0_ode_smoothness",
            ],
        )

    summary = {
        "experiment": "NFGS modal manufactured solutions",
        "device": str(device),
        "dtypes": [_dtype_name(dtype) for dtype in dtypes],
        "grids": grids,
        "timing_repeats": args.timing_repeats,
        "semidiscrete_horizontal_basis": True,
        "horizontal_wavenumber_definition": (
            "k_h=sqrt(-lambda_h) from endpoint-Neumann DCT-I spectrum"
        ),
        "vertical_operator": "second-order centered interior with second-order backward top Robin",
        "horizontal_zero_mode_policy": args.zero_mode_policy,
        "horizontal_zero_mode_discretization": (
            "analytic_constant_Az_linear_lift"
            if args.zero_mode_policy == "frozen_linear_harmonic"
            else "staggered_face_D1_true_three_point_curvature"
        ),
        "top_boundary": "(d_z+k_h)Phi=g_R",
        "cases": {
            key: [list(component) for component in value]
            for key, value in CASE_COMPONENTS.items()
        },
        "verification": _verification_summary(convergence, longwave, zero_mode),
        "convergence": convergence,
        "long_wavelength": longwave,
        "zero_mode": zero_mode,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"results={output_dir}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
