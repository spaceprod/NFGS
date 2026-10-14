#!/usr/bin/env python3
"""Manufactured Gaussian source-attribution test with two compact current loops.

One closed filament is entirely below z=0 and the other entirely above z=0.
Their Biot--Savart fields are evaluated independently, so the source support
and the correct boundary-field components are known without using the NFGS
operator itself.
"""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs_solver import NFGSSolverGPU  # noqa: E402


def circular_filament_field(
    points: np.ndarray,
    *,
    center: tuple[float, float, float],
    normal: tuple[float, float, float],
    radius: float,
    current: float,
    segments: int,
    chunk_segments: int = 256,
) -> np.ndarray:
    """Midpoint-quadrature Biot--Savart field of an arbitrarily tilted loop."""
    target = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    axis = np.asarray(normal, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    trial = np.asarray((1.0, 0.0, 0.0) if abs(axis[0]) < 0.8 else (0.0, 1.0, 0.0))
    u = np.cross(axis, trial)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = (np.arange(int(segments), dtype=np.float64) + 0.5) * (2.0 * np.pi / segments)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    source = np.asarray(center)[None, :] + radius * (cos_t[:, None] * u + sin_t[:, None] * v)
    dl = radius * (2.0 * np.pi / segments) * (-sin_t[:, None] * u + cos_t[:, None] * v)
    result = np.zeros((target.shape[0], 3), dtype=np.float64)
    for start in range(0, int(segments), int(chunk_segments)):
        stop = min(start + int(chunk_segments), int(segments))
        displacement = target[:, None, :] - source[None, start:stop, :]
        distance3 = np.maximum(np.sum(displacement**2, axis=-1), 1.0e-24) ** 1.5
        result += np.sum(
            np.cross(dl[None, start:stop, :], displacement) / distance3[..., None],
            axis=1,
        )
    return (float(current) / (4.0 * np.pi)) * result.reshape(np.asarray(points).shape)


def vector_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)[:, mask].T
    cand = np.asarray(candidate, dtype=np.float64)[:, mask].T
    diff = cand - ref
    ref2 = float(np.sum(ref * ref))
    cand2 = float(np.sum(cand * cand))
    return {
        "C_vec": float(np.sum(ref * cand)) / max(math.sqrt(ref2 * cand2), 1.0e-30),
        "vector_nrmse": float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))
        / max(float(np.sqrt(np.mean(np.sum(ref * ref, axis=-1)))), 1.0e-30),
        "amplitude_ratio": math.sqrt(cand2 / max(ref2, 1.0e-30)),
    }


def run_resolution(n: int, segments: int, extent: float) -> tuple[list[dict[str, float | int | str]], dict]:
    axis = np.linspace(-extent, extent, int(n))
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    below_xyz = circular_filament_field(
        points, center=(-0.18, 0.08, -0.78), normal=(0.25, -0.15, 1.0),
        radius=0.24, current=1.0, segments=segments,
    )
    above_xyz = circular_filament_field(
        points, center=(0.22, -0.12, 0.82), normal=(-0.20, 0.30, 1.0),
        radius=0.22, current=-0.85, segments=segments,
    )
    total_xyz = below_xyz + above_xyz
    total = np.moveaxis(total_xyz, -1, 0)
    below = np.moveaxis(below_xyz, -1, 0)
    above = np.moveaxis(above_xyz, -1, 0)
    spacing = float(axis[1] - axis[0])
    solver = NFGSSolverGPU(
        n, n, max(8, n // 2), spacing, spacing, spacing,
        device="cpu", dtype=torch.float64, buffer_thickness=0,
        padded_buffer_zone=False, pad_calculation_domain=False,
        mean_bz_policy="uniform_open",
        horizontal_zero_mode_policy="frozen_linear_harmonic",
        gaussian_horizontal_operator="spectral",
    )
    solver.set_bottom_vector_boundary(total[0], total[1], total[2], smooth_passes=0)
    recovered_above = solver.gaussian_above_bottom_field().detach().cpu().numpy()
    recovered_below = solver.B_background[:, 0].detach().cpu().numpy()
    margin = max(2, int(round(0.12 * n)))
    mask = np.zeros((n, n), dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    rows: list[dict[str, float | int | str]] = []
    for component, reference, recovered in (
        ("below_source", below, recovered_below),
        ("above_source", above, recovered_above),
    ):
        rows.append(
            {
                "resolution": n,
                "segments": segments,
                "component": component,
                "domain": "central_76_percent",
                **vector_metrics(reference, recovered, mask),
            }
        )
    closure = recovered_below + recovered_above
    rows.append(
        {
            "resolution": n,
            "segments": segments,
            "component": "sum_closure",
            "domain": "central_76_percent",
            **vector_metrics(total, closure, mask),
        }
    )
    support = {
        "below_loop_z_support": [-1.02, -0.54],
        "above_loop_z_support": [0.60, 1.04],
        "plane_z": 0.0,
        "support_disjoint_from_plane": True,
        "quadrature": "closed circular filament midpoint Biot-Savart",
        "window_extent": extent,
        "spacing": spacing,
        "source_diagnostics": solver.compute_source_diagnostics(),
    }
    return rows, support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "nfgs_paper" / "referee_revision" / "source_support_manufactured",
    )
    parser.add_argument("--resolutions", default="33,65,97")
    parser.add_argument("--segments", type=int, default=2048)
    parser.add_argument("--extent", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolutions = sorted({int(value) for value in args.resolutions.split(",") if value.strip()})
    if any(value < 9 for value in resolutions) or args.segments < 32 or args.extent <= 0.0:
        raise SystemExit("invalid manufactured-source grid/quadrature controls")
    if args.dry_run:
        print(json.dumps({"resolutions": resolutions, "segments": args.segments, "extent": args.extent}, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    supports = []
    for index, n in enumerate(resolutions, 1):
        print(f"manufactured source support: n={n}, segments={args.segments}", flush=True)
        result, support = run_resolution(n, args.segments, args.extent)
        rows.extend(result)
        supports.append({"resolution": n, **support})
        (args.output_dir / "progress.json").write_text(
            json.dumps({"status": "running", "completed": index, "total": len(resolutions)}, indent=2) + "\n"
        )
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "independent compact current source-support attribution",
        "resolutions": resolutions,
        "segments": args.segments,
        "supports": supports,
        "runner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "claim_scope": (
            "Tests finite-window Gaussian attribution against independently "
            "computed fields of sources whose support is strictly above/below z=0."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "progress.json").write_text(
        json.dumps({"status": "complete", "completed": len(resolutions), "total": len(resolutions)}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
