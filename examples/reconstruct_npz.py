#!/usr/bin/env python3
"""Reconstruct a vector photospheric boundary stored as Bx/By/Bz, dx/dy/dz and Nz."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from nfgs_solver import NFGSSolverGPU

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("boundary", type=Path)
p.add_argument("--output", type=Path, default=Path("reconstruction.npz"))
p.add_argument("--rho", type=float, default=0.5)
p.add_argument("--device", default="auto")
p.add_argument("--max-iter", type=int, default=3000)
a = p.parse_args()
if not 0 <= a.rho <= 1:
    p.error("--rho must be in [0,1]")
with np.load(a.boundary) as d:
    bx, by, bz = (np.asarray(d[k]) for k in ["Bx", "By", "Bz"])
    ny, nx = bz.shape
    nz = int(d["Nz"])
    dx, dy, dz = (float(d[k]) for k in ["dx", "dy", "dz"])
device = (
    ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
)
solver = NFGSSolverGPU(
    nx,
    ny,
    nz,
    dx,
    dy,
    dz,
    device=device,
    dtype=torch.float32,
    calculation_domain_mode="mudpack",
    bottom_divergence_robin_weight=a.rho,
    horizontal_zero_mode_policy="frozen_linear_harmonic",
    vertical_current_operator="staggered_face",
)
solver.set_bottom_vector_boundary(
    bx, by, bz, zero_net_current=False, smooth_passes=0, ijzbcor=3
)
with torch.inference_mode():
    history = solver.iterate(
        init_mode="gs_cfcf",
        max_iter=a.max_iter,
        tol=1e-5,
        mu=1,
        patience=40,
        sigma_improvement_tol=1e-5,
        stop_mode="sigma_best",
        verbose=True,
    )
a.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.output, **solver.fields_to_numpy(crop_to_observed_fov=True))
a.output.with_suffix(".json").write_text(
    json.dumps(
        {
            "rho": a.rho,
            "best_iter": history["best_iter"],
            "best_sigma": history["best_sigma"],
            "stop_reason": history["stop_reason"],
        },
        indent=2,
    )
    + "\n"
)
