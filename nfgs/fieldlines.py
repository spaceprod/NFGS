"""Render common-seed total-field views from completed reconstruction caches."""

import argparse, json, os
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field-dir", type=Path, required=True)
    p.add_argument("--nfpt-field-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights", default="1,0.5,0.1,0")
    p.add_argument("--tube-radius", type=float, default=3.8)
    a = p.parse_args()
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    from nfgs.diagnostics.common_fd2 import curl

    a.output_dir.mkdir(parents=True, exist_ok=True)
    weights = [float(x) for x in a.weights.split(",")]
    plotter = pv.Plotter(
        shape=(3, len(weights) + 2),
        off_screen=True,
        window_size=(2400, 1200),
        border=False,
    )
    records = []
    for row, case in enumerate(["td1", "td2", "td3"]):
        meta = json.loads((a.field_dir / f"{case}__case_meta.json").read_text())
        dx, dy, dz = map(float, meta["dims"][3:])
        origin = (float(meta["x_bounds"][0]), float(meta["y_bounds"][0]), 0.0)
        with np.load(a.field_dir / f"{case}__reference.npz") as f:
            reference = {k: f[k] for k in ["Bx", "By", "Bz"]}
        jz = curl(reference, (dx, dy, dz))["Jz"][0]
        flat = np.abs(jz).ravel()
        indices = np.argsort(flat)[-100:][::5]
        iy, ix = np.unravel_index(indices, jz.shape)
        seeds = np.column_stack(
            [origin[0] + ix * dx, origin[1] + iy * dy, np.full(len(ix), 0.5 * dz)]
        )
        specs = [("Reference", a.field_dir / f"{case}__reference.npz")] + [
            (
                f"rho={rho:g}",
                a.field_dir
                / f"{case}__nfgs__gs_cfcf__robin_w{rho:.1f}".replace(".", "p"),
            )
            for rho in weights
        ]
        specs = [
            (label, path if str(path).endswith(".npz") else Path(str(path) + ".npz"))
            for label, path in specs
        ]
        specs.append(("NFPT", a.nfpt_field_dir / f"{case}__nfpt_fortran_symmetric.npz"))
        for col, (label, path) in enumerate(specs):
            with np.load(path) as f:
                values = np.stack([f[k] for k in ["Bx", "By", "Bz"]], axis=-1)
            nz, ny, nx = values.shape[:3]
            grid = pv.ImageData(
                dimensions=(nx, ny, nz), spacing=(dx, dy, dz), origin=origin
            )
            grid["B"] = values.reshape(-1, 3)
            lines = grid.streamlines_from_source(
                pv.PolyData(seeds),
                vectors="B",
                integration_direction="both",
                max_steps=3500,
                terminal_speed=1e-12,
                compute_vorticity=False,
            )
            if case == "td1" and label == "NFPT":
                lines.points[:, 0] *= -1
            plotter.subplot(row, col)
            plotter.set_background("white")
            if lines.n_points:
                plotter.add_mesh(
                    lines.tube(radius=a.tube_radius, n_sides=12, capping=True),
                    color="#dc792d",
                    smooth_shading=True,
                    ambient=0.28,
                    diffuse=0.63,
                    specular=0.82,
                    specular_power=44,
                )
            bottom = pv.ImageData(
                dimensions=(nx, ny, 1), spacing=(dx, dy, 1), origin=origin
            )
            bottom["Bz"] = reference["Bz"][0].ravel()
            limit = float(np.percentile(np.abs(reference["Bz"][0]), 99))
            plotter.add_mesh(
                bottom,
                scalars="Bz",
                cmap="gray",
                clim=(-limit, limit),
                show_scalar_bar=False,
                opacity=0.8,
            )
            plotter.add_text(f"{case.upper()} / {label}", font_size=11, color="black")
            plotter.view_isometric()
            plotter.enable_parallel_projection()
            plotter.reset_camera()
            records.append(
                {
                    "case": case,
                    "method": label,
                    "seeds": seeds.tolist(),
                    "source": path.name,
                    "TD1_NFPT_display_x_reflection": case == "td1" and label == "NFPT",
                }
            )
    path = a.output_dir / "fieldline_comparison.png"
    plotter.screenshot(path)
    plotter.close()
    (a.output_dir / "fieldline_manifest.json").write_text(
        json.dumps(
            {
                "shared_reference_derived_seeds": True,
                "projection": "parallel",
                "tube_radius_mm": a.tube_radius,
                "specular": 0.82,
                "specular_power": 44,
                "views": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
