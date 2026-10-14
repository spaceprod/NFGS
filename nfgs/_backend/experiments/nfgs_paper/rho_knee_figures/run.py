#!/usr/bin/env python3
"""Build the paper figures for the practical rho selected on the CCI-error / force-free plane."""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.scripts.plot_nfgs_td_robin_metrics_vs_er import (
    plot_sigmaj_tradeoff_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = ROOT / "paper_writing" / "core_figures" / "frozen_linear_harmonic_vertical_staggered"
    imprint_path = core / "figure_nfgs_td_bgt_robin_w1p0_w0p5_w0p1_w0p0_summary.png"
    imprint = plt.imread(imprint_path)
    height_px, width_px = imprint.shape[:2]
    # Match the source aspect so the imprint grid is not letterboxed in a wide canvas.
    fig_width_in = 7.5
    fig_height_in = fig_width_in * (height_px / width_px)
    figure, axis = plt.subplots(1, 1, figsize=(fig_width_in, fig_height_in))
    axis.imshow(imprint)
    axis.axis("off")
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(
        args.output_dir / "figure_rho_imprint_knee.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)

    raw = read_csv(args.diagnostic_dir / "headline_common_fd2_protocol.csv")
    nfgs = []
    nfpt: dict[str, dict[str, float]] = {}
    keys = ("C_vec", "C_CS", "Eprime_n", "Eprime_m", "epsilon", "sigma_J")
    for row in raw:
        method = row["method"]
        if method.startswith("nfgs_rho_"):
            nfgs.append({
                "benchmark": row["benchmark"], "robin_weight": float(row["rho"]),
                "E_rel_Bgt": float(row["E_rel_Bgt_domain"]),
                **{key: float(row[key]) for key in keys},
            })
        elif method == "nfpt_fortran_symmetric":
            nfpt[row["benchmark"]] = {
                "E_rel_Bgt": float(row["E_rel_Bgt_domain"]),
                **{key: float(row[key]) for key in keys},
            }
    if {row["benchmark"] for row in nfgs} != {"td1", "td2", "td3"} or set(nfpt) != {"td1", "td2", "td3"}:
        raise RuntimeError("incomplete common-protocol NFGS or NFPT metric rows")
    plot_sigmaj_tradeoff_pair(
        nfgs,
        args.output_dir / "figure_nfgs_td_rho_metrics_with_nfpt.png",
    )


if __name__ == "__main__":
    main()
