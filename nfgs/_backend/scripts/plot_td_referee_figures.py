#!/usr/bin/env python3
"""Create manuscript figures from the cache-only TD referee diagnostics."""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_td_referee_diagnostics_frozen_linear_harmonic_vertical_staggered"
        ),
    )
    parser.add_argument(
        "--source-metrics",
        type=Path,
        default=(
            ROOT
            / "results"
            / "nfgs_td_robin_weight_sweep_frozen_linear_harmonic_vertical_staggered"
            / "robin_w0p0.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "paper_writing"
            / "core_figures"
            / "frozen_linear_harmonic_vertical_staggered"
        ),
    )
    parser.add_argument(
        "--allow-legacy-k0",
        action="store_true",
        help=(
            "Allow a diagnostic-only plot from pre-staggered coupled-k0 caches. "
            "Without this explicit flag, legacy checkerboard-prone profiles are rejected."
        ),
    )
    parser.add_argument(
        "--horizontal-zero-mode-policy",
        choices=(
            "coupled_k0",
            "omit",
            "frozen_linear_harmonic",
            "constrained_spline_harmonic",
        ),
        default="frozen_linear_harmonic",
        help="Expected source-cache horizontal k=0 protocol.",
    )
    parser.add_argument(
        "--tradeoff-only",
        action="store_true",
        help="Create only figure_nfgs_td_imprint_tradeoff.png.",
    )
    parser.add_argument(
        "--allow-legacy-vertical-current",
        action="store_true",
        help="Allow diagnostic-only figures from collocated-current caches.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_tradeoff(rows: list[dict[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(11.2, 6.0), sharex="col", layout="constrained"
    )
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)
    for column, case in enumerate(("td1", "td2", "td3")):
        group = [
            row
            for row in rows
            if row["benchmark"] == case
            and row["domain"] == "protocol"
            and row["current_source"] == "common_fd2_curl"
            and row["method"].startswith("nfgs_rho_")
        ]
        group.sort(key=lambda row: float(row["rho"]))
        x = np.asarray([float(row["E_rel_Bgt_domain"]) for row in group])
        rho = np.asarray([float(row["rho"]) for row in group])
        for row_index, metric in enumerate(("C_vec", "sigma_J")):
            ax = axes[row_index, column]
            y = np.asarray([float(row[metric]) for row in group])
            ax.plot(x, y, color="0.55", linewidth=1.0, zorder=1)
            ax.scatter(
                x,
                y,
                c=rho,
                cmap=cmap,
                norm=norm,
                s=42,
                edgecolors="k",
                linewidths=0.35,
                zorder=2,
            )
            # Corrected diagnostics use the untouched archived-original
            # method name; historical development files used the symmetric
            # cache name.  Prefer the strict original control if both exist.
            nfpt_candidates = [
                row
                for row in rows
                if row["benchmark"] == case
                and row["method"] in {"nfpt_fortran", "nfpt_fortran_symmetric"}
                and row["domain"] == "protocol"
                and row["current_source"] == "common_fd2_curl"
            ]
            nfpt = next(
                (
                    row
                    for row in nfpt_candidates
                    if row["method"] == "nfpt_fortran"
                ),
                None,
            )
            if nfpt is None:
                nfpt = next(iter(nfpt_candidates), None)
            if nfpt is None:
                raise RuntimeError(
                    f"{case}: trade-off plot requires one common-FD2 "
                    "original-NFPT control row."
                )
            ax.scatter(
                [float(nfpt["E_rel_Bgt_domain"])],
                [float(nfpt[metric])],
                marker="x",
                s=75,
                linewidths=1.5,
                color="#00a6d6",
                label="NFPT control" if column == 0 and row_index == 0 else None,
                zorder=3,
            )
            # Offsets are for raw C_vec (top) and sigma_J (bottom).  Do not
            # transform C_vec to 1-C_vec: manuscript tables and captions use
            # the Schrijver correlation itself.
            annotation_offsets = {
                "td1": {
                    "C_vec": {0.0: (-45, -12), 0.4: (4, -12), 1.0: (4, 5)},
                    "sigma_J": {0.0: (-45, 5), 0.4: (4, 5), 1.0: (4, -11)},
                },
                "td2": {
                    "C_vec": {0.0: (-45, -12), 0.4: (4, 5), 1.0: (4, -11)},
                    "sigma_J": {0.0: (-45, 5), 0.4: (4, 5), 1.0: (4, -11)},
                },
                # TD3's rho=0 and rho=0.4 points are close in both rows.
                # Keep their labels on opposite sides of the local curve.
                "td3": {
                    "C_vec": {0.0: (5, 7), 0.4: (5, -12), 1.0: (5, 7)},
                    "sigma_J": {0.0: (5, -12), 0.4: (5, 7), 1.0: (5, -12)},
                },
            }
            for target in (0.0, 0.4, 1.0):
                offset = annotation_offsets[case][metric][target]
                index = int(np.argmin(np.abs(rho - target)))
                ax.annotate(
                    rf"$\rho={rho[index]:.1f}$",
                    (x[index], y[index]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7,
                )
            ax.grid(alpha=0.25, linewidth=0.6)
            if row_index == 0:
                ax.set_title(case.upper())
        axes[1, column].set_xlabel(r"evaluation-domain $E^{\rm rel}_{B^{>}}$")
    axes[0, 0].set_ylabel(r"evaluation-domain $C_{\rm vec}$")
    axes[1, 0].set_ylabel(r"evaluation-domain $\sigma_J$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, fontsize=8, loc="lower left")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.025,
    )
    colorbar.set_label(r"mixed-row weight $\rho$")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _record_horizontal_zero_mode_policy(record: dict) -> str:
    coupled = record.get("source_diagnostics", {}).get("coupled_k0", {})
    return str(
        record.get(
            "horizontal_zero_mode_policy",
            "coupled_k0" if coupled else "legacy_unspecified",
        )
    )


def validate_k0_records(
    source_metrics: Path,
    *,
    horizontal_zero_mode_policy: str | None = None,
    allow_legacy_k0: bool = False,
    allow_legacy_vertical_current: bool = False,
) -> list[dict]:
    records = json.loads(source_metrics.read_text())
    expected_policy = horizontal_zero_mode_policy
    for record in records:
        case = str(record["benchmark"])
        coupled = record.get("source_diagnostics", {}).get("coupled_k0", {})
        # Older coupled-k0 records predate the explicit top-level provenance.
        # Preserve their existing validation path rather than silently treating
        # them as omit records.
        policy = _record_horizontal_zero_mode_policy(record)
        if expected_policy is None:
            expected_policy = policy
        if policy != expected_policy:
            raise RuntimeError(
                f"{case}: expected horizontal zero-mode policy "
                f"{expected_policy!r}, found {policy!r}."
            )
        corrected = coupled.get("discretization") == "staggered_face_D1_true_three_point_curvature"
        if (
            expected_policy == "coupled_k0"
            and not corrected
            and not allow_legacy_k0
        ):
            raise RuntimeError(
                f"{case}: refusing publication plot from legacy coupled-k0 cache. "
                "Rerun NFGS with the staggered k0 solver, or pass "
                "--allow-legacy-k0 for a diagnostic-only figure."
            )
        vertical = str(record.get("vertical_current_operator", "legacy_unspecified"))
        if vertical != "staggered_face" and not allow_legacy_vertical_current:
            raise RuntimeError(
                f"{case}: refusing publication plot from vertical-current "
                f"operator {vertical!r}; rerun with staggered_face."
            )
    return records


def plot_zero_mode(
    source_metrics: Path,
    path: Path,
    *,
    horizontal_zero_mode_policy: str | None = None,
    allow_legacy_k0: bool = False,
    allow_legacy_vertical_current: bool = False,
) -> None:
    records = validate_k0_records(
        source_metrics,
        horizontal_zero_mode_policy=horizontal_zero_mode_policy,
        allow_legacy_k0=allow_legacy_k0,
        allow_legacy_vertical_current=allow_legacy_vertical_current,
    )
    resolved_policy = horizontal_zero_mode_policy
    if resolved_policy is None:
        if not records:
            raise RuntimeError("Zero-mode plot requires at least one source record")
        resolved_policy = _record_horizontal_zero_mode_policy(records[0])
    fig, axes = plt.subplots(
        2, 3, figsize=(10.8, 5.8), sharex="col", layout="constrained"
    )
    for column, record in enumerate(records):
        case = str(record["benchmark"])
        diagnostics = record.get("source_diagnostics", {})
        if resolved_policy == "omit":
            faces = int(
                record.get("source_diagnostics", {})
                .get("vertical_staggered_face", {})
                .get("face_planes", 1)
            )
            hx = np.zeros(max(faces + 1, 2), dtype=float)
            hy = np.zeros_like(hx)
            corrected = True
        else:
            closure = diagnostics.get(resolved_policy, {})
            profile = closure.get("profile_by_height")
            if profile is None:
                raise RuntimeError(
                    f"{case}: missing {resolved_policy} height profile"
                )
            hx = np.asarray(profile[0], dtype=float)
            hy = np.asarray(profile[1], dtype=float)
            corrected = (
                closure.get("discretization")
                == "staggered_face_D1_true_three_point_curvature"
                if resolved_policy == "coupled_k0"
                else True
            )
        spacing = 360.0 / max(hx.size - 1, 1)
        z = np.arange(hx.size, dtype=float) * spacing
        z_face = 0.5 * (z[:-1] + z[1:])
        # The corrected ODE is constrained on cell faces.  Plot the same
        # face derivative so a one-cell alternating mode cannot disappear in
        # a centred-gradient null space.
        jm = np.hypot(
            np.diff(hy) / spacing,
            np.diff(hx) / spacing,
        )
        axes[0, column].plot(z, hx, label=r"$H_x$", linewidth=1.15)
        axes[0, column].plot(z, hy, label=r"$H_y$", linewidth=1.15)
        axes[1, column].plot(z_face, jm, color="#d95f02", linewidth=1.15)
        axes[0, column].set_title(case.upper())
        if resolved_policy == "omit":
            axes[0, column].text(
                0.02, 0.04, r"explicit $k_h=0$ omitted",
                transform=axes[0, column].transAxes, fontsize=6.5, color="#555555",
            )
        elif resolved_policy == "frozen_linear_harmonic":
            axes[0, column].text(
                0.02, 0.04, r"frozen lift: $H'=0$, $J_M=0$",
                transform=axes[0, column].transAxes, fontsize=6.5, color="#555555",
            )
        if not corrected:
            axes[0, column].text(
                0.02,
                0.04,
                "legacy centered-$D_z$ cache",
                transform=axes[0, column].transAxes,
                fontsize=6.5,
                color="#b2182b",
            )
        axes[1, column].set_xlabel("Height [Mm]")
        for ax in axes[:, column]:
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0, 0].set_ylabel(r"mean field $H$ [model units]")
    axes[1, 0].set_ylabel(r"face $|\mathbf{J}_M|$ [model units Mm$^{-1}$]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    # Validate the nonlinear source generation before writing even the first
    # paper figure, preventing a partial legacy output directory.
    validate_k0_records(
        args.source_metrics,
        horizontal_zero_mode_policy=args.horizontal_zero_mode_policy,
        allow_legacy_k0=args.allow_legacy_k0,
        allow_legacy_vertical_current=args.allow_legacy_vertical_current,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.diagnostic_dir / "metrics_by_domain.csv")
    plot_tradeoff(rows, args.output_dir / "figure_nfgs_td_imprint_tradeoff.png")
    if args.tradeoff_only:
        return
    plot_zero_mode(
        args.source_metrics,
        args.output_dir / "figure_nfgs_td_zero_mode_profiles.png",
        horizontal_zero_mode_policy=args.horizontal_zero_mode_policy,
        allow_legacy_k0=args.allow_legacy_k0,
        allow_legacy_vertical_current=args.allow_legacy_vertical_current,
    )


if __name__ == "__main__":
    main()
