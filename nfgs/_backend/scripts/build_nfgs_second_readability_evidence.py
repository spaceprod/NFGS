#!/usr/bin/env python3
"""Reproduce the second reader revision from frozen fields; no solver iteration."""
from __future__ import annotations

from nfgs.paths import project_root

import csv
import gc
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = project_root()
sys.path.insert(0, str(ROOT))
from nfgs.diagnostics.common_fd2 import curl, sigma_j
from nfgs.diagnostics.topology_metrics import q_map_metrics
from nfgs._backend.scripts.recompute_td_referee_diagnostics import profile_and_budgets

DATA = ROOT / "results/nfgs_paper/referee_revision"
OUT = DATA / "second_reader_revision"
GEN = ROOT / "paper_writing/nfgs/manuscript/generated"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def repair_low_lou() -> list[dict]:
    audit = []
    for precision in ("float32", "float64"):
        base = DATA / "low_lou_family_rho0p5" / precision
        manifest_path = base / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        records = json.loads((base / "metrics.json").read_text())
        with (base / "metrics.csv").open() as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames
            rows = list(reader)
        previous = {
            row["benchmark"]: row for row in manifest.get("anisotropic_fd2_correction", [])
        }
        corrections = []
        for record in records:
            name = record["benchmark"]
            contract = manifest["configuration"]["case_contracts"][name]
            spacing = tuple(contract["spacing_xyz"])
            slices = tuple(slice(*entry) for entry in contract["metric_slices_zyx"])
            path = base / "fields" / f"{name}__nfgs__gs_cfcf.npz"
            with np.load(path) as bundle:
                fields = {key: bundle[key] for key in ("Bx", "By", "Bz")}
            value = sigma_j(fields, spacing, slices)
            correction = {
                "benchmark": name, "precision": precision,
                "original_sigma_J_common_fd2": previous.get(name, {}).get(
                    "original_sigma_J_common_fd2", record["sigma_J_common_fd2"]
                ),
                "corrected_sigma_J_common_fd2": value,
                "spacing_xyz": spacing,
                "field_sha256": sha(path),
                "operator_sha256": sha(ROOT / "nfgs/diagnostics/common_fd2.py"),
                "method": "cache-only FD2 curl with separate dx, dy, dz; no reconstruction rerun",
            }
            record["sigma_J_common_fd2"] = value
            for row in rows:
                if row["benchmark"] == name:
                    row["sigma_J_common_fd2"] = str(value)
            corrections.append(correction)
            audit.append(correction)
            print(precision, name, value, flush=True)
        # Keep the original solver-run fingerprints and native-current metrics.
        manifest["anisotropic_fd2_correction"] = corrections
        json_write(base / "metrics.json", records)
        with (base / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        json_write(manifest_path, manifest)
    return audit


def td_profiles(selected_weight=.5) -> tuple[list[dict], list[dict]]:
    profiles, budgets = [], []
    base = DATA / "rho_full/fields"
    for case in ("td1", "td2", "td3"):
        meta = json.loads((base / f"{case}__case_meta.json").read_text())
        spacing = float(meta["dims"][3])
        slices = tuple(slice(*entry) for entry in meta["metric_slices"])
        for rho in (None, 0.0, selected_weight, 1.0):
            method = "analytic_reference" if rho is None else f"nfgs_rho_{rho:.1f}"
            suffix = "reference" if rho is None else f"nfgs__gs_cfcf__robin_w{rho:.1f}".replace(".", "p")
            path = base / f"{case}__{suffix}.npz"
            print("height profiles", case, method, flush=True)
            with np.load(path) as bundle:
                fields = {key: bundle[key] for key in ("Bx", "By", "Bz")}
            current = curl(fields, spacing)
            p, b = profile_and_budgets(case, method, fields, current, spacing, slices)
            profiles.extend(p)
            budgets.extend(b)
            del fields, current
            gc.collect()
    write_csv(OUT / "height_profiles.csv", profiles)
    write_csv(OUT / "force_free_region_budgets.csv", budgets)
    plt.rcParams.update({"font.size": 13, "text.color": "black", "axes.labelcolor": "black"})
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.1), layout="constrained")
    styles = [
        ("analytic_reference", "Reference", "black"),
        ("nfgs_rho_0.0", r"$\rho=0$", "#2477b5"),
        (f"nfgs_rho_{selected_weight:.1f}", rf"$\rho={selected_weight:g}$", "#803bb2"),
        ("nfgs_rho_1.0", r"$\rho=1$", "#dc792d"),
    ]
    for col, case in enumerate(("td1", "td2", "td3")):
        for method, label, color in styles:
            rows = [r for r in profiles if r["benchmark"] == case and r["method"] == method]
            z = np.array([r["z_Mm"] for r in rows])
            axes[0, col].plot(z, [r["sigma_J_plane"] for r in rows], color=color, label=label)
            cumulative = np.cumsum([r["numerator_fraction"] for r in rows]) * 100
            axes[1, col].plot(z, cumulative, color=color)
        axes[0, col].set(title=case.upper(), yscale="log", ylim=(0.002, 1.1))
        axes[1, col].set(xlabel="Height z (Mm)", ylim=(0, 103))
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
            ax.tick_params(colors="black")
        axes[0, col].legend(fontsize=11, frameon=False)
    axes[0, 0].set_ylabel(r"Plane current-weighted $\sigma_J(z)$")
    axes[1, 0].set_ylabel("Cumulative residual numerator (%)")
    for suffix in ("pdf", "png"):
        fig.savefig(GEN / f"figure_force_free_height_reader.{suffix}", dpi=220)
    plt.close(fig)
    return profiles, budgets


def q_thresholds() -> list[dict]:
    rows = []
    base = DATA / "quantitative_topology"
    for case in ("td1", "td2", "td3"):
        with np.load(base / f"{case}__analytic_reference.npz") as bundle:
            reference = bundle["q"]
        for threshold in (2.0, 2.5, 3.0):
            for rho in (0.0, 0.5, 1.0):
                with np.load(base / f"{case}__nfgs_rho_{rho:.1f}.npz") as bundle:
                    candidate = bundle["q"]
                rows.append({"benchmark": case, "rho": rho, "high_logq": threshold,
                             **q_map_metrics(reference, candidate, high_logq=threshold)})
    write_csv(OUT / "q_threshold_sensitivity.csv", rows)
    return rows


def selection_sensitivity() -> list[dict]:
    path = DATA / "common_diagnostics/metrics_by_domain.csv"
    with path.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["current_source"] == "common_fd2_curl"
                and r["rho"] and r["domain"] == ("protocol" if r["benchmark"] == "td2" else "central_half")]
    assert len(rows) == 33
    values = np.array([[float(r["E_rel_Bgt_domain"]), float(r["sigma_J"])] for r in rows])
    scaled = (values - values.min(axis=0)) / np.ptp(values, axis=0)
    result = []
    for weight in (0.5, 1.0, 2.0):
        distances = np.sqrt(weight * scaled[:, 0] ** 2 + scaled[:, 1] ** 2)
        for rho in sorted(set(float(r["rho"]) for r in rows)):
            mean = float(np.mean([v for r, v in zip(rows, distances) if float(r["rho"]) == rho]))
            result.append({"cci_squared_term_weight": weight, "rho": rho, "mean_distance": mean})
    write_csv(OUT / "selection_sensitivity.csv", result)
    return result


def write_latex_tables(thresholds: list[dict], selection: list[dict]) -> None:
    lines = [
        r"\begin{deluxetable*}{p{0.10\textwidth}>{\centering\arraybackslash}p{0.24\textwidth}>{\centering\arraybackslash}p{0.12\textwidth}>{\centering\arraybackslash}p{0.12\textwidth}>{\centering\arraybackslash}p{0.12\textwidth}}",
        r"\tablewidth{0.8\textwidth}",
        r"\tablecaption{High-\(Q\) threshold sensitivity\label{tab:q_threshold_reader}}",
        r"\tablehead{\colhead{Case}&\colhead{Minimum \(\log_{10}Q\)}&\colhead{\(\rho=0\)}&\colhead{\(\rho=0.5\)}&\colhead{\(\rho=1\)}}",
        r"\startdata",
    ]
    for case in ("td1", "td2", "td3"):
        for threshold in (2.0, 2.5, 3.0):
            values = [next(r["high_q_jaccard"] for r in thresholds
                           if r["benchmark"] == case and r["high_logq"] == threshold and r["rho"] == rho)
                      for rho in (0.0, 0.5, 1.0)]
            lines.append(f"{case.upper()} & {threshold:g} & " + " & ".join(f"{v:.4f}" for v in values) + r"\\")
    lines += [r"\enddata", r"\tablecomments{Entries are high-\(Q\) Jaccard indices from the same cached maps and original pairwise finite-pixel masks. No smoothing or retracing is applied.}", r"\end{deluxetable*}"]
    (GEN / "table_q_threshold_reader.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{deluxetable*}{>{\centering\arraybackslash}p{0.08\textwidth}>{\centering\arraybackslash}p{0.28\textwidth}>{\centering\arraybackslash}p{0.34\textwidth}}",
        r"\tablewidth{0.8\textwidth}",
        r"\tablecaption{Sensitivity to relative CCI weighting in the selection objective\label{tab:selection_sensitivity_reader}}",
        r"\tablehead{\colhead{\(w_E\)}&\colhead{Minimizing \(\rho\)}&\colhead{Minimum \(\overline d_{w_E}\)}}",
        r"\startdata",
    ]
    for w in (0.5, 1.0, 2.0):
        row = min((r for r in selection if r["cci_squared_term_weight"] == w), key=lambda r: r["mean_distance"])
        lines.append(f"{w:g} & {row['rho']:g} & {row['mean_distance']:.4f}" + r"\\")
    lines += [r"\enddata", r"\tablecomments{Only the diagnostic-selection coefficient changes. Shared min/max normalization, TD fields, primary domains, and the equal mean over the three cases remain fixed; \(w_E=1\) reproduces the main-text criterion.}", r"\end{deluxetable*}"]
    (GEN / "table_selection_sensitivity_reader.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    low_lou = repair_low_lou()
    _, budgets = td_profiles()
    thresholds = q_thresholds()
    selection = selection_sensitivity()
    write_latex_tables(thresholds, selection)
    summary = {
        "included_review_items": [1, 2, 4, 5, 6, 7, 8, 9, 10],
        "excluded_review_items": [3],
        "low_lou_correction": low_lou,
        "rho_0p5_region_budgets": [r for r in budgets if r["method"] == "nfgs_rho_0.5"],
        "q_threshold_sensitivity": [{k: r[k] for k in ("benchmark", "rho", "high_logq", "high_q_jaccard", "valid_pixels")} for r in thresholds],
        "selection_minima": [min((r for r in selection if r["cci_squared_term_weight"] == w), key=lambda r: r["mean_distance"]) for w in (0.5, 1.0, 2.0)],
        "primary_domains_unchanged": True,
        "no_solver_iteration_or_fieldline_retracing": True,
        "builder_sha256": sha(Path(__file__)),
    }
    json_write(OUT / "manifest.json", summary)
    print("Evidence written:", OUT, flush=True)


if __name__ == "__main__":
    main()
