"""Load and visualize one fresh or archived experiment campaign."""

from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

TABLES = {
    "td": "common_diagnostics/metrics_by_domain.csv",
    "runtime": "rho_full/metrics_all.csv",
    "low_lou_fp32": "low_lou_family_rho0p5/float32/metrics.csv",
    "low_lou_fp64": "low_lou_family_rho0p5/float64/metrics.csv",
    "height": "second_reader_revision/height_profiles.csv",
    "budgets": "second_reader_revision/force_free_region_budgets.csv",
    "topology": "quantitative_topology/metrics.csv",
    "twist": "twist_apex/metrics.csv",
    "helicity": "six_face_helicity/metrics.csv",
    "selection": "second_reader_revision/selection_sensitivity.csv",
    "Q_threshold": "second_reader_revision/q_threshold_sensitivity.csv",
    "regularization": "regularization/regularization_metrics.csv",
    "factorized": "factorized_ablation/metrics_all.csv",
    "resolution": "rho_gamma_scaling/metrics_all.csv",
}


def load_tables(root):
    root = Path(root)
    tables = {}
    missing = []
    for key, name in TABLES.items():
        path = root / name
        if path.is_file():
            tables[key] = pd.read_csv(path)
        else:
            missing.append(name)
    return tables, missing


def primary_td(frame):
    keep = (frame["current_source"] == "common_fd2_curl") & frame["rho"].notna()
    keep &= frame["domain"] == "protocol"
    return frame.loc[keep].sort_values(["benchmark", "rho"]).copy()


def plot_tradeoff(frame, selected_rho=0.5):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout="constrained")
    for ax, case in zip(axes, ["td1", "td2", "td3"]):
        group = frame[frame.benchmark == case].sort_values("rho")
        ax.plot(group.E_rel_Bgt_domain, group.sigma_J, color="0.65", zorder=1)
        points = ax.scatter(
            group.E_rel_Bgt_domain,
            group.sigma_J,
            c=group.rho,
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=65,
        )
        selected = group[np.isclose(group.rho, selected_rho)]
        ax.scatter(
            selected.E_rel_Bgt_domain,
            selected.sigma_J,
            s=140,
            facecolors="none",
            edgecolors="black",
            linewidth=2,
        )
        ax.set(
            title=case.upper(),
            xlabel="Relative CCI error",
            ylabel=r"Common-curl $\sigma_J$",
        )
        ax.grid(alpha=0.2)
    fig.colorbar(points, ax=axes, label=r"$\rho$")
    return fig


def plot_field_metrics(frame):
    metrics = ["C_vec", "C_CS", "Eprime_n", "Eprime_m", "epsilon"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, key in zip(axes.flat, metrics):
        for case, group in frame.groupby("benchmark"):
            ax.plot(group.rho, group[key], "-o", label=case.upper(), markersize=4)
        ax.set(title=key, xlabel=r"$\rho$")
        ax.grid(alpha=0.2)
        if key == "epsilon":
            ax.axhline(1, color="black", linestyle=":", label="Reference energy")
        ax.legend()
    axes.flat[-1].axis("off")
    return fig


def plot_runtime(frame):
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for case, group in frame.groupby("benchmark"):
        ax.plot(group.robin_weight, group.wall_time_s, "-o", label=case.upper())
    ax.set(
        xlabel=r"$\rho$",
        ylabel="Timed solver interval (s)",
        title="Absolute runtime; hardware and interval depend on the run",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def plot_grouped(frame, columns, title):
    columns = [
        c for c in columns if c in frame and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not columns:
        print("No matching numerical columns:", title)
        return None
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(max(6, 5 * len(columns)), 4.5),
        squeeze=False,
        layout="constrained",
    )
    case_col = "benchmark" if "benchmark" in frame else "case"
    method_col = "method" if "method" in frame else None
    for ax, key in zip(axes.flat, columns):
        data = frame.copy()
        data["label"] = data[case_col].astype(str) + (
            ("\n" + data[method_col].astype(str)) if method_col else ""
        )
        ax.bar(np.arange(len(data)), data[key])
        ax.set_xticks(np.arange(len(data)), data.label, rotation=70, fontsize=8)
        ax.set(title=key)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    return fig


def plot_height(frame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, case in zip(axes, ["td1", "td2", "td3"]):
        for method, group in frame[frame.benchmark == case].groupby("method"):
            ax.semilogy(
                group.z_Mm,
                group.sigma_J_plane,
                label=method.replace("nfgs_rho_", r"$\rho=$"),
            )
        ax.set(title=case.upper(), xlabel="Height z (Mm)", ylabel=r"Plane $\sigma_J$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    return fig


def plot_selection(frame):
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for coefficient, group in frame.groupby("cci_squared_term_weight"):
        group = group.sort_values("rho")
        ax.plot(
            group.rho,
            group.mean_distance,
            "-o",
            label=f"CCI squared-term weight = {coefficient}",
        )
        best = group.loc[group.mean_distance.idxmin()]
        ax.scatter(
            [best.rho],
            [best.mean_distance],
            s=140,
            facecolors="none",
            edgecolors="black",
        )
    ax.set(
        xlabel=r"$\rho$",
        ylabel="Mean normalized distance",
        title="Diagnostic selection sensitivity",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    return fig
