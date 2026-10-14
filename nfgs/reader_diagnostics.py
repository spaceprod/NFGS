"""Reproduce the final manuscript's cache-only diagnostics."""

import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    from nfgs._backend.scripts import build_nfgs_second_readability_evidence as evidence

    evidence.DATA = a.output_root.resolve()
    evidence.OUT = evidence.DATA / "second_reader_revision"
    evidence.GEN = evidence.DATA / "figures"
    evidence.OUT.mkdir(parents=True, exist_ok=True)
    evidence.GEN.mkdir(parents=True, exist_ok=True)
    corrected = evidence.repair_low_lou()
    _, budgets = evidence.td_profiles()
    thresholds = evidence.q_thresholds()
    selection = evidence.selection_sensitivity()
    evidence.json_write(
        evidence.OUT / "manifest.json",
        {
            "low_lou_anisotropic_curl": corrected,
            "Q_thresholds": [2, 2.5, 3],
            "selection_squared_CCI_weights": [0.5, 1, 2],
            "selection_minima": [
                min(
                    (r for r in selection if r["cci_squared_term_weight"] == w),
                    key=lambda r: r["mean_distance"],
                )
                for w in (0.5, 1, 2)
            ],
            "no_new_solver_iteration": True,
        },
    )


if __name__ == "__main__":
    main()
