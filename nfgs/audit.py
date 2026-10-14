"""Check that all current manuscript reconstruction rows and diagnostics exist."""

import argparse, csv, json, math
from pathlib import Path
from nfgs.workflow import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    root = a.output_root
    failures = []
    count = 0
    matrix = set()
    for path in sorted((root / "rho_full").glob("robin_w*.json")):
        records = json.loads(path.read_text())
        records = records if isinstance(records, list) else [records]
        for record in records:
            if "result" in record:
                record = record["result"]
            case = record["benchmark"]
            rho = float(record.get("robin_weight", record.get("mixed_robin_weight")))
            matrix.add((case, round(rho, 1)))
            count += 1
            if record.get("stop_reason") not in [
                "tol",
                "sigma_improvement_tol",
                "sigma_stall",
            ]:
                failures.append(f"{case} rho={rho}: inadmissible stopping reason")
            if record.get("stop_mode") != "sigma_best":
                failures.append(f"{case} rho={rho}: incorrect monitor selection")
            if (
                record.get(
                    "selected_state", record.get("history", {}).get("selected_state")
                )
                != "best_sigma"
            ):
                failures.append(f"{case} rho={rho}: final state is not best_sigma")
            for key in [
                "C_vec",
                "C_CS",
                "Eprime_n",
                "Eprime_m",
                "epsilon",
                "sigma_J_common_fd2",
            ]:
                if not math.isfinite(float(record[key])):
                    failures.append(f"{case} rho={rho}: nonfinite {key}")
    expected = {(case, i / 10) for case in ["td1", "td2", "td3"] for i in range(11)}
    if matrix != expected:
        failures.append("Incomplete 3-case x 11-weight matrix")
    for precision in ["float32", "float64"]:
        path = root / "low_lou_family_rho0p5" / precision / "metrics.json"
        if not path.is_file():
            failures.append("Missing Low--Lou " + precision)
            continue
        records = json.loads(path.read_text())
        if {r["benchmark"] for r in records} != {
            "low_lou_c2",
            "low_lou_ll1",
            "low_lou_ll2",
        }:
            failures.append("Incomplete Low--Lou family " + precision)
        for r in records:
            if float(r["mixed_robin_weight"]) != 0.5:
                failures.append("Low--Lou did not use rho=0.5")
            if r["stop_reason"] not in ["tol", "sigma_improvement_tol", "sigma_stall"]:
                failures.append("Low--Lou iteration cap reached")
    for relative in [
        "common_diagnostics/metrics_by_domain.csv",
        "quantitative_topology/metrics.csv",
        "twist_apex/metrics.csv",
        "six_face_helicity/metrics.csv",
        "second_reader_revision/selection_sensitivity.csv",
    ]:
        if not (root / relative).is_file():
            failures.append("Missing " + relative)
    report = {
        "status": "passed" if not failures else "failed",
        "TD_rows": count,
        "expected_TD_rows": 33,
        "low_lou_rho": 0.5,
        "failures": failures,
        "criteria": "Completeness and configuration checks; does not prove mathematical fixed-point convergence.",
    }
    atomic_json(root / "reproduction_audit.json", report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
