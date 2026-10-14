#!/usr/bin/env python3
"""Run TD rho/gamma scaling anchors on fixed physical domains."""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs._backend.experiments.nfgs_paper.td_resolution_sweep import run as resolution  # noqa: E402


def items(text: str) -> list[float]:
    return sorted({float(value.strip()) for value in text.split(",") if value.strip()})


def tag(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def rho_from_reference(name: str, ratio: float, rho_ref: float) -> tuple[float, float | None]:
    if rho_ref <= 0.0:
        return 0.0, 0.0
    if rho_ref >= 1.0:
        return 1.0, None
    _xr, _yr, z_ref = resolution.reference_axes(name)
    _xt, _yt, z_target = resolution.target_axes(name, ratio)
    dz_ref = float(z_ref[1] - z_ref[0])
    dz_target = float(z_target[1] - z_target[0])
    gamma = rho_ref / ((1.0 - rho_ref) * dz_ref)
    rho = gamma * dz_target / (1.0 + gamma * dz_target)
    return float(rho), float(gamma)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "nfgs_paper" / "referee_revision" / "rho_gamma_scaling",
    )
    parser.add_argument("--benchmarks", default="td1,td2,td3")
    parser.add_argument("--ratios", default="0.5,0.75,1.0")
    parser.add_argument("--rho-reference-anchors", default="0,0.2,0.4,0.7,1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmarks = tuple(dict.fromkeys(x.strip().lower() for x in args.benchmarks.split(",") if x.strip()))
    ratios = items(args.ratios)
    anchors = items(args.rho_reference_anchors)
    if set(benchmarks) - {"td1", "td2", "td3"}:
        raise SystemExit("--benchmarks must be a subset of td1,td2,td3")
    if any(not 0.0 <= value <= 1.0 for value in anchors):
        raise SystemExit("rho reference anchors must be in [0,1]")

    jobs = []
    for rho_ref in anchors:
        for name in benchmarks:
            for ratio in ratios:
                rho, gamma = rho_from_reference(name, ratio, rho_ref)
                out = args.output_dir / f"rho_ref_{tag(rho_ref)}" / name / f"ratio_{tag(ratio)}"
                cmd = [
                    sys.executable,
                    "nfgs/_backend/experiments/nfgs_paper/td_resolution_sweep/run.py",
                    "--output-dir", str(out),
                    "--benchmarks", name,
                    "--ratios", str(ratio),
                    "--dtypes", args.dtype,
                    "--device", args.device,
                    "--rho", repr(rho),
                    "--max-iter", str(args.max_iter),
                    "--tol", "1e-5",
                    "--patience", "40",
                    "--sigma-improvement-tol", "1e-5",
                    "--stop-mode", "sigma_best",
                    "--horizontal-zero-mode-policy", "frozen_linear_harmonic",
                    "--td-calculation-domain-mode", "mudpack",
                ]
                if args.force:
                    cmd.append("--force")
                jobs.append({
                    "benchmark": name, "ratio": ratio, "rho_reference": rho_ref,
                    "rho_realized": rho, "gamma": gamma, "output_dir": str(out),
                    "command": cmd,
                })

    if args.dry_run:
        for job in jobs:
            print("$ " + " ".join(job["command"]))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, job in enumerate(jobs, 1):
        subprocess.run(job["command"], cwd=ROOT, check=True)
        metrics_path = Path(job["output_dir"]) / "metrics.csv"
        with metrics_path.open(newline="") as stream:
            metric_rows = list(csv.DictReader(stream))
        if len(metric_rows) != 1:
            raise RuntimeError(f"{metrics_path}: expected one row, got {len(metric_rows)}")
        row = dict(metric_rows[0])
        row.update({key: value for key, value in job.items() if key != "command"})
        rows.append(row)
        progress = {"status": "running", "completed_runs": index, "total_runs": len(jobs)}
        (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")

    fields = list(rows[0]) if rows else []
    with (args.output_dir / "metrics_all.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "scaling": "rho(dz)=gamma*dz/(1+gamma*dz)",
        "stopping": {
            "mode": "sigma_best",
            "sigma_tol": 1.0e-5,
            "patience": 40,
            "sigma_improvement_tol": 1.0e-5,
            "selected_state": "best_sigma",
        },
        "rho_reference_anchors": anchors,
        "ratios": ratios,
        "jobs": [{**job, "command": job["command"]} for job in jobs],
        "runner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "progress.json").write_text(json.dumps({"status": "complete", "completed_runs": len(jobs), "total_runs": len(jobs)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
