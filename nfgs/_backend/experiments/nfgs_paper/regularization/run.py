#!/usr/bin/env python3
"""Run alpha-clipping/weak-field sensitivity with explicit activation rates."""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "nfgs_paper" / "referee_revision" / "regularization",
    )
    parser.add_argument("--alpha-scales", default="0.5,1,2")
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--include-low-lou", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def tag(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def activation_summary(record: dict[str, Any]) -> dict[str, Any]:
    history = record.get("history", {})
    clip = [float(value) for value in history.get("alpha_clip_fraction", [])]
    weak = [float(value) for value in history.get("weak_field_floor_fraction", [])]
    source = record.get("source_diagnostics", {})
    return {
        "benchmark": record.get("benchmark"),
        "alpha_clip_scale": source.get("alpha_clip_scale"),
        "weak_field_floor_B2": source.get("weak_field_floor_B2"),
        "stop_mode": record.get("stop_mode"),
        "stop_reason": record.get("stop_reason"),
        "selected_state": record.get("selected_state")
        or record.get("history", {}).get("selected_state"),
        "n_iters": record.get("n_iters"),
        "C_vec": record.get("C_vec"),
        "epsilon": record.get("epsilon"),
        "sigma_J": record.get("sigma_J"),
        "sigma_J_common_fd2": record.get("sigma_J_common_fd2"),
        "alpha_clip_fraction_final": clip[-1] if clip else None,
        "alpha_clip_fraction_max": max(clip) if clip else None,
        "weak_field_floor_fraction_final": weak[-1] if weak else None,
        "weak_field_floor_fraction_max": max(weak) if weak else None,
        "B_harm_over_Bobs_rms": source.get("B_harm_over_Bobs_rms"),
        "Jz_projected_out_over_Jz_rms": source.get(
            "Jz_projected_out_over_Jz_rms"
        ),
        "bottom_horizontal_reconstruction_ratio": source.get(
            "bottom_horizontal_reconstruction_ratio"
        ),
    }


def main() -> int:
    args = parse_args()
    scales = sorted({float(item) for item in args.alpha_scales.split(",") if item.strip()})
    if not scales or any(value <= 0.0 for value in scales):
        raise SystemExit("--alpha-scales must contain positive values")
    jobs: list[dict[str, Any]] = []
    for scale in scales:
        output = args.output_dir / f"alpha_{tag(scale)}" / "td"
        command = [
            sys.executable, "nfgs/_backend/scripts/run_nfgs_td_robin_weight_sweep.py",
            "--output-dir", str(output), "--benchmarks", "td1,td2,td3",
            "--weights", str(args.rho), "--device", args.device,
            "--dtype", args.dtype, "--max-iter", str(args.max_iter),
            "--stop-mode", "sigma_best", "--tol", "1e-5",
            "--patience", "40", "--sigma-improvement-tol", "1e-5",
            "--record-iteration-diagnostics",
            "--alpha-clip-scale", str(scale),
            "--weak-field-floor-b2", "1e-20",
        ]
        if args.force:
            command.append("--force")
        jobs.append({"family": "td", "alpha_scale": scale, "output": output, "command": command})
        if args.include_low_lou:
            output = args.output_dir / f"alpha_{tag(scale)}" / "low_lou"
            command = [
                sys.executable, "nfgs/_backend/experiments/nfgs_paper/analytic_nlfff/run.py",
                "--output-dir", str(output), "--field-cache-dir", str(output / "fields"),
                "--benchmarks", "low_lou_c2,low_lou_ll1,low_lou_ll2",
                "--mixed-robin-weight", str(args.rho), "--device", args.device,
                "--dtype", args.dtype, "--max-iter-lnl", str(args.max_iter),
                "--stop-mode", "sigma_best", "--tol", "1e-5",
                "--patience", "40", "--sigma-improvement-tol", "1e-5",
                "--record-iteration-diagnostics",
                "--alpha-clip-scale", str(scale),
                "--weak-field-floor-b2", "1e-20",
            ]
            if args.force:
                command.append("--force")
            jobs.append({"family": "low_lou", "alpha_scale": scale, "output": output, "command": command})

    if args.dry_run:
        for job in jobs:
            print("$ " + " ".join(str(value) for value in job["command"]))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, 1):
        subprocess.run(job["command"], cwd=ROOT, check=True)
        if job["family"] == "td":
            path = Path(job["output"]) / f"robin_w{tag(args.rho)}.json"
        else:
            path = Path(job["output"]) / "metrics.json"
        records = json.loads(path.read_text())
        for record in records:
            row = activation_summary(record)
            row.update({"family": job["family"], "alpha_scale_requested": job["alpha_scale"]})
            rows.append(row)
        (args.output_dir / "progress.json").write_text(
            json.dumps({"status": "running", "completed_jobs": index, "total_jobs": len(jobs)}, indent=2) + "\n"
        )

    fields = list(rows[0]) if rows else []
    with (args.output_dir / "regularization_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "rho": args.rho,
        "alpha_scales": scales,
        "weak_field_floor_B2": 1.0e-20,
        "stopping": {
            "mode": "sigma_best",
            "sigma_tol": 1.0e-5,
            "patience": 40,
            "sigma_improvement_tol": 1.0e-5,
            "selected_state": "best_sigma",
        },
        "jobs": [{**job, "output": str(job["output"])} for job in jobs],
        "interpretation": (
            "Activation fractions quantify regularization use; alpha sensitivity "
            "must not be interpreted as an isolated source-separation effect."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (args.output_dir / "progress.json").write_text(
        json.dumps({"status": "complete", "completed_jobs": len(jobs), "total_jobs": len(jobs)}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
