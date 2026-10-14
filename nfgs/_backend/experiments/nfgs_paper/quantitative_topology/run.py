#!/usr/bin/env python3
"""Common-protocol FastQSL Q and conjugate-endpoint metrics for TD1--TD3."""

from __future__ import annotations

from nfgs.paths import project_root

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nfgs.diagnostics.topology_metrics import fastqsl_bundle_metrics  # noqa: E402
from nfgs._backend.util.fastqsl_plane import compute_bottom_qmap, compute_y0_qmap  # noqa: E402


def weight_tag(weight: float) -> str:
    return f"{weight:.1f}".replace(".", "p")


def load_field(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        if "B" in data:
            b = np.asarray(data["B"])
            return {key: b[index] for index, key in enumerate(("Bx", "By", "Bz"))}
        return {key: np.asarray(data[key]) for key in ("Bx", "By", "Bz")}


def save_bundle(path: Path, bundle: dict[str, Any], metadata: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        q=bundle["q"], flag_start=bundle["flag_start"], flag_end=bundle["flag_end"],
        start_xyz=bundle["start_xyz"], end_xyz=bundle["end_xyz"],
        line_len=bundle["line_len"],
        metadata=np.asarray(json.dumps({**metadata, **bundle["metadata"]}, indent=2)),
    )


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            key: np.asarray(data[key])
            for key in (
                "q",
                "flag_start",
                "flag_end",
                "start_xyz",
                "end_xyz",
                "line_len",
            )
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument(
        "--nfpt-field-dir", type=Path,
        default=ROOT / "results" / "nfgs_td_robin_weight_sweep" / "fields",
    )
    parser.add_argument(
        "--nfpt-variant", default="nfpt_fortran_symmetric",
        help="NFPT field variant; the default enforces mirror-symmetric x/y padding.",
    )
    parser.add_argument(
        "--allow-legacy-nfpt-padding", action="store_true",
        help="Allow the archived asymmetric nfpt_fortran field for a diagnostic-only run.",
    )
    parser.add_argument("--weights", default="0,0.4")
    parser.add_argument("--benchmarks", default="td1,td2,td3")
    parser.add_argument("--high-logq", type=float, default=3.0)
    parser.add_argument(
        "--plane", choices=("bottom", "y0"), default="bottom",
        help="Common seed plane; bottom gives photospheric conjugate-footpoint errors.",
    )
    parser.add_argument("--step-len", type=float, default=0.125)
    parser.add_argument("--tol", type=float, default=0.1)
    parser.add_argument("--block-x", type=int, default=128)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "nfgs_paper" / "referee_revision" / "quantitative_topology",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help=(
            "Recompute metrics from saved FastQSL NPZ bundles without launching "
            "CUDA field-line tracing."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing FastQSL bundle instead of tracing that field again.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.nfpt_variant != "nfpt_fortran_symmetric" and not args.allow_legacy_nfpt_padding:
        raise SystemExit(
            "Paper topology requires nfpt_fortran_symmetric so TD3 uses "
            "mirror-symmetric x/y padding.  Pass --allow-legacy-nfpt-padding "
            "only for an explicitly legacy diagnostic."
        )
    cases = tuple(dict.fromkeys(item.strip().lower() for item in args.benchmarks.split(",") if item.strip()))
    weights = sorted({float(item) for item in args.weights.split(",") if item.strip()})
    if set(cases) - {"td1", "td2", "td3"}:
        raise SystemExit("--benchmarks must be a subset of td1,td2,td3")
    specs: list[tuple[str, str, Path]] = []
    for case in cases:
        specs.append((case, "analytic_reference", args.field_dir / f"{case}__reference.npz"))
        specs.append((case, args.nfpt_variant, args.nfpt_field_dir / f"{case}__{args.nfpt_variant}.npz"))
        specs.extend(
            (
                case,
                f"nfgs_rho_{rho:.1f}",
                args.field_dir
                / f"{case}__nfgs__gs_cfcf__robin_w{weight_tag(rho)}.npz",
            )
            for rho in weights
        )
    missing = (
        [
            str(args.output_dir / f"{case}__{method}.npz")
            for case, method, _path in specs
            if not (args.output_dir / f"{case}__{method}.npz").is_file()
        ]
        if args.metrics_only
        else [str(path) for _case, _method, path in specs if not path.is_file()]
    )
    if missing:
        kind = "FastQSL bundles" if args.metrics_only else "topology field caches"
        raise SystemExit(f"Missing {kind}:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps([{"benchmark": case, "method": method, "field": str(path)} for case, method, path in specs], indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        meta = json.loads((args.field_dir / f"{case}__case_meta.json").read_text())
        dx, dy, dz = (float(value) for value in meta["dims"][3:])
        bundles: dict[str, dict[str, Any]] = {}
        for _case, method, path in [item for item in specs if item[0] == case]:
            bundle_path = args.output_dir / f"{case}__{method}.npz"
            if args.metrics_only:
                bundle = load_bundle(bundle_path)
                bundles[method] = bundle
                continue
            if args.resume and bundle_path.is_file():
                print(f"reuse FastQSL {case.upper()} {method}", flush=True)
                bundles[method] = load_bundle(bundle_path)
                continue
            print(f"FastQSL {case.upper()} {method}", flush=True)
            field = load_field(path)
            compute = compute_bottom_qmap if args.plane == "bottom" else compute_y0_qmap
            bundle = compute(
                field["Bx"], field["By"], field["Bz"],
                step_len=args.step_len, tol=args.tol, block_x=args.block_x,
                b_smooth_sigma_cells=0.0,
            )
            bundles[method] = bundle
            save_bundle(
                bundle_path,
                bundle,
                {"benchmark": case, "method": method, "field": str(path)},
            )
        reference = bundles["analytic_reference"]
        for method, bundle in bundles.items():
            metrics = fastqsl_bundle_metrics(
                reference, bundle, high_logq=args.high_logq,
                spacing_xyz=(dx, dy, dz),
            )
            rows.append(
                {
                    "benchmark": case,
                    "method": method,
                    "field_smoothing_sigma_cells": 0.0,
                    "seed_protocol": (
                        "every_node_on_photosphere_half_cell_inside"
                        if args.plane == "bottom"
                        else "every_node_on_physical_y0_plane"
                    ),
                    "high_logq": args.high_logq,
                    **metrics,
                }
            )
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "benchmarks": list(cases),
        "methods": sorted({row["method"] for row in rows}),
        "common_protocol": {
            "field_smoothing_sigma_cells": 0.0,
            "plane": args.plane,
            "seeds": (
                "identical every-node x-y photospheric grid, half-cell inside"
                if args.plane == "bottom"
                else "identical every-node x-z grid on physical y=0"
            ),
            "step_len": args.step_len,
            "tol": args.tol,
            "block_x": args.block_x,
            "endpoint_termination_flags": (
                "both reference and candidate endpoints must terminate on "
                "one of FastQSL boundary faces 1--6"
            ),
        },
        "metric_postprocess_from_saved_bundles": bool(args.metrics_only),
        "runner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
