"""Configuration-driven experiment execution and resumable manuscript campaigns."""

from __future__ import annotations
import argparse, copy, hashlib, importlib, json, os, runpy, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from nfgs.paths import project_root


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported config schema")
    return config


def resolve_device(device):
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def experiment_arguments(config, output_root, profile, device):
    context = {
        "project_root": str(project_root()),
        "output_root": str(Path(output_root).resolve()),
        "device": device,
    }
    arguments = copy.deepcopy(config.get("arguments", {}))
    arguments.update(config.get("profiles", {}).get(profile, {}).get("arguments", {}))
    argv = []
    for key, value in arguments.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif value is not None:
            argv.extend([flag, str(value).format(**context)])
    return argv


def artifact_paths(config, output_root):
    context = {
        "project_root": str(project_root()),
        "output_root": str(Path(output_root).resolve()),
    }
    return [Path(value.format(**context)) for value in config.get("artifacts", [])]


def apply_case_config(config, profile):
    if config.get("low_lou_cases"):
        from nfgs._backend.util.lnl_io import LOW_LOU_FAMILY

        LOW_LOU_FAMILY.update(copy.deepcopy(config["low_lou_cases"]))
    parameters = copy.deepcopy(config.get("td_cases", {}))
    for key, value in (
        config.get("profiles", {}).get(profile, {}).get("td_cases", {}).items()
    ):
        parameters[key] = value
    if parameters:
        from nfgs._backend.scripts import run_analytic_validation as analytic
        from nfgs._backend.td import TDModelConfig

        for name, spec in parameters.items():
            analytic.TD_TOPOLOGY[name] = {
                "config": TDModelConfig(**spec["parameters"]),
                "nodes": tuple(spec["grid_nodes_xyz"]),
            }


def run_module(config, output_root, profile, device, force=False):
    apply_case_config(config, profile)
    module = config["entrypoint"]
    argv = experiment_arguments(config, output_root, profile, device)
    if force and config.get("supports_force"):
        argv.append("--force")
    old_argv = sys.argv
    sys.argv = [module] + argv
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = old_argv


def experiment_main(identifier):
    parser = argparse.ArgumentParser(
        description=f"Run the {identifier} experiment from a reproducible JSON configuration."
    )
    parser.add_argument(
        "--config", type=Path, default=project_root() / "configs" / f"{identifier}.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=project_root() / "results" / "paper"
    )
    parser.add_argument("--profile", choices=["paper", "quick"], default="paper")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    if config["id"] != identifier:
        parser.error("script and config experiment IDs differ")
    device = resolve_device(args.device)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "id": identifier,
                    "module": config["entrypoint"],
                    "argv": experiment_arguments(
                        config, args.output_root, args.profile, device
                    ),
                    "artifacts": [
                        str(p) for p in artifact_paths(config, args.output_root)
                    ],
                    "profile": args.profile,
                },
                indent=2,
            )
        )
        return 0
    os.environ["NFGS_EXPERIMENT_CONFIG"] = str(args.config.resolve())
    os.environ["NFGS_RUN_PROFILE"] = args.profile
    run_module(config, args.output_root, args.profile, device, args.force)
    missing = [
        str(p) for p in artifact_paths(config, args.output_root) if not p.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Experiment did not produce required artifacts: " + ", ".join(missing)
        )
    return 0


def plan(campaign, configs_dir, selected=None, profile="paper"):
    available = campaign["experiments"]
    enabled = (
        campaign.get("profiles", {}).get(profile, {}).get("experiments", available)
    )
    selected = list(selected or enabled)
    result = []
    visiting = set()
    done = set()

    def add(identifier):
        if identifier in done:
            return
        if identifier in visiting:
            raise ValueError("Cyclic experiment dependency: " + identifier)
        if identifier not in available:
            raise ValueError("Unknown experiment: " + identifier)
        visiting.add(identifier)
        config = read_config(configs_dir / f"{identifier}.json")
        for dependency in config.get("depends_on", []):
            add(dependency)
        visiting.remove(identifier)
        done.add(identifier)
        result.append(config)

    for identifier in selected:
        add(identifier)
    return result


def campaign_main():
    parser = argparse.ArgumentParser(
        description="Run all manuscript experiments, record their configuration and visualize their outputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "campaign_paper.json",
    )
    parser.add_argument(
        "--output-root", type=Path, default=project_root() / "results" / "paper"
    )
    parser.add_argument("--profile", choices=["paper", "quick"], default="paper")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--only", help="Comma-separated experiment IDs; prerequisites are included"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    campaign = read_config(args.config)
    device = resolve_device(args.device)
    configs_dir = args.config.resolve().parent
    configs = plan(
        campaign, configs_dir, args.only.split(",") if args.only else None, args.profile
    )
    commands = []
    for config in configs:
        command = [
            sys.executable,
            str(project_root() / "experiments" / f"{config['id']}.py"),
            "--config",
            str(configs_dir / f"{config['id']}.json"),
            "--output-root",
            str(args.output_root.resolve()),
            "--profile",
            args.profile,
            "--device",
            device,
        ]
        if args.force:
            command.append("--force")
        commands.append((config, command))
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "id": c["id"],
                        "depends_on": c.get("depends_on", []),
                        "command": cmd,
                        "module": c["entrypoint"],
                        "arguments": experiment_arguments(
                            c, args.output_root, args.profile, device
                        ),
                    }
                    for c, cmd in commands
                ],
                indent=2,
            )
        )
        return 0
    if any(c.get("requires_cuda") for c, _ in commands):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "FastQSL stages require a CUDA GPU; use --only or --profile quick for CPU checks."
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "campaign_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"schema_version": 1, "started_at": now(), "experiments": {}}
    )
    state.update(
        {
            "profile": args.profile,
            "device": device,
            "status": "running",
            "scientific_evidence": args.profile == "paper",
        }
    )
    source_files = [project_root() / "nfgs_solver.py"] + sorted(
        (project_root() / "nfgs").rglob("*.py")
    )
    source_fingerprint = hashlib.sha256(
        "".join(
            str(p.relative_to(project_root())) + sha(p) for p in source_files
        ).encode()
    ).hexdigest()
    env = dict(
        os.environ,
        NFGS_PROJECT_ROOT=str(project_root()),
        PYTHONPATH=str(project_root()),
        MPLBACKEND="Agg",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for config, command in commands:
        identifier = config["id"]
        context = json.dumps(
            {
                "config": config,
                "profile": args.profile,
                "device": device,
                "source": source_fingerprint,
            },
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(context.encode()).hexdigest()
        previous = state["experiments"].get(identifier, {})
        artifacts = artifact_paths(config, args.output_root)
        if (
            args.resume
            and previous.get("status") == "complete"
            and previous.get("fingerprint") == fingerprint
            and all(
                p.is_file() and sha(p) == previous.get("artifacts", {}).get(str(p))
                for p in artifacts
            )
        ):
            print(f"[resume] {identifier}", flush=True)
            continue
        if (
            previous.get("status") == "complete"
            and previous.get("fingerprint") != fingerprint
            and not args.force
        ):
            raise RuntimeError(
                f"{identifier}: config or source changed; choose a fresh output root or explicitly pass --force"
            )
        log = args.output_root / "logs" / f"{identifier}.log"
        log.parent.mkdir(exist_ok=True)
        item = {
            "status": "running",
            "started_at": now(),
            "fingerprint": fingerprint,
            "command": command,
            "config": config,
            "log": str(log),
        }
        state["experiments"][identifier] = item
        atomic_json(state_path, state)
        print(f"[{identifier}] log: {log}", flush=True)
        started = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.run(
                command,
                cwd=project_root(),
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        item.update(
            {
                "returncode": process.returncode,
                "wall_time_s": time.monotonic() - started,
                "finished_at": now(),
            }
        )
        missing = [str(p) for p in artifacts if not p.is_file()]
        item["status"] = (
            "complete" if process.returncode == 0 and not missing else "failed"
        )
        item["missing_artifacts"] = missing
        item["artifacts"] = {str(p): sha(p) for p in artifacts if p.is_file()}
        if item["status"] == "failed":
            state["status"] = "failed"
            atomic_json(state_path, state)
            print(log.read_text()[-6000:], file=sys.stderr)
            return 1
        atomic_json(state_path, state)
        print(f"[{identifier}] complete ({item['wall_time_s']:.1f}s)", flush=True)
    state["status"] = "complete"
    state["finished_at"] = now()
    atomic_json(state_path, state)
    print(f"Completed campaign: {state_path}", flush=True)
    return 0
