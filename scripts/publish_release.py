#!/usr/bin/env python3
"""Publish the prepared code and checksum-pinned comparison-data release."""
from __future__ import annotations
import argparse, json, os, shlex, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(args, capture=False, check=True, env=None):
    return subprocess.run(
        args, cwd=ROOT, text=True, capture_output=capture, check=check, env=env
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repository",
        help="OWNER/NFGS; defaults to the authenticated personal account",
    )
    p.add_argument(
        "--archive", type=Path, default=ROOT / "release_data/nfpt_comparison_fields.tar"
    )
    p.add_argument("--gh", default=shutil.which("gh") or str(ROOT / ".local/bin/gh"))
    a = p.parse_args()
    if not a.archive.is_file():
        p.error("The prepared comparison data archive is required")
    env = dict(os.environ)
    if "GH_CONFIG_DIR" not in env and (ROOT / ".local/gh/hosts.yml").is_file():
        env["GH_CONFIG_DIR"] = str(ROOT / ".local/gh")
    command([a.gh, "auth", "status"], env=env)
    owner = command(
        [a.gh, "api", "user", "--jq", ".login"], capture=True, env=env
    ).stdout.strip()
    repository = a.repository or f"{owner}/NFGS"
    if len(repository.split("/")) != 2:
        p.error("Expected OWNER/REPOSITORY")
    url = f"https://github.com/{repository}"
    manifest_path = ROOT / "configs/archived_inputs_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    import hashlib

    if hashlib.sha256(a.archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        p.error("Data archive checksum mismatch")
    manifest["url"] = url + "/releases/download/v0.1.0/" + a.archive.name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cff = ROOT / "CITATION.cff"
    text = cff.read_text()
    text += 'repository-code: "' + url + '"\n' if "repository-code:" not in text else ""
    cff.write_text(text)
    availability = ROOT / "docs/manuscript_code_availability.md"
    availability.write_text(
        "The standalone NFGS solver, experiment scripts and configurations, integrated reproduction runner, and results-overview notebook are publicly available at "
        + url
        + ". Release v0.1.0 includes checksum-verified archived NFPT comparison fields and the manuscript result tables. The repository documents the numerical environment, benchmark parameters, scoring domains, and commands required to reproduce the reconstructions and diagnostics.\n\n독립 실행 가능한 NFGS 풀이기, 실험별 스크립트와 설정 파일, 통합 재현 실행기, 결과 개요 노트북은 "
        + url
        + " 에 공개되어 있다. v0.1.0 배포본에는 체크섬으로 검증하는 보관된 NFPT 비교 자기장과 논문의 결과표가 포함되어 있다. 저장소에는 복원과 진단을 재현하는 데 필요한 수치 환경, 검증 매개변수, 평가 영역 및 실행 명령을 제시한다.\n"
    )
    if not (ROOT / ".git").exists():
        command(["git", "init", "-b", "main"])
    command(["git", "config", "user.name", owner])
    command(["git", "config", "user.email", owner + "@users.noreply.github.com"])
    command(
        [
            "git",
            "config",
            "credential.https://github.com.helper",
            "!"
            + ("GH_CONFIG_DIR=" + shlex.quote(env["GH_CONFIG_DIR"]) + " " if "GH_CONFIG_DIR" in env else "")
            + shlex.quote(a.gh) + " auth git-credential",
        ]
    )
    command([sys.executable, str(ROOT / "scripts/check_publication_contents.py")])
    command(["git", "add", "."])
    changed = command(["git", "diff", "--cached", "--quiet"], check=False)
    if changed.returncode:
        command(
            [
                "git",
                "commit",
                "-m",
                "Publish standalone NFGS and manuscript reproduction suite",
            ]
        )
    exists = command(
        [a.gh, "repo", "view", repository, "--json", "visibility"],
        capture=True,
        check=False,
        env=env,
    )
    if exists.returncode:
        command(
            [
                a.gh,
                "repo",
                "create",
                repository,
                "--public",
                "--description",
                "Gaussian-separated nonlinear force-free reconstruction and reproducible manuscript experiments",
            ],
            env=env,
        )
    elif json.loads(exists.stdout)["visibility"] != "PUBLIC":
        p.error("The selected repository is not public")
    remote = command(["git", "remote", "get-url", "origin"], capture=True, check=False)
    if not remote.returncode and remote.stdout.strip() != url + ".git":
        p.error("Existing origin points to a different repository")
    if remote.returncode:
        command(["git", "remote", "add", "origin", url + ".git"])
    command(["git", "push", "--set-upstream", "origin", "main"], env=env)
    release = command(
        [a.gh, "release", "view", "v0.1.0", "--repo", repository],
        capture=True,
        check=False,
        env=env,
    )
    if release.returncode:
        command(
            [
                a.gh,
                "release",
                "create",
                "v0.1.0",
                str(a.archive),
                "--repo",
                repository,
                "--target",
                "main",
                "--title",
                "NFGS v0.1.0: manuscript reproduction suite",
                "--notes",
                "Standalone solver, experiment scripts and explicit configurations, integrated reproduction workflow, overview notebook and checksum-pinned archived NFPT comparison inputs. Archived result tables are distinguished from fresh campaign results.",
            ],
            env=env,
        )
    else:
        raise SystemExit(
            "Release v0.1.0 already exists; existing release assets are preserved."
        )
    print(
        json.dumps(
            {
                "repository": url,
                "release": url + "/releases/tag/v0.1.0",
                "commit": command(
                    ["git", "rev-parse", "HEAD"], capture=True
                ).stdout.strip(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
