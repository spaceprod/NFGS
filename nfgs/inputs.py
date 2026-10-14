"""Fetch checksum-pinned archived NFPT comparison fields without Git LFS."""

from __future__ import annotations
import argparse, hashlib, json, tarfile, tempfile, urllib.request
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    a.destination.mkdir(parents=True, exist_ok=True)
    missing = [
        r
        for r in manifest["files"]
        if not (a.destination / r["name"]).is_file()
        or sha(a.destination / r["name"]) != r["sha256"]
    ]
    if not missing:
        print("Archived comparison fields verified")
        return
    if a.archive:
        archive = a.archive
    else:
        url = manifest.get("url")
        if not url:
            raise SystemExit(
                "Archive URL has not been published yet; pass --archive or place checksum-matching files in data/nfpt."
            )
        archive = a.destination / "nfpt_comparison_fields.tar"
        print("Downloading archived NFPT comparison inputs:", url, flush=True)
        urllib.request.urlretrieve(url, archive)
    if sha(archive) != manifest["archive_sha256"]:
        raise SystemExit("Comparison archive checksum mismatch")
    expected = {r["name"] for r in manifest["files"]}
    with tarfile.open(archive) as bundle:
        for item in bundle.getmembers():
            if item.name not in expected or not item.isfile():
                raise SystemExit("Unexpected archive member: " + item.name)
            with bundle.extractfile(item) as src, (a.destination / item.name).open(
                "wb"
            ) as dst:
                shutil.copyfileobj(src, dst)
    for record in manifest["files"]:
        if sha(a.destination / record["name"]) != record["sha256"]:
            raise SystemExit("Archived field checksum mismatch: " + record["name"])
    print("Verified", len(manifest["files"]), "archived field inputs")


if __name__ == "__main__":
    main()
