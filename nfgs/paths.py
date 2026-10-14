"""Locate an editable checkout without depending on the original research project."""

from pathlib import Path
import os


def project_root():
    configured = os.environ.get("NFGS_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent
