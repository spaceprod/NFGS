#!/usr/bin/env python3
"""Manufactured modal solutions and grid convergence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfgs.workflow import experiment_main

if __name__ == "__main__":
    raise SystemExit(experiment_main("modal_verification"))
