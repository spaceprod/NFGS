#!/usr/bin/env python3
"""Check full weight coverage, numerical provenance and manuscript stopping rules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfgs.workflow import experiment_main

if __name__ == "__main__":
    raise SystemExit(experiment_main("audit"))
