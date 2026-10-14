#!/usr/bin/env python3
"""Reconstruct all three TD cases at eleven CCI weights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfgs.workflow import experiment_main

if __name__ == "__main__":
    raise SystemExit(experiment_main("td_rho_scan"))
