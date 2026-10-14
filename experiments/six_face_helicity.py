#!/usr/bin/env python3
"""Six-face potential-field energy and current-carrying helicity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfgs.workflow import experiment_main

if __name__ == "__main__":
    raise SystemExit(experiment_main("six_face_helicity"))
