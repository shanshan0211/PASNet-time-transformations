"""Run the AC17-compatible checkpoint analysis on sealed AC18 artifacts."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC18")
sys.path.insert(0, HERE)

import analyze_expAC17 as analysis  # noqa: E402


def main():
    analysis.OUT = OUT
    analysis.AUDIT_PATH = os.path.join(OUT, "hamiltonian_audit.json")
    analysis.TRAINING_PATH = os.path.join(OUT, "training.json")
    analysis.RAW_PATH = os.path.join(OUT, "raw_results.json")
    analysis.SUMMARY_PATH = os.path.join(OUT, "summary.json")
    analysis.ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
    analysis.main()


if __name__ == "__main__":
    main()
