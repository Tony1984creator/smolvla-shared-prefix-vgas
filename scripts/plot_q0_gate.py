"""Render archived candidate-zero diagnostics with the manuscript style."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
from make_submission import q0_gate


if __name__ == "__main__":
    q0_gate()
