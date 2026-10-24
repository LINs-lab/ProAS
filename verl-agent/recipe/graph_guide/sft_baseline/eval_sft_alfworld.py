#!/usr/bin/env python3
"""Evaluate an SFT checkpoint with run_sh_fair's online alfworld validator."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.graph_guide.sft_baseline.eval_sft_online import main


if __name__ == "__main__":
    main("alfworld")
