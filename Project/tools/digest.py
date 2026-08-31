#!/usr/bin/env python3
"""Owner read-only alias for the authenticated controller log."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/usr/bin/python3")
CONTROLLER = ROOT / "Project" / "harness" / "iterate.py"


def main() -> int:
    """Delegate to the one implementation that verifies authority and replay."""

    completed = subprocess.run(
        [str(PYTHON), "-I", str(CONTROLLER), "log"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
