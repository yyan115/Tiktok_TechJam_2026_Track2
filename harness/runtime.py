from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from harness.spec import ROOT, sha256_file


PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "torch",
    "lightgbm",
    "polars",
    "scikit-learn",
)

TRUSTED_CODE = (
    "harness/agent.py",
    "harness/controller.py",
    "harness/eda.py",
    "harness/ledger.py",
    "harness/metrics.py",
    "harness/policy.py",
    "harness/runtime.py",
    "harness/sandbox.py",
    "harness/spec.py",
    "harness/supervisor.py",
)


def _version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None


def trusted_code_hashes(root: Path = ROOT) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in TRUSTED_CODE}


def runtime_fingerprint() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            package: importlib.metadata.version(package) for package in PACKAGE_NAMES
        },
        "cuda": {
            "available": cuda_available,
            "torch_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda_available else None,
            "total_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory if cuda_available else None
            ),
        },
        "bubblewrap": _version("bwrap"),
        "claude": _version("claude"),
        "trusted_code_sha256": trusted_code_hashes(),
    }
