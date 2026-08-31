#!/usr/bin/env python3
"""Untrusted candidate worker executed inside a bubblewrap mount namespace.

The worker can see one candidate file, the organizer's public Python modules,
and the sanitized KuaiRand data directory.  It cannot see the raw data,
official journal, sealed artifacts, parent process, or repository checkout.
It returns prediction arrays only; scoring stays in the trusted parent.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import resource
import stat
import sys
import types
from pathlib import Path

# Pin native math runtimes before organizer code or NumPy is imported.  This
# keeps their thread creation compatible with the hard process bound even when
# the worker is invoked directly by a synthetic test.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def load_candidate(path: Path):
    source = path.read_bytes()
    module = types.ModuleType("official_candidate")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    if not hasattr(module, "run"):
        raise RuntimeError("candidate must define run(splits)")
    return module


def finite_array(values, expected: int, name: str):
    import numpy as np

    # Never materialize an unbounded generator supplied by candidate code.
    bounded = list(itertools.islice(iter(values), expected + 1))
    arr = np.asarray(bounded, dtype=np.float64)
    if arr.ndim != 1 or len(arr) != expected:
        raise RuntimeError(f"{name} predictions have shape {arr.shape}; expected ({expected},)")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name} predictions contain NaN/Inf")
    return arr


def write_fixed(path: Path, payload: bytes) -> None:
    """Replace one pre-mounted fixed output without creating filesystem names."""

    flags = (
        os.O_WRONLY
        | os.O_TRUNC
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"output is not the fixed regular file: {path.name}")
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError(f"output write made no progress: {path.name}")
            view = view[written:]
        os.fsync(fd)
        if os.fstat(fd).st_size != len(payload):
            raise RuntimeError(f"output has wrong final size: {path.name}")
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--kit", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-seconds", required=True, type=float)
    parser.add_argument("--max-output-bytes", required=True, type=int)
    args = parser.parse_args()

    kit = Path(args.kit).resolve()
    data_dir = Path(args.data).resolve()
    candidate = Path(args.candidate).resolve()
    output_dir = Path(args.output_dir).resolve()
    sys.path.insert(0, str(kit))

    # Hard process limits supplement the outer wall timeout.  Candidate code
    # cannot raise a hard limit after this point.
    cpu_limit = max(1, math.ceil(args.cpu_seconds))
    if args.max_output_bytes < 64 * 1024:
        raise RuntimeError("trusted output-size limit is invalid")
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (args.max_output_bytes, args.max_output_bytes),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024,) * 2)

    from data import load

    splits = load(str(data_dir))
    if any(row[6] != 0 for split in ("valid", "test") for row in splits[split]):
        raise RuntimeError(
            "candidate data is not feature-only: validation/test label found"
        )
    if any(row[0] > 20220421 for row in splits["train"]):
        raise RuntimeError("training split crosses the fixed training date boundary")

    module = load_candidate(candidate)
    result = module.run({name: list(rows) for name, rows in splits.items()})
    if not isinstance(result, dict) or set(result) != {"valid", "test"}:
        raise RuntimeError("candidate run() must return exactly {'valid', 'test'}")

    valid = finite_array(result["valid"], len(splits["valid"]), "validation")
    test = finite_array(result["test"], len(splits["test"]), "test")
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise RuntimeError("output directory is not the fixed sandbox directory")
    valid = valid.astype("<f8", copy=False)
    test = test.astype("<f8", copy=False)
    write_fixed(output_dir / "valid.f64", valid.tobytes(order="C"))
    write_fixed(output_dir / "test.f64", test.tobytes(order="C"))
    metadata = {
        "valid_rows": len(valid),
        "test_rows": len(test),
    }
    write_fixed(
        output_dir / "metadata.json",
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
