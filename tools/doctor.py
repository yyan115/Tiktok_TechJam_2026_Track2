from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq
import torch

from harness.runtime import runtime_fingerprint
from harness.spec import (
    DEFAULT_DERIVED_DIR,
    DEFAULT_RAW_DIR,
    ROOT,
    BenchmarkSpec,
    sha256_file,
)


class DoctorError(RuntimeError):
    pass


def _verify_data() -> dict[str, Any]:
    spec = BenchmarkSpec.load()
    spec.validate_raw_files(DEFAULT_RAW_DIR)
    manifest_path = DEFAULT_DERIVED_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("benchmark_spec_sha256") != spec.digest:
        raise DoctorError("derived data uses a different benchmark specification")
    if manifest.get("hidden_test_targets_cached") is not False:
        raise DoctorError("derived data claims to cache hidden-test targets")

    for record in manifest["raw_inputs"].values():
        path = DEFAULT_RAW_DIR / record["file"]
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise DoctorError(f"raw input differs from its manifest: {path.name}")
    for record in manifest["outputs"]:
        path = DEFAULT_DERIVED_DIR / record["path"]
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise DoctorError(f"derived file differs from its manifest: {record['path']}")

    public = DEFAULT_DERIVED_DIR / "public"
    expected_public = {
        "train.parquet",
        "validation_features.parquet",
        "test_features.parquet",
        "user_features.parquet",
        "video_features.parquet",
    }
    if {path.name for path in public.iterdir()} != expected_public:
        raise DoctorError("unexpected file in the candidate-visible data directory")
    safe_future = {"row_id", *spec.future_feature_columns}
    for split in ("validation", "test"):
        schema = set(pq.read_schema(public / f"{split}_features.parquet").names)
        if schema != safe_future:
            raise DoctorError(f"{split} feature schema is not the frozen label-free schema")
        rows = pq.ParquetFile(public / f"{split}_features.parquet").metadata.num_rows
        if rows != spec.expected_rows[split]:
            raise DoctorError(f"{split} feature row count is wrong")
    private_names = {
        path.name for path in (DEFAULT_DERIVED_DIR / "private").iterdir() if path.is_file()
    }
    if private_names != {"validation_users.npy", "validation_labels.npy"}:
        raise DoctorError("trusted data contains unexpected private arrays")
    if any("test" in path.name.lower() and "label" in path.name.lower() for path in DEFAULT_DERIVED_DIR.rglob("*")):
        raise DoctorError("a cached hidden-test target file exists")
    return {"rows": manifest["rows"], "hidden_test_targets_cached": False}


def _verify_runtime() -> dict[str, Any]:
    lock = json.loads((ROOT / "config" / "runtime-lock.json").read_text())
    actual = runtime_fingerprint()
    if actual["python"] != lock["python"]:
        raise DoctorError("Python version differs from runtime-lock.json")
    if actual["packages"] != lock["packages"]:
        raise DoctorError("package versions differ from runtime-lock.json")
    cuda = actual["cuda"]
    if not cuda["available"] or cuda["torch_runtime"] != lock["torch_cuda"]:
        raise DoctorError("the locked CUDA-enabled PyTorch runtime is unavailable")
    if cuda["device"] != lock["target_gpu"]:
        raise DoctorError("GPU differs from runtime-lock.json")
    if any(actual[name] is None for name in ("bubblewrap", "codex", "claude")):
        raise DoctorError("bubblewrap, Codex CLI, or Claude CLI is missing")
    value = torch.ones((256, 256), device="cuda").square().sum().item()
    if value != 65536.0:
        raise DoctorError("CUDA arithmetic smoke check failed")
    return {
        "python": actual["python"],
        "packages": actual["packages"],
        "cuda": cuda,
        "bubblewrap": actual["bubblewrap"],
        "codex": actual["codex"],
        "claude": actual["claude"],
    }


def _smoke(script: str) -> dict[str, Any]:
    module = f"tools.{Path(script).stem}"
    result = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-m", module],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise DoctorError(f"{script} failed: {result.stderr[-2000:]}")
    return {"status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only campaign readiness checks")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    checks: dict[str, dict[str, Any]] = {}
    tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("data", _verify_data),
        ("runtime", _verify_runtime),
    ]
    if not args.skip_smoke:
        tasks.extend(
            [
                ("candidate_sandbox", lambda: _smoke("smoke_sandbox.py")),
                ("agent_sandbox", lambda: _smoke("smoke_agent.py")),
            ]
        )
    ready = True
    for name, task in tasks:
        try:
            checks[name] = {"ok": True, "details": task()}
        except Exception as exc:
            ready = False
            checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps({"ready": ready, "checks": checks}, sort_keys=True))
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
