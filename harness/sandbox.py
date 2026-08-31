from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from harness.spec import DEFAULT_DERIVED_DIR, ROOT, sha256_file


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxResult:
    status: str
    returncode: int | None
    wall_seconds: float
    stdout_tail: str
    stderr_tail: str
    output_bytes: int


def directory_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise SandboxError(f"symlinks are not allowed in candidate output: {entry}")
        if entry.is_file():
            total += entry.stat().st_size
    return total


def directory_manifest(path: Path) -> tuple[list[dict[str, str | int]], str]:
    records: list[dict[str, str | int]] = []
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink():
            raise SandboxError(f"symlinks are not allowed: {entry}")
        if not entry.is_file():
            continue
        digest = sha256_file(entry)
        records.append(
            {"path": str(entry.relative_to(path)), "bytes": entry.stat().st_size, "sha256": digest}
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return records, hashlib.sha256(encoded).hexdigest()


class BubblewrapRunner:
    def __init__(
        self,
        derived_dir: Path = DEFAULT_DERIVED_DIR,
        venv_dir: Path = ROOT / ".venv",
    ) -> None:
        self.derived_dir = derived_dir.resolve()
        self.venv_dir = venv_dir.resolve()
        self.bwrap = Path(shutil.which("bwrap") or "")
        if not self.bwrap.is_file():
            raise SandboxError("bubblewrap is required")
        self.python = self.venv_dir / "bin" / "python"
        if not self.python.is_file():
            raise SandboxError(f"project Python environment is missing: {self.python}")

    def _base_command(
        self,
        source_dir: Path,
        output_dir: Path,
        cache_dir: Path,
        evaluation_file: Path,
        checkpoint_dir: Path | None,
        allow_gpu: bool,
    ) -> list[str]:
        public = self.derived_dir / "public"
        user_site = Path.home() / ".local" / "lib" / "python3.14" / "site-packages"
        required = [
            public / "user_features.parquet",
            public / "video_features.parquet",
            evaluation_file,
            user_site,
        ]
        if checkpoint_dir is None:
            required.append(public / "train.parquet")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SandboxError(f"sandbox runtime inputs are missing: {missing}")

        command = [
            str(self.bwrap),
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--ro-bind-try",
            "/etc/ld.so.cache",
            "/etc/ld.so.cache",
            "--proc",
            "/proc",
            "--ro-bind",
            "/sys",
            "/sys",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/etc",
            "--dir",
            "/runtime",
            "--ro-bind",
            str(self.venv_dir),
            "/runtime/venv",
            "--ro-bind",
            str(user_site),
            "/runtime/user-site",
            "--dir",
            "/data",
            "--ro-bind",
            str(public / "user_features.parquet"),
            "/data/user_features.parquet",
            "--ro-bind",
            str(public / "video_features.parquet"),
            "/data/video_features.parquet",
            "--ro-bind",
            str(evaluation_file),
            "/data/evaluation_features.parquet",
            "--ro-bind",
            str(source_dir.resolve()),
            "/candidate",
            "--bind",
            str(output_dir.resolve()),
            "/output",
            "--bind",
            str(cache_dir.resolve()),
            "/cache",
            "--dir",
            "/home",
            "--dir",
            "/home/researcher",
            "--setenv",
            "HOME",
            "/home/researcher",
            "--setenv",
            "PATH",
            "/runtime/venv/bin:/usr/bin",
            "--setenv",
            "PYTHONPATH",
            "/runtime/user-site",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "OMP_NUM_THREADS",
            "12",
            "--setenv",
            "MKL_NUM_THREADS",
            "12",
            "--setenv",
            "OPENBLAS_NUM_THREADS",
            "12",
        ]
        # Final inference must use the frozen checkpoint. Withholding train data
        # mechanically prevents retraining after the run becomes terminal.
        if checkpoint_dir is None:
            command.extend(
                ["--ro-bind", str(public / "train.parquet"), "/data/train.parquet"]
            )
        if checkpoint_dir is not None:
            command.extend(["--ro-bind", str(checkpoint_dir.resolve()), "/checkpoint"])
        if allow_gpu:
            for device in (
                "/dev/nvidia0",
                "/dev/nvidiactl",
                "/dev/nvidia-uvm",
                "/dev/nvidia-uvm-tools",
                "/dev/nvidia-modeset",
            ):
                if Path(device).exists():
                    command.extend(["--dev-bind", device, device])
            if Path("/dev/nvidia-caps").is_dir():
                command.extend(["--dev-bind", "/dev/nvidia-caps", "/dev/nvidia-caps"])
        command.extend(["--chdir", "/candidate"])
        return command

    @staticmethod
    def _limits(max_file_bytes: int) -> None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))
        # RLIMIT_NPROC is per host user, not per sandbox. A desktop session already
        # owns many processes, so a tiny value can prevent Bubblewrap itself from
        # creating its namespace. This remains a broad emergency ceiling.
        resource.setrlimit(resource.RLIMIT_NPROC, (4096, 4096))
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))

    def run(
        self,
        *,
        source_dir: Path,
        output_dir: Path,
        cache_dir: Path,
        mode: str,
        timeout_seconds: int,
        max_output_bytes: int,
        checkpoint_dir: Path | None = None,
        allow_gpu: bool = True,
    ) -> SandboxResult:
        if mode not in {"attempt", "checkpoint_check", "final"}:
            raise SandboxError("mode must be attempt, checkpoint_check or final")
        if mode != "attempt" and checkpoint_dir is None:
            raise SandboxError("checkpoint inference requires a frozen checkpoint directory")
        entrypoint = source_dir / "candidate.py"
        if not entrypoint.is_file():
            raise SandboxError("candidate.py is missing")
        output_dir.mkdir(parents=True, exist_ok=False)
        cache_dir.mkdir(parents=True, exist_ok=True)
        evaluation_file = self.derived_dir / "public" / (
            "test_features.parquet" if mode == "final" else "validation_features.parquet"
        )
        command = self._base_command(
            source_dir, output_dir, cache_dir, evaluation_file, checkpoint_dir, allow_gpu
        )
        command.extend(
            [
                "/runtime/venv/bin/python",
                "/candidate/candidate.py",
                "--mode",
                "attempt" if mode == "attempt" else "final",
                "--data-root",
                "/data",
                "--output-dir",
                "/output",
                "--cache-dir",
                "/cache",
            ]
        )
        if checkpoint_dir is not None:
            command.extend(["--checkpoint-dir", "/checkpoint"])

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                start_new_session=True,
                preexec_fn=lambda: self._limits(max_output_bytes),
                check=False,
            )
            status = "ok" if completed.returncode == 0 else "candidate_error"
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            returncode = None
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        elapsed = time.monotonic() - started
        output_bytes = directory_size(output_dir)
        if output_bytes > max_output_bytes:
            status = "output_limit"
        return SandboxResult(
            status=status,
            returncode=returncode,
            wall_seconds=elapsed,
            stdout_tail=stdout[-65536:],
            stderr_tail=stderr[-65536:],
            output_bytes=output_bytes,
        )
