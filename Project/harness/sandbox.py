"""OS-isolated execution of one exact Track 2 candidate byte string."""

from __future__ import annotations

import base64
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import signal
import site
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path, PurePosixPath

import numpy as np


SAFE_KIT_FILES = (
    "data.py",
    "baseline.py",
    "evaluate.py",
)


class SandboxError(RuntimeError):
    pass


NUMPY_SITE = "/opt/numpy-site"
OUTPUT_FILES = ("valid.f64", "test.f64", "metadata.json")
TMPFS_BYTES = 32 * 1024 * 1024
BROAD_HOST_BIND_SOURCES = (Path("/usr"),)
RUNTIME_ATTESTATION_FORMAT = "track2.candidate-runtime.v1"
MAX_NUMPY_ARTIFACTS = 5_000
MAX_NUMPY_BYTES = 512 * 1024 * 1024
MAX_STDLIB_ARTIFACTS = 5_000
MAX_STDLIB_BYTES = 512 * 1024 * 1024


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_stable_regular(
    path: Path, *, maximum_bytes: int, allow_hardlinks: bool = False
) -> tuple[str, int]:
    """Hash one bounded regular file through a stable, no-follow descriptor."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise SandboxError(f"runtime artifact is missing: {path}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink < 1
        or (not allow_hardlinks and before.st_nlink != 1)
        or before.st_size > maximum_bytes
    ):
        raise SandboxError(f"runtime artifact is unsafe: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SandboxError(f"runtime artifact cannot be opened: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(fd)
        if (
            (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
            != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink)
            or opened.st_size > maximum_bytes
        ):
            raise SandboxError(f"runtime artifact changed during open: {path}")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum_bytes:
                raise SandboxError(f"runtime artifact exceeds its size cap: {path}")
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
             opened.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns)
            or size != opened.st_size
        ):
            raise SandboxError(f"runtime artifact changed while hashing: {path}")
    finally:
        os.close(fd)
    return digest.hexdigest(), size


def _walk_runtime_tree(
    root: Path,
    *,
    prefix: str,
    max_files: int,
    max_bytes: int,
    excluded_directories: frozenset[str] = frozenset(),
) -> tuple[tuple[Path, str, str], ...]:
    """Return a deterministic no-symlink file inventory for one runtime tree."""

    try:
        root_metadata = root.lstat()
        root = root.resolve(strict=True)
    except OSError as exc:
        raise SandboxError(f"runtime tree is unavailable: {root}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise SandboxError(f"runtime tree is unsafe: {root}")
    artifacts: list[tuple[Path, str, str]] = []
    total_bytes = 0

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        nonlocal total_bytes
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise SandboxError(f"runtime tree cannot be listed: {directory}") from exc
        for entry in entries:
            if entry.name in {"", ".", ".."}:
                raise SandboxError("runtime tree contains an unsafe name")
            relative = (*relative_parts, entry.name)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise SandboxError(
                    f"runtime tree entry cannot be inspected: {entry.path}"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                if entry.name in excluded_directories:
                    continue
                visit(Path(entry.path), relative)
                continue
            if entry.name.endswith(".pyc"):
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink < 1:
                raise SandboxError(f"runtime tree entry is unsafe: {entry.path}")
            remaining = max_bytes - total_bytes
            if remaining < 0:
                raise SandboxError("runtime tree exceeds its total byte cap")
            # Package managers may install identical license/header files as
            # hardlinks. Every logical alias is retained in the manifest and
            # mounted separately; no inode is silently deduplicated.
            digest, size = _hash_stable_regular(
                Path(entry.path), maximum_bytes=remaining, allow_hardlinks=True
            )
            total_bytes += size
            artifacts.append((
                Path(entry.path).resolve(strict=True),
                f"{prefix}/{'/'.join(relative)}",
                digest,
            ))
            if len(artifacts) > max_files or total_bytes > max_bytes:
                raise SandboxError("runtime tree exceeds its inventory cap")

    visit(root, ())
    if not artifacts:
        raise SandboxError(f"runtime tree is empty: {root}")
    return tuple(artifacts)


def _mkdir_args(path: str, created: set[str]) -> list[str]:
    current = ""
    args: list[str] = []
    for part in Path(path).parts:
        if part == "/":
            current = "/"
            continue
        current = str(Path(current) / part)
        if current not in created:
            args += ["--dir", current]
            created.add(current)
    return args


def _reject_sensitive_broad_bind_overlap(
    paths: tuple[tuple[Path, str], ...],
) -> None:
    """Fail if a private path would reappear through a broad host bind."""

    for path, label in paths:
        resolved = path.resolve(strict=False)
        for source in BROAD_HOST_BIND_SOURCES:
            if (
                resolved == source
                or resolved.is_relative_to(source)
                or source.is_relative_to(resolved)
            ):
                raise SandboxError(
                    f"{label} overlaps broadly mounted host runtime {source}"
                )


def exact_numpy_artifacts() -> tuple[tuple[Path, str, str], ...]:
    """Return an exact bounded NumPy runtime inventory.

    Mounting the entire mutable user ``site-packages`` directory exposed
    unrelated packages and stray files. NumPy's installed RECORD provides an
    explicit allowlist when available. Distribution-managed installations
    such as RPM may legitimately have no RECORD; for those, inventory only the
    resolved ``numpy`` and adjacent ``numpy.libs`` trees and hash every file.
    """

    try:
        distribution = importlib_metadata.distribution("numpy")
    except importlib_metadata.PackageNotFoundError as exc:
        raise SandboxError("the trusted host has no NumPy distribution") from exc
    files = distribution.files
    distribution_root = Path(distribution.locate_file("")).resolve(strict=True)

    if not files:
        try:
            package_root = Path(np.__file__).parent
            package_metadata = package_root.lstat()
            package_root = package_root.resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise SandboxError("the installed NumPy package root is unsafe") from exc
        if (
            stat.S_ISLNK(package_metadata.st_mode)
            or not stat.S_ISDIR(package_metadata.st_mode)
            or package_root.name != "numpy"
            or not package_root.is_relative_to(distribution_root)
        ):
            raise SandboxError("the installed NumPy package root is unsafe")
        artifacts = list(_walk_runtime_tree(
            package_root,
            prefix="numpy",
            max_files=MAX_NUMPY_ARTIFACTS,
            max_bytes=MAX_NUMPY_BYTES,
            excluded_directories=frozenset({"__pycache__"}),
        ))
        libraries = package_root.parent / "numpy.libs"
        if libraries.exists():
            remaining_files = MAX_NUMPY_ARTIFACTS - len(artifacts)
            used_bytes = sum(path.stat().st_size for path, _, _ in artifacts)
            artifacts.extend(_walk_runtime_tree(
                libraries,
                prefix="numpy.libs",
                max_files=remaining_files,
                max_bytes=MAX_NUMPY_BYTES - used_bytes,
                excluded_directories=frozenset({"__pycache__"}),
            ))
        if not any(
            relative == "numpy/__init__.py" for _, relative, _ in artifacts
        ):
            raise SandboxError("NumPy inventory did not include numpy/__init__.py")
        return tuple(sorted(artifacts, key=lambda value: value[1]))

    artifacts: list[tuple[Path, str, str]] = []
    total_bytes = 0
    for item in files:
        relative = item.as_posix()
        relative_path = PurePosixPath(relative)
        if not (relative.startswith("numpy/") or relative.startswith("numpy.libs/")):
            continue
        if relative_path.is_absolute() or any(
            part in {"", ".", ".."} for part in relative_path.parts
        ):
            raise SandboxError(f"NumPy RECORD has an unsafe path: {relative}")
        if "/__pycache__/" in f"/{relative}" or relative.endswith(".pyc"):
            continue
        if item.hash is None or item.hash.mode != "sha256" or item.size is None:
            raise SandboxError(f"NumPy RECORD lacks a hash/size for {relative}")
        source = Path(distribution.locate_file(item))
        if source.is_symlink():
            raise SandboxError(f"NumPy artifact is a symlink: {relative}")
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise SandboxError(f"NumPy artifact is missing: {relative}") from exc
        if not resolved.is_relative_to(distribution_root):
            raise SandboxError(f"NumPy artifact escapes its distribution root: {relative}")
        digest, size = _hash_stable_regular(
            resolved,
            maximum_bytes=MAX_NUMPY_BYTES - total_bytes,
            allow_hardlinks=True,
        )
        if size != item.size:
            raise SandboxError(f"NumPy artifact type/size mismatch: {relative}")
        digest_bytes = bytes.fromhex(digest)
        record_digest = base64.urlsafe_b64encode(digest_bytes).rstrip(b"=").decode()
        if record_digest != item.hash.value:
            raise SandboxError(f"NumPy artifact hash mismatch: {relative}")
        total_bytes += size
        artifacts.append((resolved, relative, digest))
        if len(artifacts) > MAX_NUMPY_ARTIFACTS or total_bytes > MAX_NUMPY_BYTES:
            raise SandboxError("NumPy RECORD exceeds its inventory cap")
    if not artifacts or not any(
        relative == "numpy/__init__.py" for _, relative, _ in artifacts
    ):
        raise SandboxError("NumPy RECORD did not yield a complete runtime package")
    return tuple(sorted(artifacts, key=lambda value: value[1]))


def numpy_manifest_sha256(artifacts: tuple[tuple[Path, str, str], ...]) -> str:
    payload = json.dumps(
        [(relative, digest) for _, relative, digest in artifacts],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_attestation(
    numpy_artifacts: tuple[tuple[Path, str, str], ...] | None = None,
) -> dict:
    """Bind the interpreter, stdlib, NumPy, and sandbox launcher bytes."""

    numpy_artifacts = (
        exact_numpy_artifacts() if numpy_artifacts is None else numpy_artifacts
    )
    executable = Path(sys.executable).resolve(strict=True)
    bwrap = Path("/usr/bin/bwrap").resolve(strict=True)
    executable_sha, executable_size = _hash_stable_regular(
        executable, maximum_bytes=128 * 1024 * 1024
    )
    bwrap_sha, bwrap_size = _hash_stable_regular(
        bwrap, maximum_bytes=128 * 1024 * 1024
    )
    stdlib_root = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    stdlib_artifacts = _walk_runtime_tree(
        stdlib_root,
        prefix="stdlib",
        max_files=MAX_STDLIB_ARTIFACTS,
        max_bytes=MAX_STDLIB_BYTES,
        excluded_directories=frozenset({"__pycache__", "site-packages"}),
    )
    descriptor = {
        "format": RUNTIME_ATTESTATION_FORMAT,
        "python": {
            "cache_tag": sys.implementation.cache_tag,
            "executable": str(executable),
            "sha256": executable_sha,
            "size": executable_size,
            "version": sys.version,
        },
        "stdlib": {
            "root": str(stdlib_root),
            "file_count": len(stdlib_artifacts),
            "manifest_sha256": _canonical_sha256([
                (relative, digest) for _, relative, digest in stdlib_artifacts
            ]),
        },
        "numpy": {
            "version": np.__version__,
            "file_count": len(numpy_artifacts),
            "manifest_sha256": numpy_manifest_sha256(numpy_artifacts),
        },
        "bubblewrap": {
            "path": str(bwrap),
            "sha256": bwrap_sha,
            "size": bwrap_size,
        },
    }
    return {
        "runtime_manifest_sha256": _canonical_sha256(descriptor),
        "runtime": descriptor,
    }


def validate_runtime_attestation(value: dict) -> str:
    """Validate and return the canonical root of one runtime attestation."""

    if (
        not isinstance(value, dict)
        or set(value) != {"runtime_manifest_sha256", "runtime"}
    ):
        raise SandboxError("runtime attestation has the wrong shape")
    root = value.get("runtime_manifest_sha256")
    descriptor = value.get("runtime")
    if (
        not isinstance(root, str)
        or len(root) != 64
        or any(char not in "0123456789abcdef" for char in root)
        or not isinstance(descriptor, dict)
        or set(descriptor) != {
            "format", "python", "stdlib", "numpy", "bubblewrap"
        }
        or descriptor.get("format") != RUNTIME_ATTESTATION_FORMAT
    ):
        raise SandboxError("runtime attestation root/descriptor is malformed")
    expected_keys = {
        "python": {"cache_tag", "executable", "sha256", "size", "version"},
        "stdlib": {"root", "file_count", "manifest_sha256"},
        "numpy": {"version", "file_count", "manifest_sha256"},
        "bubblewrap": {"path", "sha256", "size"},
    }
    for section, keys in expected_keys.items():
        record = descriptor.get(section)
        if not isinstance(record, dict) or set(record) != keys:
            raise SandboxError(f"runtime {section} binding is malformed")
    for path_key in (
        ("python", "executable"),
        ("stdlib", "root"),
        ("bubblewrap", "path"),
    ):
        path = descriptor[path_key[0]].get(path_key[1])
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise SandboxError("runtime attestation contains a non-absolute path")
    for section, key in (
        ("python", "sha256"),
        ("stdlib", "manifest_sha256"),
        ("numpy", "manifest_sha256"),
        ("bubblewrap", "sha256"),
    ):
        digest = descriptor[section].get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise SandboxError("runtime attestation contains a malformed SHA-256")
    for section, key, maximum in (
        ("python", "size", 128 * 1024 * 1024),
        ("bubblewrap", "size", 128 * 1024 * 1024),
        ("stdlib", "file_count", MAX_STDLIB_ARTIFACTS),
        ("numpy", "file_count", MAX_NUMPY_ARTIFACTS),
    ):
        number = descriptor[section].get(key)
        if type(number) is not int or not 1 <= number <= maximum:
            raise SandboxError("runtime attestation contains an invalid bound")
    for section, key in (
        ("python", "cache_tag"),
        ("python", "version"),
        ("numpy", "version"),
    ):
        text = descriptor[section].get(key)
        if not isinstance(text, str) or not 1 <= len(text) <= 1024:
            raise SandboxError("runtime attestation contains invalid version text")
    if _canonical_sha256(descriptor) != root:
        raise SandboxError("runtime attestation root does not match its descriptor")
    return root


def verify_runtime(expected_manifest_sha256: str) -> dict:
    if (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_manifest_sha256)
    ):
        raise SandboxError("expected runtime manifest SHA-256 is malformed")
    current = runtime_attestation()
    validate_runtime_attestation(current)
    if current["runtime_manifest_sha256"] != expected_manifest_sha256:
        raise SandboxError("candidate/trusted runtime drifted after run_start")
    return current


def system_site_directories() -> tuple[Path, ...]:
    """Return existing host Python package roots that must be hidden."""

    directories = []
    for value in site.getsitepackages():
        path = Path(value).resolve()
        if path.exists() and path.is_dir() and path.is_relative_to(Path("/usr")):
            directories.append(path)
    return tuple(sorted(set(directories)))


def _base_command() -> list[str]:
    command = [
        "/usr/bin/bwrap",
        "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--unshare-net", "--disable-userns", "--die-with-parent", "--new-session",
        "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "HOME", "/tmp",
        "--setenv", "PYTHONNOUSERSITE", "1",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "OPENBLAS_NUM_THREADS", "1",
        "--setenv", "OMP_NUM_THREADS", "1",
        "--setenv", "MKL_NUM_THREADS", "1",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--proc", "/proc", "--dev", "/dev", "--remount-ro", "/dev",
        "--size", str(TMPFS_BYTES), "--tmpfs", "/tmp",
    ]
    for path in system_site_directories():
        command += [
            "--size", "4096", "--tmpfs", str(path),
            "--remount-ro", str(path),
        ]
    return command


def check_capability(
    timeout_seconds: float = 15,
    *,
    attestation: dict | None = None,
    numpy_artifacts: tuple[tuple[Path, str, str], ...] | None = None,
) -> dict:
    """Smoke-test the exact mounted NumPy runtime inside all namespaces."""

    if not Path("/usr/bin/bwrap").exists():
        raise SandboxError("bubblewrap (/usr/bin/bwrap) is required")
    if numpy_artifacts is None:
        numpy_artifacts = exact_numpy_artifacts()
    if attestation is None:
        attestation = runtime_attestation(numpy_artifacts)
    validate_runtime_attestation(attestation)
    command = _base_command()
    created = {"/usr", "/lib", "/lib64", "/bin", "/proc", "/dev", "/tmp"}
    for source, relative, _digest in numpy_artifacts:
        destination = f"{NUMPY_SITE}/{relative}"
        command += _mkdir_args(str(Path(destination).parent), created)
        command += ["--ro-bind", str(source), destination]
    command += [
        "--setenv", "PYTHONPATH", NUMPY_SITE,
        "--remount-ro", "/",
        "/usr/bin/python3", "-S", "-c",
        (
            "import os,numpy as np;"
            "assert not os.path.exists('/home');"
            "assert np.__file__.startswith('/opt/numpy-site/numpy/');"
            "a=np.array([[1.0,2.0]]);b=np.array([[3.0],[4.0]]);"
            "assert float(np.dot(a,b)[0,0])==11.0;"
            "print('sandbox-ok')"
        ),
    ]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout_seconds, check=False, env={},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxError(f"sandbox capability probe failed: {exc}") from exc
    if result.returncode != 0 or result.stdout.strip() != "sandbox-ok":
        detail = (result.stderr or result.stdout)[-2000:]
        raise SandboxError(
            "required PID/mount/network isolation is unavailable; official execution "
            f"fails closed ({detail})"
        )
    post_smoke = runtime_attestation()
    if (
        post_smoke["runtime_manifest_sha256"]
        != attestation["runtime_manifest_sha256"]
    ):
        raise SandboxError("candidate runtime changed during capability smoke test")
    return {
        "engine": "bubblewrap",
        "mount_namespace": True,
        "new_pid_namespace": True,
        "network_namespace": True,
        "raw_dataset_mounted": False,
        "parent_repo_mounted": False,
        "runtime_manifest_sha256": attestation["runtime_manifest_sha256"],
        "runtime": attestation["runtime"],
    }


def build_command(
    *,
    candidate_snapshot: Path,
    worker_snapshot: Path,
    kit_snapshot: Path,
    sanitized_snapshot: Path,
    sanitized_names: tuple[str, ...],
    numpy_artifacts: tuple[tuple[Path, str, str], ...],
    output_dir: Path,
    candidate_name: str,
    timeout_seconds: float,
    max_output_bytes: int,
) -> tuple[list[str], str]:
    if type(max_output_bytes) is not int or max_output_bytes < 64 * 1024:
        raise SandboxError("candidate output file limit is invalid")
    _reject_sensitive_broad_bind_overlap((
        (candidate_snapshot, "candidate snapshot"),
        (worker_snapshot, "candidate worker snapshot"),
        (kit_snapshot, "organizer runtime snapshot"),
        (sanitized_snapshot, "candidate data snapshot"),
        (output_dir, "candidate output staging"),
    ))
    sandbox_candidate = f"/repo/Project/solutions/{candidate_name}"
    command = _base_command()
    created = {"/usr", "/lib", "/lib64", "/bin", "/proc", "/dev", "/tmp"}
    for path in (
        "/repo/Project/solutions",
        "/repo/Project/harness",
        "/repo/kuairand-starter-kit/KuaiRand-Pure/data_sanitized",
        "/out",
    ):
        command += _mkdir_args(path, created)
    command += ["--ro-bind", str(candidate_snapshot), sandbox_candidate]
    command += [
        "--ro-bind", str(worker_snapshot), "/repo/Project/harness/candidate_worker.py"
    ]
    for name in SAFE_KIT_FILES:
        command += [
            "--ro-bind", str(kit_snapshot / name), f"/repo/kuairand-starter-kit/{name}"
        ]
    for name in sanitized_names:
        command += [
            "--ro-bind", str(sanitized_snapshot / name),
            f"/repo/kuairand-starter-kit/KuaiRand-Pure/data_sanitized/{name}",
        ]

    for source, relative, _digest in numpy_artifacts:
        destination = f"{NUMPY_SITE}/{relative}"
        command += _mkdir_args(str(Path(destination).parent), created)
        command += ["--ro-bind", str(source), destination]
    command += [
        "--setenv", "PYTHONPATH", f"/repo/kuairand-starter-kit:{NUMPY_SITE}"
    ]
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise SandboxError("candidate output staging directory is unsafe")
    for name in OUTPUT_FILES:
        target = output_dir / name
        metadata = target.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != 0
        ):
            raise SandboxError("candidate output staging file is unsafe")
    # The directory is read-only and contains only three pre-created writable
    # file mounts. Candidate code cannot create an unbounded number of files.
    command += ["--ro-bind", str(output_dir), "/out"]
    for name in OUTPUT_FILES:
        command += ["--bind", str(output_dir / name), f"/out/{name}"]
    # The root becomes read-only. /tmp is a separate bounded tmpfs and the
    # three /out file mounts remain writable submounts.
    command += ["--remount-ro", "/", "--chdir", "/repo"]
    command += [
        "/usr/bin/python3", "-S", "/repo/Project/harness/candidate_worker.py",
        "--candidate", sandbox_candidate,
        "--kit", "/repo/kuairand-starter-kit",
        "--data", "/repo/kuairand-starter-kit/KuaiRand-Pure/data_sanitized",
        "--output-dir", "/out",
        "--cpu-seconds", str(timeout_seconds),
        "--max-output-bytes", str(max_output_bytes),
    ]
    return command, sandbox_candidate


def _read_exact_regular(path: Path, expected_sha256: str, maximum: int) -> bytes:
    """Read one pinned trusted input without following links or races."""

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise SandboxError("trusted input has an invalid expected SHA-256")
    try:
        before = path.lstat()
    except OSError as exc:
        raise SandboxError(f"trusted input is missing: {path}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > maximum
    ):
        raise SandboxError(f"trusted input is not a bounded unique regular file: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SandboxError(f"trusted input cannot be safely opened: {path}") from exc
    try:
        opened = os.fstat(fd)
        if (
            (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
            != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink)
            or opened.st_size > maximum
        ):
            raise SandboxError(f"trusted input changed during safe open: {path}")
        remaining = opened.st_size
        chunks: list[bytes] = []
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                raise SandboxError(f"trusted input ended early: {path}")
            chunks.append(block)
            remaining -= len(block)
        if os.read(fd, 1):
            raise SandboxError(f"trusted input grew while being read: {path}")
        payload = b"".join(chunks)
    finally:
        os.close(fd)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SandboxError(f"trusted input hash mismatch: {path}")
    return payload


def validate_sanitized_snapshot(
    snapshot: Path, expected_sha256: dict[str, str]
) -> tuple[Path, tuple[str, ...], str]:
    """Validate one flat, exact, caller-created sanitized-data snapshot."""

    if not isinstance(expected_sha256, dict) or not expected_sha256:
        raise SandboxError("sanitized snapshot manifest must be non-empty")
    expected: dict[str, str] = {}
    for name, digest in expected_sha256.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {".", ".."}
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise SandboxError("sanitized snapshot manifest has an invalid name/hash")
        expected[name] = digest
    if snapshot.is_symlink():
        raise SandboxError("sanitized snapshot directory cannot be a symlink")
    try:
        resolved = snapshot.resolve(strict=True)
    except OSError as exc:
        raise SandboxError("sanitized snapshot directory is missing") from exc
    if not resolved.is_dir():
        raise SandboxError("sanitized snapshot path is not a directory")
    directory_flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(resolved, directory_flags)
    try:
        names = tuple(sorted(os.listdir(directory_fd)))
        if names != tuple(sorted(expected)):
            raise SandboxError("sanitized snapshot contains missing or unexpected entries")
        frozen: list[tuple[str, str]] = []
        for name in names:
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise SandboxError(
                    f"sanitized snapshot entry is not a unique regular file: {name}"
                )
            flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise SandboxError(
                    f"sanitized snapshot entry cannot be safely opened: {name}"
                ) from exc
            try:
                opened = os.fstat(fd)
                if (
                    (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
                    != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink)
                    or not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                ):
                    raise SandboxError(
                        f"sanitized snapshot entry changed during safe open: {name}"
                    )
                digest_state = hashlib.sha256()
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest_state.update(chunk)
                digest = digest_state.hexdigest()
            finally:
                os.close(fd)
            if digest != expected[name]:
                raise SandboxError(f"sanitized snapshot hash mismatch: {name}")
            frozen.append((name, digest))
    finally:
        os.close(directory_fd)
    manifest_digest = hashlib.sha256(
        json.dumps(frozen, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return resolved, names, manifest_digest


def _read_regular_output(
    directory_fd: int,
    name: str,
    *,
    exact_size: int | None = None,
    maximum_size: int | None = None,
) -> bytes:
    """Read one candidate output without following links or special files."""

    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise SandboxError(f"candidate output is missing: {name}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SandboxError(f"candidate output is not a unique regular file: {name}")
    if exact_size is not None and before.st_size != exact_size:
        raise SandboxError(f"candidate output has the wrong exact size: {name}")
    if maximum_size is not None and before.st_size > maximum_size:
        raise SandboxError(f"candidate output exceeds its size limit: {name}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise SandboxError(f"candidate output cannot be safely opened: {name}") from exc
    try:
        opened = os.fstat(fd)
        identity = (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
        )
        if (
            identity != opened_identity
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise SandboxError(f"candidate output changed during safe open: {name}")
        if exact_size is not None and opened.st_size != exact_size:
            raise SandboxError(f"candidate output changed size during safe open: {name}")
        if maximum_size is not None and opened.st_size > maximum_size:
            raise SandboxError(f"candidate output changed beyond its size limit: {name}")
        remaining = opened.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise SandboxError(f"candidate output ended early: {name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise SandboxError(f"candidate output grew while being read: {name}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_output_packet(
    output_dir: Path, expected_valid_rows: int, expected_test_rows: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Consume exact raw-f64 outputs through safely opened file descriptors."""

    if any(
        type(value) is not int or value < 0
        for value in (expected_valid_rows, expected_test_rows)
    ):
        raise SandboxError("expected output row counts must be nonnegative integers")

    flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(output_dir, flags)
    except OSError as exc:
        raise SandboxError("candidate output directory cannot be safely opened") from exc
    try:
        if set(os.listdir(directory_fd)) != set(OUTPUT_FILES):
            raise SandboxError("candidate output directory has missing or unexpected entries")
        valid_bytes = _read_regular_output(
            directory_fd, "valid.f64", exact_size=8 * expected_valid_rows
        )
        test_bytes = _read_regular_output(
            directory_fd, "test.f64", exact_size=8 * expected_test_rows
        )
        metadata_bytes = _read_regular_output(
            directory_fd, "metadata.json", maximum_size=64 * 1024
        )
    finally:
        os.close(directory_fd)

    valid = np.frombuffer(valid_bytes, dtype="<f8").copy()
    test = np.frombuffer(test_bytes, dtype="<f8").copy()
    if valid.shape != (expected_valid_rows,) or test.shape != (expected_test_rows,):
        raise SandboxError("candidate output packet has incorrect array dimensions")
    if not np.all(np.isfinite(valid)) or not np.all(np.isfinite(test)):
        raise SandboxError("candidate output packet contains NaN/Inf")

    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise SandboxError(f"candidate metadata repeats key {key!r}")
            value[key] = item
        return value

    try:
        metadata = json.loads(
            metadata_bytes.decode("utf-8"), object_pairs_hook=no_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SandboxError(f"candidate metadata is invalid JSON: {exc}") from exc
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"valid_rows", "test_rows"}
        or type(metadata.get("valid_rows")) is not int
        or type(metadata.get("test_rows")) is not int
        or metadata.get("valid_rows") != expected_valid_rows
        or metadata.get("test_rows") != expected_test_rows
    ):
        raise SandboxError("candidate metadata is inconsistent")
    return valid, test, metadata


def run_candidate(
    *,
    root: Path,
    candidate_name: str,
    candidate_bytes: bytes,
    candidate_sha256: str,
    organizer_snapshot: Path,
    organizer_sha256: dict[str, str],
    worker_bytes: bytes,
    worker_sha256: str,
    sanitized_snapshot: Path,
    sanitized_sha256: dict[str, str],
    timeout_seconds: float,
    expected_valid_rows: int,
    expected_test_rows: int,
    expected_runtime_manifest_sha256: str,
) -> dict:
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise SandboxError("repository root is unavailable") from exc
    _reject_sensitive_broad_bind_overlap(((root, "repository root"),))
    if timeout_seconds <= 0:
        raise SandboxError("candidate has no remaining execution time")
    if any(
        type(value) is not int or value < 0
        for value in (expected_valid_rows, expected_test_rows)
    ):
        raise SandboxError("expected row counts must be nonnegative integers")
    if Path(candidate_name).name != candidate_name or not candidate_name.endswith(".py"):
        raise SandboxError("candidate_name must be one plain Python filename")
    if hashlib.sha256(candidate_bytes).hexdigest() != candidate_sha256:
        raise SandboxError("candidate byte snapshot does not match its reviewed SHA-256")
    if hashlib.sha256(worker_bytes).hexdigest() != worker_sha256:
        raise SandboxError("candidate worker bytes do not match their trusted SHA-256")
    if not set(SAFE_KIT_FILES).issubset(organizer_sha256):
        raise SandboxError("organizer snapshot does not bind every required starter file")
    if organizer_snapshot.is_symlink():
        raise SandboxError("organizer snapshot directory cannot be a symlink")
    try:
        organizer_snapshot = organizer_snapshot.resolve(strict=True)
    except OSError as exc:
        raise SandboxError("organizer snapshot directory is missing") from exc
    if organizer_snapshot == (root / "kuairand-starter-kit").resolve():
        raise SandboxError("live organizer files are forbidden; pass the frozen snapshot")
    organizer_bytes = {
        name: _read_exact_regular(
            organizer_snapshot / name, organizer_sha256[name], 2 * 1024 * 1024
        )
        for name in SAFE_KIT_FILES
    }
    sanitized_snapshot, sanitized_names, sanitized_manifest_sha256 = (
        validate_sanitized_snapshot(sanitized_snapshot, sanitized_sha256)
    )
    live_sanitized = (
        root / "kuairand-starter-kit" / "KuaiRand-Pure" / "data_sanitized"
    ).resolve()
    if sanitized_snapshot == live_sanitized:
        raise SandboxError("live sanitized data is forbidden; pass a frozen snapshot")
    numpy_files = exact_numpy_artifacts()
    numpy_digest = numpy_manifest_sha256(numpy_files)
    attestation = runtime_attestation(numpy_files)
    if (
        attestation["runtime_manifest_sha256"]
        != expected_runtime_manifest_sha256
    ):
        raise SandboxError("candidate/trusted runtime drifted after run_start")
    deadline = time.monotonic() + timeout_seconds
    check_capability(
        timeout_seconds=min(15.0, max(0.001, deadline - time.monotonic())),
        attestation=attestation,
        numpy_artifacts=numpy_files,
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("sandbox capability probe consumed the candidate deadline")
    with tempfile.TemporaryDirectory(prefix="track2-candidate-", dir="/tmp") as tmp:
        work = Path(tmp)
        snapshot_dir = work / "snapshot"
        kit_snapshot = snapshot_dir / "kit"
        kit_snapshot.mkdir(parents=True)
        candidate_snapshot = snapshot_dir / candidate_name
        candidate_snapshot.write_bytes(candidate_bytes)
        worker_snapshot = snapshot_dir / "candidate_worker.py"
        worker_snapshot.write_bytes(worker_bytes)
        for name in SAFE_KIT_FILES:
            (kit_snapshot / name).write_bytes(organizer_bytes[name])
        for path in [candidate_snapshot, worker_snapshot, *kit_snapshot.iterdir()]:
            path.chmod(0o444)

        output_dir = work / "output"
        output_dir.mkdir()
        for name in OUTPUT_FILES:
            (output_dir / name).touch(mode=0o600, exist_ok=False)
        command, sandbox_candidate = build_command(
            candidate_snapshot=candidate_snapshot,
            worker_snapshot=worker_snapshot,
            kit_snapshot=kit_snapshot,
            sanitized_snapshot=sanitized_snapshot,
            sanitized_names=sanitized_names,
            numpy_artifacts=numpy_files,
            output_dir=output_dir,
            candidate_name=candidate_name,
            timeout_seconds=max(1.0, deadline - time.monotonic()),
            max_output_bytes=max(
                64 * 1024,
                8 * expected_valid_rows,
                8 * expected_test_rows,
            ),
        )
        log_path = work / "candidate.log"
        process = None
        try:
            with log_path.open("wb") as log_handle:
                process = subprocess.Popen(
                    command, cwd=root, stdin=subprocess.DEVNULL,
                    stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
                    env={},
                )
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            if process.returncode != 0 or time.monotonic() > deadline:
                raise SandboxError("candidate-controlled execution did not succeed")
            valid, test, metadata = read_output_packet(
                output_dir, expected_valid_rows, expected_test_rows
            )
            if time.monotonic() > deadline:
                raise SandboxError("candidate-controlled verification missed deadline")
            verify_runtime(expected_runtime_manifest_sha256)
            if time.monotonic() > deadline:
                raise SandboxError("post-execution runtime verification missed deadline")
            return {
                "valid": valid,
                "test": test,
                "sandbox": {
                    "engine": "bubblewrap",
                    "candidate_path_inside": sandbox_candidate,
                    "raw_dataset_mounted": False,
                    "parent_repo_mounted": False,
                    "new_pid_namespace": True,
                    "network_namespace": True,
                    "reviewed_candidate_sha256": candidate_sha256,
                    "sanitized_snapshot_manifest_sha256": sanitized_manifest_sha256,
                    "numpy_manifest_sha256": numpy_digest,
                    "runtime_manifest_sha256": expected_runtime_manifest_sha256,
                    "worker_sha256": worker_sha256,
                    "organizer_manifest_sha256": hashlib.sha256(
                        json.dumps(
                            [(name, organizer_sha256[name]) for name in SAFE_KIT_FILES],
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                },
            }
        except (KeyboardInterrupt, SystemExit):
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            raise
        except Exception:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            # Exit status, timing class, output length/content, packet shape,
            # and validation error are all candidate-selectable channels.  No
            # distinction crosses back to the researcher.
            raise SandboxError("candidate sandbox execution failed") from None
