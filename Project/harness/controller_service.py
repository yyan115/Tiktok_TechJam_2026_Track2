#!/usr/bin/env python3
"""Privileged outer boundary for the Track 2 researcher.

The researcher process does not receive the repository controller, results,
datasets, manifest, or Git metadata.  It receives a capability to this Unix
socket instead.  The protocol deliberately exposes only:

* ``log`` -- the controller's read-only state summary; and
* ``run`` -- commit exactly one whitelisted solution/card pair and ask the
  frozen controller to consume that attempt.

``start-run`` and ``final`` are intentionally absent.  The service never uses
a shell and always invokes the controller with ``/usr/bin/python3 -I``.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import selectors
import signal
import socket
import socketserver
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_VERSION = 2
ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/usr/bin/python3")
GIT = Path("/usr/bin/git")
GIT_FIXED_OPTIONS = (
    "--no-replace-objects",
    "--no-lazy-fetch",
    "-c", "core.hooksPath=/dev/null",
    "-c", "commit.gpgsign=false",
    "-c", "core.fsync=all",
    "-c", "core.fsyncMethod=fsync",
)
MAX_REQUEST_BYTES = 32 * 1024
MAX_RESPONSE_STREAM_BYTES = 128 * 1024
MAX_PROCESS_STREAM_BYTES = 2 * 1024 * 1024
MAX_AUDIT_BYTES = 64 * 1024 * 1024
MAX_COMPLETION_ROW_BYTES = (
    6 * (2 * MAX_RESPONSE_STREAM_BYTES) + 64 * 1024
)
MAX_CONCURRENT_CONNECTIONS = 8
SOCKET_READ_TIMEOUT_SECONDS = 10.0
LOG_TIMEOUT_SECONDS = 120.0
RUN_TIMEOUT_SECONDS = 7 * 3600.0
MAX_SOLUTION_BYTES = 512 * 1024
MAX_CARD_BYTES = 256 * 1024
MAX_PROTECTED_BYTES = 4 * 1024 * 1024
REQUEST_ID_RE = re.compile(r"[0-9a-f]{32}")
SOLUTION_RE = re.compile(r"Project/solutions/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.py")
CARD_RE = re.compile(
    r"Project/research/attempts/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json"
)
BANK_CATALOG = "Project/research/bank/catalog.json"
BANK_NOTE_RE = re.compile(
    r"Project/research/bank/notes/[a-z0-9][a-z0-9._-]{0,124}\.md"
)
WORKSPACE_SUBDIRS = ("solutions", "attempts", "memory", "scratch")
WORKSPACE_MANIFEST = ".track2-workspace.json"
WORKSPACE_PORTFOLIO = ".track2-portfolio.json"
WORKSPACE_FORMAT = "track2.researcher-workspace.v2"
INHERITED_LOCK_FD_ENV = "TRACK2_CONTROLLER_LOCK_FD"
SERVICE_LOCK_NAME = ".track2-controller.lock"
REPOSITORY_SERVICE_LOCK_NAME = ".controller-service.lock"
SERVICE_BINDING_NAME = "controller-service-binding.json"
SERVICE_BINDING_FORMAT = "track2.controller-service-binding.v1"
MAX_SERVICE_BINDING_BYTES = 16 * 1024
OUTER_ALLOWED_IMPORTS = {
    "__future__", "array", "baseline", "bisect", "collections", "copy", "csv",
    "data", "dataclasses", "decimal", "evaluate", "fractions", "functools",
    "heapq", "itertools", "math", "numpy", "operator", "random", "statistics",
    "typing",
}
OUTER_BLOCKED_CALLS = {
    "breakpoint", "compile", "eval", "exec", "input", "open", "__import__"
}
OUTER_BLOCKED_ATTRIBUTES = {
    "__builtins__", "__code__", "__globals__", "__subclasses__", "_getframe",
}
OUTER_BLOCKED_TEXT = (
    r"KuaiRand-Pure/data(?:/|['\"])",
    r"(?:https?://|ftp://|localhost|127\.0\.0\.1)",
    r"\b(?:subprocess|socket|requests|urllib|httpx|os|sys|inspect|ctypes|multiprocessing|signal|resource|importlib)\b",
    r"/(?:proc|sys|dev)(?:/|['\"])",
)

# This list is deliberately independent of caller input.  The service captures
# these bytes at startup and checks them before/after every transition and Git
# commit.  The inner controller performs its own, more detailed checks too.
PROTECTED_PATHS = (
    "CLAUDE.md",
    "README.md",
    "Project/PLAN.md",
    "Project/RESEARCH_PROTOCOL.md",
    "Project/RUNBOOK.md",
    "Project/RESEARCHER_BRIEF.md",
    "Project/research/templates/attempt.template.json",
    "Project/harness/iterate.py",
    "Project/harness/policy.py",
    "Project/harness/sandbox.py",
    "Project/harness/candidate_worker.py",
    "Project/harness/authority.py",
    "Project/harness/input_snapshot.py",
    "Project/harness/research_bank.py",
    "Project/harness/controller_service.py",
    "Project/harness/researcher_shell.py",
    "Project/harness/controller_mcp_config.json",
    "Project/harness/claude_runtime.json",
    "Project/tools/preflight_review.py",
    "Project/tools/control.py",
    "Project/tools/controller_mcp.py",
    "Project/tools/init_researcher_workspace.py",
    "Project/audits/preflight_schema.json",
    "Project/manifest.json",
    "kuairand-starter-kit/data.py",
    "kuairand-starter-kit/evaluate.py",
    "kuairand-starter-kit/submit.py",
    "kuairand-starter-kit/baseline.py",
    "kuairand-starter-kit/ablation_features.py",
    "kuairand-starter-kit/baseline_scores.json",
    "kuairand-starter-kit/README.md",
)

AUDIT_FORMAT = "track2-controller-wal-v2"


class ServiceError(RuntimeError):
    """Fail-closed request, integrity, or controller-launch error."""


class IndeterminateTransition(ServiceError):
    """A durable outer side effect may have occurred; keep the WAL pending."""


class AuditCapacityError(ServiceError):
    """A WAL request was safely refused before any append was attempted."""


def _load_exact_module(name: str, path: Path):
    """Load one protected helper by absolute path, never import search order."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ServiceError(f"cannot load protected module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise ServiceError(f"protected module identity mismatch: {path.name}")
    return module


outer_policy = _load_exact_module(
    "track2_outer_policy", ROOT / "Project" / "harness" / "policy.py"
)
MAX_AUDIT_REQUESTS = outer_policy.MAX_OUTER_RUN_REQUESTS
outer_research_bank = _load_exact_module(
    "track2_outer_research_bank",
    ROOT / "Project" / "harness" / "research_bank.py",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _materialization_temp_leaf(relative: str, payload_sha256: str) -> str:
    """Return the one reserved same-directory staging name for an artifact."""

    pure = PurePosixPath(relative)
    if (
        len(pure.parts) < 2
        or any(part in ("", ".", "..") for part in pure.parts)
        or re.fullmatch(r"[0-9a-f]{64}", payload_sha256) is None
    ):
        raise ServiceError("materialization staging identity is malformed")
    leaf = f".{pure.name}.track2-materializing-{payload_sha256[:32]}"
    if len(os.fsencode(leaf)) > 255:
        raise ServiceError("materialization staging name is too long")
    return leaf


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    """Atomically publish one same-directory file without replacing a path."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise ServiceError(
            "atomic no-replace publication is unavailable on this host"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        directory_fd,
        os.fsencode(source),
        directory_fd,
        os.fsencode(destination),
        1,  # Linux RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise ServiceError("artifact destination appeared during publication")
    raise ServiceError(
        "atomic artifact publication failed: "
        f"{os.strerror(error_number) if error_number else 'unknown error'}"
    )


def _fallocate_keep_size(fd: int, offset: int, length: int) -> None:
    """Durably reserve Linux filesystem blocks without extending logical EOF."""

    if offset < 0 or length <= 0:
        raise AuditCapacityError("physical WAL reservation bounds are invalid")
    try:
        fallocate = ctypes.CDLL(None, use_errno=True).fallocate
    except (AttributeError, OSError) as exc:
        raise AuditCapacityError(
            "this host lacks fallocate KEEP_SIZE for durable WAL reservation"
        ) from exc
    fallocate.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
    )
    fallocate.restype = ctypes.c_int
    result = fallocate(fd, 1, offset, length)  # Linux FALLOC_FL_KEEP_SIZE
    if result == 0:
        return
    error_number = ctypes.get_errno()
    raise AuditCapacityError(
        "physical WAL completion reservation failed: "
        f"{os.strerror(error_number) if error_number else 'unknown error'}"
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ServiceError(f"value is not canonical finite JSON: {exc}") from exc


def _admission_binding(
    admission: tuple[str, str, bytes, bytes, str] | dict[str, Any]
) -> dict[str, Any]:
    """Return the exact durable intent bound to one consuming request."""

    if isinstance(admission, tuple) and len(admission) == 5:
        solution_sha, card_sha, solution_bytes, card_bytes, parent_revision = admission
        value = {
            "solution_sha256": solution_sha,
            "solution_size": len(solution_bytes) if isinstance(solution_bytes, bytes) else -1,
            "card_sha256": card_sha,
            "card_size": len(card_bytes) if isinstance(card_bytes, bytes) else -1,
            "parent_revision": parent_revision,
        }
        if (
            not isinstance(solution_bytes, bytes)
            or not isinstance(card_bytes, bytes)
            or _sha256(solution_bytes) != solution_sha
            or _sha256(card_bytes) != card_sha
        ):
            raise ServiceError("outer admission bytes do not match their hashes")
    elif isinstance(admission, dict):
        value = dict(admission)
    else:
        raise ServiceError("consuming request lacks an exact admission binding")
    if (
        set(value) != {
            "solution_sha256", "solution_size", "card_sha256", "card_size",
            "parent_revision",
        }
        or re.fullmatch(r"[0-9a-f]{64}", value.get("solution_sha256", "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", value.get("card_sha256", "")) is None
        or re.fullmatch(r"[0-9a-f]{40}", value.get("parent_revision", "")) is None
        or type(value.get("solution_size")) is not int
        or not 1 <= value["solution_size"] <= MAX_SOLUTION_BYTES
        or type(value.get("card_size")) is not int
        or not 1 <= value["card_size"] <= MAX_CARD_BYTES
    ):
        raise ServiceError("consuming request admission binding is malformed")
    return value


def _strict_object(data: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ServiceError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def no_constants(value):
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            data, object_pairs_hook=no_duplicates, parse_constant=no_constants
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ServiceError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ServiceError(f"{label} must be one JSON object")
    return value


def _validate_relative(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ServiceError(f"{label} is outside its fixed path allowlist")
    if "\\" in value or "\x00" in value or "//" in value:
        raise ServiceError(f"{label} contains a forbidden path spelling")
    return value


def validate_request(value: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized request and reject every non-protocol field."""

    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ServiceError("unsupported protocol_version")
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or REQUEST_ID_RE.fullmatch(request_id) is None:
        raise ServiceError("request_id must be 32 lowercase hexadecimal characters")
    workspace_id = value.get("workspace_id")
    if not isinstance(workspace_id, str) or REQUEST_ID_RE.fullmatch(workspace_id) is None:
        raise ServiceError("workspace_id must be 32 lowercase hexadecimal characters")
    workspace_binding = value.get("workspace_binding")
    if not isinstance(workspace_binding, str) or re.fullmatch(
        r"[0-9a-f]{64}", workspace_binding
    ) is None:
        raise ServiceError("workspace_binding must be 64 lowercase hexadecimal characters")
    command = value.get("command")
    if command == "log":
        if set(value) != {
            "protocol_version", "request_id", "workspace_id",
            "workspace_binding", "command",
        }:
            raise ServiceError("log request has missing or extra fields")
        return dict(value)
    if command == "run":
        if set(value) != {
            "protocol_version", "request_id", "workspace_id", "command",
            "workspace_binding", "solution", "card",
        }:
            raise ServiceError("run request has missing or extra fields")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "workspace_id": workspace_id,
            "workspace_binding": workspace_binding,
            "command": "run",
            "solution": _validate_relative(value.get("solution"), SOLUTION_RE, "solution"),
            "card": _validate_relative(value.get("card"), CARD_RE, "card"),
        }
    raise ServiceError("only run and log RPCs exist")


def _run_process(
    command: list[str], *, root: Path, timeout_seconds: float, env: dict[str, str],
    input_bytes: bytes | None = None,
    pass_fds: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[str]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
        or not isinstance(command, list)
        or not command
        or not isinstance(pass_fds, tuple)
        or any(type(fd) is not int or fd < 0 for fd in pass_fds)
    ):
        raise ServiceError("fixed process launch has invalid bounds")
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=env,
            start_new_session=True,
            pass_fds=pass_fds,
        )
    except OSError as exc:
        raise ServiceError(f"fixed process launch failed: {type(exc).__name__}: {exc}") from exc
    if input_bytes is not None:
        if len(input_bytes) > MAX_PROCESS_STREAM_BYTES:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise ServiceError("fixed process stdin exceeds its size limit")
        assert process.stdin is not None
        try:
            process.stdin.write(input_bytes)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    assert process.stdout is not None and process.stderr is not None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    streams = {
        process.stdout.fileno(): ("stdout", stdout_buffer),
        process.stderr.fileno(): ("stderr", stderr_buffer),
    }
    selector = selectors.DefaultSelector()
    for fd in streams:
        os.set_blocking(fd, False)
        selector.register(fd, selectors.EVENT_READ)
    deadline = time.monotonic() + float(timeout_seconds)
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("fixed process deadline elapsed")
            events = selector.select(min(remaining, 0.25))
            if not events and process.poll() is not None:
                # Pipes may still contain buffered bytes; keep selecting until
                # EOF rather than treating process exit as stream completion.
                continue
            for key, _mask in events:
                fd = key.fd
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(fd)
                    continue
                name, buffer = streams[fd]
                buffer.extend(chunk)
                if len(buffer) > MAX_PROCESS_STREAM_BYTES:
                    raise ServiceError(f"fixed process {name} exceeds its size limit")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fixed process deadline elapsed")
        returncode = process.wait(timeout=remaining)
    except (ServiceError, TimeoutError, subprocess.TimeoutExpired) as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        if isinstance(exc, ServiceError):
            raise
        raise ServiceError(f"fixed process failed: {type(exc).__name__}: {exc}") from exc
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=bytes(stdout_buffer).decode("utf-8", errors="replace"),
        stderr=bytes(stderr_buffer).decode("utf-8", errors="replace"),
    )


def fixed_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Build a small owner-supplied environment, excluding Python/loader overrides."""

    source = os.environ if source is None else source
    result = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "NO_COLOR": "1",
        "TERM": "dumb",
        # Fixed Git commands must neither inherit owner aliases/configuration
        # nor contact a promisor remote to fill a missing object.
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    # These values belong to the owner process that starts the service, never a
    # researcher RPC.  Endpoint/provider override variables are not inherited.
    for name in (
        "HOME", "CODEX_HOME", "OPENAI_API_KEY", "LANG", "LC_ALL", "TZ",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy",
    ):
        value = source.get(name)
        if value:
            result[name] = value
    return result


def _git_command(args: list[str]) -> list[str]:
    """Build one owner-fixed Git invocation with durable write semantics."""

    return [str(GIT), *GIT_FIXED_OPTIONS, *args]


def _git(
    root: Path,
    args: list[str],
    *,
    env: dict[str, str],
    timeout_seconds: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    command = _git_command(args)
    result = _run_process(command, root=root, timeout_seconds=timeout_seconds, env=env)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-4000:]
        raise ServiceError(f"fixed Git command failed ({args[0]}): {detail}")
    return result


def _git_blob(root: Path, relative: str, env: dict[str, str]) -> bytes:
    command = _git_command(["show", f"HEAD:{relative}"])
    result = _run_process(
        command, root=root, timeout_seconds=60.0, env=env
    )
    if result.returncode != 0:
        raise ServiceError(f"protected or requested path is absent from HEAD: {relative}")
    try:
        return result.stdout.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ServiceError(f"protected path is not exact UTF-8 text: {relative}") from exc


def _validate_git_repository(root: Path, env: dict[str, str]) -> None:
    """Reject repository features that can substitute or fetch trusted objects."""

    common_raw = _git(
        root,
        ["rev-parse", "--path-format=absolute", "--git-common-dir"],
        env=env,
    ).stdout.strip()
    if not common_raw or "\x00" in common_raw or "\n" in common_raw:
        raise ServiceError("Git common directory is malformed")
    common_unresolved = Path(common_raw)
    if not common_unresolved.is_absolute():
        raise ServiceError("Git common directory is not absolute")
    try:
        common_metadata = common_unresolved.lstat()
        common = common_unresolved.resolve(strict=True)
    except OSError as exc:
        raise ServiceError("Git common directory is unavailable") from exc
    if (
        stat.S_ISLNK(common_metadata.st_mode)
        or not stat.S_ISDIR(common_metadata.st_mode)
        or common != common_unresolved
    ):
        raise ServiceError("Git common directory is unsafe")
    expected_unresolved = root / ".git"
    try:
        expected_metadata = expected_unresolved.lstat()
        expected = expected_unresolved.resolve(strict=True)
    except OSError as exc:
        raise ServiceError(
            "official controller requires the repository's primary checkout"
        ) from exc
    if (
        stat.S_ISLNK(expected_metadata.st_mode)
        or not stat.S_ISDIR(expected_metadata.st_mode)
        or expected != expected_unresolved
        or common != expected
    ):
        raise ServiceError(
            "official controller requires a primary checkout with its Git common "
            "directory at repository/.git"
        )

    # Replacement refs are disabled on every command, but their presence is
    # still an integrity smell and therefore refused explicitly.
    replacements = _git(
        root, ["for-each-ref", "--format=%(refname)", "refs/replace"], env=env
    ).stdout.strip()
    if replacements:
        raise ServiceError("Git replacement refs are forbidden")

    grafts = common / "info" / "grafts"
    try:
        grafts.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ServiceError("legacy Git graft state is unreadable") from exc
    else:
        raise ServiceError("legacy Git grafts are forbidden")

    for relative in ("objects/info/alternates", "objects/info/http-alternates"):
        alternate = common / relative
        try:
            alternate.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ServiceError("Git alternate-object state is unreadable") from exc
        else:
            raise ServiceError("Git alternate object stores are forbidden")

    partial = _run_process(
        _git_command([
            "config", "--get-regexp",
            r"^(extensions\.partialclone|remote\..*\.(promisor|partialclonefilter))$",
        ]),
        root=root,
        timeout_seconds=60,
        env=env,
    )
    if partial.returncode == 0:
        raise ServiceError("partial/promisor Git repositories are forbidden")
    if partial.returncode != 1:
        raise ServiceError("Git partial-clone configuration could not be verified")


def _regular_repo_file(
    root: Path, relative: str, *, maximum_bytes: int | None = None
) -> tuple[Path, bytes]:
    maximum_bytes = MAX_PROTECTED_BYTES if maximum_bytes is None else maximum_bytes
    pure = PurePosixPath(relative)
    if (
        not isinstance(relative, str)
        or not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ServiceError("requested artifact path is unsafe")
    path = root / relative
    root_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    )
    directory_fd = root_fd
    try:
        for part in pure.parts[:-1]:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            next_fd = os.open(part, flags, dir_fd=directory_fd)
            if directory_fd != root_fd:
                os.close(directory_fd)
            directory_fd = next_fd
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NONBLOCK", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_fd = os.open(pure.parts[-1], flags, dir_fd=directory_fd)
        try:
            metadata = os.fstat(file_fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > maximum_bytes
            ):
                raise ServiceError(
                    f"requested artifact is not a bounded unique regular file: {relative}"
                )
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(file_fd, min(65536, maximum_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum_bytes:
                    raise ServiceError(
                        f"requested artifact exceeds {maximum_bytes} bytes: {relative}"
                    )
                chunks.append(chunk)
            after = os.fstat(file_fd)
            if (
                (after.st_dev, after.st_ino, after.st_size, after.st_nlink)
                != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_nlink)
                or size != metadata.st_size
            ):
                raise ServiceError(f"requested artifact changed while read: {relative}")
            return path, b"".join(chunks)
        finally:
            os.close(file_fd)
    except OSError as exc:
        raise ServiceError(f"requested artifact is unavailable or unsafe: {relative}") from exc
    finally:
        if directory_fd != root_fd:
            os.close(directory_fd)
        os.close(root_fd)


def _private_workspace_root(path: Path) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(os.fspath(path))) != path:
        raise ServiceError("researcher workspace must use a canonical absolute path")
    if path == Path(path.anchor):
        raise ServiceError("researcher workspace may not be the filesystem root")
    effective_uid = os.geteuid()
    current = Path(path.anchor)
    root_metadata = current.lstat()
    root_uid = root_metadata.st_uid
    for index, part in enumerate(path.parts[1:]):
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ServiceError(f"researcher workspace path is unavailable: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ServiceError("researcher workspace path must contain only real directories")
        if metadata.st_uid not in {root_uid, effective_uid}:
            raise ServiceError("researcher workspace path has an untrusted owner")
        if metadata.st_mode & 0o022:
            sticky_root_boundary = (
                metadata.st_uid == root_uid and bool(metadata.st_mode & stat.S_ISVTX)
            )
            if not sticky_root_boundary:
                raise ServiceError("researcher workspace has a replaceable ancestor")
        if index == len(path.parts[1:]) - 1 and (
            metadata.st_uid != effective_uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ServiceError("researcher workspace must be owner-controlled mode 0700")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ServiceError("researcher workspace changed while validating")
    return resolved


def _read_workspace_root_file(
    root: Path, name: str, *, maximum_bytes: int, label: str
) -> bytes:
    root_fd = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        flags = (
            os.O_RDONLY | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        fd = os.open(name, flags, dir_fd=root_fd)
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or before.st_mode & 0o022
                or not 1 <= before.st_size <= maximum_bytes
            ):
                raise ServiceError(f"{label} is unsafe")
            def read_exact() -> bytes:
                payload = bytearray()
                while len(payload) <= maximum_bytes:
                    chunk = os.read(
                        fd, min(65536, maximum_bytes + 1 - len(payload))
                    )
                    if not chunk:
                        break
                    payload.extend(chunk)
                return bytes(payload)

            first = read_exact()
            os.lseek(fd, 0, os.SEEK_SET)
            second = read_exact()
            after = os.fstat(fd)
            identity_before = (
                before.st_dev, before.st_ino, before.st_size, before.st_nlink,
                before.st_mtime_ns, before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev, after.st_ino, after.st_size, after.st_nlink,
                after.st_mtime_ns, after.st_ctime_ns,
            )
            if (
                first != second
                or len(first) != before.st_size
                or len(first) > maximum_bytes
                or identity_after != identity_before
            ):
                raise ServiceError(f"{label} changed while being read")
            return first
        finally:
            os.close(fd)
    except OSError as exc:
        raise ServiceError(f"{label} is unavailable") from exc
    finally:
        os.close(root_fd)


def _workspace_binding_value(root: Path, manifest_payload: bytes) -> str:
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or root.resolve(strict=True) != root:
        raise ServiceError("researcher workspace identity is unsafe")
    value = {
        "format": "track2.workspace-physical-binding.v1",
        "canonical_path": str(root),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "manifest_sha256": _sha256(manifest_payload),
    }
    return _sha256(_canonical_bytes(value))


class ArtifactWorkspace:
    """Read exact model-authored artifacts from a clean external workspace."""

    def __init__(self, root: Path):
        self.root = _private_workspace_root(root)
        root_metadata = self.root.lstat()
        self.root_identity = (root_metadata.st_dev, root_metadata.st_ino)
        try:
            entries = {entry.name: entry for entry in os.scandir(self.root)}
        except OSError as exc:
            raise ServiceError("researcher workspace cannot be enumerated") from exc
        if set(entries) != set(WORKSPACE_SUBDIRS) | {
            WORKSPACE_MANIFEST, WORKSPACE_PORTFOLIO
        }:
            raise ServiceError(
                "researcher workspace must contain exactly solutions, attempts, "
                "memory, scratch, the run-binding manifest, and frozen portfolio"
            )
        manifest_payload = _read_workspace_root_file(
            self.root, WORKSPACE_MANIFEST, maximum_bytes=4096,
            label="researcher workspace run-binding manifest",
        )
        portfolio_payload = _read_workspace_root_file(
            self.root, WORKSPACE_PORTFOLIO,
            maximum_bytes=outer_policy.MAX_PORTFOLIO_VIEW_BYTES,
            label="researcher workspace frozen portfolio",
        )
        try:
            manifest = _strict_object(
                manifest_payload, "researcher workspace manifest"
            )
            portfolio = _strict_object(
                portfolio_payload, "researcher workspace frozen portfolio"
            )
        except (OSError, ServiceError) as exc:
            raise ServiceError("researcher workspace run-binding manifest is unreadable") from exc
        if (
            set(manifest) != {
                "format", "workspace_id", "run_id", "repository_head",
                "portfolio_sha256", "created_ns",
            }
            or manifest.get("format") != WORKSPACE_FORMAT
            or not isinstance(manifest.get("workspace_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", manifest["workspace_id"]) is None
            or not isinstance(manifest.get("run_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", manifest["run_id"]) is None
            or not isinstance(manifest.get("repository_head"), str)
            or re.fullmatch(r"[0-9a-f]{40}", manifest["repository_head"]) is None
            or type(manifest.get("created_ns")) is not int
            or manifest["created_ns"] <= 0
            or not isinstance(manifest.get("portfolio_sha256"), str)
            or manifest["portfolio_sha256"] != _sha256(portfolio_payload)
            or not isinstance(portfolio, dict)
        ):
            raise ServiceError("researcher workspace run-binding manifest has wrong shape")
        try:
            outer_policy.validate_portfolio(portfolio)
        except Exception as exc:
            raise ServiceError("researcher workspace frozen portfolio is invalid") from exc
        self.workspace_id = manifest["workspace_id"]
        self.binding = _workspace_binding_value(self.root, manifest_payload)
        self.run_id = manifest["run_id"]
        self.repository_head = manifest["repository_head"]
        self.portfolio_sha256 = manifest["portfolio_sha256"]
        self.portfolio = portfolio
        for name in WORKSPACE_SUBDIRS:
            metadata = entries[name].stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise ServiceError(
                    f"researcher workspace {name} must be a real owner-only directory"
                )

    def read(self, relative: str, *, card: bool) -> bytes:
        pattern = CARD_RE if card else SOLUTION_RE
        label = "card" if card else "solution"
        relative = _validate_relative(relative, pattern, label)
        subdir = "attempts" if card else "solutions"
        filename = PurePosixPath(relative).name
        maximum = MAX_CARD_BYTES if card else MAX_SOLUTION_BYTES
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        directory_fd = None
        try:
            opened_root = os.fstat(root_fd)
            if (opened_root.st_dev, opened_root.st_ino) != self.root_identity:
                raise ServiceError("researcher workspace root identity changed")
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            directory_fd = os.open(subdir, flags, dir_fd=root_fd)
            file_flags = (
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NONBLOCK", 0)
            )
            if hasattr(os, "O_NOFOLLOW"):
                file_flags |= os.O_NOFOLLOW
            file_fd = os.open(filename, file_flags, dir_fd=directory_fd)
            try:
                metadata = os.fstat(file_fd)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or metadata.st_mode & 0o022
                    or not 1 <= metadata.st_size <= maximum
                ):
                    raise ServiceError(
                        f"staged {label} is not a bounded private unique regular file"
                    )
                def read_exact() -> bytes:
                    payload = bytearray()
                    while len(payload) <= maximum:
                        chunk = os.read(
                            file_fd, min(65536, maximum + 1 - len(payload))
                        )
                        if not chunk:
                            break
                        payload.extend(chunk)
                    return bytes(payload)

                first = read_exact()
                os.lseek(file_fd, 0, os.SEEK_SET)
                second = read_exact()
                after = os.fstat(file_fd)
                identity_before = (
                    metadata.st_dev, metadata.st_ino, metadata.st_size,
                    metadata.st_nlink, metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
                identity_after = (
                    after.st_dev, after.st_ino, after.st_size, after.st_nlink,
                    after.st_mtime_ns, after.st_ctime_ns,
                )
                if (
                    first != second
                    or len(first) != metadata.st_size
                    or len(first) > maximum
                    or identity_after != identity_before
                ):
                    raise ServiceError(f"staged {label} changed while being read")
                return first
            finally:
                os.close(file_fd)
        except OSError as exc:
            raise ServiceError(f"staged {label} is unavailable or unsafe") from exc
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
            os.close(root_fd)


def _outer_candidate_findings(source: str) -> list[str]:
    findings: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in OUTER_ALLOWED_IMPORTS:
                    findings.append("import outside allowlist")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in OUTER_ALLOWED_IMPORTS:
                findings.append("import outside allowlist")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in OUTER_BLOCKED_CALLS:
                findings.append("dynamic execution call")
        elif isinstance(node, ast.Attribute) and node.attr in OUTER_BLOCKED_ATTRIBUTES:
            findings.append("blocked introspection attribute")
    if any(re.search(pattern, source, re.IGNORECASE) for pattern in OUTER_BLOCKED_TEXT):
        findings.append("blocked source reference")
    return sorted(set(findings))


def _outer_candidate_fingerprint(source: str) -> str:
    tree = ast.parse(source)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return _sha256(normalized.encode("utf-8"))


def _outer_resolve_attempt_research(
    bank: Any,
    card: dict[str, Any],
    portfolio: dict[str, Any],
    registrations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Mirror the inner controller's deterministic bank/topic checks."""

    resolved = bank.resolve_basis(
        card["research_basis"],
        allowed_relationships=outer_policy.RESEARCH_RELATIONSHIPS,
        allowed_targets=outer_policy.ATTEMPT_RESEARCH_TARGETS,
    )
    cited_topics = {
        topic
        for citation in resolved["citations"]
        for topic in citation["topics"]
    }
    seed = next(
        (
            family
            for family in portfolio["families"]
            if family["family_id"] == card["family_id"]
        ),
        None,
    )
    if seed is not None:
        applicable_topics = set(seed["bank_topics"])
    else:
        registration = registrations.get(card["family_id"])
        extension = (
            registration["extension"]
            if registration is not None
            else card["family_extension"]
        )
        applicable_topics = set(extension["bank_topics"])
    if not applicable_topics.intersection(cited_topics):
        raise ServiceError(
            "attempt citations share no topic with their applicable mechanism family"
        )
    if seed is None and not applicable_topics.issubset(cited_topics):
        raise ServiceError(
            "new-family bank topics are not fully covered by attempt citations"
        )
    return resolved


def _validate_controller_commit_chain(
    root: Path, base: str, current: str, env: dict[str, str]
) -> None:
    """Accept only the initial workspace HEAD plus controller-authored commits."""

    if re.fullmatch(r"[0-9a-f]{40}", base) is None or re.fullmatch(
        r"[0-9a-f]{40}", current
    ) is None:
        raise ServiceError("workspace Git binding is malformed")
    ancestor = _run_process(
        _git_command(["merge-base", "--is-ancestor", base, current]),
        root=root,
        timeout_seconds=60,
        env=env,
    )
    if ancestor.returncode != 0:
        raise ServiceError("workspace base is not an ancestor of current HEAD")
    revisions_raw = _git(
        root, ["rev-list", "--reverse", f"{base}..{current}"], env=env
    ).stdout.splitlines()
    if len(revisions_raw) > MAX_AUDIT_REQUESTS:
        raise ServiceError("controller-authored descendant chain is unexpectedly long")
    previous = base
    message_re = re.compile(
        r"track2 official attempt artifacts "
        r"solution=([0-9a-f]{64}) card=([0-9a-f]{64})"
    )
    for revision in revisions_raw:
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ServiceError("controller descendant revision is malformed")
        parents = _git(
            root, ["rev-list", "--parents", "-n", "1", revision], env=env
        ).stdout.strip().split()
        if parents != [revision, previous]:
            raise ServiceError("workspace descendant chain is not linear")
        message = _git(root, ["show", "-s", "--format=%B", revision], env=env).stdout
        match = message_re.fullmatch(message.rstrip("\n"))
        if match is None:
            raise ServiceError("workspace descendant is not controller-authored")
        changes = _git(
            root,
            ["diff-tree", "--no-commit-id", "--name-status", "-r", revision],
            env=env,
        ).stdout.splitlines()
        if len(changes) != 2 or any(not row.startswith("A\t") for row in changes):
            raise ServiceError("controller descendant has an unexpected tree delta")
        names = {row[2:] for row in changes}
        solutions = [name for name in names if SOLUTION_RE.fullmatch(name)]
        cards = [name for name in names if CARD_RE.fullmatch(name)]
        if len(solutions) != 1 or len(cards) != 1:
            raise ServiceError("controller descendant has unexpected artifact paths")
        if (
            _sha256(_git_blob(root, solutions[0], env)) != match.group(1)
            or _sha256(_git_blob(root, cards[0], env)) != match.group(2)
        ):
            raise ServiceError("controller descendant message/hash binding is invalid")
        previous = revision
    if previous != current:
        raise ServiceError("workspace descendant chain does not reach current HEAD")


def frozen_bank_paths(root: Path, env: dict[str, str]) -> tuple[str, ...]:
    """Resolve the controller-owned bank allowlist once at service startup."""

    _path, payload = _regular_repo_file(
        root, BANK_CATALOG, maximum_bytes=256 * 1024
    )
    if _git_blob(root, BANK_CATALOG, env) != payload:
        raise ServiceError("research bank catalog differs from committed HEAD")
    catalog = _strict_object(payload, "research bank catalog")
    if (
        set(catalog) != {"schema_version", "benchmark", "claims"}
        or type(catalog.get("schema_version")) is not int
        or catalog.get("schema_version") != 1
        or catalog.get("benchmark") != "KuaiRand-Pure"
        or not isinstance(catalog.get("claims"), list)
        or not 1 <= len(catalog["claims"]) <= 256
    ):
        raise ServiceError("research bank catalog has an invalid fixed schema")
    notes: set[str] = set()
    for claim in catalog["claims"]:
        note = claim.get("note_path") if isinstance(claim, dict) else None
        if not isinstance(note, str) or BANK_NOTE_RE.fullmatch(note) is None:
            raise ServiceError("research bank catalog has an unsafe note path")
        notes.add(note)
    if not 1 <= len(notes) <= 64:
        raise ServiceError("research bank catalog selects an invalid note count")
    return (BANK_CATALOG, *sorted(notes))


class ProtectedState:
    """Startup snapshot that detects protected drift or module substitution."""

    def __init__(
        self,
        root: Path,
        *,
        paths: tuple[str, ...] = PROTECTED_PATHS,
        env: dict[str, str] | None = None,
    ):
        self.root = root.resolve(strict=True)
        self.paths = tuple(paths)
        self.env = fixed_environment() if env is None else dict(env)
        self.hashes = self._current(require_head_match=True)

    def _current(self, *, require_head_match: bool) -> dict[str, str]:
        result: dict[str, str] = {}
        for relative in self.paths:
            path, data = _regular_repo_file(self.root, relative)
            if require_head_match and _git_blob(self.root, relative, self.env) != data:
                raise ServiceError(f"protected file differs from committed HEAD: {relative}")
            result[relative] = _sha256(data)
            if path.is_symlink():  # defensive; lstat above already refuses it
                raise ServiceError(f"protected file is a symlink: {relative}")
        return result

    def verify(self) -> None:
        current = self._current(require_head_match=True)
        if current != self.hashes:
            changed = sorted(set(current) | set(self.hashes))
            changed = [name for name in changed if current.get(name) != self.hashes.get(name)]
            raise ServiceError(f"protected-component drift since service startup: {changed}")


def _nul_names(payload: str) -> list[str]:
    return [name for name in payload.split("\0") if name]


class ArtifactCommitter:
    """Build one exact commit from captured bytes without using the live index."""

    def __init__(self, root: Path, protected: ProtectedState, env: dict[str, str]):
        self.root = root.resolve(strict=True)
        self.protected = protected
        self.env = dict(env)

    @staticmethod
    def _write_private(path: Path, payload: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise ServiceError("private Git object input write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _hash_exact_file(self, path: Path, payload: bytes, env: dict[str, str]) -> str:
        result = _git(
            self.root,
            ["hash-object", "-w", "--no-filters", "--", str(path)],
            env=env,
        )
        oid = result.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", oid) is None:
            raise ServiceError("Git returned a malformed blob object ID")
        check = _run_process(
            _git_command(["cat-file", "blob", oid]),
            root=self.root,
            timeout_seconds=60,
            env=env,
        )
        if check.returncode != 0 or check.stdout.encode("utf-8") != payload:
            raise ServiceError("Git blob bytes differ from the captured artifact")
        return oid

    def _tree_blob(self, tree: str, relative: str, env: dict[str, str]) -> str:
        result = _git(
            self.root, ["ls-tree", tree, "--", relative], env=env
        ).stdout.strip()
        match = re.fullmatch(
            r"100644 blob ([0-9a-f]{40})\t" + re.escape(relative), result
        )
        if match is None:
            raise ServiceError(f"private Git tree did not bind exact path: {relative}")
        return match.group(1)

    def commit_pair(
        self,
        solution: str,
        card: str,
        *,
        expected_solution_sha256: str | None = None,
        expected_card_sha256: str | None = None,
        captured_solution_bytes: bytes | None = None,
        captured_card_bytes: bytes | None = None,
        materialize_new_worktree_files: bool = False,
        expected_parent_revision: str | None = None,
    ) -> dict[str, str]:
        solution = _validate_relative(solution, SOLUTION_RE, "solution")
        card = _validate_relative(card, CARD_RE, "card")
        self.protected.verify()
        if (captured_solution_bytes is None) != (captured_card_bytes is None):
            raise ServiceError("captured artifact pair must be complete")
        if captured_solution_bytes is None:
            _solution_path, solution_bytes = _regular_repo_file(
                self.root, solution, maximum_bytes=MAX_SOLUTION_BYTES
            )
            _card_path, card_bytes = _regular_repo_file(
                self.root, card, maximum_bytes=MAX_CARD_BYTES
            )
        else:
            solution_bytes = captured_solution_bytes
            card_bytes = captured_card_bytes
            if (
                not isinstance(solution_bytes, bytes)
                or not 1 <= len(solution_bytes) <= MAX_SOLUTION_BYTES
                or not isinstance(card_bytes, bytes)
                or not 1 <= len(card_bytes) <= MAX_CARD_BYTES
            ):
                raise ServiceError("captured artifact pair has invalid byte bounds")
        solution_sha = _sha256(solution_bytes)
        card_sha = _sha256(card_bytes)
        if (
            expected_solution_sha256 is not None
            and solution_sha != expected_solution_sha256
        ) or (
            expected_card_sha256 is not None and card_sha != expected_card_sha256
        ):
            raise ServiceError("attempt artifacts changed after outer admission")

        old_revision = _git(
            self.root, ["rev-parse", "HEAD"], env=self.env
        ).stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", old_revision) is None:
            raise ServiceError("repository HEAD is malformed")
        if (
            expected_parent_revision is not None
            and old_revision != expected_parent_revision
        ):
            raise ServiceError("repository HEAD changed after outer admission")
        if materialize_new_worktree_files:
            for relative in (solution, card):
                tracked = _git(
                    self.root, ["ls-tree", old_revision, "--", relative], env=self.env
                ).stdout
                if tracked:
                    raise ServiceError(
                        f"staged artifact destination already exists in HEAD: {relative}"
                    )
                try:
                    (self.root / relative).lstat()
                except FileNotFoundError:
                    pass
                else:
                    raise ServiceError(
                        f"staged artifact destination already exists in worktree: {relative}"
                    )
        with tempfile.TemporaryDirectory(
            prefix="track2-git-object-", dir="/tmp"
        ) as temporary:
            private = Path(temporary)
            os.chmod(private, 0o700)
            solution_input = private / "solution.bin"
            card_input = private / "card.bin"
            self._write_private(solution_input, solution_bytes)
            self._write_private(card_input, card_bytes)
            private_env = dict(self.env)
            private_env["GIT_INDEX_FILE"] = str(private / "index")
            solution_oid = self._hash_exact_file(
                solution_input, solution_bytes, private_env
            )
            card_oid = self._hash_exact_file(card_input, card_bytes, private_env)
            _git(self.root, ["read-tree", old_revision], env=private_env)
            for relative, oid in ((solution, solution_oid), (card, card_oid)):
                _git(
                    self.root,
                    ["update-index", "--add", "--cacheinfo", f"100644,{oid},{relative}"],
                    env=private_env,
                )
            tree = _git(self.root, ["write-tree"], env=private_env).stdout.strip()
            if re.fullmatch(r"[0-9a-f]{40}", tree) is None:
                raise ServiceError("private Git tree ID is malformed")
            if (
                self._tree_blob(tree, solution, private_env) != solution_oid
                or self._tree_blob(tree, card, private_env) != card_oid
            ):
                raise ServiceError("private Git tree failed exact blob verification")
            message = (
                "track2 official attempt artifacts "
                f"solution={solution_sha} card={card_sha}"
            )
            revision = _git(
                self.root,
                [
                    "-c", "user.name=Track2 Controller",
                    "-c", "user.email=track2-controller@localhost",
                    "commit-tree", tree, "-p", old_revision, "-m", message,
                ],
                env=private_env,
            ).stdout.strip()
            if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                raise ServiceError("private Git commit ID is malformed")
            # This compare-and-swap is the only operation that advances HEAD.
            # Every validation above happens first, and no worktree/index bytes
            # are consulted by the commit construction.
            materialized: list[tuple[str, tuple[int, int]]] = []
            if materialize_new_worktree_files:
                try:
                    materialized.append((
                        solution, self._materialize_new_file(solution, solution_bytes)
                    ))
                    materialized.append((
                        card, self._materialize_new_file(card, card_bytes)
                    ))
                except BaseException:
                    try:
                        for relative, identity in reversed(materialized):
                            self._remove_materialized_file(relative, identity)
                    except BaseException as cleanup_exc:
                        raise IndeterminateTransition(
                            "failed artifact materialization could not be rolled back"
                        ) from cleanup_exc
                    raise
            try:
                _git(
                    self.root,
                    ["update-ref", "HEAD", revision, old_revision],
                    env=private_env,
                )
            except BaseException as exc:
                try:
                    current_head = _git(
                        self.root, ["rev-parse", "HEAD"], env=private_env
                    ).stdout.strip()
                except BaseException as probe_exc:
                    raise IndeterminateTransition(
                        "Git compare-and-swap outcome could not be determined"
                    ) from probe_exc
                if current_head == revision:
                    raise IndeterminateTransition(
                        "Git advanced but its completion acknowledgement was ambiguous"
                    ) from exc
                if current_head != old_revision:
                    raise IndeterminateTransition(
                        "repository HEAD changed to an unrelated revision during Git CAS"
                    ) from exc
                try:
                    for relative, identity in reversed(materialized):
                        self._remove_materialized_file(relative, identity)
                except BaseException as cleanup_exc:
                    raise IndeterminateTransition(
                        "failed Git transition could not roll back worktree artifacts"
                    ) from cleanup_exc
                raise
        if materialize_new_worktree_files:
            try:
                _path, current_solution = _regular_repo_file(
                    self.root, solution, maximum_bytes=MAX_SOLUTION_BYTES
                )
                _path, current_card = _regular_repo_file(
                    self.root, card, maximum_bytes=MAX_CARD_BYTES
                )
                if current_solution != solution_bytes or current_card != card_bytes:
                    raise ServiceError("post-commit worktree bytes differ from Git inputs")
            except ServiceError as exc:
                raise IndeterminateTransition(
                    "committed artifacts became unavailable after Git advanced"
                ) from exc
        return {
            "git_revision": revision,
            "solution_sha256": solution_sha,
            "card_sha256": card_sha,
        }

    def _materialize_new_file(
        self, relative: str, payload: bytes
    ) -> tuple[int, int]:
        """Atomically publish one exact worktree file without following links."""

        pure = PurePosixPath(relative)
        temporary_leaf = _materialization_temp_leaf(relative, _sha256(payload))
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        directory_fd = root_fd
        staged_identity: tuple[int, int] | None = None
        published = False
        fd: int | None = None
        try:
            for part in pure.parts[:-1]:
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                next_fd = os.open(part, flags, dir_fd=directory_fd)
                if directory_fd != root_fd:
                    os.close(directory_fd)
                directory_fd = next_fd
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(temporary_leaf, flags, 0o600, dir_fd=directory_fd)
            os.fchmod(fd, 0o600)
            opened = os.fstat(fd)
            staged_identity = (opened.st_dev, opened.st_ino)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise ServiceError("worktree artifact write made no progress")
                view = view[written:]
            os.fsync(fd)
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size != len(payload)
            ):
                raise ServiceError("materialized worktree artifact is unsafe")
            _rename_noreplace(
                directory_fd, temporary_leaf, pure.parts[-1]
            )
            published = True
            os.fsync(directory_fd)
            assert staged_identity is not None
            return staged_identity
        except BaseException as exc:
            if staged_identity is not None:
                try:
                    cleanup_leaf = pure.parts[-1] if published else temporary_leaf
                    self._remove_exact_leaf_at(
                        directory_fd, cleanup_leaf, staged_identity
                    )
                except BaseException as cleanup_exc:
                    raise IndeterminateTransition(
                        f"failed materialization left an unsafe worktree path: {relative}"
                    ) from cleanup_exc
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError(
                f"artifact could not be materialized safely: {relative}"
            ) from exc
        finally:
            if fd is not None:
                os.close(fd)
            if directory_fd != root_fd:
                os.close(directory_fd)
            os.close(root_fd)

    def _remove_exact_leaf_at(
        self,
        directory_fd: int,
        leaf: str,
        identity: tuple[int, int],
    ) -> None:
        metadata = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            raise ServiceError("artifact cleanup identity changed")
        os.unlink(leaf, dir_fd=directory_fd)
        os.fsync(directory_fd)

    def remove_abandoned_materialization(
        self, relative: str, payload_sha256: str
    ) -> bool:
        """Remove only the reserved staging path left by a killed publisher."""

        pure = PurePosixPath(relative)
        temporary_leaf = _materialization_temp_leaf(relative, payload_sha256)
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        directory_fd = root_fd
        fd: int | None = None
        try:
            for part in pure.parts[:-1]:
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                next_fd = os.open(part, flags, dir_fd=directory_fd)
                if directory_fd != root_fd:
                    os.close(directory_fd)
                directory_fd = next_fd
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                fd = os.open(temporary_leaf, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                return False
            opened = os.fstat(fd)
            identity = (opened.st_dev, opened.st_ino)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
                or stat.S_IMODE(opened.st_mode) != 0o600
            ):
                raise ServiceError("abandoned materialization path is unsafe")
            self._remove_exact_leaf_at(directory_fd, temporary_leaf, identity)
            return True
        finally:
            if fd is not None:
                os.close(fd)
            if directory_fd != root_fd:
                os.close(directory_fd)
            os.close(root_fd)

    def _remove_materialized_file(
        self, relative: str, identity: tuple[int, int]
    ) -> None:
        pure = PurePosixPath(relative)
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        directory_fd = root_fd
        try:
            for part in pure.parts[:-1]:
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                next_fd = os.open(part, flags, dir_fd=directory_fd)
                if directory_fd != root_fd:
                    os.close(directory_fd)
                directory_fd = next_fd
            metadata = os.stat(
                pure.parts[-1], dir_fd=directory_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino) != identity
                or metadata.st_nlink != 1
            ):
                raise ServiceError("refusing to remove a replaced worktree artifact")
            os.unlink(pure.parts[-1], dir_fd=directory_fd)
            os.fsync(directory_fd)
        except OSError as exc:
            raise ServiceError("could not roll back a new worktree artifact") from exc
        finally:
            if directory_fd != root_fd:
                os.close(directory_fd)
            os.close(root_fd)


def controller_command(root: Path, request: dict[str, Any]) -> list[str]:
    request = validate_request(request)
    command = [
        str(PYTHON), "-I", str(root.resolve() / "Project" / "harness" / "iterate.py")
    ]
    if request["command"] == "log":
        return [*command, "log"]
    return [
        *command,
        "run",
        "--solution", request["solution"],
        "--card", request["card"],
    ]


def admission_state_command(root: Path) -> list[str]:
    """Private state surface used only by the trusted outer service."""

    return [
        str(PYTHON),
        "-I",
        str(root.resolve() / "Project" / "harness" / "iterate.py"),
        "_admission-state",
    ]


def _bounded(text: str) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= MAX_RESPONSE_STREAM_BYTES:
        return text
    suffix = b"\n...[controller output truncated by outer service]"
    prefix = data[: MAX_RESPONSE_STREAM_BYTES - len(suffix)].decode(
        "utf-8", errors="ignore"
    )
    return prefix + suffix.decode("ascii")


def _safe_response_text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ServiceError(f"{label} must be text")
    try:
        payload = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ServiceError(f"{label} is not valid Unicode text") from exc
    if len(payload) > maximum:
        raise ServiceError(f"{label} exceeds its byte limit")
    return value


def error_response(
    request: dict[str, Any], error: str, *, recovery_required: bool = False
) -> dict[str, Any]:
    """Build the one exact failure envelope accepted by the client."""

    request = validate_request(request)
    try:
        error_bytes = str(error).encode("utf-8", errors="replace")[:16 * 1024]
    except Exception:
        error_bytes = b"controller error could not be rendered"
    safe_error = error_bytes.decode("utf-8", errors="ignore")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request["request_id"],
        "command": request["command"],
        "ok": False,
        "returncode": 125,
        "error": safe_error,
        "recovery_required": bool(recovery_required),
    }


def validate_response(
    request: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    """Validate a response before it crosses the socket or enters the WAL."""

    request = validate_request(request)
    if not isinstance(response, dict):
        raise ServiceError("controller response must be one object")
    common = {
        "protocol_version", "request_id", "command", "ok", "returncode"
    }
    if response.get("ok") is False:
        if set(response) != common | {"error", "recovery_required"}:
            raise ServiceError("controller error response has the wrong shape")
        if response.get("returncode") != 125 or isinstance(
            response.get("returncode"), bool
        ):
            raise ServiceError("controller error response has the wrong status")
        if type(response.get("recovery_required")) is not bool:
            raise ServiceError("controller recovery flag must be boolean")
        _safe_response_text(response.get("error"), "controller error", 16 * 1024)
    elif response.get("ok") is True:
        expected = common | {
            "stdout", "stderr", "elapsed_seconds", "artifact_commit"
        }
        if set(response) != expected:
            raise ServiceError("controller success response has the wrong shape")
        if response.get("returncode") != 0 or isinstance(
            response.get("returncode"), bool
        ):
            raise ServiceError("controller success response has the wrong status")
        _safe_response_text(
            response.get("stdout"), "controller stdout", MAX_RESPONSE_STREAM_BYTES
        )
        _safe_response_text(
            response.get("stderr"), "controller stderr", MAX_RESPONSE_STREAM_BYTES
        )
        elapsed = response.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
            or float(elapsed) > RUN_TIMEOUT_SECONDS + 3600.0
        ):
            raise ServiceError("controller elapsed time is invalid")
        commit = response.get("artifact_commit")
        if request["command"] == "log":
            if commit is not None:
                raise ServiceError("log response cannot contain an artifact commit")
        else:
            if not isinstance(commit, dict) or set(commit) != {
                "git_revision", "solution_sha256", "card_sha256"
            }:
                raise ServiceError("run response has an invalid artifact commit")
            if re.fullmatch(r"[0-9a-f]{40}", commit.get("git_revision", "")) is None:
                raise ServiceError("run response has an invalid Git revision")
            for key in ("solution_sha256", "card_sha256"):
                if re.fullmatch(r"[0-9a-f]{64}", commit.get(key, "")) is None:
                    raise ServiceError(f"run response has an invalid {key}")
    else:
        raise ServiceError("controller response ok flag must be literal boolean")
    if (
        type(response.get("protocol_version")) is not int
        or response.get("protocol_version") != PROTOCOL_VERSION
        or response.get("request_id") != request["request_id"]
        or response.get("command") != request["command"]
    ):
        raise ServiceError("controller response is not bound to its request")
    _canonical_bytes(response)
    return response


def validate_response_admission(
    request: dict[str, Any],
    response: dict[str, Any],
    admission: tuple[str, str, bytes, bytes, str] | dict[str, Any],
) -> dict[str, Any]:
    """Bind a successful consuming response back to its durable intent."""

    response = validate_response(request, response)
    binding = _admission_binding(admission)
    if response["ok"]:
        commit = response["artifact_commit"]
        if (
            commit["solution_sha256"] != binding["solution_sha256"]
            or commit["card_sha256"] != binding["card_sha256"]
        ):
            raise ServiceError(
                "successful artifact commit does not match its durable admission"
            )
    return response


class ControllerAuthority:
    """Own protected state, exact commits, and fixed controller subprocesses."""

    def __init__(
        self,
        root: Path,
        *,
        workspace_root: Path | None = None,
        allow_repo_artifacts_for_tests: bool = False,
        protected_paths: tuple[str, ...] | None = None,
        env: dict[str, str] | None = None,
    ):
        self.root = root.resolve(strict=True)
        self.env = fixed_environment() if env is None else dict(env)
        if not PYTHON.is_file() or not GIT.is_file():
            raise ServiceError("fixed /usr/bin/python3 and /usr/bin/git are required")
        _validate_git_repository(self.root, self.env)
        if protected_paths is None:
            protected_paths = (
                *PROTECTED_PATHS,
                *frozen_bank_paths(self.root, self.env),
            )
        if len(protected_paths) != len(set(protected_paths)):
            raise ServiceError("protected path registry contains duplicates")
        self.protected = ProtectedState(
            self.root, paths=protected_paths, env=self.env
        )
        self.committer = ArtifactCommitter(self.root, self.protected, self.env)
        self._transaction_fd: int | None = None
        self.bank = None
        if workspace_root is None:
            if not allow_repo_artifacts_for_tests:
                raise ServiceError("external researcher workspace is required")
            self.workspace = None
        else:
            self.workspace = ArtifactWorkspace(workspace_root)
            state_result = _run_process(
                admission_state_command(self.root),
                root=self.root,
                timeout_seconds=LOG_TIMEOUT_SECONDS,
                env=self.env,
            )
            if state_result.returncode != 0:
                raise ServiceError("could not bind researcher workspace to official state")
            state = _strict_object(
                state_result.stdout.encode("utf-8"), "inner controller state"
            )
            self.bank = outer_research_bank.load(self.root)
            expected_bank = {
                "snapshot_sha256": self.bank.snapshot_sha256,
                "descriptor": self.bank.descriptor,
                "claim_count": len(self.bank.known_claims),
                "known_topics": list(self.bank.known_topics),
            }
            if (
                state.get("official_run_started") is not True
                or state.get("run_id") != self.workspace.run_id
                or state.get("run_start_git_revision")
                != self.workspace.repository_head
                or state.get("state") not in {
                    "ACTIVE", "TERMINAL_DUE", "TERMINAL"
                }
                or state.get("open_attempt") is not False
                or state.get("research_bank") != expected_bank
                or state.get("portfolio") != self.workspace.portfolio
            ):
                raise ServiceError(
                    "researcher workspace is stale or official state is not safely active"
                )
            head = _git(self.root, ["rev-parse", "HEAD"], env=self.env).stdout.strip()
            _validate_controller_commit_chain(
                self.root, self.workspace.repository_head, head, self.env
            )

    @contextlib.contextmanager
    def transaction(self):
        """Hold the inner controller's lock across admission, Git, and execution."""

        if self._transaction_fd is not None:
            raise ServiceError("controller transaction lock is already held")
        lock_path = self.root / "Project" / "results" / ".controller.lock"
        parent = lock_path.parent
        try:
            parent_metadata = parent.lstat()
        except OSError as exc:
            raise ServiceError("inner controller lock parent is unavailable") from exc
        if (
            stat.S_ISLNK(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or parent.resolve(strict=True) != parent
        ):
            raise ServiceError("inner controller lock parent is unsafe")
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise ServiceError("inner controller lock cannot be opened safely") from exc
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o022
            ):
                raise ServiceError("inner controller lock file is unsafe")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ServiceError(
                    "another owner controller command is active; retry only after it exits"
                ) from exc
            self._transaction_fd = fd
            yield
        finally:
            self._transaction_fd = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _admit_run(
        self, request: dict[str, Any]
    ) -> tuple[str, str, bytes, bytes, str]:
        """Reject every deterministic non-attempt before advancing Git HEAD."""

        state_result = _run_process(
            admission_state_command(self.root),
            root=self.root,
            timeout_seconds=LOG_TIMEOUT_SECONDS,
            env=self.env,
        )
        if state_result.returncode != 0:
            raise ServiceError("inner controller state precheck failed")
        state = _strict_object(
            state_result.stdout.encode("utf-8"), "inner controller state"
        )
        if (
            state.get("official_run_started") is not True
            or state.get("state") != "ACTIVE"
            or state.get("open_attempt") is not False
            or state.get("would_trigger_now") != []
            or type(state.get("attempt_review_budget_remaining")) is not int
            or state["attempt_review_budget_remaining"] <= 0
            or type(state.get("attempt_review_failure_budget_remaining")) is not int
            or state["attempt_review_failure_budget_remaining"] <= 0
        ):
            raise ServiceError("inner controller is not admitting another attempt")
        if self.workspace is not None:
            artifact_paths = (request["solution"], request["card"])
            if _git(
                self.root, ["ls-tree", "HEAD", "--", *artifact_paths], env=self.env
            ).stdout.strip() or _git(
                self.root,
                ["ls-files", "--stage", "--", *artifact_paths],
                env=self.env,
            ).stdout.strip():
                raise ServiceError(
                    "attempt artifact destination already exists in repository "
                    "history or index"
                )
            for relative in artifact_paths:
                try:
                    (self.root / relative).lstat()
                except FileNotFoundError:
                    continue
                raise ServiceError(
                    "attempt artifact destination already exists in the repository "
                    "worktree"
                )
        if self.workspace is None:
            _solution_path, solution_bytes = _regular_repo_file(
                self.root, request["solution"], maximum_bytes=MAX_SOLUTION_BYTES
            )
            _card_path, card_bytes = _regular_repo_file(
                self.root, request["card"], maximum_bytes=MAX_CARD_BYTES
            )
        else:
            solution_bytes = self.workspace.read(request["solution"], card=False)
            card_bytes = self.workspace.read(request["card"], card=True)
        try:
            source = solution_bytes.decode("utf-8", errors="strict")
            ast.parse(source)
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise ServiceError("candidate is not strict parseable UTF-8 Python") from exc
        static_findings = _outer_candidate_findings(source)
        if static_findings:
            raise ServiceError(
                f"candidate fails deterministic outer source policy: {static_findings}"
            )
        card = _strict_object(card_bytes, "attempt card")
        if self.workspace is not None:
            expected_iteration = state.get("official_iterations")
            if type(expected_iteration) is not int or not 0 <= expected_iteration < 50:
                raise ServiceError("inner controller returned an invalid iteration count")
            expected_iteration += 1
            solution_name = PurePosixPath(request["solution"]).name
            card_name = PurePosixPath(request["card"]).name
            if not solution_name.startswith(f"s{expected_iteration:03d}_") or not (
                card_name.startswith(f"i{expected_iteration:03d}_")
            ):
                raise ServiceError(
                    "staged artifact names must carry the next official iteration prefix"
                )
            if state.get("next_iteration") != expected_iteration:
                raise ServiceError("inner controller next-iteration binding is inconsistent")
            portfolio = state.get("portfolio")
            registrations = state.get("registered_family_records")
            if not isinstance(registrations, dict):
                raise ServiceError("inner controller family registry is malformed")
            extensions: dict[str, dict[str, Any]] = {}
            for family_id, registration in registrations.items():
                if (
                    not isinstance(family_id, str)
                    or not isinstance(registration, dict)
                    or set(registration) != {
                        "family_id", "extension", "extension_sha256", "first_iteration"
                    }
                    or registration.get("family_id") != family_id
                    or registration.get("extension_sha256")
                    != outer_policy.canonical_sha256(registration.get("extension"))
                    or type(registration.get("first_iteration")) is not int
                ):
                    raise ServiceError("inner controller family registry is malformed")
                extensions[family_id] = registration["extension"]
            candidate_sha = _sha256(solution_bytes)
            try:
                outer_policy.validate_attempt_card(
                    card,
                    run_id=state.get("run_id"),
                    expected_iteration=expected_iteration,
                    portfolio=portfolio,
                    candidate_path=request["solution"],
                    candidate_sha256=candidate_sha,
                    registered_families=extensions,
                )
            except Exception as exc:
                raise ServiceError(f"attempt card violates frozen policy: {exc}") from exc
            expected_prior = state.get("prior_outcomes_considered")
            if card["prior_outcomes_considered"] != expected_prior:
                raise ServiceError(
                    "attempt card must list every prior official outcome chronologically"
                )
            opening = state.get("portfolio_opening_order")
            if expected_iteration <= 4 and (
                not isinstance(opening, list)
                or len(opening) < expected_iteration
                or card.get("family_id") != opening[expected_iteration - 1]
            ):
                raise ServiceError("attempt card violates the frozen opening order")
            if self.bank is None:
                raise ServiceError("frozen research bank was not initialized")
            try:
                resolved_research = _outer_resolve_attempt_research(
                    self.bank, card, portfolio, registrations
                )
            except Exception as exc:
                raise ServiceError(
                    f"attempt research basis violates frozen policy: {exc}"
                ) from exc
            seed_ids = {family["family_id"] for family in portfolio["families"]}
            if card["family_id"] in seed_ids:
                family_registration = None
            elif card["family_id"] in registrations:
                family_registration = registrations[card["family_id"]]
            else:
                family_registration = {
                    "family_id": card["family_id"],
                    "extension": card["family_extension"],
                    "extension_sha256": outer_policy.canonical_sha256(
                        card["family_extension"]
                    ),
                    "first_iteration": expected_iteration,
                }
            reviews = state.get("current_attempt_rejections")
            if not isinstance(reviews, list) or any(
                not isinstance(review, dict) for review in reviews
            ):
                raise ServiceError("inner controller rejection summary is malformed")
            rejected_same_candidate = [
                review for review in reviews
                if review.get("candidate_sha256") == candidate_sha
            ]
            if rejected_same_candidate and card.get("corrects_review_id") not in {
                review.get("review_id") for review in rejected_same_candidate
            }:
                raise ServiceError(
                    "a previously rejected candidate needs a correction card"
                )
            semantics = {
                "policy_id": outer_policy.POLICY_ID,
                "run_id": state.get("run_id"),
                "iteration": expected_iteration,
                "candidate_sha256": candidate_sha,
                "candidate_ast_sha256": _outer_candidate_fingerprint(source),
                "card_canonical_sha256": outer_policy.canonical_sha256(card),
                "prior_outcome_ids": expected_prior,
                "research_bank_snapshot_sha256": self.bank.snapshot_sha256,
                "resolved_research_sha256": outer_policy.canonical_sha256(
                    resolved_research
                ),
                "family_registration": family_registration,
            }
            review_request_id = outer_policy.canonical_sha256({
                "review_protocol": "track2.no-tools.consensus.v1",
                "kind": "attempt",
                "semantics": semantics,
            })
            if any(review.get("request_id") == review_request_id for review in reviews):
                raise ServiceError("this exact attempt already has a sticky verdict")
        expected_parent = _git(
            self.root, ["rev-parse", "HEAD"], env=self.env
        ).stdout.strip()
        if self.workspace is not None:
            _validate_controller_commit_chain(
                self.root,
                self.workspace.repository_head,
                expected_parent,
                self.env,
            )
        return (
            _sha256(solution_bytes), _sha256(card_bytes), solution_bytes,
            card_bytes, expected_parent,
        )

    def pre_admit(
        self, request: dict[str, Any]
    ) -> tuple[str, str, bytes, bytes, str] | None:
        """Perform side-effect-free admission before a request occupies the WAL."""

        request = validate_request(request)
        if (
            self.workspace is not None
            and (
                request["workspace_id"] != self.workspace.workspace_id
                or request["workspace_binding"] != self.workspace.binding
            )
        ):
            raise ServiceError("request belongs to a different researcher workspace")
        self.protected.verify()
        return self._admit_run(request) if request["command"] == "run" else None

    def _require_closed_inner_transition(self) -> dict[str, Any]:
        """Prove that a consuming inner command left no open attempt."""

        state_result = _run_process(
            admission_state_command(self.root),
            root=self.root,
            timeout_seconds=LOG_TIMEOUT_SECONDS,
            env=self.env,
        )
        if state_result.returncode != 0:
            raise IndeterminateTransition(
                "inner state could not be read after the consuming command"
            )
        try:
            state = _strict_object(
                state_result.stdout.encode("utf-8"), "post-run inner controller state"
            )
        except ServiceError as exc:
            raise IndeterminateTransition(
                "inner state was malformed after the consuming command"
            ) from exc
        if (
            state.get("official_run_started") is not True
            or state.get("open_attempt") is not False
        ):
            raise IndeterminateTransition(
                "inner command may have left an interrupted official attempt"
            )
        return state

    def execute(
        self,
        request: dict[str, Any],
        *,
        admission: tuple[str, str, bytes, bytes, str] | None = None,
    ) -> dict[str, Any]:
        request = validate_request(request)
        if (
            self.workspace is not None
            and (
                request["workspace_id"] != self.workspace.workspace_id
                or request["workspace_binding"] != self.workspace.binding
            )
        ):
            raise ServiceError("request belongs to a different researcher workspace")
        self.protected.verify()
        commit = None
        if request["command"] == "run":
            if admission is None:
                admission = self._admit_run(request)
            solution_sha, card_sha, solution_bytes, card_bytes, parent_revision = admission
            self.protected.verify()
            commit = self.committer.commit_pair(
                request["solution"], request["card"],
                expected_solution_sha256=solution_sha,
                expected_card_sha256=card_sha,
                captured_solution_bytes=(
                    solution_bytes if self.workspace is not None else None
                ),
                captured_card_bytes=card_bytes if self.workspace is not None else None,
                materialize_new_worktree_files=self.workspace is not None,
                expected_parent_revision=parent_revision,
            )
        command = controller_command(self.root, request)
        started = time.monotonic()
        try:
            process_env = dict(self.env)
            pass_fds: tuple[int, ...] = ()
            if request["command"] == "run" and self.workspace is not None:
                if self._transaction_fd is None:
                    raise IndeterminateTransition(
                        "outer run was invoked without the shared controller lock"
                    )
                process_env[INHERITED_LOCK_FD_ENV] = str(self._transaction_fd)
                pass_fds = (self._transaction_fd,)
            result = _run_process(
                command,
                root=self.root,
                timeout_seconds=(
                    LOG_TIMEOUT_SECONDS
                    if request["command"] == "log" else RUN_TIMEOUT_SECONDS
                ),
                env=process_env,
                pass_fds=pass_fds,
            )
            self.protected.verify()
        except Exception as exc:
            if commit is not None:
                raise IndeterminateTransition(
                    "inner controller completion could not be established"
                ) from exc
            raise
        if request["command"] == "run":
            # A textual ``REFUSED:`` is not proof that no mutation happened:
            # the inner controller can fail after durably opening an attempt.
            # Re-read protected state while the outer transaction lock is still
            # held and keep the WAL pending unless closure is explicit.
            self._require_closed_inner_transition()
        if result.returncode != 0:
            controlled_refusal = (
                result.returncode == 1 and result.stderr.startswith("REFUSED:")
            )
            recorded_candidate_failure = result.returncode == 2 and bool(
                result.stdout.strip().startswith("{")
            )
            if commit is not None and not (
                controlled_refusal or recorded_candidate_failure
            ):
                raise IndeterminateTransition(
                    "inner controller exited without a recognized durable outcome"
                )
            detail = _bounded((result.stderr or result.stdout or "controller refused").strip())
            return error_response(
                request,
                f"inner controller refused request: {detail}",
                recovery_required=False,
            )
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request["request_id"],
            "command": request["command"],
            "ok": True,
            "returncode": 0,
            "stdout": _bounded(result.stdout),
            "stderr": _bounded(result.stderr),
            "elapsed_seconds": time.monotonic() - started,
            "artifact_commit": commit,
        }
        return validate_response(request, response)


class AuditStore:
    """Bounded two-phase WAL for consuming ``run`` requests.

    A pending record is fsynced before any authority side effect.  A process
    death after that point leaves an intentionally indeterminate request that
    is never executed again automatically.
    """

    def __init__(self, path: Path):
        self.path = path
        _prepare_private_parent(self.path, "service audit log")
        self.cache: dict[str, dict[str, Any]] = {}
        self.failed = False
        try:
            fd = os.open(
                self.path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
                | getattr(os, "O_NONBLOCK", 0),
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ServiceError("audit path is unavailable or unsafe") from exc
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size > MAX_AUDIT_BYTES
            ):
                raise ServiceError("audit path must be one bounded private regular file")
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    raise ServiceError("audit log ended before its recorded size")
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            after = os.fstat(fd)
            if (
                (after.st_dev, after.st_ino, after.st_size, after.st_nlink)
                != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_nlink)
            ):
                raise ServiceError("audit log changed while loading")
        finally:
            os.close(fd)
        if payload and not payload.endswith(b"\n"):
            raise ServiceError("audit log has a torn final record")
        for line_number, raw in enumerate(payload.splitlines(), 1):
            if not raw:
                raise ServiceError(f"audit line {line_number} is empty")
            self._load_row(_strict_object(raw, f"audit line {line_number}"), line_number)
        if len(self.cache) > MAX_AUDIT_REQUESTS:
            raise ServiceError("audit log exceeds its request-count limit")

    @staticmethod
    def _request_binding(request: dict[str, Any]) -> tuple[dict[str, Any], str]:
        request = validate_request(request)
        if request["command"] != "run":
            raise ServiceError("only consuming run requests belong in the WAL")
        return request, _sha256(_canonical_bytes(request))

    @staticmethod
    def _base_row(row: dict[str, Any], expected: set[str], line_number: int) -> None:
        if set(row) != expected:
            raise ServiceError(f"audit line {line_number} has the wrong shape")
        if row.get("format") != AUDIT_FORMAT or type(row.get("recorded_ns")) is not int:
            raise ServiceError(f"audit line {line_number} has invalid metadata")
        if row["recorded_ns"] <= 0:
            raise ServiceError(f"audit line {line_number} has invalid time")
        if (
            not isinstance(row.get("request_id"), str)
            or REQUEST_ID_RE.fullmatch(row["request_id"]) is None
            or not isinstance(row.get("request_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", row["request_sha256"]) is None
        ):
            raise ServiceError(f"audit line {line_number} has an invalid binding")

    def _load_row(self, row: dict[str, Any], line_number: int) -> None:
        phase = row.get("phase")
        pending_keys = {
            "format", "phase", "recorded_ns", "request_id",
            "request_sha256", "request", "admission",
        }
        completed_keys = {
            "format", "phase", "recorded_ns", "request_id",
            "request_sha256", "response",
        }
        if phase == "pending":
            self._base_row(row, pending_keys, line_number)
            request, request_sha = self._request_binding(row.get("request"))
            admission = _admission_binding(row.get("admission"))
            if (
                row["request_id"] != request["request_id"]
                or row["request_sha256"] != request_sha
                or _canonical_bytes(row["request"]) != _canonical_bytes(request)
                or request["request_id"] in self.cache
            ):
                raise ServiceError(f"audit line {line_number} has inconsistent pending data")
            self.cache[request["request_id"]] = {
                "request": request,
                "request_sha256": request_sha,
                "admission": admission,
                "response": None,
            }
            return
        if phase == "completed":
            self._base_row(row, completed_keys, line_number)
            record = self.cache.get(row["request_id"])
            if (
                record is None
                or record["response"] is not None
                or row["request_sha256"] != record["request_sha256"]
            ):
                raise ServiceError(f"audit line {line_number} lacks one pending predecessor")
            response = validate_response_admission(
                record["request"], row.get("response"), record["admission"]
            )
            record["response"] = response
            return
        raise ServiceError(f"audit line {line_number} has an invalid phase")

    def lookup(
        self, request: dict[str, Any]
    ) -> tuple[str, dict[str, Any] | None]:
        if self.failed:
            raise ServiceError("WAL is latched failed and requires owner inspection")
        request, request_sha = self._request_binding(request)
        record = self.cache.get(request["request_id"])
        if record is None:
            return "absent", None
        if (
            record["request_sha256"] != request_sha
            or _canonical_bytes(record["request"]) != _canonical_bytes(request)
        ):
            raise ServiceError("request_id was already used for different bytes")
        response = record["response"]
        return ("pending", None) if response is None else ("completed", response)

    def _append_row(self, row: dict[str, Any]) -> None:
        payload = _canonical_bytes(row) + b"\n"
        # O_NONBLOCK is security-relevant even though a valid WAL is a regular
        # file: a same-UID process must not be able to replace the path with a
        # FIFO between validation and this open and hang the controller.
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | os.O_CLOEXEC
            | getattr(os, "O_NONBLOCK", 0)
        )
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            try:
                fd = os.open(self.path, flags, 0o600)
            except OSError as exc:
                raise ServiceError("audit log cannot be opened safely") from exc
            created_empty = False
            try:
                metadata = os.fstat(fd)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_size + len(payload) > MAX_AUDIT_BYTES
                ):
                    raise ServiceError("audit log is not a bounded private regular file")
                created_empty = metadata.st_size == 0
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise ServiceError("audit append made no progress")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            if created_empty:
                parent_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
        except BaseException:
            self.failed = True
            raise

    def _current_size_for_admission(self) -> int:
        """Return the exact safe WAL size without creating or modifying it."""

        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NONBLOCK", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            return 0
        except OSError as exc:
            raise ServiceError("audit log cannot be inspected safely") from exc
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size > MAX_AUDIT_BYTES
            ):
                raise ServiceError(
                    "audit log is not a bounded private regular file"
                )
            return metadata.st_size
        finally:
            os.close(fd)

    def _reserve_physical_capacity(self, length: int) -> None:
        """Preallocate append blocks while preserving the WAL's logical size."""

        flags = (
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NONBLOCK
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise AuditCapacityError(
                "WAL cannot be opened for physical reservation"
            ) from exc
        try:
            os.fchmod(fd, 0o600)
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size > MAX_AUDIT_BYTES
                or metadata.st_size + length > MAX_AUDIT_BYTES
            ):
                raise AuditCapacityError(
                    "WAL is unsafe or too large for physical reservation"
                )
            _fallocate_keep_size(fd, metadata.st_size, length)
            os.fsync(fd)
        finally:
            os.close(fd)
        parent_fd = os.open(
            self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def _reserve_completion_capacity(self, pending_row: dict[str, Any]) -> int:
        """Refuse before append unless pending plus worst-case completion fit."""

        pending_bytes = len(_canonical_bytes(pending_row)) + 1
        current_bytes = self._current_size_for_admission()
        if (
            current_bytes
            + pending_bytes
            + MAX_COMPLETION_ROW_BYTES
            > MAX_AUDIT_BYTES
        ):
            raise AuditCapacityError(
                "WAL lacks space for both the pending record and its reserved "
                "completion record"
            )
        reservation = pending_bytes + MAX_COMPLETION_ROW_BYTES
        self._reserve_physical_capacity(reservation)
        return reservation

    def begin(
        self,
        request: dict[str, Any],
        admission: tuple[str, str, bytes, bytes, str] | dict[str, Any],
    ) -> None:
        if self.failed:
            raise ServiceError("WAL is latched failed and requires owner inspection")
        request, request_sha = self._request_binding(request)
        admission_value = _admission_binding(admission)
        if request["request_id"] in self.cache:
            raise ServiceError("WAL begin attempted to reuse request_id")
        if len(self.cache) >= MAX_AUDIT_REQUESTS:
            raise AuditCapacityError(
                "WAL request-count limit reached; do not delete deduplication history"
            )
        row = {
            "format": AUDIT_FORMAT,
            "phase": "pending",
            "recorded_ns": time.time_ns(),
            "request_id": request["request_id"],
            "request_sha256": request_sha,
            "request": request,
            "admission": admission_value,
        }
        self._reserve_completion_capacity(row)
        self._append_row(row)
        self.cache[request["request_id"]] = {
            "request": request,
            "request_sha256": request_sha,
            "admission": admission_value,
            "response": None,
        }

    def complete(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        if self.failed:
            raise ServiceError("WAL is latched failed and requires owner inspection")
        request, request_sha = self._request_binding(request)
        record = self.cache.get(request["request_id"])
        if (
            record is None
            or record["request_sha256"] != request_sha
            or record["response"] is not None
        ):
            raise ServiceError("WAL completion lacks one exact pending request")
        response = validate_response_admission(
            request, response, record["admission"]
        )
        row = {
            "format": AUDIT_FORMAT,
            "phase": "completed",
            "recorded_ns": time.time_ns(),
            "request_id": request["request_id"],
            "request_sha256": request_sha,
            "response": response,
        }
        completion_bytes = len(_canonical_bytes(row)) + 1
        if completion_bytes > MAX_COMPLETION_ROW_BYTES:
            raise ServiceError(
                "validated completion record exceeds its pre-reserved WAL bound"
            )
        # Reassert the still-durable post-pending extent after a process
        # restart. On supporting filesystems this is idempotent and requires no
        # new blocks when the original reservation survived.
        self._reserve_physical_capacity(completion_bytes)
        self._append_row(row)
        record["response"] = response


def repair_torn_audit_tail(path: Path) -> dict[str, int]:
    """Owner-only removal of one non-newline WAL suffix after power loss.

    Newline-complete malformed records are never repaired.  The intact prefix
    is fully replay-validated before the exact final suffix is truncated.
    """

    _prepare_private_parent(path, "service audit log")
    flags = (
        os.O_RDWR
        | os.O_CLOEXEC
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ServiceError("audit log cannot be opened for torn-tail repair") from exc
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 1 <= before.st_size <= MAX_AUDIT_BYTES
        ):
            raise ServiceError("audit log is not a bounded private regular file")
        payload = bytearray()
        while len(payload) < before.st_size:
            chunk = os.read(fd, min(65536, before.st_size - len(payload)))
            if not chunk:
                raise ServiceError("audit log ended during torn-tail inspection")
            payload.extend(chunk)
        after = os.fstat(fd)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_nlink)
            != (before.st_dev, before.st_ino, before.st_size, before.st_nlink)
        ):
            raise ServiceError("audit log changed during torn-tail inspection")
        raw = bytes(payload)
        if raw.endswith(b"\n"):
            raise ServiceError("audit log has no torn non-newline tail")
        prefix_length = raw.rfind(b"\n") + 1
        prefix = raw[:prefix_length]

        validator = object.__new__(AuditStore)
        validator.path = path
        validator.cache = {}
        validator.failed = False
        for line_number, line in enumerate(prefix.splitlines(), 1):
            if not line:
                raise ServiceError(
                    f"intact audit prefix line {line_number} is empty"
                )
            validator._load_row(
                _strict_object(line, f"intact audit prefix line {line_number}"),
                line_number,
            )
        if len(validator.cache) > MAX_AUDIT_REQUESTS:
            raise ServiceError("intact audit prefix exceeds its request-count limit")

        os.ftruncate(fd, prefix_length)
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return {
        "previous_bytes": len(raw),
        "repaired_bytes": prefix_length,
        "removed_bytes": len(raw) - prefix_length,
    }


def dispatch_request(
    authority: ControllerAuthority,
    audit: AuditStore,
    transition_lock: threading.Lock,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Execute one validated request under the serialized outer authority."""

    request = validate_request(request)
    transaction = getattr(authority, "transaction", None)
    transaction_context = (
        transaction() if callable(transaction) else contextlib.nullcontext()
    )
    with transition_lock, transaction_context:
        if request["command"] == "log":
            try:
                response = authority.execute(request)
            except Exception as exc:
                response = error_response(request, f"{type(exc).__name__}: {exc}")
            return validate_response(request, response)

        try:
            status, cached = audit.lookup(request)
        except Exception as exc:
            return error_response(
                request, f"WAL lookup failed: {type(exc).__name__}: {exc}",
                recovery_required=True,
            )
        if status == "completed":
            assert cached is not None
            return validate_response(request, cached)
        if status == "pending":
            return error_response(
                request,
                "request is indeterminate after an interrupted owner transition; "
                "owner recovery is required and automatic replay is forbidden",
                recovery_required=True,
            )
        if any(record["response"] is None for record in audit.cache.values()):
            return error_response(
                request,
                "another consuming request is indeterminate; all new runs are "
                "latched until owner reconciliation",
                recovery_required=True,
            )
        pre_admit = getattr(authority, "pre_admit", None)
        admission = None
        if callable(pre_admit):
            try:
                admission = pre_admit(request)
            except Exception as exc:
                # No durable side effect has happened, so malformed or stale
                # requests do not consume permanent deduplication capacity.
                return error_response(
                    request,
                    f"side-effect-free admission refused: {type(exc).__name__}: {exc}",
                    recovery_required=False,
                )
        try:
            if admission is None:
                raise ServiceError("authority did not provide an admission binding")
            audit.begin(request, admission)
        except AuditCapacityError as exc:
            return error_response(
                request,
                f"WAL capacity exhausted before admission: {exc}",
                recovery_required=False,
            )
        except Exception as exc:
            return error_response(
                request, f"WAL begin failed: {type(exc).__name__}: {exc}",
                recovery_required=True,
            )
        try:
            response = (
                authority.execute(request, admission=admission)
                if callable(pre_admit)
                else authority.execute(request)
            )
        except IndeterminateTransition as exc:
            return error_response(
                request,
                f"outer transition is indeterminate: {exc}",
                recovery_required=True,
            )
        except Exception as exc:
            response = error_response(request, f"{type(exc).__name__}: {exc}")
        # BaseException is deliberately not caught: a simulated or real process
        # death leaves the fsynced pending row behind.
        try:
            audit.complete(request, response)
        except Exception as exc:
            # Authority may already have changed Git or official state.  Never
            # tell the client this request is safely complete unless the
            # completion record itself is durable.
            return error_response(
                request,
                f"completion durability is indeterminate: {type(exc).__name__}: {exc}",
                recovery_required=True,
            )
        return validate_response(request, response)


def _repository_authority_state_dir(root: Path) -> Path:
    identity = _sha256(str(root.resolve(strict=True)).encode("utf-8"))[:24]
    return (
        Path.home()
        / ".local" / "state" / "tiktok-techjam-2026-track2" / identity
    ).absolute()


def _service_binding_value(
    *,
    root: Path,
    audit_path: Path,
) -> dict[str, Any]:
    return {
        "format": SERVICE_BINDING_FORMAT,
        "repository": str(root),
        "audit_log": str(audit_path),
    }


def _read_service_binding(state_dir: Path) -> dict[str, Any]:
    payload = _read_workspace_root_file(
        state_dir,
        SERVICE_BINDING_NAME,
        maximum_bytes=MAX_SERVICE_BINDING_BYTES,
        label="controller service binding",
    )
    if not payload.endswith(b"\n"):
        raise ServiceError("controller service binding is not a complete line")
    value = _strict_object(payload[:-1], "controller service binding")
    if (
        payload != _canonical_bytes(value) + b"\n"
        or set(value) != {"format", "repository", "audit_log"}
        or value.get("format") != SERVICE_BINDING_FORMAT
        or any(
            not isinstance(value.get(name), str)
            for name in ("repository", "audit_log")
        )
    ):
        raise ServiceError("controller service binding has the wrong shape")
    return value


def _create_service_binding(state_dir: Path, value: dict[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > MAX_SERVICE_BINDING_BYTES:
        raise ServiceError("controller service binding exceeds its size cap")
    directory_fd = os.open(
        state_dir,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary = f".{SERVICE_BINDING_NAME}.{secrets.token_hex(12)}.tmp"
    fd: int | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(fd, 0o600)
        opened = os.fstat(fd)
        temporary_identity = (opened.st_dev, opened.st_ino)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ServiceError("controller service binding write made no progress")
            view = view[written:]
        os.fsync(fd)
        _rename_noreplace(directory_fd, temporary, SERVICE_BINDING_NAME)
        temporary_identity = None
        os.fsync(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if temporary_identity is not None:
            try:
                metadata = os.stat(
                    temporary, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    stat.S_ISREG(metadata.st_mode)
                    and (metadata.st_dev, metadata.st_ino) == temporary_identity
                ):
                    os.unlink(temporary, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _enforce_service_binding(
    *,
    state_dir: Path,
    root: Path,
    audit_path: Path,
    create: bool,
) -> dict[str, Any]:
    """Permanently bind every restart/recovery to one repository WAL."""

    try:
        state_dir = _private_workspace_root(state_dir)
    except ServiceError as exc:
        raise ServiceError("external authority state directory is unsafe") from exc
    root = root.resolve(strict=True)
    _prepare_private_parent(audit_path, "service audit log")
    audit_path = audit_path.absolute()
    expected = _service_binding_value(
        root=root,
        audit_path=audit_path,
    )
    binding_path = state_dir / SERVICE_BINDING_NAME
    try:
        binding_path.lstat()
    except FileNotFoundError:
        if not create:
            raise ServiceError(
                "controller service binding is missing; offline recovery cannot "
                "select a new WAL"
            )
        _create_service_binding(state_dir, expected)
    current = _read_service_binding(state_dir)
    for name in ("repository", "audit_log"):
        if current.get(name) != expected[name]:
            raise ServiceError(
                f"controller service restart changed its frozen {name}"
            )
    return current


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "ControllerServer" = self.server  # type: ignore[assignment]
        request: dict[str, Any] | None = None
        try:
            self.request.settimeout(SOCKET_READ_TIMEOUT_SECONDS)
            if hasattr(socket, "SO_PEERCRED"):
                credentials = self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                _peer_pid, peer_uid, _peer_gid = struct.unpack("3i", credentials)
                if peer_uid != server.allowed_uid:
                    raise ServiceError("Unix peer uid is not authorized")
            raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
            if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                raise ServiceError("request is empty, oversized, or not newline-terminated")
            request = validate_request(_strict_object(raw[:-1], "RPC request"))
            response = dispatch_request(
                server.authority, server.audit, server.transition_lock, request
            )
        except Exception as exc:  # every protocol failure becomes one bounded response
            if request is not None:
                response = error_response(
                    request,
                    f"{type(exc).__name__}: {exc}",
                    recovery_required=request.get("command") == "run",
                )
            else:
                # Malformed callers have no request binding with which to build
                # the strict client envelope.  Keep even this diagnostic bounded.
                response = {
                    "protocol_version": PROTOCOL_VERSION,
                    "ok": False,
                    "returncode": 125,
                    "error": str(exc)[:4000],
                }
        try:
            self.wfile.write(_canonical_bytes(response) + b"\n")
        except (BrokenPipeError, ConnectionError, OSError):
            pass


class ControllerServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Bounded threaded transport around one serialized authority."""

    allow_reuse_address = False
    daemon_threads = False
    block_on_close = True
    request_queue_size = MAX_CONCURRENT_CONNECTIONS

    def __init__(
        self,
        socket_path: Path,
        authority: ControllerAuthority,
        audit_path: Path,
        *,
        authority_state_dir: Path,
        allowed_uid: int | None = None,
    ):
        self.socket_path = socket_path
        self.authority = authority
        self.allowed_uid = os.getuid() if allowed_uid is None else allowed_uid
        self._repository_lock_fd: int | None = None
        self._lock_fd: int | None = None
        self._bound_identity: tuple[int, int] | None = None
        _validate_runtime_endpoints(socket_path, audit_path)
        _prepare_private_parent(socket_path, "controller runtime")
        _prepare_private_parent(audit_path, "controller runtime")
        try:
            # The repository lock is deliberately acquired first. A runtime
            # directory is caller-selected, so its lock alone cannot prevent
            # two services with separate WALs from controlling the same Git
            # authority. Offline recovery follows the same lock order.
            self._repository_lock_fd = _acquire_repository_service_lock(
                authority.root
            )
            self._lock_fd = _acquire_service_lock(socket_path.parent)
            if authority.workspace is None:
                raise ServiceError("controller service requires a bound workspace")
            _enforce_service_binding(
                state_dir=authority_state_dir,
                root=authority.root,
                audit_path=audit_path,
                create=True,
            )
            self.audit = AuditStore(audit_path)
            self.transition_lock = threading.Lock()
            self.connection_slots = threading.BoundedSemaphore(
                MAX_CONCURRENT_CONNECTIONS
            )
            _remove_stale_socket(socket_path)
            previous_umask = os.umask(0o177)
            try:
                super().__init__(str(socket_path), _RequestHandler)
            finally:
                os.umask(previous_umask)
            os.chmod(socket_path, 0o600)
            metadata = socket_path.lstat()
            if (
                not stat.S_ISSOCK(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise ServiceError("bound controller socket failed its identity check")
            self._bound_identity = (metadata.st_dev, metadata.st_ino)
        except BaseException:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            if self._repository_lock_fd is not None:
                os.close(self._repository_lock_fd)
                self._repository_lock_fd = None
            raise

    def process_request(self, request, client_address) -> None:
        if not self.connection_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connection_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connection_slots.release()

    def server_close(self) -> None:
        super().server_close()
        try:
            metadata = self.socket_path.lstat()
            if (
                self._bound_identity is not None
                and stat.S_ISSOCK(metadata.st_mode)
                and metadata.st_uid == os.getuid()
                and (metadata.st_dev, metadata.st_ino) == self._bound_identity
            ):
                self.socket_path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._bound_identity = None
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            if self._repository_lock_fd is not None:
                os.close(self._repository_lock_fd)
                self._repository_lock_fd = None


def _prepare_socket_path(path: Path) -> None:
    if not path.is_absolute():
        raise ServiceError("controller socket path must be absolute")
    _prepare_private_parent(path, "controller socket")
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise ServiceError("controller socket path already exists; refusing replacement")


def _remove_stale_socket(path: Path) -> None:
    """Remove one safely identified stale endpoint while service flock is held."""

    if not path.is_absolute():
        raise ServiceError("controller socket path must be absolute")
    _prepare_private_parent(path, "controller socket")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ServiceError("existing controller endpoint is not a safe stale socket")
    path.unlink()
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _prepare_private_parent(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ServiceError(f"{label} path must be absolute")
    try:
        parent = _private_workspace_root(path.parent)
    except (OSError, ServiceError) as exc:
        raise ServiceError(f"{label} parent must already exist safely") from exc
    if parent != path.parent:
        raise ServiceError(f"{label} parent must use its canonical spelling")


def _validate_runtime_endpoints(socket_path: Path, audit_path: Path) -> None:
    """Reserve the socket, WAL, and service-lock names as distinct objects."""

    if socket_path.parent != audit_path.parent:
        raise ServiceError(
            "socket and audit log must share one private runtime parent"
        )
    if socket_path == audit_path:
        raise ServiceError("socket and service audit log must be different paths")
    if (
        socket_path.name == SERVICE_LOCK_NAME
        or audit_path.name == SERVICE_LOCK_NAME
    ):
        raise ServiceError("runtime endpoint collides with the reserved service lock")


def _acquire_service_lock(parent: Path) -> int:
    lock_path = parent / SERVICE_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(lock_path, flags, 0o600)
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ServiceError("controller service lock is not private and unique")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException as exc:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if isinstance(exc, ServiceError):
            raise
        raise ServiceError("another controller service owns this runtime") from exc


def _acquire_repository_service_lock(root: Path) -> int:
    """Own the one controller/offline-recovery lifetime for a repository."""

    parent = root / "Project" / "results"
    try:
        metadata = parent.lstat()
        canonical_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise ServiceError(
            "repository service lock parent is unavailable"
        ) from exc
    if (
        not root.is_absolute()
        or root.resolve(strict=True) != root
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or canonical_parent != parent
    ):
        raise ServiceError("repository service lock parent is unsafe")

    lock_path = parent / REPOSITORY_SERVICE_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(lock_path, flags, 0o600)
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise ServiceError(
                "repository service lock is not private and unique"
            )
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException as exc:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if isinstance(exc, ServiceError):
            raise
        raise ServiceError(
            "another controller service or offline recovery owns this repository"
        ) from exc


@contextlib.contextmanager
def _owner_inner_controller_lock(root: Path):
    """Serialize offline repair with every direct inner controller mutation."""

    path = root / "Project" / "results" / ".controller.lock"
    parent = path.parent
    try:
        metadata = parent.lstat()
    except OSError as exc:
        raise ServiceError("inner controller lock parent is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or parent.resolve(strict=True) != parent
    ):
        raise ServiceError("inner controller lock parent is unsafe")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or opened.st_mode & 0o022
        ):
            raise ServiceError("inner controller lock file is unsafe")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ServiceError("another inner controller command is active") from exc
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def reconcile_pending_request(
    root: Path,
    audit_path: Path,
    request_id: str,
    env: dict[str, str],
    workspace: ArtifactWorkspace,
) -> dict[str, Any]:
    """Owner-only repair and closure of one globally latched WAL request."""

    if REQUEST_ID_RE.fullmatch(request_id) is None:
        raise ServiceError("pending request id must be 32 lowercase hexadecimal characters")
    _validate_git_repository(root, env)
    protected = ProtectedState(
        root,
        paths=(*PROTECTED_PATHS, *frozen_bank_paths(root, env)),
        env=env,
    )
    audit = AuditStore(audit_path)
    record = audit.cache.get(request_id)
    if record is None or record.get("response") is not None:
        raise ServiceError("requested WAL row is not pending")
    request = validate_request(record["request"])
    admission = _admission_binding(record.get("admission"))
    if request["command"] != "run":
        raise ServiceError("only a consuming run request can be reconciled")
    if request["workspace_id"] != workspace.workspace_id:
        raise ServiceError(
            "pending request belongs to a different researcher workspace"
        )
    state_result = _run_process(
        admission_state_command(root),
        root=root,
        timeout_seconds=LOG_TIMEOUT_SECONDS,
        env=env,
    )
    if state_result.returncode != 0:
        raise ServiceError("inner controller state could not be read for reconciliation")
    state = _strict_object(
        state_result.stdout.encode("utf-8"), "inner controller state"
    )
    if state.get("official_run_started") is not True:
        raise ServiceError("pending request has no official run to reconcile")
    if state.get("run_id") != workspace.run_id:
        raise ServiceError("pending request workspace belongs to another official run")
    if state.get("run_start_git_revision") != workspace.repository_head:
        raise ServiceError(
            "pending request workspace base differs from the frozen run start"
        )
    bank = outer_research_bank.load(root)
    expected_bank = {
        "snapshot_sha256": bank.snapshot_sha256,
        "descriptor": bank.descriptor,
        "claim_count": len(bank.known_claims),
        "known_topics": list(bank.known_topics),
    }
    if (
        state.get("portfolio") != workspace.portfolio
        or state.get("research_bank") != expected_bank
    ):
        raise ServiceError(
            "pending request workspace does not match frozen run resources"
        )
    head = _git(root, ["rev-parse", "HEAD"], env=env).stdout.strip()
    _validate_controller_commit_chain(
        root, workspace.repository_head, head, env
    )
    if state.get("open_attempt") is not False:
        raise ServiceError(
            "inner attempt is still open; run owner-only iterate.py recover first"
        )

    committer = ArtifactCommitter(root, protected, env)
    tracked: dict[str, bytes | None] = {}
    for relative in (request["solution"], request["card"]):
        listing = _git(root, ["ls-tree", "HEAD", "--", relative], env=env).stdout
        tracked[relative] = _git_blob(root, relative, env) if listing else None
    if (tracked[request["solution"]] is None) != (tracked[request["card"]] is None):
        raise ServiceError("pending artifact pair is only partly present in Git HEAD")
    pair_committed = tracked[request["solution"]] is not None
    if not pair_committed and head != admission["parent_revision"]:
        raise ServiceError(
            "repository HEAD moved despite the pending pair not being committed"
        )
    if pair_committed:
        parents = _git(
            root, ["rev-list", "--parents", "-n", "1", head], env=env
        ).stdout.strip().split()
        if parents != [head, admission["parent_revision"]]:
            raise ServiceError(
                "pending artifact commit is not the direct child of durable intent"
            )

    for relative, payload in tracked.items():
        is_card = relative == request["card"]
        expected_sha = admission["card_sha256" if is_card else "solution_sha256"]
        expected_size = admission["card_size" if is_card else "solution_size"]
        committer.remove_abandoned_materialization(relative, expected_sha)
        if payload is not None and (
            len(payload) != expected_size or _sha256(payload) != expected_sha
        ):
            raise ServiceError(
                "pending Git artifact does not match the durable admission binding"
            )
        path = root / relative
        try:
            metadata = path.lstat()
            exists = True
        except FileNotFoundError:
            metadata = None
            exists = False
        if payload is not None:
            if not exists:
                committer._materialize_new_file(relative, payload)
            else:
                _path, live = _regular_repo_file(
                    root,
                    relative,
                    maximum_bytes=(MAX_CARD_BYTES if is_card else MAX_SOLUTION_BYTES),
                )
                if live != payload:
                    raise ServiceError(
                        "pending worktree artifact differs from committed Git bytes"
                    )
        elif exists:
            assert metadata is not None
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ServiceError("uncommitted pending artifact path is unsafe")
            _path, live = _regular_repo_file(
                root,
                relative,
                maximum_bytes=(MAX_CARD_BYTES if is_card else MAX_SOLUTION_BYTES),
            )
            if len(live) != expected_size or _sha256(live) != expected_sha:
                raise ServiceError(
                    "uncommitted pending artifact does not match durable intent"
                )
            committer._remove_materialized_file(
                relative, (metadata.st_dev, metadata.st_ino)
            )
    protected.verify()
    response = error_response(
        request,
        "owner reconciliation verified a closed inner state and repaired the exact "
        "artifact pair; the original response remains unavailable",
        recovery_required=False,
    )
    audit.complete(request, response)
    return response


def _paths_overlap(left: Path, right: Path) -> bool:
    """Whether either canonical path contains the other, including equality."""

    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validate_external_layout(
    root: Path, runtime_parent: Path, workspace: Path, authority_root: Path
) -> None:
    """Keep repository, runtime, workspace, and authority in separate domains."""

    if _paths_overlap(root, authority_root):
        raise ServiceError("repository must be disjoint from authority state")
    for external, label in (
        (runtime_parent, "controller runtime"),
        (workspace, "researcher workspace"),
    ):
        if _paths_overlap(external, root):
            raise ServiceError(f"{label} must be disjoint from the repository")
    if _paths_overlap(workspace, runtime_parent):
        raise ServiceError(
            "researcher workspace must be disjoint from controller runtime"
        )
    if _paths_overlap(workspace, authority_root):
        raise ServiceError(
            "researcher workspace must be disjoint from authority state"
        )
    if _paths_overlap(runtime_parent, authority_root):
        raise ServiceError(
            "controller runtime must be disjoint from authority state"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Track 2 outer controller authority service")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--audit-log", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    owner_action = parser.add_mutually_exclusive_group()
    owner_action.add_argument(
        "--resolve-pending",
        metavar="REQUEST_ID",
        help="owner-only offline reconciliation; the controller service must be stopped",
    )
    owner_action.add_argument(
        "--repair-torn-audit-tail",
        action="store_true",
        help=(
            "owner-only removal of one non-newline WAL suffix; the controller "
            "service must be stopped"
        ),
    )
    args = parser.parse_args()
    root = ROOT.resolve(strict=True)
    socket_path = args.socket.absolute()
    audit_path = args.audit_log.absolute()
    workspace_path = args.workspace.absolute()
    runtime_parent = socket_path.parent
    authority_root = (
        Path.home() / ".local" / "state" / "tiktok-techjam-2026-track2"
    ).resolve(strict=False)
    authority_state_dir = _repository_authority_state_dir(root)
    _validate_external_layout(
        root, runtime_parent, workspace_path, authority_root
    )
    _validate_runtime_endpoints(socket_path, audit_path)
    if args.repair_torn_audit_tail:
        repository_lock_fd = _acquire_repository_service_lock(root)
        lock_fd: int | None = None
        try:
            lock_fd = _acquire_service_lock(runtime_parent)
            _enforce_service_binding(
                state_dir=authority_state_dir,
                root=root,
                audit_path=audit_path,
                create=False,
            )
            _remove_stale_socket(socket_path)
            print(json.dumps(repair_torn_audit_tail(audit_path), indent=2))
            return 0
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(repository_lock_fd)
    if args.resolve_pending:
        repository_lock_fd = _acquire_repository_service_lock(root)
        lock_fd = None
        try:
            lock_fd = _acquire_service_lock(runtime_parent)
            _enforce_service_binding(
                state_dir=authority_state_dir,
                root=root,
                audit_path=audit_path,
                create=False,
            )
            _remove_stale_socket(socket_path)
            workspace = ArtifactWorkspace(workspace_path)
            with _owner_inner_controller_lock(root):
                response = reconcile_pending_request(
                    root,
                    audit_path,
                    args.resolve_pending,
                    fixed_environment(),
                    workspace,
                )
            print(json.dumps(response, indent=2))
            return 0
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(repository_lock_fd)
    authority = ControllerAuthority(root, workspace_root=workspace_path)
    with ControllerServer(
        socket_path,
        authority,
        audit_path,
        authority_state_dir=authority_state_dir,
    ) as server:
        print(json.dumps({
            "protocol_version": PROTOCOL_VERSION,
            "socket": str(socket_path),
            "audit_log": str(audit_path),
            "workspace": str(workspace_path),
            "allowed_rpc": ["log", "run"],
            "start_and_final_exposed": False,
        }, indent=2), flush=True)
        server.serve_forever(poll_interval=0.25)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ServiceError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
