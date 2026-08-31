#!/usr/bin/env python3
"""Unprivileged run/log client exposed inside the researcher sandbox.

Every new RPC is durably recorded outside the repository before any socket
write.  ``retry`` replays an unresolved request byte-for-byte.  ``recover``
does the same for an unresolved request, but returns the locally persisted,
strictly bound response for a completed request instead of risking a second
consuming RPC.  Server-side write-ahead request handling is still required for
idempotence across a server crash.

The client directory is an owner-private crash-recovery aid, not an authority
boundary: the researcher runs as that owner and can delete its own files, and
the default sandbox home is ephemeral unless the launcher is given a persistent
``--agent-home``.  Exactly-once safety therefore remains enforced by the outer
service's write-ahead state; client persistence prevents accidental new IDs.

The Unix endpoint is authenticated as an owner-only socket in an owner-private
directory, with a same-UID peer check and pre/post-connect inode comparison.
This is deliberately not described as cryptographic server authentication: a
malicious process already running as the same effective UID and able to alter
the private socket directory can still substitute an endpoint.  Service and
client state directories therefore remain part of the trusted outer boundary.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import stat
import struct
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_VERSION = 2
SOCKET_ENV = "TRACK2_CONTROLLER_SOCKET"
# This must cover the controller's fully escaped worst-case completion row
# (currently < 2 MiB) so a durable response can always be received and replayed.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024
MAX_SOLUTION_BYTES = 512 * 1024
CLIENT_STATE_DIRNAME = ".track2-controller-client"
CLIENT_STATE_ENV = "TRACK2_CONTROLLER_CLIENT_STATE"
PENDING_REQUEST = "pending-request.json"
LAST_REQUEST = "last-request.json"
LAST_RESPONSE = "last-response.json"
STATE_LOCK = "client.lock"
WORKSPACE_ENV = "TRACK2_RESEARCHER_WORKSPACE"
WORKSPACE_ID_ENV = "TRACK2_RESEARCHER_WORKSPACE_ID"
WORKSPACE_BINDING_ENV = "TRACK2_RESEARCHER_WORKSPACE_BINDING"
MOUNTED_WORKSPACE = Path("/workspace")
SOLUTION_RE = re.compile(r"Project/solutions/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.py")
CARD_RE = re.compile(
    r"Project/research/attempts/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json"
)


class ClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class SocketIdentity:
    path: Path
    device: int
    inode: int


def _path(value: str, pattern: re.Pattern[str], label: str) -> str:
    if pattern.fullmatch(value) is None or "\\" in value or "\x00" in value:
        raise ClientError(f"{label} is outside the controller allowlist")
    return value


def hash_solution(
    relative: str, *, workspace_root: Path = MOUNTED_WORKSPACE
) -> dict[str, Any]:
    """Hash one safely opened staged solution without controller side effects."""

    relative = _path(relative, SOLUTION_RE, "solution")
    parts = PurePosixPath(relative).parts
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(workspace_root, directory_flags)
    except OSError as exc:
        raise ClientError("researcher workspace cannot be opened safely") from exc
    directory_fd = root_fd
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise ClientError("solution parent path is unavailable or unsafe") from exc
            if directory_fd != root_fd:
                os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        except OSError as exc:
            raise ClientError("solution is unavailable or unsafe") from exc
        try:
            before = os.fstat(file_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or before.st_mode & 0o022
                or not 1 <= before.st_size <= MAX_SOLUTION_BYTES
            ):
                raise ClientError(
                    "solution must be one bounded private unique regular file"
                )

            def read_exact() -> bytes:
                chunks: list[bytes] = []
                total = 0
                while total <= MAX_SOLUTION_BYTES:
                    chunk = os.read(
                        file_fd, min(65536, MAX_SOLUTION_BYTES + 1 - total)
                    )
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                payload = b"".join(chunks)
                if len(payload) > MAX_SOLUTION_BYTES:
                    raise ClientError("solution exceeds its byte limit")
                return payload

            first = read_exact()
            os.lseek(file_fd, 0, os.SEEK_SET)
            second = read_exact()
            after = os.fstat(file_fd)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_nlink,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_nlink,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if (
                first != second
                or len(first) != before.st_size
                or identity_after != identity_before
            ):
                raise ClientError("solution changed while being hashed")
            return {
                "path": relative,
                "sha256": hashlib.sha256(first).hexdigest(),
                "size": len(first),
            }
        finally:
            os.close(file_fd)
    finally:
        if directory_fd != root_fd:
            os.close(directory_fd)
        os.close(root_fd)


def current_workspace_id() -> str:
    workspace_id = os.environ.get(WORKSPACE_ID_ENV)
    if not isinstance(workspace_id, str) or re.fullmatch(
        r"[0-9a-f]{32}", workspace_id
    ) is None:
        raise ClientError(
            "workspace binding is unavailable; use the restricted researcher shell"
        )
    return workspace_id


def current_workspace_binding() -> str:
    binding = os.environ.get(WORKSPACE_BINDING_ENV)
    if not isinstance(binding, str) or re.fullmatch(r"[0-9a-f]{64}", binding) is None:
        raise ClientError(
            "physical workspace binding is unavailable; use the restricted researcher shell"
        )
    return binding


def require_current_workspace(request: dict[str, Any]) -> None:
    if (
        request.get("workspace_id") != current_workspace_id()
        or request.get("workspace_binding") != current_workspace_binding()
    ):
        raise ClientError(
            "persisted controller state belongs to a different researcher workspace"
        )


def build_request(
    command: str,
    *,
    solution: str | None = None,
    card: str | None = None,
    request_id: str | None = None,
    workspace_id: str | None = None,
    workspace_binding: str | None = None,
) -> dict[str, Any]:
    request_id = uuid.uuid4().hex if request_id is None else request_id
    if re.fullmatch(r"[0-9a-f]{32}", request_id) is None:
        raise ClientError("request_id must be 32 lowercase hexadecimal characters")
    workspace_id = current_workspace_id() if workspace_id is None else workspace_id
    if not isinstance(workspace_id, str) or re.fullmatch(
        r"[0-9a-f]{32}", workspace_id
    ) is None:
        raise ClientError(
            "workspace binding is unavailable; use the restricted researcher shell"
        )
    workspace_binding = (
        current_workspace_binding()
        if workspace_binding is None else workspace_binding
    )
    if not isinstance(workspace_binding, str) or re.fullmatch(
        r"[0-9a-f]{64}", workspace_binding
    ) is None:
        raise ClientError("workspace_binding must be 64 lowercase hexadecimal characters")
    request: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "workspace_id": workspace_id,
        "workspace_binding": workspace_binding,
        "command": command,
    }
    if command == "log":
        if solution is not None or card is not None:
            raise ClientError("log takes no artifact paths")
        return request
    if command == "run":
        if solution is None or card is None:
            raise ClientError("run requires both --solution and --card")
        request["solution"] = _path(solution, SOLUTION_RE, "solution")
        request["card"] = _path(card, CARD_RE, "card")
        return request
    raise ClientError("only run and log are available")


def validate_request(request: Any) -> dict[str, Any]:
    """Validate the client-side request schema without ever minting an ID."""

    if not isinstance(request, dict):
        raise ClientError("request must be one object")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or re.fullmatch(
        r"[0-9a-f]{32}", request_id
    ) is None:
        raise ClientError("request_id must be 32 lowercase hexadecimal characters")
    command = request.get("command")
    if command == "log":
        expected = build_request(
            "log", request_id=request_id,
            workspace_id=request.get("workspace_id"),
            workspace_binding=request.get("workspace_binding"),
        )
    elif command == "run":
        expected = build_request(
            "run",
            solution=request.get("solution"),
            card=request.get("card"),
            request_id=request_id,
            workspace_id=request.get("workspace_id"),
            workspace_binding=request.get("workspace_binding"),
        )
    else:
        raise ClientError("only run and log are available")
    if request != expected:
        raise ClientError(f"{command} request has missing or extra fields")
    return expected


def request_bytes(request: dict[str, Any]) -> bytes:
    request = validate_request(request)
    try:
        payload = json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ClientError(f"request is not finite canonical JSON: {exc}") from exc
    if len(payload) > MAX_REQUEST_BYTES:
        raise ClientError("request exceeds its size limit")
    return payload


def _strict_object(data: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ClientError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    def no_constants(value):
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            data, object_pairs_hook=no_duplicates, parse_constant=no_constants
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ClientError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ClientError(f"{label} must be one JSON object")
    return value


def _valid_utf8_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _request_from_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if (
        not payload
        or len(payload) > MAX_REQUEST_BYTES
        or not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
    ):
        raise ClientError(f"{label} is not one bounded newline-terminated request")
    request = validate_request(_strict_object(payload[:-1], label))
    if request_bytes(request) != payload:
        raise ClientError(f"{label} is not the exact canonical request encoding")
    return request


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite_number(value: Any, *, positive: bool) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(converted) and (converted > 0 if positive else converted >= 0)


def validate_response(request: dict[str, Any], response: Any) -> dict[str, Any]:
    """Bind one response to the exact request and one of two fixed envelopes."""

    request = validate_request(request)
    if not isinstance(response, dict):
        raise ClientError("controller response must be one object")
    common = {
        "protocol_version", "request_id", "command", "ok", "returncode"
    }
    execution_keys = common | {
        "stdout", "stderr", "elapsed_seconds", "artifact_commit"
    }
    error_keys = common | {"error", "recovery_required"}
    if frozenset(response) not in {
        frozenset(execution_keys), frozenset(error_keys)
    }:
        raise ClientError("controller response has missing or extra fields")
    if (
        type(response.get("protocol_version")) is not int
        or response["protocol_version"] != PROTOCOL_VERSION
        or response.get("request_id") != request["request_id"]
        or response.get("command") != request["command"]
        or type(response.get("ok")) is not bool
        or type(response.get("returncode")) is not int
        or not -255 <= response["returncode"] <= 255
    ):
        raise ClientError("controller response identity or status is invalid")

    if set(response) == error_keys:
        error = response.get("error")
        if (
            response["ok"] is not False
            or response["returncode"] != 125
            or not _valid_utf8_string(error)
            or not 1 <= len(error) <= MAX_RESPONSE_BYTES
            or type(response.get("recovery_required")) is not bool
        ):
            raise ClientError("controller error envelope is invalid")
        return response

    elapsed = response.get("elapsed_seconds")
    if (
        response["ok"] is not (response["returncode"] == 0)
        or not _valid_utf8_string(response.get("stdout"))
        or not _valid_utf8_string(response.get("stderr"))
        or not _finite_number(elapsed, positive=False)
    ):
        raise ClientError("controller execution envelope is invalid")
    commit = response.get("artifact_commit")
    if request["command"] == "log":
        if commit is not None:
            raise ClientError("log response cannot contain an artifact commit")
    elif (
        not isinstance(commit, dict)
        or set(commit) != {"git_revision", "solution_sha256", "card_sha256"}
        or not _valid_utf8_string(commit.get("git_revision"))
        or re.fullmatch(r"[0-9a-f]{40}", commit["git_revision"]) is None
        or not _valid_sha256(commit.get("solution_sha256"))
        or not _valid_sha256(commit.get("card_sha256"))
    ):
        raise ClientError("run response has an invalid artifact commit")
    return response


def response_bytes(request: dict[str, Any], response: dict[str, Any]) -> bytes:
    response = validate_response(request, response)
    try:
        payload = json.dumps(
            response,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ClientError(f"response is not finite canonical JSON: {exc}") from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ClientError("response exceeds its persistence size limit")
    return payload


def _response_from_bytes(
    request: dict[str, Any], payload: bytes, label: str
) -> dict[str, Any]:
    if (
        not payload
        or len(payload) > MAX_RESPONSE_BYTES
        or not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
    ):
        raise ClientError(f"{label} is not one bounded newline-terminated response")
    response = validate_response(request, _strict_object(payload[:-1], label))
    if response_bytes(request, response) != payload:
        raise ClientError(f"{label} is not the exact canonical response encoding")
    return response


def _socket_identity(socket_path: Path) -> SocketIdentity:
    """Validate an owner-only socket and the filesystem chain containing it."""

    if not socket_path.is_absolute():
        raise ClientError("controller socket path must be absolute")
    normalized = Path(os.path.abspath(os.fspath(socket_path)))
    if normalized != socket_path:
        raise ClientError("controller socket path must use a canonical absolute spelling")
    effective_uid = os.geteuid()
    current = Path(normalized.anchor)
    try:
        root_metadata = current.lstat()
        root_uid = root_metadata.st_uid
    except OSError as exc:
        raise ClientError("controller socket filesystem root is unavailable") from exc
    components = list(normalized.parts[1:])
    if len(components) == 1 and (
        root_uid != effective_uid or stat.S_IMODE(root_metadata.st_mode) != 0o700
    ):
        raise ClientError("controller socket parent must be owner-controlled mode 0700")
    socket_metadata = None
    for index, part in enumerate(components):
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ClientError(f"controller socket path is unavailable: {current}") from exc
        is_socket = index == len(components) - 1
        if stat.S_ISLNK(metadata.st_mode):
            raise ClientError(f"controller socket path traverses a symlink: {current}")
        if is_socket:
            if not stat.S_ISSOCK(metadata.st_mode):
                raise ClientError("controller capability path is not a Unix socket")
            socket_metadata = metadata
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise ClientError(f"controller socket parent is not a directory: {current}")
        if metadata.st_uid not in {root_uid, effective_uid}:
            raise ClientError(
                f"controller socket parent has an untrusted owner: {current}"
            )
        mode = stat.S_IMODE(metadata.st_mode)
        is_immediate_parent = index == len(components) - 2
        if is_immediate_parent and (
            metadata.st_uid != effective_uid or mode != 0o700
        ):
            raise ClientError(
                "controller socket parent must be owner-controlled mode 0700"
            )
        if metadata.st_mode & 0o022:
            # A filesystem-root-owned sticky directory such as /tmp cannot
            # replace an entry owned by this UID.  Other writable ancestors
            # are unsafe.  root_uid may be the overflow UID in a user namespace.
            sticky_boundary = metadata.st_uid == root_uid and bool(
                metadata.st_mode & stat.S_ISVTX
            )
            if not sticky_boundary:
                raise ClientError(
                    f"controller socket parent chain is group/world writable: {current}"
                )
    if socket_metadata is None:
        raise ClientError("controller socket path is invalid")
    if (
        socket_metadata.st_uid != effective_uid
        or stat.S_IMODE(socket_metadata.st_mode) != 0o600
        or socket_metadata.st_nlink != 1
    ):
        raise ClientError("controller socket must be owned by the effective uid mode 0600")
    return SocketIdentity(normalized, socket_metadata.st_dev, socket_metadata.st_ino)


def rpc(
    socket_path: Path,
    request: dict[str, Any],
    timeout_seconds: float = 7 * 3600,
    *,
    exact_request_bytes: bytes | None = None,
) -> dict:
    request = validate_request(request)
    if (
        isinstance(timeout_seconds, bool)
        or not _finite_number(timeout_seconds, positive=True)
    ):
        raise ClientError("controller RPC timeout must be finite and positive")
    payload = request_bytes(request)
    if exact_request_bytes is not None:
        loaded = _request_from_bytes(exact_request_bytes, "persisted request")
        if loaded != request:
            raise ClientError("persisted request bytes differ from the requested RPC")
        payload = exact_request_bytes
    before = _socket_identity(socket_path)
    chunks: list[bytes] = []
    received = 0
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_seconds)
            client.connect(str(before.path))
            if hasattr(socket, "SO_PEERCRED"):
                credentials = client.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
                )
                _peer_pid, peer_uid, _peer_gid = struct.unpack("3i", credentials)
                if peer_uid != os.geteuid():
                    raise ClientError("controller Unix peer uid is not authorized")
            after = _socket_identity(before.path)
            if (after.device, after.inode) != (before.device, before.inode):
                raise ClientError("controller socket changed while connecting")
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                received += len(chunk)
                if received > MAX_RESPONSE_BYTES:
                    raise ClientError("controller response exceeds its size limit")
                chunks.append(chunk)
    except (OSError, TimeoutError) as exc:
        raise ClientError(f"controller RPC failed: {type(exc).__name__}: {exc}") from exc
    raw = b"".join(chunks)
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ClientError("controller response is not one newline-terminated JSON object")
    response = _strict_object(raw[:-1], "controller response")
    return validate_response(request, response)


def _known_repository_roots() -> tuple[Path, ...]:
    # /workspace is the fixed repository spelling in researcher_shell.  It is
    # included unconditionally so a caller cannot redirect state into the repo
    # merely by deleting or overriding environment variables.
    roots: set[Path] = {MOUNTED_WORKSPACE}
    workspace = os.environ.get(WORKSPACE_ENV)
    if workspace:
        candidate = Path(workspace)
        if candidate.is_absolute():
            roots.add(candidate.resolve(strict=False))
    module_path = Path(__file__).resolve(strict=True)
    if len(module_path.parents) >= 3:
        candidate = module_path.parents[2]
        if (candidate / ".git").exists():
            roots.add(candidate)
    current = Path.cwd().resolve(strict=True)
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            roots.add(candidate)
            break
    return tuple(sorted(roots, key=str))


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_state_parent(parent: Path) -> Path:
    if parent != Path(os.path.abspath(os.fspath(parent))):
        raise ClientError("client state path must use a canonical absolute spelling")
    effective_uid = os.geteuid()
    current = Path(parent.anchor)
    try:
        root_metadata = current.lstat()
        root_uid = root_metadata.st_uid
    except OSError as exc:
        raise ClientError("client state filesystem root is unavailable") from exc
    parts = list(parent.parts[1:])
    if not parts and root_uid != effective_uid:
        raise ClientError("client state parent must be owned by the effective uid")
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ClientError(f"client state parent is unavailable: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ClientError(f"client state parent must be a real directory: {current}")
        if metadata.st_uid not in {root_uid, effective_uid}:
            raise ClientError(f"client state parent has an untrusted owner: {current}")
        if metadata.st_mode & 0o022:
            sticky_boundary = metadata.st_uid == root_uid and bool(
                metadata.st_mode & stat.S_ISVTX
            )
            if not sticky_boundary:
                raise ClientError(
                    f"client state parent chain is group/world writable: {current}"
                )
        if index == len(parts) - 1 and metadata.st_uid != effective_uid:
            raise ClientError("client state parent must be owned by the effective uid")
    return parent.resolve(strict=True)


class RequestStore:
    """One owner-private durable request slot plus the last completed request."""

    def __init__(self, state_dir: Path):
        if not state_dir.is_absolute():
            raise ClientError("client state directory must be absolute")
        parent = _validate_state_parent(state_dir.parent)
        normalized = parent / state_dir.name
        for root in _known_repository_roots():
            if normalized == root or normalized.is_relative_to(root):
                raise ClientError("client request state may never live in the repository")
        try:
            normalized.mkdir(mode=0o700)
            _fsync_directory(parent)
        except FileExistsError:
            pass
        metadata = normalized.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ClientError("client state directory must be owned mode 0700")
        self.state_dir = normalized
        self.pending_path = normalized / PENDING_REQUEST
        self.last_path = normalized / LAST_REQUEST
        self.response_path = normalized / LAST_RESPONSE
        self.lock_path = normalized / STATE_LOCK

    @classmethod
    def default(cls) -> "RequestStore":
        configured = os.environ.get(CLIENT_STATE_ENV)
        state_root = Path(configured) if configured else Path.home() / CLIENT_STATE_DIRNAME
        if not state_root.is_absolute():
            raise ClientError("client state path does not resolve to an absolute path")
        parent = _validate_state_parent(state_root.parent)
        state_root = parent / state_root.name
        try:
            state_root.mkdir(mode=0o700)
            _fsync_directory(parent)
        except FileExistsError:
            pass
        metadata = state_root.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ClientError("client state root must be owned mode 0700")
        namespace = (
            f"{current_workspace_id()}-{current_workspace_binding()}"
        )
        return cls(state_root / namespace)

    @contextlib.contextmanager
    def _locked(self):
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.lock_path, flags, 0o600)
        except OSError as exc:
            raise ClientError(f"client state lock is unavailable: {exc}") from exc
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise ClientError("client state lock must be one owned mode-0600 file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _read_private(self, path: Path, maximum_bytes: int = MAX_REQUEST_BYTES) -> bytes:
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NONBLOCK", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
        except FileNotFoundError as exc:
            raise ClientError(f"persisted request is unavailable: {path.name}") from exc
        except OSError as exc:
            raise ClientError(f"persisted request is unsafe: {path.name}: {exc}") from exc
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or not 1 <= metadata.st_size <= maximum_bytes
            ):
                raise ClientError(
                    f"persisted request must be one owned bounded mode-0600 file: {path.name}"
                )
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    raise ClientError("persisted request changed while being read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise ClientError("persisted request grew while being read")
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _write_atomic(self, path: Path, payload: bytes) -> None:
        temporary = self.state_dir / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(temporary, flags, 0o600)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise ClientError("client request persistence made no progress")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, path)
            _fsync_directory(self.state_dir)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _present(path: Path) -> bool:
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        return True

    def persist_new(self, request: dict[str, Any]) -> bytes:
        request = validate_request(request)
        require_current_workspace(request)
        payload = request_bytes(request)
        with self._locked():
            if self._present(self.pending_path):
                raise ClientError(
                    "an unresolved request already exists; use retry or recover"
                )
            if self._present(self.last_path):
                last_payload = self._read_private(self.last_path)
                last_request = _request_from_bytes(last_payload, self.last_path.name)
                require_current_workspace(last_request)
                response_payload = self._read_private(
                    self.response_path, MAX_RESPONSE_BYTES
                )
                _response_from_bytes(
                    last_request, response_payload, self.response_path.name
                )
            elif self._present(self.response_path):
                raise ClientError("completed response exists without its exact request")
            self._write_atomic(self.pending_path, payload)
        return payload

    def load(
        self, *, recover_completed: bool
    ) -> tuple[dict[str, Any], bytes, Path, dict[str, Any] | None]:
        with self._locked():
            if self._present(self.pending_path):
                selected = self.pending_path
            elif recover_completed and self._present(self.last_path):
                selected = self.last_path
            else:
                action = "recover" if recover_completed else "retry"
                raise ClientError(f"there is no persisted request available to {action}")
            payload = self._read_private(selected)
            request = _request_from_bytes(payload, selected.name)
            require_current_workspace(request)
            cached_response = None
            if self._present(self.response_path):
                cached_payload = self._read_private(
                    self.response_path, MAX_RESPONSE_BYTES
                )
                try:
                    cached_response = _response_from_bytes(
                        request, cached_payload, self.response_path.name
                    )
                except ClientError:
                    # A newly pending request legitimately coexists with the
                    # preceding completed response until its own response is
                    # received.  Only that exact prior pairing may be ignored.
                    if selected != self.pending_path or not self._present(self.last_path):
                        raise
                    previous_payload = self._read_private(self.last_path)
                    previous_request = _request_from_bytes(
                        previous_payload, self.last_path.name
                    )
                    require_current_workspace(previous_request)
                    _response_from_bytes(
                        previous_request, cached_payload, self.response_path.name
                    )
            elif selected == self.last_path:
                raise ClientError("completed request has no durable bound response")
            if selected == self.pending_path and cached_response is not None:
                if cached_response.get("recovery_required") is True:
                    raise ClientError(
                        "indeterminate response cannot complete a pending request"
                    )
                os.replace(self.pending_path, self.last_path)
                _fsync_directory(self.state_dir)
                selected = self.last_path
        return request, payload, selected, cached_response

    def mark_completed(
        self, expected_payload: bytes, response: dict[str, Any]
    ) -> None:
        request = _request_from_bytes(expected_payload, "completed request")
        require_current_workspace(request)
        if response.get("recovery_required") is True:
            raise ClientError("indeterminate response cannot mark a request completed")
        cached_payload = response_bytes(request, response)
        with self._locked():
            if self._present(self.pending_path):
                actual = self._read_private(self.pending_path)
            else:
                if not self._present(self.last_path):
                    raise ClientError("pending request disappeared before completion")
                last = self._read_private(self.last_path)
                if last != expected_payload:
                    raise ClientError("a different request completed concurrently")
                existing = self._read_private(self.response_path, MAX_RESPONSE_BYTES)
                if existing != cached_payload:
                    raise ClientError("completed response changed during retry")
                _response_from_bytes(request, existing, self.response_path.name)
                return
            if actual != expected_payload:
                raise ClientError("pending request changed before completion")
            if self._present(self.last_path):
                last_payload = self._read_private(self.last_path)
                last_request = _request_from_bytes(last_payload, self.last_path.name)
                require_current_workspace(last_request)
            self._write_atomic(self.response_path, cached_payload)
            os.replace(self.pending_path, self.last_path)
            _fsync_directory(self.state_dir)


def issue_request(
    socket_path: Path,
    request: dict[str, Any],
    *,
    store: RequestStore | None = None,
    timeout_seconds: float = 7 * 3600,
) -> dict[str, Any]:
    store = RequestStore.default() if store is None else store
    payload = store.persist_new(request)
    response = rpc(
        socket_path,
        request,
        timeout_seconds=timeout_seconds,
        exact_request_bytes=payload,
    )
    if response.get("recovery_required") is not True:
        store.mark_completed(payload, response)
    return response


def retry_persisted(
    socket_path: Path,
    *,
    recover_completed: bool,
    store: RequestStore | None = None,
    timeout_seconds: float = 7 * 3600,
) -> dict[str, Any]:
    store = RequestStore.default() if store is None else store
    request, payload, selected, cached_response = store.load(
        recover_completed=recover_completed
    )
    if cached_response is not None:
        return cached_response
    if recover_completed and selected == store.pending_path:
        raise ClientError(
            "recover is local-only and cannot replay a pending request; use retry"
        )
    if selected == store.last_path:
        raise ClientError("completed request recovery must never issue a new RPC")
    response = rpc(
        socket_path,
        request,
        timeout_seconds=timeout_seconds,
        exact_request_bytes=payload,
    )
    if (
        selected == store.pending_path
        and response.get("recovery_required") is not True
    ):
        store.mark_completed(payload, response)
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description="Track 2 restricted controller client")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("log", help="read the official controller state")
    run = sub.add_parser("run", help="commit and consume one exact official attempt")
    run.add_argument("--solution", required=True)
    run.add_argument("--card", required=True)
    hash_parser = sub.add_parser(
        "hash-solution", help="hash one safely opened staged solution"
    )
    hash_parser.add_argument("--solution", required=True)
    sub.add_parser(
        "retry",
        help="replay the unresolved durably persisted request byte-for-byte",
    )
    sub.add_parser(
        "recover",
        help=(
            "return only the durable response of the last completed request; "
            "never replay a pending request"
        ),
    )
    args = parser.parse_args()

    if args.command == "hash-solution":
        print(json.dumps(hash_solution(args.solution), indent=2))
        return 0
    socket_value = os.environ.get(SOCKET_ENV)
    if not socket_value:
        raise ClientError(f"{SOCKET_ENV} is not set by the researcher shell")
    socket_path = Path(socket_value)
    if args.command in {"retry", "recover"}:
        response = retry_persisted(
            socket_path, recover_completed=args.command == "recover"
        )
    else:
        request = build_request(
            args.command,
            solution=getattr(args, "solution", None),
            card=getattr(args, "card", None),
        )
        response = issue_request(socket_path, request)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    returncode = response.get("returncode", 125)
    return returncode if isinstance(returncode, int) and 0 <= returncode <= 255 else 125


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClientError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(125) from exc
