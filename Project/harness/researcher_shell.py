#!/usr/bin/env python3
"""Launch the untrusted researcher in a least-privilege bubblewrap view.

The sandbox deliberately does *not* mount the repository root.  It constructs a
small ``/workspace`` containing writable solutions, research, and memory only;
selected documentation and public starter source are read-only.  Git metadata,
controller/reviewer code, results, manifest, raw data, and sanitized data are
absent.  The sole controller capability is a Unix socket plus ``control.py``.

The launcher authenticates the external socket's ownership, mode, inode, and
real parent chain before constructing the mount and again immediately before
launch.  A pathname-only bind still has an unavoidable final TOCTOU interval:
a malicious same-UID process able to race the private runtime directory could
substitute the path before bubblewrap resolves it.  The runtime directory is
therefore trusted outer state, not a cryptographic identity mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BWRAP = Path("/usr/bin/bwrap")
INSIDE_SOCKET = "/run/track2/controller.sock"
INSIDE_CLIENT_STATE = "/tmp/researcher-home/.track2-controller-client"
WORKSPACE_MOUNTS = (
    ("solutions", "/workspace/Project/solutions"),
    ("attempts", "/workspace/Project/research/attempts"),
    ("memory", "/workspace/Project/memory"),
    ("scratch", "/workspace/Project/research/scratch"),
)
READONLY_REPO_FILES = (
    "CLAUDE.md",
    "Project/RESEARCHER_BRIEF.md",
    "Project/PLAN.md",
    "Project/RESEARCH_PROTOCOL.md",
    "Project/research/templates/attempt.template.json",
)
PUBLIC_KIT_FILES = (
    "data.py",
    "evaluate.py",
)
MCP_SERVER = "Project/tools/controller_mcp.py"
MCP_CONFIG = "Project/harness/controller_mcp_config.json"
CLAUDE_ATTESTATION = "Project/harness/claude_runtime.json"
INSIDE_MCP_SERVER = "/control/controller_mcp.py"
INSIDE_MCP_CONFIG = "/control/controller_mcp.json"
CLAUDE_TOOLS = (
    "Read", "Write", "Edit", "Glob", "Grep",
    "mcp__track2_controller__hash_solution",
    "mcp__track2_controller__log",
    "mcp__track2_controller__run",
    "mcp__track2_controller__retry",
    "mcp__track2_controller__recover",
)
MAX_CONSUMING_ARTIFACTS = 300
MAX_PORTFOLIO_VIEW_BYTES = 1024 * 1024
RESEARCHER_TMPFS_BYTES = 2 * 1024 * 1024 * 1024
RESEARCH_BANK_TMPFS_BYTES = 64 * 1024 * 1024
BANK_NOTE_RE = re.compile(
    r"Project/research/bank/notes/[a-z0-9][a-z0-9._-]{0,124}\.md"
)
WORKSPACE_FILE_RULES = {
    "solutions": (
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.py"),
        MAX_CONSUMING_ARTIFACTS,
        512 * 1024,
    ),
    "attempts": (
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json"),
        MAX_CONSUMING_ARTIFACTS,
        256 * 1024,
    ),
    "memory": (
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.md"),
        32,
        512 * 1024,
    ),
    "scratch": (
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.(?:md|json)"),
        128,
        512 * 1024,
    ),
}
WORKSPACE_MANIFEST = ".track2-workspace.json"
WORKSPACE_FORMAT = "track2.researcher-workspace.v2"
WORKSPACE_PORTFOLIO = ".track2-portfolio.json"
WORKSPACE_ID_ENV = "TRACK2_RESEARCHER_WORKSPACE_ID"
WORKSPACE_BINDING_ENV = "TRACK2_RESEARCHER_WORKSPACE_BINDING"
BROAD_HOST_BIND_SOURCES = (Path("/usr"), Path("/etc"))


class ShellError(RuntimeError):
    pass


@dataclass(frozen=True)
class SocketIdentity:
    path: Path
    device: int
    inode: int


def _mkdir_args(path: str, created: set[str]) -> list[str]:
    current = ""
    result: list[str] = []
    for part in Path(path).parts:
        if part == "/":
            current = "/"
            continue
        current = str(Path(current) / part)
        if current not in created:
            result += ["--dir", current]
            created.add(current)
    return result


def _reject_sensitive_broad_bind_overlap(
    paths: tuple[tuple[Path, str], ...],
) -> None:
    """Fail if private state would reappear through /usr or /etc binds."""

    for path, label in paths:
        resolved = path.resolve(strict=False)
        for source in BROAD_HOST_BIND_SOURCES:
            if (
                resolved == source
                or resolved.is_relative_to(source)
                or source.is_relative_to(resolved)
            ):
                raise ShellError(
                    f"{label} overlaps broadly mounted host runtime {source}"
                )


def _real_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ShellError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ShellError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def _real_file(path: Path, label: str, *, allow_symlink: bool = False) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ShellError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        if not allow_symlink:
            raise ShellError(f"{label} cannot be a symlink: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ShellError(f"{label} symlink does not resolve to a file: {path}")
        return resolved
    if not stat.S_ISREG(metadata.st_mode):
        raise ShellError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _real_executable(path: Path, label: str) -> Path:
    resolved = _real_file(path, label, allow_symlink=True)
    if not os.access(resolved, os.X_OK):
        raise ShellError(f"{label} must be executable: {resolved}")
    return resolved


def _attested_claude(path: Path, root: Path) -> Path:
    """Pin the exact owner-reviewed Claude Code binary and version."""

    if path.name != "claude":
        raise ShellError("official restricted researcher executable must be named claude")
    attestation_path = _real_file(
        root / CLAUDE_ATTESTATION, "Claude runtime attestation"
    )
    try:
        raw = attestation_path.read_bytes()
        attestation = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShellError("Claude runtime attestation is unreadable") from exc
    if (
        not isinstance(attestation, dict)
        or set(attestation) != {"format", "version", "sha256", "size"}
        or attestation.get("format") != "track2.claude-runtime.v1"
        or not isinstance(attestation.get("version"), str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+ \(Claude Code\)", attestation["version"])
        is None
        or not isinstance(attestation.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", attestation["sha256"]) is None
        or type(attestation.get("size")) is not int
        or not 1 <= attestation["size"] <= 512 * 1024 * 1024
    ):
        raise ShellError("Claude runtime attestation has the wrong shape")
    resolved = _real_executable(path, "Claude researcher executable")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(resolved, flags)
    except OSError as exc:
        raise ShellError("attested Claude executable cannot be opened safely") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid not in {0, os.geteuid()}
            or before.st_nlink != 1
            or before.st_mode & 0o022
            or before.st_size != attestation["size"]
        ):
            raise ShellError("attested Claude executable metadata is unsafe")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size)
            != (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size)
        ):
            raise ShellError("attested Claude executable changed while hashing")
    finally:
        os.close(fd)
    if digest.hexdigest() != attestation["sha256"]:
        raise ShellError("Claude executable differs from the frozen attestation")
    try:
        completed = subprocess.run(
            [str(resolved), "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ShellError("attested Claude version probe failed") from exc
    if completed.returncode != 0 or completed.stdout.strip() != attestation["version"]:
        raise ShellError("Claude version differs from the frozen attestation")
    current = resolved.lstat()
    if (current.st_dev, current.st_ino, current.st_size) != (
        before.st_dev, before.st_ino, before.st_size
    ):
        raise ShellError("Claude executable changed after attestation")
    return resolved


def _socket_identity(path: Path) -> SocketIdentity:
    if not path.is_absolute():
        raise ShellError("controller socket must be an absolute external path")
    normalized = Path(os.path.abspath(os.fspath(path)))
    if normalized != path:
        raise ShellError("controller socket path must use a canonical absolute spelling")
    effective_uid = os.geteuid()
    current = Path(normalized.anchor)
    try:
        root_metadata = current.lstat()
        root_uid = root_metadata.st_uid
    except OSError as exc:
        raise ShellError("controller socket filesystem root is unavailable") from exc
    parts = list(normalized.parts[1:])
    if len(parts) == 1 and (
        root_uid != effective_uid or stat.S_IMODE(root_metadata.st_mode) != 0o700
    ):
        raise ShellError("controller socket parent must be owner-controlled mode 0700")
    socket_metadata = None
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ShellError(f"controller socket path is unavailable: {current}") from exc
        is_socket = index == len(parts) - 1
        if stat.S_ISLNK(metadata.st_mode):
            raise ShellError(f"controller socket path traverses a symlink: {current}")
        if is_socket:
            if not stat.S_ISSOCK(metadata.st_mode):
                raise ShellError("controller capability path is not a Unix socket")
            socket_metadata = metadata
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise ShellError(f"controller socket parent is not a directory: {current}")
        if metadata.st_uid not in {root_uid, effective_uid}:
            raise ShellError(
                f"controller socket parent has an untrusted owner: {current}"
            )
        mode = stat.S_IMODE(metadata.st_mode)
        if index == len(parts) - 2 and (
            metadata.st_uid != effective_uid or mode != 0o700
        ):
            raise ShellError(
                "controller socket parent must be owner-controlled mode 0700"
            )
        if metadata.st_mode & 0o022:
            sticky_boundary = metadata.st_uid == root_uid and bool(
                metadata.st_mode & stat.S_ISVTX
            )
            if not sticky_boundary:
                raise ShellError(
                    f"controller socket parent chain is group/world writable: {current}"
                )
    if socket_metadata is None:
        raise ShellError("controller socket path is invalid")
    if (
        socket_metadata.st_uid != effective_uid
        or stat.S_IMODE(socket_metadata.st_mode) != 0o600
        or socket_metadata.st_nlink != 1
    ):
        raise ShellError("controller socket must be owned by the effective uid mode 0600")
    return SocketIdentity(normalized, socket_metadata.st_dev, socket_metadata.st_ino)


def _socket_file(path: Path) -> Path:
    return _socket_identity(path).path


def _private_owned_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(os.fspath(path))) != path:
        raise ShellError(f"{label} must use a canonical absolute spelling")
    effective_uid = os.geteuid()
    current = Path(path.anchor)
    try:
        root_metadata = current.lstat()
    except OSError as exc:
        raise ShellError(f"{label} filesystem root is unavailable") from exc
    root_uid = root_metadata.st_uid
    parts = list(path.parts[1:])
    if not parts:
        raise ShellError(f"{label} may not be the filesystem root")
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ShellError(f"{label} path is unavailable: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ShellError(f"{label} path must contain only real directories")
        if metadata.st_uid not in {root_uid, effective_uid}:
            raise ShellError(f"{label} path has an untrusted owner: {current}")
        mode = stat.S_IMODE(metadata.st_mode)
        if metadata.st_mode & 0o022:
            sticky_boundary = metadata.st_uid == root_uid and bool(
                metadata.st_mode & stat.S_ISVTX
            )
            if not sticky_boundary:
                raise ShellError(f"{label} path has a replaceable ancestor: {current}")
        if index == len(parts) - 1 and (
            metadata.st_uid != effective_uid or mode != 0o700
        ):
            raise ShellError(f"{label} must be owner-controlled mode 0700")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ShellError(f"{label} path changed during validation")
    return resolved


def _private_agent_home(path: Path, root: Path, socket_parent: Path) -> Path:
    """Accept only a dedicated, owner-private home outside the repository."""

    home = _private_owned_directory(path, "researcher home")
    authority_root = (
        Path.home() / ".local" / "state" / "tiktok-techjam-2026-track2"
    ).resolve(strict=False)
    if (
        home == root
        or home.is_relative_to(root)
        or root.is_relative_to(home)
        or home == authority_root
        or home.is_relative_to(authority_root)
        or authority_root.is_relative_to(home)
        or home == socket_parent
        or home.is_relative_to(socket_parent)
        or socket_parent.is_relative_to(home)
    ):
        raise ShellError(
            "researcher home must be disjoint from repository, authority, and socket state"
        )
    return home


def _read_workspace_regular(
    workspace: Path, name: str, *, maximum_bytes: int, label: str
) -> tuple[Path, bytes]:
    """Read one root workspace file through a stable no-follow descriptor."""

    root_fd = os.open(
        workspace,
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
                raise ShellError(f"{label} is unsafe")
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
                raise ShellError(f"{label} changed while being read")
            path = workspace / name
            if path.resolve(strict=True) != path:
                raise ShellError(f"{label} path changed while being read")
            return path, first
        finally:
            os.close(fd)
    except OSError as exc:
        raise ShellError(f"{label} is unavailable") from exc
    finally:
        os.close(root_fd)


def _workspace_binding_value(workspace: Path, manifest_payload: bytes) -> str:
    metadata = workspace.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or workspace.resolve(strict=True) != workspace:
        raise ShellError("researcher workspace identity is unsafe")
    value = {
        "format": "track2.workspace-physical-binding.v1",
        "canonical_path": str(workspace),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _researcher_workspace(
    path: Path, root: Path, socket_parent: Path, agent_home: Path | None
) -> tuple[Path, dict, Path, str]:
    """Validate the clean host workspace mounted into the model namespace."""

    workspace = _private_owned_directory(path, "researcher workspace")
    authority_root = (
        Path.home() / ".local" / "state" / "tiktok-techjam-2026-track2"
    ).resolve(strict=False)
    for other in (
        root, socket_parent, authority_root, *((agent_home,) if agent_home else ())
    ):
        if (
            workspace == other
            or workspace.is_relative_to(other)
            or other.is_relative_to(workspace)
        ):
            raise ShellError(
                "researcher workspace must be disjoint from repository, socket, "
                "and agent home"
            )
    try:
        entries = {entry.name: entry for entry in os.scandir(workspace)}
    except OSError as exc:
        raise ShellError("researcher workspace cannot be enumerated safely") from exc
    if set(entries) != set(WORKSPACE_FILE_RULES) | {
        WORKSPACE_MANIFEST, WORKSPACE_PORTFOLIO
    }:
        raise ShellError(
            "researcher workspace must contain exactly solutions, attempts, "
            "memory, scratch, the run-binding manifest, and frozen portfolio"
        )
    _manifest_path, manifest_payload = _read_workspace_regular(
        workspace, WORKSPACE_MANIFEST, maximum_bytes=4096,
        label="researcher workspace run-binding manifest",
    )
    portfolio_path, portfolio_payload = _read_workspace_regular(
        workspace, WORKSPACE_PORTFOLIO,
        maximum_bytes=MAX_PORTFOLIO_VIEW_BYTES,
        label="researcher workspace frozen portfolio",
    )
    try:
        manifest = json.loads(manifest_payload)
        portfolio = json.loads(portfolio_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShellError("researcher workspace run-binding manifest is unreadable") from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {
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
        or manifest["portfolio_sha256"]
        != hashlib.sha256(portfolio_payload).hexdigest()
        or not isinstance(portfolio, dict)
    ):
        raise ShellError("researcher workspace run-binding manifest has the wrong shape")
    effective_uid = os.geteuid()
    for name, (pattern, maximum_files, maximum_bytes) in WORKSPACE_FILE_RULES.items():
        entry = entries[name]
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ShellError(f"researcher workspace {name} is unsafe") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != effective_uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ShellError(
                f"researcher workspace {name} must be a real owner-only directory"
            )
        try:
            children = list(os.scandir(entry.path))
        except OSError as exc:
            raise ShellError(
                f"researcher workspace {name} cannot be enumerated safely"
            ) from exc
        if len(children) > maximum_files:
            raise ShellError(f"researcher workspace {name} exceeds its file limit")
        total = 0
        for child in children:
            if pattern.fullmatch(child.name) is None:
                raise ShellError(
                    f"researcher workspace {name} contains a forbidden filename"
                )
            try:
                child_metadata = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise ShellError(
                    f"researcher workspace {name} contains an unsafe entry"
                ) from exc
            if (
                not stat.S_ISREG(child_metadata.st_mode)
                or stat.S_ISLNK(child_metadata.st_mode)
                or child_metadata.st_uid != effective_uid
                or child_metadata.st_nlink != 1
                or child_metadata.st_mode & 0o022
                or child_metadata.st_size > maximum_bytes
            ):
                raise ShellError(
                    f"researcher workspace {name} contains a non-private, linked, "
                    "special, or oversized file"
                )
            total += child_metadata.st_size
        if total > maximum_files * maximum_bytes:
            raise ShellError(f"researcher workspace {name} exceeds its byte limit")
    return (
        workspace,
        manifest,
        portfolio_path,
        _workspace_binding_value(workspace, manifest_payload),
    )


def _base_command() -> list[str]:
    return [
        str(BWRAP),
        "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--disable-userns", "--die-with-parent", "--new-session",
        "--clearenv",
        "--setenv", "PATH", "/usr/bin:/bin:/control",
        "--setenv", "HOME", "/tmp/researcher-home",
        "--setenv", "PYTHONNOUSERSITE", "1",
        "--setenv", "NO_COLOR", "1",
        "--setenv", "DISABLE_UPDATES", "1",
        "--setenv", "DISABLE_AUTOUPDATER", "1",
        "--setenv", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1",
        "--setenv", "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL", "1",
        "--setenv", "CLAUDE_CODE_DISABLE_ARTIFACT", "1",
        "--setenv", "CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1",
        "--setenv", "ENABLE_CLAUDEAI_MCP_SERVERS", "false",
        "--setenv", "CLAUDE_CODE_AUTO_CONNECT_IDE", "false",
        "--setenv", "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS", "1",
        "--setenv", "TRACK2_CONTROLLER_SOCKET", INSIDE_SOCKET,
        "--setenv", "TRACK2_CONTROLLER_CLIENT_STATE", INSIDE_CLIENT_STATE,
        "--setenv", "TRACK2_RESEARCHER_WORKSPACE", "/workspace",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--ro-bind", "/etc", "/etc",
        "--proc", "/proc",
        "--dev", "/dev",
        "--remount-ro", "/dev",
        "--size", str(RESEARCHER_TMPFS_BYTES), "--tmpfs", "/tmp",
    ]


def hardened_claude_args(agent_args: list[str]) -> list[str]:
    """Return one exact invocation; no resume, subcommand, or option is caller-set."""

    if agent_args:
        raise ShellError(
            "official researcher launch accepts no Claude arguments; model and "
            "boundary are owner-fixed"
        )
    tools = ",".join(CLAUDE_TOOLS)
    return [
        "--restricted",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--mcp-config", INSIDE_MCP_CONFIG,
        "--tools", tools,
        "--allowedTools", tools,
        "--disallowedTools", "Bash,WebFetch,WebSearch,NotebookEdit",
        "--permission-mode", "dontAsk",
        "--no-chrome",
        "--model", "fable",
    ]


def _frozen_bank_files(root: Path) -> tuple[str, ...]:
    catalog_relative = "Project/research/bank/catalog.json"
    catalog_path = _real_file(root / catalog_relative, "research bank catalog")
    try:
        raw = catalog_path.read_bytes()
        catalog = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShellError("research bank catalog is unreadable") from exc
    if (
        not isinstance(catalog, dict)
        or set(catalog) != {"schema_version", "benchmark", "claims"}
        or catalog.get("schema_version") != 1
        or catalog.get("benchmark") != "KuaiRand-Pure"
        or not isinstance(catalog.get("claims"), list)
        or not 1 <= len(catalog["claims"]) <= 256
    ):
        raise ShellError("research bank catalog has the wrong frozen shape")
    notes = set()
    for claim in catalog["claims"]:
        note = claim.get("note_path") if isinstance(claim, dict) else None
        if not isinstance(note, str) or BANK_NOTE_RE.fullmatch(note) is None:
            raise ShellError("research bank catalog has an unsafe note path")
        _real_file(root / note, "research bank note")
        notes.add(note)
    if not 1 <= len(notes) <= 64:
        raise ShellError("research bank catalog selects an invalid note count")
    return (catalog_relative, *sorted(notes))


def build_command(
    *,
    root: Path,
    socket_path: Path,
    agent_executable: Path,
    agent_args: list[str],
    workspace_root: Path,
    agent_home: Path | None = None,
    restrict_claude: bool = False,
) -> list[str]:
    """Construct the exact researcher mount namespace; perform no execution."""

    root = _real_directory(root, "repository root")
    socket_path = _socket_file(socket_path)
    socket_parent = socket_path.parent
    if (
        socket_parent == root
        or socket_parent.is_relative_to(root)
        or root.is_relative_to(socket_parent)
    ):
        raise ShellError("controller socket runtime must be disjoint from the repository")
    resolved_agent_home = (
        _private_agent_home(agent_home, root, socket_parent)
        if agent_home is not None else None
    )
    (
        resolved_workspace, workspace_manifest, portfolio_path,
        workspace_binding,
    ) = _researcher_workspace(
        workspace_root, root, socket_parent, resolved_agent_home
    )
    authority_root = (
        Path.home() / ".local" / "state" / "tiktok-techjam-2026-track2"
    ).resolve(strict=False)
    sensitive_paths: list[tuple[Path, str]] = [
        (root, "repository root"),
        (socket_parent, "controller runtime"),
        (resolved_workspace, "researcher workspace"),
        (authority_root, "controller authority state"),
    ]
    if resolved_agent_home is not None:
        sensitive_paths.append((resolved_agent_home, "researcher home"))
    _reject_sensitive_broad_bind_overlap(tuple(sensitive_paths))
    control = _real_file(root / "Project" / "tools" / "control.py", "control client")
    mcp_server = _real_file(root / MCP_SERVER, "controller MCP bridge")
    mcp_config = _real_file(root / MCP_CONFIG, "controller MCP config")
    if not BWRAP.is_file():
        raise ShellError("/usr/bin/bwrap is required")

    command = _base_command()
    command += [
        "--setenv", WORKSPACE_ID_ENV, workspace_manifest["workspace_id"]
    ]
    command += ["--setenv", WORKSPACE_BINDING_ENV, workspace_binding]
    created = {"/usr", "/lib", "/lib64", "/bin", "/etc", "/proc", "/dev", "/tmp"}
    for destination in (
        "/workspace/Project",
        "/workspace/kuairand-starter-kit",
        "/control",
        "/tmp/researcher-home",
        "/agent/bin",
    ):
        command += _mkdir_args(destination, created)

    command += _mkdir_args("/run", created)
    command += ["--perms", "0700", "--dir", "/run/track2"]
    created.add("/run/track2")

    if resolved_agent_home is not None:
        command += ["--bind", str(resolved_agent_home), "/tmp/researcher-home"]

    for source_name, destination in WORKSPACE_MOUNTS:
        source = _real_directory(
            resolved_workspace / source_name,
            f"researcher workspace {source_name}",
        )
        command += _mkdir_args(str(Path(destination).parent), created)
        command += ["--bind", str(source), destination]

    # Give the evidence corpus its own empty mount, then add only catalog-bound
    # files.  This hides every legacy/unreferenced bank sibling.
    bank_destination = "/workspace/Project/research/bank"
    command += _mkdir_args(bank_destination, created)
    command += [
        "--size", str(RESEARCH_BANK_TMPFS_BYTES), "--tmpfs", bank_destination
    ]
    if restrict_claude:
        for relative in _frozen_bank_files(root):
            source = _real_file(root / relative, f"frozen bank file {relative}")
            destination = f"/workspace/{relative}"
            command += _mkdir_args(str(Path(destination).parent), created)
            command += ["--ro-bind", str(source), destination]
    else:
        for relative in (
            "Project/research/bank/catalog.template.json",
            "Project/research/bank/notes/README.md",
        ):
            source = _real_file(root / relative, f"frozen bank file {relative}")
            destination = f"/workspace/{relative}"
            command += _mkdir_args(str(Path(destination).parent), created)
            command += ["--ro-bind", str(source), destination]

    for relative in READONLY_REPO_FILES:
        source = _real_file(root / relative, f"read-only {relative}")
        destination = f"/workspace/{relative}"
        command += _mkdir_args(str(Path(destination).parent), created)
        command += ["--ro-bind", str(source), destination]

    command += _mkdir_args("/workspace/Project/research", created)
    command += [
        "--ro-bind", str(portfolio_path),
        "/workspace/Project/research/portfolio.json",
    ]

    for name in PUBLIC_KIT_FILES:
        source = _real_file(
            root / "kuairand-starter-kit" / name, f"public starter file {name}"
        )
        command += [
            "--ro-bind", str(source), f"/workspace/kuairand-starter-kit/{name}"
        ]

    command += ["--ro-bind", str(control), "/control/control.py"]
    command += ["--ro-bind", str(mcp_server), INSIDE_MCP_SERVER]
    command += ["--ro-bind", str(mcp_config), INSIDE_MCP_CONFIG]
    command += ["--ro-bind", str(socket_path), INSIDE_SOCKET]

    resolved_agent = (
        _attested_claude(agent_executable, root)
        if restrict_claude
        else _real_executable(agent_executable, "researcher executable")
    )
    if resolved_agent.is_relative_to(Path("/usr")):
        inside_agent = str(resolved_agent)
    elif resolved_agent == control:
        inside_agent = "/control/control.py"
    else:
        inside_agent = "/agent/bin/researcher"
        command += ["--ro-bind", str(resolved_agent), inside_agent]

    # Bubblewrap applies mount operations in order.  Every source bind must be
    # installed before the root is remounted read-only, otherwise an agent
    # executable outside /usr cannot be created at its in-sandbox destination.
    command += ["--remount-ro", "/workspace/Project/research/bank"]
    command += ["--remount-ro", "/"]

    if restrict_claude:
        if resolved_agent == control:
            raise ShellError("the control client is not a Claude researcher executable")
        agent_args = hardened_claude_args(agent_args)
    command += ["--chdir", "/workspace", inside_agent, *agent_args]
    return command


def boundary_manifest(command: list[str]) -> dict:
    """Summarize the intentional boundary without exposing environment secrets."""

    writable = []
    readonly = []
    for index, item in enumerate(command):
        if item == "--bind" and index + 2 < len(command):
            writable.append(command[index + 2])
        if item == "--ro-bind" and index + 2 < len(command):
            readonly.append(command[index + 2])
    return {
        "writable_repo_paths": [
            path for path in writable if path.startswith("/workspace/")
        ],
        "controller_tools": ["hash_solution", "log", "run", "retry", "recover"],
        # The model client itself needs network access. Arbitrary Bash and web
        # tools are absent in the hard Claude profile; OS-level egress remains
        # an explicitly reported residual rather than a false claim.
        "researcher_network_shared": "--unshare-net" not in command,
        "persistent_agent_home": "/tmp/researcher-home" in writable,
        "claude_hard_tool_surface": (
            "--restricted" in command
            and "--strict-mcp-config" in command
            and "Bash,WebFetch,WebSearch,NotebookEdit" in command
        ),
        "research_bank_readonly": any(
            path == "/workspace/Project/research/bank"
            or path.startswith("/workspace/Project/research/bank/")
            for path in readonly
        ),
        "socket_inside": INSIDE_SOCKET,
        "git_history_mounted": any(path.endswith("/.git") for path in readonly + writable),
        "raw_or_sanitized_data_mounted": any(
            "KuaiRand-Pure" in path for path in readonly + writable
        ),
        "results_mounted": any("/Project/results" in path for path in readonly + writable),
        "harness_mounted": any("/Project/harness" in path for path in readonly + writable),
        "manifest_mounted": any(path.endswith("/Project/manifest.json") for path in readonly),
    }


def _preflight_service_binding(
    root: Path, socket_path: Path, workspace_id: str, workspace_binding: str
) -> None:
    """Prove the live service is bound to this exact logical workspace."""

    client_path = root / "Project" / "tools" / "control.py"
    spec = importlib.util.spec_from_file_location(
        "track2_control_for_shell_preflight", client_path
    )
    if spec is None or spec.loader is None:
        raise ShellError("controller client cannot be loaded for launch preflight")
    client = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = client
    try:
        spec.loader.exec_module(client)
        request = client.build_request(
            "log", request_id=secrets.token_hex(16),
            workspace_id=workspace_id,
            workspace_binding=workspace_binding,
        )
        response = client.rpc(
            socket_path,
            request,
            timeout_seconds=30.0,
            exact_request_bytes=client.request_bytes(request),
        )
    except Exception as exc:
        raise ShellError(
            "live controller service is not bound to this researcher workspace"
        ) from exc
    if response.get("ok") is not True:
        raise ShellError(
            "live controller service refused this researcher workspace binding"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Track 2 restricted researcher shell")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument(
        "--agent-executable", type=Path, required=True,
        help="owner-selected Claude executable; paths outside /usr are mounted read-only",
    )
    parser.add_argument(
        "--agent-home", type=Path,
        help=(
            "required dedicated mode-0700 home outside the repo for agent auth/state; "
            "never point this at a normal home containing unrelated conversations"
        ),
    )
    parser.add_argument(
        "--workspace", required=True, type=Path,
        help=(
            "dedicated external mode-0700 run workspace containing exactly "
            "solutions, attempts, memory, and scratch mode-0700 directories"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("agent_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    agent_args = list(args.agent_args)
    if agent_args and agent_args[0] == "--":
        agent_args = agent_args[1:]
    if args.agent_home is None:
        raise ShellError("official researcher launch requires a persistent --agent-home")
    socket_identity = _socket_identity(args.socket)
    root = ROOT.resolve(strict=True)
    resolved_agent_home = _private_agent_home(
        args.agent_home, root, socket_identity.path.parent
    )
    (
        workspace_before, workspace_manifest, _portfolio_path,
        workspace_binding,
    ) = _researcher_workspace(
        args.workspace, root, socket_identity.path.parent, resolved_agent_home
    )
    command = build_command(
        root=root,
        socket_path=socket_identity.path,
        agent_executable=args.agent_executable,
        agent_args=agent_args,
        workspace_root=args.workspace,
        agent_home=args.agent_home,
        restrict_claude=True,
    )
    manifest = boundary_manifest(command)
    if manifest != {
        "writable_repo_paths": [
            "/workspace/Project/solutions",
            "/workspace/Project/research/attempts",
            "/workspace/Project/memory",
            "/workspace/Project/research/scratch",
        ],
        "controller_tools": ["hash_solution", "log", "run", "retry", "recover"],
        "researcher_network_shared": True,
        "persistent_agent_home": args.agent_home is not None,
        "claude_hard_tool_surface": True,
        "research_bank_readonly": True,
        "socket_inside": INSIDE_SOCKET,
        "git_history_mounted": False,
        "raw_or_sanitized_data_mounted": False,
        "results_mounted": False,
        "harness_mounted": False,
        "manifest_mounted": False,
    }:
        raise ShellError(f"researcher boundary manifest is unsafe: {manifest}")
    current_socket = _socket_identity(socket_identity.path)
    if current_socket != socket_identity:
        raise ShellError("controller socket changed while preparing the researcher shell")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    _preflight_service_binding(
        root, socket_identity.path, workspace_manifest["workspace_id"],
        workspace_binding,
    )
    (
        workspace_after, manifest_after, _portfolio_after,
        binding_after,
    ) = _researcher_workspace(
        args.workspace, root, socket_identity.path.parent, resolved_agent_home
    )
    if (
        workspace_after != workspace_before
        or manifest_after != workspace_manifest
        or binding_after != workspace_binding
    ):
        raise ShellError("researcher workspace changed during launch preflight")
    current_socket = _socket_identity(socket_identity.path)
    if current_socket != socket_identity:
        raise ShellError("controller socket changed during launch preflight")
    try:
        completed = subprocess.run(command, check=False, env={}, umask=0o077)
    except OSError as exc:
        raise ShellError(f"could not launch researcher sandbox: {exc}") from exc
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ShellError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(125) from exc
