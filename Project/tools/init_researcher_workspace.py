#!/usr/bin/env python3
"""Owner-only creation of one fresh, run-bound researcher workspace."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/usr/bin/python3")
GIT = Path("/usr/bin/git")
FORMAT = "track2.researcher-workspace.v2"
MANIFEST = ".track2-workspace.json"
PORTFOLIO = ".track2-portfolio.json"
SUBDIRS = ("solutions", "attempts", "memory", "scratch")


class InitError(RuntimeError):
    pass


def _load_controller_service():
    path = ROOT / "Project" / "harness" / "controller_service.py"
    spec = importlib.util.spec_from_file_location(
        "track2_controller_service_for_workspace_init", path
    )
    if spec is None or spec.loader is None:
        raise InitError("controller service verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise InitError("controller service verifier identity changed")
    return module


@contextlib.contextmanager
def _controller_lock():
    """Serialize the state/HEAD snapshot with every official transition."""

    path = ROOT / "Project" / "results" / ".controller.lock"
    parent = path.parent
    metadata = parent.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or parent.resolve(strict=True) != parent
    ):
        raise InitError("controller lock parent is unsafe")
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
            raise InitError("controller lock file is unsafe")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InitError("another controller transition is active") from exc
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _private_parent(target: Path) -> Path:
    if not target.is_absolute() or Path(os.path.abspath(os.fspath(target))) != target:
        raise InitError("workspace path must use a canonical absolute spelling")
    parent = target.parent
    effective_uid = os.geteuid()
    current = Path(parent.anchor)
    root_metadata = current.lstat()
    root_uid = root_metadata.st_uid
    for index, part in enumerate(parent.parts[1:]):
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise InitError("workspace parent path must contain only real directories")
        if metadata.st_uid not in {root_uid, effective_uid}:
            raise InitError("workspace parent path has an untrusted owner")
        if metadata.st_mode & 0o022:
            sticky_root_boundary = (
                metadata.st_uid == root_uid and bool(metadata.st_mode & stat.S_ISVTX)
            )
            if not sticky_root_boundary:
                raise InitError("workspace parent path has a replaceable ancestor")
        if index == len(parent.parts[1:]) - 1 and (
            metadata.st_uid != effective_uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise InitError("workspace parent must be owner-controlled mode 0700")
    if parent.resolve(strict=True) != parent:
        raise InitError("workspace parent path changed while validating")
    return parent


def _run(command: list[str]) -> bytes:
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/tmp"),
            "PYTHONNOUSERSITE": "1",
            "NO_COLOR": "1",
        },
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-4000:].decode(
            "utf-8", errors="replace"
        )
        raise InitError(f"owner state check failed: {detail}")
    return result.stdout


def _strict_object(payload: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InitError("state response contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InitError("state response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise InitError("state response must be one JSON object")
    return value


def create_workspace(
    target: Path, *, recover_workspace_id: str | None = None
) -> dict:
    target = Path(target)
    if recover_workspace_id is not None and re.fullmatch(
        r"[0-9a-f]{32}", recover_workspace_id
    ) is None:
        raise InitError("recovery workspace_id must be 32 lowercase hexadecimal characters")
    if not target.is_absolute() or Path(os.path.abspath(os.fspath(target))) != target:
        raise InitError("workspace path must use a canonical absolute spelling")
    parent = _private_parent(target)
    authority_root = (
        Path.home() / ".local" / "state" / "tiktok-techjam-2026-track2"
    ).resolve(strict=False)
    for other, label in ((ROOT.resolve(strict=True), "repository"), (authority_root, "authority")):
        if target == other or target.is_relative_to(other) or other.is_relative_to(target):
            raise InitError(f"workspace must be disjoint from {label} state")
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    else:
        raise InitError("workspace target already exists; refusing reuse")

    with _controller_lock():
        state = _strict_object(_run([
            str(PYTHON), "-I", str(ROOT / "Project/harness/iterate.py"),
            "_admission-state",
        ]))
        if (
            state.get("official_run_started") is not True
            or state.get("state") not in {
                "ACTIVE", "TERMINAL_DUE", "TERMINAL"
            }
            or type(state.get("official_iterations")) is not int
            or not 0 <= state["official_iterations"] <= 50
            or state.get("open_attempt") is not False
            or not isinstance(state.get("run_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", state["run_id"]) is None
            or not isinstance(state.get("run_start_git_revision"), str)
            or re.fullmatch(
                r"[0-9a-f]{40}", state["run_start_git_revision"]
            ) is None
            or not isinstance(state.get("portfolio"), dict)
        ):
            raise InitError(
                "workspace creation requires one safely closed official run state"
            )
        head = _run([
            str(GIT), "-c", "core.hooksPath=/dev/null", "rev-parse", "HEAD"
        ]).decode("ascii", errors="strict").strip()
        if re.fullmatch(r"[0-9a-f]{40}", head) is None:
            raise InitError("repository HEAD is malformed")
        base_revision = state["run_start_git_revision"]
        verifier = _load_controller_service()
        try:
            verifier._validate_controller_commit_chain(
                ROOT, base_revision, head, verifier.fixed_environment()
            )
        except Exception as exc:
            raise InitError(
                "repository contains a non-controller commit after run start"
            ) from exc

        try:
            portfolio_payload = json.dumps(
                state["portfolio"], sort_keys=True, indent=2,
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise InitError("frozen portfolio cannot be copied exactly") from exc
        if not 1 <= len(portfolio_payload) <= verifier.outer_policy.MAX_PORTFOLIO_VIEW_BYTES:
            raise InitError("frozen portfolio copy exceeds its fixed byte bound")
        manifest = {
            "format": FORMAT,
            "workspace_id": recover_workspace_id or secrets.token_hex(16),
            "run_id": state["run_id"],
            "repository_head": base_revision,
            "portfolio_sha256": hashlib.sha256(portfolio_payload).hexdigest(),
            "created_ns": time.time_ns(),
        }
        manifest_payload = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8") + b"\n"

        previous_umask = os.umask(0o077)
        try:
            os.mkdir(target, 0o700)
            for name in SUBDIRS:
                os.mkdir(target / name, 0o700)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            for name, payload in (
                (PORTFOLIO, portfolio_payload), (MANIFEST, manifest_payload)
            ):
                fd = os.open(target / name, flags, 0o600)
                try:
                    view = memoryview(payload)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise InitError(f"workspace {name} write made no progress")
                        view = view[written:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
            for directory in (target, parent):
                directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            os.umask(previous_umask)
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create one fresh Track 2 researcher workspace after start-run"
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument(
        "--recover-workspace-id",
        help=(
            "reuse the exact durable workspace_id only when rebuilding a lost "
            "workspace with the controller service stopped"
        ),
    )
    args = parser.parse_args()
    manifest = create_workspace(
        args.workspace, recover_workspace_id=args.recover_workspace_id
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InitError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
