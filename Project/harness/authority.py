"""External authority for an append-only, HMAC-authenticated JSONL journal.

The official controller and its journal live in the repository.  This module
keeps the journal secret and authoritative tail *outside* the repository so a
repository checkout cannot silently rewrite, truncate, or replace history.

Crash ordering is deliberate:

1. append and fsync one or more individually HMAC-authenticated journal rows;
2. atomically replace and directory-fsync the external head anchor.

If the process dies between those steps, the journal may contain a suffix past
the external head.  Such a suffix may be reconciled only when every byte is a
canonical JSON line and every row has the next sequence number, previous HMAC,
and a valid HMAC under the external secret.  Truncation, mutation, unsigned
rows, partial lines, and reordered rows fail closed.

This module is intentionally independent of benchmark data and model code.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


FORMAT = "track2.external-journal-authority.v1"
ROW_AUTHORITY_KEY = "journal_authority"
DOMAIN = b"track2-journal-row-v1\x00"
HEAD_DOMAIN = b"track2-journal-head-v1\x00"
KEY_BYTES = 32
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class AuthorityError(RuntimeError):
    """The external authority or protected journal failed closed."""


class CrashSuffixPending(AuthorityError):
    """A valid post-head suffix exists and requires explicit reconciliation."""

    def __init__(self, pending_rows: int):
        self.pending_rows = pending_rows
        super().__init__(
            f"journal contains {pending_rows} HMAC-valid crash-suffix row(s); "
            "explicit reconciliation is required"
        )


@dataclass(frozen=True)
class Verification:
    """Summary of a verified journal and external authority head."""

    journal_id: str
    protected_prefix_bytes: int
    protected_prefix_sha256: str
    protected_rows: int
    tail_hmac_sha256: str
    pending_rows: int = 0
    reconciled_rows: int = 0


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuthorityError(f"value is not canonical finite JSON: {exc}") from exc


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate key {key!r}")
            out[key] = value
        return out

    def no_constants(value: str):
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_constants,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AuthorityError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AuthorityError(f"{label} must be one JSON object")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_hex(key: bytes, domain: bytes, payload: bytes) -> str:
    return hmac.new(key, domain + payload, hashlib.sha256).hexdigest()


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise AuthorityError("durable write made no progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class JournalAuthority:
    """HMAC journal authority backed by a private external state directory."""

    KEY_NAME = "journal-authority.key"
    HEAD_NAME = "journal-authority-head.json"
    LOCK_NAME = "journal-authority.lock"

    def __init__(self, *, journal_path: Path, state_dir: Path, repo_root: Path):
        self.repo_root = repo_root.resolve(strict=True)
        self.journal_path = journal_path.resolve(strict=False)
        self.state_dir = state_dir.resolve(strict=False)
        if not self.repo_root.is_dir():
            raise AuthorityError("repo_root must be an existing directory")
        if not self.journal_path.is_relative_to(self.repo_root):
            raise AuthorityError("protected journal must be inside repo_root")
        if self.state_dir == self.repo_root or self.state_dir.is_relative_to(self.repo_root):
            raise AuthorityError("authority state directory must be outside the repository")
        if self.journal_path.exists() and self.journal_path.is_symlink():
            raise AuthorityError("protected journal may not be a symlink")
        self.key_path = self.state_dir / self.KEY_NAME
        self.head_path = self.state_dir / self.HEAD_NAME
        self.lock_path = self.state_dir / self.LOCK_NAME

    @classmethod
    def create(
        cls, *, journal_path: Path, state_dir: Path, repo_root: Path
    ) -> "JournalAuthority":
        """Create a new authority and freeze every current journal byte as prefix.

        Creation is a one-time, pre-run operation.  Existing key/head material is
        never overwritten because doing so would permit history reset.
        """

        authority = cls(
            journal_path=journal_path, state_dir=state_dir, repo_root=repo_root
        )
        if authority.state_dir.exists() and authority.state_dir.is_symlink():
            raise AuthorityError("authority state directory may not be a symlink")
        authority.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        authority._validate_state_directory()
        with authority._locked():
            if authority.key_path.exists() or authority.head_path.exists():
                raise AuthorityError("external journal authority already exists")
            key = secrets.token_bytes(KEY_BYTES)
            authority._create_private_file(authority.key_path, key)
            prefix = authority._read_journal_bytes(allow_missing=True)
            if prefix and not prefix.endswith(b"\n"):
                raise AuthorityError(
                    "existing journal prefix must end at a complete newline boundary"
                )
            prefix_sha = _sha256(prefix)
            head = {
                "format": FORMAT,
                "journal_id": uuid.uuid4().hex,
                "journal_path": str(authority.journal_path),
                "repo_root": str(authority.repo_root),
                "key_sha256": _sha256(key),
                "protected_prefix_bytes": len(prefix),
                "protected_prefix_sha256": prefix_sha,
                "protected_rows": 0,
                "tail_hmac_sha256": prefix_sha,
                "updated_unix_ns": time.time_ns(),
            }
            authority._write_head_atomic(authority._sign_head(head, key))
            # A concurrent non-authority writer cannot be made safe.  Detect a
            # create-time race and leave the new authority fail-closed.
            if authority._read_journal_bytes(allow_missing=True) != prefix:
                raise AuthorityError("journal changed while its protected prefix was frozen")
        return authority

    @classmethod
    def open_existing(
        cls, *, journal_path: Path, state_dir: Path, repo_root: Path
    ) -> "JournalAuthority":
        """Open an existing authority; missing material is never recreated."""

        authority = cls(
            journal_path=journal_path, state_dir=state_dir, repo_root=repo_root
        )
        authority._validate_state_directory()
        with authority._locked():
            key = authority._read_key()
            authority._read_head(key)
        return authority

    def verify(self, *, reconcile_crash_suffix: bool = False) -> Verification:
        """Verify the journal and optionally advance over a valid crash suffix."""

        with self._locked():
            key = self._read_key()
            head = self._read_head(key)
            result = self._verify_journal(head, key)
            if result.pending_rows and not reconcile_crash_suffix:
                raise CrashSuffixPending(result.pending_rows)
            if result.pending_rows:
                reconciled = result.pending_rows
                head = self._advanced_head(
                    head,
                    protected_rows=result.protected_rows,
                    tail_hmac=result.tail_hmac_sha256,
                    key=key,
                )
                self._write_head_atomic(head)
                return Verification(
                    journal_id=result.journal_id,
                    protected_prefix_bytes=result.protected_prefix_bytes,
                    protected_prefix_sha256=result.protected_prefix_sha256,
                    protected_rows=result.protected_rows,
                    tail_hmac_sha256=result.tail_hmac_sha256,
                    pending_rows=0,
                    reconciled_rows=reconciled,
                )
            return result

    def append(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Authenticate, append, fsync, and externally anchor one row batch."""

        if not rows:
            raise AuthorityError("authority append requires at least one row")
        with self._locked():
            key = self._read_key()
            head = self._read_head(key)
            before = self._verify_journal(head, key)
            if before.pending_rows:
                head = self._advanced_head(
                    head,
                    protected_rows=before.protected_rows,
                    tail_hmac=before.tail_hmac_sha256,
                    key=key,
                )
                self._write_head_atomic(head)

            signed, payload = self._sign_rows(
                rows,
                journal_id=head["journal_id"],
                first_sequence=int(head["protected_rows"]) + 1,
                previous_hmac=head["tail_hmac_sha256"],
                key=key,
            )
            expected_size = len(self._read_journal_bytes(allow_missing=True))
            self._append_journal(payload, expected_size=expected_size)

            # Re-read from disk before advancing the external anchor.  Only the
            # exact HMAC-valid suffix just prepared is acceptable.
            after = self._verify_journal(head, key)
            if (
                after.pending_rows != len(signed)
                or after.protected_rows != int(head["protected_rows"]) + len(signed)
                or after.tail_hmac_sha256
                != signed[-1][ROW_AUTHORITY_KEY]["hmac_sha256"]
            ):
                raise AuthorityError("journal changed concurrently during authority append")
            new_head = self._advanced_head(
                head,
                protected_rows=after.protected_rows,
                tail_hmac=after.tail_hmac_sha256,
                key=key,
            )
            self._write_head_atomic(new_head)
            return signed

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        if not self.state_dir.is_dir():
            raise AuthorityError("external authority state directory is missing")
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            self._validate_private_fd(fd, self.lock_path, exact_size=None)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _validate_state_directory(self) -> None:
        if not self.state_dir.is_dir() or self.state_dir.is_symlink():
            raise AuthorityError("external authority state directory is missing or unsafe")
        info = self.state_dir.stat()
        if info.st_uid != os.geteuid():
            raise AuthorityError("external authority state directory has the wrong owner")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise AuthorityError("external authority state directory must have mode 0700")

    @staticmethod
    def _validate_private_fd(fd: int, path: Path, exact_size: int | None) -> None:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise AuthorityError(f"authority file is not regular: {path}")
        if info.st_nlink != 1:
            raise AuthorityError(f"authority file must not have hard links: {path}")
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise AuthorityError(f"authority file is not private to its owner: {path}")
        if exact_size is not None and info.st_size != exact_size:
            raise AuthorityError(f"authority file has unexpected size: {path}")

    def _create_private_file(self, path: Path, payload: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_directory(self.state_dir)

    def _read_private_file(self, path: Path, exact_size: int | None = None) -> bytes:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise AuthorityError(f"required external authority file is unavailable: {path}") from exc
        try:
            self._validate_private_fd(fd, path, exact_size)
            chunks = []
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _read_key(self) -> bytes:
        return self._read_private_file(self.key_path, exact_size=KEY_BYTES)

    def _sign_head(self, head: dict[str, Any], key: bytes) -> dict[str, Any]:
        unsigned = dict(head)
        unsigned.pop("head_hmac_sha256", None)
        return {
            **unsigned,
            "head_hmac_sha256": _hmac_hex(key, HEAD_DOMAIN, _canonical(unsigned)),
        }

    def _read_head(self, key: bytes) -> dict[str, Any]:
        raw = self._read_private_file(self.head_path)
        if not raw.endswith(b"\n"):
            raise AuthorityError("external authority head is not a complete line")
        head = _strict_object(raw[:-1], "external authority head")
        if raw != _canonical(head) + b"\n":
            raise AuthorityError("external authority head is not canonical JSON")
        required = {
            "format",
            "journal_id",
            "journal_path",
            "repo_root",
            "key_sha256",
            "protected_prefix_bytes",
            "protected_prefix_sha256",
            "protected_rows",
            "tail_hmac_sha256",
            "updated_unix_ns",
            "head_hmac_sha256",
        }
        if set(head) != required:
            raise AuthorityError("external authority head has missing or extra fields")
        supplied = head.get("head_hmac_sha256")
        unsigned = dict(head)
        unsigned.pop("head_hmac_sha256")
        expected = _hmac_hex(key, HEAD_DOMAIN, _canonical(unsigned))
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
            raise AuthorityError("external authority head HMAC mismatch")
        if (
            head.get("format") != FORMAT
            or head.get("journal_path") != str(self.journal_path)
            or head.get("repo_root") != str(self.repo_root)
            or head.get("key_sha256") != _sha256(key)
        ):
            raise AuthorityError("external authority head identity mismatch")
        numeric = ("protected_prefix_bytes", "protected_rows", "updated_unix_ns")
        if any(type(head.get(name)) is not int or head[name] < 0 for name in numeric):
            raise AuthorityError("external authority head contains an invalid counter")
        for name in (
            "protected_prefix_sha256",
            "tail_hmac_sha256",
            "head_hmac_sha256",
            "key_sha256",
        ):
            if not isinstance(head.get(name), str) or SHA256_RE.fullmatch(head[name]) is None:
                raise AuthorityError(f"external authority head has invalid {name}")
        try:
            uuid.UUID(hex=head["journal_id"])
        except (ValueError, TypeError, AttributeError) as exc:
            raise AuthorityError("external authority head has invalid journal_id") from exc
        if head["protected_rows"] == 0 and (
            head["tail_hmac_sha256"] != head["protected_prefix_sha256"]
        ):
            raise AuthorityError("zero-row authority head has an invalid tail anchor")
        return head

    def _write_head_atomic(self, head: dict[str, Any]) -> None:
        payload = _canonical(head) + b"\n"
        temporary = self.state_dir / (
            f".{self.HEAD_NAME}.tmp-{os.getpid()}-{secrets.token_hex(6)}"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(temporary, flags, 0o600)
        try:
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.replace(temporary, self.head_path)
            _fsync_directory(self.state_dir)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_journal_bytes(self, *, allow_missing: bool) -> bytes:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.journal_path, flags)
        except FileNotFoundError:
            if allow_missing:
                return b""
            raise AuthorityError("protected journal is missing")
        except OSError as exc:
            raise AuthorityError(f"protected journal cannot be read: {exc}") from exc
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise AuthorityError("protected journal is not a regular file")
            chunks = []
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _verify_journal(self, head: dict[str, Any], key: bytes) -> Verification:
        data = self._read_journal_bytes(allow_missing=True)
        prefix_size = int(head["protected_prefix_bytes"])
        if len(data) < prefix_size:
            raise AuthorityError("protected journal prefix was truncated")
        prefix = data[:prefix_size]
        if _sha256(prefix) != head["protected_prefix_sha256"]:
            raise AuthorityError("protected journal prefix digest mismatch")
        suffix = data[prefix_size:]
        if suffix and not suffix.endswith(b"\n"):
            raise AuthorityError("protected journal has a partial trailing row")

        previous = head["protected_prefix_sha256"]
        total = 0
        anchored_tail = previous
        for line_number, framed in enumerate(suffix.splitlines(keepends=True), 1):
            if not framed.endswith(b"\n") or framed == b"\n":
                raise AuthorityError(
                    f"protected journal suffix line {line_number} is incomplete or blank"
                )
            raw = framed[:-1]
            row = _strict_object(raw, f"protected journal suffix line {line_number}")
            if raw != _canonical(row):
                raise AuthorityError(
                    f"protected journal suffix line {line_number} is not canonical JSON"
                )
            authority = row.get(ROW_AUTHORITY_KEY)
            if not isinstance(authority, dict) or set(authority) != {
                "format", "journal_id", "sequence", "previous_hmac_sha256", "hmac_sha256"
            }:
                raise AuthorityError(
                    f"protected journal suffix line {line_number} has invalid authority fields"
                )
            total += 1
            if (
                authority.get("format") != FORMAT
                or authority.get("journal_id") != head["journal_id"]
                or type(authority.get("sequence")) is not int
                or authority["sequence"] != total
                or authority.get("previous_hmac_sha256") != previous
                or not isinstance(authority.get("hmac_sha256"), str)
                or SHA256_RE.fullmatch(authority["hmac_sha256"]) is None
            ):
                raise AuthorityError(
                    f"protected journal suffix line {line_number} breaks authority sequence"
                )
            unsigned = dict(row)
            unsigned_authority = dict(authority)
            supplied = unsigned_authority.pop("hmac_sha256")
            unsigned[ROW_AUTHORITY_KEY] = unsigned_authority
            expected = _hmac_hex(key, DOMAIN, _canonical(unsigned))
            if not hmac.compare_digest(supplied, expected):
                raise AuthorityError(
                    f"protected journal suffix line {line_number} HMAC mismatch"
                )
            previous = supplied
            if total == head["protected_rows"]:
                anchored_tail = supplied

        if total < head["protected_rows"]:
            raise AuthorityError("HMAC-protected journal suffix was truncated")
        if head["protected_rows"] == 0:
            anchored_tail = head["protected_prefix_sha256"]
        if not hmac.compare_digest(anchored_tail, head["tail_hmac_sha256"]):
            raise AuthorityError("journal history disagrees with external head anchor")
        return Verification(
            journal_id=head["journal_id"],
            protected_prefix_bytes=prefix_size,
            protected_prefix_sha256=head["protected_prefix_sha256"],
            protected_rows=total,
            tail_hmac_sha256=previous,
            pending_rows=total - int(head["protected_rows"]),
        )

    def _sign_rows(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        journal_id: str,
        first_sequence: int,
        previous_hmac: str,
        key: bytes,
    ) -> tuple[list[dict[str, Any]], bytes]:
        signed: list[dict[str, Any]] = []
        framed: list[bytes] = []
        previous = previous_hmac
        for offset, original in enumerate(rows):
            if not isinstance(original, dict):
                raise AuthorityError("every appended journal row must be an object")
            if ROW_AUTHORITY_KEY in original:
                raise AuthorityError(f"caller may not supply reserved {ROW_AUTHORITY_KEY!r}")
            # Canonical round-trip freezes nested caller values and rejects
            # NaN, non-string keys, and non-JSON objects before any disk write.
            frozen = _strict_object(_canonical(original), "journal row")
            unsigned_authority = {
                "format": FORMAT,
                "journal_id": journal_id,
                "sequence": first_sequence + offset,
                "previous_hmac_sha256": previous,
            }
            unsigned = {**frozen, ROW_AUTHORITY_KEY: unsigned_authority}
            digest = _hmac_hex(key, DOMAIN, _canonical(unsigned))
            row = {
                **frozen,
                ROW_AUTHORITY_KEY: {
                    **unsigned_authority,
                    "hmac_sha256": digest,
                },
            }
            signed.append(row)
            framed.append(_canonical(row) + b"\n")
            previous = digest
        return signed, b"".join(framed)

    def _append_journal(self, payload: bytes, *, expected_size: int) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.journal_path, flags, 0o644)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
                raise AuthorityError("protected journal changed before authority append")
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_directory(self.journal_path.parent)

    def _advanced_head(
        self,
        head: dict[str, Any],
        *,
        protected_rows: int,
        tail_hmac: str,
        key: bytes,
    ) -> dict[str, Any]:
        unsigned = dict(head)
        unsigned.pop("head_hmac_sha256", None)
        unsigned.update(
            {
                "protected_rows": protected_rows,
                "tail_hmac_sha256": tail_hmac,
                "updated_unix_ns": time.time_ns(),
            }
        )
        return self._sign_head(unsigned, key)
