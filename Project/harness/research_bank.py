"""Frozen, content-addressed research evidence for the trusted controller.

The research bank is deliberately much smaller than a general document
loader.  One fixed catalog names a bounded set of flat Markdown notes and
line ranges.  Both the working-tree bytes and the bytes committed at ``HEAD``
must agree before any claim can be used.  Callers may then cite only a known
claim identifier; they never choose a path, hash, or line range.

``head_reader`` is injectable so unit tests and other trusted callers can
provide committed bytes without invoking Git.  Its contract is simply
``head_reader(repository_relative_posix_path) -> bytes``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping


SCHEMA_VERSION = 1
BENCHMARK = "KuaiRand-Pure"
CATALOG_PATH = "Project/research/bank/catalog.json"
NOTES_PREFIX = "Project/research/bank/notes/"

MAX_CATALOG_BYTES = 256 * 1024
MAX_CLAIMS = 256
MAX_NOTES = 64
MAX_NOTE_BYTES = 128 * 1024
MAX_TOTAL_NOTE_BYTES = 4 * 1024 * 1024
MAX_EXCERPT_LINES = 12
MAX_EXCERPT_BYTES = 4_000
MAX_TOPICS = 8
MAX_BASIS_ENTRIES = 6

CLAIM_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{2,63}")
TOPIC_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,31}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
# The complete basename, including ``.md``, is at most 128 characters.
NOTE_PATH_RE = re.compile(
    r"Project/research/bank/notes/[a-z0-9][a-z0-9._-]{0,124}\.md"
)

CATALOG_KEYS = {"schema_version", "benchmark", "claims"}
CLAIM_KEYS = {
    "claim_id",
    "note_path",
    "note_sha256",
    "line_start",
    "line_end",
    "topics",
}
BASIS_KEYS = {"claim_id", "relationship", "target"}

HeadReader = Callable[[str], bytes]


class BankError(RuntimeError):
    """A stable, fail-closed research-bank validation error.

    ``code`` is suitable for controller records.  ``message`` and ``str(exc)``
    are deterministic and intentionally omit platform-dependent exception
    text.
    """

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise BankError(code, message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    """Encode a finite JSON value in the controller's canonical form."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # defensive: descriptors are internal
        raise BankError(
            "BANK_CANONICAL",
            "research-bank value cannot be encoded as finite canonical JSON",
        ) from exc


def _strict_catalog(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BankError("CATALOG_UTF8", "catalog must be valid UTF-8") from exc

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(
                    "CATALOG_DUPLICATE_KEY",
                    f"catalog contains duplicate JSON key {key!r}",
                )
            result[key] = value
        return result

    def no_constants(value: str) -> None:
        _fail(
            "CATALOG_NONFINITE",
            f"catalog contains non-finite JSON constant {value}",
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=no_constants,
        )
    except BankError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise BankError("CATALOG_JSON", "catalog must be strict JSON") from exc
    if type(value) is not dict:
        _fail("CATALOG_SCHEMA", "catalog must contain one JSON object")
    return value


def _root(path: Path) -> Path:
    try:
        supplied = Path(path)
    except (TypeError, ValueError, OSError) as exc:
        raise BankError(
            "BANK_ROOT", "repository root must be an existing real directory"
        ) from exc
    try:
        metadata = supplied.lstat()
    except OSError as exc:
        raise BankError(
            "BANK_ROOT", "repository root must be an existing real directory"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        _fail("BANK_ROOT", "repository root must be an existing real directory")
    try:
        return supplied.resolve(strict=True)
    except OSError as exc:  # pragma: no cover - lstat already covers normal cases
        raise BankError(
            "BANK_ROOT", "repository root must be an existing real directory"
        ) from exc


def _read_live(root: Path, relative: str, maximum_bytes: int) -> bytes:
    """Securely walk and read one fixed repository-relative regular file."""

    parts = relative.split("/")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    file_flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    try:
        directory_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise BankError(
            "BANK_ROOT", "repository root cannot be securely opened"
        ) from exc
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(
                    part,
                    directory_flags,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError as exc:
                raise BankError(
                    "BANK_FILE_MISSING", f"bank file is missing: {relative}"
                ) from exc
            except OSError as exc:
                raise BankError(
                    "BANK_FILE_UNSAFE",
                    f"bank file has an unsafe parent directory: {relative}",
                ) from exc
            previous_fd = directory_fd
            directory_fd = next_fd
            os.close(previous_fd)

        try:
            descriptor = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        except FileNotFoundError as exc:
            raise BankError(
                "BANK_FILE_MISSING", f"bank file is missing: {relative}"
            ) from exc
        except OSError as exc:
            raise BankError(
                "BANK_FILE_UNSAFE",
                f"bank file cannot be securely opened: {relative}",
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                _fail(
                    "BANK_FILE_UNSAFE",
                    f"bank file must be a regular non-symlink: {relative}",
                )
            if opened.st_size > maximum_bytes:
                _fail(
                    "BANK_FILE_SIZE",
                    f"bank file exceeds {maximum_bytes} bytes: {relative}",
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                try:
                    chunk = os.read(
                        descriptor,
                        min(64 * 1024, maximum_bytes + 1 - total),
                    )
                except OSError as exc:
                    raise BankError(
                        "BANK_FILE_READ", f"bank file cannot be read: {relative}"
                    ) from exc
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > maximum_bytes:
                    _fail(
                        "BANK_FILE_SIZE",
                        f"bank file exceeds {maximum_bytes} bytes: {relative}",
                    )
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)


class _GitHeadReader:
    """Read exact ``HEAD`` blobs using one fixed, isolated Git invocation."""

    def __init__(self, root: Path):
        self.root = root
        self._validate_repository()

    def _validate_repository(self) -> None:
        """Recheck repository layout so later verification also fails closed."""

        root = self.root
        git_directory = root / ".git"
        try:
            metadata = git_directory.lstat()
        except OSError as exc:
            raise BankError(
                "GIT_REPOSITORY",
                "default HEAD reader requires a standalone repository",
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail(
                "GIT_REPOSITORY",
                "default HEAD reader requires a standalone repository",
            )
        alternates = git_directory / "objects" / "info" / "alternates"
        if alternates.exists() or alternates.is_symlink():
            _fail(
                "GIT_ALTERNATES",
                "default HEAD reader forbids alternate Git object stores",
            )

    def __call__(self, relative: str) -> bytes:
        self._validate_repository()
        environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        try:
            result = subprocess.run(
                [
                    "/usr/bin/git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "core.attributesFile=/dev/null",
                    "show",
                    f"HEAD:{relative}",
                ],
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
                shell=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BankError(
                "HEAD_READ", f"cannot read committed HEAD bytes: {relative}"
            ) from exc
        if result.returncode != 0:
            _fail("HEAD_READ", f"cannot read committed HEAD bytes: {relative}")
        return result.stdout


def _read_exact_pair(
    root: Path,
    relative: str,
    maximum_bytes: int,
    head_reader: HeadReader,
) -> bytes:
    live = _read_live(root, relative, maximum_bytes)
    try:
        committed = head_reader(relative)
    except BankError:
        raise
    except Exception as exc:
        raise BankError(
            "HEAD_READ", f"cannot read committed HEAD bytes: {relative}"
        ) from exc
    if type(committed) is not bytes:
        _fail("HEAD_RESULT", "HEAD reader must return immutable bytes")
    if len(committed) > maximum_bytes:
        _fail(
            "BANK_FILE_SIZE",
            f"committed bank file exceeds {maximum_bytes} bytes: {relative}",
        )
    if live != committed:
        _fail(
            "HEAD_MISMATCH",
            f"live bank bytes differ from committed HEAD: {relative}",
        )
    return live


@dataclass(frozen=True)
class _Claim:
    claim_id: str
    note_path: str
    note_sha256: str
    line_start: int
    line_end: int
    topics: tuple[str, ...]
    excerpt: str
    excerpt_sha256: str

    def public(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "note_path": self.note_path,
            "note_sha256": self.note_sha256,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "topics": list(self.topics),
            "excerpt": self.excerpt,
            "excerpt_sha256": self.excerpt_sha256,
        }


def _validate_claims(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    if set(catalog) != CATALOG_KEYS:
        _fail(
            "CATALOG_SCHEMA",
            "catalog must have exactly schema_version, benchmark, and claims",
        )
    if type(catalog["schema_version"]) is not int or catalog["schema_version"] != 1:
        _fail("CATALOG_VERSION", "catalog schema_version must be integer 1")
    if catalog["benchmark"] != BENCHMARK or type(catalog["benchmark"]) is not str:
        _fail("CATALOG_BENCHMARK", f"catalog benchmark must be {BENCHMARK}")
    claims = catalog["claims"]
    if type(claims) is not list or not 1 <= len(claims) <= MAX_CLAIMS:
        _fail("CLAIMS_COUNT", "catalog must contain between 1 and 256 claims")

    seen_ids: set[str] = set()
    seen_ranges: set[tuple[str, int, int]] = set()
    for index, claim in enumerate(claims):
        label = f"claim {index + 1}"
        if type(claim) is not dict or set(claim) != CLAIM_KEYS:
            _fail("CLAIM_SCHEMA", f"{label} has missing or extra fields")
        claim_id = claim["claim_id"]
        if type(claim_id) is not str or CLAIM_ID_RE.fullmatch(claim_id) is None:
            _fail("CLAIM_ID", f"{label} has an invalid claim_id")
        if claim_id in seen_ids:
            _fail("DUPLICATE_CLAIM_ID", f"duplicate claim_id: {claim_id}")
        seen_ids.add(claim_id)

        note_path = claim["note_path"]
        if type(note_path) is not str or NOTE_PATH_RE.fullmatch(note_path) is None:
            _fail("NOTE_PATH", f"{label} has an invalid note_path")
        note_sha256 = claim["note_sha256"]
        if (
            type(note_sha256) is not str
            or SHA256_RE.fullmatch(note_sha256) is None
        ):
            _fail("NOTE_SHA256", f"{label} has an invalid note_sha256")

        line_start = claim["line_start"]
        line_end = claim["line_end"]
        if (
            type(line_start) is not int
            or type(line_end) is not int
            or line_start < 1
            or line_end < line_start
            or line_end - line_start + 1 > MAX_EXCERPT_LINES
        ):
            _fail(
                "LINE_RANGE",
                f"{label} must select between 1 and 12 positive ordered lines",
            )
        range_key = (note_path, line_start, line_end)
        if range_key in seen_ranges:
            _fail(
                "DUPLICATE_RANGE",
                f"duplicate note line range: {note_path}:{line_start}-{line_end}",
            )
        seen_ranges.add(range_key)

        topics = claim["topics"]
        if type(topics) is not list or not 1 <= len(topics) <= MAX_TOPICS:
            _fail("TOPICS", f"{label} must have between 1 and 8 topics")
        if any(
            type(topic) is not str or TOPIC_RE.fullmatch(topic) is None
            for topic in topics
        ):
            _fail("TOPICS", f"{label} has an invalid topic")
        if len(set(topics)) != len(topics):
            _fail("TOPICS", f"{label} has duplicate topics")
    return claims


def _note_lines(payload: bytes) -> list[bytes]:
    """Return LF-addressed logical lines without terminator bytes.

    A single final LF terminates the preceding line; it does not create an
    additional empty line.  Internal empty lines remain addressable.  Selected
    lines are later joined with one LF and no terminal LF, giving citations a
    single deterministic textual representation regardless of whether the
    source note has a final newline.
    """

    lines = payload.split(b"\n")
    if payload.endswith(b"\n"):
        lines.pop()
    return lines


class ResearchBank:
    """A verified immutable view of the committed research-bank snapshot."""

    def __init__(
        self,
        *,
        root: Path,
        head_reader: HeadReader,
        catalog_sha256: str,
        note_hashes: Mapping[str, str],
        claims: Mapping[str, _Claim],
    ):
        self.root = root
        self._head_reader = head_reader
        self._catalog_sha256 = catalog_sha256
        self._note_hashes = MappingProxyType(dict(sorted(note_hashes.items())))
        self._claim_records = MappingProxyType(dict(claims))
        public_claims: dict[str, Mapping[str, Any]] = {}
        for claim_id, claim in claims.items():
            value = claim.public()
            value["topics"] = tuple(value["topics"])
            public_claims[claim_id] = MappingProxyType(value)
        self._claims = MappingProxyType(public_claims)
        self._topics = tuple(
            sorted({topic for claim in claims.values() for topic in claim.topics})
        )
        self._known_claims = tuple(sorted(claims))
        self._snapshot_sha256 = _sha256(_canonical(self._descriptor()))

    @classmethod
    def load(
        cls,
        root: Path,
        head_reader: HeadReader | None = None,
    ) -> "ResearchBank":
        repository = _root(root)
        if head_reader is None:
            reader: HeadReader = _GitHeadReader(repository)
        elif callable(head_reader):
            reader = head_reader
        else:
            _fail("HEAD_READER", "head_reader must be callable")

        catalog_bytes = _read_exact_pair(
            repository, CATALOG_PATH, MAX_CATALOG_BYTES, reader
        )
        catalog = _strict_catalog(catalog_bytes)
        raw_claims = _validate_claims(catalog)

        expected_notes: dict[str, str] = {}
        for claim in raw_claims:
            path = claim["note_path"]
            digest = claim["note_sha256"]
            prior = expected_notes.get(path)
            if prior is not None and prior != digest:
                _fail(
                    "NOTE_HASH_CONFLICT",
                    f"catalog assigns conflicting hashes to note: {path}",
                )
            expected_notes[path] = digest
        if len(expected_notes) > MAX_NOTES:
            _fail("NOTES_COUNT", "catalog may reference at most 64 notes")

        note_payloads: dict[str, bytes] = {}
        note_hashes: dict[str, str] = {}
        total_note_bytes = 0
        for path in sorted(expected_notes):
            payload = _read_exact_pair(repository, path, MAX_NOTE_BYTES, reader)
            total_note_bytes += len(payload)
            if total_note_bytes > MAX_TOTAL_NOTE_BYTES:
                _fail(
                    "NOTES_TOTAL_SIZE",
                    "referenced notes exceed 4194304 total bytes",
                )
            if b"\r" in payload:
                _fail("NOTE_NEWLINE", f"note must use LF-only newlines: {path}")
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise BankError("NOTE_UTF8", f"note must be valid UTF-8: {path}") from exc
            actual = _sha256(payload)
            if actual != expected_notes[path]:
                _fail("NOTE_HASH_MISMATCH", f"note hash does not match catalog: {path}")
            note_payloads[path] = payload
            note_hashes[path] = actual

        claim_records: dict[str, _Claim] = {}
        for raw_claim in raw_claims:
            path = raw_claim["note_path"]
            lines = _note_lines(note_payloads[path])
            start = raw_claim["line_start"]
            end = raw_claim["line_end"]
            if end > len(lines):
                _fail(
                    "EXCERPT_RANGE",
                    f"claim range exceeds note lines: {raw_claim['claim_id']}",
                )
            excerpt_bytes = b"\n".join(lines[start - 1 : end])
            excerpt = excerpt_bytes.decode("utf-8")
            if not excerpt.strip():
                _fail(
                    "EXCERPT_BLANK",
                    f"claim excerpt must be nonblank: {raw_claim['claim_id']}",
                )
            if len(excerpt_bytes) > MAX_EXCERPT_BYTES:
                _fail(
                    "EXCERPT_SIZE",
                    f"claim excerpt exceeds 4000 bytes: {raw_claim['claim_id']}",
                )
            record = _Claim(
                claim_id=raw_claim["claim_id"],
                note_path=path,
                note_sha256=raw_claim["note_sha256"],
                line_start=start,
                line_end=end,
                topics=tuple(raw_claim["topics"]),
                excerpt=excerpt,
                excerpt_sha256=_sha256(excerpt_bytes),
            )
            claim_records[record.claim_id] = record

        # Re-read the fixed catalog after all note I/O.  This closes the useful
        # catalog-swap window and ensures the constructed claim set was derived
        # from the same exact live/HEAD bytes throughout loading.
        final_catalog = _read_exact_pair(
            repository, CATALOG_PATH, MAX_CATALOG_BYTES, reader
        )
        if final_catalog != catalog_bytes:
            _fail("BANK_FILE_RACE", "catalog changed while loading research bank")

        return cls(
            root=repository,
            head_reader=reader,
            catalog_sha256=_sha256(catalog_bytes),
            note_hashes=note_hashes,
            claims=claim_records,
        )

    def _descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "catalog": {
                "path": CATALOG_PATH,
                "sha256": self._catalog_sha256,
            },
            "notes": [
                {"path": path, "sha256": digest}
                for path, digest in self._note_hashes.items()
            ],
        }

    @property
    def descriptor(self) -> dict[str, Any]:
        """Return a caller-owned copy of the canonical snapshot descriptor."""

        return self._descriptor()

    @property
    def snapshot_sha256(self) -> str:
        return self._snapshot_sha256

    @property
    def bank_snapshot_sha256(self) -> str:
        """Explicit alias matching the field stored in controller records."""

        return self._snapshot_sha256

    @property
    def claims(self) -> Mapping[str, Mapping[str, Any]]:
        return self._claims

    @property
    def topics(self) -> tuple[str, ...]:
        return self._topics

    @property
    def known_claims(self) -> tuple[str, ...]:
        return self._known_claims

    @property
    def known_topics(self) -> tuple[str, ...]:
        return self._topics

    def verify(self) -> dict[str, Any]:
        """Revalidate live and committed bytes against this frozen snapshot."""

        current = type(self).load(self.root, head_reader=self._head_reader)
        if current.snapshot_sha256 != self.snapshot_sha256:
            _fail(
                "BANK_SNAPSHOT_DRIFT",
                "research bank changed since this snapshot was loaded",
            )
        return current.descriptor

    @staticmethod
    def _allowed(values: Iterable[str], label: str) -> frozenset[str]:
        if isinstance(values, (str, bytes)):
            _fail("BASIS_ALLOWLIST", f"{label} must be a nonempty string collection")
        try:
            items = list(values)
        except (TypeError, ValueError) as exc:
            raise BankError(
                "BASIS_ALLOWLIST", f"{label} must be a nonempty string collection"
            ) from exc
        if not items or any(type(item) is not str or not item for item in items):
            _fail("BASIS_ALLOWLIST", f"{label} must be a nonempty string collection")
        return frozenset(items)

    def resolve_basis(
        self,
        basis: Any,
        *,
        allowed_relationships: Iterable[str],
        allowed_targets: Iterable[str],
    ) -> dict[str, Any]:
        """Resolve caller citations to controller-owned exact excerpts.

        Caller entries contain only ``claim_id``, ``relationship``, and
        ``target``.  Every other returned field comes from the frozen bank.
        """

        relationships = self._allowed(allowed_relationships, "allowed_relationships")
        targets = self._allowed(allowed_targets, "allowed_targets")
        if type(basis) is not list or not 1 <= len(basis) <= MAX_BASIS_ENTRIES:
            _fail("BASIS_SCHEMA", "research_basis must contain between 1 and 6 entries")

        normalized: list[tuple[str, str, str]] = []
        seen_claim_ids: set[str] = set()
        for index, entry in enumerate(basis):
            label = f"research_basis entry {index + 1}"
            if type(entry) is not dict or set(entry) != BASIS_KEYS:
                _fail("BASIS_SCHEMA", f"{label} has missing or extra fields")
            claim_id = entry["claim_id"]
            relationship = entry["relationship"]
            target = entry["target"]
            if type(claim_id) is not str or CLAIM_ID_RE.fullmatch(claim_id) is None:
                _fail("BASIS_CLAIM_ID", f"{label} has an invalid claim_id")
            if claim_id not in self._claim_records:
                _fail("BASIS_UNKNOWN_CLAIM", f"unknown research claim: {claim_id}")
            if type(relationship) is not str or relationship not in relationships:
                _fail("BASIS_RELATIONSHIP", f"{label} has a disallowed relationship")
            if type(target) is not str or target not in targets:
                _fail("BASIS_TARGET", f"{label} has a disallowed target")
            if claim_id in seen_claim_ids:
                _fail("BASIS_DUPLICATE", f"duplicate requested claim_id: {claim_id}")
            seen_claim_ids.add(claim_id)
            normalized.append((claim_id, relationship, target))

        # Never emit cached evidence after live/HEAD drift.
        self.verify()
        citations: list[dict[str, Any]] = []
        for claim_id, relationship, target in normalized:
            claim = self._claim_records[claim_id]
            citations.append(
                {
                    "claim_id": claim_id,
                    "relationship": relationship,
                    "target": target,
                    "note_path": claim.note_path,
                    "note_sha256": claim.note_sha256,
                    "line_start": claim.line_start,
                    "line_end": claim.line_end,
                    "topics": list(claim.topics),
                    "excerpt": claim.excerpt,
                    "excerpt_sha256": claim.excerpt_sha256,
                }
            )
        return {
            "bank_snapshot_sha256": self.snapshot_sha256,
            "citations": citations,
        }

    def validate_basis(
        self,
        basis: Any,
        *,
        allowed_relationships: Iterable[str],
        allowed_targets: Iterable[str],
    ) -> dict[str, Any]:
        return self.resolve_basis(
            basis,
            allowed_relationships=allowed_relationships,
            allowed_targets=allowed_targets,
        )


def load(root: Path, head_reader: HeadReader | None = None) -> ResearchBank:
    """Load the one fixed research bank beneath ``root``."""

    return ResearchBank.load(root, head_reader=head_reader)


def validate_basis(
    bank: ResearchBank,
    basis: Any,
    *,
    allowed_relationships: Iterable[str],
    allowed_targets: Iterable[str],
) -> dict[str, Any]:
    """Functional wrapper for trusted-controller integration."""

    if not isinstance(bank, ResearchBank):
        _fail("BASIS_BANK", "validate_basis requires a ResearchBank")
    return bank.resolve_basis(
        basis,
        allowed_relationships=allowed_relationships,
        allowed_targets=allowed_targets,
    )
