"""Exact immutable organizer/data snapshots held outside the researcher sandbox."""

from __future__ import annotations

import hashlib
import csv
import json
import os
import secrets
import stat
from pathlib import Path, PurePosixPath
from typing import Any


FORMAT = "track2.input-snapshot.v2"
MANIFEST_NAME = "SNAPSHOT.json"
VALID_START = 20220422
FEEDBACK_COLUMNS = {
    "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "long_view", "play_time_ms", "profile_stay_time",
    "comment_stay_time", "is_profile_enter",
}
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_RECORDS = 10_000
CHUNK_BYTES = 1024 * 1024
MAX_CANDIDATE_CSV_BYTES = 512 * 1024 * 1024
MAX_CANDIDATE_CSV_ROWS = 10_000_000

# The source snapshot retains every manifest-pinned organizer input for audit
# and trusted evaluation.  The candidate receives a deliberately narrower
# derivative.  In particular, KuaiRand's official documentation says the
# statistic file averages engagement outcomes over a month, so it is not a
# time-safe feature for this date split.  The random log overlaps both public
# validation and hidden-test dates and its competition use remains unresolved.
# Neither file crosses the candidate boundary unless the policy is formally
# revised before the run is frozen.
SANITIZED_SOURCE_PREFIX = (
    "kuairand-starter-kit/KuaiRand-Pure/data_sanitized/"
)
CANDIDATE_DATA_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
)
MAX_SOURCE_RECORDS = MAX_RECORDS - len(CANDIDATE_DATA_FILES)
WITHHELD_CANDIDATE_DATA_FILES = (
    "log_random_4_22_to_5_08_pure.csv",
    "video_features_statistic_pure.csv",
)
EXPECTED_SANITIZED_SOURCE_FILES = frozenset(
    SANITIZED_SOURCE_PREFIX + name
    for name in (*CANDIDATE_DATA_FILES, *WITHHELD_CANDIDATE_DATA_FILES)
)
LOG_HEADER = (
    "user_id", "video_id", "date", "hourmin", "time_ms", "is_click",
    "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
    "comment_stay_time", "is_profile_enter", "is_rand", "tab",
)
USER_HEADER = (
    "user_id", "user_active_degree", "is_lowactive_period",
    "is_live_streamer", "is_video_author", "follow_user_num",
    "follow_user_num_range", "fans_user_num", "fans_user_num_range",
    "friend_user_num", "friend_user_num_range", "register_days",
    "register_days_range", "onehot_feat0", "onehot_feat1", "onehot_feat2",
    "onehot_feat3", "onehot_feat4", "onehot_feat5", "onehot_feat6",
    "onehot_feat7", "onehot_feat8", "onehot_feat9", "onehot_feat10",
    "onehot_feat11", "onehot_feat12", "onehot_feat13", "onehot_feat14",
    "onehot_feat15", "onehot_feat16", "onehot_feat17",
)
VIDEO_BASIC_HEADER = (
    "video_id", "author_id", "video_type", "upload_dt", "upload_type",
    "visible_status", "video_duration", "server_width", "server_height",
    "music_id", "music_type", "tag",
)
APPROVED_CANDIDATE_HEADERS = {
    "log_standard_4_08_to_4_21_pure.csv": LOG_HEADER,
    "log_standard_4_22_to_5_08_pure.csv": LOG_HEADER,
    "user_features_pure.csv": USER_HEADER,
    "video_features_basic_pure.csv": VIDEO_BASIC_HEADER,
}


class SnapshotError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SnapshotError(f"snapshot value is not finite JSON: {exc}") from exc


def _candidate_policy_sha256() -> str:
    descriptor = {
        "format": FORMAT,
        "candidate_files": list(CANDIDATE_DATA_FILES),
        "withheld_files": list(WITHHELD_CANDIDATE_DATA_FILES),
        "valid_start": VALID_START,
        "feedback_columns": sorted(FEEDBACK_COLUMNS),
        "headers": {
            name: list(header)
            for name, header in sorted(APPROVED_CANDIDATE_HEADERS.items())
        },
        "log_transform": (
            "strict CSV parse; preserve feature values/order; set every listed "
            "feedback column to string zero when date >= valid_start; emit LF"
        ),
        "static_transform": "strict schema/row-width validation; byte-exact copy",
    }
    return _sha256(_canonical(descriptor))


CANDIDATE_POLICY_SHA256 = _candidate_policy_sha256()


def _open_regular(path: Path, *, maximum_bytes: int | None = None) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SnapshotError(f"cannot securely open snapshot input {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SnapshotError(f"snapshot input is not a regular file: {path}")
        if maximum_bytes is not None and info.st_size > maximum_bytes:
            raise SnapshotError(f"snapshot input exceeds its size limit: {path}")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_regular(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    fd = _open_regular(path, maximum_bytes=maximum_bytes)
    try:
        chunks = []
        size = 0
        while True:
            chunk = os.read(fd, CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if maximum_bytes is not None and size > maximum_bytes:
                raise SnapshotError(f"snapshot input exceeds its size limit: {path}")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _hash_regular(path: Path, *, expected_size: int | None = None) -> tuple[str, int]:
    fd = _open_regular(path)
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = os.read(fd, CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(fd)
    if expected_size is not None and size != expected_size:
        raise SnapshotError(f"snapshot file size mismatch: {path}")
    return digest.hexdigest(), size


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise SnapshotError("snapshot write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_regular_exclusive(source: Path, destination: Path) -> tuple[str, int]:
    """Copy through safe descriptors while hashing the exact copied bytes."""

    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_fd = _open_regular(source)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    destination_fd = os.open(destination, flags, 0o400)
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = os.read(source_fd, CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise SnapshotError("snapshot copy made no progress")
                view = view[written:]
        os.fsync(destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    return digest.hexdigest(), size


def _candidate_visible_csv(source_path: Path, destination_path: Path, name: str) -> tuple[str, int]:
    """Remove every validation/test feedback field while preserving features."""

    destination_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_fd = _open_regular(
        source_path, maximum_bytes=MAX_CANDIDATE_CSV_BYTES
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    destination_fd = os.open(destination_path, flags, 0o400)
    digest = hashlib.sha256()
    size = 0

    class HashingWriter:
        def write(self, value: str) -> int:
            nonlocal size
            payload = value.encode("utf-8")
            digest.update(payload)
            size += len(payload)
            view = memoryview(payload)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise SnapshotError("candidate CSV write made no progress")
                view = view[written:]
            return len(value)

    try:
        with os.fdopen(source_fd, "r", encoding="utf-8", errors="strict", newline="") as source:
            source_fd = -1
            reader = csv.DictReader(source, strict=True)
            if tuple(reader.fieldnames or ()) != APPROVED_CANDIDATE_HEADERS.get(name):
                raise SnapshotError(f"candidate CSV has an unapproved schema: {name}")
            writer = csv.DictWriter(
                HashingWriter(), fieldnames=reader.fieldnames, lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row_number, row in enumerate(reader, 1):
                if row_number > MAX_CANDIDATE_CSV_ROWS:
                    raise SnapshotError(f"candidate CSV has too many rows: {name}")
                if None in row:
                    raise SnapshotError(f"candidate log has extra CSV fields: {name}")
                if any(value is None for value in row.values()):
                    raise SnapshotError(f"candidate log has missing CSV fields: {name}")
                try:
                    date = int(row["date"])
                except (TypeError, ValueError) as exc:
                    raise SnapshotError(f"candidate log has invalid date: {name}") from exc
                if date >= VALID_START:
                    for column in FEEDBACK_COLUMNS:
                        if column in row:
                            row[column] = "0"
                writer.writerow(row)
        os.fsync(destination_fd)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SnapshotError(f"candidate CSV is malformed: {name}") from exc
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        os.close(destination_fd)
    return digest.hexdigest(), size


def _expected_candidate_csv(source_path: Path, name: str) -> tuple[str, int]:
    """Re-derive the canonical candidate CSV digest without writing state."""

    source_fd = _open_regular(
        source_path, maximum_bytes=MAX_CANDIDATE_CSV_BYTES
    )
    digest = hashlib.sha256()
    size = 0

    class HashingWriter:
        def write(self, value: str) -> int:
            nonlocal size
            payload = value.encode("utf-8")
            digest.update(payload)
            size += len(payload)
            return len(value)

    try:
        with os.fdopen(
            source_fd, "r", encoding="utf-8", errors="strict", newline=""
        ) as source:
            source_fd = -1
            reader = csv.DictReader(source, strict=True)
            if tuple(reader.fieldnames or ()) != APPROVED_CANDIDATE_HEADERS.get(name):
                raise SnapshotError(f"candidate CSV has an unapproved schema: {name}")
            writer = csv.DictWriter(
                HashingWriter(), fieldnames=reader.fieldnames, lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row_number, row in enumerate(reader, 1):
                if row_number > MAX_CANDIDATE_CSV_ROWS:
                    raise SnapshotError(f"candidate CSV has too many rows: {name}")
                if None in row or any(value is None for value in row.values()):
                    raise SnapshotError(f"candidate log has malformed fields: {name}")
                try:
                    date = int(row["date"])
                except (TypeError, ValueError) as exc:
                    raise SnapshotError(f"candidate log has invalid date: {name}") from exc
                if date >= VALID_START:
                    for column in FEEDBACK_COLUMNS:
                        if column in row:
                            row[column] = "0"
                writer.writerow(row)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SnapshotError(f"candidate CSV is malformed: {name}") from exc
    finally:
        if source_fd >= 0:
            os.close(source_fd)
    return digest.hexdigest(), size


def _validate_static_candidate_schema(source_path: Path, name: str) -> None:
    source_fd = _open_regular(
        source_path, maximum_bytes=MAX_CANDIDATE_CSV_BYTES
    )
    try:
        with os.fdopen(
            source_fd, "r", encoding="utf-8", errors="strict", newline=""
        ) as source:
            source_fd = -1
            reader = csv.reader(source, strict=True)
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                raise SnapshotError(f"candidate CSV is empty: {name}") from exc
            if header != APPROVED_CANDIDATE_HEADERS.get(name):
                raise SnapshotError(f"candidate CSV has an unapproved schema: {name}")
            for row_number, row in enumerate(reader, 1):
                if row_number > MAX_CANDIDATE_CSV_ROWS:
                    raise SnapshotError(f"candidate CSV has too many rows: {name}")
                if len(row) != len(header):
                    raise SnapshotError(
                        f"candidate CSV row has the wrong width: {name}"
                    )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SnapshotError(f"candidate CSV is malformed: {name}") from exc
    finally:
        if source_fd >= 0:
            os.close(source_fd)


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SnapshotError(f"unsafe {label} path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotError(f"unsafe {label} path: {value}")
    if path.as_posix() != value:
        raise SnapshotError(f"non-canonical {label} path: {value}")
    return value


class InputSnapshot:
    """A content-addressed copy of only manifest-listed required inputs."""

    def __init__(self, path: Path):
        # Keep the lexical final component so verify() can reject a symlink;
        # resolving here would erase exactly the evidence we need to inspect.
        self.path = path.absolute()
        self.manifest_path = self.path / MANIFEST_NAME

    @classmethod
    def create(cls, *, root: Path, state_dir: Path, manifest: dict) -> "InputSnapshot":
        root = root.resolve(strict=True)
        state_dir = state_dir.resolve(strict=True)
        target = state_dir / "input-snapshot"
        sanitized_policy_group = manifest.get("dataset_files_sanitized")
        if (
            not isinstance(sanitized_policy_group, dict)
            or set(sanitized_policy_group) != EXPECTED_SANITIZED_SOURCE_FILES
        ):
            raise SnapshotError(
                "sanitized source manifest differs from the frozen Pure data policy"
            )
        selected: dict[str, str] = {}
        for section in ("files", "dataset_files", "dataset_files_sanitized"):
            group = manifest.get(section)
            if not isinstance(group, dict) or not group:
                raise SnapshotError(f"manifest section {section} is missing")
            for rel, digest in group.items():
                if rel == "kuairand-starter-kit.zip":
                    continue
                rel = _safe_relative(rel, "source manifest")
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                ):
                    raise SnapshotError(f"invalid manifest entry in {section}")
                unresolved = root / rel
                try:
                    path = unresolved.resolve(strict=True)
                    metadata = unresolved.lstat()
                except OSError as exc:
                    raise SnapshotError(f"manifest input is missing: {rel}") from exc
                if (
                    not path.is_relative_to(root)
                    or stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                ):
                    raise SnapshotError(f"manifest path escapes repository: {rel}")
                if rel in selected and selected[rel] != digest:
                    raise SnapshotError(f"manifest sections disagree about {rel}")
                selected[rel] = digest
        if not selected or len(selected) > MAX_SOURCE_RECORDS:
            raise SnapshotError("manifest selects an invalid number of files")
        if target.exists():
            existing = cls(target)
            frozen = existing.verify(expected_sources=selected)
            current_manifest_sha256 = _sha256(_canonical(manifest))
            if frozen.get("source_manifest_sha256") != current_manifest_sha256:
                raise SnapshotError(
                    "existing input snapshot belongs to a different source manifest"
                )
            return existing
        temporary = state_dir / f".input-snapshot-{secrets.token_hex(8)}"
        temporary.mkdir(mode=0o700)
        try:
            records = {}
            for rel, expected in sorted(selected.items()):
                source = root / rel
                destination = temporary / rel
                actual, size = _copy_regular_exclusive(source, destination)
                if actual != expected:
                    raise SnapshotError(f"source hash mismatch while snapshotting: {rel}")
                records[rel] = {"sha256": actual, "size": size}
            sanitized_group = manifest["dataset_files_sanitized"]
            candidate_names: set[str] = set()
            for rel, expected in sorted(sanitized_group.items()):
                rel = _safe_relative(rel, "sanitized manifest")
                name = Path(rel).name
                if name not in CANDIDATE_DATA_FILES:
                    continue
                if name in candidate_names:
                    raise SnapshotError(f"duplicate candidate-data basename: {name}")
                candidate_names.add(name)
                source_copy = temporary / rel
                source_digest, _ = _hash_regular(source_copy)
                if source_digest != expected:
                    raise SnapshotError(
                        f"sanitized source changed while deriving candidate data: {rel}"
                    )
                candidate_path = temporary / "candidate-data" / name
                if name.startswith("log_") and name.endswith(".csv"):
                    candidate_digest, candidate_size = _candidate_visible_csv(
                        source_copy, candidate_path, name
                    )
                else:
                    _validate_static_candidate_schema(source_copy, name)
                    candidate_digest, candidate_size = _copy_regular_exclusive(
                        source_copy, candidate_path
                    )
                candidate_rel = f"candidate-data/{name}"
                records[candidate_rel] = {
                    "sha256": candidate_digest,
                    "size": candidate_size,
                }
            snapshot_manifest = {
                "format": FORMAT,
                "source_manifest_sha256": _sha256(_canonical(manifest)),
                "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
                "source_files": sorted(selected),
                "files": records,
            }
            manifest_payload = _canonical(snapshot_manifest) + b"\n"
            if len(manifest_payload) > MAX_MANIFEST_BYTES:
                raise SnapshotError("snapshot manifest exceeds its byte limit")
            _write_exclusive(
                temporary / MANIFEST_NAME,
                manifest_payload,
            )
            os.replace(temporary, target)
            fd = os.open(state_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            # A unique incomplete directory is never treated as authoritative;
            # retain it for forensics rather than performing broad deletion.
            raise
        created = cls(target)
        created.verify(expected_sources=selected)
        return created

    @classmethod
    def open_existing(cls, *, state_dir: Path) -> "InputSnapshot":
        snapshot = cls(state_dir.resolve(strict=True) / "input-snapshot")
        snapshot.verify()
        return snapshot

    def _manifest(self) -> dict:
        raw = _read_regular(self.manifest_path, maximum_bytes=MAX_MANIFEST_BYTES)
        if not raw.endswith(b"\n"):
            raise SnapshotError("snapshot manifest is not a complete line")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SnapshotError(f"snapshot manifest is invalid JSON: {exc}") from exc
        if not isinstance(value, dict) or set(value) != {
            "format", "source_manifest_sha256", "candidate_policy_sha256",
            "source_files", "files"
        }:
            raise SnapshotError("snapshot manifest has the wrong shape")
        if raw != _canonical(value) + b"\n" or value.get("format") != FORMAT:
            raise SnapshotError("snapshot manifest is not canonical or has wrong format")
        source_digest = value.get("source_manifest_sha256")
        if (
            not isinstance(source_digest, str)
            or len(source_digest) != 64
            or any(char not in "0123456789abcdef" for char in source_digest)
        ):
            raise SnapshotError("snapshot source manifest digest is malformed")
        if value.get("candidate_policy_sha256") != CANDIDATE_POLICY_SHA256:
            raise SnapshotError("snapshot candidate policy is obsolete or altered")
        source_files = value.get("source_files")
        if (
            not isinstance(source_files, list)
            or not source_files
            or len(source_files) > MAX_SOURCE_RECORDS
            or not all(isinstance(item, str) for item in source_files)
            or source_files != sorted(source_files)
            or len(source_files) != len(set(source_files))
            or any(_safe_relative(item, "snapshot source") != item for item in source_files)
        ):
            raise SnapshotError("snapshot source file registry is malformed")
        return value

    def verify(self, *, expected_sources: dict[str, str] | None = None) -> dict:
        try:
            root_metadata = self.path.lstat()
        except FileNotFoundError as exc:
            raise SnapshotError("input snapshot is missing or unsafe") from exc
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
            raise SnapshotError("input snapshot is missing or unsafe")
        manifest = self._manifest()
        records = manifest.get("files")
        if not isinstance(records, dict) or not 1 <= len(records) <= MAX_RECORDS:
            raise SnapshotError("input snapshot has no file records")
        expected_paths = {MANIFEST_NAME}
        for rel, record in records.items():
            rel = _safe_relative(rel, "snapshot record")
            if (
                not isinstance(record, dict)
                or set(record) != {"sha256", "size"}
                or type(record.get("size")) is not int
                or record["size"] < 0
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in record["sha256"])
            ):
                raise SnapshotError("snapshot file record is malformed")
            path = self.path
            parts = PurePosixPath(rel).parts
            for offset, part in enumerate(parts):
                path = path / part
                try:
                    metadata = path.lstat()
                except OSError as exc:
                    raise SnapshotError(f"snapshot path is missing: {rel}") from exc
                final = offset == len(parts) - 1
                if stat.S_ISLNK(metadata.st_mode) or (
                    final and not stat.S_ISREG(metadata.st_mode)
                ) or (not final and not stat.S_ISDIR(metadata.st_mode)):
                    raise SnapshotError(f"snapshot path is unsafe: {rel}")
            actual, _ = _hash_regular(path, expected_size=record["size"])
            if actual != record["sha256"]:
                raise SnapshotError(f"snapshot file hash/size mismatch: {rel}")
            expected_paths.add(rel)
        candidate_records = {
            rel for rel in records if rel.startswith("candidate-data/")
        }
        expected_candidate_records = {
            f"candidate-data/{name}" for name in CANDIDATE_DATA_FILES
        }
        if candidate_records != expected_candidate_records:
            raise SnapshotError("snapshot candidate file registry differs from policy")
        source_records = set(records) - candidate_records
        if source_records != set(manifest["source_files"]):
            raise SnapshotError("snapshot source file registry differs from records")
        if expected_sources is not None:
            if source_records != set(expected_sources):
                raise SnapshotError("snapshot source files differ from current manifest")
            for rel, digest in expected_sources.items():
                if records[rel]["sha256"] != digest:
                    raise SnapshotError(
                        f"snapshot source digest differs from current manifest: {rel}"
                    )

        # The manifest hashes bind identity, but policy truth is independently
        # recomputed from the frozen source copies.  A self-consistent legacy
        # manifest therefore cannot authorize visible validation/test labels.
        for name in CANDIDATE_DATA_FILES:
            source_rel = SANITIZED_SOURCE_PREFIX + name
            candidate_rel = f"candidate-data/{name}"
            if source_rel not in records:
                raise SnapshotError(f"candidate source is absent from snapshot: {name}")
            source_path = self.path / source_rel
            candidate_record = records[candidate_rel]
            if name.startswith("log_"):
                expected_digest, expected_size = _expected_candidate_csv(
                    source_path, name
                )
            else:
                _validate_static_candidate_schema(source_path, name)
                expected_digest = records[source_rel]["sha256"]
                expected_size = records[source_rel]["size"]
            if candidate_record != {
                "sha256": expected_digest, "size": expected_size
            }:
                raise SnapshotError(
                    f"candidate derivative differs from current policy: {name}"
                )
        expected_directories = {""}
        for rel in expected_paths:
            parent = PurePosixPath(rel).parent
            while parent.as_posix() != ".":
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        actual_paths: set[str] = set()
        actual_directories = {""}
        for path in self.path.rglob("*"):
            rel = path.relative_to(self.path).as_posix()
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                actual_paths.add(rel)
            elif stat.S_ISDIR(metadata.st_mode):
                actual_directories.add(rel)
            else:
                raise SnapshotError(f"snapshot contains an unsafe special path: {rel}")
        if actual_paths != expected_paths or actual_directories != expected_directories:
            raise SnapshotError(
                "snapshot contains missing or unexpected paths: "
                f"files={sorted(actual_paths ^ expected_paths)} "
                f"directories={sorted(actual_directories ^ expected_directories)}"
            )
        return manifest

    def file(self, relative_to_repo: str) -> Path:
        relative_to_repo = _safe_relative(relative_to_repo, "requested snapshot")
        manifest = self._manifest()
        if relative_to_repo not in manifest["files"]:
            raise SnapshotError(f"file is not part of exact input snapshot: {relative_to_repo}")
        return self.path / relative_to_repo

    @property
    def canonical_sha256(self) -> str:
        return _sha256(_read_regular(self.manifest_path, maximum_bytes=MAX_MANIFEST_BYTES))

    @property
    def sanitized_dir(self) -> Path:
        return self.path / "kuairand-starter-kit" / "KuaiRand-Pure" / "data_sanitized"

    @property
    def candidate_dir(self) -> Path:
        return self.path / "candidate-data"

    def candidate_sha256(self) -> dict[str, str]:
        files = self.verify()["files"]
        return {
            Path(rel).name: record["sha256"]
            for rel, record in files.items()
            if rel.startswith("candidate-data/")
        }

    def kit_sha256(self) -> dict[str, str]:
        files = self.verify()["files"]
        prefix = "kuairand-starter-kit/"
        return {
            rel[len(prefix):]: record["sha256"]
            for rel, record in files.items()
            if rel.startswith(prefix) and "/" not in rel[len(prefix):]
        }

    @property
    def raw_dir(self) -> Path:
        return self.path / "kuairand-starter-kit" / "KuaiRand-Pure" / "data"

    @property
    def kit_dir(self) -> Path:
        return self.path / "kuairand-starter-kit"
