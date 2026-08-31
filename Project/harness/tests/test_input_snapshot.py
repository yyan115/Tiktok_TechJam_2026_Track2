from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import input_snapshot


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            state.update(block)
    return state.hexdigest()


class SyntheticInputs:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="track2-inputs-", dir="/tmp")
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        self.state = self.base / "state"
        self.state.mkdir(mode=0o700)
        kit = self.root / "kuairand-starter-kit"
        sanitized = kit / "KuaiRand-Pure" / "data_sanitized"
        raw = kit / "KuaiRand-Pure" / "data"
        sanitized.mkdir(parents=True)
        raw.mkdir(parents=True)
        (kit / "data.py").write_text("# exact organizer loader\n")
        (raw / "raw.csv").write_text("date,long_view\n20220429,1\n")
        header = ",".join(input_snapshot.LOG_HEADER) + "\n"
        train_row = {
            name: "0" for name in input_snapshot.LOG_HEADER
        }
        train_row.update({
            "user_id": "u1", "video_id": "v1", "date": "20220420",
            "duration_ms": "1000", "long_view": "1", "is_click": "1",
            "play_time_ms": "900", "tab": "home,feed",
        })
        valid_row = dict(train_row)
        valid_row.update({
            "video_id": "v2", "date": "20220422", "duration_ms": "1100",
            "play_time_ms": "1000",
        })
        test_row = dict(train_row)
        test_row.update({
            "video_id": "v3", "date": "20220429", "duration_ms": "1200",
            "play_time_ms": "1100",
        })
        def csv_row(row):
            # csv.writer-compatible quoting for the synthetic comma in tab.
            return ",".join(
                f'"{row[name]}"' if "," in row[name] else row[name]
                for name in input_snapshot.LOG_HEADER
            ) + "\n"
        (sanitized / "log_standard_4_08_to_4_21_pure.csv").write_text(
            header + csv_row(train_row)
        )
        (sanitized / "log_standard_4_22_to_5_08_pure.csv").write_text(
            header + csv_row(valid_row) + csv_row(test_row)
        )
        (sanitized / "log_random_4_22_to_5_08_pure.csv").write_text(
            header + csv_row(test_row)
        )
        (sanitized / "user_features_pure.csv").write_text(
            ",".join(input_snapshot.USER_HEADER) + "\n"
            + ",".join(
                "u1" if name == "user_id" else
                "full_active" if name == "user_active_degree" else "0"
                for name in input_snapshot.USER_HEADER
            ) + "\n"
        )
        (sanitized / "video_features_basic_pure.csv").write_text(
            ",".join(input_snapshot.VIDEO_BASIC_HEADER) + "\n"
            + ",".join(
                "v1" if name == "video_id" else
                "a1" if name == "author_id" else "0"
                for name in input_snapshot.VIDEO_BASIC_HEADER
            ) + "\n"
        )
        (sanitized / "video_features_statistic_pure.csv").write_text(
            "video_id,show_cnt,long_time_play_cnt\nv1,100,80\n"
        )
        self.manifest = {
            "files": {
                "kuairand-starter-kit/data.py": digest(kit / "data.py"),
            },
            "dataset_files": {
                "kuairand-starter-kit/KuaiRand-Pure/data/raw.csv": digest(raw / "raw.csv"),
            },
            "dataset_files_sanitized": {
                path.relative_to(self.root).as_posix(): digest(path)
                for path in sorted(sanitized.iterdir())
            },
        }

    def close(self):
        self.temp.cleanup()


class InputSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticInputs()

    def tearDown(self):
        self.fixture.close()

    def create(self, state: Path | None = None):
        return input_snapshot.InputSnapshot.create(
            root=self.fixture.root,
            state_dir=state or self.fixture.state,
            manifest=self.fixture.manifest,
        )

    def test_candidate_copy_hides_all_post_train_feedback_only(self):
        snapshot = self.create()
        source = snapshot.sanitized_dir / "log_standard_4_22_to_5_08_pure.csv"
        candidate = snapshot.candidate_dir / source.name
        with source.open(newline="") as handle:
            trusted_rows = list(csv.DictReader(handle))
        with candidate.open(newline="") as handle:
            candidate_rows = list(csv.DictReader(handle))
        self.assertEqual([row["long_view"] for row in trusted_rows], ["1", "1"])
        self.assertEqual([row["long_view"] for row in candidate_rows], ["0", "0"])
        self.assertEqual([row["is_click"] for row in candidate_rows], ["0", "0"])
        self.assertEqual([row["play_time_ms"] for row in candidate_rows], ["0", "0"])
        self.assertEqual(candidate_rows[0]["tab"], "home,feed")

        train = snapshot.candidate_dir / "log_standard_4_08_to_4_21_pure.csv"
        with train.open(newline="") as handle:
            train_row = next(csv.DictReader(handle))
        self.assertEqual(train_row["long_view"], "1")
        self.assertEqual(train_row["is_click"], "1")
        self.assertEqual(train_row["play_time_ms"], "900")

    def test_exact_records_and_deterministic_manifest(self):
        first = self.create()
        second_state = self.fixture.base / "state-two"
        second_state.mkdir(mode=0o700)
        second = self.create(second_state)
        self.assertEqual(first.canonical_sha256, second.canonical_sha256)
        self.assertEqual(first.candidate_sha256(), second.candidate_sha256())
        records = first.verify()["files"]
        self.assertEqual(
            set(records),
            set(self.fixture.manifest["files"])
            | set(self.fixture.manifest["dataset_files"])
            | set(self.fixture.manifest["dataset_files_sanitized"])
            | {"candidate-data/" + name for name in input_snapshot.CANDIDATE_DATA_FILES},
        )

    def test_future_aggregate_and_unresolved_random_log_are_not_candidate_visible(self):
        snapshot = self.create()
        self.assertEqual(
            set(snapshot.candidate_sha256()),
            set(input_snapshot.CANDIDATE_DATA_FILES),
        )
        for name in input_snapshot.WITHHELD_CANDIDATE_DATA_FILES:
            self.assertTrue((snapshot.sanitized_dir / name).is_file())
            self.assertFalse((snapshot.candidate_dir / name).exists())

    def test_candidate_policy_fails_closed_on_unknown_or_missing_source_file(self):
        for mutation in ("missing", "unknown"):
            with self.subTest(mutation=mutation):
                state = self.fixture.base / f"state-{mutation}"
                state.mkdir(mode=0o700)
                changed = json.loads(json.dumps(self.fixture.manifest))
                group = changed["dataset_files_sanitized"]
                if mutation == "missing":
                    group.pop(
                        input_snapshot.SANITIZED_SOURCE_PREFIX
                        + "video_features_statistic_pure.csv"
                    )
                else:
                    group[
                        input_snapshot.SANITIZED_SOURCE_PREFIX + "unexpected.csv"
                    ] = "0" * 64
                with self.assertRaisesRegex(
                    input_snapshot.SnapshotError, "frozen Pure data policy"
                ):
                    input_snapshot.InputSnapshot.create(
                        root=self.fixture.root,
                        state_dir=state,
                        manifest=changed,
                    )

    def test_allowed_basename_with_changed_schema_is_rejected(self):
        target = (
            self.fixture.root / input_snapshot.SANITIZED_SOURCE_PREFIX
            / "user_features_pure.csv"
        )
        target.write_text("user_id,future_outcome\nu1,1\n")
        self.fixture.manifest["dataset_files_sanitized"][
            input_snapshot.SANITIZED_SOURCE_PREFIX + "user_features_pure.csv"
        ] = digest(target)
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "unapproved schema"):
            self.create()

    def test_static_candidate_rows_must_match_the_approved_width(self):
        target = (
            self.fixture.root / input_snapshot.SANITIZED_SOURCE_PREFIX
            / "user_features_pure.csv"
        )
        with target.open("a") as handle:
            handle.write(",".join(["0"] * (len(input_snapshot.USER_HEADER) + 1)) + "\n")
        self.fixture.manifest["dataset_files_sanitized"][
            input_snapshot.SANITIZED_SOURCE_PREFIX + "user_features_pure.csv"
        ] = digest(target)
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "wrong width"):
            self.create()

    def test_log_derivation_rejects_malformed_csv_quotes(self):
        target = (
            self.fixture.root / input_snapshot.SANITIZED_SOURCE_PREFIX
            / "log_standard_4_22_to_5_08_pure.csv"
        )
        target.write_text(
            ",".join(input_snapshot.LOG_HEADER) + "\n\"unterminated\n"
        )
        self.fixture.manifest["dataset_files_sanitized"][
            input_snapshot.SANITIZED_SOURCE_PREFIX
            + "log_standard_4_22_to_5_08_pure.csv"
        ] = digest(target)
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "malformed"):
            self.create()
        self.assertFalse((self.fixture.state / "input-snapshot").exists())

    def test_source_and_serialized_manifest_caps_fail_before_install(self):
        self.assertEqual(
            input_snapshot.MAX_SOURCE_RECORDS
            + len(input_snapshot.CANDIDATE_DATA_FILES),
            input_snapshot.MAX_RECORDS,
        )
        with mock.patch.object(input_snapshot, "MAX_SOURCE_RECORDS", 7):
            with self.assertRaisesRegex(
                input_snapshot.SnapshotError, "invalid number of files"
            ):
                self.create()
        self.assertFalse((self.fixture.state / "input-snapshot").exists())

        with mock.patch.object(input_snapshot, "MAX_MANIFEST_BYTES", 128):
            with self.assertRaisesRegex(
                input_snapshot.SnapshotError, "manifest exceeds its byte limit"
            ):
                self.create()
        self.assertFalse((self.fixture.state / "input-snapshot").exists())

    def test_existing_snapshot_cannot_be_reused_for_a_new_manifest(self):
        snapshot = self.create()
        changed = json.loads(json.dumps(self.fixture.manifest))
        changed["_revision"] = "different organizer manifest bytes"
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "different"):
            input_snapshot.InputSnapshot.create(
                root=self.fixture.root,
                state_dir=self.fixture.state,
                manifest=changed,
            )
        self.assertEqual(snapshot.verify()["format"], input_snapshot.FORMAT)

    def test_legacy_or_self_declared_candidate_policy_cannot_be_reused(self):
        snapshot = self.create()
        candidate = (
            snapshot.candidate_dir / "log_standard_4_22_to_5_08_pure.csv"
        )
        source = snapshot.sanitized_dir / candidate.name
        candidate.chmod(0o600)
        candidate.write_bytes(source.read_bytes())
        manifest = json.loads(snapshot.manifest_path.read_text())
        manifest["files"][f"candidate-data/{candidate.name}"] = {
            "sha256": digest(candidate), "size": candidate.stat().st_size
        }
        snapshot.manifest_path.chmod(0o600)
        snapshot.manifest_path.write_bytes(
            input_snapshot._canonical(manifest) + b"\n"
        )
        with self.assertRaisesRegex(
            input_snapshot.SnapshotError, "derivative differs"
        ):
            input_snapshot.InputSnapshot.create(
                root=self.fixture.root,
                state_dir=self.fixture.state,
                manifest=self.fixture.manifest,
            )

        legacy_state = self.fixture.base / "state-legacy"
        legacy_state.mkdir(mode=0o700)
        legacy = self.create(legacy_state)
        old_manifest = json.loads(legacy.manifest_path.read_text())
        old_manifest["format"] = "track2.input-snapshot.v1"
        old_manifest.pop("candidate_policy_sha256")
        old_manifest.pop("source_files")
        legacy.manifest_path.chmod(0o600)
        legacy.manifest_path.write_bytes(
            input_snapshot._canonical(old_manifest) + b"\n"
        )
        with self.assertRaises(input_snapshot.SnapshotError):
            input_snapshot.InputSnapshot.create(
                root=self.fixture.root,
                state_dir=legacy_state,
                manifest=self.fixture.manifest,
            )

    def test_open_existing_detects_hash_drift_extra_symlink_and_truncation(self):
        snapshot = self.create()
        target = snapshot.candidate_dir / "video_features_basic_pure.csv"
        target.chmod(0o600)
        target.write_bytes(b"changed")
        with self.assertRaises(input_snapshot.SnapshotError):
            input_snapshot.InputSnapshot.open_existing(state_dir=self.fixture.state)

        # Rebuild independently for each mutation; authoritative targets are
        # intentionally immutable rather than repairable in place.
        state = self.fixture.base / "state-extra"
        state.mkdir(mode=0o700)
        snapshot = self.create(state)
        (snapshot.path / "extra").write_bytes(b"x")
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "unexpected"):
            snapshot.verify()

        state = self.fixture.base / "state-special"
        state.mkdir(mode=0o700)
        snapshot = self.create(state)
        os.mkfifo(snapshot.path / "unexpected-fifo")
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "special"):
            snapshot.verify()

        state = self.fixture.base / "state-empty-directory"
        state.mkdir(mode=0o700)
        snapshot = self.create(state)
        (snapshot.path / "unexpected-empty").mkdir()
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "unexpected paths"):
            snapshot.verify()

        state = self.fixture.base / "state-link"
        state.mkdir(mode=0o700)
        snapshot = self.create(state)
        target = snapshot.candidate_dir / "video_features_basic_pure.csv"
        target.unlink()
        target.symlink_to("../kuairand-starter-kit/data.py")
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "unsafe"):
            snapshot.verify()

        state = self.fixture.base / "state-truncated"
        state.mkdir(mode=0o700)
        snapshot = self.create(state)
        snapshot.manifest_path.chmod(0o600)
        snapshot.manifest_path.write_bytes(b'{"format":')
        with self.assertRaises(input_snapshot.SnapshotError):
            snapshot.verify()

    def test_creation_rejects_manifest_escape_symlink_and_hash_drift(self):
        bad = json.loads(json.dumps(self.fixture.manifest))
        bad["files"]["../escape"] = "0" * 64
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "unsafe"):
            input_snapshot.InputSnapshot.create(
                root=self.fixture.root, state_dir=self.fixture.state, manifest=bad
            )

        data_file = self.fixture.root / "kuairand-starter-kit" / "data.py"
        data_file.unlink()
        data_file.symlink_to("/etc/passwd")
        with self.assertRaises(input_snapshot.SnapshotError):
            self.create()

        data_file.unlink()
        data_file.write_text("# changed bytes\n")
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "hash mismatch"):
            self.create()

    def test_open_existing_rejects_snapshot_directory_symlink(self):
        snapshot = self.create()
        alternate_state = self.fixture.base / "state-symlink-root"
        alternate_state.mkdir(mode=0o700)
        (alternate_state / "input-snapshot").symlink_to(snapshot.path)
        with self.assertRaisesRegex(input_snapshot.SnapshotError, "unsafe"):
            input_snapshot.InputSnapshot.open_existing(state_dir=alternate_state)


if __name__ == "__main__":
    unittest.main()
