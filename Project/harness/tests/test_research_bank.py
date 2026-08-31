from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research_bank


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class ResearchBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="track2-research-bank-", dir="/tmp")
        self.root = Path(self.temp.name) / "repo"
        self.notes = self.root / "Project" / "research" / "bank" / "notes"
        self.notes.mkdir(parents=True)
        self.catalog_path = self.root / research_bank.CATALOG_PATH
        self.note_rel = "Project/research/bank/notes/ranking.md"
        self.note_path = self.root / self.note_rel
        self.note_bytes = "heading\nα evidence line\nsecond evidence line\n".encode()
        self.note_path.write_bytes(self.note_bytes)
        self.claim = {
            "claim_id": "within-user.ranking",
            "note_path": self.note_rel,
            "note_sha256": sha256(self.note_bytes),
            "line_start": 2,
            "line_end": 3,
            "topics": ["ranking", "user.history"],
        }
        self.catalog = {
            "schema_version": 1,
            "benchmark": "KuaiRand-Pure",
            "claims": [self.claim],
        }
        self._write_catalog()
        self._freeze_head()

    def tearDown(self):
        self.temp.cleanup()

    def _write_catalog(self, raw: bytes | None = None) -> None:
        payload = canonical(self.catalog) + b"\n" if raw is None else raw
        self.catalog_path.write_bytes(payload)

    def _freeze_head(self) -> None:
        self.head: dict[str, bytes] = {
            research_bank.CATALOG_PATH: self.catalog_path.read_bytes()
        }
        for path in self.notes.iterdir():
            if path.is_file():
                relative = path.relative_to(self.root).as_posix()
                self.head[relative] = path.read_bytes()

    def _head_reader(self, relative: str) -> bytes:
        return self.head[relative]

    def _load(self) -> research_bank.ResearchBank:
        return research_bank.load(self.root, head_reader=self._head_reader)

    def _assert_code(self, code: str, operation) -> research_bank.BankError:
        with self.assertRaises(research_bank.BankError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(str(caught.exception).startswith(f"{code}: "))
        return caught.exception

    def _configure_note(self, payload: bytes, *, start: int = 1, end: int = 1) -> None:
        self.note_bytes = payload
        self.note_path.write_bytes(payload)
        self.claim["note_sha256"] = sha256(payload)
        self.claim["line_start"] = start
        self.claim["line_end"] = end
        self._write_catalog()
        self._freeze_head()

    def test_load_descriptor_and_resolution_are_exact_and_canonical(self):
        bank = self._load()
        expected_descriptor = {
            "schema_version": 1,
            "catalog": {
                "path": research_bank.CATALOG_PATH,
                "sha256": sha256(self.catalog_path.read_bytes()),
            },
            "notes": [{"path": self.note_rel, "sha256": sha256(self.note_bytes)}],
        }
        self.assertEqual(bank.descriptor, expected_descriptor)
        self.assertEqual(bank.snapshot_sha256, sha256(canonical(expected_descriptor)))
        self.assertEqual(bank.bank_snapshot_sha256, bank.snapshot_sha256)
        self.assertEqual(bank.known_claims, ("within-user.ranking",))
        self.assertEqual(bank.known_topics, ("ranking", "user.history"))
        self.assertEqual(bank.verify(), expected_descriptor)

        resolved = bank.resolve_basis(
            [
                {
                    "claim_id": "within-user.ranking",
                    "relationship": "supports",
                    "target": "mechanism",
                }
            ],
            allowed_relationships={"supports", "contradicts"},
            allowed_targets={"mechanism", "hypothesis"},
        )
        excerpt = "α evidence line\nsecond evidence line"
        self.assertEqual(
            resolved,
            {
                "bank_snapshot_sha256": bank.snapshot_sha256,
                "citations": [
                    {
                        "claim_id": "within-user.ranking",
                        "relationship": "supports",
                        "target": "mechanism",
                        "note_path": self.note_rel,
                        "note_sha256": sha256(self.note_bytes),
                        "line_start": 2,
                        "line_end": 3,
                        "topics": ["ranking", "user.history"],
                        "excerpt": excerpt,
                        "excerpt_sha256": sha256(excerpt.encode("utf-8")),
                    }
                ],
            },
        )

        # All caller-visible containers are copies or immutable views.
        resolved["citations"][0]["topics"].append("poison")
        descriptor = bank.descriptor
        descriptor["notes"].clear()
        again = bank.validate_basis(
            [
                {
                    "claim_id": "within-user.ranking",
                    "relationship": "supports",
                    "target": "mechanism",
                }
            ],
            allowed_relationships={"supports"},
            allowed_targets={"mechanism"},
        )
        self.assertEqual(again["citations"][0]["topics"], ["ranking", "user.history"])
        self.assertEqual(bank.descriptor, expected_descriptor)
        with self.assertRaises(TypeError):
            bank.claims["within-user.ranking"]["line_start"] = 99

    def test_descriptor_sorts_notes_and_snapshot_is_reload_stable(self):
        other_rel = "Project/research/bank/notes/aaa.md"
        other_bytes = b"other evidence\n"
        (self.root / other_rel).write_bytes(other_bytes)
        other = {
            "claim_id": "another.claim",
            "note_path": other_rel,
            "note_sha256": sha256(other_bytes),
            "line_start": 1,
            "line_end": 1,
            "topics": ["ranking"],
        }
        self.catalog["claims"] = [self.claim, other]
        self._write_catalog()
        self._freeze_head()
        first = self._load()
        second = self._load()
        self.assertEqual(
            [entry["path"] for entry in first.descriptor["notes"]],
            [other_rel, self.note_rel],
        )
        self.assertEqual(first.snapshot_sha256, second.snapshot_sha256)
        self.assertEqual(
            first.snapshot_sha256,
            sha256(canonical(first.descriptor)),
        )

    def test_note_path_traversal_and_nonflat_spellings_are_rejected(self):
        bad_paths = [
            "Project/research/bank/notes/../escape.md",
            "Project/research/bank/notes/nested/escape.md",
            "Project/research/bank/notes\\escape.md",
            "Project/research/bank/notes//escape.md",
            "/Project/research/bank/notes/escape.md",
            "Project/research/bank/notes/Upper.md",
            "Project/research/bank/notes/.hidden.md",
            "Project/research/bank/notes/escape.txt",
            "Project/research/bank/notes/nul\x00.md",
        ]
        for bad in bad_paths:
            with self.subTest(path=repr(bad)):
                self.claim["note_path"] = bad
                self._write_catalog()
                self._freeze_head()
                self._assert_code("NOTE_PATH", self._load)

    def test_catalog_and_note_symlinks_are_rejected(self):
        catalog_bytes = self.catalog_path.read_bytes()
        catalog_target = self.root / "catalog-target.json"
        catalog_target.write_bytes(catalog_bytes)
        self.catalog_path.unlink()
        self.catalog_path.symlink_to(catalog_target)
        self.head[research_bank.CATALOG_PATH] = catalog_bytes
        self._assert_code("BANK_FILE_UNSAFE", self._load)

        self.catalog_path.unlink()
        self.catalog_path.write_bytes(catalog_bytes)
        target = self.root / "note-target.md"
        target.write_bytes(self.note_bytes)
        self.note_path.unlink()
        self.note_path.symlink_to(target)
        self._freeze_head()
        self._assert_code("BANK_FILE_UNSAFE", self._load)

        self.note_path.unlink()
        self.note_path.write_bytes(self.note_bytes)
        real_notes = self.notes.with_name("notes-real")
        self.notes.rename(real_notes)
        self.notes.symlink_to(real_notes, target_is_directory=True)
        self._freeze_head()
        self._assert_code("BANK_FILE_UNSAFE", self._load)

    def test_catalog_note_hash_drift_is_rejected(self):
        self.claim["note_sha256"] = "0" * 64
        self._write_catalog()
        self._freeze_head()
        error = self._assert_code("NOTE_HASH_MISMATCH", self._load)
        self.assertEqual(
            str(error),
            f"NOTE_HASH_MISMATCH: note hash does not match catalog: {self.note_rel}",
        )

    def test_live_and_committed_mismatches_fail_for_catalog_and_note(self):
        self.catalog_path.write_bytes(self.catalog_path.read_bytes() + b" ")
        self._assert_code("HEAD_MISMATCH", self._load)

        self._write_catalog()
        self._freeze_head()
        self.note_path.write_bytes(self.note_bytes + b"changed\n")
        self._assert_code("HEAD_MISMATCH", self._load)

    def test_verify_detects_a_valid_new_committed_snapshot(self):
        bank = self._load()
        changed = b"new evidence\n"
        self.note_path.write_bytes(changed)
        self.claim["note_sha256"] = sha256(changed)
        self.claim["line_start"] = 1
        self.claim["line_end"] = 1
        self._write_catalog()
        self._freeze_head()
        self._assert_code("BANK_SNAPSHOT_DRIFT", bank.verify)

    def test_duplicate_claim_ids_and_duplicate_ranges_are_rejected(self):
        duplicate_id = copy.deepcopy(self.claim)
        duplicate_id["line_start"] = 1
        duplicate_id["line_end"] = 1
        self.catalog["claims"] = [self.claim, duplicate_id]
        self._write_catalog()
        self._freeze_head()
        self._assert_code("DUPLICATE_CLAIM_ID", self._load)

        duplicate_range = copy.deepcopy(self.claim)
        duplicate_range["claim_id"] = "different.claim"
        self.catalog["claims"] = [self.claim, duplicate_range]
        self._write_catalog()
        self._freeze_head()
        self._assert_code("DUPLICATE_RANGE", self._load)

    def test_crlf_and_invalid_utf8_notes_are_rejected(self):
        self._configure_note(b"one\r\ntwo\r\n", start=1, end=1)
        self._assert_code("NOTE_NEWLINE", self._load)

        self._configure_note(b"valid\n\xffbad\n", start=1, end=1)
        self._assert_code("NOTE_UTF8", self._load)

    def test_excerpt_range_blank_size_and_line_count_bounds(self):
        self._configure_note(b"one\ntwo\n", start=2, end=3)
        self._assert_code("EXCERPT_RANGE", self._load)

        self._configure_note(b"one\n  \nthree\n", start=2, end=2)
        self._assert_code("EXCERPT_BLANK", self._load)

        self._configure_note(b"x" * 4_001 + b"\n", start=1, end=1)
        self._assert_code("EXCERPT_SIZE", self._load)

        thirteen = b"\n".join([b"x"] * 13) + b"\n"
        self._configure_note(thirteen, start=1, end=13)
        self._assert_code("LINE_RANGE", self._load)

        self._configure_note(b"x" * 4_000 + b"\n", start=1, end=1)
        bank = self._load()
        self.assertEqual(len(bank.claims["within-user.ranking"]["excerpt"].encode()), 4_000)

    def test_excerpt_final_lf_is_never_in_citation(self):
        for payload in (b"one\ntwo", b"one\ntwo\n"):
            with self.subTest(final_lf=payload.endswith(b"\n")):
                self._configure_note(payload, start=1, end=2)
                bank = self._load()
                citation = bank.resolve_basis(
                    [
                        {
                            "claim_id": "within-user.ranking",
                            "relationship": "supports",
                            "target": "hypothesis",
                        }
                    ],
                    allowed_relationships=["supports"],
                    allowed_targets=["hypothesis"],
                )["citations"][0]
                self.assertEqual(citation["excerpt"], "one\ntwo")
                self.assertEqual(citation["excerpt_sha256"], sha256(b"one\ntwo"))

    def test_bool_integer_fields_and_wrong_shapes_are_rejected(self):
        self.catalog["schema_version"] = True
        self._write_catalog()
        self._freeze_head()
        self._assert_code("CATALOG_VERSION", self._load)

        self.catalog["schema_version"] = 1
        self.claim["line_start"] = True
        self._write_catalog()
        self._freeze_head()
        self._assert_code("LINE_RANGE", self._load)

        self.claim["line_start"] = 2
        self.claim["extra"] = "forbidden"
        self._write_catalog()
        self._freeze_head()
        self._assert_code("CLAIM_SCHEMA", self._load)

        del self.claim["extra"]
        self.catalog["extra"] = "forbidden"
        self._write_catalog()
        self._freeze_head()
        self._assert_code("CATALOG_SCHEMA", self._load)

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        raw = (
            b'{"schema_version":1,"schema_version":1,'
            b'"benchmark":"KuaiRand-Pure","claims":[]}'
        )
        self._write_catalog(raw)
        self._freeze_head()
        self._assert_code("CATALOG_DUPLICATE_KEY", self._load)

        raw = b'{"schema_version":1,"benchmark":"KuaiRand-Pure","claims":NaN}'
        self._write_catalog(raw)
        self._freeze_head()
        self._assert_code("CATALOG_NONFINITE", self._load)

    def test_topic_id_hash_and_note_count_schemas_are_strict(self):
        self.claim["claim_id"] = "UPPER"
        self._write_catalog()
        self._freeze_head()
        self._assert_code("CLAIM_ID", self._load)

        self.claim["claim_id"] = "valid.claim"
        self.claim["topics"] = ["ranking", "ranking"]
        self._write_catalog()
        self._freeze_head()
        self._assert_code("TOPICS", self._load)

        self.claim["topics"] = ["ranking"]
        self.claim["note_sha256"] = "A" * 64
        self._write_catalog()
        self._freeze_head()
        self._assert_code("NOTE_SHA256", self._load)

        claims = []
        for index in range(65):
            claims.append(
                {
                    "claim_id": f"claim.{index:02d}",
                    "note_path": f"Project/research/bank/notes/note.{index:02d}.md",
                    "note_sha256": "0" * 64,
                    "line_start": 1,
                    "line_end": 1,
                    "topics": ["ranking"],
                }
            )
        self.catalog["claims"] = claims
        self._write_catalog()
        self._freeze_head()
        self._assert_code("NOTES_COUNT", self._load)

    def test_unknown_duplicate_and_malformed_basis_entries_fail_closed(self):
        bank = self._load()
        common = {
            "allowed_relationships": {"supports", "contradicts"},
            "allowed_targets": {"mechanism", "hypothesis"},
        }
        self._assert_code(
            "BASIS_UNKNOWN_CLAIM",
            lambda: bank.resolve_basis(
                [
                    {
                        "claim_id": "unknown.claim",
                        "relationship": "supports",
                        "target": "mechanism",
                    }
                ],
                **common,
            ),
        )
        duplicate = [
            {
                "claim_id": "within-user.ranking",
                "relationship": "supports",
                "target": "mechanism",
            },
            {
                "claim_id": "within-user.ranking",
                "relationship": "contradicts",
                "target": "hypothesis",
            },
        ]
        self._assert_code(
            "BASIS_DUPLICATE", lambda: bank.resolve_basis(duplicate, **common)
        )
        extra = [dict(duplicate[0], note_path=self.note_rel)]
        self._assert_code(
            "BASIS_SCHEMA", lambda: bank.resolve_basis(extra, **common)
        )
        self._assert_code(
            "BASIS_SCHEMA", lambda: bank.resolve_basis([], **common)
        )
        self._assert_code(
            "BASIS_SCHEMA", lambda: bank.resolve_basis([duplicate[0]] * 7, **common)
        )
        disallowed = [dict(duplicate[0], relationship="proves")]
        self._assert_code(
            "BASIS_RELATIONSHIP", lambda: bank.resolve_basis(disallowed, **common)
        )


if __name__ == "__main__":
    unittest.main()
