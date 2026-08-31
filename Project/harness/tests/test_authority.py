from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import authority


class ExternalJournalAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="track2-authority-", dir="/tmp")
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.journal = self.repo / "Project" / "results" / "JOURNAL.jsonl"
        self.journal.parent.mkdir(parents=True)
        self.legacy = b'{"entry_id":"legacy","type":"setup"}\n'
        self.journal.write_bytes(self.legacy)
        self.external = self.base / "external-authority"
        self.auth = authority.JournalAuthority.create(
            journal_path=self.journal,
            state_dir=self.external,
            repo_root=self.repo,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_external_head_protects_prefix_and_hmac_rows(self):
        signed = self.auth.append(
            [
                {"entry_id": "one", "type": "iteration", "primary": 0.6},
                {"entry_id": "two", "type": "run_terminated"},
            ]
        )
        result = self.auth.verify()
        self.assertEqual(result.protected_prefix_bytes, len(self.legacy))
        self.assertEqual(result.protected_prefix_sha256, authority._sha256(self.legacy))
        self.assertEqual(result.protected_rows, 2)
        self.assertEqual(result.tail_hmac_sha256, signed[-1][authority.ROW_AUTHORITY_KEY]["hmac_sha256"])
        self.assertEqual(signed[0][authority.ROW_AUTHORITY_KEY]["sequence"], 1)
        self.assertEqual(signed[1][authority.ROW_AUTHORITY_KEY]["sequence"], 2)

    def test_tail_content_modification_is_detected(self):
        self.auth.append([{"entry_id": "one", "payload": "alpha"}])
        changed = self.journal.read_bytes().replace(b"alpha", b"bravo")
        self.assertNotEqual(changed, self.journal.read_bytes())
        self.journal.write_bytes(changed)
        with self.assertRaisesRegex(authority.AuthorityError, "HMAC mismatch"):
            self.auth.verify()

    def test_tail_json_reformatting_is_detected(self):
        self.auth.append([{"entry_id": "one", "payload": "alpha"}])
        lines = self.journal.read_bytes().splitlines(keepends=True)
        protected = json.loads(lines[-1])
        lines[-1] = json.dumps(protected, sort_keys=True).encode() + b"\n"
        self.journal.write_bytes(b"".join(lines))
        with self.assertRaisesRegex(authority.AuthorityError, "not canonical JSON"):
            self.auth.verify()

    def test_tail_truncation_is_detected(self):
        self.auth.append(
            [{"entry_id": "one", "value": 1}, {"entry_id": "two", "value": 2}]
        )
        lines = self.journal.read_bytes().splitlines(keepends=True)
        self.journal.write_bytes(b"".join(lines[:-1]))
        with self.assertRaisesRegex(authority.AuthorityError, "suffix was truncated"):
            self.auth.verify()

    def test_protected_legacy_prefix_modification_is_detected(self):
        changed = self.journal.read_bytes().replace(b"legacy", b"LEGACY")
        self.assertEqual(len(changed), len(self.journal.read_bytes()))
        self.journal.write_bytes(changed)
        with self.assertRaisesRegex(authority.AuthorityError, "prefix digest mismatch"):
            self.auth.verify()

    def test_only_hmac_valid_crash_suffix_can_be_reconciled(self):
        with mock.patch.object(
            self.auth,
            "_write_head_atomic",
            side_effect=OSError("synthetic crash after journal fsync"),
        ):
            with self.assertRaisesRegex(OSError, "synthetic crash"):
                self.auth.append([{"entry_id": "crash-row", "value": 7}])

        with self.assertRaises(authority.CrashSuffixPending) as caught:
            self.auth.verify()
        self.assertEqual(caught.exception.pending_rows, 1)
        reconciled = self.auth.verify(reconcile_crash_suffix=True)
        self.assertEqual(reconciled.reconciled_rows, 1)
        self.assertEqual(reconciled.protected_rows, 1)
        self.assertEqual(self.auth.verify().protected_rows, 1)

    def test_unsigned_or_partial_crash_suffix_is_never_reconciled(self):
        with self.journal.open("ab") as handle:
            handle.write(b'{"entry_id":"forged"}\n')
            handle.flush()
            os.fsync(handle.fileno())
        with self.assertRaisesRegex(authority.AuthorityError, "invalid authority fields"):
            self.auth.verify(reconcile_crash_suffix=True)

        # Restore only inside the isolated test fixture, then prove a partial
        # line also fails rather than being silently chopped off.
        self.journal.write_bytes(self.legacy)
        with self.journal.open("ab") as handle:
            handle.write(b'{"entry_id":"partial"')
            handle.flush()
            os.fsync(handle.fileno())
        with self.assertRaisesRegex(authority.AuthorityError, "partial trailing row"):
            self.auth.verify(reconcile_crash_suffix=True)

    def test_external_head_tampering_is_detected(self):
        head_path = self.external / authority.JournalAuthority.HEAD_NAME
        value = json.loads(head_path.read_text())
        value["protected_rows"] = 99
        head_path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        )
        os.chmod(head_path, 0o600)
        with self.assertRaisesRegex(authority.AuthorityError, "head HMAC mismatch"):
            self.auth.verify()

    def test_authority_state_inside_repo_is_refused(self):
        with self.assertRaisesRegex(authority.AuthorityError, "outside the repository"):
            authority.JournalAuthority.create(
                journal_path=self.journal,
                state_dir=self.repo / ".authority",
                repo_root=self.repo,
            )

    def test_existing_authority_cannot_be_reset(self):
        with self.assertRaisesRegex(authority.AuthorityError, "already exists"):
            authority.JournalAuthority.create(
                journal_path=self.journal,
                state_dir=self.external,
                repo_root=self.repo,
            )
        reopened = authority.JournalAuthority.open_existing(
            journal_path=self.journal,
            state_dir=self.external,
            repo_root=self.repo,
        )
        self.assertEqual(reopened.verify().protected_rows, 0)


if __name__ == "__main__":
    unittest.main()
