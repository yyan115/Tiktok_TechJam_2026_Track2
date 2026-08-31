from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sandbox


class SandboxStructureTests(unittest.TestCase):
    def test_broad_usr_bind_rejects_overlapping_private_paths(self):
        with self.assertRaisesRegex(
            sandbox.SandboxError, "overlaps broadly mounted host runtime /usr"
        ):
            sandbox._reject_sensitive_broad_bind_overlap((
                (Path("/usr/src/private-track2-repo"), "repository root"),
            ))
        sandbox._reject_sensitive_broad_bind_overlap((
            (Path("/home/admin/private-track2-state"), "private state"),
        ))

    def test_network_namespace_is_mandatory(self):
        command = sandbox._base_command()
        self.assertIn("--unshare-net", command)
        self.assertIn("--size", command)
        self.assertIn(str(sandbox.TMPFS_BYTES), command)

    def test_host_python_site_directories_are_hidden(self):
        command = sandbox._base_command()
        hidden = {
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--tmpfs"
        }
        for path in sandbox.system_site_directories():
            self.assertIn(str(path), hidden)
            self.assertTrue(any(
                command[index:index + 2] == ["--remount-ro", str(path)]
                for index in range(len(command) - 1)
            ))
        self.assertTrue(any(
            command[index:index + 2] == ["--remount-ro", "/dev"]
            for index in range(len(command) - 1)
        ))

    def test_numpy_mount_is_exact_allowlist_not_site_packages(self):
        artifacts = sandbox.exact_numpy_artifacts()
        self.assertGreater(len(artifacts), 100)
        self.assertTrue(any(relative == "numpy/__init__.py" for _, relative, _ in artifacts))
        self.assertTrue(
            all(
                relative.startswith(("numpy/", "numpy.libs/"))
                and "__pycache__" not in relative
                and not relative.endswith(".pyc")
                and source.is_file()
                for source, relative, _ in artifacts
            )
        )

    def test_official_isolated_interpreter_can_attest_rpm_numpy(self):
        module_path = Path(sandbox.__file__).resolve()
        code = (
            "import importlib.util;"
            f"p={str(module_path)!r};"
            "s=importlib.util.spec_from_file_location('track2_exact_probe',p);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "a=m.runtime_attestation();m.validate_runtime_attestation(a);"
            "print(a['runtime_manifest_sha256'])"
        )
        result = subprocess.run(
            ["/usr/bin/python3", "-I", "-c", code],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout.strip(), r"^[0-9a-f]{64}$")

    def test_snapshot_rejects_extra_symlink_and_hash_drift(self):
        with tempfile.TemporaryDirectory(prefix="track2-snapshot-test-", dir="/tmp") as tmp:
            snapshot = Path(tmp)
            payload = b"sanitized"
            (snapshot / "data.csv").write_bytes(payload)
            manifest = {"data.csv": hashlib.sha256(payload).hexdigest()}
            resolved, names, digest = sandbox.validate_sanitized_snapshot(
                snapshot, manifest
            )
            self.assertEqual(resolved, snapshot.resolve())
            self.assertEqual(names, ("data.csv",))
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

            (snapshot / "extra.csv").write_bytes(b"extra")
            with self.assertRaises(sandbox.SandboxError):
                sandbox.validate_sanitized_snapshot(snapshot, manifest)
            (snapshot / "extra.csv").unlink()

            (snapshot / "data.csv").write_bytes(b"changed")
            with self.assertRaises(sandbox.SandboxError):
                sandbox.validate_sanitized_snapshot(snapshot, manifest)
            (snapshot / "data.csv").unlink()
            (snapshot / "data.csv").symlink_to("target.csv")
            with self.assertRaises(sandbox.SandboxError):
                sandbox.validate_sanitized_snapshot(snapshot, manifest)


class OutputPacketTests(unittest.TestCase):
    @staticmethod
    def write_packet(output: Path) -> None:
        (output / "valid.f64").write_bytes(struct.pack("<2d", 0.1, 0.2))
        (output / "test.f64").write_bytes(struct.pack("<2d", 0.3, 0.4))
        (output / "metadata.json").write_text(
            json.dumps({
                "valid_rows": 2,
                "test_rows": 2,
            })
        )

    def test_reads_exact_raw_f64_packet(self):
        with tempfile.TemporaryDirectory(prefix="track2-output-test-", dir="/tmp") as tmp:
            output = Path(tmp)
            self.write_packet(output)
            valid, test, metadata = sandbox.read_output_packet(output, 2, 2)
            self.assertEqual(valid.tolist(), [0.1, 0.2])
            self.assertEqual(test.tolist(), [0.3, 0.4])
            self.assertEqual(metadata, {"valid_rows": 2, "test_rows": 2})

    def test_rejects_symlink_fifo_hardlink_and_extra_output(self):
        mutations = ("symlink", "fifo", "hardlink", "extra")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix="track2-output-redteam-", dir="/tmp"
            ) as tmp:
                root = Path(tmp)
                output = root / "output"
                output.mkdir()
                self.write_packet(output)
                target = output / "valid.f64"
                if mutation == "extra":
                    (output / "unexpected").write_bytes(b"x")
                else:
                    target.unlink()
                    outside = root / "outside.f64"
                    outside.write_bytes(struct.pack("<2d", 9.0, 9.0))
                    if mutation == "symlink":
                        target.symlink_to(outside)
                    elif mutation == "fifo":
                        os.mkfifo(target)
                    else:
                        os.link(outside, target)
                with self.assertRaises(sandbox.SandboxError):
                    sandbox.read_output_packet(output, 2, 2)

    def test_rejects_wrong_size_nonfinite_and_duplicate_metadata(self):
        mutations = ("size", "nonfinite", "duplicate")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix="track2-output-content-", dir="/tmp"
            ) as tmp:
                output = Path(tmp)
                self.write_packet(output)
                if mutation == "size":
                    (output / "valid.f64").write_bytes(b"short")
                elif mutation == "nonfinite":
                    (output / "valid.f64").write_bytes(
                        struct.pack("<2d", float("nan"), 0.2)
                    )
                else:
                    (output / "metadata.json").write_text(
                        '{"valid_rows":2,"valid_rows":2,"test_rows":2}'
                    )
                with self.assertRaises(sandbox.SandboxError):
                    sandbox.read_output_packet(output, 2, 2)


class CandidateWorkerProtocolTests(unittest.TestCase):
    def test_worker_emits_only_fixed_size_raw_outputs(self):
        with tempfile.TemporaryDirectory(prefix="track2-worker-test-", dir="/tmp") as tmp:
            root = Path(tmp)
            kit = root / "kit"
            data_dir = root / "data"
            output = root / "output"
            for directory in (kit, data_dir, output):
                directory.mkdir()
            for name in sandbox.OUTPUT_FILES:
                (output / name).touch()
            (kit / "data.py").write_text(
                "def load(_):\n"
                " return {'train':[(20220420,'u','v','a','t',1.0,1)],"
                "'valid':[(20220422,'u','v','a','t',0.0,0),(20220422,'u','w','a','t',0.0,0)],"
                "'test':[(20220429,'u','v','a','t',1.0,0),(20220429,'u','w','a','t',1.0,0)]}\n"
            )
            candidate = root / "candidate.py"
            candidate.write_text(
                "def run(splits):\n"
                " return {'valid':[0.1,0.2], 'test':[0.3,0.4]}\n"
            )
            worker = Path(sandbox.__file__).with_name("candidate_worker.py")
            result = subprocess.run(
                [
                    "/usr/bin/python3",
                    str(worker),
                    "--candidate",
                    str(candidate),
                    "--kit",
                    str(kit),
                    "--data",
                    str(data_dir),
                    "--output-dir",
                    str(output),
                    "--cpu-seconds",
                    "30",
                    "--max-output-bytes",
                    str(64 * 1024),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(set(os.listdir(output)), set(sandbox.OUTPUT_FILES))
            self.assertEqual((output / "valid.f64").stat().st_size, 16)
            self.assertEqual((output / "test.f64").stat().st_size, 16)
            valid, test, _ = sandbox.read_output_packet(output, 2, 2)
            self.assertEqual(valid.tolist(), [0.1, 0.2])
            self.assertEqual(test.tolist(), [0.3, 0.4])

@unittest.skipUnless(os.environ.get("TRACK2_TEST_BWRAP") == "1", "set TRACK2_TEST_BWRAP=1")
class SandboxExecutionTests(unittest.TestCase):
    def test_synthetic_candidate_cannot_see_host_repo_or_raw_marker(self):
        with tempfile.TemporaryDirectory(prefix="track2-sandbox-test-", dir="/tmp") as tmp:
            root = Path(tmp)
            kit = root / "kuairand-starter-kit"
            sanitized = kit / "KuaiRand-Pure" / "data_sanitized"
            raw = kit / "KuaiRand-Pure" / "data"
            sanitized_snapshot = root / "trusted-sanitized-snapshot"
            organizer_snapshot = root / "trusted-organizer-snapshot"
            harness = root / "Project" / "harness"
            results = root / "Project" / "results"
            for directory in (
                sanitized, raw, sanitized_snapshot, organizer_snapshot, harness, results
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (raw / "RAW_TEST_LABEL_MARKER").write_text("must-not-be-visible")
            (results / "JOURNAL.jsonl").write_text("must-not-be-visible")
            sanitized_payload = b"synthetic sanitized snapshot"
            (sanitized_snapshot / "synthetic.csv").write_bytes(sanitized_payload)
            worker_source = Path(sandbox.__file__).with_name("candidate_worker.py").read_bytes()
            (harness / "candidate_worker.py").write_bytes(worker_source)
            (kit / "data.py").write_text(
                "def load(_):\n"
                " return {'train':[(20220420,'u','v','a','t',1.0,1)],"
                "'valid':[(20220422,'u','v','a','t',0.0,0),(20220422,'u','w','a','t',0.0,0)],"
                "'test':[(20220429,'u','v','a','t',1.0,0),(20220429,'u','w','a','t',1.0,0)]}\n"
            )
            for name in sandbox.SAFE_KIT_FILES:
                path = kit / name
                if not path.exists():
                    path.write_text("{}" if name.endswith(".json") else "# synthetic\n")
                (organizer_snapshot / name).write_bytes(path.read_bytes())
            candidate = b"""
import os
HYPOTHESIS = 'synthetic boundary probe'
def run(splits):
    forbidden = [
        '/repo/Project/results/JOURNAL.jsonl',
        '/repo/kuairand-starter-kit/KuaiRand-Pure/data/RAW_TEST_LABEL_MARKER',
    ]
    assert all(not os.path.exists(path) for path in forbidden)
    assert not os.path.exists('/home/admin/.local/lib/python3.14/site-packages/openai')
    with open('/proc/net/route') as handle:
        assert len(handle.read().strip().splitlines()) <= 1
    write_probes = ['/dev/track2-probe', '/repo/track2-probe']
    write_probes += [
        '/usr/lib/python3.14/site-packages/track2-probe',
        '/usr/local/lib/python3.14/site-packages/track2-probe',
    ]
    for path in write_probes:
        try:
            with open(path, 'w') as handle:
                handle.write('forbidden')
        except OSError:
            pass
        else:
            raise AssertionError('unexpected writable mount: ' + path)
    return {'valid': [0.1, 0.2], 'test': [0.3, 0.4]}
"""
            runtime_manifest_sha256 = sandbox.runtime_attestation()[
                "runtime_manifest_sha256"
            ]
            result = sandbox.run_candidate(
                root=root,
                candidate_name="candidate.py",
                candidate_bytes=candidate,
                candidate_sha256=hashlib.sha256(candidate).hexdigest(),
                organizer_snapshot=organizer_snapshot,
                organizer_sha256={
                    name: hashlib.sha256((organizer_snapshot / name).read_bytes()).hexdigest()
                    for name in sandbox.SAFE_KIT_FILES
                },
                worker_bytes=worker_source,
                worker_sha256=hashlib.sha256(worker_source).hexdigest(),
                sanitized_snapshot=sanitized_snapshot,
                sanitized_sha256={
                    "synthetic.csv": hashlib.sha256(sanitized_payload).hexdigest()
                },
                timeout_seconds=30,
                expected_valid_rows=2,
                expected_test_rows=2,
                expected_runtime_manifest_sha256=runtime_manifest_sha256,
            )
            self.assertEqual(result["valid"].tolist(), [0.1, 0.2])
            self.assertTrue(result["sandbox"]["network_namespace"])
            self.assertFalse(result["sandbox"]["raw_dataset_mounted"])

            marker = "CANDIDATE_CONTROLLED_DIAGNOSTIC_MUST_NOT_ESCAPE"
            failing = f"""
HYPOTHESIS = 'synthetic diagnostic-channel probe'
def run(splits):
    print({marker!r})
    raise RuntimeError({marker!r})
""".encode()
            with self.assertRaises(sandbox.SandboxError) as caught:
                sandbox.run_candidate(
                    root=root,
                    candidate_name="candidate.py",
                    candidate_bytes=failing,
                    candidate_sha256=hashlib.sha256(failing).hexdigest(),
                    organizer_snapshot=organizer_snapshot,
                    organizer_sha256={
                        name: hashlib.sha256(
                            (organizer_snapshot / name).read_bytes()
                        ).hexdigest()
                        for name in sandbox.SAFE_KIT_FILES
                    },
                    worker_bytes=worker_source,
                    worker_sha256=hashlib.sha256(worker_source).hexdigest(),
                    sanitized_snapshot=sanitized_snapshot,
                    sanitized_sha256={
                        "synthetic.csv": hashlib.sha256(sanitized_payload).hexdigest()
                    },
                    timeout_seconds=30,
                    expected_valid_rows=2,
                    expected_test_rows=2,
                    expected_runtime_manifest_sha256=runtime_manifest_sha256,
                )
            self.assertEqual(
                str(caught.exception), "candidate sandbox execution failed"
            )
            self.assertNotIn(marker, str(caught.exception))

            clean_exit_without_packet = b"raise SystemExit(0)\n"
            with self.assertRaises(sandbox.SandboxError) as clean_exit:
                sandbox.run_candidate(
                    root=root,
                    candidate_name="candidate.py",
                    candidate_bytes=clean_exit_without_packet,
                    candidate_sha256=hashlib.sha256(
                        clean_exit_without_packet
                    ).hexdigest(),
                    organizer_snapshot=organizer_snapshot,
                    organizer_sha256={
                        name: hashlib.sha256(
                            (organizer_snapshot / name).read_bytes()
                        ).hexdigest()
                        for name in sandbox.SAFE_KIT_FILES
                    },
                    worker_bytes=worker_source,
                    worker_sha256=hashlib.sha256(worker_source).hexdigest(),
                    sanitized_snapshot=sanitized_snapshot,
                    sanitized_sha256={
                        "synthetic.csv": hashlib.sha256(sanitized_payload).hexdigest()
                    },
                    timeout_seconds=30,
                    expected_valid_rows=2,
                    expected_test_rows=2,
                    expected_runtime_manifest_sha256=runtime_manifest_sha256,
                )
            self.assertEqual(
                str(clean_exit.exception), "candidate sandbox execution failed"
            )


if __name__ == "__main__":
    unittest.main()
