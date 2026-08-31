from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Project" / "harness"))

import controller_service
import researcher_shell


class ResearcherShellHardeningTests(unittest.TestCase):
    def test_device_mount_is_immediately_remounted_read_only(self):
        command = researcher_shell._base_command()
        device_index = command.index("--dev")
        self.assertEqual(
            command[device_index:device_index + 4],
            ["--dev", "/dev", "--remount-ro", "/dev"],
        )

    def test_broad_runtime_binds_reject_sensitive_usr_and_etc_paths(self):
        for path, label, source in (
            (Path("/usr/src/private-repo"), "repository root", "/usr"),
            (Path("/etc/track2-private"), "controller authority", "/etc"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    researcher_shell.ShellError,
                    rf"{label} overlaps broadly mounted host runtime {source}",
                ):
                    researcher_shell._reject_sensitive_broad_bind_overlap(
                        ((path, label),)
                    )

    def test_broad_runtime_binds_allow_disjoint_home_path(self):
        researcher_shell._reject_sensitive_broad_bind_overlap((
            (Path("/home/admin/private-track2-state"), "private state"),
        ))

    def test_workspace_read_rejects_same_size_concurrent_edit(self):
        with tempfile.TemporaryDirectory(
            prefix="track2-researcher-read-test-", dir="/tmp"
        ) as temporary:
            workspace = Path(temporary).resolve()
            target = workspace / "manifest.json"
            original = b'{"value":"AAAA"}\n'
            replacement = b'{"value":"BBBB"}\n'
            self.assertEqual(len(original), len(replacement))
            target.write_bytes(original)
            target.chmod(0o600)

            real_read = os.read
            changed = False

            def racing_read(fd: int, size: int) -> bytes:
                nonlocal changed
                chunk = real_read(fd, size)
                if chunk and not changed:
                    changed = True
                    target.write_bytes(replacement)
                    target.chmod(0o600)
                return chunk

            with mock.patch.object(researcher_shell.os, "read", side_effect=racing_read):
                with self.assertRaisesRegex(
                    researcher_shell.ShellError, "changed while being read"
                ):
                    researcher_shell._read_workspace_regular(
                        workspace,
                        target.name,
                        maximum_bytes=1024,
                        label="workspace manifest",
                    )

    def test_workspace_root_fifo_is_opened_nonblocking_and_refused(self):
        with tempfile.TemporaryDirectory(
            prefix="track2-researcher-fifo-test-", dir="/tmp"
        ) as temporary:
            workspace = Path(temporary).resolve()
            target = workspace / "manifest.json"
            os.mkfifo(target, 0o600)
            real_open = os.open

            def checked_open(path, flags, *args, **kwargs):
                if path == target.name and kwargs.get("dir_fd") is not None:
                    self.assertTrue(flags & os.O_NONBLOCK)
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(
                researcher_shell.os, "open", side_effect=checked_open
            ):
                with self.assertRaisesRegex(researcher_shell.ShellError, "unsafe"):
                    researcher_shell._read_workspace_regular(
                        workspace,
                        target.name,
                        maximum_bytes=1024,
                        label="workspace manifest",
                    )

    def test_unicode_workspace_binding_matches_controller(self):
        with tempfile.TemporaryDirectory(
            prefix="track2-binding-test-", dir="/tmp"
        ) as temporary:
            workspace = Path(temporary) / "研究-工作区"
            workspace.mkdir(mode=0o700)
            manifest_payload = '{"label":"实验"}\n'.encode("utf-8")
            self.assertEqual(
                researcher_shell._workspace_binding_value(
                    workspace, manifest_payload
                ),
                controller_service._workspace_binding_value(
                    workspace, manifest_payload
                ),
            )


@unittest.skipUnless(
    os.environ.get("TRACK2_TEST_RESEARCHER_BWRAP") == "1",
    "set TRACK2_TEST_RESEARCHER_BWRAP=1",
)
class ResearcherShellExecutionTests(unittest.TestCase):
    def test_real_mount_namespace_has_only_intended_workspace_capabilities(self):
        with tempfile.TemporaryDirectory(
            prefix="track2-researcher-bwrap-test-", dir="/tmp"
        ) as temporary:
            base = Path(temporary).resolve()
            runtime = base / "runtime"
            runtime.mkdir(mode=0o700)
            socket_path = runtime / "controller.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(socket_path))
                socket_path.chmod(0o600)
                workspace = base / "workspace"
                workspace.mkdir(mode=0o700)
                for name in researcher_shell.WORKSPACE_FILE_RULES:
                    (workspace / name).mkdir(mode=0o700)
                portfolio = (
                    ROOT / "Project" / "research" / "templates"
                    / "portfolio.template.json"
                ).read_bytes()
                portfolio_path = workspace / researcher_shell.WORKSPACE_PORTFOLIO
                portfolio_path.write_bytes(portfolio)
                portfolio_path.chmod(0o600)
                manifest_path = workspace / researcher_shell.WORKSPACE_MANIFEST
                manifest_path.write_text(json.dumps({
                    "format": researcher_shell.WORKSPACE_FORMAT,
                    "workspace_id": "c" * 32,
                    "run_id": "a" * 32,
                    "repository_head": "b" * 40,
                    "portfolio_sha256": hashlib.sha256(portfolio).hexdigest(),
                    "created_ns": 1,
                }) + "\n")
                manifest_path.chmod(0o600)
                probe = """
import os, socket, stat
assert os.getcwd() == '/workspace'
assert stat.S_ISSOCK(os.stat('/run/track2/controller.sock').st_mode)
for forbidden in (
    '/workspace/.git', '/workspace/Project/results',
    '/workspace/Project/harness',
    '/workspace/kuairand-starter-kit/KuaiRand-Pure',
):
    assert not os.path.exists(forbidden), forbidden
try:
    fd = os.open('/dev/track2-write-probe', os.O_WRONLY | os.O_CREAT, 0o600)
except OSError:
    pass
else:
    os.close(fd)
    raise AssertionError('/dev accepted a new file')
with open('/workspace/Project/solutions/probe.py', 'w') as handle:
    handle.write('# intended writable mount\\n')
assert os.path.isfile('/workspace/Project/research/bank/catalog.template.json')
print('researcher-sandbox-ok')
"""
                command = researcher_shell.build_command(
                    root=ROOT,
                    socket_path=socket_path,
                    agent_executable=Path("/usr/bin/python3"),
                    agent_args=["-c", probe],
                    workspace_root=workspace,
                )
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    env={},
                )
                probe_written = (
                    workspace / "solutions" / "probe.py"
                ).read_text()
            finally:
                listener.close()
        self.assertEqual(
            completed.returncode, 0, completed.stdout + completed.stderr
        )
        self.assertEqual(completed.stdout.strip(), "researcher-sandbox-ok")
        self.assertEqual(probe_written, "# intended writable mount\n")


if __name__ == "__main__":
    unittest.main()
