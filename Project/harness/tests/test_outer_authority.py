from __future__ import annotations

import copy
import contextlib
import fcntl
import hashlib
import io
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Project" / "harness"))
sys.path.insert(0, str(ROOT / "Project" / "tools"))

import controller_service
import control
import controller_mcp
import iterate
import init_researcher_workspace
import researcher_shell


os.environ.setdefault(control.WORKSPACE_ID_ENV, "c" * 32)
os.environ.setdefault(control.WORKSPACE_BINDING_ENV, "e" * 64)


def make_researcher_workspace(base: Path) -> Path:
    workspace = base / "workspace"
    workspace.mkdir(mode=0o700)
    for name in researcher_shell.WORKSPACE_FILE_RULES:
        (workspace / name).mkdir(mode=0o700)
    portfolio_payload = (
        ROOT / "Project" / "research" / "templates" / "portfolio.template.json"
    ).read_bytes()
    (workspace / researcher_shell.WORKSPACE_PORTFOLIO).write_bytes(portfolio_payload)
    (workspace / researcher_shell.WORKSPACE_PORTFOLIO).chmod(0o600)
    (workspace / researcher_shell.WORKSPACE_MANIFEST).write_text(json.dumps({
        "format": researcher_shell.WORKSPACE_FORMAT,
        "workspace_id": "c" * 32,
        "run_id": "a" * 32,
        "repository_head": "b" * 40,
        "portfolio_sha256": hashlib.sha256(portfolio_payload).hexdigest(),
        "created_ns": 1,
    }) + "\n")
    (workspace / researcher_shell.WORKSPACE_MANIFEST).chmod(0o600)
    return workspace


def execution_response(request: dict, **changes) -> dict:
    response = {
        "protocol_version": control.PROTOCOL_VERSION,
        "request_id": request["request_id"],
        "command": request["command"],
        "ok": True,
        "returncode": 0,
        "stdout": "official output\n",
        "stderr": "",
        "elapsed_seconds": 0.125,
        "artifact_commit": None,
    }
    if request["command"] == "run":
        response["artifact_commit"] = {
            "git_revision": "1" * 40,
            "solution_sha256": "2" * 64,
            "card_sha256": "3" * 64,
        }
    response.update(changes)
    return response


def error_response(request: dict, **changes) -> dict:
    response = {
        "protocol_version": control.PROTOCOL_VERSION,
        "request_id": request["request_id"],
        "command": request["command"],
        "ok": False,
        "returncode": 125,
        "error": "refused",
        "recovery_required": False,
    }
    response.update(changes)
    return response


class SyntheticGitRepo:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="track2-authority-test-", dir="/tmp")
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        for relative in (
            "Project/harness",
            "Project/solutions",
            "Project/research/attempts",
            "Project/memory",
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        self.iterate = self.root / "Project" / "harness" / "iterate.py"
        self.iterate.write_text(
            "import json, sys\n"
            "if sys.argv[1:] == ['_admission-state']:\n"
            " print(json.dumps({'official_run_started':True,'state':'ACTIVE',"
            "'open_attempt':False,'would_trigger_now':[],"
            "'attempt_review_budget_remaining':3,"
            "'attempt_review_failure_budget_remaining':3}))\n"
            "elif sys.argv[1:] == ['log']:\n"
            " print(json.dumps({'state':'ACTIVE','open_attempt':False,"
            "'would_trigger_now':[],'attempt_review_budget_remaining':3}))\n"
            "else:\n"
            " print(json.dumps({'argv': sys.argv[1:]}))\n"
        )
        self._git("init", "-q")
        self._git("config", "user.name", "Synthetic Test")
        self._git("config", "user.email", "synthetic@example.invalid")
        self._git("add", "--", "Project/harness/iterate.py")
        self._git("commit", "-q", "-m", "synthetic root")
        self.env = controller_service.fixed_environment(
            {"HOME": str(self.base), "LANG": "C.UTF-8"}
        )

    def close(self):
        self.temp.cleanup()

    def _git(self, *args: str, binary: bool = False):
        return subprocess.run(
            ["/usr/bin/git", *args],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=not binary,
        ).stdout

    def artifacts(self, suffix: str = "one") -> tuple[str, str, bytes, bytes]:
        solution = f"Project/solutions/s001_{suffix}.py"
        card = f"Project/research/attempts/i001_{suffix}.json"
        solution_bytes = b"HYPOTHESIS='synthetic'\ndef run(s): return {}\n"
        card_bytes = json.dumps({"synthetic": suffix}, sort_keys=True).encode()
        (self.root / solution).write_bytes(solution_bytes)
        (self.root / card).write_bytes(card_bytes)
        return solution, card, solution_bytes, card_bytes

    def protected(self) -> controller_service.ProtectedState:
        return controller_service.ProtectedState(
            self.root,
            paths=("Project/harness/iterate.py",),
            env=self.env,
        )


class RequestSurfaceTests(unittest.TestCase):
    def test_only_exact_run_and_log_requests_are_accepted(self):
        request_id = "a" * 32
        self.assertEqual(
            controller_service.validate_request(
                {
                    "protocol_version": control.PROTOCOL_VERSION,
                    "request_id": request_id,
                    "workspace_id": "c" * 32,
                    "workspace_binding": "e" * 64,
                    "command": "log",
                }
            )["command"],
            "log",
        )
        valid = {
            "protocol_version": control.PROTOCOL_VERSION,
            "request_id": request_id,
            "workspace_id": "c" * 32,
            "workspace_binding": "e" * 64,
            "command": "run",
            "solution": "Project/solutions/s001_ok.py",
            "card": "Project/research/attempts/i001_ok.json",
        }
        self.assertEqual(controller_service.validate_request(valid), valid)
        for changed in (
            {key: value for key, value in valid.items() if key != "workspace_id"},
            {key: value for key, value in valid.items() if key != "workspace_binding"},
            {**valid, "workspace_id": "0" * 31},
            {**valid, "command": "start-run"},
            {**valid, "command": "final"},
            {**valid, "solution": "../Project/solutions/s001_ok.py"},
            {**valid, "solution": "Project/solutions/nested/s001.py"},
            {**valid, "card": "Project/research/attempts/../../manifest.json"},
            {**valid, "override": True},
        ):
            with self.assertRaises(controller_service.ServiceError):
                controller_service.validate_request(changed)

    def test_fixed_controller_command_has_isolated_python_and_no_shell(self):
        request = {
            "protocol_version": control.PROTOCOL_VERSION,
            "request_id": "b" * 32,
            "workspace_id": "c" * 32,
            "workspace_binding": "e" * 64,
            "command": "run",
            "solution": "Project/solutions/s001_ok.py",
            "card": "Project/research/attempts/i001_ok.json",
        }
        command = controller_service.controller_command(ROOT, request)
        self.assertEqual(command[:2], ["/usr/bin/python3", "-I"])
        self.assertEqual(command[-5:], [
            "run", "--solution", request["solution"], "--card", request["card"]
        ])
        self.assertNotIn("bash", " ".join(command))

    def test_unprivileged_client_has_the_same_narrow_surface(self):
        request = control.build_request(
            "run",
            solution="Project/solutions/s001_ok.py",
            card="Project/research/attempts/i001_ok.json",
            request_id="c" * 32,
        )
        self.assertEqual(request["command"], "run")
        with self.assertRaises(control.ClientError):
            control.build_request("final", request_id="d" * 32)

    def test_authority_refuses_a_different_valid_workspace_id(self):
        authority = object.__new__(controller_service.ControllerAuthority)
        authority.workspace = mock.Mock(
            workspace_id="c" * 32, binding="e" * 64
        )
        authority.protected = mock.Mock()
        request = control.build_request(
            "log", request_id="1" * 32, workspace_id="d" * 32
        )
        with self.assertRaisesRegex(
            controller_service.ServiceError, "different researcher workspace"
        ):
            authority.pre_admit(request)
        authority.protected.verify.assert_not_called()


class StrictClientResponseTests(unittest.TestCase):
    def setUp(self):
        self.log_request = control.build_request("log", request_id="e" * 32)
        self.run_request = control.build_request(
            "run",
            solution="Project/solutions/s001_ok.py",
            card="Project/research/attempts/i001_ok.json",
            request_id="f" * 32,
        )

    def test_accepts_only_the_two_exact_bound_envelopes(self):
        success = execution_response(self.log_request)
        failure = error_response(self.log_request)
        self.assertEqual(
            control.validate_response(self.log_request, success), success
        )
        self.assertEqual(
            control.validate_response(self.log_request, failure), failure
        )

    def test_rejects_mismatched_spoofed_or_nonfinite_execution_response(self):
        base = execution_response(self.log_request)
        mutations = {
            "request id": {**base, "request_id": "0" * 32},
            "command": {**base, "command": "run"},
            "bool protocol": {**base, "protocol_version": True},
            "extra key": {**base, "private_predictions": []},
            "missing key": {key: value for key, value in base.items() if key != "stderr"},
            "ok status": {**base, "ok": False},
            "bool return code": {**base, "returncode": False},
            "nan elapsed": {**base, "elapsed_seconds": float("nan")},
            "infinite elapsed": {**base, "elapsed_seconds": float("inf")},
            "huge elapsed": {**base, "elapsed_seconds": 10**4000},
            "negative elapsed": {**base, "elapsed_seconds": -0.1},
            "wrong output type": {**base, "stdout": ["spoofed"]},
            "lone surrogate": {**base, "stdout": "\ud800"},
            "log commit": {**base, "artifact_commit": {
                "git_revision": "1" * 40,
                "solution_sha256": "2" * 64,
                "card_sha256": "3" * 64,
            }},
        }
        for label, response in mutations.items():
            with self.subTest(label=label), self.assertRaises(control.ClientError):
                control.validate_response(self.log_request, response)

    def test_error_and_run_commit_shapes_are_exact(self):
        error = error_response(self.log_request)
        invalid_errors = (
            {key: value for key, value in error.items() if key != "recovery_required"},
            {**error, "recovery_required": 1},
            {**error, "stdout": "not part of the error envelope"},
            {**error, "returncode": 1},
            {**error, "request_id": "0" * 32},
        )
        for response in invalid_errors:
            with self.assertRaises(control.ClientError):
                control.validate_response(self.log_request, response)

        valid_run = execution_response(self.run_request)
        self.assertEqual(
            control.validate_response(self.run_request, valid_run), valid_run
        )
        for change in (
            None,
            {"git_revision": "1" * 39,
             "solution_sha256": "2" * 64, "card_sha256": "3" * 64},
            {**valid_run["artifact_commit"], "extra": "not allowed"},
        ):
            invalid = copy.deepcopy(valid_run)
            invalid["artifact_commit"] = change
            with self.assertRaises(control.ClientError):
                control.validate_response(self.run_request, invalid)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(self):
        with self.assertRaises(control.ClientError):
            control._strict_object(b'{"ok":true,"ok":false}', "response")
        with self.assertRaises(control.ClientError):
            control._strict_object(b'{"elapsed":NaN}', "response")

    def test_worst_escaped_server_response_fits_client_replay_limit(self):
        response = execution_response(
            self.log_request,
            stdout="\x00" * controller_service.MAX_RESPONSE_STREAM_BYTES,
            stderr="\x1f" * controller_service.MAX_RESPONSE_STREAM_BYTES,
        )
        validated = controller_service.validate_response(self.log_request, response)
        payload = control.response_bytes(self.log_request, validated)
        self.assertGreater(len(payload), 512 * 1024)
        self.assertLessEqual(
            controller_service.MAX_COMPLETION_ROW_BYTES,
            control.MAX_RESPONSE_BYTES,
        )
        self.assertEqual(
            control._response_from_bytes(
                self.log_request, payload, "worst-case server response"
            ),
            response,
        )


class HashSolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="track2-hash-solution-", dir="/tmp"
        )
        self.root = Path(self.temp.name)
        self.solutions = self.root / "Project" / "solutions"
        self.solutions.mkdir(parents=True)
        self.relative = "Project/solutions/s001_hash.py"
        self.path = self.root / self.relative
        self.payload = b"def run(splits):\n    return {}\n"
        self.path.write_bytes(self.payload)
        self.path.chmod(0o600)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_hash_and_size_are_returned(self):
        result = control.hash_solution(self.relative, workspace_root=self.root)
        self.assertEqual(result, {
            "path": self.relative,
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "size": len(self.payload),
        })

    def test_symlink_hardlink_mode_and_size_are_refused(self):
        sibling = self.solutions / "s001_sibling.py"
        sibling.write_bytes(self.payload)
        sibling.chmod(0o600)
        cases = ("symlink", "hardlink", "writable", "oversized")
        for case in cases:
            with self.subTest(case=case):
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                if case == "symlink":
                    self.path.symlink_to(sibling)
                elif case == "hardlink":
                    os.link(sibling, self.path)
                elif case == "writable":
                    self.path.write_bytes(self.payload)
                    self.path.chmod(0o620)
                else:
                    self.path.write_bytes(b"x" * (control.MAX_SOLUTION_BYTES + 1))
                    self.path.chmod(0o600)
                with self.assertRaises(control.ClientError):
                    control.hash_solution(self.relative, workspace_root=self.root)
                if case == "hardlink":
                    self.path.unlink()

    def test_fifo_is_opened_nonblocking_and_refused(self):
        self.path.unlink()
        os.mkfifo(self.path, 0o600)
        real_open = os.open

        def checked_open(path, flags, *args, **kwargs):
            if path == self.path.name and kwargs.get("dir_fd") is not None:
                self.assertTrue(flags & os.O_NONBLOCK)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(control.os, "open", side_effect=checked_open):
            with self.assertRaisesRegex(control.ClientError, "regular file"):
                control.hash_solution(self.relative, workspace_root=self.root)


class DurableClientRequestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="track2-client-state-test-", dir="/tmp"
        )
        self.base = Path(self.temp.name)
        self.base.chmod(0o700)
        self.store = control.RequestStore(self.base / "client-state")
        self.request = control.build_request("log", request_id="7" * 32)
        self.payload = control.request_bytes(self.request)

    def tearDown(self):
        self.temp.cleanup()

    def test_new_request_is_fsynced_before_send_and_response_is_durable(self):
        response = execution_response(self.request)
        observations = []

        def fake_rpc(socket_path, request, timeout_seconds, *, exact_request_bytes):
            observations.append((socket_path, request.copy(), exact_request_bytes))
            self.assertEqual(self.store.pending_path.read_bytes(), self.payload)
            self.assertFalse(self.store.last_path.exists())
            return response

        with mock.patch.object(control, "rpc", side_effect=fake_rpc):
            actual = control.issue_request(
                Path("/unused/controller.sock"), self.request, store=self.store
            )
        self.assertEqual(actual, response)
        self.assertEqual(observations[0][1], self.request)
        self.assertEqual(observations[0][2], self.payload)
        self.assertFalse(self.store.pending_path.exists())
        self.assertEqual(self.store.last_path.read_bytes(), self.payload)
        self.assertEqual(
            control._response_from_bytes(
                self.request,
                self.store.response_path.read_bytes(),
                "test response",
            ),
            response,
        )
        self.assertEqual(os.stat(self.store.state_dir).st_mode & 0o777, 0o700)
        for path in (
            self.store.last_path, self.store.response_path, self.store.lock_path
        ):
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_failed_send_leaves_exact_pending_and_retry_reuses_id_and_bytes(self):
        with mock.patch.object(
            control, "rpc", side_effect=control.ClientError("simulated crash")
        ):
            with self.assertRaises(control.ClientError):
                control.issue_request(
                    Path("/unused/controller.sock"), self.request, store=self.store
                )
        self.assertEqual(self.store.pending_path.read_bytes(), self.payload)

        response = execution_response(self.request)
        calls = []

        def fake_retry(socket_path, request, timeout_seconds, *, exact_request_bytes):
            calls.append((request["request_id"], exact_request_bytes))
            return response

        with mock.patch.object(control, "rpc", side_effect=fake_retry):
            actual = control.retry_persisted(
                Path("/unused/controller.sock"),
                recover_completed=False,
                store=self.store,
            )
        self.assertEqual(actual, response)
        self.assertEqual(calls, [(self.request["request_id"], self.payload)])

        with mock.patch.object(
            control, "rpc", side_effect=AssertionError("recover must be local")
        ):
            recovered = control.retry_persisted(
                Path("/unused/controller.sock"),
                recover_completed=True,
                store=self.store,
            )
        self.assertEqual(recovered, response)

    def test_crash_after_response_fsync_recovers_without_resending(self):
        self.store.persist_new(self.request)
        response = execution_response(self.request)
        self.store._write_atomic(
            self.store.response_path, control.response_bytes(self.request, response)
        )
        with mock.patch.object(
            control, "rpc", side_effect=AssertionError("must use durable response")
        ):
            recovered = control.retry_persisted(
                Path("/unused/controller.sock"),
                recover_completed=False,
                store=self.store,
            )
        self.assertEqual(recovered, response)
        self.assertFalse(self.store.pending_path.exists())
        self.assertEqual(self.store.last_path.read_bytes(), self.payload)

    def test_indeterminate_server_wal_response_keeps_request_pending(self):
        response = error_response(
            self.request, error="owner recovery required", recovery_required=True
        )
        with mock.patch.object(control, "rpc", return_value=response):
            actual = control.issue_request(
                Path("/unused/controller.sock"), self.request, store=self.store
            )
        self.assertEqual(actual, response)
        self.assertEqual(self.store.pending_path.read_bytes(), self.payload)
        self.assertFalse(self.store.last_path.exists())
        self.assertFalse(self.store.response_path.exists())

        with mock.patch.object(
            control, "rpc", side_effect=AssertionError("recover must never replay")
        ):
            with self.assertRaisesRegex(control.ClientError, "local-only"):
                control.retry_persisted(
                    Path("/unused/controller.sock"),
                    recover_completed=True,
                    store=self.store,
                )

    def test_completion_is_idempotent_for_concurrent_exact_retry(self):
        self.store.persist_new(self.request)
        response = execution_response(self.request)
        self.store.mark_completed(self.payload, response)
        self.store.mark_completed(self.payload, response)
        self.assertEqual(self.store.last_path.read_bytes(), self.payload)

    def test_store_refuses_repository_and_unsafe_parent_locations(self):
        repository_state = ROOT / "Project" / "research" / "client-state-test"
        self.assertFalse(repository_state.exists())
        with self.assertRaisesRegex(control.ClientError, "repository"):
            control.RequestStore(repository_state)
        self.assertFalse(repository_state.exists())

        unsafe_parent = self.base / "unsafe"
        unsafe_parent.mkdir(mode=0o777)
        unsafe_parent.chmod(0o777)
        with self.assertRaises(control.ClientError):
            control.RequestStore(unsafe_parent / "state")

    def test_persisted_state_cannot_cross_workspace_bindings(self):
        pending_store = control.RequestStore(self.base / "pending-cross-workspace")
        pending_store.persist_new(self.request)
        with mock.patch.dict(
            os.environ, {control.WORKSPACE_ID_ENV: "d" * 32}, clear=False
        ):
            with self.assertRaisesRegex(control.ClientError, "different researcher"):
                pending_store.load(recover_completed=False)

        completed_store = control.RequestStore(self.base / "done-cross-workspace")
        payload = completed_store.persist_new(self.request)
        completed_store.mark_completed(payload, execution_response(self.request))
        with mock.patch.dict(
            os.environ, {control.WORKSPACE_ID_ENV: "d" * 32}, clear=False
        ):
            with self.assertRaisesRegex(control.ClientError, "different researcher"):
                completed_store.load(recover_completed=True)

    def test_default_store_is_namespaced_by_workspace_id(self):
        root = self.base / "namespaced-client-state"
        with mock.patch.dict(
            os.environ,
            {
                control.CLIENT_STATE_ENV: str(root),
                control.WORKSPACE_ID_ENV: "c" * 32,
            },
            clear=False,
        ):
            first = control.RequestStore.default()
            first_request = control.build_request("log", request_id="1" * 32)
            first_payload = first.persist_new(first_request)
        with mock.patch.dict(
            os.environ,
            {
                control.CLIENT_STATE_ENV: str(root),
                control.WORKSPACE_ID_ENV: "d" * 32,
                control.WORKSPACE_BINDING_ENV: "f" * 64,
            },
            clear=False,
        ):
            second = control.RequestStore.default()
            second_request = control.build_request("log", request_id="2" * 32)
            second_payload = second.persist_new(second_request)
        self.assertNotEqual(first.state_dir, second.state_dir)
        self.assertEqual(first.state_dir.parent, second.state_dir.parent)
        self.assertEqual(first.pending_path.read_bytes(), first_payload)
        self.assertEqual(second.pending_path.read_bytes(), second_payload)

    def test_cli_routes_new_and_replay_commands_without_regenerating_replay(self):
        response = execution_response(self.request)
        with (
            mock.patch.dict(os.environ, {control.SOCKET_ENV: "/unused/socket"}),
            mock.patch.object(sys, "argv", ["control.py", "log"]),
            mock.patch.object(control, "issue_request", return_value=response) as issue,
            mock.patch.object(sys, "stdout", new=io.StringIO()),
        ):
            self.assertEqual(control.main(), 0)
        issue.assert_called_once()

        for command, recover_completed in (("retry", False), ("recover", True)):
            with (
                self.subTest(command=command),
                mock.patch.dict(os.environ, {control.SOCKET_ENV: "/unused/socket"}),
                mock.patch.object(sys, "argv", ["control.py", command]),
                mock.patch.object(
                    control, "build_request",
                    side_effect=AssertionError("replay minted a request"),
                ),
                mock.patch.object(
                    control, "retry_persisted", return_value=response
                ) as replay,
                mock.patch.object(sys, "stdout", new=io.StringIO()),
            ):
                self.assertEqual(control.main(), 0)
            replay.assert_called_once_with(
                Path("/unused/socket"), recover_completed=recover_completed
            )


class ArtifactWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="track2-workspace-test-", dir="/tmp"
        )
        self.base = Path(self.temp.name)
        self.workspace = make_researcher_workspace(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def test_reads_exact_private_file_and_refuses_symlink_replacement(self):
        relative = "Project/solutions/s001_workspace.py"
        payload = b"HYPOTHESIS='workspace'\ndef run(splits): return {}\n"
        staged = self.workspace / "solutions" / Path(relative).name
        staged.write_bytes(payload)
        staged.chmod(0o600)
        workspace = controller_service.ArtifactWorkspace(self.workspace)
        self.assertEqual(workspace.read(relative, card=False), payload)

        staged.unlink()
        staged.symlink_to("/etc/passwd")
        with self.assertRaisesRegex(controller_service.ServiceError, "unsafe"):
            workspace.read(relative, card=False)

    def test_same_size_concurrent_edit_is_refused(self):
        relative = "Project/solutions/s001_workspace.py"
        original = b"HYPOTHESIS='workspace'\ndef run(splits): return {}\n"
        replacement = b"X" * len(original)
        staged = self.workspace / "solutions" / Path(relative).name
        staged.write_bytes(original)
        staged.chmod(0o600)
        workspace = controller_service.ArtifactWorkspace(self.workspace)
        real_read = os.read
        changed = False

        def racing_read(fd, count):
            nonlocal changed
            chunk = real_read(fd, count)
            if chunk and not changed:
                changed = True
                staged.write_bytes(replacement)
            return chunk

        with mock.patch.object(controller_service.os, "read", side_effect=racing_read):
            with self.assertRaisesRegex(
                controller_service.ServiceError, "changed while being read"
            ):
                workspace.read(relative, card=False)
        self.assertTrue(changed)

    def test_fifo_artifact_is_opened_nonblocking_and_refused(self):
        relative = "Project/solutions/s001_fifo.py"
        staged = self.workspace / "solutions" / Path(relative).name
        os.mkfifo(staged, 0o600)
        workspace = controller_service.ArtifactWorkspace(self.workspace)
        real_open = os.open

        def checked_open(path, flags, *args, **kwargs):
            if path == staged.name and kwargs.get("dir_fd") is not None:
                self.assertTrue(flags & os.O_NONBLOCK)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            controller_service.os, "open", side_effect=checked_open
        ):
            with self.assertRaisesRegex(controller_service.ServiceError, "regular file"):
                workspace.read(relative, card=False)

    def test_rejects_nonprivate_workspace_root(self):
        self.workspace.chmod(0o750)
        with self.assertRaisesRegex(controller_service.ServiceError, "mode 0700"):
            controller_service.ArtifactWorkspace(self.workspace)

    def test_filesystem_root_cannot_be_used_as_private_authority_state(self):
        with self.assertRaisesRegex(controller_service.ServiceError, "filesystem root"):
            controller_service._private_workspace_root(Path("/"))

    def test_frozen_portfolio_copy_is_exact_and_hash_bound(self):
        portfolio_path = self.workspace / researcher_shell.WORKSPACE_PORTFOLIO
        expected = json.loads(portfolio_path.read_bytes())
        workspace = controller_service.ArtifactWorkspace(self.workspace)
        self.assertEqual(workspace.portfolio, expected)

        portfolio_path.write_bytes(portfolio_path.read_bytes() + b" ")
        with self.assertRaisesRegex(controller_service.ServiceError, "wrong shape"):
            controller_service.ArtifactWorkspace(self.workspace)

    def test_near_source_cap_portfolio_remains_fully_visible(self):
        portfolio_path = self.workspace / researcher_shell.WORKSPACE_PORTFOLIO
        portfolio = json.loads(portfolio_path.read_bytes())
        long_text = "mechanism-evidence-" + ("x" * 11_450)
        for family in portfolio["families"]:
            for field in (
                "mechanism", "causal_claim", "smallest_experiment",
                "falsifier", "known_risks",
            ):
                family[field] = long_text
        portfolio["selection_rubric"] = "rubric-" + ("y" * 9_900)
        iterate.policy.validate_portfolio(portfolio)
        source = json.dumps(
            portfolio, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        self.assertLessEqual(source.__len__(), iterate.policy.MAX_PORTFOLIO_SOURCE_BYTES)
        self.assertGreater(source.__len__(), 200 * 1024)
        view = json.dumps(
            portfolio, sort_keys=True, indent=2, ensure_ascii=False
        ).encode("utf-8") + b"\n"
        self.assertLessEqual(len(view), researcher_shell.MAX_PORTFOLIO_VIEW_BYTES)
        portfolio_path.write_bytes(view)
        manifest_path = self.workspace / researcher_shell.WORKSPACE_MANIFEST
        manifest = json.loads(manifest_path.read_bytes())
        manifest["portfolio_sha256"] = hashlib.sha256(view).hexdigest()
        manifest_path.write_text(json.dumps(manifest) + "\n")
        loaded = controller_service.ArtifactWorkspace(self.workspace)
        self.assertEqual(loaded.portfolio, portfolio)
        self.assertNotIn("[truncated]", view.decode("utf-8"))

    def test_same_logical_id_has_distinct_physical_workspace_bindings(self):
        other_base = self.base / "other"
        other_base.mkdir(mode=0o700)
        other = make_researcher_workspace(other_base)
        first = controller_service.ArtifactWorkspace(self.workspace)
        second = controller_service.ArtifactWorkspace(other)
        _root, _manifest, _portfolio, shell_binding = (
            researcher_shell._researcher_workspace(
                self.workspace, ROOT, self.base / "runtime-placeholder", None
            )
        )
        self.assertEqual(first.binding, shell_binding)
        self.assertEqual(first.workspace_id, second.workspace_id)
        self.assertNotEqual(first.binding, second.binding)

        authority = object.__new__(controller_service.ControllerAuthority)
        authority.workspace = first
        authority.protected = mock.Mock()
        request = control.build_request(
            "log", request_id="2" * 32,
            workspace_id=second.workspace_id,
            workspace_binding=second.binding,
        )
        with self.assertRaisesRegex(
            controller_service.ServiceError, "different researcher workspace"
        ):
            authority.execute(request)
        authority.protected.verify.assert_not_called()

    def test_non_ascii_workspace_path_has_one_physical_binding(self):
        unicode_base = self.base / "research-研究"
        unicode_base.mkdir(mode=0o700)
        workspace_path = make_researcher_workspace(unicode_base)
        service_workspace = controller_service.ArtifactWorkspace(workspace_path)
        _root, _manifest, _portfolio, shell_binding = (
            researcher_shell._researcher_workspace(
                workspace_path, ROOT, self.base / "runtime-placeholder", None
            )
        )
        self.assertEqual(service_workspace.binding, shell_binding)

    def test_runtime_and_authority_state_must_be_disjoint(self):
        root = self.base / "repo"
        workspace = self.base / "workspace-other"
        authority = self.base / "authority"
        with self.assertRaisesRegex(
            controller_service.ServiceError,
            "controller runtime must be disjoint from authority state",
        ):
            controller_service._validate_external_layout(
                root, authority / "runtime", workspace, authority
            )

    def test_repository_and_authority_state_must_be_disjoint(self):
        authority = self.base / "authority"
        with self.assertRaisesRegex(
            controller_service.ServiceError,
            "repository must be disjoint from authority state",
        ):
            controller_service._validate_external_layout(
                authority / "checkout",
                self.base / "runtime",
                self.base / "workspace-other",
                authority,
            )


class WorkspaceInitializerTests(unittest.TestCase):
    def test_terminal_rebuild_preserves_requested_logical_id_and_frozen_base(self):
        with tempfile.TemporaryDirectory(
            prefix="track2-workspace-init-test-", dir="/tmp"
        ) as tmp:
            parent = Path(tmp)
            parent.chmod(0o700)
            target = parent / "rebuilt"
            portfolio = json.loads((
                ROOT / "Project/research/templates/portfolio.template.json"
            ).read_bytes())
            base_revision = "a" * 40
            current_revision = "b" * 40
            state = {
                "official_run_started": True,
                "state": "TERMINAL",
                "official_iterations": 4,
                "open_attempt": False,
                "run_id": "1" * 32,
                "run_start_git_revision": base_revision,
                "portfolio": portfolio,
            }
            verifier = mock.Mock()
            verifier.outer_policy.MAX_PORTFOLIO_VIEW_BYTES = (
                iterate.policy.MAX_PORTFOLIO_VIEW_BYTES
            )
            verifier.fixed_environment.return_value = {}
            verifier._validate_controller_commit_chain.return_value = None
            with (
                mock.patch.object(
                    init_researcher_workspace,
                    "_controller_lock",
                    return_value=contextlib.nullcontext(),
                ),
                mock.patch.object(
                    init_researcher_workspace,
                    "_run",
                    side_effect=[
                        json.dumps(state).encode("utf-8"),
                        (current_revision + "\n").encode("ascii"),
                    ],
                ),
                mock.patch.object(
                    init_researcher_workspace,
                    "_load_controller_service",
                    return_value=verifier,
                ),
            ):
                manifest = init_researcher_workspace.create_workspace(
                    target, recover_workspace_id="c" * 32
                )
            self.assertEqual(manifest["workspace_id"], "c" * 32)
            self.assertEqual(manifest["repository_head"], base_revision)
            verifier._validate_controller_commit_chain.assert_called_once_with(
                ROOT, base_revision, current_revision, {}
            )
            copied = json.loads((
                target / init_researcher_workspace.PORTFOLIO
            ).read_bytes())
            self.assertEqual(copied, portfolio)
            self.assertEqual(
                hashlib.sha256((
                    target / init_researcher_workspace.PORTFOLIO
                ).read_bytes()).hexdigest(),
                manifest["portfolio_sha256"],
            )
            self.assertEqual(
                set(path.name for path in target.iterdir()),
                set(init_researcher_workspace.SUBDIRS)
                | {
                    init_researcher_workspace.MANIFEST,
                    init_researcher_workspace.PORTFOLIO,
                },
            )


class FixedGitBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticGitRepo()

    def tearDown(self):
        self.fixture.close()

    def test_fixed_options_override_unsafe_repository_configuration(self):
        self.fixture._git("config", "core.fsync", "none")
        self.fixture._git("config", "core.fsyncMethod", "writeout-only")
        self.assertEqual(
            controller_service._git(
                self.fixture.root, ["config", "--get", "core.fsync"],
                env=self.fixture.env,
            ).stdout.strip(),
            "all",
        )
        self.assertEqual(
            controller_service._git(
                self.fixture.root, ["config", "--get", "core.fsyncMethod"],
                env=self.fixture.env,
            ).stdout.strip(),
            "fsync",
        )
        command = controller_service._git_command(["version"])
        self.assertIn("--no-replace-objects", command)
        self.assertIn("--no-lazy-fetch", command)
        self.assertIn("core.fsync=all", command)
        self.assertIn("core.fsyncMethod=fsync", command)
        self.assertEqual(self.fixture.env["GIT_CONFIG_GLOBAL"], "/dev/null")
        self.assertEqual(self.fixture.env["GIT_CONFIG_NOSYSTEM"], "1")

    def test_replacement_refs_are_rejected_even_though_commands_disable_them(self):
        head = self.fixture._git("rev-parse", "HEAD").strip()
        self.fixture._git("update-ref", f"refs/replace/{head}", head)
        with self.assertRaisesRegex(
            controller_service.ServiceError, "replacement refs"
        ):
            controller_service._validate_git_repository(
                self.fixture.root, self.fixture.env
            )

    def test_legacy_grafts_are_rejected(self):
        head = self.fixture._git("rev-parse", "HEAD").strip()
        grafts = self.fixture.root / ".git" / "info" / "grafts"
        grafts.write_text(head + "\n")
        with self.assertRaisesRegex(controller_service.ServiceError, "grafts"):
            controller_service._validate_git_repository(
                self.fixture.root, self.fixture.env
            )

    def test_external_common_directory_and_alternates_are_rejected(self):
        external = self.fixture.base / "external-git"
        external.mkdir()
        fake = subprocess.CompletedProcess(
            ["git", "rev-parse"], 0, stdout=str(external) + "\n", stderr=""
        )
        with mock.patch.object(controller_service, "_git", return_value=fake):
            with self.assertRaisesRegex(
                controller_service.ServiceError, "primary checkout"
            ):
                controller_service._validate_git_repository(
                    self.fixture.root, self.fixture.env
                )

        alternates = self.fixture.root / ".git" / "objects" / "info" / "alternates"
        alternates.write_text("/tmp/untrusted-object-store\n")
        with self.assertRaisesRegex(
            controller_service.ServiceError, "alternate object stores"
        ):
            controller_service._validate_git_repository(
                self.fixture.root, self.fixture.env
            )

    def test_promisor_configuration_is_rejected(self):
        self.fixture._git("config", "remote.origin.promisor", "true")
        with self.assertRaisesRegex(
            controller_service.ServiceError, "partial/promisor"
        ):
            controller_service._validate_git_repository(
                self.fixture.root, self.fixture.env
            )


class ExactCommitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticGitRepo()

    def tearDown(self):
        self.fixture.close()

    def test_commits_exact_pair_with_fixed_hash_message(self):
        solution, card, solution_bytes, card_bytes = self.fixture.artifacts()
        index_before = (self.fixture.root / ".git" / "index").read_bytes()
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        record = committer.commit_pair(solution, card)
        self.assertEqual(
            self.fixture._git("show", f"HEAD:{solution}", binary=True), solution_bytes
        )
        self.assertEqual(
            self.fixture._git("show", f"HEAD:{card}", binary=True), card_bytes
        )
        message = self.fixture._git("log", "-1", "--pretty=%B")
        self.assertIn(hashlib.sha256(solution_bytes).hexdigest(), message)
        self.assertIn(hashlib.sha256(card_bytes).hexdigest(), message)
        self.assertRegex(record["git_revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(
            (self.fixture.root / ".git" / "index").read_bytes(), index_before
        )

    def test_preserves_unrelated_staged_index_bytes(self):
        solution, card, _, _ = self.fixture.artifacts()
        outside = self.fixture.root / "outside.txt"
        outside.write_text("must not ride along")
        self.fixture._git("add", "--", "outside.txt")
        index_before = (self.fixture.root / ".git" / "index").read_bytes()
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        committer.commit_pair(solution, card)
        self.assertEqual(
            (self.fixture.root / ".git" / "index").read_bytes(), index_before
        )
        result = subprocess.run(
            ["/usr/bin/git", "show", "HEAD:outside.txt"],
            cwd=self.fixture.root,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_refuses_symlink_artifact(self):
        solution, card, _, _ = self.fixture.artifacts()
        path = self.fixture.root / solution
        path.unlink()
        path.symlink_to("/etc/passwd")
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        with self.assertRaisesRegex(controller_service.ServiceError, "unsafe"):
            committer.commit_pair(solution, card)

    def test_refuses_protected_drift(self):
        solution, card, _, _ = self.fixture.artifacts()
        protected = self.fixture.protected()
        self.fixture.iterate.write_text("raise SystemExit('changed')\n")
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, protected, self.fixture.env
        )
        with self.assertRaisesRegex(controller_service.ServiceError, "protected"):
            committer.commit_pair(solution, card)

    def test_live_rewrite_cannot_change_captured_commit_bytes(self):
        solution, card, solution_bytes, card_bytes = self.fixture.artifacts("race")
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        original = committer._hash_exact_file
        calls = 0

        def mutate_after_capture(path, payload, env):
            nonlocal calls
            calls += 1
            if calls == 1:
                (self.fixture.root / solution).write_text("raise SystemExit('raced')\n")
                (self.fixture.root / card).write_text('{"raced":true}\n')
            return original(path, payload, env)

        with mock.patch.object(
            committer, "_hash_exact_file", side_effect=mutate_after_capture
        ):
            committer.commit_pair(solution, card)
        self.assertEqual(
            self.fixture._git("show", f"HEAD:{solution}", binary=True), solution_bytes
        )
        self.assertEqual(
            self.fixture._git("show", f"HEAD:{card}", binary=True), card_bytes
        )

    def test_failed_private_tree_verification_leaves_head_unchanged(self):
        solution, card, _, _ = self.fixture.artifacts("treefail")
        old_head = self.fixture._git("rev-parse", "HEAD").strip()
        index_before = (self.fixture.root / ".git" / "index").read_bytes()
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        with mock.patch.object(committer, "_tree_blob", return_value="0" * 40):
            with self.assertRaisesRegex(
                controller_service.ServiceError, "exact blob verification"
            ):
                committer.commit_pair(solution, card)
        self.assertEqual(self.fixture._git("rev-parse", "HEAD").strip(), old_head)
        self.assertEqual(
            (self.fixture.root / ".git" / "index").read_bytes(), index_before
        )

    def test_gitattributes_cannot_transform_captured_candidate(self):
        attributes = self.fixture.root / ".gitattributes"
        attributes.write_text("*.py text eol=crlf\n")
        self.fixture._git("add", "--", ".gitattributes")
        self.fixture._git("commit", "-q", "-m", "attributes")
        solution, card, solution_bytes, _ = self.fixture.artifacts("attrs")
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        committer.commit_pair(solution, card)
        self.assertEqual(
            self.fixture._git("show", f"HEAD:{solution}", binary=True), solution_bytes
        )

    def test_second_materialization_failure_rolls_back_first_file_and_head(self):
        solution, card, solution_bytes, card_bytes = self.fixture.artifacts("partial")
        (self.fixture.root / solution).unlink()
        (self.fixture.root / card).unlink()
        old_head = self.fixture._git("rev-parse", "HEAD").strip()
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        original = committer._materialize_new_file
        calls = 0

        def fail_second(relative, payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise controller_service.ServiceError("simulated second-file failure")
            return original(relative, payload)

        with mock.patch.object(
            committer, "_materialize_new_file", side_effect=fail_second
        ):
            with self.assertRaisesRegex(
                controller_service.ServiceError, "second-file failure"
            ):
                committer.commit_pair(
                    solution,
                    card,
                    captured_solution_bytes=solution_bytes,
                    captured_card_bytes=card_bytes,
                    materialize_new_worktree_files=True,
                    expected_parent_revision=old_head,
                )
        self.assertEqual(self.fixture._git("rev-parse", "HEAD").strip(), old_head)
        self.assertFalse((self.fixture.root / solution).exists())
        self.assertFalse((self.fixture.root / card).exists())

    def test_materialization_publishes_only_after_full_staging_write(self):
        solution, _card, solution_bytes, _card_bytes = self.fixture.artifacts(
            "atomic"
        )
        final_path = self.fixture.root / solution
        final_path.unlink()
        parent = final_path.parent
        temporary_leaf = controller_service._materialization_temp_leaf(
            solution, hashlib.sha256(solution_bytes).hexdigest()
        )
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )

        def refuse_publish(_directory_fd, source, destination):
            self.assertEqual(source, temporary_leaf)
            self.assertEqual(destination, final_path.name)
            self.assertFalse(final_path.exists())
            self.assertEqual((parent / source).read_bytes(), solution_bytes)
            raise controller_service.ServiceError("simulated publication refusal")

        with mock.patch.object(
            controller_service, "_rename_noreplace", side_effect=refuse_publish
        ):
            with self.assertRaisesRegex(
                controller_service.ServiceError, "publication refusal"
            ):
                committer._materialize_new_file(solution, solution_bytes)
        self.assertFalse(final_path.exists())
        self.assertFalse((parent / temporary_leaf).exists())

    def test_failed_head_cas_rolls_back_both_materialized_files(self):
        solution, card, solution_bytes, card_bytes = self.fixture.artifacts("cas")
        (self.fixture.root / solution).unlink()
        (self.fixture.root / card).unlink()
        old_head = self.fixture._git("rev-parse", "HEAD").strip()
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        original_git = controller_service._git

        def refuse_update(root, args, **kwargs):
            if args[0] == "update-ref":
                raise controller_service.ServiceError("simulated CAS refusal")
            return original_git(root, args, **kwargs)

        with mock.patch.object(controller_service, "_git", side_effect=refuse_update):
            with self.assertRaisesRegex(controller_service.ServiceError, "CAS refusal"):
                committer.commit_pair(
                    solution,
                    card,
                    captured_solution_bytes=solution_bytes,
                    captured_card_bytes=card_bytes,
                    materialize_new_worktree_files=True,
                    expected_parent_revision=old_head,
                )
        self.assertEqual(self.fixture._git("rev-parse", "HEAD").strip(), old_head)
        self.assertFalse((self.fixture.root / solution).exists())
        self.assertFalse((self.fixture.root / card).exists())

    def test_parent_revision_binding_refuses_precommit_head_drift(self):
        solution, card, solution_bytes, card_bytes = self.fixture.artifacts("parent")
        (self.fixture.root / solution).unlink()
        (self.fixture.root / card).unlink()
        old_head = self.fixture._git("rev-parse", "HEAD").strip()
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        with self.assertRaisesRegex(controller_service.ServiceError, "HEAD changed"):
            committer.commit_pair(
                solution,
                card,
                captured_solution_bytes=solution_bytes,
                captured_card_bytes=card_bytes,
                materialize_new_worktree_files=True,
                expected_parent_revision="0" * 40,
            )
        self.assertEqual(self.fixture._git("rev-parse", "HEAD").strip(), old_head)
        self.assertFalse((self.fixture.root / solution).exists())
        self.assertFalse((self.fixture.root / card).exists())


class PreAdmissionTests(unittest.TestCase):
    def test_reused_repository_destinations_are_refused_before_workspace_read(self):
        fixture = SyntheticGitRepo()
        try:
            solution, card, _solution_bytes, _card_bytes = fixture.artifacts("reused")
            fixture._git("add", "--", solution, card)
            fixture._git("commit", "-q", "-m", "existing attempt artifacts")
            authority = object.__new__(controller_service.ControllerAuthority)
            authority.root = fixture.root
            authority.env = fixture.env
            authority.workspace = mock.Mock()
            request = control.build_request(
                "run", solution=solution, card=card, request_id="3" * 32
            )
            state = subprocess.CompletedProcess(
                ["iterate.py", "_admission-state"],
                0,
                stdout=json.dumps({
                    "official_run_started": True,
                    "state": "ACTIVE",
                    "open_attempt": False,
                    "would_trigger_now": [],
                    "attempt_review_budget_remaining": 3,
                    "attempt_review_failure_budget_remaining": 3,
                }),
                stderr="",
            )
            with mock.patch.object(
                controller_service, "_run_process", return_value=state
            ):
                with self.assertRaisesRegex(
                    controller_service.ServiceError, "already exists"
                ):
                    authority._admit_run(request)
            authority.workspace.read.assert_not_called()
        finally:
            fixture.close()


class SharedControllerLockTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticGitRepo()
        (self.fixture.root / "Project" / "results").mkdir()

    def tearDown(self):
        self.fixture.close()

    def test_outer_lock_is_inherited_and_blocks_an_independent_owner_command(self):
        authority = object.__new__(controller_service.ControllerAuthority)
        authority.root = self.fixture.root
        authority._transaction_fd = None
        lock_path = self.fixture.root / "Project" / "results" / ".controller.lock"

        with authority.transaction():
            inherited_fd = authority._transaction_fd
            self.assertIsNotNone(inherited_fd)
            assert inherited_fd is not None
            with (
                mock.patch.object(iterate, "LOCK_PATH", lock_path),
                mock.patch.dict(
                    os.environ,
                    {iterate.INHERITED_LOCK_FD_ENV: str(inherited_fd)},
                    clear=False,
                ),
                iterate.controller_lock(),
            ):
                competing = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(competing)
        self.assertIsNone(authority._transaction_fd)


class ServerLifetimeTests(unittest.TestCase):
    def test_graceful_close_joins_request_threads_before_releasing_authority(self):
        self.assertFalse(controller_service.ControllerServer.daemon_threads)
        self.assertTrue(controller_service.ControllerServer.block_on_close)

    def test_runtime_endpoints_cannot_collide_with_service_lock(self):
        runtime = Path("/tmp/track2-controller-layout-test")
        with self.assertRaisesRegex(
            controller_service.ServiceError, "reserved service lock"
        ):
            controller_service._validate_runtime_endpoints(
                runtime / "controller.sock",
                runtime / controller_service.SERVICE_LOCK_NAME,
            )
        with self.assertRaisesRegex(
            controller_service.ServiceError, "reserved service lock"
        ):
            controller_service._validate_runtime_endpoints(
                runtime / controller_service.SERVICE_LOCK_NAME,
                runtime / "controller.audit.jsonl",
            )

    def test_distinct_runtime_locks_cannot_bypass_repository_lifetime_lock(self):
        fixture = SyntheticGitRepo()
        try:
            (fixture.root / "Project" / "results").mkdir()
            runtime_a = fixture.base / "runtime-a"
            runtime_b = fixture.base / "runtime-b"
            runtime_a.mkdir(mode=0o700)
            runtime_b.mkdir(mode=0o700)
            repository_fd = controller_service._acquire_repository_service_lock(
                fixture.root
            )
            runtime_a_fd = controller_service._acquire_service_lock(runtime_a)
            runtime_b_fd = controller_service._acquire_service_lock(runtime_b)
            try:
                with self.assertRaisesRegex(
                    controller_service.ServiceError, "owns this repository"
                ):
                    controller_service._acquire_repository_service_lock(
                        fixture.root
                    )
            finally:
                os.close(runtime_b_fd)
                os.close(runtime_a_fd)
                os.close(repository_fd)
        finally:
            fixture.close()


class ServiceBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticGitRepo()
        self.state = self.fixture.base / "authority-state"
        self.runtime = self.fixture.base / "runtime"
        self.other_runtime = self.fixture.base / "other-runtime"
        self.workspace_root = self.fixture.base / "workspace-binding"
        for path in (
            self.state, self.runtime, self.other_runtime, self.workspace_root
        ):
            path.mkdir(mode=0o700)
        self.audit = self.runtime / "controller.audit.jsonl"
        self.workspace = type("BindingWorkspace", (), {})()
        self.workspace.root = self.workspace_root
        self.workspace.workspace_id = "c" * 32
        self.workspace.binding = "e" * 64
        self.workspace.run_id = "a" * 32

    def tearDown(self):
        self.fixture.close()

    def enforce(self, audit: Path | None = None, *, create: bool = True):
        return controller_service._enforce_service_binding(
            state_dir=self.state,
            root=self.fixture.root,
            audit_path=self.audit if audit is None else audit,
            create=create,
        )

    def test_first_service_atomically_freezes_repository_wal(self):
        first = self.enforce()
        second = self.enforce()
        self.assertEqual(first, second)
        binding_path = self.state / controller_service.SERVICE_BINDING_NAME
        self.assertEqual(stat.S_IMODE(binding_path.stat().st_mode), 0o600)
        self.assertEqual(first["audit_log"], str(self.audit))

    def test_restart_cannot_select_a_fresh_wal_in_another_runtime(self):
        self.enforce()
        changed = self.other_runtime / "fresh.audit.jsonl"
        with self.assertRaisesRegex(
            controller_service.ServiceError, "frozen audit_log"
        ):
            self.enforce(changed)

    def test_offline_recovery_requires_the_existing_binding(self):
        with self.assertRaisesRegex(
            controller_service.ServiceError, "binding is missing"
        ):
            self.enforce(create=False)
        self.enforce()
        recovered = self.enforce(create=False)
        self.assertEqual(recovered["audit_log"], str(self.audit))


class InnerTransitionClassificationTests(unittest.TestCase):
    def test_refused_child_with_open_attempt_keeps_outer_transition_pending(self):
        authority = object.__new__(controller_service.ControllerAuthority)
        authority.root = ROOT
        authority.env = controller_service.fixed_environment(
            {"HOME": "/tmp", "LANG": "C.UTF-8"}
        )
        authority.workspace = mock.Mock(
            workspace_id="c" * 32, binding="e" * 64
        )
        authority._transaction_fd = 7
        authority.protected = mock.Mock()
        authority.committer = mock.Mock()
        authority.committer.commit_pair.return_value = {
            "git_revision": "1" * 40,
            "solution_sha256": "2" * 64,
            "card_sha256": "3" * 64,
        }
        solution_bytes = b"def run(splits): return {}\n"
        card_bytes = b'{"schema_version":2}\n'
        request = control.build_request(
            "run",
            solution="Project/solutions/s001_open.py",
            card="Project/research/attempts/i001_open.json",
            request_id="5" * 32,
        )
        admission = (
            hashlib.sha256(solution_bytes).hexdigest(),
            hashlib.sha256(card_bytes).hexdigest(),
            solution_bytes,
            card_bytes,
            "4" * 40,
        )
        refused = subprocess.CompletedProcess(
            ["iterate.py", "run"], 1, stdout="", stderr="REFUSED: after open"
        )
        open_state = subprocess.CompletedProcess(
            ["iterate.py", "_admission-state"],
            0,
            stdout=json.dumps({
                "official_run_started": True,
                "open_attempt": True,
            }),
            stderr="",
        )
        with mock.patch.object(
            controller_service, "_run_process", side_effect=[refused, open_state]
        ):
            with self.assertRaisesRegex(
                controller_service.IndeterminateTransition, "interrupted official attempt"
            ):
                authority.execute(request, admission=admission)


class BankProtectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticGitRepo()
        notes = self.fixture.root / "Project" / "research" / "bank" / "notes"
        notes.mkdir(parents=True)
        self.note = notes / "ranking.md"
        self.note.write_text("frozen evidence\n")
        catalog = {
            "schema_version": 1,
            "benchmark": "KuaiRand-Pure",
            "claims": [{"note_path": "Project/research/bank/notes/ranking.md"}],
        }
        (notes.parent / "catalog.json").write_text(json.dumps(catalog))
        self.fixture._git("add", "--", "Project/research/bank")
        self.fixture._git("commit", "-q", "-m", "bank")

    def tearDown(self):
        self.fixture.close()

    def test_catalog_resolves_fixed_note_protection(self):
        paths = controller_service.frozen_bank_paths(
            self.fixture.root, self.fixture.env
        )
        self.assertEqual(paths, (
            "Project/research/bank/catalog.json",
            "Project/research/bank/notes/ranking.md",
        ))
        protected = controller_service.ProtectedState(
            self.fixture.root, paths=paths, env=self.fixture.env
        )
        self.note.write_text("mutated evidence\n")
        with self.assertRaises(controller_service.ServiceError):
            protected.verify()


class ServerWalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="track2-server-wal-test-", dir="/tmp"
        )
        self.runtime = Path(self.temp.name)
        self.runtime.chmod(0o700)
        self.path = self.runtime / "controller.audit.jsonl"
        self.request = control.build_request(
            "run",
            solution="Project/solutions/s001_wal.py",
            card="Project/research/attempts/i001_wal.json",
            request_id="9" * 32,
        )
        solution_bytes = b"def run(splits):\n    return {}\n"
        card_bytes = b'{"schema_version":2}\n'
        self.admission = (
            hashlib.sha256(solution_bytes).hexdigest(),
            hashlib.sha256(card_bytes).hexdigest(),
            solution_bytes,
            card_bytes,
            "a" * 40,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_two_phase_record_reloads_and_replays_exact_response(self):
        store = controller_service.AuditStore(self.path)
        response = execution_response(
            self.request,
            artifact_commit={
                "git_revision": "1" * 40,
                "solution_sha256": self.admission[0],
                "card_sha256": self.admission[1],
            },
        )
        store.begin(self.request, self.admission)
        self.assertEqual(store.lookup(self.request), ("pending", None))
        store.complete(self.request, response)
        self.assertEqual(store.lookup(self.request), ("completed", response))
        reloaded = controller_service.AuditStore(self.path)
        self.assertEqual(reloaded.lookup(self.request), ("completed", response))
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual([row["phase"] for row in rows], ["pending", "completed"])

    def test_success_response_must_match_its_durable_admission(self):
        store = controller_service.AuditStore(self.path)
        store.begin(self.request, self.admission)
        with self.assertRaisesRegex(
            controller_service.ServiceError, "does not match its durable admission"
        ):
            store.complete(self.request, execution_response(self.request))
        self.assertEqual(store.lookup(self.request), ("pending", None))

    def test_wal_and_workspace_caps_cover_the_policy_maximum(self):
        expected = iterate.policy.ITERATION_CAP * (
            iterate.policy.MAX_CONCLUSIVE_REVIEWS_PER_STAGE
            + iterate.policy.MAX_FAILED_REVIEWS_PER_STAGE
        )
        self.assertEqual(controller_service.MAX_AUDIT_REQUESTS, expected)
        self.assertGreaterEqual(
            researcher_shell.WORKSPACE_FILE_RULES["solutions"][1], expected
        )
        self.assertGreaterEqual(
            researcher_shell.WORKSPACE_FILE_RULES["attempts"][1], expected
        )
        self.assertEqual(
            researcher_shell.MAX_PORTFOLIO_VIEW_BYTES,
            iterate.policy.MAX_PORTFOLIO_VIEW_BYTES,
        )

    def test_wal_refuses_an_admission_hash_that_does_not_match_captured_bytes(self):
        bad = list(self.admission)
        bad[0] = "0" * 64
        store = controller_service.AuditStore(self.path)
        with self.assertRaisesRegex(controller_service.ServiceError, "do not match"):
            store.begin(self.request, tuple(bad))
        self.assertFalse(self.path.exists())

    def test_wal_capacity_refusal_is_clean_and_never_executes(self):
        class Authority:
            def __init__(self, admission):
                self.admission = admission
                self.pre_admit_calls = 0
                self.execute_calls = 0

            def pre_admit(self, _request):
                self.pre_admit_calls += 1
                return self.admission

            def execute(self, _request, *, admission=None):
                self.execute_calls += 1
                raise AssertionError("capacity refusal must not execute")

        authority = Authority(self.admission)
        store = controller_service.AuditStore(self.path)
        with mock.patch.object(
            store,
            "_current_size_for_admission",
            return_value=controller_service.MAX_AUDIT_BYTES,
        ):
            response = controller_service.dispatch_request(
                authority, store, threading.Lock(), self.request
            )

        self.assertFalse(response["ok"])
        self.assertFalse(response["recovery_required"])
        self.assertIn("capacity exhausted before admission", response["error"])
        self.assertEqual(authority.pre_admit_calls, 1)
        self.assertEqual(authority.execute_calls, 0)
        self.assertFalse(store.failed)
        self.assertFalse(self.path.exists())

    def test_physical_completion_reservation_keeps_logical_eof_unchanged(self):
        store = controller_service.AuditStore(self.path)
        store._reserve_physical_capacity(64 * 1024)
        metadata = self.path.stat()
        self.assertEqual(metadata.st_size, 0)
        self.assertGreater(metadata.st_blocks, 0)

    def test_physical_reservation_failure_is_clean_and_never_executes(self):
        class Authority:
            calls = 0

            def pre_admit(inner_self, _request):
                return self.admission

            def execute(inner_self, _request, *, admission=None):
                inner_self.calls += 1
                raise AssertionError("physical capacity refusal must not execute")

        authority = Authority()
        store = controller_service.AuditStore(self.path)
        with mock.patch.object(
            controller_service,
            "_fallocate_keep_size",
            side_effect=controller_service.AuditCapacityError("simulated ENOSPC"),
        ):
            response = controller_service.dispatch_request(
                authority, store, threading.Lock(), self.request
            )
        self.assertFalse(response["ok"])
        self.assertFalse(response["recovery_required"])
        self.assertIn("capacity exhausted before admission", response["error"])
        self.assertEqual(authority.calls, 0)
        self.assertFalse(store.failed)
        self.assertEqual(self.path.stat().st_size, 0)

    def test_wal_count_limit_is_a_clean_pre_append_refusal(self):
        store = controller_service.AuditStore(self.path)
        store.cache = {
            f"{index:032x}": {"response": {}}
            for index in range(controller_service.MAX_AUDIT_REQUESTS)
        }
        with self.assertRaises(controller_service.AuditCapacityError):
            store.begin(self.request, self.admission)
        self.assertFalse(store.failed)
        self.assertFalse(self.path.exists())

    def test_reserved_completion_bound_covers_worst_case_json_escaping(self):
        store = controller_service.AuditStore(self.path)
        response = execution_response(
            self.request,
            stdout="\x00" * controller_service.MAX_RESPONSE_STREAM_BYTES,
            stderr="\x1f" * controller_service.MAX_RESPONSE_STREAM_BYTES,
            artifact_commit={
                "git_revision": "1" * 40,
                "solution_sha256": self.admission[0],
                "card_sha256": self.admission[1],
            },
        )
        store.begin(self.request, self.admission)
        before = self.path.stat().st_size
        store.complete(self.request, response)
        completion_bytes = self.path.stat().st_size - before
        self.assertLessEqual(
            completion_bytes, controller_service.MAX_COMPLETION_ROW_BYTES
        )
        client_payload = control.response_bytes(self.request, response)
        self.assertLessEqual(len(client_payload), control.MAX_RESPONSE_BYTES)

    def test_pending_after_process_death_never_reexecutes(self):
        class SimulatedProcessDeath(BaseException):
            pass

        class Authority:
            def __init__(self):
                self.calls = 0

            def pre_admit(self, _request):
                return self.self_admission

            def execute(self, _request, *, admission=None):
                self.calls += 1
                raise SimulatedProcessDeath()

        authority = Authority()
        authority.self_admission = self.admission
        with self.assertRaises(SimulatedProcessDeath):
            controller_service.dispatch_request(
                authority,
                controller_service.AuditStore(self.path),
                threading.Lock(),
                self.request,
            )
        self.assertEqual(authority.calls, 1)
        reloaded = controller_service.AuditStore(self.path)
        response = controller_service.dispatch_request(
            authority, reloaded, threading.Lock(), self.request
        )
        self.assertFalse(response["ok"])
        self.assertTrue(response["recovery_required"])
        self.assertEqual(authority.calls, 1)

    def test_completion_failure_after_side_effect_is_indeterminate(self):
        class Authority:
            def __init__(self):
                self.calls = 0

            def pre_admit(self, _request):
                return self.self_admission

            def execute(self, supplied, *, admission=None):
                self.calls += 1
                return execution_response(supplied)

        authority = Authority()
        authority.self_admission = self.admission
        store = controller_service.AuditStore(self.path)
        with mock.patch.object(
            store, "complete", side_effect=OSError("simulated fsync failure")
        ):
            response = controller_service.dispatch_request(
                authority, store, threading.Lock(), self.request
            )
        self.assertEqual(authority.calls, 1)
        self.assertFalse(response["ok"])
        self.assertTrue(response["recovery_required"])
        self.assertEqual(store.lookup(self.request), ("pending", None))

    def test_request_id_collision_and_corrupt_audit_fail_closed(self):
        store = controller_service.AuditStore(self.path)
        store.begin(self.request, self.admission)
        changed = {**self.request, "solution": "Project/solutions/s001_other.py"}
        with self.assertRaisesRegex(controller_service.ServiceError, "different bytes"):
            store.lookup(changed)

        payload = self.path.read_bytes()
        self.path.write_bytes(payload[:-2])
        with self.assertRaisesRegex(controller_service.ServiceError, "torn"):
            controller_service.AuditStore(self.path)

    def test_torn_completion_repair_preserves_the_valid_pending_prefix(self):
        store = controller_service.AuditStore(self.path)
        store.begin(self.request, self.admission)
        valid_prefix_size = self.path.stat().st_size
        with self.path.open("ab") as stream:
            stream.write(b'{"format":"track2-controller-wal-v2","phase":"completed"')
            stream.flush()
            os.fsync(stream.fileno())

        result = controller_service.repair_torn_audit_tail(self.path)

        self.assertEqual(result["repaired_bytes"], valid_prefix_size)
        self.assertGreater(result["removed_bytes"], 0)
        self.assertEqual(
            controller_service.AuditStore(self.path).lookup(self.request),
            ("pending", None),
        )

    def test_torn_repair_never_changes_newline_complete_corruption(self):
        self.path.write_bytes(b'{"malformed":true}\n')
        self.path.chmod(0o600)
        before = self.path.read_bytes()
        with self.assertRaises(controller_service.ServiceError):
            controller_service.repair_torn_audit_tail(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_one_pending_request_globally_latches_different_run_ids(self):
        store = controller_service.AuditStore(self.path)
        store.begin(self.request, self.admission)
        second = control.build_request(
            "run",
            solution="Project/solutions/s001_second.py",
            card="Project/research/attempts/i001_second.json",
            request_id="8" * 32,
        )

        class Authority:
            calls = 0

            def pre_admit(self, _request):
                self.calls += 1
                return self.admission

        authority = Authority()
        authority.admission = self.admission
        response = controller_service.dispatch_request(
            authority, store, threading.Lock(), second
        )
        self.assertFalse(response["ok"])
        self.assertTrue(response["recovery_required"])
        self.assertEqual(authority.calls, 0)
        self.assertEqual(store.lookup(second), ("absent", None))

    def test_read_only_log_does_not_enter_consuming_wal(self):
        request = control.build_request("log", request_id="8" * 32)

        class Authority:
            def execute(self, supplied):
                return execution_response(supplied)

        response = controller_service.dispatch_request(
            Authority(), controller_service.AuditStore(self.path),
            threading.Lock(), request,
        )
        self.assertTrue(response["ok"])
        self.assertFalse(self.path.exists())

    def test_preexisting_fifo_wal_is_opened_nonblocking_and_refused(self):
        os.mkfifo(self.path, 0o600)
        real_open = os.open

        def checked_open(path, flags, *args, **kwargs):
            if Path(path) == self.path:
                self.assertTrue(flags & os.O_NONBLOCK)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            controller_service.os, "open", side_effect=checked_open
        ):
            with self.assertRaisesRegex(
                controller_service.ServiceError, "regular file"
            ):
                controller_service.AuditStore(self.path)

    def test_fifo_replacement_cannot_block_wal_size_admission(self):
        store = controller_service.AuditStore(self.path)
        os.mkfifo(self.path, 0o600)
        real_open = os.open

        def checked_open(path, flags, *args, **kwargs):
            if Path(path) == self.path:
                self.assertTrue(flags & os.O_NONBLOCK)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            controller_service.os, "open", side_effect=checked_open
        ):
            with self.assertRaisesRegex(
                controller_service.ServiceError, "regular file"
            ):
                store._current_size_for_admission()

    def test_fifo_replacement_cannot_block_wal_append(self):
        store = controller_service.AuditStore(self.path)
        os.mkfifo(self.path, 0o600)
        real_open = os.open

        def checked_open(path, flags, *args, **kwargs):
            if Path(path) == self.path:
                self.assertTrue(flags & os.O_NONBLOCK)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            controller_service.os, "open", side_effect=checked_open
        ):
            with self.assertRaisesRegex(
                controller_service.ServiceError, "cannot be opened safely"
            ):
                store._append_row({"hostile": "fifo"})

    def test_fifo_replacement_cannot_block_offline_torn_tail_repair(self):
        os.mkfifo(self.path, 0o600)
        real_open = os.open

        def checked_open(path, flags, *args, **kwargs):
            if Path(path) == self.path:
                self.assertTrue(flags & os.O_NONBLOCK)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            controller_service.os, "open", side_effect=checked_open
        ):
            with self.assertRaisesRegex(
                controller_service.ServiceError, "regular file"
            ):
                controller_service.repair_torn_audit_tail(self.path)


class PendingReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticGitRepo()
        (self.fixture.root / "Project" / "results").mkdir()
        self.runtime = self.fixture.base / "runtime"
        self.runtime.mkdir(mode=0o700)
        self.audit_path = self.runtime / "controller.audit.jsonl"
        (
            self.solution,
            self.card,
            self.solution_bytes,
            self.card_bytes,
        ) = self.fixture.artifacts("pending")
        self.parent_revision = self.fixture._git("rev-parse", "HEAD").strip()
        self.request = control.build_request(
            "run",
            solution=self.solution,
            card=self.card,
            request_id="6" * 32,
        )
        self.admission = (
            hashlib.sha256(self.solution_bytes).hexdigest(),
            hashlib.sha256(self.card_bytes).hexdigest(),
            self.solution_bytes,
            self.card_bytes,
            self.parent_revision,
        )
        controller_service.AuditStore(self.audit_path).begin(
            self.request, self.admission
        )
        self.workspace = type("Workspace", (), {})()
        self.workspace.workspace_id = "c" * 32
        self.workspace.run_id = "a" * 32
        self.workspace.repository_head = self.parent_revision
        self.workspace.portfolio = {"synthetic": True}
        self.bank = mock.Mock(
            snapshot_sha256="e" * 64,
            descriptor={"synthetic": True},
            known_claims=("claim",),
            known_topics=("topic",),
        )
        self.original_run_process = controller_service._run_process

    def tearDown(self):
        self.fixture.close()

    def _run_process(self, command, **kwargs):
        if command[-1] == "_admission-state":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    "official_run_started": True,
                    "run_id": self.workspace.run_id,
                    "run_start_git_revision": self.workspace.repository_head,
                    "open_attempt": False,
                    "portfolio": self.workspace.portfolio,
                    "research_bank": {
                        "snapshot_sha256": self.bank.snapshot_sha256,
                        "descriptor": self.bank.descriptor,
                        "claim_count": len(self.bank.known_claims),
                        "known_topics": list(self.bank.known_topics),
                    },
                }),
                stderr="",
            )
        return self.original_run_process(command, **kwargs)

    def _reconcile(self):
        protected = self.fixture.protected()
        with (
            mock.patch.object(
                controller_service, "ProtectedState", return_value=protected
            ),
            mock.patch.object(
                controller_service, "frozen_bank_paths", return_value=()
            ),
            mock.patch.object(
                controller_service, "_run_process", side_effect=self._run_process
            ),
            mock.patch.object(
                controller_service.outer_research_bank,
                "load",
                return_value=self.bank,
            ),
        ):
            return controller_service.reconcile_pending_request(
                self.fixture.root,
                self.audit_path,
                self.request["request_id"],
                self.fixture.env,
                self.workspace,
            )

    def test_no_commit_reconciliation_removes_exact_uncommitted_pair(self):
        response = self._reconcile()
        self.assertFalse(response["ok"])
        self.assertFalse(response["recovery_required"])
        self.assertEqual(
            self.fixture._git("rev-parse", "HEAD").strip(), self.parent_revision
        )
        self.assertFalse((self.fixture.root / self.solution).exists())
        self.assertFalse((self.fixture.root / self.card).exists())
        status, replay = controller_service.AuditStore(self.audit_path).lookup(
            self.request
        )
        self.assertEqual(status, "completed")
        self.assertEqual(replay, response)

    def test_reconciliation_removes_partial_reserved_staging_file(self):
        solution_path = self.fixture.root / self.solution
        card_path = self.fixture.root / self.card
        solution_path.unlink()
        card_path.unlink()
        temporary_leaf = controller_service._materialization_temp_leaf(
            self.solution, hashlib.sha256(self.solution_bytes).hexdigest()
        )
        temporary_path = solution_path.parent / temporary_leaf
        temporary_path.write_bytes(self.solution_bytes[:3])
        temporary_path.chmod(0o600)

        response = self._reconcile()

        self.assertFalse(response["ok"])
        self.assertFalse(temporary_path.exists())
        self.assertFalse(solution_path.exists())
        self.assertFalse(card_path.exists())
        self.assertEqual(
            controller_service.AuditStore(self.audit_path).lookup(self.request)[0],
            "completed",
        )

    def test_reconciliation_refuses_a_different_workspace_id(self):
        self.workspace.workspace_id = "d" * 32
        with self.assertRaisesRegex(
            controller_service.ServiceError, "different researcher workspace"
        ):
            self._reconcile()
        self.assertEqual(
            controller_service.AuditStore(self.audit_path).lookup(self.request),
            ("pending", None),
        )

    def test_committed_bytes_must_match_the_durable_admission_hashes(self):
        (self.fixture.root / self.solution).unlink()
        (self.fixture.root / self.card).unlink()
        changed_solution = self.solution_bytes + b"# changed\n"
        changed_card = self.card_bytes + b"\n"
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        committer.commit_pair(
            self.solution,
            self.card,
            captured_solution_bytes=changed_solution,
            captured_card_bytes=changed_card,
            materialize_new_worktree_files=True,
            expected_parent_revision=self.parent_revision,
        )
        with self.assertRaisesRegex(
            controller_service.ServiceError, "durable admission binding"
        ):
            self._reconcile()
        self.assertEqual(
            controller_service.AuditStore(self.audit_path).lookup(self.request),
            ("pending", None),
        )

    def test_pending_commit_must_be_the_direct_child_of_admitted_parent(self):
        (self.fixture.root / self.solution).unlink()
        (self.fixture.root / self.card).unlink()
        committer = controller_service.ArtifactCommitter(
            self.fixture.root, self.fixture.protected(), self.fixture.env
        )
        committer.commit_pair(
            self.solution,
            self.card,
            captured_solution_bytes=self.solution_bytes,
            captured_card_bytes=self.card_bytes,
            materialize_new_worktree_files=True,
            expected_parent_revision=self.parent_revision,
        )
        later_solution, later_card, _, _ = self.fixture.artifacts("later")
        committer.commit_pair(later_solution, later_card)

        with self.assertRaisesRegex(
            controller_service.ServiceError, "direct child of durable intent"
        ):
            self._reconcile()
        self.assertEqual(
            controller_service.AuditStore(self.audit_path).lookup(self.request),
            ("pending", None),
        )


@unittest.skipUnless(
    os.environ.get("TRACK2_TEST_UNIX_SOCKET") == "1",
    "set TRACK2_TEST_UNIX_SOCKET=1 where AF_UNIX bind is permitted",
)
class RpcIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticGitRepo()
        (self.fixture.root / "Project" / "results").mkdir()
        self.runtime = self.fixture.base / "runtime"
        self.runtime.mkdir(mode=0o700)
        self.authority_state_dir = self.fixture.base / "authority-state"
        self.authority_state_dir.mkdir(mode=0o700)
        self.socket_path = self.runtime / "controller.sock"
        self.audit_path = self.runtime / "controller.audit.jsonl"
        execution_authority = controller_service.ControllerAuthority(
            self.fixture.root,
            allow_repo_artifacts_for_tests=True,
            protected_paths=("Project/harness/iterate.py",),
            env=self.fixture.env,
        )
        binding_workspace_root = self.fixture.base / "binding-workspace"
        binding_workspace_root.mkdir(mode=0o700)
        binding_workspace = type("BindingWorkspace", (), {})()
        binding_workspace.root = binding_workspace_root
        binding_workspace.workspace_id = "c" * 32
        binding_workspace.binding = "e" * 64
        binding_workspace.run_id = "a" * 32
        self.authority = type("BoundTestAuthority", (), {})()
        self.authority.root = execution_authority.root
        self.authority.workspace = binding_workspace
        self.authority.transaction = execution_authority.transaction
        self.authority.pre_admit = execution_authority.pre_admit
        self.authority.execute = execution_authority.execute
        self.server = controller_service.ControllerServer(
            self.socket_path,
            self.authority,
            self.audit_path,
            authority_state_dir=self.authority_state_dir,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.fixture.close()

    def test_log_and_idempotent_run_end_to_end(self):
        log_request = control.build_request("log", request_id=uuid.uuid4().hex)
        log_response = control.rpc(self.socket_path, log_request, timeout_seconds=10)
        self.assertTrue(log_response["ok"], log_response)
        self.assertIn('"state": "ACTIVE"', log_response["stdout"])

        solution, card, _, _ = self.fixture.artifacts("rpc")
        run_request = control.build_request(
            "run", solution=solution, card=card, request_id=uuid.uuid4().hex
        )
        first = control.rpc(self.socket_path, run_request, timeout_seconds=30)
        second = control.rpc(self.socket_path, run_request, timeout_seconds=30)
        self.assertTrue(first["ok"], first)
        self.assertEqual(second, first)
        self.assertIn('"run"', first["stdout"])
        self.assertEqual(int(self.fixture._git("rev-list", "--count", "HEAD")), 2)
        self.assertEqual(len(self.audit_path.read_text().splitlines()), 2)

    def test_second_runtime_for_same_repository_is_refused(self):
        second_runtime = self.fixture.base / "runtime-second"
        second_runtime.mkdir(mode=0o700)
        second_socket = second_runtime / "controller.sock"
        second_audit = second_runtime / "controller.audit.jsonl"
        with self.assertRaisesRegex(
            controller_service.ServiceError, "owns this repository"
        ):
            controller_service.ControllerServer(
                second_socket,
                self.authority,
                second_audit,
                authority_state_dir=self.authority_state_dir,
            )
        self.assertFalse(second_socket.exists())
        self.assertFalse(second_audit.exists())


class SocketAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="track2-private-socket-test-", dir="/tmp"
        )
        self.base = Path(self.temp.name)
        self.base.chmod(0o700)
        self.runtime = self.base / "runtime"
        self.runtime.mkdir(mode=0o700)
        self.socket_path = self.runtime / "controller.sock"
        self.socket_path.touch(mode=0o600)
        self.socket_path.chmod(0o600)
        self.original_lstat = Path.lstat

    def tearDown(self):
        self.runtime.chmod(0o700)
        self.temp.cleanup()

    def socket_metadata(self, *, owner=None):
        original_lstat = self.original_lstat
        target = self.socket_path

        def virtual_lstat(path):
            metadata = original_lstat(path)
            if Path(path) == target:
                fields = list(metadata)
                fields[0] = stat.S_IFSOCK | stat.S_IMODE(metadata.st_mode)
                if owner is not None:
                    fields[4] = owner
                return os.stat_result(fields)
            return metadata

        return mock.patch.object(Path, "lstat", new=virtual_lstat)

    def assert_rejected_by_both(self):
        with self.assertRaises(control.ClientError):
            control._socket_identity(self.socket_path)
        with self.assertRaises(researcher_shell.ShellError):
            researcher_shell._socket_file(self.socket_path)

    def test_private_owner_only_socket_is_accepted_by_client_and_shell(self):
        with self.socket_metadata():
            client = control._socket_identity(self.socket_path)
            mounted = researcher_shell._socket_file(self.socket_path)
        self.assertEqual(client.path, self.socket_path)
        self.assertEqual(mounted, self.socket_path)

    def test_group_accessible_socket_is_rejected(self):
        self.socket_path.chmod(0o660)
        with self.socket_metadata():
            self.assert_rejected_by_both()

    def test_nonprivate_parent_is_rejected(self):
        self.runtime.chmod(0o755)
        with self.socket_metadata():
            self.assert_rejected_by_both()

    def test_symlinked_parent_is_rejected(self):
        alias = self.base / "runtime-alias"
        alias.symlink_to(self.runtime, target_is_directory=True)
        aliased_socket = alias / self.socket_path.name
        with self.socket_metadata():
            with self.assertRaises(control.ClientError):
                control._socket_identity(aliased_socket)
            with self.assertRaises(researcher_shell.ShellError):
                researcher_shell._socket_file(aliased_socket)

    def test_spoofed_socket_owner_is_rejected(self):
        with self.socket_metadata(owner=os.geteuid() + 1):
            self.assert_rejected_by_both()


class ControllerLogSurfaceTests(unittest.TestCase):
    def test_log_lists_chronological_official_ids_without_private_evidence(self):
        attempts = [
            {
                "entry_id": "outcome-early", "iteration": 1,
                "valid_metrics": {"primary": 0.1}, "card": {},
            },
            {
                "entry_id": "outcome-late", "iteration": 2,
                "valid_metrics": {"primary": 0.2}, "card": {},
            },
        ]
        journal = [{
            "entry_id": "private-row",
            "labels": [1, 0, 1],
            "reviewer_private_reasoning": "must never cross the boundary",
        }]
        output = io.StringIO()
        with tempfile.TemporaryDirectory(
            prefix="track2-log-surface-test-", dir="/tmp"
        ) as tmp:
            with (
                mock.patch.object(
                    iterate, "authority_state_dir", return_value=Path(tmp) / "missing"
                ),
                mock.patch.object(iterate, "read_journal", return_value=journal),
                mock.patch.object(iterate, "validate_ledger"),
                mock.patch.object(
                    iterate.policy, "first_run_start",
                    return_value={"run_id": "run-1"},
                ),
                mock.patch.object(
                    iterate.policy, "official_iterations", return_value=attempts
                ),
                mock.patch.object(
                    iterate.policy, "first_terminal", return_value=None
                ),
                mock.patch.object(
                    iterate.policy, "best_eligible", return_value=attempts[-1]
                ),
                mock.patch.object(
                    iterate.policy, "triggered_reasons", return_value=[]
                ),
                mock.patch.object(
                    iterate.policy, "primary_score", return_value=0.2
                ),
                mock.patch.object(
                    iterate.policy, "elapsed_seconds", return_value=12.0
                ),
                mock.patch.object(
                    iterate, "_portfolio_from_start",
                    return_value={"families": [], "opening_order": []},
                ),
                mock.patch.object(sys, "stdout", new=output),
            ):
                self.assertEqual(iterate.cmd_log(None), 0)
        raw = output.getvalue()
        logged = json.loads(raw)
        self.assertEqual(
            logged["prior_outcomes_considered"],
            ["outcome-early", "outcome-late"],
        )
        self.assertNotIn("labels", raw)
        self.assertNotIn("reviewer_private_reasoning", raw)
        self.assertNotIn("must never cross", raw)


class ResearcherBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(
        Path("/home/admin/.local/bin/claude").exists(),
        "owner-pinned Claude runtime is not installed on this host",
    )
    def test_installed_claude_matches_frozen_runtime_attestation(self):
        resolved = researcher_shell._attested_claude(
            Path("/home/admin/.local/bin/claude"), ROOT
        )
        self.assertTrue(resolved.is_file())

    def test_mount_graph_exposes_only_intended_repository_paths(self):
        with tempfile.TemporaryDirectory(prefix="track2-socket-test-", dir="/tmp") as tmp:
            base = Path(tmp)
            runtime = base / "runtime"
            runtime.mkdir(mode=0o700)
            socket_path = runtime / "controller.sock"
            workspace = make_researcher_workspace(base)
            with mock.patch.object(
                researcher_shell, "_socket_file", return_value=socket_path
            ):
                command = researcher_shell.build_command(
                    root=ROOT,
                    socket_path=socket_path,
                    agent_executable=Path("/usr/bin/true"),
                    agent_args=[],
                    workspace_root=workspace,
                )
        manifest = researcher_shell.boundary_manifest(command)
        self.assertEqual(manifest["writable_repo_paths"], [
            "/workspace/Project/solutions",
            "/workspace/Project/research/attempts",
            "/workspace/Project/memory",
            "/workspace/Project/research/scratch",
        ])
        self.assertTrue(manifest["research_bank_readonly"])
        self.assertTrue(manifest["researcher_network_shared"])
        self.assertFalse(manifest["persistent_agent_home"])
        self.assertFalse(manifest["claude_hard_tool_surface"])
        for key in (
            "git_history_mounted",
            "raw_or_sanitized_data_mounted",
            "results_mounted",
            "harness_mounted",
            "manifest_mounted",
        ):
            self.assertFalse(manifest[key])
        flattened = "\n".join(command)
        self.assertNotIn("/.git", flattened)
        self.assertNotIn("/Project/results", flattened)
        self.assertNotIn("/Project/manifest.json", flattened)
        self.assertNotIn("KuaiRand-Pure", flattened)
        self.assertNotIn(
            str(ROOT / "kuairand-starter-kit" / "README.md"), flattened
        )
        self.assertEqual(
            command.count("/control/control.py"), 1,
            "the control client must have exactly one read-only bind destination",
        )
        self.assertIn(
            ["--perms", "0700", "--dir", "/run/track2"],
            [command[index:index + 4] for index in range(len(command) - 3)],
        )
        self.assertIn("TRACK2_CONTROLLER_CLIENT_STATE", command)
        self.assertIn(researcher_shell.INSIDE_CLIENT_STATE, command)
        self.assertIn("TRACK2_RESEARCHER_WORKSPACE", command)
        self.assertIn(researcher_shell.WORKSPACE_ID_ENV, command)
        self.assertIn("/workspace/Project/research/portfolio.json", command)

    def test_official_claude_profile_removes_shell_and_web_tools(self):
        with tempfile.TemporaryDirectory(
            prefix="track2-claude-boundary-test-", dir="/tmp"
        ) as tmp:
            base = Path(tmp)
            runtime = base / "runtime"
            runtime.mkdir(mode=0o700)
            socket_path = runtime / "controller.sock"
            agent_home = base / "agent-home"
            agent_home.mkdir(mode=0o700)
            workspace = make_researcher_workspace(base)
            claude = base / "claude"
            claude.write_text("#!/bin/sh\nexit 0\n")
            claude.chmod(0o700)
            with (
                mock.patch.object(
                    researcher_shell, "_socket_file", return_value=socket_path
                ),
                mock.patch.object(
                    researcher_shell, "_attested_claude", return_value=claude
                ),
                mock.patch.object(
                    researcher_shell,
                    "_frozen_bank_files",
                    return_value=(
                        "Project/research/bank/catalog.template.json",
                        "Project/research/bank/notes/README.md",
                    ),
                ),
            ):
                command = researcher_shell.build_command(
                    root=ROOT,
                    socket_path=socket_path,
                    agent_executable=claude,
                    agent_args=[],
                    workspace_root=workspace,
                    agent_home=agent_home,
                    restrict_claude=True,
                )
                for unsafe in (
                    ["--tools", "Bash"], ["--resume"], ["mcp", "--help"]
                ):
                    with self.assertRaisesRegex(
                        researcher_shell.ShellError, "accepts no Claude arguments"
                    ):
                        researcher_shell.build_command(
                            root=ROOT,
                            socket_path=socket_path,
                            agent_executable=claude,
                            agent_args=unsafe,
                            workspace_root=workspace,
                            agent_home=agent_home,
                            restrict_claude=True,
                        )
        manifest = researcher_shell.boundary_manifest(command)
        self.assertTrue(manifest["claude_hard_tool_surface"])
        self.assertIn(researcher_shell.INSIDE_MCP_CONFIG, command)
        self.assertIn(researcher_shell.INSIDE_MCP_SERVER, command)
        self.assertIn("--restricted", command)
        self.assertIn("--disable-slash-commands", command)
        self.assertIn("dontAsk", command)
        self.assertIn("Bash,WebFetch,WebSearch,NotebookEdit", command)
        self.assertIn("DISABLE_UPDATES", command)
        self.assertIn("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", command)
        agent_bind = command.index(str(claude))
        root_remount = next(
            index for index in range(len(command) - 1)
            if command[index:index + 2] == ["--remount-ro", "/"]
        )
        self.assertLess(agent_bind, root_remount)
        config = json.loads(
            (ROOT / researcher_shell.MCP_CONFIG).read_text(encoding="utf-8")
        )
        self.assertNotIn("env", config["mcpServers"]["track2_controller"])


class ControllerMcpTests(unittest.TestCase):
    class FakeControl:
        SOCKET_ENV = "TRACK2_TEST_FAKE_SOCKET"

        def __init__(self):
            self.calls = []

        def build_request(self, command, **kwargs):
            request = {
                "protocol_version": control.PROTOCOL_VERSION,
                "request_id": "a" * 32,
                "command": command,
            }
            request.update({key: value for key, value in kwargs.items() if value is not None})
            return request

        def issue_request(self, socket_path, request):
            self.calls.append(("issue", socket_path, request))
            return execution_response(request)

        def retry_persisted(self, socket_path, *, recover_completed):
            self.calls.append(("replay", socket_path, recover_completed))
            request = {
                "protocol_version": control.PROTOCOL_VERSION,
                "request_id": "b" * 32,
                "command": "log",
            }
            return execution_response(request)

    def test_exact_tools_and_protocol_handshake(self):
        self.assertEqual(
            [tool["name"] for tool in controller_mcp.tool_definitions()],
            ["hash_solution", "log", "run", "retry", "recover"],
        )
        initialized = controller_mcp.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        })
        self.assertEqual(
            initialized["result"]["protocolVersion"], "2025-06-18"
        )
        listed = controller_mcp.handle_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
        })
        self.assertEqual(len(listed["result"]["tools"]), 5)
        self.assertIsNone(controller_mcp.handle_message({
            "jsonrpc": "2.0", "method": "notifications/initialized"
        }))

    def test_tool_calls_use_only_narrow_client_operations(self):
        fake = self.FakeControl()
        with mock.patch.dict(os.environ, {fake.SOCKET_ENV: "/run/controller.sock"}):
            result = controller_mcp.call_tool("run", {
                "solution": "Project/solutions/s001_mcp.py",
                "card": "Project/research/attempts/i001_mcp.json",
            }, fake)
            controller_mcp.call_tool("retry", {}, fake)
            controller_mcp.call_tool("recover", {}, fake)
        self.assertFalse(result["isError"])
        self.assertEqual(fake.calls[0][0], "issue")
        self.assertEqual(fake.calls[1][-1], False)
        self.assertEqual(fake.calls[2][-1], True)
        with self.assertRaises(controller_mcp.McpError):
            controller_mcp.call_tool("run", {
                "solution": "Project/solutions/s001_mcp.py",
                "card": "Project/research/attempts/i001_mcp.json",
                "override": True,
            }, fake)

    def test_worst_valid_controller_response_fits_mcp_message_limit(self):
        fake = self.FakeControl()

        def worst_response(_socket_path, request):
            return execution_response(
                request,
                stdout="\x00" * controller_service.MAX_RESPONSE_STREAM_BYTES,
                stderr="\x1f" * controller_service.MAX_RESPONSE_STREAM_BYTES,
            )

        fake.issue_request = worst_response
        with mock.patch.dict(os.environ, {fake.SOCKET_ENV: "/run/controller.sock"}):
            result = controller_mcp.call_tool("log", {}, fake)
        wire = controller_mcp._canonical({
            "jsonrpc": "2.0", "id": 1, "result": result,
        }) + b"\n"
        self.assertGreater(len(wire), 512 * 1024)
        self.assertLessEqual(len(wire), controller_mcp.MAX_MESSAGE_BYTES)

    def test_dedicated_agent_home_must_be_private_and_outside_repo(self):
        with tempfile.TemporaryDirectory(prefix="track2-socket-test-", dir="/tmp") as tmp:
            base = Path(tmp)
            runtime = base / "runtime"
            runtime.mkdir(mode=0o700)
            socket_path = runtime / "controller.sock"
            agent_home = base / "agent-home"
            agent_home.mkdir(mode=0o700)
            workspace = make_researcher_workspace(base)
            with mock.patch.object(
                researcher_shell, "_socket_file", return_value=socket_path
            ):
                command = researcher_shell.build_command(
                    root=ROOT,
                    socket_path=socket_path,
                    agent_executable=Path("/usr/bin/true"),
                    agent_args=[],
                    workspace_root=workspace,
                    agent_home=agent_home,
                )
                self.assertTrue(
                    researcher_shell.boundary_manifest(command)[
                        "persistent_agent_home"
                    ]
                )
                agent_home.chmod(0o755)
                with self.assertRaisesRegex(researcher_shell.ShellError, "0700"):
                    researcher_shell.build_command(
                        root=ROOT,
                        socket_path=socket_path,
                        agent_executable=Path("/usr/bin/true"),
                        agent_args=[],
                        workspace_root=workspace,
                        agent_home=agent_home,
                    )

    def test_agent_home_cannot_overlap_socket_runtime(self):
        with tempfile.TemporaryDirectory(
            prefix="track2-socket-test-", dir="/tmp"
        ) as tmp:
            base = Path(tmp)
            runtime = base / "runtime"
            runtime.mkdir(mode=0o700)
            socket_path = runtime / "controller.sock"
            workspace = make_researcher_workspace(base)
            with mock.patch.object(
                researcher_shell, "_socket_file", return_value=socket_path
            ):
                with self.assertRaisesRegex(researcher_shell.ShellError, "socket"):
                    researcher_shell.build_command(
                        root=ROOT,
                        socket_path=socket_path,
                        agent_executable=Path("/usr/bin/true"),
                        agent_args=[],
                        workspace_root=workspace,
                        agent_home=runtime,
                    )

    def test_socket_runtime_cannot_overlap_repository(self):
        socket_path = ROOT / "Project" / "research" / "controller.sock"
        with tempfile.TemporaryDirectory(
            prefix="track2-workspace-test-", dir="/tmp"
        ) as tmp:
            workspace = make_researcher_workspace(Path(tmp))
            with mock.patch.object(
                researcher_shell, "_socket_file", return_value=socket_path
            ):
                with self.assertRaisesRegex(researcher_shell.ShellError, "repository"):
                    researcher_shell.build_command(
                        root=ROOT,
                        socket_path=socket_path,
                        agent_executable=Path("/usr/bin/true"),
                        agent_args=[],
                        workspace_root=workspace,
                    )

    def test_nonexecutable_researcher_binary_is_rejected(self):
        with tempfile.TemporaryDirectory(
            prefix="track2-socket-test-", dir="/tmp"
        ) as tmp:
            base = Path(tmp)
            runtime = base / "runtime"
            runtime.mkdir(mode=0o700)
            socket_path = runtime / "controller.sock"
            workspace = make_researcher_workspace(base)
            candidate = base / "researcher"
            candidate.write_text("#!/bin/sh\nexit 0\n")
            candidate.chmod(0o600)
            with mock.patch.object(
                researcher_shell, "_socket_file", return_value=socket_path
            ):
                with self.assertRaisesRegex(researcher_shell.ShellError, "executable"):
                    researcher_shell.build_command(
                        root=ROOT,
                        socket_path=socket_path,
                        agent_executable=candidate,
                        agent_args=[],
                        workspace_root=workspace,
                    )

    def test_normal_home_or_repository_path_cannot_be_mounted(self):
        with tempfile.TemporaryDirectory(prefix="track2-socket-test-", dir="/tmp") as tmp:
            base = Path(tmp)
            runtime = base / "runtime"
            runtime.mkdir(mode=0o700)
            socket_path = runtime / "controller.sock"
            workspace = make_researcher_workspace(base)
            with mock.patch.object(
                researcher_shell, "_socket_file", return_value=socket_path
            ):
                for unsafe in (Path.home(), ROOT / "Project" / "memory"):
                    with self.assertRaises(researcher_shell.ShellError):
                        researcher_shell.build_command(
                            root=ROOT,
                            socket_path=socket_path,
                            agent_executable=Path("/usr/bin/true"),
                            agent_args=[],
                            workspace_root=workspace,
                            agent_home=unsafe,
                        )


if __name__ == "__main__":
    unittest.main()
