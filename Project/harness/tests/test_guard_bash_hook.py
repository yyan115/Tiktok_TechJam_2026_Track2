from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HOOK = ROOT / ".claude" / "hooks" / "guard_bash.py"
sys.path.insert(0, str(ROOT / "Project" / "harness"))
import controller_service


def hook(command: str | None = None, *, raw: bytes | None = None):
    if raw is None:
        raw = json.dumps({"tool_input": {"command": command}}).encode("utf-8")
    result = subprocess.run(
        [sys.executable, "-I", str(HOOK)],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        timeout=5,
        check=False,
    )
    if not result.stdout:
        return None, result
    return json.loads(result.stdout), result


def denied(command: str) -> bool:
    payload, result = hook(command)
    if result.returncode != 0 or result.stderr:
        raise AssertionError(
            f"hook failed for {command!r}: {result.returncode=} {result.stderr!r}"
        )
    return (
        payload is not None
        and payload["hookSpecificOutput"]["permissionDecision"] == "deny"
    )


class GuardBashHookTests(unittest.TestCase):
    def test_literal_protected_list_matches_controller_and_every_path_denies(self):
        tree = ast.parse(HOOK.read_bytes())
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "CONTROLLER_PROTECTED_PATHS"
                for target in node.targets
            )
        )
        hook_paths = ast.literal_eval(assignment.value)
        self.assertEqual(hook_paths, controller_service.PROTECTED_PATHS)
        for relative in hook_paths:
            with self.subTest(relative=relative):
                self.assertTrue(denied(f"touch -- {relative}"))

    def test_directory_nodes_ancestors_and_data_are_protected(self):
        commands = (
            "mv Project/harness /tmp/harness-old",
            "rm -r Project/results",
            "rm -f Project/results/JOURNAL.jsonl",
            "truncate -s 0 .git/HEAD",
            "rm -f kuairand-starter-kit/KuaiRand-Pure/data/test.csv",
            "touch kuairand-starter-kit/KuaiRand-Pure/data_sanitized/valid.csv",
            "rm -r Project",
            "cd Project && rm results",
            "cd Project/harness && mv controller_service.py /tmp/controller.py",
            "mv Project/./harness /tmp/harness-old",
            "rm -f Project/harness/../results/JOURNAL.jsonl",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(denied(command))

    def test_malformed_payload_and_shell_text_deny(self):
        malformed = (
            b"not json",
            b"{}",
            b'{"tool_input":null}',
            b'{"tool_input":{"command":7}}',
            b'{"tool_input":{"command":"echo ok"},"tool_input":{"command":"echo bad"}}',
        )
        for raw in malformed:
            with self.subTest(raw=raw):
                payload, result = hook(raw=raw)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(
                    payload["hookSpecificOutput"]["permissionDecision"], "deny"
                )
        self.assertTrue(denied("printf '%s"))

    def test_obvious_alternate_writers_deny(self):
        commands = (
            "printf x > Project/results/JOURNAL.jsonl",
            "printf x 2>>Project/results/JOURNAL.jsonl",
            "dd if=/dev/zero of=Project/manifest.json bs=1 count=1",
            "find Project/harness -delete",
            "sed -i 's/x/y/' Project/harness/policy.py",
            "perl -pi -e 's/x/y/' Project/harness/policy.py",
            "python3 -c \"from pathlib import Path; Path('Project/results/x').write_text('x')\"",
            "sh -c 'mv Project/harness /tmp/h'",
            "echo x | tee Project/results/x",
            "install source Project/harness/iterate.py",
            "tar -xf payload.tar -C Project/results",
            "unzip payload.zip -d Project/harness",
            "echo $(rm Project/results/JOURNAL.jsonl)",
            "printf Project/results/JOURNAL.jsonl | xargs rm",
            "echo 'rm Project/results/JOURNAL.jsonl' | sh",
            "echo \"open('Project/results/x','w')\" | python3 -",
            "python3 - <<'EOF'\nopen('Project/results/x', 'w')\nEOF",
            "curl -o Project/results/download.bin https://example.invalid/x",
            "wget -OProject/harness/download.py https://example.invalid/x",
            "stdbuf -o0 rm Project/results/JOURNAL.jsonl",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(denied(command))

    def test_git_writers_deny_but_readers_remain_available(self):
        for command in (
            "git add Project/solutions/s001.py",
            "git -C . reset --hard HEAD",
            "git update-ref refs/heads/x HEAD",
            "sudo git clean -fd",
        ):
            with self.subTest(command=command):
                self.assertTrue(denied(command))
        for command in (
            "git status --short",
            "git diff -- Project/harness/controller_service.py",
            "git show HEAD:Project/harness/controller_service.py",
        ):
            with self.subTest(command=command):
                self.assertFalse(denied(command))

    def test_recursive_delete_is_deny_biased_and_tmp_exception_is_narrow(self):
        for command in (
            "rm -rf Project/solutions/scratch",
            "/bin/rm --recursive --force -- -outside",
            "sudo rm --recur /home/admin/scratch",
            "rm -rf /tmp/../home/admin/scratch",
            "rm -rf /tmp/*",
        ):
            with self.subTest(command=command):
                self.assertTrue(denied(command))
        self.assertFalse(denied("rm -rf -- /tmp/track2-disposable-hook-test"))

    def test_external_authority_runtime_and_workspace_markers_deny(self):
        home_authority = (
            Path.home() / ".local" / "state" / "tiktok-techjam-2026-track2"
        )
        commands = (
            f"truncate -s 0 {home_authority}/key.json",
            "mv /absolute/private/track2-runtime /tmp/runtime-old",
            "rm -f /absolute/private/runtime/controller.audit.jsonl",
            "touch /absolute/private/work/.track2-workspace.json",
            "chmod 777 /absolute/private/agent/.track2-controller-client",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(denied(command))

    def test_arbitrarily_named_workspace_is_discovered_from_marker(self):
        with tempfile.TemporaryDirectory(prefix="track2-hook-workspace-", dir="/tmp") as temporary:
            workspace = Path(temporary) / "opaque-name"
            workspace.mkdir()
            (workspace / ".track2-workspace.json").write_text("{}\n")
            self.assertTrue(denied(f"touch {workspace}/scratch/new.md"))
            self.assertTrue(denied(f"mv {workspace} /tmp/renamed-workspace"))

    def test_normal_read_and_candidate_authoring_commands_remain_available(self):
        commands = (
            "cat Project/harness/controller_service.py",
            "rg convergence Project/results/JOURNAL.jsonl",
            "python3 Project/harness/controller_service.py --help",
            "touch Project/solutions/s999_candidate.py",
            "printf x > Project/solutions/s999_candidate.py",
            "cp Project/solutions/s001_fm_baseline.py /tmp/baseline-copy.py",
            "cd Project && touch solutions/s999_candidate.py",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(denied(command))


if __name__ == "__main__":
    unittest.main()
