from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from harness.agent import ClaudeAgentRunner, CodexAgentRunner


def _probe_boundary(command: list[str], executable: str) -> None:
    executable_index = max(
        index for index, value in enumerate(command) if value == executable
    )
    probe = subprocess.run(
        [
            *command[:executable_index],
            "/usr/bin/python3",
            "-c",
            """
import json
from pathlib import Path
result = {}
try:
    Path('/evidence/STATE.json').write_text('tampered')
    result['evidence_read_only'] = False
except OSError:
    result['evidence_read_only'] = True
Path('/workspace/write-check.txt').write_text('ok')
result['workspace_writable'] = Path('/workspace/write-check.txt').read_text() == 'ok'
result['raw_data_blocked'] = not Path('/home/yy/Desktop/Repos/Tiktok_TechJam_2026_Track2/datasets').exists()
print(json.dumps(result))
""",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    expected = {
        "evidence_read_only": True,
        "workspace_writable": True,
        "raw_data_blocked": True,
    }
    try:
        probe_value = json.loads(probe.stdout)
    except json.JSONDecodeError:
        probe_value = None
    if probe.returncode != 0 or probe_value != expected:
        print(probe.stderr[-2000:])
        raise SystemExit(1)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="techjam-agent-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        evidence = root / "evidence"
        workspace.mkdir()
        evidence.mkdir()
        (evidence / "STATE.json").write_text("trusted")
        claude = ClaudeAgentRunner()
        command = claude._command(
            workspace,
            evidence_dir=evidence,
            model="fable",
            tools="Read",
            output_format="stream-json",
            schema_path=None,
        )
        _probe_boundary(command, "/runtime/claude")
        executable_index = max(
            index for index, value in enumerate(command) if value == "/runtime/claude"
        )
        network_probe = subprocess.run(
            [
                *command[:executable_index],
                "/usr/bin/curl",
                "-IsS",
                "--max-time",
                "10",
                "https://api.anthropic.com/",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if network_probe.returncode != 0:
            print(network_probe.stderr[-2000:])
            raise SystemExit(1)
        smoke_command = [*command[: executable_index + 1], "auth", "status"]
        completed = subprocess.run(
            smoke_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        # Do not print account details; only report whether isolated authentication works.
        print(f"agent_auth_status={completed.returncode}")
        if completed.returncode != 0:
            print(completed.stderr[-2000:])
            raise SystemExit(1)

        codex = CodexAgentRunner()
        codex_command = codex._command(
            workspace,
            evidence_dir=evidence,
            model="gpt-5.6-sol",
        )
        codex_executable = str(codex.codex)
        _probe_boundary(codex_command, codex_executable)
        codex_index = codex_command.index(codex_executable)
        codex_auth = subprocess.run(
            [*codex_command[:codex_index], codex_executable, "login", "status"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        print(f"codex_auth_status={codex_auth.returncode}")
        if codex_auth.returncode != 0:
            print(codex_auth.stderr[-2000:])
            raise SystemExit(1)


if __name__ == "__main__":
    main()
