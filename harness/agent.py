from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.spec import ROOT


class AgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentResult:
    status: str
    returncode: int | None
    wall_seconds: float
    stdout: str
    stderr: str
    structured: dict[str, Any] | None = None


class ClaudeAgentRunner:
    """Run a model inside a narrow filesystem view.

    The researcher gets guidance and its own workspace, but not the repository,
    controller ledger, raw benchmark, validation labels, or hidden-test features.
    Bash is deliberately absent, so model executions cannot become uncounted
    scientific experiments. Candidate execution is owned by the controller.
    """

    def __init__(self, guidance_dir: Path = ROOT / "guidance") -> None:
        self.guidance_dir = guidance_dir.resolve()
        self.bwrap = Path(shutil.which("bwrap") or "")
        self.claude = Path(shutil.which("claude") or "").resolve()
        if not self.bwrap.is_file() or not self.claude.is_file():
            raise AgentError("claude and bubblewrap are required")

    def _command(
        self,
        workspace: Path,
        *,
        evidence_dir: Path | None = None,
        model: str,
        tools: str,
        output_format: str,
        schema_path: Path | None,
    ) -> list[str]:
        credentials = Path.home() / ".claude" / ".credentials.json"
        account = Path.home() / ".claude.json"
        if not credentials.is_file() or not account.is_file():
            raise AgentError("Claude authentication files are missing")
        command = [
            str(self.bwrap),
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--ro-bind",
            "/etc",
            "/etc",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--ro-bind",
            "/sys",
            "/sys",
            "--dir",
            "/run",
            "--dir",
            "/run/systemd",
            "--dir",
            "/run/systemd/resolve",
            "--ro-bind",
            "/run/systemd/resolve/stub-resolv.conf",
            "/run/systemd/resolve/stub-resolv.conf",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/runtime",
            "--ro-bind",
            str(self.claude),
            "/runtime/claude",
            "--dir",
            "/home",
            "--dir",
            "/home/researcher",
            "--dir",
            "/home/researcher/.claude",
            "--ro-bind",
            str(credentials),
            "/home/researcher/.claude/.credentials.json",
            "--ro-bind",
            str(account),
            "/home/researcher/.claude.json",
            "--ro-bind",
            str(self.guidance_dir),
            "/guidance",
            "--bind",
            str(workspace.resolve()),
            "/workspace",
            "--chdir",
            "/workspace",
            "--setenv",
            "HOME",
            "/home/researcher",
            "--setenv",
            "PATH",
            "/usr/bin:/runtime",
            "/runtime/claude",
            "--print",
            "--safe-mode",
            "--no-session-persistence",
            "--model",
            model,
            "--effort",
            "max",
            "--permission-mode",
            "acceptEdits",
            "--tools",
            tools,
            "--allowedTools",
            tools,
            "--output-format",
            output_format,
        ]
        if output_format == "stream-json":
            command.append("--verbose")
        if evidence_dir is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            executable_index = max(
                index for index, value in enumerate(command) if value == "/runtime/claude"
            )
            command[executable_index:executable_index] = [
                "--ro-bind",
                str(evidence_dir.resolve()),
                "/evidence",
            ]
        if schema_path is not None:
            command.extend(["--json-schema", schema_path.read_text()])
        return command

    def run(
        self,
        workspace: Path,
        prompt: str,
        *,
        evidence_dir: Path | None = None,
        model: str = "fable",
        tools: str = "Read,Write,Edit,Glob,Grep,WebSearch,WebFetch",
        timeout_seconds: int = 1800,
        structured_schema: Path | None = None,
    ) -> AgentResult:
        workspace.mkdir(parents=True, exist_ok=True)
        output_format = "json" if structured_schema else "stream-json"
        command = self._command(
            workspace,
            evidence_dir=evidence_dir,
            model=model,
            tools=tools,
            output_format=output_format,
            schema_path=structured_schema,
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return AgentResult(
                status="timeout",
                returncode=None,
                wall_seconds=time.monotonic() - started,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            )
        structured = None
        status = "ok" if completed.returncode == 0 else "agent_error"
        if status == "ok" and structured_schema is not None:
            try:
                envelope = json.loads(completed.stdout)
                value = envelope.get("structured_output")
                if value is None and isinstance(envelope.get("result"), str):
                    value = json.loads(envelope["result"])
                if not isinstance(value, dict):
                    raise ValueError("structured result is missing")
                structured = value
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                status = "invalid_output"
                completed.stderr += f"\nStructured-output parse failed: {exc}"
        return AgentResult(
            status=status,
            returncode=completed.returncode,
            wall_seconds=time.monotonic() - started,
            stdout=completed.stdout,
            stderr=completed.stderr,
            structured=structured,
        )
