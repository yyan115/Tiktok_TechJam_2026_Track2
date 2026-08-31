from harness.agent import ClaudeAgentRunner


def test_researcher_command_has_narrow_filesystem_and_no_bash(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = ClaudeAgentRunner()
    command = runner._command(
        workspace,
        evidence_dir=None,
        model="fable",
        tools="Read,Write,Edit,Glob,Grep,WebSearch,WebFetch",
        output_format="stream-json",
        schema_path=None,
    )
    rendered = " ".join(command)
    assert "Bash" not in rendered
    assert "--verbose" in command
    assert "/workspace" in command
    assert "/guidance" in command
    assert "datasets/KuaiRand-1K" not in rendered
    assert "derived_1k" not in rendered


def test_controller_evidence_is_mounted_read_only(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    evidence = tmp_path / "evidence"
    workspace.mkdir()
    evidence.mkdir()
    runner = ClaudeAgentRunner()
    command = runner._command(
        workspace,
        evidence_dir=evidence,
        model="fable",
        tools="Read,Write,Edit",
        output_format="stream-json",
        schema_path=None,
    )
    evidence_index = command.index(str(evidence.resolve()))
    assert command[evidence_index - 1] == "--ro-bind"
    assert command[evidence_index + 1] == "/evidence"
