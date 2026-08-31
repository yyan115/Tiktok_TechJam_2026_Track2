#!/usr/bin/env python3
"""Best-effort PreToolUse guard for legacy host-side Bash access.

The official researcher has no Bash tool and cannot see these paths.  This
hook is only a deny-biased accident guard for any host-side Claude session;
the external controller, immutable snapshots, and authenticated state remain
the authority.  It deliberately prefers a false denial over allowing a
malformed or obviously destructive command.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAX_PAYLOAD_BYTES = 512 * 1024
MAX_COMMAND_CHARS = 256 * 1024
MAX_TOKENS = 16_384

# Keep this literal tuple in parity with controller_service.PROTECTED_PATHS.
# A subprocess-level regression test compares the two lists and then probes
# every path against these actual hook bytes.
CONTROLLER_PROTECTED_PATHS = (
    "CLAUDE.md",
    "README.md",
    "Project/PLAN.md",
    "Project/RESEARCH_PROTOCOL.md",
    "Project/RUNBOOK.md",
    "Project/RESEARCHER_BRIEF.md",
    "Project/research/templates/attempt.template.json",
    "Project/harness/iterate.py",
    "Project/harness/policy.py",
    "Project/harness/sandbox.py",
    "Project/harness/candidate_worker.py",
    "Project/harness/authority.py",
    "Project/harness/input_snapshot.py",
    "Project/harness/research_bank.py",
    "Project/harness/controller_service.py",
    "Project/harness/researcher_shell.py",
    "Project/harness/controller_mcp_config.json",
    "Project/harness/claude_runtime.json",
    "Project/tools/preflight_review.py",
    "Project/tools/control.py",
    "Project/tools/controller_mcp.py",
    "Project/tools/init_researcher_workspace.py",
    "Project/audits/preflight_schema.json",
    "Project/manifest.json",
    "kuairand-starter-kit/data.py",
    "kuairand-starter-kit/evaluate.py",
    "kuairand-starter-kit/submit.py",
    "kuairand-starter-kit/baseline.py",
    "kuairand-starter-kit/ablation_features.py",
    "kuairand-starter-kit/baseline_scores.json",
    "kuairand-starter-kit/README.md",
)

# Directory nodes are intentional: both the node and every descendant are
# protected.  Path-relation checks also protect an ancestor when a command
# could remove or relocate the entire protected tree.
EXTRA_PROTECTED_REPO_NODES = (
    ".claude",
    ".git",
    "Project/harness",
    "Project/results",
    "Project/audits/preflight",
    "Project/research/bank",
    "Project/research/templates/portfolio.template.json",
    "kuairand-starter-kit",
    "kuairand-starter-kit.zip",
    "kuairand-starter-kit/KuaiRand-Pure/data",
    "kuairand-starter-kit/KuaiRand-Pure/data_sanitized",
)
PROTECTED_REPO_NODES = tuple(dict.fromkeys(
    (*CONTROLLER_PROTECTED_PATHS, *EXTRA_PROTECTED_REPO_NODES)
))

SENSITIVE_ENV_PATHS = (
    "TRACK2_CONTROLLER_SOCKET",
    "TRACK2_CONTROLLER_CLIENT_STATE",
    "TRACK2_RESEARCHER_WORKSPACE",
)
SENSITIVE_PATH_COMPONENTS = frozenset({
    ".controller-service.lock",
    ".controller.lock",
    ".track2-controller-client",
    ".track2-controller.lock",
    ".track2-portfolio.json",
    ".track2-workspace.json",
    "controller.audit.jsonl",
    "controller.sock",
    "tiktok-techjam-2026-track2",
    "track2-agent-home",
    "track2-runtime",
    "track2-workspaces",
})
DISCOVERY_MARKERS = (
    ".track2-workspace.json",
    ".track2-portfolio.json",
    ".track2-controller.lock",
    ".track2-controller-client",
    "controller.sock",
)

DIRECT_MUTATORS = frozenset({
    "7z", "chgrp", "chmod", "chown", "cmake", "cp", "cpio", "dd", "ed",
    "ex", "install", "link", "ln", "make", "mkdir", "mkfifo", "mknod",
    "mv", "nano", "ninja", "npm", "patch", "pax", "pip", "pip3",
    "pnpm", "rename", "rmdir", "rm", "rsync", "scp", "setfacl",
    "setfattr", "shred", "sponge", "sftp", "sqlite3", "tee", "touch",
    "truncate", "unlink", "uv", "vi", "vim", "yarn", "zip",
})
SHELLS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
INLINE_INTERPRETERS = {
    "awk": None,
    "gawk": None,
    "node": "-e",
    "nodejs": "-e",
    "perl": "-e",
    "php": "-r",
    "python": "-c",
    "python2": "-c",
    "python3": "-c",
    "ruby": "-e",
}
GIT_MUTATING_SUBCOMMANDS = frozenset({
    "add", "am", "apply", "bisect", "branch", "checkout", "cherry-pick",
    "clean", "clone", "commit", "fetch", "gc", "init", "maintenance",
    "merge", "mv", "notes", "pack-refs", "pull", "push", "rebase",
    "reflog", "remote", "repack", "replace", "reset", "restore", "revert",
    "rm", "stash", "submodule", "switch", "tag", "update-index",
    "update-ref", "worktree",
})
GIT_OPTIONS_WITH_VALUE = frozenset({
    "-C", "-c", "--exec-path", "--git-dir", "--namespace", "--work-tree",
})
WRAPPERS = frozenset({
    "chrt", "command", "exec", "ionice", "nohup", "stdbuf", "sudo", "time",
})
CONTROL_CHARS = frozenset(";&|()\n")
OUTPUT_REDIRECTIONS = frozenset({">", ">>", ">|", "<>", "&>"})


class GuardInputError(ValueError):
    """The hook invocation or shell text is not safe to interpret."""


def _deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"Blocked by Track 2 safety guard: {reason}",
        }
    }, separators=(",", ":")))


def _strict_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise GuardInputError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str):
    raise GuardInputError("non-finite JSON value")


def _read_command() -> str:
    raw = sys.stdin.buffer.read(MAX_PAYLOAD_BYTES + 1)
    if not raw or len(raw) > MAX_PAYLOAD_BYTES:
        raise GuardInputError("missing or oversized hook payload")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardInputError("malformed hook JSON") from exc
    if not isinstance(payload, dict):
        raise GuardInputError("hook payload must be an object")
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        raise GuardInputError("tool_input must be an object")
    command = tool_input.get("command")
    if (
        not isinstance(command, str)
        or not command.strip()
        or len(command) > MAX_COMMAND_CHARS
        or "\x00" in command
    ):
        raise GuardInputError("command must be a bounded non-empty string")
    return command


def _tokenize(command: str) -> list[str]:
    try:
        # Keep newlines as shell separators, including while quoted text stays
        # one token.  This avoids treating a second command as an argument to
        # the first command.
        lexer = shlex.shlex(
            command, posix=True, punctuation_chars=";&|<>()\n"
        )
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError as exc:
        raise GuardInputError("shell text has malformed quoting") from exc
    if not tokens or len(tokens) > MAX_TOKENS:
        raise GuardInputError("shell token stream is empty or oversized")
    return tokens


def _is_control(token: str) -> bool:
    return bool(token) and all(char in CONTROL_CHARS for char in token)


def _segments(tokens: list[str]) -> list[list[str]]:
    result: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if _is_control(token):
            if current:
                result.append(current)
                current = []
        else:
            current.append(token)
    if current:
        result.append(current)
    return result


def _basename(token: str) -> str:
    return token.rstrip("/").rsplit("/", 1)[-1]


def _repo_roots() -> tuple[str, ...]:
    roots = {posixpath.normpath(str(ROOT))}
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if project_dir and project_dir.startswith("/") and "\x00" not in project_dir:
        roots.add(posixpath.normpath(project_dir))
    return tuple(sorted(roots))


def _sensitive_external_nodes() -> tuple[str, ...]:
    nodes = {
        posixpath.normpath(str(
            Path.home() / ".local" / "state" / "tiktok-techjam-2026-track2"
        ))
    }
    for name in SENSITIVE_ENV_PATHS:
        value = os.environ.get(name)
        if value and value.startswith("/") and "\x00" not in value:
            normalized = posixpath.normpath(value)
            nodes.add(normalized)
            if name == "TRACK2_CONTROLLER_SOCKET":
                nodes.add(posixpath.dirname(normalized))
    return tuple(sorted(nodes))


def _relation(left: str, right: str) -> bool:
    """True when either normalized absolute path contains the other."""

    try:
        return posixpath.commonpath((left, right)) in {left, right}
    except ValueError:
        return False


def _strip_assignment(token: str) -> str:
    if "=" not in token:
        return token
    key, value = token.split("=", 1)
    if re.fullmatch(r"(?:--?[A-Za-z][A-Za-z0-9_-]*|[A-Za-z_][A-Za-z0-9_]*)", key):
        return value
    return token


def _normalized_operand(token: str, cwd: str) -> str | None:
    value = _strip_assignment(token).strip()
    if len(value) > 2 and value[:2] in {"-O", "-P", "-o"}:
        value = value[2:]
    if not value or value in {"-", "--"}:
        return None
    for root in _repo_roots():
        value = value.replace("${CLAUDE_PROJECT_DIR}", root)
        value = value.replace("$CLAUDE_PROJECT_DIR", root)
    if value == "~" or value.startswith("~/"):
        value = posixpath.join(str(Path.home()), value[2:]) if value != "~" else str(Path.home())
    if any(marker in value for marker in ("$", "`", "$(")):
        return None
    # Inline snippets are handled by raw node matching, not mistaken for a
    # literal path here.
    if any(char.isspace() for char in value) or any(char in value for char in "'\"(),{}"):
        return None
    glob_at = min(
        (index for index, char in enumerate(value) if char in "*?["),
        default=-1,
    )
    if glob_at >= 0:
        prefix = value[:glob_at]
        if not prefix:
            prefix = "."
        elif not prefix.endswith("/"):
            prefix = posixpath.dirname(prefix) or "."
        value = prefix
    if value.startswith("/"):
        return posixpath.normpath(value)
    return posixpath.normpath(posixpath.join(cwd, value))


def _repo_node_pattern(node: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in node.split("/")]
    joined = r"/+(?:\./+)*".join(pieces)
    return re.compile(
        rf"(?<![A-Za-z0-9_.-])(?:\./+)*{joined}(?=$|[/\s'\"`:,;)\]}}])"
    )


RAW_NODE_PATTERNS = tuple(_repo_node_pattern(node) for node in PROTECTED_REPO_NODES)


def _raw_protected_reference(text: str, *, include_ancestors: bool = False) -> bool:
    if any(pattern.search(text) for pattern in RAW_NODE_PATTERNS):
        return True
    if include_ancestors and re.search(
        r"(?<![A-Za-z0-9_.-])(?:Project|kuairand-starter-kit)(?=$|[/\s'\"`:,;)\]}])",
        text,
    ):
        return True
    if any(
        f"${name}" in text or f"${{{name}}}" in text
        for name in SENSITIVE_ENV_PATHS
    ):
        return True
    return any(
        re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(component)}(?=$|[/\s'\"`:,;)\]}}])", text)
        for component in SENSITIVE_PATH_COMPONENTS
    )


def _discoverable_sensitive_tree(path: str) -> bool:
    """Recognize arbitrary external runtime/workspace names by marker files."""

    current = Path(path)
    if not current.is_dir():
        current = current.parent
    for _ in range(16):
        for marker in DISCOVERY_MARKERS:
            try:
                (current / marker).lstat()
            except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
                continue
            else:
                return True
        if current.parent == current:
            break
        current = current.parent
    return False


def _protected_operand(token: str, cwd: str) -> bool:
    if _raw_protected_reference(token):
        return True
    normalized = _normalized_operand(token, cwd)
    if normalized is None:
        return False
    if any(
        component in SENSITIVE_PATH_COMPONENTS
        for component in Path(normalized).parts
    ):
        return True
    for root in _repo_roots():
        for relative in PROTECTED_REPO_NODES:
            if _relation(normalized, posixpath.join(root, relative)):
                return True
    if any(_relation(normalized, node) for node in _sensitive_external_nodes()):
        return True
    return normalized.startswith("/") and _discoverable_sensitive_tree(normalized)


def _effective_command(segment: list[str]) -> tuple[int, str] | None:
    index = 0
    while index < len(segment):
        token = segment[index]
        if token in OUTPUT_REDIRECTIONS or token in {"<", "<<", "<<<"}:
            index += 2
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token, re.DOTALL):
            index += 1
            continue
        break
    if index >= len(segment):
        return None
    command = _basename(segment[index])
    if command == "env":
        index += 1
        while index < len(segment):
            token = segment[index]
            if token == "--":
                index += 1
                break
            if token.startswith("-") or re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*=.*", token, re.DOTALL
            ):
                index += 1
                continue
            break
        if index < len(segment):
            return index, _basename(segment[index])
        return None
    if command in WRAPPERS or command in {"busybox", "nice", "xargs"}:
        known = (
            DIRECT_MUTATORS | SHELLS | frozenset(INLINE_INTERPRETERS)
            | {"cd", "eval", "find", "git", "perl", "sed", "tar", "unzip"}
        )
        for nested in range(index + 1, len(segment)):
            candidate = _basename(segment[nested])
            if candidate in known:
                return nested, candidate
    return index, command


def _git_subcommand(arguments: list[str]) -> str | None:
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            index += 1
            break
        if token in GIT_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if any(token.startswith(option + "=") for option in GIT_OPTIONS_WITH_VALUE):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return _basename(token)
    return _basename(arguments[index]) if index < len(arguments) else None


def _recursive_rm_is_unsafe(arguments: list[str], cwd: str) -> bool:
    recursive = False
    operands: list[str] = []
    after_double_dash = False
    for token in arguments:
        if not after_double_dash and token == "--":
            after_double_dash = True
            continue
        if not after_double_dash and token.startswith("--"):
            option = token[2:].split("=", 1)[0]
            if option and "recursive".startswith(option):
                recursive = True
            continue
        if not after_double_dash and token.startswith("-") and token != "-":
            if any(char in "rR" for char in token[1:]):
                recursive = True
            continue
        operands.append(token)
    if not recursive:
        return False
    if not operands:
        return True
    for token in operands:
        if ".." in Path(token).parts or any(char in token for char in "$`*?["):
            return True
        normalized = _normalized_operand(token, cwd)
        if (
            normalized is None
            or normalized == "/tmp"
            or not normalized.startswith("/tmp/")
            or any(_relation(normalized, root) for root in _repo_roots())
        ):
            return True
    return False


def _redirection_writes_protected(segment: list[str], cwd: str) -> bool:
    for index, token in enumerate(segment[:-1]):
        if token in OUTPUT_REDIRECTIONS and _protected_operand(segment[index + 1], cwd):
            return True
    return False


def _in_place_editor(command: str, arguments: list[str]) -> bool:
    if command == "sed":
        return any(
            token == "-i" or token.startswith("-i") or token.startswith("--in-place")
            for token in arguments
        )
    if command == "perl":
        return any(
            token.startswith("-") and "i" in token[1:]
            for token in arguments
        )
    return False


def _archive_extracts(command: str, arguments: list[str]) -> bool:
    if command == "unzip":
        return not any(token in {"-l", "-t", "-v", "-Z"} for token in arguments)
    if command != "tar":
        return False
    for token in arguments:
        if token.startswith("--"):
            if token.split("=", 1)[0] in {"--extract", "--create", "--append", "--update", "--delete"}:
                return True
        elif token.startswith("-") and any(char in "xcrudA" for char in token[1:]):
            return True
        elif token and not token.startswith("-") and any(char in "xcrudA" for char in token):
            return True
    return False


def _downloader_writes_named_output(command: str, arguments: list[str]) -> bool:
    if command == "curl":
        return any(
            token in {"-o", "--output", "--output-dir"}
            or token.startswith("--output=")
            or token.startswith("--output-dir=")
            for token in arguments
        )
    if command == "wget":
        return any(
            token in {"-O", "-P", "--directory-prefix", "--output-document"}
            or token.startswith("-O")
            or token.startswith("-P")
            or token.startswith("--directory-prefix=")
            or token.startswith("--output-document=")
            for token in arguments
        )
    return False


def _segment_denial(segment: list[str], cwd: str, depth: int) -> str | None:
    effective = _effective_command(segment)
    if effective is None:
        return None
    command_index, command = effective
    arguments = segment[command_index + 1:]

    leading = _basename(segment[0]) if segment else ""
    if leading == "xargs" and command in DIRECT_MUTATORS:
        return "xargs would feed uninspectable targets to a filesystem writer"

    if command == "rm" and _recursive_rm_is_unsafe(arguments, cwd):
        return "recursive deletion is allowed only for an explicit safe path below /tmp"
    if command == "git" and _git_subcommand(arguments) in GIT_MUTATING_SUBCOMMANDS:
        return "a Git writer could alter protected repository metadata or the worktree"
    if _redirection_writes_protected(segment, cwd):
        return "an output redirection targets protected state"

    if command in SHELLS and "-c" in arguments:
        index = arguments.index("-c")
        if index + 1 >= len(arguments):
            return "a shell -c invocation has no inspectable command"
        if depth >= 4:
            return "nested shell depth exceeds the guard's inspection bound"
        return _command_denial(arguments[index + 1], depth + 1)
    if command == "eval":
        if depth >= 4:
            return "nested eval depth exceeds the guard's inspection bound"
        return _command_denial(" ".join(arguments), depth + 1)

    mutates = command in DIRECT_MUTATORS
    mutates = mutates or _in_place_editor(command, arguments)
    mutates = mutates or (
        command == "find"
        and any(token in {"-delete", "-exec", "-execdir", "-ok", "-okdir"} for token in arguments)
    )
    mutates = mutates or _archive_extracts(command, arguments)
    mutates = mutates or _downloader_writes_named_output(command, arguments)
    inline_flag = INLINE_INTERPRETERS.get(command, "not-an-interpreter")
    inline = command in INLINE_INTERPRETERS and (
        inline_flag is None or inline_flag in arguments
    )
    if inline:
        mutates = True

    if mutates:
        include_ancestors = inline
        if any(_protected_operand(token, cwd) for token in arguments):
            return "a command that can write, delete, or relocate targets protected state"
        if _raw_protected_reference(
            " ".join(arguments), include_ancestors=include_ancestors
        ):
            return "inline code or a writer references protected state"
    return None


def _cd_target(segment: list[str], cwd: str) -> str:
    effective = _effective_command(segment)
    if effective is None or effective[1] not in {"cd", "pushd"}:
        return cwd
    arguments = segment[effective[0] + 1:]
    operands = [token for token in arguments if token != "--" and not token.startswith("-")]
    if not operands:
        return str(Path.home())
    normalized = _normalized_operand(operands[0], cwd)
    return normalized if normalized is not None else cwd


def _command_denial(command: str, depth: int = 0) -> str | None:
    tokens = _tokenize(command)
    cwd = _repo_roots()[0]
    segments = _segments(tokens)
    for segment in segments:
        denial = _segment_denial(segment, cwd, depth)
        if denial is not None:
            return denial
        cwd = _cd_target(segment, cwd)

    raw_reference = _raw_protected_reference(command, include_ancestors=True)
    if raw_reference and "<<" in tokens:
        for segment in segments:
            effective = _effective_command(segment)
            if effective is not None and (
                effective[1] in SHELLS or effective[1] in INLINE_INTERPRETERS
            ):
                return "an interpreter heredoc contains a protected path"
    if raw_reference and any(token in {"|", "|&"} for token in tokens):
        for segment in segments:
            effective = _effective_command(segment)
            if effective is None:
                continue
            command_index, executable = effective
            arguments = segment[command_index + 1:]
            if executable in SHELLS and "-c" not in arguments:
                return "a piped shell would execute uninspectable text mentioning protected state"
            if executable in INLINE_INTERPRETERS and (
                not arguments or arguments == ["-"]
            ):
                return "a piped interpreter would execute uninspectable text mentioning protected state"

    # Command substitutions are not a full shell grammar in shlex.  Catch the
    # obvious nested-writer form without pretending this advisory hook is a
    # parser or a security boundary.
    if (
        ("$(" in command or "`" in command)
        and _raw_protected_reference(command, include_ancestors=True)
        and re.search(
        r"(?:\$\(|`)\s*(?:(?:/\S+/)?(?:sudo|env|command)\s+)*(?:/\S+/)?"
        r"(?:rm|mv|cp|install|ln|truncate|touch|chmod|chown|tee|dd|patch)\b",
        command,
        )
    ):
        return "an obvious nested writer references protected state"
    return None


def main() -> None:
    try:
        command = _read_command()
        denial = _command_denial(command)
    except Exception as exc:
        _deny(str(exc))
        return
    if denial is not None:
        _deny(denial)


if __name__ == "__main__":
    main()
