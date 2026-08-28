#!/usr/bin/env python3
"""PreToolUse guard for Bash commands (Track 2).

Blocks shell commands that would WRITE to protected files (organizer starter
kit, README.md, manifest, results ledger). Reads are fine. Guards against
accidents, not malice — see Project/PLAN.md trust model.
"""
import json
import re
import sys

PROTECTED = [
    r"kuairand-starter-kit/evaluate\.py",
    r"kuairand-starter-kit/data\.py",
    r"kuairand-starter-kit/baseline\.py",
    r"kuairand-starter-kit/submit\.py",
    r"kuairand-starter-kit/ablation_features\.py",
    r"kuairand-starter-kit/baseline_scores\.json",
    r"kuairand-starter-kit\.zip",
    r"README\.md",
    r"manifest\.json",
    r"Project/results/",
    r"JOURNAL\.jsonl",
    r"\.claude/",
]
PROT = "(" + "|".join(PROTECTED) + ")"

WRITE_PATTERNS = [
    r">>?\s*\S*" + PROT,
    r"\btee\b(\s+-\S+)*\s+\S*" + PROT,
    r"\bsed\b[^|;&]*-i[^|;&]*" + PROT,
    r"\brm\b[^|;&]*" + PROT,
    r"\bmv\b[^|;&]*" + PROT,
    r"\bcp\b[^|;&]*\s\S*" + PROT + r"\s*($|[;&|])",
    r"\btruncate\b[^|;&]*" + PROT,
    r"\bchmod\b[^|;&]*" + PROT,
    r"\bln\b[^|;&]*" + PROT,
]


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    command = payload.get("tool_input", {}).get("command", "") or ""
    for pattern in WRITE_PATTERNS:
        if re.search(pattern, command):
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                "Blocked: this command writes to a protected file "
                                "(organizer starter kit / README / manifest / results "
                                "ledger). See Project/PLAN.md."
                            ),
                        }
                    }
                )
            )
            return


if __name__ == "__main__":
    main()
