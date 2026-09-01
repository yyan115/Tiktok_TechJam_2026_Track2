import json

import pytest

from harness.supervisor import SupervisorError, observed_tool_uses, validate_research_plan


def _valid_plan():
    return {
        "problem_summary": "Rank logged impressions by long_view.",
        "eda_observations": [
            {"result_id": f"q{i}", "fact": f"fact {i}", "implication": f"decision {i}"}
            for i in range(3)
        ],
        "sources": [
            {
                "url": f"https://example.org/{i}",
                "title": f"source {i}",
                "claim": f"claim {i}",
                "limitations": f"limit {i}",
                "decision": f"choice {i}",
            }
            for i in range(3)
        ],
        "portfolio": [
            {
                "family": f"family {i}",
                "mechanism": f"mechanism {i}",
                "evidence": [f"q{i}"],
                "falsifier": f"falsifier {i}",
                "resource_plan": f"resource {i}",
            }
            for i in range(3)
        ],
        "first_attempt": "family 0 because it is the strongest test",
    }


def test_research_plan_requires_evidence_to_decision_chain(tmp_path) -> None:
    path = tmp_path / "research_plan.json"
    path.write_text(json.dumps(_valid_plan()))
    assert validate_research_plan(path)["first_attempt"].startswith("family 0")


def test_research_plan_rejects_citation_theatre(tmp_path) -> None:
    value = _valid_plan()
    del value["sources"][0]["decision"]
    path = tmp_path / "research_plan.json"
    path.write_text(json.dumps(value))
    with pytest.raises(SupervisorError, match="each source"):
        validate_research_plan(path)


def test_tool_trace_distinguishes_real_read_from_prompt_text() -> None:
    trace = "\n".join(
        [
            json.dumps({"type": "user", "text": "Read /guidance/PROBLEM.md"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {"file_path": "/guidance/PROBLEM.md"},
                            }
                        ]
                    },
                }
            ),
        ]
    )
    uses = observed_tool_uses(trace)
    assert uses == [{"name": "Read", "input": {"file_path": "/guidance/PROBLEM.md"}}]


def test_tool_trace_reads_codex_commands_and_web_search() -> None:
    trace = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "cat /guidance/PROBLEM.md",
                        "aggregated_output": "problem text",
                        "exit_code": 0,
                        "status": "completed",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "web_search",
                        "action": {"type": "search", "query": "KuaiRand paper"},
                    },
                }
            ),
        ]
    )
    assert observed_tool_uses(trace) == [
        {
            "name": "CommandExecution",
            "input": {
                "command": "cat /guidance/PROBLEM.md",
                "output": "problem text",
            },
        },
        {"name": "WebSearch", "input": {"query": "KuaiRand paper"}},
    ]
