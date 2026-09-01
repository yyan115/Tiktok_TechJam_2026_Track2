from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from harness.agent import AgentResult, ClaudeAgentRunner, CodexAgentRunner
from harness.controller import Controller, ControllerError, _validate_source
from harness.eda import EDAService
from harness.spec import ROOT, canonical_json_bytes, sha256_file


class SupervisorError(RuntimeError):
    pass


REQUIRED_RESEARCH_READS = (
    "/guidance/PROBLEM.md",
    "/guidance/ORGANIZER_REFERENCE.md",
    "/guidance/WORKFLOW.md",
    "/guidance/RESEARCHER.md",
)

RESEARCHER_MODEL = "gpt-5.6-sol"
CRITIC_MODEL = "opus"


def observed_tool_uses(stream_json: str) -> list[dict[str, Any]]:
    uses: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "tool_use" and isinstance(value.get("name"), str):
                uses.append({"name": value["name"], "input": value.get("input", {})})
            if (
                value.get("type") == "command_execution"
                and value.get("status") == "completed"
                and value.get("exit_code") == 0
            ):
                uses.append(
                    {
                        "name": "CommandExecution",
                        "input": {
                            "command": value.get("command", ""),
                            "output": value.get("aggregated_output", ""),
                        },
                    }
                )
            if value.get("type") == "web_search":
                action = value.get("action", {})
                query = value.get("query") or (
                    action.get("query", "") if isinstance(action, dict) else ""
                )
                if query:
                    uses.append({"name": "WebSearch", "input": {"query": query}})
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for line in stream_json.splitlines():
        try:
            visit(json.loads(line))
        except json.JSONDecodeError:
            continue
    return uses


def validate_research_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorError("research_plan.json is missing or invalid") from exc
    required = {"problem_summary", "eda_observations", "sources", "portfolio", "first_attempt"}
    missing = sorted(required.difference(plan))
    if missing:
        raise SupervisorError(f"research plan fields are missing: {missing}")
    if not isinstance(plan["problem_summary"], str) or not plan["problem_summary"].strip():
        raise SupervisorError("problem_summary must be non-empty")
    if not isinstance(plan["first_attempt"], str) or not plan["first_attempt"].strip():
        raise SupervisorError("first_attempt must be non-empty")
    observations = plan["eda_observations"]
    if not isinstance(observations, list) or len(observations) < 3:
        raise SupervisorError("research plan needs at least three EDA observations")
    for observation in observations:
        if not isinstance(observation, dict) or not all(
            isinstance(observation.get(field), str) and observation[field].strip()
            for field in ("result_id", "fact", "implication")
        ):
            raise SupervisorError("each EDA observation needs result_id, fact and implication")
    sources = plan["sources"]
    if not isinstance(sources, list) or len(sources) < 3:
        raise SupervisorError("research plan needs at least three sources")
    for source in sources:
        if not isinstance(source, dict) or not all(
            isinstance(source.get(field), str) and source[field].strip()
            for field in ("url", "title", "claim", "limitations", "decision")
        ):
            raise SupervisorError("each source needs URL, title, claim, limitations and decision")
        if not source["url"].startswith(("https://", "http://")):
            raise SupervisorError("research source URL is invalid")
    portfolio = plan["portfolio"]
    if not isinstance(portfolio, list) or len(portfolio) < 3:
        raise SupervisorError("research portfolio needs at least three families")
    names: set[str] = set()
    for family in portfolio:
        if not isinstance(family, dict) or not all(
            isinstance(family.get(field), str) and family[field].strip()
            for field in ("family", "mechanism", "falsifier", "resource_plan")
        ):
            raise SupervisorError("each portfolio family is incomplete")
        if not isinstance(family.get("evidence"), list) or not family["evidence"]:
            raise SupervisorError("each portfolio family needs evidence")
        normalized = family["family"].strip().lower()
        if normalized in names:
            raise SupervisorError("portfolio family names must be distinct")
        names.add(normalized)
    return plan


class Supervisor:
    def __init__(
        self,
        run_dir: Path,
        workspace: Path,
        *,
        controller: Controller | None = None,
        researcher_agent: CodexAgentRunner | None = None,
        critic_agent: ClaudeAgentRunner | None = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.workspace = workspace.resolve()
        self.controller = controller or Controller(self.run_dir)
        self.researcher_agent = researcher_agent or CodexAgentRunner()
        self.critic_agent = critic_agent or ClaudeAgentRunner()
        self.traces = self.run_dir / "agent-traces"
        self.evidence = self.run_dir / "evidence"

    def _write_evidence_json(self, relative: str, value: Any) -> None:
        path = self.evidence / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value))

    def _record_agent_result(self, role: str, result: AgentResult) -> None:
        self.traces.mkdir(parents=True, exist_ok=True)
        number = len(list(self.traces.glob("*.json"))) + 1
        payload = {
            "role": role,
            "status": result.status,
            "returncode": result.returncode,
            "wall_seconds": result.wall_seconds,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "structured": result.structured,
            "provider": result.provider,
            "model": result.model,
            "effort": result.effort,
        }
        (self.traces / f"{number:04d}-{role}.json").write_bytes(canonical_json_bytes(payload))

    def _remaining_seconds(self, ceiling: int) -> int:
        remaining = self.controller.policy.max_wall_seconds - self.controller.status()[
            "elapsed_seconds"
        ]
        return max(1, min(ceiling, int(remaining)))

    def _researcher(
        self,
        task: str,
        required_reads: tuple[str, ...] = (),
        require_web_research: bool = False,
    ) -> bool:
        prompt = (
            "Read /guidance/RESEARCHER.md first. Follow the current phase exactly. "
            "Do not merely explain what you would do: create the requested workspace file, "
            "then stop.\n\nCURRENT PHASE:\n" + task
        )
        result = self.researcher_agent.run(
            self.workspace,
            prompt,
            evidence_dir=self.evidence,
            model=RESEARCHER_MODEL,
            timeout_seconds=self._remaining_seconds(1800),
        )
        self._record_agent_result("researcher", result)
        if result.status != "ok":
            return False
        uses = observed_tool_uses(result.stdout)
        read_inputs = [
            json.dumps(use.get("input", {}), sort_keys=True)
            for use in uses
            if use["name"].lower() in {"read", "commandexecution"}
        ]
        if not all(any(path in item for item in read_inputs) for path in required_reads):
            return False
        if require_web_research and not any(
            use["name"].lower() in {"websearch", "webfetch", "web_search"}
            for use in uses
        ):
            return False
        return True

    def _critic(self, packet: dict[str, Any], purpose: str) -> dict[str, Any] | None:
        prompt = (
            (ROOT / "guidance" / "CRITIC.md").read_text()
            + "\n\nReview the packet below for "
            + purpose
            + ". Return only the required structured verdict.\n\n"
            + json.dumps(packet, sort_keys=True)
        )
        schema = ROOT / "guidance" / "critic_schema.json"
        for _ in range(2):
            result = self.critic_agent.run(
                self.workspace,
                prompt,
                evidence_dir=self.evidence,
                model=CRITIC_MODEL,
                tools="Read,WebSearch,WebFetch",
                timeout_seconds=self._remaining_seconds(600),
                structured_schema=schema,
            )
            self._record_agent_result("critic", result)
            if result.status == "ok" and result.structured is not None:
                return result.structured
        return None

    def _sync_state(self, last_outcome: dict[str, Any] | None = None) -> None:
        view = self.controller.view()
        attempts = []
        starts = {int(event["attempt"]): event for event in view.attempt_starts}
        for outcome in sorted(view.attempt_finishes, key=lambda event: int(event["attempt"])):
            attempt = int(outcome["attempt"])
            attempts.append(
                {
                    "attempt": attempt,
                    "card": starts.get(attempt, {}).get("card"),
                    "status": outcome.get("status"),
                    "metrics": outcome.get("metrics"),
                    "error": outcome.get("error"),
                    "stderr_tail": outcome.get("stderr_tail"),
                }
            )
        self._write_evidence_json(
            "STATE.json",
            {
                "controller": self.controller.status(),
                "attempts": attempts,
                "last_outcome": last_outcome,
                "instruction": "Reflect on evidence and choose keep, refine, debug, ensemble, or abandon.",
            },
        )

    def _ensure_eda(self) -> bool:
        results_dir = self.evidence / "eda_results"
        if list(results_dir.glob("*.json")) if results_dir.exists() else []:
            return True
        request_path = self.workspace / "eda_request.json"
        if not request_path.is_file():
            ok = self._researcher(
                "Read /guidance/PROBLEM.md, /guidance/ORGANIZER_REFERENCE.md, "
                "/guidance/WORKFLOW.md and /guidance/EDA.md "
                "completely. Decide which bounded EDA answers would materially change model "
                "choices. Write /workspace/eda_request.json, then stop.",
                required_reads=(*REQUIRED_RESEARCH_READS, "/guidance/EDA.md"),
            )
            if not ok or not request_path.is_file():
                return False
        try:
            request = json.loads(request_path.read_text())
            result = EDAService().execute(request)
        except Exception as exc:
            self._write_evidence_json(
                f"feedback/eda-error-{time.time_ns()}.json",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            request_path.rename(self.workspace / f"eda_request.invalid-{int(time.time())}.json")
            return False
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "round-001.json").write_bytes(canonical_json_bytes(result))
        return True

    def _ensure_research_plan(self) -> bool:
        accepted_path = self.evidence / "research_plan.json"
        if accepted_path.is_file():
            return True
        plan_path = self.workspace / "research_plan.json"
        if not plan_path.is_file():
            eda_paths = tuple(
                f"/evidence/eda_results/{path.name}"
                for path in sorted((self.evidence / "eda_results").glob("*.json"))
            )
            ok = self._researcher(
                "Read /guidance/PROBLEM.md, /guidance/ORGANIZER_REFERENCE.md, "
                "/guidance/WORKFLOW.md, /guidance/RESEARCHER.md, every JSON file in "
                "/evidence/eda_results, and /guidance/RESEARCH_PLAN.md. "
                "Interpret the measurements, perform primary/official-source web research, and "
                "write /workspace/research_plan.json. Every source must change a concrete design "
                "decision. Then stop.",
                required_reads=(
                    *REQUIRED_RESEARCH_READS,
                    "/guidance/RESEARCH_PLAN.md",
                    *eda_paths,
                ),
                require_web_research=True,
            )
            if not ok or not plan_path.is_file():
                return False
        try:
            plan = validate_research_plan(plan_path)
        except SupervisorError as exc:
            self._write_evidence_json(
                f"feedback/research-plan-error-{time.time_ns()}.json", {"error": str(exc)}
            )
            plan_path.rename(self.workspace / f"research_plan.invalid-{time.time_ns()}.json")
            return False
        eda_paths = sorted((self.evidence / "eda_results").glob("*.json"))
        eda = [json.loads(path.read_text()) for path in eda_paths]
        verdict = self._critic({"research_plan": plan, "eda_results": eda}, "research plan quality")
        if verdict is None:
            verdict = {
                "hard_block": False,
                "decision": "accept_with_notes",
                "findings": [],
                "strategy_notes": ["critic unavailable after two infrastructure retries"],
            }
        if verdict.get("hard_block") is True:
            self._write_evidence_json(
                f"feedback/research-plan-blocked-{time.time_ns()}.json", verdict
            )
            history = self.workspace / "research_history"
            history.mkdir(exist_ok=True)
            shutil.move(plan_path, history / f"blocked-{time.time_ns()}.json")
            return False
        self._write_evidence_json("feedback/research_plan_review.json", verdict)
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        accepted_path.write_bytes(canonical_json_bytes(plan))
        with self.controller.ledger.locked():
            self.controller.ledger.append(
                "research_plan_accepted",
                research_plan_sha256=sha256_file(accepted_path),
                eda_results=[
                    {"path": path.name, "sha256": sha256_file(path)} for path in eda_paths
                ],
                critic_decision=verdict.get("decision"),
            )
        history = self.workspace / "research_history"
        history.mkdir(exist_ok=True)
        shutil.move(plan_path, history / "accepted-source.json")
        return True

    def _candidate_review_required(self, card: dict[str, Any]) -> bool:
        return card["kind"] in {"baseline", "new_family", "ensemble"}

    def _review_candidate(self, candidate_dir: Path, card: dict[str, Any]) -> bool:
        if not self._candidate_review_required(card):
            return True
        plan = json.loads((self.evidence / "research_plan.json").read_text())
        source = (candidate_dir / "candidate.py").read_text()
        verdict = self._critic(
            {"proposal": card, "candidate_py": source, "research_plan": plan, "state": self.controller.status()},
            "candidate validity, evidence and resource feasibility",
        )
        if verdict is None:
            verdict = {
                "hard_block": False,
                "decision": "accept_with_notes",
                "findings": [],
                "strategy_notes": ["critic unavailable after two infrastructure retries"],
            }
        self._write_evidence_json(
            f"feedback/candidate-review-{time.time_ns()}.json", verdict
        )
        return verdict.get("hard_block") is not True

    def _archive_candidate(self, candidate_dir: Path, category: str, attempt: int | None = None) -> None:
        suffix = f"attempt-{attempt:03d}" if attempt is not None else str(time.time_ns())
        destination = self.workspace / category / suffix
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(candidate_dir, destination)

    def _ensure_candidate(self) -> Path | None:
        candidate = self.workspace / "candidate"
        try:
            _validate_source(candidate)
            return candidate
        except ControllerError:
            if candidate.exists():
                self._archive_candidate(candidate, "invalid")
        ok = self._researcher(
            "Read /evidence/research_plan.json, /evidence/STATE.json, all files in "
            "/evidence/feedback, and the "
            "candidate contract in /guidance/WORKFLOW.md. Reflect on all prior outcomes. Write one "
            "complete /workspace/candidate/proposal.json and candidate.py for the highest-value "
            "next experiment. Do not run it. Then stop.",
            required_reads=(
                "/guidance/WORKFLOW.md",
                "/evidence/research_plan.json",
                "/evidence/STATE.json",
            ),
        )
        if not ok:
            return None
        try:
            _validate_source(candidate)
        except ControllerError as exc:
            self._write_evidence_json(
                f"feedback/candidate-error-{time.time_ns()}.json", {"error": str(exc)}
            )
            if candidate.exists():
                self._archive_candidate(candidate, "invalid")
            return None
        return candidate

    def run(self) -> dict[str, Any]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.controller.view().started is None:
            self.controller.start()
        self.controller.recover()
        self._sync_state()
        final_exception_count = 0

        while True:
            self.controller.recover()
            status = self.controller.status()
            if status["terminal"] is not None:
                if status["final_scored"] is None and status["best_attempt"] is not None:
                    final_failures = len(
                        [
                            event
                            for event in self.controller.view().events
                            if event["type"] == "final_failed"
                        ]
                    )
                    if final_failures >= 3 or final_exception_count >= 3:
                        raise SupervisorError(
                            "the frozen best checkpoint failed finalization three times"
                        )
                    try:
                        result = self.controller.finalize()
                    except ControllerError:
                        raise
                    except Exception as exc:
                        final_exception_count += 1
                        self._write_evidence_json(
                            f"feedback/finalizer-error-{time.time_ns()}.json",
                            {"error": f"{type(exc).__name__}: {exc}"},
                        )
                        time.sleep(2)
                        continue
                    if result["type"] != "final_scored":
                        time.sleep(2)
                        continue
                return self.controller.status()
            if not self._ensure_eda():
                time.sleep(2)
                continue
            if not self._ensure_research_plan():
                time.sleep(2)
                continue
            candidate = self._ensure_candidate()
            if candidate is None:
                time.sleep(2)
                continue
            card, _, _ = _validate_source(candidate)
            if not self._review_candidate(candidate, card):
                self._archive_candidate(candidate, "blocked")
                continue
            try:
                outcome = self.controller.submit(candidate)
            except ControllerError as exc:
                self._write_evidence_json(
                    f"feedback/controller-rejection-{time.time_ns()}.json",
                    {"error": str(exc)},
                )
                self._archive_candidate(candidate, "controller-rejected")
                self.controller.recover()
                continue
            self._archive_candidate(candidate, "submitted", int(outcome["attempt"]))
            self._sync_state(outcome)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-go autonomous KuaiRand-1K campaign")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    controller = Controller(args.run_dir)
    if controller.view().initialized is None:
        if args.policy is None:
            raise SystemExit("--policy is required for a new run")
        controller.initialize(args.policy)
    supervisor = Supervisor(
        args.run_dir,
        args.workspace,
        controller=controller,
    )
    print(json.dumps(supervisor.run(), sort_keys=True))


if __name__ == "__main__":
    main()
