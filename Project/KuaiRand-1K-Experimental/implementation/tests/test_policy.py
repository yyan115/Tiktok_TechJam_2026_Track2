import json

import pytest

from harness.policy import CampaignPolicy, PolicyError, convergence_details


def _policy(tmp_path, **changes) -> CampaignPolicy:
    value = {
        "benchmark_id": "kuairand-1k",
        "epsilon": 0.002,
        "window_scored_iterations": 3,
        "minimum_scored_iterations": 4,
        "max_attempts": 50,
        "max_wall_seconds": 21600,
        "candidate_timeout_seconds": 3600,
        "max_candidate_output_bytes": 1000000,
        "selection": "earliest validation-best checkpoint at terminal",
        "tie_break": "earliest attempt",
        "failed_attempts_advance_convergence": False,
    }
    value.update(changes)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value))
    return CampaignPolicy.load(path)


def _score(attempt: int, primary: float) -> dict:
    return {"attempt": attempt, "metrics": {"primary": primary}}


def test_failures_do_not_enter_scored_convergence_window(tmp_path) -> None:
    policy = _policy(tmp_path)
    # Attempts 2 and 4 failed and are intentionally absent.
    scored = [_score(1, 0.60), _score(3, 0.601), _score(5, 0.6015), _score(6, 0.6016)]
    details = convergence_details(scored, policy)
    assert details is not None
    assert details["window_attempts"] == [3, 5, 6]
    assert details["improvement"] == pytest.approx(0.0016)


def test_large_recent_improvement_prevents_convergence(tmp_path) -> None:
    policy = _policy(tmp_path)
    scored = [_score(1, 0.60), _score(2, 0.601), _score(3, 0.602), _score(4, 0.603)]
    assert convergence_details(scored, policy) is None


def test_template_placeholders_are_rejected(tmp_path) -> None:
    with pytest.raises(PolicyError, match="epsilon"):
        _policy(tmp_path, epsilon=None)
