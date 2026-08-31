import json
from pathlib import Path

import numpy as np
import pytest

from harness.controller import Controller, ControllerError
from harness.sandbox import SandboxResult
from harness.spec import BenchmarkSpec, canonical_json_bytes


class FakeRunner:
    def __init__(self, predictions: list[np.ndarray | None]) -> None:
        self.predictions = list(predictions)
        self.last_prediction: np.ndarray | None = None

    def run(self, **kwargs) -> SandboxResult:
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        if kwargs["mode"] == "checkpoint_check":
            assert self.last_prediction is not None
            np.save(output / "test_predictions.npy", self.last_prediction.astype(np.float32))
            return SandboxResult("ok", 0, 0.01, "replayed", "", 100)
        prediction = self.predictions.pop(0)
        if prediction is None:
            return SandboxResult("candidate_error", 1, 0.01, "", "boom", 0)
        self.last_prediction = prediction
        np.save(output / "validation_predictions.npy", prediction.astype(np.float32))
        checkpoint = output / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.bin").write_bytes(b"frozen")
        return SandboxResult("ok", 0, 0.01, "trained", "", 100)


class FinalFakeRunner:
    def __init__(self, validation: np.ndarray, test: np.ndarray) -> None:
        self.validation = validation
        self.test = test
        self.modes: list[str] = []

    def run(self, **kwargs) -> SandboxResult:
        mode = kwargs["mode"]
        self.modes.append(mode)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        if mode == "attempt":
            np.save(output / "validation_predictions.npy", self.validation.astype(np.float32))
            checkpoint = output / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.bin").write_bytes(b"exact-final-checkpoint")
        elif mode == "checkpoint_check":
            np.save(output / "test_predictions.npy", self.validation.astype(np.float32))
        else:
            assert kwargs["checkpoint_dir"].is_dir()
            np.save(output / "test_predictions.npy", self.test.astype(np.float32))
        return SandboxResult("ok", 0, 0.01, mode, "", 100)


class BadReplayRunner(FakeRunner):
    def run(self, **kwargs) -> SandboxResult:
        if kwargs["mode"] != "checkpoint_check":
            return super().run(**kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        assert self.last_prediction is not None
        np.save(output / "test_predictions.npy", np.zeros_like(self.last_prediction, dtype=np.float32))
        return SandboxResult("ok", 0, 0.01, "wrong replay", "", 100)


def _policy(
    path: Path,
    *,
    epsilon: float = 0.002,
    window: int = 3,
    minimum: int = 4,
    max_attempts: int = 50,
) -> Path:
    value = {
        "benchmark_id": "kuairand-1k",
        "epsilon": epsilon,
        "window_scored_iterations": window,
        "minimum_scored_iterations": minimum,
        "max_attempts": max_attempts,
        "max_wall_seconds": 21600,
        "candidate_timeout_seconds": 3600,
        "max_candidate_output_bytes": 1000000,
        "selection": "earliest validation-best checkpoint at terminal",
        "tie_break": "earliest attempt",
        "failed_attempts_advance_convergence": False,
    }
    path.write_text(json.dumps(value))
    return path


def _derived(path: Path, *, test_rows: int = 2) -> Path:
    spec = BenchmarkSpec.load()
    (path / "private").mkdir(parents=True)
    np.save(path / "private" / "validation_users.npy", np.array([1, 1, 2, 2]))
    np.save(path / "private" / "validation_labels.npy", np.array([0, 1, 0, 1], dtype=np.int8))
    manifest = {
        "benchmark_spec_sha256": spec.digest,
        "hidden_test_targets_cached": False,
        "rows": {"train": 1, "validation": 4, "test": test_rows},
    }
    (path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return path


def _proposal(path: Path, name: str = "baseline") -> Path:
    path.mkdir()
    (path / "proposal.json").write_text(
        json.dumps(
            {
                "kind": name,
                "hypothesis": "A deterministic ordering provides a controller test.",
                "evidence": ["synthetic fixture"],
                "change": "Emit the supplied prediction vector.",
                "falsifier": "The exact metric does not improve.",
            }
        )
    )
    (path / "candidate.py").write_text("print('fixture')\n")
    return path


def _controller(tmp_path: Path, runner: FakeRunner, **policy_args) -> Controller:
    derived = _derived(tmp_path / "derived")
    controller = Controller(tmp_path / "run", derived_dir=derived, runner=runner)
    controller.initialize(_policy(tmp_path / "policy.json", **policy_args))
    controller.start()
    return controller


def test_attempt_is_counted_before_candidate_failure_and_recovery_continues(tmp_path) -> None:
    runner = FakeRunner([None, np.array([0.0, 1.0, 0.0, 1.0])])
    controller = _controller(tmp_path, runner)
    failed = controller.submit(_proposal(tmp_path / "p1"))
    scored = controller.submit(_proposal(tmp_path / "p2"))
    assert failed["attempt"] == 1
    assert failed["status"] == "candidate_error"
    assert scored["attempt"] == 2
    assert scored["status"] == "scored"
    assert controller.status()["attempts"] == 2
    assert controller.status()["scored_iterations"] == 1


def test_earliest_equal_best_is_selected_and_terminal_cannot_continue(tmp_path) -> None:
    predictions = np.array([0.0, 1.0, 0.0, 1.0])
    runner = FakeRunner([predictions, predictions, predictions])
    controller = _controller(tmp_path, runner, epsilon=0.0, window=2, minimum=3)
    for number in range(1, 4):
        controller.submit(_proposal(tmp_path / f"p{number}"))
    status = controller.status()
    assert status["terminal"]["reasons"] == ["convergence"]
    assert status["best_attempt"] == 1
    with pytest.raises(ControllerError, match="terminal"):
        controller.submit(_proposal(tmp_path / "p4"))


def test_checkpoint_must_reproduce_the_scored_predictions(tmp_path) -> None:
    controller = _controller(
        tmp_path, BadReplayRunner([np.array([0.0, 1.0, 0.0, 1.0])])
    )
    outcome = controller.submit(_proposal(tmp_path / "proposal"))
    assert outcome["status"] == "invalid_checkpoint"
    assert controller.status()["scored_iterations"] == 0


def test_pending_attempt_is_durably_recovered_as_counted_failure(tmp_path) -> None:
    controller = _controller(tmp_path, FakeRunner([]))
    with controller.ledger.locked():
        controller.ledger.append(
            "attempt_started",
            attempt=1,
            card={"hypothesis": "pending"},
            source_manifest_sha256="abc",
            source_files=[],
        )
    controller.recover()
    view = controller.view()
    assert view.attempt_count == 1
    assert view.pending_attempt is None
    assert view.outcomes[1]["status"] == "controller_interrupted"


def test_frozen_policy_cannot_be_modified_after_initialization(tmp_path) -> None:
    derived = _derived(tmp_path / "derived")
    policy = _policy(tmp_path / "policy.json")
    controller = Controller(tmp_path / "run", derived_dir=derived, runner=FakeRunner([]))
    controller.initialize(policy)
    frozen = json.loads(controller.policy_path.read_text())
    frozen["epsilon"] = 0.5
    controller.policy_path.chmod(0o644)
    controller.policy_path.write_text(json.dumps(frozen))
    with pytest.raises(ControllerError, match="policy was modified"):
        controller.start()


def test_finalizer_seals_predictions_before_labels_and_resumes_without_reinference(
    tmp_path,
) -> None:
    validation = np.array([0.0, 1.0, 0.0, 1.0])
    test = np.array([0.0, 1.0, 0.0, 1.0])
    runner = FinalFakeRunner(validation, test)
    derived = _derived(tmp_path / "derived", test_rows=4)
    raw = tmp_path / "raw"
    raw.mkdir()
    future_name = BenchmarkSpec.load().raw_files["future_log"]
    (raw / future_name).write_text(
        "user_id,video_id,date,long_view\n"
        "1,10,20220429,0\n"
        "1,11,20220429,1\n"
        "2,20,20220430,0\n"
        "2,21,20220430,1\n"
    )
    controller = Controller(
        tmp_path / "run", derived_dir=derived, raw_dir=raw, runner=runner
    )
    controller.initialize(_policy(tmp_path / "policy.json", max_attempts=1))
    controller.start()
    outcome = controller.submit(_proposal(tmp_path / "proposal"))
    assert outcome["status"] == "scored"
    assert controller.status()["terminal"]["reasons"] == ["attempt_cap"]

    original_loader = controller._load_hidden_test_targets

    def fail_after_seal():
        assert controller.view().hidden_score_started is not None
        raise RuntimeError("synthetic crash after sealing")

    controller._load_hidden_test_targets = fail_after_seal  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="synthetic crash"):
        controller.finalize()

    controller._load_hidden_test_targets = original_loader  # type: ignore[method-assign]
    final = controller.finalize()
    assert final["attempt"] == 1
    assert final["metrics"]["primary"] == pytest.approx(1.0)
    assert runner.modes == ["attempt", "checkpoint_check", "final"]
    assert len(
        [event for event in controller.view().events if event["type"] == "hidden_score_started"]
    ) == 1
    assert not list(derived.rglob("*test*label*"))
    with pytest.raises(ControllerError, match="already been scored"):
        controller.finalize()
