from __future__ import annotations

import argparse
import ast
import difflib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from harness.ledger import EventLedger, LedgerError, export_jsonl
from harness.metrics import evaluate
from harness.policy import CampaignPolicy, convergence_details
from harness.runtime import runtime_fingerprint, trusted_code_hashes
from harness.sandbox import (
    BubblewrapRunner,
    SandboxError,
    SandboxResult,
    directory_manifest,
    directory_size,
)
from harness.spec import (
    DEFAULT_DERIVED_DIR,
    DEFAULT_RAW_DIR,
    BenchmarkSpec,
    canonical_json_bytes,
    sha256_file,
)


class ControllerError(RuntimeError):
    pass


class CandidateRunner(Protocol):
    def run(self, **kwargs: Any) -> SandboxResult: ...


REQUIRED_CARD_FIELDS = ("kind", "hypothesis", "evidence", "change", "falsifier")


@dataclass(frozen=True)
class RunView:
    events: list[dict[str, Any]]

    @property
    def initialized(self) -> dict[str, Any] | None:
        return next((event for event in self.events if event["type"] == "run_initialized"), None)

    @property
    def started(self) -> dict[str, Any] | None:
        return next((event for event in self.events if event["type"] == "run_started"), None)

    @property
    def terminal(self) -> dict[str, Any] | None:
        return next((event for event in self.events if event["type"] == "run_terminal"), None)

    @property
    def final_scored(self) -> dict[str, Any] | None:
        return next((event for event in self.events if event["type"] == "final_scored"), None)

    @property
    def hidden_score_started(self) -> dict[str, Any] | None:
        return next((event for event in self.events if event["type"] == "hidden_score_started"), None)

    @property
    def attempt_starts(self) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == "attempt_started"]

    @property
    def attempt_finishes(self) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == "attempt_finished"]

    @property
    def attempt_count(self) -> int:
        return len(self.attempt_starts)

    @property
    def outcomes(self) -> dict[int, dict[str, Any]]:
        values: dict[int, dict[str, Any]] = {}
        for event in self.attempt_finishes:
            attempt = int(event["attempt"])
            if attempt in values:
                raise ControllerError(f"duplicate outcome for attempt {attempt}")
            values[attempt] = event
        return values

    @property
    def pending_attempt(self) -> dict[str, Any] | None:
        outcomes = self.outcomes
        pending = [event for event in self.attempt_starts if int(event["attempt"]) not in outcomes]
        if len(pending) > 1:
            raise ControllerError("multiple pending attempts")
        return pending[0] if pending else None

    @property
    def scored_attempts(self) -> list[dict[str, Any]]:
        return sorted(
            (event for event in self.attempt_finishes if event.get("status") == "scored"),
            key=lambda event: int(event["attempt"]),
        )

    @property
    def best(self) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for event in self.scored_attempts:
            if best is None or float(event["metrics"]["primary"]) > float(
                best["metrics"]["primary"]
            ):
                best = event
        return best

    def elapsed(self, now: float | None = None) -> float:
        if self.started is None:
            return 0.0
        end = self.terminal["timestamp_unix"] if self.terminal else (now or time.time())
        return max(0.0, float(end) - float(self.started["timestamp_unix"]))


def _write_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_manifest(derived_dir: Path) -> dict[str, Any]:
    path = derived_dir / "manifest.json"
    if not path.is_file():
        raise ControllerError(f"derived-data manifest is missing: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("hidden_test_targets_cached") is not False:
        raise ControllerError("derived data must not cache hidden-test targets")
    return manifest


def _validate_card(path: Path) -> dict[str, Any]:
    try:
        card = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerError("proposal.json is missing or invalid") from exc
    missing = [field for field in REQUIRED_CARD_FIELDS if field not in card]
    if missing:
        raise ControllerError(f"proposal card is missing fields: {missing}")
    for field in ("kind", "hypothesis", "change", "falsifier"):
        if not isinstance(card[field], str) or not card[field].strip():
            raise ControllerError(f"proposal field must be non-empty text: {field}")
    if card["kind"] not in {"baseline", "new_family", "refine", "debug", "ensemble"}:
        raise ControllerError("proposal kind is invalid")
    if not isinstance(card["evidence"], list) or not card["evidence"]:
        raise ControllerError("proposal evidence must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in card["evidence"]):
        raise ControllerError("proposal evidence entries must be non-empty text")
    return card


def _validate_source(source_dir: Path) -> tuple[dict[str, Any], list[dict[str, str | int]], str]:
    if not source_dir.is_dir():
        raise ControllerError(f"proposal directory does not exist: {source_dir}")
    card = _validate_card(source_dir / "proposal.json")
    entrypoint = source_dir / "candidate.py"
    if not entrypoint.is_file():
        raise ControllerError("candidate.py is missing")
    try:
        ast.parse(entrypoint.read_text(), filename="candidate.py")
    except (OSError, SyntaxError) as exc:
        raise ControllerError(f"candidate.py does not parse: {exc}") from exc
    records, digest = directory_manifest(source_dir)
    if not records or len(records) > 100:
        raise ControllerError("proposal must contain between 1 and 100 files")
    if sum(int(record["bytes"]) for record in records) > 20 * 1024 * 1024:
        raise ControllerError("proposal source exceeds 20 MiB")
    return card, records, digest


def _snapshot_source(source_dir: Path, destination: Path) -> tuple[list[dict[str, str | int]], str]:
    if destination.exists():
        raise ControllerError(f"attempt source already exists: {destination}")
    shutil.copytree(source_dir, destination, symlinks=False)
    records, digest = directory_manifest(destination)
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    destination.chmod(0o555)
    return records, digest


def _write_code_diff(run_dir: Path, attempt: int, source_snapshot: Path) -> tuple[Path, str, str]:
    current = (source_snapshot / "candidate.py").read_text().splitlines(keepends=True)
    if attempt == 1:
        previous: list[str] = []
        previous_name = "/dev/null"
    else:
        previous_path = run_dir / "artifacts" / f"attempt-{attempt - 1:03d}" / "source" / "candidate.py"
        previous = previous_path.read_text().splitlines(keepends=True)
        previous_name = f"attempt-{attempt - 1:03d}/candidate.py"
    diff = "".join(
        difflib.unified_diff(
            previous,
            current,
            fromfile=previous_name,
            tofile=f"attempt-{attempt:03d}/candidate.py",
        )
    )
    path = source_snapshot.parent / "code.diff"
    path.write_text(diff)
    return path, sha256_file(path), diff[:200000]


def _validate_predictions(path: Path, expected_rows: int) -> np.ndarray:
    if not path.is_file():
        raise ControllerError(f"prediction file is missing: {path.name}")
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ControllerError("prediction file is not a valid NumPy array") from exc
    if values.ndim != 1 or len(values) != expected_rows:
        raise ControllerError(
            f"prediction shape mismatch: expected ({expected_rows},), found {values.shape}"
        )
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ControllerError("predictions must use float32 or float64")
    if not np.all(np.isfinite(values)):
        raise ControllerError("predictions contain NaN or infinity")
    return values


def _make_read_only(path: Path) -> None:
    for entry in path.rglob("*"):
        if entry.is_file():
            entry.chmod(0o444)
        elif entry.is_dir():
            entry.chmod(0o555)
    path.chmod(0o555)


class Controller:
    def __init__(
        self,
        run_dir: Path,
        *,
        derived_dir: Path = DEFAULT_DERIVED_DIR,
        raw_dir: Path = DEFAULT_RAW_DIR,
        runner: CandidateRunner | None = None,
        clock: Any = time.time,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.derived_dir = derived_dir.resolve()
        self.raw_dir = raw_dir.resolve()
        self.ledger = EventLedger(self.run_dir)
        self.runner = runner
        self.clock = clock

    @property
    def policy_path(self) -> Path:
        return self.run_dir / "policy.json"

    @property
    def policy(self) -> CampaignPolicy:
        return CampaignPolicy.load(self.policy_path)

    def view(self) -> RunView:
        return RunView(self.ledger.read())

    def _verify_frozen_authority(self) -> None:
        initialized = self.view().initialized
        if initialized is None:
            raise ControllerError("run is not initialized")
        if sha256_file(self.policy_path) != initialized["policy_sha256"]:
            raise ControllerError("frozen campaign policy was modified")
        benchmark_path = self.run_dir / "benchmark.json"
        if sha256_file(benchmark_path) != initialized["benchmark_spec_sha256"]:
            raise ControllerError("frozen benchmark specification was modified")
        manifest_path = self.derived_dir / "manifest.json"
        if sha256_file(manifest_path) != initialized["derived_manifest_sha256"]:
            raise ControllerError("derived-data manifest was modified")
        runtime_path = self.run_dir / "runtime.json"
        if sha256_file(runtime_path) != initialized["runtime_sha256"]:
            raise ControllerError("runtime record was modified")
        runtime = json.loads(runtime_path.read_text())
        if runtime.get("trusted_code_sha256") != trusted_code_hashes():
            raise ControllerError("trusted controller code changed after initialization")

    def initialize(self, policy_path: Path, spec: BenchmarkSpec | None = None) -> None:
        spec = spec or BenchmarkSpec.load()
        policy = CampaignPolicy.load(policy_path)
        manifest = _load_manifest(self.derived_dir)
        if manifest.get("benchmark_spec_sha256") != spec.digest:
            raise ControllerError("derived data was built from a different benchmark spec")
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise ControllerError(f"run directory is not empty: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(self.policy_path, policy.raw)
        _write_atomic(self.run_dir / "benchmark.json", spec.raw)
        _write_atomic(self.run_dir / "runtime.json", runtime_fingerprint())
        with self.ledger.locked():
            self.ledger.append(
                "run_initialized",
                benchmark_id=spec.raw["benchmark_id"],
                benchmark_spec_sha256=spec.digest,
                policy_sha256=policy.digest,
                derived_manifest_sha256=sha256_file(self.derived_dir / "manifest.json"),
                runtime_sha256=sha256_file(self.run_dir / "runtime.json"),
            )

    def start(self) -> None:
        with self.ledger.locked():
            view = self.view()
            if view.initialized is None:
                raise ControllerError("run is not initialized")
            self._verify_frozen_authority()
            if view.started is not None:
                raise ControllerError("run has already started")
            self.ledger.append("run_started", policy_sha256=self.policy.digest)

    def _append_terminal_if_needed(self) -> dict[str, Any] | None:
        view = self.view()
        if view.terminal is not None:
            return view.terminal
        if view.started is None:
            return None
        reasons: list[str] = []
        details: dict[str, Any] = {}
        if view.elapsed(self.clock()) >= self.policy.max_wall_seconds:
            reasons.append("wall_clock")
        if view.attempt_count >= self.policy.max_attempts:
            reasons.append("attempt_cap")
        convergence = convergence_details(view.scored_attempts, self.policy)
        if convergence is not None:
            reasons.append("convergence")
            details["convergence"] = convergence
        if not reasons:
            return None
        best = view.best
        return self.ledger.append(
            "run_terminal",
            reasons=reasons,
            attempts=view.attempt_count,
            scored_iterations=len(view.scored_attempts),
            best_attempt=int(best["attempt"]) if best else None,
            best_primary=float(best["metrics"]["primary"]) if best else None,
            elapsed_seconds=view.elapsed(self.clock()),
            **details,
        )

    def recover(self) -> None:
        with self.ledger.locked():
            view = self.view()
            self._verify_frozen_authority()
            pending = view.pending_attempt
            if pending is not None:
                self.ledger.append(
                    "attempt_finished",
                    attempt=int(pending["attempt"]),
                    status="controller_interrupted",
                    failure_kind="infrastructure",
                    error="controller restarted before an outcome was durably recorded",
                )
            self._append_terminal_if_needed()

    def submit(self, proposal_dir: Path) -> dict[str, Any]:
        card, original_records, original_digest = _validate_source(proposal_dir)
        with self.ledger.locked():
            view = self.view()
            self._verify_frozen_authority()
            if view.started is None:
                raise ControllerError("run has not started")
            if view.terminal is not None:
                raise ControllerError("run is terminal")
            if view.pending_attempt is not None:
                raise ControllerError("a pending attempt must be recovered before another submission")
            if self._append_terminal_if_needed() is not None:
                raise ControllerError("run reached a terminal condition before submission")

            attempt = view.attempt_count + 1
            attempt_dir = self.run_dir / "artifacts" / f"attempt-{attempt:03d}"
            source_snapshot = attempt_dir / "source"
            snapshot_records, snapshot_digest = _snapshot_source(proposal_dir, source_snapshot)
            if original_records != snapshot_records or original_digest != snapshot_digest:
                raise ControllerError("proposal changed while it was being snapshotted")
            diff_path, diff_digest, diff_excerpt = _write_code_diff(
                self.run_dir, attempt, source_snapshot
            )
            self.ledger.append(
                "attempt_started",
                attempt=attempt,
                card=card,
                source_manifest_sha256=snapshot_digest,
                source_files=snapshot_records,
                code_diff_path=str(diff_path.relative_to(self.run_dir)),
                code_diff_sha256=diff_digest,
                code_diff=diff_excerpt,
            )
            remaining_seconds = max(
                1,
                math.floor(self.policy.max_wall_seconds - self.view().elapsed(self.clock())),
            )

        runner = self.runner or BubblewrapRunner(self.derived_dir)
        output_dir = attempt_dir / "output"
        cache_dir = attempt_dir / "scratch"
        try:
            sandbox_result = runner.run(
                source_dir=source_snapshot,
                output_dir=output_dir,
                cache_dir=cache_dir,
                mode="attempt",
                timeout_seconds=min(
                    self.policy.candidate_timeout_seconds, remaining_seconds
                ),
                max_output_bytes=self.policy.max_candidate_output_bytes,
                checkpoint_dir=None,
                allow_gpu=True,
            )
            outcome = self._score_attempt(attempt, output_dir, sandbox_result)
            if outcome.get("status") == "scored":
                replay_remaining = max(
                    1,
                    math.floor(
                        self.policy.max_wall_seconds - self.view().elapsed(self.clock())
                    ),
                )
                outcome = self._verify_checkpoint_replay(
                    attempt,
                    source_snapshot,
                    output_dir,
                    attempt_dir / "checkpoint-check",
                    attempt_dir / "checkpoint-check-scratch",
                    runner,
                    outcome,
                    replay_remaining,
                )
        except Exception as exc:
            outcome = {
                "attempt": attempt,
                "status": "controller_error",
                "failure_kind": "infrastructure",
                "error": f"{type(exc).__name__}: {exc}",
            }

        with self.ledger.locked():
            if self.view().pending_attempt is None:
                raise ControllerError("attempt outcome was already recorded")
            event = self.ledger.append("attempt_finished", **outcome)
            self._append_terminal_if_needed()
            return event

    def _verify_checkpoint_replay(
        self,
        attempt: int,
        source_dir: Path,
        output_dir: Path,
        replay_dir: Path,
        scratch_dir: Path,
        runner: CandidateRunner,
        outcome: dict[str, Any],
        remaining_seconds: int,
    ) -> dict[str, Any]:
        result = runner.run(
            source_dir=source_dir,
            output_dir=replay_dir,
            cache_dir=scratch_dir,
            mode="checkpoint_check",
            timeout_seconds=min(
                self.policy.candidate_timeout_seconds, remaining_seconds
            ),
            max_output_bytes=self.policy.max_candidate_output_bytes,
            checkpoint_dir=output_dir / "checkpoint",
            allow_gpu=True,
        )
        replay_evidence = {
            "checkpoint_check_status": result.status,
            "checkpoint_check_returncode": result.returncode,
            "checkpoint_check_wall_seconds": result.wall_seconds,
            "checkpoint_check_stdout_tail": result.stdout_tail,
            "checkpoint_check_stderr_tail": result.stderr_tail,
        }
        if result.status != "ok":
            return {
                **{key: value for key, value in outcome.items() if key != "metrics"},
                **replay_evidence,
                "status": "invalid_checkpoint",
                "failure_kind": "candidate",
                "error": "frozen checkpoint could not run validation inference",
            }
        try:
            expected_rows = int(_load_manifest(self.derived_dir)["rows"]["validation"])
            original = _validate_predictions(
                output_dir / "validation_predictions.npy", expected_rows
            )
            replay = _validate_predictions(replay_dir / "test_predictions.npy", expected_rows)
            if not np.allclose(original, replay, rtol=1e-5, atol=1e-6):
                raise ControllerError(
                    "frozen checkpoint does not reproduce validation predictions"
                )
            _make_read_only(replay_dir)
            return {
                **outcome,
                **replay_evidence,
                "checkpoint_replay_predictions_sha256": sha256_file(
                    replay_dir / "test_predictions.npy"
                ),
            }
        except Exception as exc:
            return {
                **{key: value for key, value in outcome.items() if key != "metrics"},
                **replay_evidence,
                "status": "invalid_checkpoint",
                "failure_kind": "candidate",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _score_attempt(
        self, attempt: int, output_dir: Path, sandbox_result: SandboxResult
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "attempt": attempt,
            "sandbox_status": sandbox_result.status,
            "returncode": sandbox_result.returncode,
            "candidate_wall_seconds": sandbox_result.wall_seconds,
            "output_bytes": sandbox_result.output_bytes,
            "stdout_tail": sandbox_result.stdout_tail,
            "stderr_tail": sandbox_result.stderr_tail,
        }
        if sandbox_result.status != "ok":
            return {**base, "status": sandbox_result.status, "failure_kind": "candidate"}
        try:
            expected_rows = int(_load_manifest(self.derived_dir)["rows"]["validation"])
            prediction_path = output_dir / "validation_predictions.npy"
            predictions = _validate_predictions(prediction_path, expected_rows)
            checkpoint_dir = output_dir / "checkpoint"
            if not checkpoint_dir.is_dir() or not any(checkpoint_dir.rglob("*")):
                raise ControllerError("candidate did not produce a checkpoint directory")
            if directory_size(output_dir) > self.policy.max_candidate_output_bytes:
                raise ControllerError("candidate output exceeds the fixed output limit")
            users = np.load(
                self.derived_dir / "private" / "validation_users.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            labels = np.load(
                self.derived_dir / "private" / "validation_labels.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            metrics = evaluate(users, labels, predictions)
            checkpoint_digest = directory_manifest(checkpoint_dir)[1]
            _make_read_only(output_dir)
            return {
                **base,
                "status": "scored",
                "metrics": metrics,
                "validation_predictions_sha256": sha256_file(prediction_path),
                "checkpoint_manifest_sha256": checkpoint_digest,
            }
        except Exception as exc:
            return {
                **base,
                "status": "invalid_output",
                "failure_kind": "candidate",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _load_hidden_test_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Open hidden labels only from the finalization path; never cache them."""

        spec = BenchmarkSpec.load(self.run_dir / "benchmark.json")
        source = self.raw_dir / spec.raw_files["future_log"]
        convert = pacsv.ConvertOptions(include_columns=["user_id", "video_id", "date", "long_view"])
        reader = pacsv.open_csv(
            source,
            read_options=pacsv.ReadOptions(block_size=32 * 1024 * 1024, use_threads=True),
            convert_options=convert,
        )
        users: list[np.ndarray] = []
        videos: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        lo, hi = spec.splits["test"]
        for batch in reader:
            dates = batch.column(batch.schema.get_field_index("date")).to_numpy(zero_copy_only=False)
            mask = (dates >= lo) & (dates <= hi)
            if not mask.any():
                continue
            selected = batch.filter(pa.array(mask))
            users.append(selected.column(0).to_numpy(zero_copy_only=False).astype(np.int64))
            videos.append(selected.column(1).to_numpy(zero_copy_only=False).astype(np.int64))
            labels.append(selected.column(3).to_numpy(zero_copy_only=False).astype(np.int8))
        return np.concatenate(users), np.concatenate(videos), np.concatenate(labels)

    def finalize(self) -> dict[str, Any]:
        with self.ledger.locked():
            view = self.view()
            self._verify_frozen_authority()
            if view.terminal is None:
                raise ControllerError("run is not terminal")
            if view.final_scored is not None:
                raise ControllerError("hidden test has already been scored")
            best = view.best
            if best is None:
                raise ControllerError("no scored checkpoint is available")
            attempt = int(best["attempt"])
            hidden_started = view.hidden_score_started

        attempt_dir = self.run_dir / "artifacts" / f"attempt-{attempt:03d}"
        source_dir = attempt_dir / "source"
        checkpoint_dir = attempt_dir / "output" / "checkpoint"
        start_event = next(
            event for event in view.attempt_starts if int(event["attempt"]) == attempt
        )
        if directory_manifest(source_dir)[1] != start_event["source_manifest_sha256"]:
            raise ControllerError("frozen final source no longer matches its logged hash")
        if directory_manifest(checkpoint_dir)[1] != best["checkpoint_manifest_sha256"]:
            raise ControllerError("frozen final checkpoint no longer matches its logged hash")

        if hidden_started is None:
            with self.ledger.locked():
                self.ledger.append(
                    "final_started",
                    attempt=attempt,
                    source_manifest_sha256=start_event["source_manifest_sha256"],
                    checkpoint_manifest_sha256=best["checkpoint_manifest_sha256"],
                )

            final_number = len(
                [event for event in self.view().events if event["type"] == "final_started"]
            )
            final_output = self.run_dir / "final" / f"try-{final_number:02d}"
            runner = self.runner or BubblewrapRunner(self.derived_dir)
            result = runner.run(
                source_dir=source_dir,
                output_dir=final_output,
                cache_dir=final_output / "scratch",
                mode="final",
                timeout_seconds=self.policy.candidate_timeout_seconds,
                max_output_bytes=self.policy.max_candidate_output_bytes,
                checkpoint_dir=checkpoint_dir,
                allow_gpu=True,
            )
            if result.status != "ok":
                with self.ledger.locked():
                    return self.ledger.append(
                        "final_failed",
                        attempt=attempt,
                        status=result.status,
                        returncode=result.returncode,
                        stderr_tail=result.stderr_tail,
                    )

            expected_rows = int(_load_manifest(self.derived_dir)["rows"]["test"])
            prediction_path = final_output / "test_predictions.npy"
            _validate_predictions(prediction_path, expected_rows)
            prediction_digest = sha256_file(prediction_path)
            with self.ledger.locked():
                # If two finalizers overlap, only the first sealed prediction
                # artifact is authoritative. The other run is discarded.
                current = self.view().hidden_score_started
                if current is None:
                    hidden_started = self.ledger.append(
                        "hidden_score_started",
                        attempt=attempt,
                        prediction_path=str(prediction_path.relative_to(self.run_dir)),
                        test_predictions_sha256=prediction_digest,
                    )
                else:
                    hidden_started = current
                    prediction_path = self.run_dir / hidden_started["prediction_path"]
        else:
            prediction_path = self.run_dir / hidden_started["prediction_path"]

        final_output = prediction_path.parent
        expected_rows = int(_load_manifest(self.derived_dir)["rows"]["test"])
        predictions = _validate_predictions(prediction_path, expected_rows)
        if sha256_file(prediction_path) != hidden_started["test_predictions_sha256"]:
            raise ControllerError("sealed hidden-test predictions no longer match their logged hash")
        users, videos, labels = self._load_hidden_test_targets()
        if len(labels) != expected_rows:
            raise ControllerError("hidden-test target count differs from the frozen manifest")
        metrics = evaluate(users, labels, predictions)
        submission_path = final_output / "submission.csv"
        with submission_path.open("w") as handle:
            handle.write("row_id,user_id,video_id,score\n")
            for row_id, (user, video, score) in enumerate(zip(users, videos, predictions)):
                handle.write(f"{row_id},{int(user)},{int(video)},{float(score):.10g}\n")
        with self.ledger.locked():
            return self.ledger.append(
                "final_scored",
                attempt=attempt,
                metrics=metrics,
                test_predictions_sha256=hidden_started["test_predictions_sha256"],
                submission_sha256=sha256_file(submission_path),
            )

    def status(self) -> dict[str, Any]:
        view = self.view()
        best = view.best
        return {
            "initialized": view.initialized is not None,
            "started": view.started is not None,
            "terminal": view.terminal,
            "attempts": view.attempt_count,
            "scored_iterations": len(view.scored_attempts),
            "pending_attempt": int(view.pending_attempt["attempt"]) if view.pending_attempt else None,
            "best_attempt": int(best["attempt"]) if best else None,
            "best_primary": float(best["metrics"]["primary"]) if best else None,
            "elapsed_seconds": view.elapsed(self.clock()),
            "final_scored": view.final_scored,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal KuaiRand-1K campaign controller")
    parser.add_argument("--run-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--policy", type=Path, required=True)
    subparsers.add_parser("start")
    submit = subparsers.add_parser("submit")
    submit.add_argument("proposal", type=Path)
    subparsers.add_parser("recover")
    subparsers.add_parser("finalize")
    export = subparsers.add_parser("export-jsonl")
    export.add_argument("destination", type=Path)
    subparsers.add_parser("status")
    args = parser.parse_args()

    controller = Controller(args.run_dir)
    if args.command == "init":
        controller.initialize(args.policy)
        result: Any = controller.status()
    elif args.command == "start":
        controller.start()
        result = controller.status()
    elif args.command == "submit":
        result = controller.submit(args.proposal)
    elif args.command == "recover":
        controller.recover()
        result = controller.status()
    elif args.command == "finalize":
        result = controller.finalize()
    elif args.command == "export-jsonl":
        export_jsonl(args.run_dir, args.destination)
        result = {"exported": str(args.destination)}
    else:
        result = controller.status()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
