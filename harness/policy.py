from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.spec import canonical_json_bytes, sha256_bytes


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class CampaignPolicy:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "CampaignPolicy":
        policy = cls(json.loads(path.read_text()))
        policy.validate()
        return policy

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.raw))

    @property
    def epsilon(self) -> float:
        return float(self.raw["epsilon"])

    @property
    def window(self) -> int:
        return int(self.raw["window_scored_iterations"])

    @property
    def minimum_scored(self) -> int:
        return int(self.raw["minimum_scored_iterations"])

    @property
    def max_attempts(self) -> int:
        return int(self.raw["max_attempts"])

    @property
    def max_wall_seconds(self) -> int:
        return int(self.raw["max_wall_seconds"])

    @property
    def candidate_timeout_seconds(self) -> int:
        return int(self.raw["candidate_timeout_seconds"])

    @property
    def max_candidate_output_bytes(self) -> int:
        return int(self.raw["max_candidate_output_bytes"])

    def validate(self) -> None:
        required = {
            "benchmark_id",
            "epsilon",
            "window_scored_iterations",
            "minimum_scored_iterations",
            "max_attempts",
            "max_wall_seconds",
            "candidate_timeout_seconds",
            "max_candidate_output_bytes",
            "selection",
            "tie_break",
            "failed_attempts_advance_convergence",
        }
        missing = sorted(required.difference(self.raw))
        if missing:
            raise PolicyError(f"missing campaign policy fields: {missing}")
        if self.raw["benchmark_id"] != "kuairand-1k":
            raise PolicyError("campaign benchmark_id must be kuairand-1k")
        for field in ("epsilon", "window_scored_iterations", "minimum_scored_iterations"):
            if self.raw[field] is None:
                raise PolicyError(f"{field} must be fixed before the run")
        if not (0.0 <= self.epsilon < 1.0):
            raise PolicyError("epsilon must be in [0, 1)")
        if self.window < 1:
            raise PolicyError("window_scored_iterations must be positive")
        if self.minimum_scored < self.window + 1:
            raise PolicyError("minimum_scored_iterations must be at least window + 1")
        if not (1 <= self.max_attempts <= 50):
            raise PolicyError("max_attempts must be between 1 and the hard cap of 50")
        if not (1 <= self.max_wall_seconds <= 21600):
            raise PolicyError("max_wall_seconds must not exceed six hours")
        if not (1 <= self.candidate_timeout_seconds <= self.max_wall_seconds):
            raise PolicyError("candidate timeout must fit inside the wall-clock cap")
        if self.max_candidate_output_bytes < 1:
            raise PolicyError("max_candidate_output_bytes must be positive")
        if self.raw["failed_attempts_advance_convergence"] is not False:
            raise PolicyError("failed attempts cannot advance or reset convergence")
        if self.raw["selection"] != "earliest validation-best checkpoint at terminal":
            raise PolicyError("final selection rule cannot be changed")
        if self.raw["tie_break"] != "earliest attempt":
            raise PolicyError("tie break must select the earliest attempt")


def convergence_details(
    scored_attempts: list[dict[str, Any]], policy: CampaignPolicy
) -> dict[str, Any] | None:
    """Return convergence evidence using only scored attempts.

    Failures are absent from ``scored_attempts`` by construction, so they count
    against wall time and the attempt cap without advancing or resetting this window.
    """

    if len(scored_attempts) < policy.minimum_scored or len(scored_attempts) <= policy.window:
        return None
    scores = [float(attempt["metrics"]["primary"]) for attempt in scored_attempts]
    prior_best = max(scores[: -policy.window])
    window_best = max(scores[-policy.window :])
    improvement = window_best - prior_best
    if improvement > policy.epsilon:
        return None
    return {
        "epsilon": policy.epsilon,
        "window_scored_iterations": policy.window,
        "minimum_scored_iterations": policy.minimum_scored,
        "scored_iterations": len(scores),
        "prior_best": prior_best,
        "window_best": window_best,
        "improvement": improvement,
        "window_attempts": [attempt["attempt"] for attempt in scored_attempts[-policy.window :]],
    }
