from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = ROOT / "config" / "benchmark.json"
DEFAULT_RAW_DIR = ROOT / "datasets" / "KuaiRand-1K" / "data"
DEFAULT_DERIVED_DIR = ROOT / "datasets" / "derived_1k"


class SpecError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkSpec:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path = DEFAULT_SPEC_PATH) -> "BenchmarkSpec":
        value = json.loads(path.read_text())
        spec = cls(raw=value, path=path.resolve())
        spec.validate()
        return spec

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.raw))

    @property
    def splits(self) -> dict[str, tuple[int, int]]:
        return {name: tuple(bounds) for name, bounds in self.raw["splits"].items()}

    @property
    def raw_files(self) -> dict[str, str]:
        return dict(self.raw["raw_files"])

    @property
    def expected_rows(self) -> dict[str, int]:
        return {name: int(count) for name, count in self.raw["expected_rows"].items()}

    @property
    def future_feature_columns(self) -> tuple[str, ...]:
        return tuple(self.raw["future_feature_columns"])

    def validate(self) -> None:
        if self.raw.get("benchmark_id") != "kuairand-1k":
            raise SpecError("benchmark_id must be kuairand-1k")
        if self.raw.get("label") != "long_view":
            raise SpecError("label must be long_view")
        if self.raw.get("metrics") != ["GAUC", "nDCG@5"]:
            raise SpecError("metrics must be GAUC and nDCG@5")
        expected_splits = {
            "train": [20220408, 20220421],
            "validation": [20220422, 20220428],
            "test": [20220429, 20220508],
        }
        if self.raw.get("splits") != expected_splits:
            raise SpecError("date splits differ from the supplied benchmark contract")
        limits = self.raw.get("hard_limits", {})
        if limits.get("attempts") != 50 or limits.get("wall_seconds") != 21600:
            raise SpecError("hard limits must remain 50 attempts and six hours")
        if limits.get("failed_attempts_advance_convergence") is not False:
            raise SpecError("failed attempts must not advance convergence")
        required = {"train_log", "future_log", "user_features", "video_features"}
        if set(self.raw_files) != required:
            raise SpecError("raw_files must contain only the four approved inputs")
        forbidden_future = {
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "long_view",
            "play_time_ms",
            "profile_stay_time",
            "comment_stay_time",
            "is_profile_enter",
        }
        overlap = forbidden_future.intersection(self.future_feature_columns)
        if overlap:
            raise SpecError(f"post-impression columns exposed in future data: {sorted(overlap)}")

    def validate_raw_files(self, raw_dir: Path = DEFAULT_RAW_DIR) -> None:
        missing = [name for name in self.raw_files.values() if not (raw_dir / name).is_file()]
        if missing:
            raise SpecError(f"missing raw benchmark files: {missing}")
