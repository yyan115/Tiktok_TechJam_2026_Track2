import json

import pytest

from harness.spec import BenchmarkSpec, SpecError


def test_checked_in_benchmark_spec_is_valid() -> None:
    spec = BenchmarkSpec.load()
    assert spec.raw["benchmark_id"] == "kuairand-1k"
    assert spec.raw["official_1k_baseline"] is None
    assert "long_view" not in spec.future_feature_columns


def test_future_outcome_column_is_rejected(tmp_path) -> None:
    spec = BenchmarkSpec.load()
    value = json.loads(json.dumps(spec.raw))
    value["future_feature_columns"].append("long_view")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(SpecError, match="post-impression"):
        BenchmarkSpec.load(path)
