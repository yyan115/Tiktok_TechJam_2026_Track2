import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from harness.prepare_data import build_derived_data
from harness.spec import BenchmarkSpec


LOG_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
]


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _log_row(user: int, video: int, date: int, target: int) -> list[object]:
    return [
        user,
        video,
        date,
        1200,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        target,
        100,
        200,
        0,
        0,
        0,
        0,
        1,
    ]


def test_derived_view_excludes_future_outcomes_and_hidden_targets(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_csv(
        raw_dir / "log_standard_4_08_to_4_21_1k.csv",
        LOG_COLUMNS,
        [_log_row(1, 10, 20220408, 1), _log_row(1, 11, 20220421, 0)],
    )
    _write_csv(
        raw_dir / "log_standard_4_22_to_5_08_1k.csv",
        LOG_COLUMNS,
        [_log_row(1, 12, 20220422, 1), _log_row(2, 13, 20220429, 1)],
    )
    _write_csv(raw_dir / "user_features_1k.csv", ["user_id", "feature"], [[1, 2]])
    _write_csv(
        raw_dir / "video_features_basic_1k.csv", ["video_id", "author_id"], [[10, 7]]
    )

    base = BenchmarkSpec.load()
    value = json.loads(json.dumps(base.raw))
    value["expected_rows"] = {"train": 2, "validation": 1, "test": 1}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(value))
    output = tmp_path / "derived"
    manifest = build_derived_data(BenchmarkSpec.load(spec_path), raw_dir, output)

    validation = pq.read_table(output / "public" / "validation_features.parquet")
    test = pq.read_table(output / "public" / "test_features.parquet")
    assert validation.column_names == ["row_id", *base.future_feature_columns]
    assert test.column_names == ["row_id", *base.future_feature_columns]
    assert "long_view" not in validation.column_names
    assert "long_view" not in test.column_names
    assert np.load(output / "private" / "validation_labels.npy").tolist() == [1]
    assert not (output / "private" / "test_labels.npy").exists()
    assert manifest["hidden_test_targets_cached"] is False
