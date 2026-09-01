import json

import polars as pl
import pytest

from harness.eda import EDAError, EDAService


def _public_data(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    pl.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "video_id": [10, 11, 10, 12],
            "date": [20220408, 20220409, 20220408, 20220410],
            "hourmin": [100, 200, 300, 400],
            "time_ms": [1, 2, 3, 4],
            "long_view": [0, 1, 1, 1],
            "is_click": [1, 1, 0, 1],
            "play_time_ms": [10, 20, 30, 40],
            "duration_ms": [100, 100, 200, 200],
        }
    ).write_parquet(public / "train.parquet")
    pl.DataFrame({"row_id": [0, 1], "video_id": [10, 99]}).write_parquet(
        public / "validation_features.parquet"
    )
    pl.DataFrame({"video_id": [10, 11, 12], "author_id": [5, 5, 6]}).write_parquet(
        public / "video_features.parquet"
    )
    pl.DataFrame({"user_id": [1, 2]}).write_parquet(public / "user_features.parquet")
    return public


def test_bounded_eda_returns_data_selected_by_the_researcher(tmp_path) -> None:
    service = EDAService(_public_data(tmp_path))
    result = service.execute(
        {
            "queries": [
                {"id": "overview", "type": "overview"},
                {"id": "history", "type": "user_history"},
                {"id": "cold", "type": "cold_item_rate"},
            ]
        }
    )
    by_id = {row["id"]: row["result"] for row in result["queries"]}
    assert by_id["overview"]["rows"] == 4
    assert by_id["overview"]["long_view_rate"] == 0.75
    assert by_id["history"]["users"] == 2
    assert by_id["cold"]["unseen_item_row_fraction"] == 0.5


def test_eda_rejects_unbounded_or_unknown_queries(tmp_path) -> None:
    service = EDAService(_public_data(tmp_path))
    with pytest.raises(EDAError, match="unsupported"):
        service.execute({"queries": [{"id": "x", "type": "sql"}]})
    with pytest.raises(EDAError, match="outside"):
        service.execute(
            {"queries": [{"id": "x", "type": "top_values", "column": "tab", "top_k": 999}]}
        )
