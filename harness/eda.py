from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from harness.spec import DEFAULT_DERIVED_DIR, canonical_json_bytes


class EDAError(ValueError):
    pass


MAX_QUERIES = 20
MAX_TOP_K = 30
MAX_SAMPLE_ROWS = 50
TRAIN_GROUP_COLUMNS = {"date", "tab", "hourmin", "user_id", "video_id"}
NUMERIC_COLUMNS = {
    "date",
    "hourmin",
    "time_ms",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
}
FEEDBACK_COLUMNS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return _clean(frame.to_dicts())


class EDAService:
    """Bounded, read-only EDA over training labels and label-free future features."""

    def __init__(self, public_dir: Path = DEFAULT_DERIVED_DIR / "public") -> None:
        self.public_dir = public_dir.resolve()
        self.train_path = self.public_dir / "train.parquet"
        self.validation_path = self.public_dir / "validation_features.parquet"
        self.video_path = self.public_dir / "video_features.parquet"
        self.user_path = self.public_dir / "user_features.parquet"
        for path in (self.train_path, self.validation_path, self.video_path, self.user_path):
            if not path.is_file():
                raise EDAError(f"EDA input is missing: {path}")

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        queries = request.get("queries")
        if not isinstance(queries, list) or not (1 <= len(queries) <= MAX_QUERIES):
            raise EDAError(f"queries must contain between 1 and {MAX_QUERIES} entries")
        ids: set[str] = set()
        results = []
        for query in queries:
            if not isinstance(query, dict):
                raise EDAError("each EDA query must be an object")
            query_id = query.get("id")
            if not isinstance(query_id, str) or not query_id.strip() or query_id in ids:
                raise EDAError("EDA query ids must be unique non-empty strings")
            ids.add(query_id)
            started = time.monotonic()
            value = self._execute_one(query)
            results.append(
                {
                    "id": query_id,
                    "type": query.get("type"),
                    "result": _clean(value),
                    "wall_seconds": time.monotonic() - started,
                }
            )
        return {"queries": results}

    def _execute_one(self, query: dict[str, Any]) -> Any:
        query_type = query.get("type")
        if query_type == "schema":
            return {
                name: str(field.type)
                for name, field in zip(
                    pq.read_schema(self.train_path).names, pq.read_schema(self.train_path)
                )
            }
        if query_type == "overview":
            frame = (
                pl.scan_parquet(self.train_path)
                .select(
                    pl.len().alias("rows"),
                    pl.col("user_id").n_unique().alias("users"),
                    pl.col("video_id").n_unique().alias("videos"),
                    pl.col("date").min().alias("first_date"),
                    pl.col("date").max().alias("last_date"),
                    pl.col("long_view").mean().alias("long_view_rate"),
                )
                .collect()
            )
            return _records(frame)[0]
        if query_type == "label_by_date":
            frame = (
                pl.scan_parquet(self.train_path)
                .group_by("date")
                .agg(pl.len().alias("rows"), pl.col("long_view").mean().alias("long_view_rate"))
                .sort("date")
                .collect()
            )
            return _records(frame)
        if query_type == "user_history":
            grouped = (
                pl.scan_parquet(self.train_path)
                .group_by("user_id")
                .agg(pl.len().alias("rows"), pl.col("long_view").mean().alias("long_view_rate"))
                .collect()
            )
            return {
                "users": grouped.height,
                "rows_quantiles": self._quantiles(grouped["rows"]),
                "long_view_rate_quantiles": self._quantiles(grouped["long_view_rate"]),
            }
        if query_type == "item_frequency":
            grouped = (
                pl.scan_parquet(self.train_path)
                .group_by("video_id")
                .agg(pl.len().alias("rows"))
                .collect()
            )
            return {
                "items": grouped.height,
                "frequency_quantiles": self._quantiles(grouped["rows"]),
                "singleton_fraction": float((grouped["rows"] == 1).mean()),
            }
        if query_type == "feedback_correlations":
            schema = set(pq.read_schema(self.train_path).names)
            columns = [column for column in FEEDBACK_COLUMNS if column in schema]
            expressions = [pl.corr(column, "long_view").alias(column) for column in columns]
            frame = pl.scan_parquet(self.train_path).select(expressions).collect()
            return _records(frame)[0]
        if query_type == "cardinality":
            columns = query.get("columns")
            schema = set(pq.read_schema(self.train_path).names)
            if not isinstance(columns, list) or not (1 <= len(columns) <= 10):
                raise EDAError("cardinality requires 1-10 columns")
            if any(column not in schema for column in columns):
                raise EDAError("cardinality requested an unknown column")
            frame = (
                pl.scan_parquet(self.train_path)
                .select([pl.col(column).n_unique().alias(column) for column in columns])
                .collect()
            )
            return _records(frame)[0]
        if query_type == "numeric_quantiles":
            column = query.get("column")
            if column not in NUMERIC_COLUMNS:
                raise EDAError("numeric_quantiles requested a disallowed column")
            series = pl.read_parquet(self.train_path, columns=[column])[column]
            return {"column": column, "quantiles": self._quantiles(series)}
        if query_type == "top_values":
            column = query.get("column")
            top_k = int(query.get("top_k", 10))
            if column not in TRAIN_GROUP_COLUMNS or not (1 <= top_k <= MAX_TOP_K):
                raise EDAError("top_values request is outside its bounds")
            frame = (
                pl.scan_parquet(self.train_path)
                .group_by(column)
                .agg(pl.len().alias("rows"), pl.col("long_view").mean().alias("long_view_rate"))
                .sort("rows", descending=True)
                .head(top_k)
                .collect()
            )
            return _records(frame)
        if query_type == "cold_item_rate":
            train_items = (
                pl.scan_parquet(self.train_path)
                .select("video_id")
                .unique()
                .with_columns(pl.lit(True).alias("seen_in_train"))
            )
            validation = pl.scan_parquet(self.validation_path).select("video_id")
            frame = (
                validation.join(train_items, on="video_id", how="left")
                .select(
                    pl.len().alias("rows"),
                    pl.col("seen_in_train").is_null().mean().alias("unseen_item_row_fraction"),
                    pl.col("video_id").n_unique().alias("unique_items"),
                )
                .collect()
            )
            return _records(frame)[0]
        if query_type == "metadata_overview":
            video = pl.scan_parquet(self.video_path)
            schema = set(pq.read_schema(self.video_path).names)
            unique_columns = [
                column for column in ("video_id", "author_id", "music_id", "video_type", "music_type")
                if column in schema
            ]
            expressions = [pl.len().alias("rows")]
            expressions.extend(pl.col(column).n_unique().alias(f"unique_{column}") for column in unique_columns)
            frame = video.select(expressions).collect()
            return {"summary": _records(frame)[0], "columns": sorted(schema)}
        if query_type == "sample_train":
            rows = int(query.get("rows", 10))
            if not (1 <= rows <= MAX_SAMPLE_ROWS):
                raise EDAError(f"sample_train rows must be 1-{MAX_SAMPLE_ROWS}")
            return _records(pl.read_parquet(self.train_path).head(rows))
        raise EDAError(f"unsupported EDA query type: {query_type}")

    @staticmethod
    def _quantiles(series: pl.Series) -> dict[str, float | int | None]:
        values: dict[str, float | int | None] = {
            "min": series.min(),
            "p10": series.quantile(0.10),
            "p25": series.quantile(0.25),
            "median": series.quantile(0.50),
            "p75": series.quantile(0.75),
            "p90": series.quantile(0.90),
            "p99": series.quantile(0.99),
            "max": series.max(),
        }
        return _clean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute bounded training-only EDA requests")
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_DERIVED_DIR / "public")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    result = EDAService(args.public_dir).execute(request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))
    print(json.dumps({"status": "complete", "queries": len(result["queries"])}, sort_keys=True))


if __name__ == "__main__":
    main()
