#!/usr/bin/env python3
"""Conservative rank ensemble with ordered causal multi-step session history."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy.stats import rankdata


VERSION = 13
PARENT_ATTEMPTS = (11,)
SEED = 20260901
HOLDOUT_DAYS = 3
MAX_BOOST_ROUNDS = 240
MIN_SELECTION_ITERATION = 20
ITERATION_SELECTION_INTERVAL = 10
ITERATION_GRID = tuple(
    range(
        MIN_SELECTION_ITERATION,
        MAX_BOOST_ROUNDS + 1,
        ITERATION_SELECTION_INTERVAL,
    )
)
SESSION_GAP_MS = 30 * 60 * 1000
LONG_VIEW_THRESHOLD_MS = 18_000
RECENT_PROXY_WINDOW = 4
BASE_MODEL_FILE = "attempt7_base_model.txt"
PROXY_MODEL_FILE = "attempt8_proxy_model.txt"
PERSONALIZED_MODEL_FILE = "target_free_user_context_model.txt"
JOINT_PERSONALIZED_MODEL_FILE = "ordered_multistep_session_model.txt"
BASE_MEMBER_NAME = "attempt7_context_session"
PROXY_MEMBER_NAME = "attempt8_lagged_engagement"
PERSONALIZED_MEMBER_NAME = "target_free_user_context"
JOINT_PERSONALIZED_MEMBER_NAME = "ordered_multistep_session_history"
USER_VOCAB_FILE = "training_user_ids.npy"
ATTEMPT9_BASE_BLEND_WEIGHT = 0.25
BASE_BLEND_WEIGHT_GRID = (
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
)
ENSEMBLE_WEIGHT_UNITS = 8
MIN_LOCAL_PERSONALIZATION_GAIN = 0.0005
MIN_LOCAL_JOINT_GAIN = 0.0005
TAB_TRANSITION_STRIDE = 128
MAX_PROXY_STREAK = 8
MANIFEST_FILE = "manifest.json"
SCHEMA_AUDIT_FILE = "inference_schema.json"
DIAGNOSTICS_FILE = "training_diagnostics.json"

TRAIN_REQUIRED = (
    "date",
    "duration_ms",
    "hourmin",
    "is_rand",
    "tab",
    "time_ms",
    "user_id",
    "long_view",
)
EVALUATION_REQUIRED = (
    "date",
    "duration_ms",
    "hourmin",
    "is_rand",
    "tab",
    "time_ms",
    "user_id",
)

DURATION_EDGES_MS = np.asarray(
    [
        0,
        3_000,
        5_000,
        7_000,
        10_000,
        15_000,
        18_000,
        30_000,
        60_000,
        120_000,
        300_000,
        600_000,
        1_200_000,
    ],
    dtype=np.int64,
)
PREVIOUS_GAP_EDGES_MS = np.asarray(
    [
        1_000,
        3_000,
        10_000,
        30_000,
        60_000,
        5 * 60_000,
        SESSION_GAP_MS,
        2 * 60 * 60_000,
        24 * 60 * 60_000,
    ],
    dtype=np.int64,
)
SESSION_POSITION_EDGES = np.asarray(
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=np.int64
)

# The first 24 features exactly preserve attempt 7. The final six describe
# only already-completed predecessor events; no future row is consulted.
FEATURE_NAMES = [
    "tab_code",
    "is_rand_code",
    "hour_code",
    "weekday_code",
    "daypart_code",
    "duration_bucket_code",
    "log_duration_ms",
    "sqrt_duration_seconds",
    "duration_missing",
    "duration_zero",
    "duration_at_least_18s",
    "duration_at_least_60s",
    "duration_outlier",
    "minute_fraction",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "previous_gap_bucket_code",
    "session_position_bucket_code",
    "log_previous_gap_ms",
    "previous_gap_missing",
    "session_start",
    "log_session_position",
    "previous_long_view_proxy_code",
    "previous_duration_bucket_code",
    "signed_log_previous_threshold_margin_ms",
    "session_prior_proxy_positive_rate",
    "recent4_prior_proxy_positive_rate",
    "log_prior_proxy_observations",
]
PERSONALIZED_FEATURE_NAMES = FEATURE_NAMES + [
    "user_code",
    "user_tab_context_code",
    "user_duration_context_code",
    "user_hour_context_code",
    "user_session_position_context_code",
    "user_previous_proxy_state_context_code",
]
# These six additions refine attempt 11 without labels, item IDs, metadata, or
# any future row. They retain order and transition information that the existing
# recent-four proxy mean deliberately discards.
JOINT_PERSONALIZED_FEATURE_NAMES = PERSONALIZED_FEATURE_NAMES + [
    "previous_tab_code",
    "tab_transition_code",
    "recent4_proxy_pattern_code",
    "signed_proxy_streak_code",
    "log_session_elapsed_ms",
    "recent4_mean_log_duration_ms",
]
CATEGORICAL_FEATURE_NAMES = [
    "tab_code",
    "is_rand_code",
    "hour_code",
    "weekday_code",
    "daypart_code",
    "duration_bucket_code",
    "previous_gap_bucket_code",
    "session_position_bucket_code",
    "previous_long_view_proxy_code",
    "previous_duration_bucket_code",
]
PERSONALIZED_CATEGORICAL_FEATURE_NAMES = CATEGORICAL_FEATURE_NAMES + [
    "user_code",
    "user_tab_context_code",
    "user_duration_context_code",
    "user_hour_context_code",
    "user_session_position_context_code",
    "user_previous_proxy_state_context_code",
]
JOINT_PERSONALIZED_CATEGORICAL_FEATURE_NAMES = (
    PERSONALIZED_CATEGORICAL_FEATURE_NAMES
    + [
        "previous_tab_code",
        "tab_transition_code",
        "recent4_proxy_pattern_code",
        "signed_proxy_streak_code",
    ]
)
JOINT_HISTORY_CONTRACT = {
    "tab_transition_stride": TAB_TRANSITION_STRIDE,
    "proxy_pattern_base": 3,
    "proxy_pattern_window": RECENT_PROXY_WINDOW,
    "maximum_proxy_streak": MAX_PROXY_STREAK,
    "recent_duration_window": RECENT_PROXY_WINDOW,
}
BASE_FEATURE_NAMES = FEATURE_NAMES[:24]
BASE_CATEGORICAL_FEATURE_NAMES = CATEGORICAL_FEATURE_NAMES[:8]

if ITERATION_GRID[-1] != MAX_BOOST_ROUNDS:
    raise RuntimeError("iteration selection grid must include the round ceiling")
if ATTEMPT9_BASE_BLEND_WEIGHT not in BASE_BLEND_WEIGHT_GRID:
    raise RuntimeError("blend grid must include attempt 9's frozen base weight")
if ENSEMBLE_WEIGHT_UNITS <= 0:
    raise RuntimeError("ensemble weight denominator must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("attempt", "final"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="/tmp")
    parser.add_argument("--checkpoint-dir")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parquet_schema_audit(path: Path, required: Iterable[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required parquet file not found: {path}")
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    names = schema.names
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(
            f"{path.name} lacks required column(s) {missing}; "
            f"available columns are {names}"
        )

    fields: dict[str, Any] = {}
    for name in required:
        index = names.index(name)
        null_count = 0
        null_count_known = True
        for row_group_index in range(parquet_file.num_row_groups):
            statistics = (
                parquet_file.metadata.row_group(row_group_index)
                .column(index)
                .statistics
            )
            if statistics is None or statistics.null_count is None:
                null_count_known = False
                break
            null_count += int(statistics.null_count)
        fields[name] = {
            "dtype": str(schema.field(name).type),
            "null_count_from_statistics": (
                null_count if null_count_known else None
            ),
        }
    return {
        "path_name": path.name,
        "rows": int(parquet_file.metadata.num_rows),
        "all_columns": names,
        "required_fields": fields,
    }


def inference_preflight(data_root: Path) -> dict[str, Any]:
    return {
        "evaluation_features": parquet_schema_audit(
            data_root / "evaluation_features.parquet", EVALUATION_REQUIRED
        )
    }


def grouped_cumulative_counts(
    values: np.ndarray,
    session_start: np.ndarray,
) -> np.ndarray:
    """Cumulative sum within each causal session, including the current value."""
    values64 = np.asarray(values, dtype=np.int64)
    cumulative = np.cumsum(values64, dtype=np.int64)
    totals_before_start = np.where(
        session_start, cumulative - values64, 0
    )
    baseline = np.maximum.accumulate(totals_before_start)
    return cumulative - baseline


def add_backward_sequence_features(events: pl.DataFrame) -> pl.DataFrame:
    """Build features available when the current impression begins."""
    rows = events.height
    if rows == 0:
        return events.with_columns(
            pl.Series("previous_gap_ms", np.empty(0, dtype=np.int64)),
            pl.Series("previous_gap_bucket", np.empty(0, dtype=np.int16)),
            pl.Series("previous_gap_missing", np.empty(0, dtype=np.int8)),
            pl.Series("session_start", np.empty(0, dtype=np.int8)),
            pl.Series("session_position", np.empty(0, dtype=np.int32)),
            pl.Series("session_position_bucket", np.empty(0, dtype=np.int16)),
            pl.Series("previous_duration_ms", np.empty(0, dtype=np.int64)),
            pl.Series(
                "previous_duration_bucket", np.empty(0, dtype=np.int16)
            ),
            pl.Series(
                "previous_long_view_proxy_code", np.empty(0, dtype=np.int8)
            ),
            pl.Series(
                "previous_threshold_margin_ms", np.empty(0, dtype=np.int64)
            ),
            pl.Series(
                "session_prior_proxy_positive_rate",
                np.empty(0, dtype=np.float32),
            ),
            pl.Series(
                "recent4_prior_proxy_positive_rate",
                np.empty(0, dtype=np.float32),
            ),
            pl.Series(
                "prior_proxy_observation_count", np.empty(0, dtype=np.int32)
            ),
            pl.Series("previous_tab_code", np.empty(0, dtype=np.int16)),
            pl.Series("tab_transition_code", np.empty(0, dtype=np.int32)),
            pl.Series(
                "recent4_proxy_pattern_code", np.empty(0, dtype=np.int16)
            ),
            pl.Series(
                "signed_proxy_streak_code", np.empty(0, dtype=np.int16)
            ),
            pl.Series("session_elapsed_ms", np.empty(0, dtype=np.int64)),
            pl.Series(
                "recent4_mean_log_duration_ms",
                np.empty(0, dtype=np.float32),
            ),
        )

    # Physical order breaks equal-timestamp ties. For a feature on row i, this
    # function uses row i and rows strictly preceding i in this timeline only.
    timeline = events.select(
        ["__row_order", "user_id", "time_ms", "duration_ms", "tab"]
    ).sort(["user_id", "time_ms", "__row_order"])
    users = timeline.get_column("user_id").to_numpy().astype(
        np.int64, copy=False
    )
    times = timeline.get_column("time_ms").to_numpy().astype(
        np.int64, copy=False
    )
    durations = timeline.get_column("duration_ms").to_numpy().astype(
        np.int64, copy=False
    )
    tabs = timeline.get_column("tab").to_numpy().astype(np.int64, copy=False)

    same_user = np.zeros(rows, dtype=bool)
    if rows > 1:
        same_user[1:] = users[1:] == users[:-1]
    valid_transition = np.zeros(rows, dtype=bool)
    if rows > 1:
        valid_transition[1:] = (
            same_user[1:]
            & (times[1:] >= 0)
            & (times[:-1] >= 0)
            & (times[1:] >= times[:-1])
        )

    previous_gap_ms = np.full(rows, -1, dtype=np.int64)
    if rows > 1:
        raw_deltas = times[1:] - times[:-1]
        previous_gap_ms[1:] = np.where(
            valid_transition[1:], raw_deltas, -1
        )

    same_session_transition = valid_transition & (
        previous_gap_ms <= SESSION_GAP_MS
    )
    session_start = ~same_session_transition
    row_numbers = np.arange(rows, dtype=np.int64)
    start_rows = np.where(session_start, row_numbers, -1)
    most_recent_start = np.maximum.accumulate(start_rows)
    session_position = (row_numbers - most_recent_start).astype(np.int32)

    session_start_times = times[most_recent_start]
    valid_session_elapsed = (
        (times >= 0)
        & (session_start_times >= 0)
        & (times >= session_start_times)
    )
    session_elapsed_ms = np.zeros(rows, dtype=np.int64)
    session_elapsed_ms[valid_session_elapsed] = (
        times[valid_session_elapsed] - session_start_times[valid_session_elapsed]
    )

    current_tab_codes = np.zeros(rows, dtype=np.int16)
    valid_current_tab = tabs >= 0
    if np.any(valid_current_tab):
        maximum_tab_code = int(tabs[valid_current_tab].max()) + 1
        if maximum_tab_code >= TAB_TRANSITION_STRIDE:
            raise RuntimeError("tab code exceeds frozen transition stride")
        current_tab_codes[valid_current_tab] = (
            tabs[valid_current_tab] + 1
        ).astype(np.int16)
    previous_tabs = np.full(rows, -1, dtype=np.int64)
    if rows > 1:
        previous_tabs[1:] = np.where(
            same_session_transition[1:], tabs[:-1], -1
        )
    previous_tab_codes = np.zeros(rows, dtype=np.int16)
    valid_previous_tab = previous_tabs >= 0
    previous_tab_codes[valid_previous_tab] = (
        previous_tabs[valid_previous_tab] + 1
    ).astype(np.int16)
    tab_transition_codes = np.zeros(rows, dtype=np.int32)
    valid_tab_transition = (
        same_session_transition & valid_previous_tab & valid_current_tab
    )
    tab_transition_codes[valid_tab_transition] = (
        previous_tab_codes[valid_tab_transition].astype(np.int32)
        * TAB_TRANSITION_STRIDE
        + current_tab_codes[valid_tab_transition].astype(np.int32)
    )

    previous_gap_bucket = np.zeros(rows, dtype=np.int16)
    valid_gap = previous_gap_ms >= 0
    previous_gap_bucket[valid_gap] = (
        np.searchsorted(
            PREVIOUS_GAP_EDGES_MS,
            previous_gap_ms[valid_gap],
            side="right",
        ).astype(np.int16)
        + 1
    )
    session_position_bucket = np.searchsorted(
        SESSION_POSITION_EDGES,
        session_position.astype(np.int64),
        side="right",
    ).astype(np.int16)

    previous_duration_ms = np.full(rows, -1, dtype=np.int64)
    if rows > 1:
        previous_duration_ms[1:] = np.where(
            same_session_transition[1:], durations[:-1], -1
        )
    previous_duration_available = (
        same_session_transition & (previous_duration_ms >= 0)
    )
    previous_duration_bucket = np.zeros(rows, dtype=np.int16)
    previous_duration_bucket[previous_duration_available] = (
        np.searchsorted(
            DURATION_EDGES_MS,
            previous_duration_ms[previous_duration_available],
            side="right",
        ).astype(np.int16)
        + 1
    )

    # Zero-duration rows are excluded because their documented threshold is
    # degenerate and prior feedback identified them as an artifact risk.
    proxy_valid = same_session_transition & (previous_duration_ms > 0)
    previous_threshold_margin_ms = np.zeros(rows, dtype=np.int64)
    if np.any(proxy_valid):
        thresholds = np.minimum(
            previous_duration_ms[proxy_valid], LONG_VIEW_THRESHOLD_MS
        )
        previous_threshold_margin_ms[proxy_valid] = (
            previous_gap_ms[proxy_valid] - thresholds
        )
    proxy_positive = proxy_valid & (previous_threshold_margin_ms >= 0)
    previous_long_view_proxy_code = np.zeros(rows, dtype=np.int8)
    previous_long_view_proxy_code[proxy_valid] = 1
    previous_long_view_proxy_code[proxy_positive] = 2

    proxy_observations = grouped_cumulative_counts(
        proxy_valid.astype(np.int8), session_start
    ).astype(np.int32)
    proxy_positives = grouped_cumulative_counts(
        proxy_positive.astype(np.int8), session_start
    ).astype(np.int32)
    session_proxy_rate = np.full(rows, 0.5, dtype=np.float32)
    observed = proxy_observations > 0
    session_proxy_rate[observed] = (
        proxy_positives[observed] / proxy_observations[observed]
    ).astype(np.float32)

    recent_count = np.zeros(rows, dtype=np.int16)
    recent_positive = np.zeros(rows, dtype=np.int16)
    recent_proxy_pattern = np.zeros(rows, dtype=np.int16)
    recent_log_duration_sum = np.zeros(rows, dtype=np.float64)
    recent_duration_count = np.zeros(rows, dtype=np.int16)
    for lag in range(RECENT_PROXY_WINDOW):
        shifted_valid = np.zeros(rows, dtype=bool)
        shifted_positive = np.zeros(rows, dtype=bool)
        shifted_proxy_code = np.zeros(rows, dtype=np.int8)
        shifted_duration = np.full(rows, -1, dtype=np.int64)
        if lag == 0:
            shifted_valid = proxy_valid
            shifted_positive = proxy_positive
            shifted_proxy_code = previous_long_view_proxy_code
            shifted_duration = previous_duration_ms
        else:
            shifted_valid[lag:] = proxy_valid[:-lag]
            shifted_positive[lag:] = proxy_positive[:-lag]
            shifted_proxy_code[lag:] = previous_long_view_proxy_code[:-lag]
            shifted_duration[lag:] = previous_duration_ms[:-lag]
        within_session = session_position > lag
        usable = within_session & shifted_valid
        recent_count += usable.astype(np.int16)
        recent_positive += (usable & shifted_positive).astype(np.int16)
        recent_proxy_pattern += (
            np.where(within_session, shifted_proxy_code, 0).astype(np.int16)
            * (3**lag)
        )
        duration_usable = within_session & (shifted_duration > 0)
        recent_duration_count += duration_usable.astype(np.int16)
        recent_log_duration_sum += np.where(
            duration_usable,
            np.log1p(
                np.minimum(np.maximum(shifted_duration, 0), 3_600_000).astype(
                    np.float64
                )
            ),
            0.0,
        )
    recent_proxy_rate = np.full(rows, 0.5, dtype=np.float32)
    recent_observed = recent_count > 0
    recent_proxy_rate[recent_observed] = (
        recent_positive[recent_observed] / recent_count[recent_observed]
    ).astype(np.float32)
    recent_mean_log_duration = np.zeros(rows, dtype=np.float32)
    recent_duration_observed = recent_duration_count > 0
    recent_mean_log_duration[recent_duration_observed] = (
        recent_log_duration_sum[recent_duration_observed]
        / recent_duration_count[recent_duration_observed]
    ).astype(np.float32)

    first_proxy_state = previous_long_view_proxy_code
    streak_length = np.zeros(rows, dtype=np.int16)
    active_streak = first_proxy_state > 0
    for lag in range(MAX_PROXY_STREAK):
        shifted_state = np.zeros(rows, dtype=np.int8)
        if lag == 0:
            shifted_state = first_proxy_state
        else:
            shifted_state[lag:] = first_proxy_state[:-lag]
        matches = (
            active_streak
            & (session_position > lag)
            & (shifted_state == first_proxy_state)
        )
        streak_length += matches.astype(np.int16)
        active_streak = matches
    signed_streak = np.where(
        first_proxy_state == 2,
        streak_length,
        np.where(first_proxy_state == 1, -streak_length, 0),
    ).astype(np.int16)
    signed_proxy_streak_code = (
        signed_streak + MAX_PROXY_STREAK
    ).astype(np.int16)
    if recent_proxy_pattern.size and (
        int(recent_proxy_pattern.min()) < 0
        or int(recent_proxy_pattern.max()) >= 3**RECENT_PROXY_WINDOW
    ):
        raise RuntimeError("ordered proxy pattern exceeds its frozen base-3 range")
    if signed_proxy_streak_code.size and (
        int(signed_proxy_streak_code.min()) < 0
        or int(signed_proxy_streak_code.max()) > 2 * MAX_PROXY_STREAK
    ):
        raise RuntimeError("signed proxy streak exceeds its frozen range")

    timeline = timeline.with_columns(
        pl.Series("previous_gap_ms", previous_gap_ms),
        pl.Series("previous_gap_bucket", previous_gap_bucket),
        pl.Series("previous_gap_missing", (~valid_gap).astype(np.int8)),
        pl.Series("session_start", session_start.astype(np.int8)),
        pl.Series("session_position", session_position),
        pl.Series("session_position_bucket", session_position_bucket),
        pl.Series("previous_duration_ms", previous_duration_ms),
        pl.Series("previous_duration_bucket", previous_duration_bucket),
        pl.Series(
            "previous_long_view_proxy_code", previous_long_view_proxy_code
        ),
        pl.Series(
            "previous_threshold_margin_ms", previous_threshold_margin_ms
        ),
        pl.Series(
            "session_prior_proxy_positive_rate", session_proxy_rate
        ),
        pl.Series(
            "recent4_prior_proxy_positive_rate", recent_proxy_rate
        ),
        pl.Series(
            "prior_proxy_observation_count", proxy_observations
        ),
        pl.Series("previous_tab_code", previous_tab_codes),
        pl.Series("tab_transition_code", tab_transition_codes),
        pl.Series("recent4_proxy_pattern_code", recent_proxy_pattern),
        pl.Series("signed_proxy_streak_code", signed_proxy_streak_code),
        pl.Series("session_elapsed_ms", session_elapsed_ms),
        pl.Series(
            "recent4_mean_log_duration_ms", recent_mean_log_duration
        ),
    ).sort("__row_order")

    restored_order = timeline.get_column("__row_order").to_numpy()
    source_order = events.get_column("__row_order").to_numpy()
    if not np.array_equal(restored_order, source_order):
        raise RuntimeError(
            "sequence transform lost, duplicated, or reordered source rows"
        )

    return events.with_columns(
        timeline.get_column("previous_gap_ms"),
        timeline.get_column("previous_gap_bucket"),
        timeline.get_column("previous_gap_missing"),
        timeline.get_column("session_start"),
        timeline.get_column("session_position"),
        timeline.get_column("session_position_bucket"),
        timeline.get_column("previous_duration_ms"),
        timeline.get_column("previous_duration_bucket"),
        timeline.get_column("previous_long_view_proxy_code"),
        timeline.get_column("previous_threshold_margin_ms"),
        timeline.get_column("session_prior_proxy_positive_rate"),
        timeline.get_column("recent4_prior_proxy_positive_rate"),
        timeline.get_column("prior_proxy_observation_count"),
        timeline.get_column("previous_tab_code"),
        timeline.get_column("tab_transition_code"),
        timeline.get_column("recent4_proxy_pattern_code"),
        timeline.get_column("signed_proxy_streak_code"),
        timeline.get_column("session_elapsed_ms"),
        timeline.get_column("recent4_mean_log_duration_ms"),
    )


def prepare_events(path: Path, require_label: bool) -> pl.DataFrame:
    columns = list(EVALUATION_REQUIRED)
    if require_label:
        columns.append("long_view")
    events = pl.read_parquet(path, columns=columns).with_row_index("__row_order")
    integer_columns = [
        "date",
        "duration_ms",
        "hourmin",
        "is_rand",
        "tab",
        "time_ms",
        "user_id",
    ]
    events = events.with_columns(
        [
            pl.col(name)
            .cast(pl.Int64, strict=False)
            .fill_null(-1)
            .alias(name)
            for name in integer_columns
        ]
    )

    if require_label:
        events = events.with_columns(
            pl.col("long_view").cast(pl.Int8, strict=False).alias("long_view")
        )
        if events.get_column("long_view").null_count() != 0:
            raise RuntimeError("training long_view contains null values")
        invalid_labels = events.filter(
            ~pl.col("long_view").is_in([0, 1])
        ).height
        if invalid_labels:
            raise RuntimeError(
                f"training long_view contains {invalid_labels} non-binary rows"
            )

    durations = events.get_column("duration_ms").to_numpy().astype(
        np.int64, copy=False
    )
    duration_missing = durations < 0
    nonnegative_duration = np.maximum(durations, 0)
    duration_bucket = np.searchsorted(
        DURATION_EDGES_MS, nonnegative_duration, side="right"
    ).astype(np.int16)
    duration_bucket[duration_missing] = 0

    hourmin = events.get_column("hourmin").to_numpy().astype(
        np.int64, copy=False
    )
    hour = np.floor_divide(np.maximum(hourmin, 0), 100)
    hour = np.clip(hour, 0, 23).astype(np.int16)
    hour[hourmin < 0] = -1

    events = events.with_columns(
        pl.Series("duration_missing", duration_missing.astype(np.int8)),
        pl.Series("duration_bucket", duration_bucket),
        pl.Series("hour", hour),
    )
    return add_backward_sequence_features(events)


def weekday_array(date_values: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_values, dtype=np.int64)
    unique_dates, inverse = np.unique(dates, return_inverse=True)
    mapped = np.full(unique_dates.shape[0], -1, dtype=np.int16)
    for index, value in enumerate(unique_dates):
        if value < 0:
            continue
        try:
            mapped[index] = datetime.strptime(
                str(int(value)), "%Y%m%d"
            ).weekday()
        except ValueError:
            mapped[index] = -1
    return mapped[inverse]


def build_user_vocabulary(frame: pl.DataFrame) -> np.ndarray:
    users = frame.get_column("user_id").to_numpy().astype(
        np.int64, copy=False
    )
    vocabulary = np.unique(users[users >= 0]).astype(np.int64, copy=False)
    if vocabulary.size == 0:
        raise RuntimeError("cannot build a user vocabulary without valid user IDs")
    if vocabulary.size > 1 and np.any(vocabulary[1:] <= vocabulary[:-1]):
        raise RuntimeError("user vocabulary is not strictly increasing")
    return vocabulary


def map_user_codes(
    user_ids: np.ndarray,
    vocabulary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    users = np.asarray(user_ids, dtype=np.int64)
    vocabulary = np.asarray(vocabulary, dtype=np.int64)
    if users.ndim != 1 or vocabulary.ndim != 1:
        raise RuntimeError("user IDs and vocabulary must be one-dimensional")
    if vocabulary.size == 0:
        raise RuntimeError("user vocabulary is empty")
    if vocabulary.size > 1 and np.any(vocabulary[1:] <= vocabulary[:-1]):
        raise RuntimeError("user vocabulary must be strictly increasing")
    positions = np.searchsorted(vocabulary, users)
    known = (users >= 0) & (positions < vocabulary.size)
    candidate_rows = np.flatnonzero(known)
    if candidate_rows.size:
        known[candidate_rows] &= (
            vocabulary[positions[candidate_rows]] == users[candidate_rows]
        )
    codes = np.zeros(users.size, dtype=np.int32)
    codes[known] = positions[known].astype(np.int32) + 1
    return codes, known


def user_vocabulary_diagnostics(
    frame: pl.DataFrame,
    vocabulary: np.ndarray,
) -> dict[str, Any]:
    users = frame.get_column("user_id").to_numpy().astype(
        np.int64, copy=False
    )
    codes, known = map_user_codes(users, vocabulary)
    return {
        "rows": frame.height,
        "vocabulary_users": int(vocabulary.size),
        "observed_users": int(np.unique(users[users >= 0]).size),
        "known_user_row_fraction": float(known.mean()) if known.size else None,
        "unknown_user_rows": int((~known).sum()),
        "maximum_user_code": int(codes.max()) if codes.size else None,
    }


def combine_user_context_codes(
    user_codes: np.ndarray,
    context_codes: np.ndarray,
    stride: int,
) -> np.ndarray:
    users = np.asarray(user_codes, dtype=np.int64)
    context = np.asarray(context_codes, dtype=np.int64)
    if users.shape != context.shape or users.ndim != 1:
        raise RuntimeError("user and context codes must be aligned vectors")
    if (
        stride <= 0
        or (context.size and int(context.min()) < 0)
        or (context.size and int(context.max()) >= stride)
    ):
        raise RuntimeError("context code exceeds its frozen interaction stride")
    combined = np.zeros(users.size, dtype=np.int32)
    known = users > 0
    combined[known] = (
        users[known] * int(stride) + context[known]
    ).astype(np.int32)
    if combined.size and int(combined.max()) > 2**24:
        raise RuntimeError(
            "categorical interaction code is not exactly representable in float32"
        )
    return combined


def build_feature_matrix(
    frame: pl.DataFrame,
    feature_names: list[str],
    user_vocabulary: np.ndarray | None = None,
) -> np.ndarray:
    if feature_names not in (
        BASE_FEATURE_NAMES,
        FEATURE_NAMES,
        PERSONALIZED_FEATURE_NAMES,
        JOINT_PERSONALIZED_FEATURE_NAMES,
    ):
        raise RuntimeError("unsupported ensemble member feature contract")
    matrix = np.empty((frame.height, len(feature_names)), dtype=np.float32)
    column = 0

    def put(name: str, values: np.ndarray) -> None:
        nonlocal column
        if column >= len(feature_names) or feature_names[column] != name:
            raise RuntimeError(
                f"feature construction order mismatch at {name}, column {column}"
            )
        values = np.asarray(values)
        if values.shape != (frame.height,):
            raise RuntimeError(
                f"feature {name} has shape {values.shape}, "
                f"expected {(frame.height,)}"
            )
        matrix[:, column] = values.astype(np.float32, copy=False)
        column += 1

    tabs = frame.get_column("tab").to_numpy().astype(np.int64, copy=False)
    tab_codes = (np.maximum(tabs, 0) + 1).astype(np.int32)
    tab_codes[tabs < 0] = 0
    put("tab_code", tab_codes)

    is_rand = frame.get_column("is_rand").to_numpy().astype(
        np.int64, copy=False
    )
    rand_codes = (np.maximum(is_rand, 0) + 1).astype(np.int32)
    rand_codes[is_rand < 0] = 0
    put("is_rand_code", rand_codes)

    hours = frame.get_column("hour").to_numpy().astype(np.int16, copy=False)
    hour_codes = (np.maximum(hours, 0) + 1).astype(np.int32)
    hour_codes[hours < 0] = 0
    put("hour_code", hour_codes)

    dates = frame.get_column("date").to_numpy().astype(np.int64, copy=False)
    weekdays = weekday_array(dates)
    weekday_codes = (np.maximum(weekdays, 0) + 1).astype(np.int32)
    weekday_codes[weekdays < 0] = 0
    put("weekday_code", weekday_codes)

    dayparts = np.floor_divide(np.maximum(hours, 0), 3).astype(np.int32) + 1
    dayparts[hours < 0] = 0
    put("daypart_code", dayparts)

    duration_buckets = frame.get_column("duration_bucket").to_numpy().astype(
        np.int16, copy=False
    )
    put("duration_bucket_code", duration_buckets.astype(np.int32) + 1)

    durations_raw = frame.get_column("duration_ms").to_numpy().astype(
        np.int64, copy=False
    )
    duration_missing = frame.get_column("duration_missing").to_numpy().astype(
        np.int8, copy=False
    )
    durations = np.maximum(durations_raw, 0).astype(np.float64)
    clipped_duration = np.minimum(durations, 3_600_000.0)
    put("log_duration_ms", np.log1p(clipped_duration))
    put("sqrt_duration_seconds", np.sqrt(clipped_duration / 1000.0))
    put("duration_missing", duration_missing)
    put("duration_zero", (durations_raw == 0).astype(np.int8))
    put("duration_at_least_18s", (durations >= 18_000.0).astype(np.int8))
    put("duration_at_least_60s", (durations >= 60_000.0).astype(np.int8))
    put("duration_outlier", (durations >= 600_000.0).astype(np.int8))

    hourmin = frame.get_column("hourmin").to_numpy().astype(
        np.int64, copy=False
    )
    minutes = np.mod(np.maximum(hourmin, 0), 100)
    minutes = np.clip(minutes, 0, 59).astype(np.float64)
    minute_fraction = minutes / 60.0
    minute_fraction[hourmin < 0] = 0.0
    put("minute_fraction", minute_fraction)

    fractional_hour = np.maximum(hours, 0).astype(np.float64) + minute_fraction
    hour_angle = 2.0 * np.pi * fractional_hour / 24.0
    valid_hour = hours >= 0
    put("hour_sin", np.where(valid_hour, np.sin(hour_angle), 0.0))
    put("hour_cos", np.where(valid_hour, np.cos(hour_angle), 0.0))

    weekday_angle = (
        2.0 * np.pi * np.maximum(weekdays, 0).astype(np.float64) / 7.0
    )
    valid_weekday = weekdays >= 0
    put("weekday_sin", np.where(valid_weekday, np.sin(weekday_angle), 0.0))
    put("weekday_cos", np.where(valid_weekday, np.cos(weekday_angle), 0.0))

    previous_gap_bucket = frame.get_column(
        "previous_gap_bucket"
    ).to_numpy().astype(np.int16, copy=False)
    put("previous_gap_bucket_code", previous_gap_bucket)

    session_position_bucket = frame.get_column(
        "session_position_bucket"
    ).to_numpy().astype(np.int16, copy=False)
    put("session_position_bucket_code", session_position_bucket)

    previous_gap_ms = frame.get_column("previous_gap_ms").to_numpy().astype(
        np.int64, copy=False
    )
    valid_gap = previous_gap_ms >= 0
    clipped_gap = np.minimum(np.maximum(previous_gap_ms, 0), 86_400_000)
    log_gap = np.log1p(clipped_gap.astype(np.float64))
    log_gap[~valid_gap] = 0.0
    put("log_previous_gap_ms", log_gap)
    put(
        "previous_gap_missing",
        frame.get_column("previous_gap_missing").to_numpy(),
    )
    put("session_start", frame.get_column("session_start").to_numpy())

    session_position = frame.get_column("session_position").to_numpy().astype(
        np.int64, copy=False
    )
    put(
        "log_session_position",
        np.log1p(np.minimum(session_position, 1024).astype(np.float64)),
    )

    if feature_names == BASE_FEATURE_NAMES:
        if column != len(feature_names):
            raise RuntimeError(
                f"filled {column} base features but declared {len(feature_names)}"
            )
        if not np.isfinite(matrix).all():
            bad = int(matrix.size - np.isfinite(matrix).sum())
            raise RuntimeError(
                f"base feature matrix contains {bad} non-finite values"
            )
        return matrix

    put(
        "previous_long_view_proxy_code",
        frame.get_column("previous_long_view_proxy_code").to_numpy(),
    )
    put(
        "previous_duration_bucket_code",
        frame.get_column("previous_duration_bucket").to_numpy(),
    )
    previous_margin = frame.get_column(
        "previous_threshold_margin_ms"
    ).to_numpy().astype(np.int64, copy=False)
    signed_log_margin = np.sign(previous_margin).astype(np.float64) * np.log1p(
        np.abs(previous_margin).astype(np.float64)
    )
    put("signed_log_previous_threshold_margin_ms", signed_log_margin)
    put(
        "session_prior_proxy_positive_rate",
        frame.get_column("session_prior_proxy_positive_rate").to_numpy(),
    )
    put(
        "recent4_prior_proxy_positive_rate",
        frame.get_column("recent4_prior_proxy_positive_rate").to_numpy(),
    )
    proxy_observations = frame.get_column(
        "prior_proxy_observation_count"
    ).to_numpy().astype(np.int64, copy=False)
    put(
        "log_prior_proxy_observations",
        np.log1p(np.minimum(proxy_observations, 1024).astype(np.float64)),
    )

    if feature_names == FEATURE_NAMES:
        if column != len(feature_names):
            raise RuntimeError(
                f"filled {column} proxy features but declared {len(feature_names)}"
            )
        if not np.isfinite(matrix).all():
            bad = int(matrix.size - np.isfinite(matrix).sum())
            raise RuntimeError(
                f"proxy feature matrix contains {bad} non-finite values"
            )
        return matrix

    if user_vocabulary is None:
        raise RuntimeError(
            "personalized feature construction requires a frozen user vocabulary"
        )
    user_ids = frame.get_column("user_id").to_numpy().astype(
        np.int64, copy=False
    )
    user_codes, _known_users = map_user_codes(user_ids, user_vocabulary)
    put("user_code", user_codes)
    put(
        "user_tab_context_code",
        combine_user_context_codes(user_codes, tab_codes, 128),
    )
    duration_context_codes = duration_buckets.astype(np.int32) + 1
    put(
        "user_duration_context_code",
        combine_user_context_codes(user_codes, duration_context_codes, 32),
    )
    put(
        "user_hour_context_code",
        combine_user_context_codes(user_codes, hour_codes, 32),
    )
    put(
        "user_session_position_context_code",
        combine_user_context_codes(
            user_codes, session_position_bucket.astype(np.int32), 16
        ),
    )
    previous_proxy_codes = frame.get_column(
        "previous_long_view_proxy_code"
    ).to_numpy().astype(np.int32, copy=False)
    put(
        "user_previous_proxy_state_context_code",
        combine_user_context_codes(user_codes, previous_proxy_codes, 4),
    )

    if feature_names == PERSONALIZED_FEATURE_NAMES:
        if column != len(feature_names):
            raise RuntimeError(
                f"filled {column} personalized features but declared "
                f"{len(feature_names)}"
            )
        if not np.isfinite(matrix).all():
            bad = int(matrix.size - np.isfinite(matrix).sum())
            raise RuntimeError(
                f"personalized feature matrix contains {bad} non-finite values"
            )
        return matrix

    # The extended member retains the ordered state of already-completed
    # impressions. The parent recent-four mean cannot distinguish recency,
    # streaks, missing observations, or tab transitions with the same average.
    put(
        "previous_tab_code",
        frame.get_column("previous_tab_code").to_numpy(),
    )
    put(
        "tab_transition_code",
        frame.get_column("tab_transition_code").to_numpy(),
    )
    put(
        "recent4_proxy_pattern_code",
        frame.get_column("recent4_proxy_pattern_code").to_numpy(),
    )
    put(
        "signed_proxy_streak_code",
        frame.get_column("signed_proxy_streak_code").to_numpy(),
    )
    session_elapsed_ms = frame.get_column(
        "session_elapsed_ms"
    ).to_numpy().astype(np.int64, copy=False)
    put(
        "log_session_elapsed_ms",
        np.log1p(
            np.minimum(np.maximum(session_elapsed_ms, 0), 86_400_000).astype(
                np.float64
            )
        ),
    )
    put(
        "recent4_mean_log_duration_ms",
        frame.get_column("recent4_mean_log_duration_ms").to_numpy(),
    )

    if column != len(feature_names):
        raise RuntimeError(
            f"filled {column} features but declared {len(feature_names)}"
        )
    if not np.isfinite(matrix).all():
        bad = int(matrix.size - np.isfinite(matrix).sum())
        raise RuntimeError(f"feature matrix contains {bad} non-finite values")
    return matrix


def sequence_diagnostics(frame: pl.DataFrame) -> dict[str, Any]:
    gaps = frame.get_column("previous_gap_ms").to_numpy().astype(
        np.int64, copy=False
    )
    positions = frame.get_column("session_position").to_numpy().astype(
        np.int64, copy=False
    )
    starts = frame.get_column("session_start").to_numpy().astype(
        np.int8, copy=False
    )
    proxy_codes = frame.get_column(
        "previous_long_view_proxy_code"
    ).to_numpy().astype(np.int8, copy=False)
    margins = frame.get_column(
        "previous_threshold_margin_ms"
    ).to_numpy().astype(np.int64, copy=False)
    observations = frame.get_column(
        "prior_proxy_observation_count"
    ).to_numpy().astype(np.int64, copy=False)
    previous_tab_codes = frame.get_column("previous_tab_code").to_numpy().astype(
        np.int16, copy=False
    )
    proxy_patterns = frame.get_column(
        "recent4_proxy_pattern_code"
    ).to_numpy().astype(np.int16, copy=False)
    streak_codes = frame.get_column(
        "signed_proxy_streak_code"
    ).to_numpy().astype(np.int16, copy=False)
    session_elapsed = frame.get_column("session_elapsed_ms").to_numpy().astype(
        np.int64, copy=False
    )
    recent_mean_duration = frame.get_column(
        "recent4_mean_log_duration_ms"
    ).to_numpy().astype(np.float32, copy=False)

    valid_gap = gaps >= 0
    valid_gaps = gaps[valid_gap]
    within_session = positions > 0
    valid_proxy = proxy_codes > 0
    valid_proxy_count = int(valid_proxy.sum())
    within_count = int(within_session.sum())
    return {
        "rows": frame.height,
        "valid_previous_gap_fraction": (
            float(valid_gap.mean()) if gaps.size else None
        ),
        "session_start_fraction": (
            float(starts.mean()) if starts.size else None
        ),
        "nonzero_session_position_fraction": (
            float(within_session.mean()) if positions.size else None
        ),
        "session_position_p50": (
            float(np.quantile(positions, 0.50)) if positions.size else None
        ),
        "session_position_p90": (
            float(np.quantile(positions, 0.90)) if positions.size else None
        ),
        "session_position_max": (
            int(positions.max()) if positions.size else None
        ),
        "valid_previous_gap_ms_p50": (
            float(np.quantile(valid_gaps, 0.50)) if valid_gaps.size else None
        ),
        "valid_previous_gap_ms_p90": (
            float(np.quantile(valid_gaps, 0.90)) if valid_gaps.size else None
        ),
        "valid_previous_proxy_fraction": (
            float(valid_proxy.mean()) if proxy_codes.size else None
        ),
        "valid_previous_proxy_given_within_session_fraction": (
            float(valid_proxy_count / within_count) if within_count else None
        ),
        "valid_previous_proxy_positive_rate": (
            float((proxy_codes[valid_proxy] == 2).mean())
            if valid_proxy_count
            else None
        ),
        "valid_previous_proxy_exact_margin_fraction": (
            float((margins[valid_proxy] == 0).mean())
            if valid_proxy_count
            else None
        ),
        "nonzero_proxy_observation_fraction": (
            float((observations > 0).mean()) if observations.size else None
        ),
        "proxy_observation_count_p50": (
            float(np.quantile(observations, 0.50))
            if observations.size
            else None
        ),
        "proxy_observation_count_p90": (
            float(np.quantile(observations, 0.90))
            if observations.size
            else None
        ),
        "previous_tab_available_fraction": (
            float((previous_tab_codes > 0).mean())
            if previous_tab_codes.size
            else None
        ),
        "nonempty_recent4_proxy_pattern_fraction": (
            float((proxy_patterns > 0).mean()) if proxy_patterns.size else None
        ),
        "nonneutral_proxy_streak_fraction": (
            float((streak_codes != MAX_PROXY_STREAK).mean())
            if streak_codes.size
            else None
        ),
        "positive_session_elapsed_fraction": (
            float((session_elapsed > 0).mean())
            if session_elapsed.size
            else None
        ),
        "recent_duration_history_fraction": (
            float((recent_mean_duration > 0).mean())
            if recent_mean_duration.size
            else None
        ),
    }


def model_parameters() -> dict[str, Any]:
    threads = max(1, int(os.cpu_count() or 1))
    return {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 500,
        "min_sum_hessian_in_leaf": 1.0,
        "feature_fraction": 0.9,
        "lambda_l1": 0.1,
        "lambda_l2": 10.0,
        "max_bin": 127,
        "deterministic": True,
        "force_col_wise": True,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "data_random_seed": SEED,
        "num_threads": threads,
        "verbosity": -1,
    }


def label_array(frame: pl.DataFrame) -> np.ndarray:
    return frame.get_column("long_view").to_numpy().astype(
        np.int8, copy=False
    )


class LocalRankingEvaluator:
    """Exact local implementation of the documented controller metric geometry."""

    def __init__(self, user_ids: np.ndarray, labels: np.ndarray) -> None:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int8)
        if user_ids.ndim != 1 or labels.shape != user_ids.shape:
            raise RuntimeError("holdout users and labels must be aligned vectors")
        if labels.size == 0:
            raise RuntimeError("chronological holdout is empty")
        if not np.isin(labels, np.asarray([0, 1], dtype=np.int8)).all():
            raise RuntimeError("chronological holdout labels are not binary")

        self.rows = int(labels.size)
        self.order = np.argsort(user_ids, kind="stable")
        sorted_users = user_ids[self.order]
        self.labels = labels[self.order]
        starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(sorted_users[1:] != sorted_users[:-1]).astype(
                    np.int64
                )
                + 1,
            )
        )
        ends = np.concatenate(
            (starts[1:], np.asarray([self.rows], dtype=np.int64))
        )
        self.groups = [
            (int(start), int(end)) for start, end in zip(starts, ends)
        ]
        self.user_count = len(self.groups)
        self.discounts = 1.0 / np.log2(np.arange(2, 7, dtype=np.float64))

        eligible_users = 0
        eligible_positive_weight = 0
        for start, end in self.groups:
            positives = int(self.labels[start:end].sum())
            negatives = (end - start) - positives
            if positives > 0 and negatives > 0:
                eligible_users += 1
                eligible_positive_weight += positives
        if eligible_positive_weight <= 0:
            raise RuntimeError("chronological holdout has no GAUC-eligible users")
        self.eligible_users = eligible_users
        self.eligible_positive_weight = eligible_positive_weight

    def evaluate(self, scores: np.ndarray) -> dict[str, Any]:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (self.rows,):
            raise RuntimeError(
                f"holdout scores have shape {scores.shape}, expected {(self.rows,)}"
            )
        if not np.isfinite(scores).all():
            raise RuntimeError("holdout scores contain non-finite values")
        sorted_scores = scores[self.order]

        weighted_auc_sum = 0.0
        positive_weight = 0
        ndcg_sum = 0.0
        for start, end in self.groups:
            group_scores = sorted_scores[start:end]
            group_labels = self.labels[start:end]
            group_size = end - start
            positives = int(group_labels.sum())
            negatives = group_size - positives

            if positives > 0 and negatives > 0:
                average_ranks = rankdata(group_scores, method="average")
                positive_rank_sum = float(
                    average_ranks[group_labels == 1].sum()
                )
                auc = (
                    positive_rank_sum - positives * (positives + 1) / 2.0
                ) / float(positives * negatives)
                weighted_auc_sum += positives * auc
                positive_weight += positives

            top_count = min(5, group_size)
            top_order = np.argsort(-group_scores, kind="stable")[:top_count]
            dcg = float(
                np.dot(
                    group_labels[top_order].astype(np.float64),
                    self.discounts[:top_count],
                )
            )
            ideal_positive_count = min(5, positives)
            if ideal_positive_count > 0:
                ideal_dcg = float(self.discounts[:ideal_positive_count].sum())
                ndcg_sum += dcg / ideal_dcg

        if positive_weight != self.eligible_positive_weight:
            raise RuntimeError("GAUC positive-weight audit failed")
        gauc = weighted_auc_sum / float(positive_weight)
        ndcg = ndcg_sum / float(self.user_count)
        primary = 0.5 * (gauc + ndcg)
        if not all(math.isfinite(value) for value in (gauc, ndcg, primary)):
            raise RuntimeError("local holdout metrics are non-finite")
        return {
            "GAUC": gauc,
            "nDCG@5": ndcg,
            "primary": primary,
            "rows": self.rows,
            "users": self.user_count,
            "gauc_eligible_users": self.eligible_users,
            "gauc_positive_weight": positive_weight,
        }


class SampledHoldoutPrimary:
    """Evaluate exact within-user metrics at a frozen round grid."""

    def __init__(self, evaluator: LocalRankingEvaluator) -> None:
        self.evaluator = evaluator
        self.iteration = 0
        self.records: list[dict[str, Any]] = []
        self.last_primary = 0.0

    def __call__(
        self, predictions: np.ndarray, _dataset: lgb.Dataset
    ) -> tuple[str, float, bool]:
        self.iteration += 1
        if self.iteration in ITERATION_GRID:
            # Final inference stores float32 scores, so selection mirrors that
            # precision before any tie handling.
            record = self.evaluator.evaluate(
                np.asarray(predictions, dtype=np.float32)
            )
            record["iteration"] = self.iteration
            self.records.append(record)
            self.last_primary = float(record["primary"])
        return "sampled_within_user_primary", self.last_primary, True


def select_iteration(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != len(ITERATION_GRID):
        raise RuntimeError(
            f"recorded {len(records)} iteration metrics, "
            f"expected {len(ITERATION_GRID)}"
        )
    observed_iterations = tuple(int(record["iteration"]) for record in records)
    if observed_iterations != ITERATION_GRID:
        raise RuntimeError("observed iteration grid does not match frozen grid")
    # Primary is authoritative; prefer GAUC and then the earlier model on exact ties.
    return max(
        records,
        key=lambda record: (
            float(record["primary"]),
            float(record["GAUC"]),
            -int(record["iteration"]),
        ),
    )


def train_pointwise_model(
    train_frame: pl.DataFrame,
    validation_frame: pl.DataFrame,
    member_name: str,
    feature_names: list[str],
    categorical_feature_names: list[str],
    holdout_evaluator: LocalRankingEvaluator,
    user_vocabulary: np.ndarray | None = None,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    train_sequence = sequence_diagnostics(train_frame)
    validation_sequence = sequence_diagnostics(validation_frame)
    train_matrix = build_feature_matrix(
        train_frame, feature_names, user_vocabulary
    )
    validation_matrix = build_feature_matrix(
        validation_frame, feature_names, user_vocabulary
    )
    train_labels = label_array(train_frame)
    validation_labels = label_array(validation_frame)
    train_set = lgb.Dataset(
        train_matrix,
        label=train_labels,
        feature_name=feature_names,
        categorical_feature=categorical_feature_names,
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        validation_matrix,
        label=validation_labels,
        feature_name=feature_names,
        categorical_feature=categorical_feature_names,
        reference=train_set,
        free_raw_data=True,
    )
    sampled_metric = SampledHoldoutPrimary(holdout_evaluator)
    booster = lgb.train(
        model_parameters(),
        train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[validation_set],
        valid_names=["chronological"],
        feval=sampled_metric,
        callbacks=[
            lgb.log_evaluation(period=20),
        ],
    )
    if sampled_metric.iteration != MAX_BOOST_ROUNDS:
        raise RuntimeError(
            f"custom metric saw {sampled_metric.iteration} rounds, "
            f"expected {MAX_BOOST_ROUNDS}"
        )
    selected_record = select_iteration(sampled_metric.records)
    best_iteration = int(selected_record["iteration"])
    selected_predictions = np.asarray(
        booster.predict(
            validation_matrix,
            num_iteration=best_iteration,
            num_threads=max(1, int(os.cpu_count() or 1)),
        ),
        dtype=np.float32,
    )
    if selected_predictions.shape != (validation_frame.height,):
        raise RuntimeError("selected holdout prediction shape is invalid")
    if not np.isfinite(selected_predictions).all():
        raise RuntimeError("selected holdout predictions contain non-finite values")
    replay_metrics = holdout_evaluator.evaluate(selected_predictions)
    replay_metric_delta = {
        metric_name: (
            float(replay_metrics[metric_name])
            - float(selected_record[metric_name])
        )
        for metric_name in ("GAUC", "nDCG@5", "primary")
    }
    diagnostics = {
        "member": member_name,
        "best_iteration": best_iteration,
        "selected_holdout_metrics": replay_metrics,
        "selected_metric_replay_delta": replay_metric_delta,
        "iteration_metric_path": sampled_metric.records,
        "iteration_grid": list(ITERATION_GRID),
        "train_rows": train_frame.height,
        "validation_rows": validation_frame.height,
        "objective": "binary",
        "row_weights": "uniform",
        "selection_metric": "chronological mean of within-user GAUC and nDCG@5",
        "selection_tie_break": "higher GAUC, then earlier iteration",
        "feature_names": feature_names,
        "categorical_feature_names": categorical_feature_names,
        "train_label_rate": float(np.mean(train_labels)),
        "validation_label_rate": float(np.mean(validation_labels)),
        "train_sequence": train_sequence,
        "validation_sequence": validation_sequence,
        "train_user_vocabulary": (
            user_vocabulary_diagnostics(train_frame, user_vocabulary)
            if user_vocabulary is not None
            else None
        ),
        "validation_user_vocabulary": (
            user_vocabulary_diagnostics(validation_frame, user_vocabulary)
            if user_vocabulary is not None
            else None
        ),
    }
    del (
        booster,
        train_set,
        validation_set,
        train_matrix,
        validation_matrix,
        train_labels,
        validation_labels,
    )
    gc.collect()
    return best_iteration, selected_predictions, diagnostics


def fit_full_pointwise_model(
    frame: pl.DataFrame,
    best_iteration: int,
    model_path: Path,
    member_name: str,
    feature_names: list[str],
    categorical_feature_names: list[str],
    user_vocabulary: np.ndarray | None = None,
) -> dict[str, Any]:
    sequence = sequence_diagnostics(frame)
    matrix = build_feature_matrix(frame, feature_names, user_vocabulary)
    labels = label_array(frame)
    dataset = lgb.Dataset(
        matrix,
        label=labels,
        feature_name=feature_names,
        categorical_feature=categorical_feature_names,
        free_raw_data=True,
    )
    booster = lgb.train(
        model_parameters(),
        dataset,
        num_boost_round=best_iteration,
        callbacks=[lgb.log_evaluation(period=25)],
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path), num_iteration=best_iteration)
    diagnostics = {
        "member": member_name,
        "rows": frame.height,
        "objective": "binary",
        "row_weights": "uniform",
        "label_rate": float(np.mean(labels)),
        "iterations": best_iteration,
        "feature_names": feature_names,
        "categorical_feature_names": categorical_feature_names,
        "sequence": sequence,
        "user_vocabulary": (
            user_vocabulary_diagnostics(frame, user_vocabulary)
            if user_vocabulary is not None
            else None
        ),
    }
    del booster, dataset, matrix, labels
    gc.collect()
    return diagnostics


def resolve_checkpoint(path: Path) -> Path:
    if (path / MANIFEST_FILE).is_file():
        return path
    nested = path / "checkpoint"
    if (nested / MANIFEST_FILE).is_file():
        return nested
    raise FileNotFoundError(
        f"could not find {MANIFEST_FILE} in {path} or {nested}"
    )


def validate_manifest(manifest: dict[str, Any]) -> None:
    if int(manifest.get("version", -1)) != VERSION:
        raise RuntimeError(
            f"unsupported checkpoint version {manifest.get('version')}; "
            f"expected {VERSION}"
        )
    if manifest.get("parent_attempts") != list(PARENT_ATTEMPTS):
        raise RuntimeError("checkpoint parent attempts do not match source")
    if manifest.get("duration_edges_ms") != DURATION_EDGES_MS.tolist():
        raise RuntimeError("checkpoint duration buckets do not match source")
    if manifest.get("previous_gap_edges_ms") != PREVIOUS_GAP_EDGES_MS.tolist():
        raise RuntimeError("checkpoint previous-gap buckets do not match source")
    if manifest.get("session_position_edges") != SESSION_POSITION_EDGES.tolist():
        raise RuntimeError("checkpoint session-position buckets do not match source")
    if int(manifest.get("session_gap_ms", -1)) != SESSION_GAP_MS:
        raise RuntimeError("checkpoint session-gap threshold does not match source")
    if int(manifest.get("long_view_threshold_ms", -1)) != LONG_VIEW_THRESHOLD_MS:
        raise RuntimeError("checkpoint long-view threshold does not match source")
    if int(manifest.get("recent_proxy_window", -1)) != RECENT_PROXY_WINDOW:
        raise RuntimeError("checkpoint recent-proxy window does not match source")
    if manifest.get("user_vocabulary_file") != USER_VOCAB_FILE:
        raise RuntimeError("checkpoint user-vocabulary file does not match source")
    if manifest.get("ordered_history_contract") != JOINT_HISTORY_CONTRACT:
        raise RuntimeError("checkpoint ordered-history contract does not match source")
    holdout_selection = manifest.get("holdout_selection")
    if not isinstance(holdout_selection, dict):
        raise RuntimeError("checkpoint lacks holdout-selection contract")
    if holdout_selection.get("iteration_grid") != list(ITERATION_GRID):
        raise RuntimeError("checkpoint iteration grid does not match source")
    if holdout_selection.get("base_weight_grid") != list(
        BASE_BLEND_WEIGHT_GRID
    ):
        raise RuntimeError("checkpoint blend-weight grid does not match source")
    if int(holdout_selection.get("simplex_weight_units", -1)) != (
        ENSEMBLE_WEIGHT_UNITS
    ):
        raise RuntimeError("checkpoint simplex weight grid does not match source")
    if not math.isclose(
        float(holdout_selection.get("minimum_local_personalization_gain", -1.0)),
        MIN_LOCAL_PERSONALIZATION_GAIN,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("checkpoint personalization gate does not match source")
    if holdout_selection.get("joint_weight_grid") != list(
        BASE_BLEND_WEIGHT_GRID
    ):
        raise RuntimeError("checkpoint joint-weight grid does not match source")
    if not math.isclose(
        float(holdout_selection.get("minimum_local_joint_gain", -1.0)),
        MIN_LOCAL_JOINT_GAIN,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("checkpoint ordered-history gate does not match source")
    if holdout_selection.get("selection_metric") != (
        "mean of within-user GAUC and nDCG@5"
    ):
        raise RuntimeError("checkpoint selection metric does not match source")

    ensemble = manifest.get("ensemble")
    if not isinstance(ensemble, dict):
        raise RuntimeError("checkpoint lacks ensemble contract")
    if ensemble.get("method") != "within_user_average_tie_percentile_rank":
        raise RuntimeError("checkpoint ensemble method does not match source")
    base_weight = float(ensemble.get("base_weight", float("nan")))
    proxy_weight = float(ensemble.get("proxy_weight", float("nan")))
    personalized_weight = float(
        ensemble.get("personalized_weight", float("nan"))
    )
    joint_weight = float(ensemble.get("joint_weight", float("nan")))
    parent_base_weight = float(
        ensemble.get("parent_base_weight", float("nan"))
    )
    parent_proxy_weight = float(
        ensemble.get("parent_proxy_weight", float("nan"))
    )
    parent_personalized_weight = float(
        ensemble.get("parent_personalized_weight", float("nan"))
    )
    for name, weight in (
        ("parent_base", parent_base_weight),
        ("parent_proxy", parent_proxy_weight),
        ("parent_personalized", parent_personalized_weight),
        ("joint", joint_weight),
    ):
        units = round(weight * ENSEMBLE_WEIGHT_UNITS)
        if (
            not math.isfinite(weight)
            or units < 0
            or units > ENSEMBLE_WEIGHT_UNITS
            or not math.isclose(
                weight,
                units / ENSEMBLE_WEIGHT_UNITS,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"checkpoint {name} weight is outside its frozen 1/8 grid"
            )
    if not math.isclose(
        parent_base_weight
        + parent_proxy_weight
        + parent_personalized_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("checkpoint reconstructed parent weights do not sum to one")
    expected_scale = 1.0 - joint_weight
    for name, observed, expected in (
        ("base", base_weight, expected_scale * parent_base_weight),
        ("proxy", proxy_weight, expected_scale * parent_proxy_weight),
        (
            "personalized",
            personalized_weight,
            expected_scale * parent_personalized_weight,
        ),
    ):
        if not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"checkpoint {name} weight is not the frozen parent shrinkage"
            )
    if not math.isclose(
        base_weight
        + proxy_weight
        + personalized_weight
        + joint_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("checkpoint final ensemble weights do not sum to one")

    members = manifest.get("members")
    if not isinstance(members, dict):
        raise RuntimeError("checkpoint lacks ensemble members")
    expected_members = {
        BASE_MEMBER_NAME: (
            BASE_MODEL_FILE,
            BASE_FEATURE_NAMES,
            BASE_CATEGORICAL_FEATURE_NAMES,
        ),
        PROXY_MEMBER_NAME: (
            PROXY_MODEL_FILE,
            FEATURE_NAMES,
            CATEGORICAL_FEATURE_NAMES,
        ),
        PERSONALIZED_MEMBER_NAME: (
            PERSONALIZED_MODEL_FILE,
            PERSONALIZED_FEATURE_NAMES,
            PERSONALIZED_CATEGORICAL_FEATURE_NAMES,
        ),
        JOINT_PERSONALIZED_MEMBER_NAME: (
            JOINT_PERSONALIZED_MODEL_FILE,
            JOINT_PERSONALIZED_FEATURE_NAMES,
            JOINT_PERSONALIZED_CATEGORICAL_FEATURE_NAMES,
        ),
    }
    if set(members) != set(expected_members):
        raise RuntimeError("checkpoint ensemble member names do not match source")
    for member_name, (
        expected_model_file,
        expected_features,
        expected_categorical,
    ) in expected_members.items():
        member = members[member_name]
        if not isinstance(member, dict):
            raise RuntimeError(f"invalid member contract for {member_name}")
        if member.get("model_file") != expected_model_file:
            raise RuntimeError(f"model file mismatch for {member_name}")
        if member.get("feature_names") != expected_features:
            raise RuntimeError(f"feature order mismatch for {member_name}")
        if member.get("categorical_feature_names") != expected_categorical:
            raise RuntimeError(f"categorical feature mismatch for {member_name}")
        iteration = int(member.get("best_iteration", 0))
        if iteration not in ITERATION_GRID:
            raise RuntimeError(f"invalid iteration count for {member_name}")


def predict_member(
    events: pl.DataFrame,
    checkpoint_dir: Path,
    member: dict[str, Any],
    feature_names: list[str],
    user_vocabulary: np.ndarray | None = None,
) -> np.ndarray:
    matrix = build_feature_matrix(events, feature_names, user_vocabulary)
    booster = lgb.Booster(
        model_file=str(checkpoint_dir / str(member["model_file"]))
    )
    predictions = booster.predict(
        matrix,
        num_iteration=int(member["best_iteration"]),
        num_threads=max(1, int(os.cpu_count() or 1)),
    )
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.shape != (events.height,):
        raise RuntimeError(
            f"member prediction shape {predictions.shape} does not match "
            f"rows {events.height}"
        )
    if not np.isfinite(predictions).all():
        bad = int(predictions.size - np.isfinite(predictions).sum())
        raise RuntimeError(f"member predictions contain {bad} non-finite values")
    del matrix, booster
    gc.collect()
    return predictions


def within_user_percentile_ranks(
    user_ids: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return average-tie ranks in [0, 1], independently within each user."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if user_ids.shape != scores.shape or user_ids.ndim != 1:
        raise RuntimeError("user IDs and member scores must be aligned vectors")
    if not np.isfinite(scores).all():
        raise RuntimeError("cannot rank non-finite member scores")
    rows = scores.size
    ranked = np.empty(rows, dtype=np.float32)
    if rows == 0:
        return ranked, 0

    user_order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[user_order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_users[1:] != sorted_users[:-1]).astype(
                np.int64
            )
            + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.asarray([rows], dtype=np.int64)))
    consumed = 0
    for start, end in zip(starts, ends):
        indices = user_order[int(start) : int(end)]
        group_size = indices.size
        if group_size == 1:
            ranked[indices] = 0.5
        else:
            group_ranks = rankdata(scores[indices], method="average")
            ranked[indices] = (
                (group_ranks - 1.0) / float(group_size - 1)
            ).astype(np.float32)
        consumed += group_size
    if consumed != rows or not np.isfinite(ranked).all():
        raise RuntimeError("within-user rank conversion lost rows or became non-finite")
    return ranked, int(starts.size)


def select_holdout_blend(
    evaluator: LocalRankingEvaluator,
    base_ranks: np.ndarray,
    proxy_ranks: np.ndarray,
) -> dict[str, Any]:
    if base_ranks.shape != proxy_ranks.shape:
        raise RuntimeError("holdout member rank vectors are not aligned")
    records: list[dict[str, Any]] = []
    for base_weight in BASE_BLEND_WEIGHT_GRID:
        proxy_weight = 1.0 - base_weight
        blended = np.asarray(
            base_weight * base_ranks + proxy_weight * proxy_ranks,
            dtype=np.float32,
        )
        record = evaluator.evaluate(blended)
        record["base_weight"] = float(base_weight)
        record["proxy_weight"] = float(proxy_weight)
        records.append(record)

    selected = max(
        records,
        key=lambda record: (
            float(record["primary"]),
            float(record["GAUC"]),
            -abs(
                float(record["base_weight"])
                - ATTEMPT9_BASE_BLEND_WEIGHT
            ),
            -float(record["base_weight"]),
        ),
    )
    fixed_attempt9 = next(
        record
        for record in records
        if math.isclose(
            float(record["base_weight"]),
            ATTEMPT9_BASE_BLEND_WEIGHT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    rank_difference = np.asarray(base_ranks - proxy_ranks, dtype=np.float64)
    if base_ranks.size and np.std(base_ranks) > 0 and np.std(proxy_ranks) > 0:
        rank_correlation = float(
            np.corrcoef(
                base_ranks.astype(np.float64),
                proxy_ranks.astype(np.float64),
            )[0, 1]
        )
    else:
        rank_correlation = None
    return {
        "selected": selected,
        "weight_metric_path": records,
        "attempt9_weight_at_selected_iterations": fixed_attempt9,
        "selection_metric": "chronological mean of within-user GAUC and nDCG@5",
        "selection_tie_break": (
            "higher GAUC, then weight nearest attempt 9, then lower base weight"
        ),
        "mean_absolute_member_rank_difference": (
            float(np.mean(np.abs(rank_difference)))
            if rank_difference.size
            else None
        ),
        "pooled_within_user_percentile_rank_correlation": rank_correlation,
    }


def select_personalized_holdout_blend(
    evaluator: LocalRankingEvaluator,
    base_ranks: np.ndarray,
    proxy_ranks: np.ndarray,
    personalized_ranks: np.ndarray,
    parent_selection: dict[str, Any],
) -> dict[str, Any]:
    if not (
        base_ranks.shape == proxy_ranks.shape == personalized_ranks.shape
    ):
        raise RuntimeError("three-member holdout rank vectors are not aligned")
    parent_base_weight = float(parent_selection["base_weight"])
    parent_proxy_weight = float(parent_selection["proxy_weight"])
    if not math.isclose(
        parent_base_weight + parent_proxy_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("reconstructed parent weights do not sum to one")

    records: list[dict[str, Any]] = []
    for base_units in range(ENSEMBLE_WEIGHT_UNITS + 1):
        for proxy_units in range(
            ENSEMBLE_WEIGHT_UNITS - base_units + 1
        ):
            personalized_units = (
                ENSEMBLE_WEIGHT_UNITS - base_units - proxy_units
            )
            base_weight = base_units / ENSEMBLE_WEIGHT_UNITS
            proxy_weight = proxy_units / ENSEMBLE_WEIGHT_UNITS
            personalized_weight = (
                personalized_units / ENSEMBLE_WEIGHT_UNITS
            )
            blended = np.asarray(
                base_weight * base_ranks
                + proxy_weight * proxy_ranks
                + personalized_weight * personalized_ranks,
                dtype=np.float32,
            )
            record = evaluator.evaluate(blended)
            record.update(
                {
                    "base_weight": float(base_weight),
                    "proxy_weight": float(proxy_weight),
                    "personalized_weight": float(personalized_weight),
                }
            )
            records.append(record)

    def distance_from_parent(record: dict[str, Any]) -> float:
        return (
            abs(float(record["base_weight"]) - parent_base_weight)
            + abs(float(record["proxy_weight"]) - parent_proxy_weight)
            + abs(float(record["personalized_weight"]))
        )

    unconstrained = max(
        records,
        key=lambda record: (
            float(record["primary"]),
            float(record["GAUC"]),
            -distance_from_parent(record),
            -float(record["personalized_weight"]),
            -float(record["base_weight"]),
        ),
    )
    parent_record = next(
        record
        for record in records
        if math.isclose(
            float(record["base_weight"]),
            parent_base_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(record["proxy_weight"]),
            parent_proxy_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(record["personalized_weight"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    local_gain = float(unconstrained["primary"]) - float(
        parent_record["primary"]
    )
    gate_passed = local_gain >= MIN_LOCAL_PERSONALIZATION_GAIN
    selected = unconstrained if gate_passed else parent_record

    def pair_audit(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        difference = np.asarray(left - right, dtype=np.float64)
        if left.size and np.std(left) > 0 and np.std(right) > 0:
            correlation = float(
                np.corrcoef(
                    left.astype(np.float64), right.astype(np.float64)
                )[0, 1]
            )
        else:
            correlation = None
        return {
            "mean_absolute_rank_difference": (
                float(np.mean(np.abs(difference)))
                if difference.size
                else None
            ),
            "pooled_rank_correlation": correlation,
        }

    return {
        "selected": selected,
        "unconstrained_best": unconstrained,
        "reconstructed_attempt10_parent": parent_record,
        "unconstrained_local_primary_gain_over_parent": local_gain,
        "minimum_local_personalization_gain": (
            MIN_LOCAL_PERSONALIZATION_GAIN
        ),
        "personalization_gate_passed": gate_passed,
        "weight_metric_path": records,
        "simplex_weight_units": ENSEMBLE_WEIGHT_UNITS,
        "selection_metric": "chronological mean of within-user GAUC and nDCG@5",
        "selection_tie_break": (
            "higher GAUC, then nearest attempt-10 parent, then lower "
            "personalized weight, then lower base weight"
        ),
        "rank_divergence": {
            "base_vs_proxy": pair_audit(base_ranks, proxy_ranks),
            "base_vs_personalized": pair_audit(
                base_ranks, personalized_ranks
            ),
            "proxy_vs_personalized": pair_audit(
                proxy_ranks, personalized_ranks
            ),
        },
    }


def select_joint_holdout_blend(
    evaluator: LocalRankingEvaluator,
    base_ranks: np.ndarray,
    proxy_ranks: np.ndarray,
    personalized_ranks: np.ndarray,
    joint_ranks: np.ndarray,
    parent_selection: dict[str, Any],
) -> dict[str, Any]:
    """Shrink the exact attempt-11 blend toward one ordered-history member."""
    if not (
        base_ranks.shape
        == proxy_ranks.shape
        == personalized_ranks.shape
        == joint_ranks.shape
    ):
        raise RuntimeError("four-member holdout rank vectors are not aligned")
    parent_base_weight = float(parent_selection["base_weight"])
    parent_proxy_weight = float(parent_selection["proxy_weight"])
    parent_personalized_weight = float(
        parent_selection["personalized_weight"]
    )
    if not math.isclose(
        parent_base_weight
        + parent_proxy_weight
        + parent_personalized_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("reconstructed attempt-11 weights do not sum to one")

    parent_ranks = np.asarray(
        parent_base_weight * base_ranks
        + parent_proxy_weight * proxy_ranks
        + parent_personalized_weight * personalized_ranks,
        dtype=np.float32,
    )
    records: list[dict[str, Any]] = []
    for joint_weight in BASE_BLEND_WEIGHT_GRID:
        parent_weight = 1.0 - joint_weight
        blended = np.asarray(
            parent_weight * parent_ranks + joint_weight * joint_ranks,
            dtype=np.float32,
        )
        record = evaluator.evaluate(blended)
        record.update(
            {
                "parent_weight": float(parent_weight),
                "joint_weight": float(joint_weight),
                "base_weight": float(parent_weight * parent_base_weight),
                "proxy_weight": float(parent_weight * parent_proxy_weight),
                "personalized_weight": float(
                    parent_weight * parent_personalized_weight
                ),
            }
        )
        records.append(record)

    unconstrained = max(
        records,
        key=lambda record: (
            float(record["primary"]),
            float(record["GAUC"]),
            -float(record["joint_weight"]),
        ),
    )
    parent_record = next(
        record
        for record in records
        if math.isclose(
            float(record["joint_weight"]), 0.0, rel_tol=0.0, abs_tol=1e-12
        )
    )
    local_gain = float(unconstrained["primary"]) - float(
        parent_record["primary"]
    )
    gate_passed = local_gain >= MIN_LOCAL_JOINT_GAIN
    selected = unconstrained if gate_passed else parent_record

    difference = np.asarray(parent_ranks - joint_ranks, dtype=np.float64)
    if (
        parent_ranks.size
        and np.std(parent_ranks) > 0
        and np.std(joint_ranks) > 0
    ):
        correlation = float(
            np.corrcoef(
                parent_ranks.astype(np.float64),
                joint_ranks.astype(np.float64),
            )[0, 1]
        )
    else:
        correlation = None
    return {
        "selected": selected,
        "unconstrained_best": unconstrained,
        "reconstructed_attempt11_parent": parent_record,
        "unconstrained_local_primary_gain_over_parent": local_gain,
        "minimum_local_joint_gain": MIN_LOCAL_JOINT_GAIN,
        "joint_gate_passed": gate_passed,
        "weight_metric_path": records,
        "joint_weight_grid": list(BASE_BLEND_WEIGHT_GRID),
        "selection_metric": "chronological mean of within-user GAUC and nDCG@5",
        "selection_tie_break": "higher GAUC, then lower history-member weight",
        "mean_absolute_parent_joint_rank_difference": (
            float(np.mean(np.abs(difference))) if difference.size else None
        ),
        "pooled_parent_joint_rank_correlation": correlation,
    }


def predict_from_checkpoint(
    data_root: Path,
    checkpoint_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = read_json(checkpoint_dir / MANIFEST_FILE)
    validate_manifest(manifest)
    user_vocabulary_path = checkpoint_dir / USER_VOCAB_FILE
    if not user_vocabulary_path.is_file():
        raise FileNotFoundError(
            f"checkpoint lacks frozen user vocabulary {USER_VOCAB_FILE}"
        )
    user_vocabulary = np.asarray(
        np.load(user_vocabulary_path, allow_pickle=False), dtype=np.int64
    )
    if (
        user_vocabulary.ndim != 1
        or user_vocabulary.size == 0
        or np.any(user_vocabulary < 0)
        or (
            user_vocabulary.size > 1
            and np.any(user_vocabulary[1:] <= user_vocabulary[:-1])
        )
    ):
        raise RuntimeError("checkpoint user vocabulary is invalid")
    schema_audit = inference_preflight(data_root)
    events = prepare_events(
        data_root / "evaluation_features.parquet", require_label=False
    )
    source_order = events.get_column("__row_order").to_numpy()
    expected_order = np.arange(events.height, dtype=source_order.dtype)
    if not np.array_equal(source_order, expected_order):
        raise RuntimeError("evaluation rows are not in authoritative source order")

    sequence_audit = sequence_diagnostics(events)
    members = manifest["members"]
    base_predictions = predict_member(
        events,
        checkpoint_dir,
        members[BASE_MEMBER_NAME],
        BASE_FEATURE_NAMES,
    )
    proxy_predictions = predict_member(
        events,
        checkpoint_dir,
        members[PROXY_MEMBER_NAME],
        FEATURE_NAMES,
    )
    personalized_predictions = predict_member(
        events,
        checkpoint_dir,
        members[PERSONALIZED_MEMBER_NAME],
        PERSONALIZED_FEATURE_NAMES,
        user_vocabulary,
    )
    joint_predictions = predict_member(
        events,
        checkpoint_dir,
        members[JOINT_PERSONALIZED_MEMBER_NAME],
        JOINT_PERSONALIZED_FEATURE_NAMES,
        user_vocabulary,
    )
    user_ids = events.get_column("user_id").to_numpy().astype(
        np.int64, copy=False
    )
    base_ranks, base_user_count = within_user_percentile_ranks(
        user_ids, base_predictions
    )
    proxy_ranks, proxy_user_count = within_user_percentile_ranks(
        user_ids, proxy_predictions
    )
    personalized_ranks, personalized_user_count = (
        within_user_percentile_ranks(user_ids, personalized_predictions)
    )
    joint_ranks, joint_user_count = within_user_percentile_ranks(
        user_ids, joint_predictions
    )
    if not (
        base_user_count
        == proxy_user_count
        == personalized_user_count
        == joint_user_count
    ):
        raise RuntimeError("ensemble members produced inconsistent user groups")
    base_weight = float(manifest["ensemble"]["base_weight"])
    proxy_weight = float(manifest["ensemble"]["proxy_weight"])
    personalized_weight = float(
        manifest["ensemble"]["personalized_weight"]
    )
    joint_weight = float(manifest["ensemble"]["joint_weight"])
    predictions = np.asarray(
        base_weight * base_ranks
        + proxy_weight * proxy_ranks
        + personalized_weight * personalized_ranks
        + joint_weight * joint_ranks,
        dtype=np.float32,
    )
    if not np.isfinite(predictions).all():
        bad = int(predictions.size - np.isfinite(predictions).sum())
        raise RuntimeError(f"predictions contain {bad} non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, predictions, allow_pickle=False)
    audit = {
        "schema": schema_audit,
        "rows": events.height,
        "users": base_user_count,
        "sequence": sequence_audit,
        "ensemble_method": manifest["ensemble"]["method"],
        "base_weight": base_weight,
        "proxy_weight": proxy_weight,
        "personalized_weight": personalized_weight,
        "joint_weight": joint_weight,
        "user_vocabulary": user_vocabulary_diagnostics(
            events, user_vocabulary
        ),
        "base_member_prediction_min": (
            float(base_predictions.min()) if base_predictions.size else None
        ),
        "base_member_prediction_max": (
            float(base_predictions.max()) if base_predictions.size else None
        ),
        "proxy_member_prediction_min": (
            float(proxy_predictions.min()) if proxy_predictions.size else None
        ),
        "proxy_member_prediction_max": (
            float(proxy_predictions.max()) if proxy_predictions.size else None
        ),
        "personalized_member_prediction_min": (
            float(personalized_predictions.min())
            if personalized_predictions.size
            else None
        ),
        "personalized_member_prediction_max": (
            float(personalized_predictions.max())
            if personalized_predictions.size
            else None
        ),
        "joint_member_prediction_min": (
            float(joint_predictions.min()) if joint_predictions.size else None
        ),
        "joint_member_prediction_max": (
            float(joint_predictions.max()) if joint_predictions.size else None
        ),
        "mean_absolute_member_rank_difference": (
            float(np.mean(np.abs(base_ranks - proxy_ranks)))
            if predictions.size
            else None
        ),
        "mean_absolute_proxy_personalized_rank_difference": (
            float(np.mean(np.abs(proxy_ranks - personalized_ranks)))
            if predictions.size
            else None
        ),
        "mean_absolute_personalized_joint_rank_difference": (
            float(np.mean(np.abs(personalized_ranks - joint_ranks)))
            if predictions.size
            else None
        ),
        "prediction_dtype": str(predictions.dtype),
        "prediction_min": float(predictions.min()) if predictions.size else None,
        "prediction_max": float(predictions.max()) if predictions.size else None,
        "prediction_mean": float(predictions.mean()) if predictions.size else None,
        "source_order_first": int(source_order[0]) if source_order.size else None,
        "source_order_last": int(source_order[-1]) if source_order.size else None,
    }
    del (
        events,
        base_predictions,
        proxy_predictions,
        personalized_predictions,
        joint_predictions,
        base_ranks,
        proxy_ranks,
        personalized_ranks,
        joint_ranks,
        user_vocabulary,
        predictions,
    )
    gc.collect()
    return audit


def run_attempt(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else output_dir / "checkpoint"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    schema_audit = inference_preflight(data_root)
    schema_audit["train"] = parquet_schema_audit(
        data_root / "train.parquet", TRAIN_REQUIRED
    )
    write_json(checkpoint_dir / SCHEMA_AUDIT_FILE, schema_audit)

    events = prepare_events(data_root / "train.parquet", require_label=True)
    unique_dates = sorted(
        int(value)
        for value in events.get_column("date").unique().to_list()
        if int(value) >= 0
    )
    if len(unique_dates) <= HOLDOUT_DAYS:
        raise RuntimeError(
            f"need more than {HOLDOUT_DAYS} training dates, found {unique_dates}"
        )
    validation_dates = unique_dates[-HOLDOUT_DAYS:]

    early_frame = events.filter(
        ~pl.col("date").is_in(validation_dates)
    ).sort(["user_id", "__row_order"])
    validation_frame = events.filter(
        pl.col("date").is_in(validation_dates)
    ).sort(["user_id", "__row_order"])
    if early_frame.height + validation_frame.height != events.height:
        raise RuntimeError("chronological partition lost training rows")
    del events
    gc.collect()

    holdout_user_ids = validation_frame.get_column("user_id").to_numpy().astype(
        np.int64, copy=False
    )
    holdout_labels = label_array(validation_frame)
    holdout_evaluator = LocalRankingEvaluator(
        holdout_user_ids, holdout_labels
    )
    early_user_vocabulary = build_user_vocabulary(early_frame)

    (
        base_best_iteration,
        base_holdout_predictions,
        base_holdout_diagnostics,
    ) = train_pointwise_model(
        early_frame,
        validation_frame,
        BASE_MEMBER_NAME,
        BASE_FEATURE_NAMES,
        BASE_CATEGORICAL_FEATURE_NAMES,
        holdout_evaluator,
    )
    (
        proxy_best_iteration,
        proxy_holdout_predictions,
        proxy_holdout_diagnostics,
    ) = train_pointwise_model(
        early_frame,
        validation_frame,
        PROXY_MEMBER_NAME,
        FEATURE_NAMES,
        CATEGORICAL_FEATURE_NAMES,
        holdout_evaluator,
    )
    (
        personalized_best_iteration,
        personalized_holdout_predictions,
        personalized_holdout_diagnostics,
    ) = train_pointwise_model(
        early_frame,
        validation_frame,
        PERSONALIZED_MEMBER_NAME,
        PERSONALIZED_FEATURE_NAMES,
        PERSONALIZED_CATEGORICAL_FEATURE_NAMES,
        holdout_evaluator,
        early_user_vocabulary,
    )
    (
        joint_best_iteration,
        joint_holdout_predictions,
        joint_holdout_diagnostics,
    ) = train_pointwise_model(
        early_frame,
        validation_frame,
        JOINT_PERSONALIZED_MEMBER_NAME,
        JOINT_PERSONALIZED_FEATURE_NAMES,
        JOINT_PERSONALIZED_CATEGORICAL_FEATURE_NAMES,
        holdout_evaluator,
        early_user_vocabulary,
    )
    base_holdout_ranks, base_holdout_users = within_user_percentile_ranks(
        holdout_user_ids, base_holdout_predictions
    )
    proxy_holdout_ranks, proxy_holdout_users = within_user_percentile_ranks(
        holdout_user_ids, proxy_holdout_predictions
    )
    personalized_holdout_ranks, personalized_holdout_users = (
        within_user_percentile_ranks(
            holdout_user_ids, personalized_holdout_predictions
        )
    )
    joint_holdout_ranks, joint_holdout_users = within_user_percentile_ranks(
        holdout_user_ids, joint_holdout_predictions
    )
    if not (
        base_holdout_users
        == proxy_holdout_users
        == personalized_holdout_users
        == joint_holdout_users
    ):
        raise RuntimeError("holdout ensemble member user counts disagree")
    parent_blend_selection = select_holdout_blend(
        holdout_evaluator,
        base_holdout_ranks,
        proxy_holdout_ranks,
    )
    personalized_blend_selection = select_personalized_holdout_blend(
        holdout_evaluator,
        base_holdout_ranks,
        proxy_holdout_ranks,
        personalized_holdout_ranks,
        parent_blend_selection["selected"],
    )
    blend_selection = select_joint_holdout_blend(
        holdout_evaluator,
        base_holdout_ranks,
        proxy_holdout_ranks,
        personalized_holdout_ranks,
        joint_holdout_ranks,
        personalized_blend_selection["selected"],
    )
    selected_blend = blend_selection["selected"]
    base_blend_weight = float(selected_blend["base_weight"])
    proxy_blend_weight = float(selected_blend["proxy_weight"])
    personalized_blend_weight = float(
        selected_blend["personalized_weight"]
    )
    joint_blend_weight = float(selected_blend["joint_weight"])
    del early_frame, validation_frame
    del (
        holdout_user_ids,
        holdout_labels,
        holdout_evaluator,
        base_holdout_predictions,
        proxy_holdout_predictions,
        personalized_holdout_predictions,
        joint_holdout_predictions,
        base_holdout_ranks,
        proxy_holdout_ranks,
        personalized_holdout_ranks,
        joint_holdout_ranks,
        early_user_vocabulary,
    )
    gc.collect()

    full_frame = prepare_events(
        data_root / "train.parquet", require_label=True
    ).sort(["user_id", "__row_order"])
    full_user_vocabulary = build_user_vocabulary(full_frame)
    np.save(
        checkpoint_dir / USER_VOCAB_FILE,
        full_user_vocabulary,
        allow_pickle=False,
    )
    base_full_diagnostics = fit_full_pointwise_model(
        full_frame,
        base_best_iteration,
        checkpoint_dir / BASE_MODEL_FILE,
        BASE_MEMBER_NAME,
        BASE_FEATURE_NAMES,
        BASE_CATEGORICAL_FEATURE_NAMES,
    )
    proxy_full_diagnostics = fit_full_pointwise_model(
        full_frame,
        proxy_best_iteration,
        checkpoint_dir / PROXY_MODEL_FILE,
        PROXY_MEMBER_NAME,
        FEATURE_NAMES,
        CATEGORICAL_FEATURE_NAMES,
    )
    personalized_full_diagnostics = fit_full_pointwise_model(
        full_frame,
        personalized_best_iteration,
        checkpoint_dir / PERSONALIZED_MODEL_FILE,
        PERSONALIZED_MEMBER_NAME,
        PERSONALIZED_FEATURE_NAMES,
        PERSONALIZED_CATEGORICAL_FEATURE_NAMES,
        full_user_vocabulary,
    )
    joint_full_diagnostics = fit_full_pointwise_model(
        full_frame,
        joint_best_iteration,
        checkpoint_dir / JOINT_PERSONALIZED_MODEL_FILE,
        JOINT_PERSONALIZED_MEMBER_NAME,
        JOINT_PERSONALIZED_FEATURE_NAMES,
        JOINT_PERSONALIZED_CATEGORICAL_FEATURE_NAMES,
        full_user_vocabulary,
    )
    training_dates_observed = sorted(
        int(value)
        for value in full_frame.get_column("date").unique().to_list()
    )
    del full_frame, full_user_vocabulary
    gc.collect()

    manifest = {
        "version": VERSION,
        "family": (
            "conservative_four_member_rank_ensemble_with_ordered_causal_"
            "multistep_session_history"
        ),
        "parent_attempts": list(PARENT_ATTEMPTS),
        "change": (
            "reconstruct attempt 11's three-member ensemble and shrink it "
            "toward one ordered multi-step causal session-history member "
            "behind a frozen chronological gain gate"
        ),
        "seed": SEED,
        "duration_edges_ms": DURATION_EDGES_MS.tolist(),
        "previous_gap_edges_ms": PREVIOUS_GAP_EDGES_MS.tolist(),
        "session_position_edges": SESSION_POSITION_EDGES.tolist(),
        "session_gap_ms": SESSION_GAP_MS,
        "long_view_threshold_ms": LONG_VIEW_THRESHOLD_MS,
        "recent_proxy_window": RECENT_PROXY_WINDOW,
        "user_vocabulary_file": USER_VOCAB_FILE,
        "ordered_history_contract": JOINT_HISTORY_CONTRACT,
        "zero_previous_duration_policy": "exclude from engagement proxy",
        "empty_proxy_history_rate": 0.5,
        "holdout_days": HOLDOUT_DAYS,
        "holdout_dates": validation_dates,
        "training_dates": training_dates_observed,
        "model_parameters": model_parameters(),
        "holdout_selection": {
            "selection_metric": "mean of within-user GAUC and nDCG@5",
            "metric_contract": (
                "GAUC excludes degenerate users and weights by positive count; "
                "nDCG@5 includes every observed holdout user"
            ),
            "iteration_grid": list(ITERATION_GRID),
            "iteration_tie_break": "higher GAUC, then earlier iteration",
            "base_weight_grid": list(BASE_BLEND_WEIGHT_GRID),
            "simplex_weight_units": ENSEMBLE_WEIGHT_UNITS,
            "minimum_local_personalization_gain": (
                MIN_LOCAL_PERSONALIZATION_GAIN
            ),
            "joint_weight_grid": list(BASE_BLEND_WEIGHT_GRID),
            "minimum_local_joint_gain": MIN_LOCAL_JOINT_GAIN,
            "weight_tie_break": (
                "reconstruct attempt 11 exactly, then higher GAUC and lower "
                "ordered-history-member weight on exact ties"
            ),
            "selected_blend_metrics": selected_blend,
            "reconstructed_attempt10_selection": (
                parent_blend_selection["selected"]
            ),
            "reconstructed_attempt11_selection": (
                personalized_blend_selection["selected"]
            ),
        },
        "members": {
            BASE_MEMBER_NAME: {
                "parent_attempt": 7,
                "model_file": BASE_MODEL_FILE,
                "best_iteration": base_best_iteration,
                "feature_names": BASE_FEATURE_NAMES,
                "categorical_feature_names": BASE_CATEGORICAL_FEATURE_NAMES,
                "mechanism": "row-local context plus backward session state",
            },
            PROXY_MEMBER_NAME: {
                "parent_attempt": 8,
                "model_file": PROXY_MODEL_FILE,
                "best_iteration": proxy_best_iteration,
                "feature_names": FEATURE_NAMES,
                "categorical_feature_names": CATEGORICAL_FEATURE_NAMES,
                "mechanism": (
                    "base member plus causal predecessor engagement proxy"
                ),
            },
            PERSONALIZED_MEMBER_NAME: {
                "parent_attempt": 11,
                "model_file": PERSONALIZED_MODEL_FILE,
                "best_iteration": personalized_best_iteration,
                "feature_names": PERSONALIZED_FEATURE_NAMES,
                "categorical_feature_names": (
                    PERSONALIZED_CATEGORICAL_FEATURE_NAMES
                ),
                "mechanism": (
                    "proxy member plus dense target-free user-by-context "
                    "categorical interactions"
                ),
            },
            JOINT_PERSONALIZED_MEMBER_NAME: {
                "parent_attempt": 11,
                "model_file": JOINT_PERSONALIZED_MODEL_FILE,
                "best_iteration": joint_best_iteration,
                "feature_names": JOINT_PERSONALIZED_FEATURE_NAMES,
                "categorical_feature_names": (
                    JOINT_PERSONALIZED_CATEGORICAL_FEATURE_NAMES
                ),
                "mechanism": (
                    "attempt-11 personalized member plus ordered proxy pattern, "
                    "proxy streak, tab transition, session elapsed time, and "
                    "recent predecessor-duration history"
                ),
            },
        },
        "ensemble": {
            "method": "within_user_average_tie_percentile_rank",
            "base_weight": base_blend_weight,
            "proxy_weight": proxy_blend_weight,
            "personalized_weight": personalized_blend_weight,
            "joint_weight": joint_blend_weight,
            "parent_base_weight": float(
                personalized_blend_selection["selected"]["base_weight"]
            ),
            "parent_proxy_weight": float(
                personalized_blend_selection["selected"]["proxy_weight"]
            ),
            "parent_personalized_weight": float(
                personalized_blend_selection["selected"][
                    "personalized_weight"
                ]
            ),
            "weight_selection": (
                "maximum chronological within-user primary on the frozen "
                "1/8 shrinkage line between the reconstructed attempt-11 "
                "ensemble and the ordered-history member, accepted only after "
                "a 0.0005 local gain; no controller labels are read"
            ),
            "singleton_user_rank": 0.5,
        },
        "source_row_order_contract": "physical parquet row order via __row_order",
        "sequence_order_contract": (
            "within user, ascending time_ms with physical row order as tie-breaker; "
            "features on a current row use only that timestamp and predecessor data"
        ),
        "proxy_contract": (
            "for a valid at-most-30-minute transition with predecessor duration > 0, "
            "compare current_time-minus-previous_time with "
            "min(previous_duration_ms, 18000); summaries include only such past events"
        ),
        "inference_columns": list(EVALUATION_REQUIRED),
        "user_id_usage": (
            "causal sequence derivation, sorting, and rank normalization in "
            "all members; personalized members additionally use a frozen "
            "dense vocabulary; the new member adds only target-free ordered "
            "session state"
        ),
        "forbidden_sequence_features": [
            "next_time_ms",
            "forward_time_delta",
            "future_session_length",
            "current_play_time_ms",
            "current_long_view",
        ],
    }
    write_json(checkpoint_dir / MANIFEST_FILE, manifest)
    write_json(
        checkpoint_dir / DIAGNOSTICS_FILE,
        {
            "chronological_holdout": {
                BASE_MEMBER_NAME: base_holdout_diagnostics,
                PROXY_MEMBER_NAME: proxy_holdout_diagnostics,
                PERSONALIZED_MEMBER_NAME: personalized_holdout_diagnostics,
                JOINT_PERSONALIZED_MEMBER_NAME: joint_holdout_diagnostics,
            },
            "reconstructed_attempt10_ensemble_selection": (
                parent_blend_selection
            ),
            "reconstructed_attempt11_ensemble_selection": (
                personalized_blend_selection
            ),
            "chronological_joint_ensemble_selection": blend_selection,
            "full_refit": {
                BASE_MEMBER_NAME: base_full_diagnostics,
                PROXY_MEMBER_NAME: proxy_full_diagnostics,
                PERSONALIZED_MEMBER_NAME: personalized_full_diagnostics,
                JOINT_PERSONALIZED_MEMBER_NAME: joint_full_diagnostics,
            },
            "holdout_dates": validation_dates,
        },
    )

    prediction_audit = predict_from_checkpoint(
        data_root,
        checkpoint_dir,
        output_dir / "validation_predictions.npy",
    )
    write_json(
        checkpoint_dir / "attempt_prediction_audit.json", prediction_audit
    )


def run_final(args: argparse.Namespace) -> None:
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required in final mode")
    checkpoint_dir = resolve_checkpoint(Path(args.checkpoint_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predict_from_checkpoint(
        Path(args.data_root),
        checkpoint_dir,
        output_dir / "test_predictions.npy",
    )


def main() -> None:
    args = parse_args()
    if args.mode == "attempt":
        run_attempt(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
