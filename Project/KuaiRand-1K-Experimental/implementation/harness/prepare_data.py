from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from harness.spec import (
    DEFAULT_DERIVED_DIR,
    DEFAULT_RAW_DIR,
    BenchmarkSpec,
    canonical_json_bytes,
    sha256_file,
)


class DataPreparationError(RuntimeError):
    pass


def _reader(path: Path, columns: list[str] | None = None) -> pacsv.CSVStreamingReader:
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=32 * 1024 * 1024, use_threads=True),
        convert_options=pacsv.ConvertOptions(include_columns=columns),
    )


def _csv_to_parquet(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in _reader(source):
            if writer is None:
                writer = pq.ParquetWriter(destination, batch.schema, compression="zstd")
            writer.write_batch(batch)
            rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise DataPreparationError(f"CSV contains no rows: {source}")
    return rows


def _with_split_row_id(batch: pa.RecordBatch, start: int) -> pa.RecordBatch:
    row_ids = pa.array(np.arange(start, start + batch.num_rows, dtype=np.int64))
    return batch.add_column(0, "row_id", row_ids)


def _split_future_features(
    source: Path,
    validation_destination: Path,
    test_destination: Path,
    feature_columns: tuple[str, ...],
    validation_dates: tuple[int, int],
    test_dates: tuple[int, int],
) -> tuple[int, int]:
    validation_destination.parent.mkdir(parents=True, exist_ok=True)
    test_destination.parent.mkdir(parents=True, exist_ok=True)
    validation_writer: pq.ParquetWriter | None = None
    test_writer: pq.ParquetWriter | None = None
    validation_rows = 0
    test_rows = 0
    try:
        for batch in _reader(source, list(feature_columns)):
            dates = batch.column(batch.schema.get_field_index("date")).to_numpy(zero_copy_only=False)
            validation_mask = (dates >= validation_dates[0]) & (dates <= validation_dates[1])
            test_mask = (dates >= test_dates[0]) & (dates <= test_dates[1])
            if not np.all(validation_mask | test_mask):
                bad_dates = np.unique(dates[~(validation_mask | test_mask)])
                raise DataPreparationError(f"future log contains dates outside the contract: {bad_dates}")

            if validation_mask.any():
                selected = batch.filter(pa.array(validation_mask))
                selected = _with_split_row_id(selected, validation_rows)
                if validation_writer is None:
                    validation_writer = pq.ParquetWriter(
                        validation_destination, selected.schema, compression="zstd"
                    )
                validation_writer.write_batch(selected)
                validation_rows += selected.num_rows

            if test_mask.any():
                selected = batch.filter(pa.array(test_mask))
                selected = _with_split_row_id(selected, test_rows)
                if test_writer is None:
                    test_writer = pq.ParquetWriter(
                        test_destination, selected.schema, compression="zstd"
                    )
                test_writer.write_batch(selected)
                test_rows += selected.num_rows
    finally:
        if validation_writer is not None:
            validation_writer.close()
        if test_writer is not None:
            test_writer.close()
    if validation_writer is None or test_writer is None:
        raise DataPreparationError("future log did not produce both validation and test rows")
    return validation_rows, test_rows


def _extract_public_validation_targets(
    source: Path, destination: Path, validation_dates: tuple[int, int]
) -> int:
    """Store public-validation targets for the trusted controller only.

    The source combines validation and test dates. The trusted preparer necessarily
    parses that file, but it filters before persisting targets. No test target array,
    statistic, metric, or report is produced.
    """

    users: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in _reader(source, ["user_id", "date", "long_view"]):
        dates = batch.column(batch.schema.get_field_index("date")).to_numpy(zero_copy_only=False)
        mask = (dates >= validation_dates[0]) & (dates <= validation_dates[1])
        if not mask.any():
            continue
        selected = batch.filter(pa.array(mask))
        users.append(
            selected.column(selected.schema.get_field_index("user_id"))
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        labels.append(
            selected.column(selected.schema.get_field_index("long_view"))
            .to_numpy(zero_copy_only=False)
            .astype(np.int8, copy=False)
        )
    if not users:
        raise DataPreparationError("no public-validation targets found")
    destination.mkdir(parents=True, exist_ok=True)
    user_values = np.concatenate(users)
    label_values = np.concatenate(labels)
    np.save(destination / "validation_users.npy", user_values, allow_pickle=False)
    np.save(destination / "validation_labels.npy", label_values, allow_pickle=False)
    return len(label_values)


def _output_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_derived_data(
    spec: BenchmarkSpec,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_DERIVED_DIR,
) -> dict[str, Any]:
    spec.validate_raw_files(raw_dir)
    if output_dir.exists():
        raise DataPreparationError(f"derived data already exists: {output_dir}")

    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    public = temporary / "public"
    private = temporary / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)

    files = spec.raw_files
    train_source = raw_dir / files["train_log"]
    future_source = raw_dir / files["future_log"]
    try:
        train_rows = _csv_to_parquet(train_source, public / "train.parquet")
        _csv_to_parquet(raw_dir / files["user_features"], public / "user_features.parquet")
        _csv_to_parquet(raw_dir / files["video_features"], public / "video_features.parquet")
        validation_rows, test_rows = _split_future_features(
            future_source,
            public / "validation_features.parquet",
            public / "test_features.parquet",
            spec.future_feature_columns,
            spec.splits["validation"],
            spec.splits["test"],
        )
        target_rows = _extract_public_validation_targets(
            future_source, private, spec.splits["validation"]
        )

        actual_rows = {
            "train": train_rows,
            "validation": validation_rows,
            "test": test_rows,
        }
        if actual_rows != spec.expected_rows:
            raise DataPreparationError(
                f"row counts differ from benchmark spec: expected={spec.expected_rows} actual={actual_rows}"
            )
        if target_rows != validation_rows:
            raise DataPreparationError("validation feature/target row counts differ")

        output_files = sorted(path for path in temporary.rglob("*") if path.is_file())
        manifest = {
            "benchmark_id": spec.raw["benchmark_id"],
            "benchmark_spec_sha256": spec.digest,
            "rows": actual_rows,
            "raw_inputs": {
                role: {
                    "file": name,
                    "bytes": (raw_dir / name).stat().st_size,
                    "sha256": sha256_file(raw_dir / name),
                }
                for role, name in files.items()
            },
            "outputs": [_output_record(path, temporary) for path in output_files],
            "hidden_test_targets_cached": False,
            "excluded_files": spec.raw["excluded_files"],
        }
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the leakage-safe KuaiRand-1K view")
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DERIVED_DIR)
    args = parser.parse_args()
    spec = BenchmarkSpec.load(args.spec) if args.spec else BenchmarkSpec.load()
    manifest = build_derived_data(spec, args.raw_dir, args.output_dir)
    print(
        json.dumps(
            {
                "status": "prepared",
                "output": str(args.output_dir),
                "rows": manifest["rows"],
                "hidden_test_targets_cached": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
