from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from harness.spec import canonical_json_bytes


class LedgerError(RuntimeError):
    pass


class EventLedger:
    """A tiny append-only ledger implemented as atomically-created event files."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.events_dir = self.run_dir / "events"
        self.lock_path = self.run_dir / "controller.lock"

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self) -> list[dict[str, Any]]:
        if not self.events_dir.exists():
            return []
        paths = sorted(self.events_dir.glob("*.json"))
        events: list[dict[str, Any]] = []
        for expected_id, path in enumerate(paths, start=1):
            expected_name = f"{expected_id:06d}.json"
            if path.name != expected_name:
                raise LedgerError(f"event sequence gap: expected {expected_name}, found {path.name}")
            try:
                event = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise LedgerError(f"invalid event file: {path}") from exc
            if event.get("event_id") != expected_id:
                raise LedgerError(f"event_id mismatch in {path}")
            if not isinstance(event.get("type"), str):
                raise LedgerError(f"event type missing in {path}")
            events.append(event)
        return events

    def append(self, event_type: str, **payload: Any) -> dict[str, Any]:
        self.events_dir.mkdir(parents=True, exist_ok=True)
        event_id = len(self.read()) + 1
        now = dt.datetime.now(dt.timezone.utc)
        event = {
            "event_id": event_id,
            "type": event_type,
            "timestamp_unix": now.timestamp(),
            "timestamp_utc": now.isoformat(),
            **payload,
        }
        temporary = self.events_dir / f".{event_id:06d}.tmp-{os.getpid()}"
        final = self.events_dir / f"{event_id:06d}.json"
        if final.exists():
            raise LedgerError(f"refusing to overwrite event {event_id}")
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(event))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)
        directory_fd = os.open(self.events_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return event


def export_jsonl(run_dir: Path, destination: Path) -> None:
    ledger = EventLedger(run_dir)
    events = ledger.read()
    with destination.open("wb") as handle:
        for event in events:
            handle.write(canonical_json_bytes(event))
