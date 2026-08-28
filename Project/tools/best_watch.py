#!/usr/bin/env python3
"""Best-experiment watcher (Track 2) — mechanical per-best audit trigger.

Same design as Track 3's champion_watch: hook-invoked after every shell
command; when the journal's best FINALIZABLE experiment changes (error-free +
sealed, last tied max — the same rule the bench's final gate uses), launches a
detached codex audit and records the verdict to Project/audits/verdicts.jsonl.
Lives outside the frozen bench; never writes results."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "Project" / "results" / "JOURNAL.jsonl"
CACHE = Path(__file__).parent / ".best_cache.json"
AUDIT_LOG_DIR = ROOT / "Project" / "audits" / "auto"


def best_finalizable():
    if not JOURNAL.exists():
        return None
    entries = []
    for line in JOURNAL.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    starts = [e for e in entries if e.get("type") == "run_start"]
    iters = []
    seen = not starts
    for e in entries:
        if starts and e.get("entry_id") == starts[0]["entry_id"]:
            seen = True
        if seen and e.get("type") == "iteration":
            iters.append(e)
    pool = [e for e in iters if e.get("valid_metrics") and e.get("error") is None
            and e.get("sealed_test_scores")]
    if not pool:
        return None
    top = max(e["valid_metrics"]["primary"] for e in pool)
    return [e for e in pool if e["valid_metrics"]["primary"] == top][-1]["entry_id"]


def main() -> int:
    best = best_finalizable()
    if best is None:
        return 0
    try:
        cache = set(json.loads(CACHE.read_text()))
    except Exception:
        cache = set()
    if best in cache:
        return 0
    CACHE.write_text(json.dumps(sorted(cache | {best})))
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = AUDIT_LOG_DIR / f"audit_{best}.log"
    subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "audit_best.py"), best],
        stdin=subprocess.DEVNULL, stdout=open(log, "a"),
        stderr=subprocess.STDOUT, start_new_session=True, cwd=str(ROOT),
    )
    print(f"[best-watch] new best experiment {best} — background audit launched "
          f"(log: {log.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
