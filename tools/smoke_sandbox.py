from __future__ import annotations

import json
import tempfile
from pathlib import Path

from harness.sandbox import BubblewrapRunner
from harness.spec import ROOT


def main() -> None:
    source = ROOT / "tests" / "fixtures" / "sandbox_candidate"
    with tempfile.TemporaryDirectory(prefix="techjam-sandbox-") as temporary:
        root = Path(temporary)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "ready.txt").write_text("synthetic checkpoint")
        outputs = {}
        for mode in ("attempt", "final"):
            result = BubblewrapRunner().run(
                source_dir=source,
                output_dir=root / f"output-{mode}",
                cache_dir=root / f"cache-{mode}",
                mode=mode,
                timeout_seconds=120,
                max_output_bytes=64 * 1024 * 1024,
                checkpoint_dir=checkpoint if mode == "final" else None,
                allow_gpu=True,
            )
            evidence_path = root / f"output-{mode}" / "isolation.json"
            evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else None
            outputs[mode] = {
                "status": result.status,
                "returncode": result.returncode,
                "wall_seconds": result.wall_seconds,
                "stderr_tail": result.stderr_tail,
                "evidence": evidence,
            }
            if result.status != "ok" or evidence != {
                "raw_data_blocked": True,
                "network_blocked": True,
                "train_visibility_correct": True,
            }:
                print(json.dumps(outputs, sort_keys=True))
                raise SystemExit(1)
        print(json.dumps(outputs, sort_keys=True))


if __name__ == "__main__":
    main()
