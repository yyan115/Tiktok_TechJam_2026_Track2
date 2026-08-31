import argparse
import json
import socket
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--cache-dir", type=Path, required=True)
parser.add_argument("--checkpoint-dir", type=Path)
args = parser.parse_args()

raw_path = Path(
    "/home/yy/Desktop/Repos/Tiktok_TechJam_2026_Track2/"
    "datasets/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv"
)
raw_blocked = False
try:
    raw_path.open("rb")
except OSError:
    raw_blocked = True

network_blocked = False
sock = socket.socket()
sock.settimeout(0.2)
try:
    sock.connect(("1.1.1.1", 53))
except OSError:
    network_blocked = True
finally:
    sock.close()

if not raw_blocked or not network_blocked:
    raise RuntimeError("sandbox isolation check failed")
train_visible = (args.data_root / "train.parquet").exists()
train_visibility_correct = train_visible if args.mode == "attempt" else not train_visible
if not train_visibility_correct:
    raise RuntimeError("training data visibility differs from the mode contract")
if args.mode == "final" and (
    args.checkpoint_dir is None or not (args.checkpoint_dir / "ready.txt").is_file()
):
    raise RuntimeError("frozen checkpoint is unavailable in final mode")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the candidate sandbox")
x = torch.ones((16, 16), device="cuda")
gpu_sum = float((x @ x).sum().cpu())

rows = pq.ParquetFile(args.data_root / "evaluation_features.parquet").metadata.num_rows
name = "validation_predictions.npy" if args.mode == "attempt" else "test_predictions.npy"
np.save(args.output_dir / name, np.zeros(rows, dtype=np.float32))
checkpoint = args.output_dir / "checkpoint"
checkpoint.mkdir()
(checkpoint / "smoke.json").write_text(json.dumps({"gpu_sum": gpu_sum}))
(args.output_dir / "isolation.json").write_text(
    json.dumps(
        {
            "raw_data_blocked": raw_blocked,
            "network_blocked": network_blocked,
            "train_visibility_correct": train_visibility_correct,
        }
    )
)
