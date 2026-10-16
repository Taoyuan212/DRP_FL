"""Run only the paired Local baselines needed for Stage-4 multi-seed NT analysis.

This intentionally reuses the frozen Stage-3 training implementation and does
not rerun the centralized or cross-client diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from local_baseline import save_json, set_reproducible_seed, sha256_file
from run_local_baseline import initialize_checkpoint, train_all_local


ROOT = Path(__file__).parents[1]


def state_dict_sha256(path: Path) -> str:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for name, value in saved["state_dict"].items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_baseline.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028])
    args = parser.parse_args()

    config_path = ROOT / args.config
    frozen_config = json.loads(config_path.read_text(encoding="utf-8"))
    for seed in args.seeds:
        config = dict(frozen_config)
        config["seed"] = seed
        config["output_dir"] = f"results/local/h1h/seed_{seed}"
        output_dir = ROOT / config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        set_reproducible_seed(seed, config["torch_num_threads"])
        initial_path = initialize_checkpoint(config, output_dir)
        results = train_all_local(config, output_dir, initial_path)
        summary = {
            "stage": 4,
            "role": "paired_local_reference_for_multiseed_negative_transfer",
            "seed": seed,
            "clients": int(len(results)),
            "mean_pinball": float(results["Pinball"].mean()),
            "median_pinball": float(results["Pinball"].median()),
            "initial_checkpoint_file_sha256": sha256_file(initial_path),
            "initial_state_dict_sha256": state_dict_sha256(initial_path),
            "frozen_stage3_config": str(config_path.resolve()),
            "frozen_stage3_config_sha256": sha256_file(config_path),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__)),
            "stage3_runner_sha256": sha256_file(ROOT / "scripts/run_local_baseline.py"),
            "stage3_component_sha256": sha256_file(ROOT / "scripts/local_baseline.py"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "device": "cpu",
        }
        save_json(output_dir / "stage4_paired_local_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
