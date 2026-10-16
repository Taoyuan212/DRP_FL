"""Shared round-boundary checkpoint policy for Stage 7D strong baselines."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SCHEMA = "stage7d-round-boundary-v1"

def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

def rng_state() -> dict:
    return {"python":random.getstate(),"numpy":np.random.get_state(),
            "torch_cpu":torch.get_rng_state(),
            "torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else "NOT_APPLICABLE"}

def restore_rng(state: dict) -> None:
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] != "NOT_APPLICABLE":
        if not torch.cuda.is_available(): raise RuntimeError("RESUME HARD FAIL: CUDA RNG unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])

def compatibility(checkpoint: dict, config: dict, method: str) -> None:
    expected={"schema":SCHEMA,"method":method,"config_hash":config_hash(config),
              "seed":config["seed"],"client_ids":config["client_ids"],
              "parameter_count":config["predictor_parameters"],"batch_size":config["batch_size"],
              "local_epochs":config["local_epochs"],"quantiles":config["quantiles"],
              "lookback":config["lookback_steps"],"horizon":config["horizon_minutes"],
              "method_config":config["methods"][method]}
    actual={key:checkpoint.get(key) for key in expected}
    bad=[key for key in expected if actual[key]!=expected[key]]
    if bad: raise RuntimeError("RESUME COMPATIBILITY HARD FAIL: "+",".join(bad))

def append_rows(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path,mode="a",header=not path.exists(),index=False)

def verify_history(path: Path, completed: int, clients: int) -> list[dict]:
    if not path.exists(): raise RuntimeError("RESUME OUTPUT HARD FAIL: missing history")
    frame=pd.read_csv(path)
    if len(frame)!=completed*clients or frame.duplicated(["round","client_id"]).any():
        raise RuntimeError("RESUME OUTPUT HARD FAIL: duplicate/missing history")
    if sorted(frame["round"].unique())!=list(range(1,completed+1)):
        raise RuntimeError("RESUME OUTPUT HARD FAIL: non-contiguous history")
    return frame.to_dict("records")
