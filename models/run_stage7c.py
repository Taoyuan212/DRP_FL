"""Stage 7C: 134-client validation-only single-parameter calibration.

The data cache contains only logical Train and Official Validation rows.  No
test array, prediction, metric, or model-selection path is implemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local_baseline import QuantileMLP, forecast_metrics, make_loader, parameter_count, pinball_loss, predict, set_reproducible_seed
from stage7b_core import (
    apple_reg_coef, clone_state, heur_fedamp_weights, read_train_validation_only,
    sha256_file, state_vector, train_state, vector_state, weighted_state,
)
from run_stage7b import rss_bytes

ALL_CLIENTS = list(range(1, 135))
MODEL_BYTES = 13_833 * 4


@dataclass
class ClientView:
    client_id: int
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray


def read_config(path: Path | None = None) -> dict:
    config = json.loads((path or ROOT / "configs/stage7c.json").read_text(encoding="utf-8"))
    if config["test_read"] is not False or config["allowed_splits"] != ["train", "validation"]:
        raise RuntimeError("Stage 7C Test lock is not active")
    if config["local_epochs"] != 1:
        raise RuntimeError("Stage 7C frozen local-epoch protocol mismatch")
    if config["client_ids"] != "all_134" and config.get("purpose") not in {
            "production_resume_audit", "apple_mu_scale_sanity"}:
        raise RuntimeError("Non-134 cohort is permitted only for a registered Stage 7C-A audit")
    return config


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def state_hash(path: Path) -> str:
    return sha256_file(path)


class TrainValidationStore:
    """Memory-mapped cache to make 134-client calibration feasible without Test."""
    def __init__(self, config: dict):
        self.config = config
        self.root = ROOT / config["output_root"] / "train_validation_cache"
        self.manifest_path = self.root / "manifest.json"

    def prepare(self) -> dict:
        expected = {"client_ids": ALL_CLIENTS, "allowed_splits": ["train", "validation"],
                    "source_processed_dir": self.config["processed_dir"], "lookback_steps": self.config["lookback_steps"]}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if all(manifest.get(k) == v for k, v in expected.items()) and all((self.root / f"client_{c:03d}_train_x.npy").exists() for c in ALL_CLIENTS):
                return manifest
        self.root.mkdir(parents=True, exist_ok=True)
        scaler = json.loads((ROOT / self.config["processed_dir"] / "scaler.json").read_text(encoding="utf-8"))
        rows = []
        for index, cid in enumerate(ALL_CLIENTS, 1):
            raw = ROOT / self.config["processed_dir"] / f"client_{cid:03d}_windows.parquet"
            view = read_train_validation_only(raw, scaler, self.config["lookback_steps"])
            # read_train_validation_only has a hard guard: returned split labels may only be Train/Validation.
            arrays = {"train_x": view.train_x, "train_y": view.train_y,
                      "validation_x": view.validation_x, "validation_y": view.validation_y}
            for name, array in arrays.items():
                destination = self.root / f"client_{cid:03d}_{name}.npy"
                temporary = self.root / f".{destination.name}.tmp.npy"
                np.save(temporary, array)
                os.replace(temporary, destination)
            rows.append({"client_id": cid, "source": str(raw.relative_to(ROOT)), "train_rows": len(view.train_y),
                         "validation_rows": len(view.validation_y), "test_rows_materialized": 0})
            if index % 10 == 0 or index == len(ALL_CLIENTS):
                print(f"cache {index}/134 clients", flush=True)
        manifest = {**expected, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "test_read": False,
                    "cache_rows": rows, "source_scaler_sha256": sha256_file(ROOT / self.config["processed_dir"] / "scaler.json")}
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def get(self, cid: int) -> ClientView:
        load = lambda name: np.load(self.root / f"client_{cid:03d}_{name}.npy", mmap_mode="r")
        return ClientView(cid, load("train_x"), load("train_y"), load("validation_x"), load("validation_y"))


def initial_state(config: dict) -> OrderedDict[str, torch.Tensor]:
    payload = torch.load(ROOT / config["initial_checkpoint"], map_location="cpu", weights_only=True)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model = QuantileMLP(); model.load_state_dict(state)
    if parameter_count(model) != config["predictor_parameters"]:
        raise RuntimeError("Predictor parameter count mismatch")
    return clone_state(state)


def validate(state: dict[str, torch.Tensor], view: ClientView, config: dict) -> dict[str, float]:
    model = QuantileMLP(); model.load_state_dict(state)
    pred, elapsed = predict(model, view.validation_x, config["batch_size"])
    return {**forecast_metrics(view.validation_y, pred, np.asarray(config["quantiles"]), 1.0), "inference_seconds": elapsed}


def score_round(rows: list[dict]) -> tuple[float, float, float, float]:
    pinball = np.asarray([x["Pinball"] for x in rows], dtype=float)
    return (float(pinball.mean()), float(np.quantile(pinball, .95)), float(pinball.max()),
            float(np.mean([x["quantile_crossing_rate"] for x in rows])))


def checkpoint_payload(method: str, parameter: dict, round_number: int, states: list,
                       core_states: list, relations: np.ndarray, score: tuple[float, float, float, float],
                       config: dict, p0: np.ndarray, best: dict | None) -> dict:
    """Artifact-only state; constructed after validation and never fed back into training."""
    return {"method": method, "parameter": parameter, "round": round_number,
            "states": states, "core_states": core_states, "relations": relations,
            "proximal_schedule": {"name": config["apple"]["schedule"], "mu": config["apple"]["mu"],
                                  "dynamic_fraction": config["apple"]["dynamic_fraction"]},
            "p0": p0.copy(), "client_order": ALL_CLIENTS.copy(),
            "config_sha256": json_hash(config), "validation_selection_score": score,
            "code_sha256": {"run_stage7c.py": sha256_file(Path(__file__)),
                            "stage7b_core.py": sha256_file(ROOT/"scripts/stage7b_core.py")},
            "seed": config["seed"], "hyperparameters": {"core_lr": parameter["core_learning_rate"],
                "lr_coef": config["apple"]["relationship_learning_rate"], "mu": config["apple"]["mu"]},
            "model": {"input_dimension": config["input_dimension"], "hidden_dimensions": config["hidden_dimensions"],
                "quantiles": config["quantiles"], "parameter_count": config["predictor_parameters"]},
            "data_identity": {"processed_dir": config["processed_dir"], "allowed_splits": config["allowed_splits"],
                "lookback_steps": config["lookback_steps"], "horizon_minutes": config["horizon_minutes"],
                "batch_size": config["batch_size"], "local_epochs": config["local_epochs"]},
            "best_so_far": best, "optimizer_state": "NOT_APPLICABLE",
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch_cpu": torch.get_rng_state(),
                          "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else "NOT_APPLICABLE"},
            "checkpoint_schema_version": "stage7c-round-boundary-v2",
            "resume_semantics": "round complete; next action is round+1; loader seed is deterministic(seed,round,client_id)"}


def update_best(best: dict | None, score: tuple[float, float, float, float], round_number: int,
                payload: dict, path: Path) -> dict:
    candidate = {"round": round_number, "score": score}
    key = (*score, round_number)
    if best is None or key < (*best["score"], best["round"]):
        torch.save(payload, path)
        return candidate
    return best


def make_fixed_bank(core_states: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    """One frozen (clients x parameters) bank for APPLE's exact mixture."""
    return torch.stack([state_vector(state).float() for state in core_states])


def vector_mapping(vector: torch.Tensor, template: dict[str, torch.Tensor]) -> OrderedDict:
    """Parameter views that retain autograd connectivity to the source vector."""
    mapping, offset = OrderedDict(), 0
    for key, value in template.items():
        size = value.numel()
        mapping[key] = vector[offset:offset + size].reshape_as(value)
        offset += size
    if offset != vector.numel():
        raise ValueError("Parameter vector length mismatch")
    return mapping


def apple_train_client_stacked(fixed_bank: torch.Tensor, core_states: list[dict[str, torch.Tensor]],
                               relation: np.ndarray, own: int, p0: np.ndarray, view: ClientView,
                               config: dict, lr_core: float, round_index: int) -> tuple[OrderedDict, np.ndarray, dict]:
    """Exact APPLE mixture, evaluated as one flattened parameter-vector product."""
    template = QuantileMLP()
    own_params = torch.nn.Parameter(state_vector(core_states[own]).float())
    p = torch.nn.Parameter(torch.tensor(relation, dtype=torch.float32))
    optimizer = torch.optim.SGD([{"params": [own_params], "lr": lr_core},
                                 {"params": [p], "lr": config["apple"]["relationship_learning_rate"]}], momentum=0.9)
    quantiles = torch.tensor(config["quantiles"], dtype=torch.float32)
    coef = apple_reg_coef(config["apple"]["mu"], round_index, config["rounds"], config["apple"]["dynamic_fraction"])
    p0_tensor = torch.tensor(p0, dtype=torch.float32)
    forward = backward = steps = 0; total = seen = 0; started = time.perf_counter()
    first_batch_fingerprint = first_step_loss = first_step_core_hash = first_step_dr_hash = None
    for bx, by in make_loader(view.train_x, view.train_y, config["batch_size"], True,
                              config["seed"] + (round_index + 1) * 100_000 + view.client_id):
        optimizer.zero_grad(set_to_none=True)
        fixed_mix = p @ fixed_bank
        mixed_vector = fixed_mix + p[own] * (own_params - fixed_bank[own])
        mixed = vector_mapping(mixed_vector, core_states[own])
        loss = pinball_loss(functional_call(template, mixed, (bx,)), by, quantiles) + coef * torch.sum((p - p0_tensor) ** 2)
        if first_batch_fingerprint is None:
            first_batch_fingerprint = hashlib.sha256(
                bx.detach().cpu().numpy().tobytes() + by.detach().cpu().numpy().tobytes()).hexdigest()
            first_step_loss = float(loss.detach())
        if not torch.isfinite(loss):
            raise FloatingPointError("APPLE non-finite objective")
        loss.backward(); optimizer.step()
        if first_step_core_hash is None:
            first_step_core_hash = hashlib.sha256(own_params.detach().cpu().numpy().tobytes()).hexdigest()
            first_step_dr_hash = hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
        forward += 1; backward += 1; steps += 1; total += float(loss.detach()) * len(by); seen += len(by)
    return vector_state(own_params.detach(), core_states[own]), p.detach().cpu().numpy().copy(), {"forward_passes": forward, "backward_passes": backward,
        "optimizer_steps": steps, "mean_objective": total / seen, "training_seconds": time.perf_counter()-started, "prox_coef": coef,
        "first_batch_fingerprint": first_batch_fingerprint, "first_step_loss": first_step_loss,
        "first_step_core_hash": first_step_core_hash, "first_step_dr_hash": first_step_dr_hash}


def apple_personalized_state_fast(core_states: list[dict[str, torch.Tensor]], relation: np.ndarray) -> OrderedDict:
    stacked = torch.stack([state_vector(s).double() for s in core_states])
    return vector_state(torch.tensor(relation, dtype=torch.float64) @ stacked, core_states[0])


def _restore_rng(payload: dict) -> None:
    random.setstate(payload["python"]); np.random.set_state(payload["numpy"]); torch.set_rng_state(payload["torch_cpu"])
    if payload["torch_cuda"] != "NOT_APPLICABLE":
        if not torch.cuda.is_available(): raise RuntimeError("Checkpoint requires CUDA RNG but CUDA is unavailable")
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _guard_resume(checkpoint: dict, config: dict, lr_core: float) -> None:
    expected = {
        "seed": config["seed"], "client_order": ALL_CLIENTS,
        "core_lr": lr_core, "lr_coef": config["apple"]["relationship_learning_rate"],
        "mu": config["apple"]["mu"], "parameter_count": config["predictor_parameters"],
        "batch_size": config["batch_size"], "local_epochs": config["local_epochs"],
        "quantiles": config["quantiles"], "processed_dir": config["processed_dir"],
        "allowed_splits": config["allowed_splits"], "lookback_steps": config["lookback_steps"],
        "horizon_minutes": config["horizon_minutes"]}
    actual = {
        "seed": checkpoint.get("seed"), "client_order": checkpoint.get("client_order"),
        "core_lr": checkpoint.get("hyperparameters",{}).get("core_lr"),
        "lr_coef": checkpoint.get("hyperparameters",{}).get("lr_coef"),
        "mu": checkpoint.get("hyperparameters",{}).get("mu"),
        "parameter_count": checkpoint.get("model",{}).get("parameter_count"),
        "batch_size": checkpoint.get("data_identity",{}).get("batch_size"),
        "local_epochs": checkpoint.get("data_identity",{}).get("local_epochs"),
        "quantiles": checkpoint.get("model",{}).get("quantiles"),
        "processed_dir": checkpoint.get("data_identity",{}).get("processed_dir"),
        "allowed_splits": checkpoint.get("data_identity",{}).get("allowed_splits"),
        "lookback_steps": checkpoint.get("data_identity",{}).get("lookback_steps"),
        "horizon_minutes": checkpoint.get("data_identity",{}).get("horizon_minutes")}
    mismatches=[key for key in expected if actual[key] != expected[key]]
    if checkpoint.get("checkpoint_schema_version") != "stage7c-round-boundary-v2": mismatches.append("checkpoint_schema_version")
    if checkpoint.get("config_sha256") != json_hash(config): mismatches.append("config_sha256")
    if mismatches: raise RuntimeError("RESUME COMPATIBILITY HARD FAIL: "+",".join(mismatches))


def _read_history(out: Path, completed: int, client_count: int) -> tuple[list, list, list, list]:
    specs=[("round_metrics.csv",client_count),("personalization_statistics.csv",client_count),
           ("communication_report.csv",1),("training_trace.csv",client_count)]
    result=[]
    for name,per_round in specs:
        path=out/name
        if not path.exists(): raise RuntimeError(f"RESUME OUTPUT HARD FAIL: missing {name}")
        rows=pd.read_csv(path).to_dict("records"); rounds=[int(x["round"]) for x in rows]
        if len(rows)!=completed*per_round or sorted(set(rounds))!=list(range(1,completed+1)):
            raise RuntimeError(f"RESUME OUTPUT HARD FAIL: inconsistent {name}")
        result.append(rows)
    return tuple(result)


def _append_rows(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path,mode="a",header=not path.exists(),index=False)


def run_apple_candidate(config: dict, store: TrainValidationStore, init: dict, lr_core: float, out: Path,
                        resume_from: Path | None = None, max_round: int | None = None) -> dict:
    n = len(ALL_CLIENTS)
    counts = np.array([len(store.get(cid).train_y) for cid in ALL_CLIENTS], dtype=float); prior = counts / counts.sum()
    if resume_from:
        checkpoint=torch.load(resume_from,map_location="cpu",weights_only=False); _guard_resume(checkpoint,config,lr_core)
        completed=int(checkpoint["round"]); core=checkpoint["core_states"]; relations=np.asarray(checkpoint["relations"])
        metrics,statistics,communication,traces=_read_history(out,completed,n)
        best=checkpoint["best_so_far"]; _restore_rng(checkpoint["rng_state"]); start=completed+1
        print(json.dumps({"RESUME MODE":True,"checkpoint":str(resume_from),"completed_round":completed,
            "next_round":start,"seed":config["seed"],"client_count":n,"core_lr":lr_core,
            "lr_coef":config["apple"]["relationship_learning_rate"],"mu":config["apple"]["mu"],
            "config_hash":json_hash(config)}),flush=True)
    else:
        core=[clone_state(init) for _ in ALL_CLIENTS]; relations=np.tile(prior,(n,1))
        metrics=[];statistics=[];communication=[];traces=[];best=None;start=1
        for name in ["round_metrics.csv","personalization_statistics.csv","communication_report.csv","training_trace.csv"]:
            if (out/name).exists(): raise RuntimeError(f"FRESH OUTPUT HARD FAIL: {name} already exists")
    stop=min(config["rounds"],max_round or config["rounds"]); best_path=out/"selected_round_candidate.pt"; payload=None
    for rnd in range(start,stop+1):
        began=time.perf_counter(); fixed=make_fixed_bank(core); next_core=[]; next_rel=[]; records=[]; peak=rss_bytes()
        for own, cid in enumerate(ALL_CLIENTS):
            state, rel, record = apple_train_client_stacked(fixed, core, relations[own], own, prior, store.get(cid), config, lr_core, rnd-1)
            next_core.append(state); next_rel.append(rel); records.append(record); peak=max(peak,rss_bytes())
        core=next_core; relations=np.stack(next_rel)
        # Each row is one ordinary weighted aggregation over the frozen
        # round-r core bank.  Build that bank once rather than stacking it
        # separately for every target client.
        personal_began=time.perf_counter()
        core_vectors=torch.stack([state_vector(state).double() for state in core])
        personalized_vectors=torch.as_tensor(relations,dtype=torch.float64) @ core_vectors
        states=[vector_state(personalized_vectors[i],core[0]) for i in range(n)]
        personal_time=time.perf_counter()-personal_began
        round_rows=[]
        for cid,state in zip(ALL_CLIENTS,states):
            item={"method":"apple","core_learning_rate":lr_core,"round":rnd,"client_id":cid,**validate(state,store.get(cid),config)}
            metrics.append(item); round_rows.append(item)
        for own,cid in enumerate(ALL_CLIENTS):
            absw=np.abs(relations[own]); probability=absw/max(absw.sum(),1e-12)
            statistics.append({"round":rnd,"client_id":cid,"self_weight":float(relations[own,own]),"entropy_abs":float(-(probability*np.log(probability+1e-12)).sum()),
                "top_collaborator":int(ALL_CLIENTS[int(np.argmax(absw))]),"weight_min":float(relations[own].min()),"weight_max":float(relations[own].max()),
                "one_hot":bool(absw.max()/max(absw.sum(),1e-12)>0.999)})
        score=score_round(round_rows)
        wall=time.perf_counter()-began
        comm={"round":rnd,"server_to_client_bytes":n*n*MODEL_BYTES,"client_to_server_bytes":n*MODEL_BYTES,
            "extra_scalar_vector_bytes":n*n*4,"model_downloads":n*n,"local_training_seconds":sum(x["training_seconds"] for x in records),
            "extra_personalization_seconds":personal_time,"server_aggregation_seconds":0.0,"wall_seconds":wall,"rss_bytes":peak,
            "forward_passes":sum(x["forward_passes"] for x in records),"backward_passes":sum(x["backward_passes"] for x in records),"optimizer_steps":sum(x["optimizer_steps"] for x in records),
            "macro_validation_pinball":score[0],"client_pinball_p95":score[1],"worst_client_pinball":score[2],"crossing_rate":score[3]}
        communication.append(comm)
        trace=[{"round":rnd,"client_id":cid,"loader_seed":config["seed"]+rnd*100_000+cid,
                "first_batch_fingerprint":record["first_batch_fingerprint"],"first_step_loss":record["first_step_loss"],
                "first_step_core_hash":record["first_step_core_hash"],"first_step_dr_hash":record["first_step_dr_hash"]}
               for cid,record in zip(ALL_CLIENTS,records)]; traces.extend(trace)
        candidate={"round":rnd,"score":score}
        is_best=best is None or (*score,rnd)<(*best["score"],best["round"])
        if is_best: best=candidate
        payload=checkpoint_payload("apple",{"core_learning_rate":lr_core},rnd,states,core,relations,score,config,prior,best)
        if is_best: torch.save(payload,best_path)
        torch.save(payload,out/f"checkpoint_round_{rnd:03d}.pt")
        _append_rows(out/"round_metrics.csv",round_rows);_append_rows(out/"personalization_statistics.csv",statistics[-n:])
        _append_rows(out/"communication_report.csv",[comm]);_append_rows(out/"training_trace.csv",trace)
        print(f"APPLE lr={lr_core:g} round {rnd:02d}/40 macro={score[0]:.8f}",flush=True)
    if payload is None: raise RuntimeError("No round executed")
    torch.save(payload,out/"final_round_checkpoint.pt")
    return persist_candidate(out,"apple",{"core_learning_rate":lr_core},metrics,statistics,communication,best,config)


def fedamp_centers_fast(states: list[dict[str, torch.Tensor]], self_weight: float, sigma: float) -> tuple[np.ndarray, list[OrderedDict]]:
    weights=heur_fedamp_weights(states,self_weight,sigma)
    vectors=torch.stack([state_vector(state).double() for state in states])
    centers_vector=torch.tensor(weights,dtype=torch.float64) @ vectors
    return weights,[vector_state(centers_vector[i],states[0]) for i in range(len(states))]


def run_fedamp_candidate(config: dict, store: TrainValidationStore, init: dict, proximal: float, out: Path) -> dict:
    states=[clone_state(init) for _ in ALL_CLIENTS]; metrics=[];statistics=[];communication=[];best=None;best_path=out/"selected_round_candidate.pt"
    n=len(ALL_CLIENTS)
    for rnd in range(1,config["rounds"]+1):
        began=time.perf_counter(); weights,centers=fedamp_centers_fast(states,.5,1.0); server=time.perf_counter()-began; local=[];records=[];peak=rss_bytes()
        for i,cid in enumerate(ALL_CLIENTS):
            state,record=train_state(centers[i],store.get(cid).train_x,store.get(cid).train_y,config,config["seed"]+rnd*100_000+cid,
                prox_center=centers[i],prox_mu=proximal)
            local.append(state);records.append(record);peak=max(peak,rss_bytes())
        states=local;round_rows=[]
        for cid,state in zip(ALL_CLIENTS,states):
            item={"method":"fedamp","proximal_coefficient":proximal,"round":rnd,"client_id":cid,**validate(state,store.get(cid),config)}
            metrics.append(item);round_rows.append(item)
        for i,cid in enumerate(ALL_CLIENTS):
            statistics.append({"round":rnd,"client_id":cid,"self_weight":float(weights[i,i]),"weight_sum":float(weights[i].sum()),
                "weight_min":float(weights[i].min()),"weight_max":float(weights[i].max()),
                "top_external_client":int(ALL_CLIENTS[int(np.argmax(np.where(np.arange(n)==i,-np.inf,weights[i])))])})
        score=score_round(round_rows)
        best=update_best(best,score,rnd,{"method":"fedamp","parameter":{"proximal_coefficient":proximal},"round":rnd,"states":states,"weights":weights},best_path)
        wall=time.perf_counter()-began
        communication.append({"round":rnd,"server_to_client_bytes":n*MODEL_BYTES,"client_to_server_bytes":n*MODEL_BYTES,
            "extra_scalar_vector_bytes":n*n*4,"model_downloads":n,"local_training_seconds":sum(x["training_seconds"] for x in records),
            "extra_personalization_seconds":0.0,"server_aggregation_seconds":server,"wall_seconds":wall,"rss_bytes":peak,
            "forward_passes":sum(x["forward_passes"] for x in records),"backward_passes":sum(x["backward_passes"] for x in records),"optimizer_steps":sum(x["optimizer_steps"] for x in records),
            "macro_validation_pinball":score[0],"client_pinball_p95":score[1],"worst_client_pinball":score[2],"crossing_rate":score[3]})
        print(f"FedAMP prox={proximal:.12g} round {rnd:02d}/40 macro={score[0]:.8f}",flush=True)
    return persist_candidate(out,"fedamp",{"proximal_coefficient":proximal},metrics,statistics,communication,best,config)


def persist_candidate(out: Path, method: str, parameter: dict, metrics: list, statistics: list, communication: list, best: dict, config: dict) -> dict:
    if method != "apple":
        pd.DataFrame(metrics).to_csv(out/"round_metrics.csv",index=False)
        pd.DataFrame(statistics).to_csv(out/"personalization_statistics.csv",index=False)
        pd.DataFrame(communication).to_csv(out/"communication_report.csv",index=False)
    resource={"method":method,"parameter":parameter,"total_runtime_seconds":float(sum(x["wall_seconds"] for x in communication)),
        "cpu_peak_rss_bytes":int(max(x["rss_bytes"] for x in communication)),"gpu_peak_memory_bytes":0,
        "communication_bytes":int(sum(x["server_to_client_bytes"]+x["client_to_server_bytes"]+x["extra_scalar_vector_bytes"] for x in communication)),
        "forward_passes":int(sum(x["forward_passes"] for x in communication)),"backward_passes":int(sum(x["backward_passes"] for x in communication)),"optimizer_steps":int(sum(x["optimizer_steps"] for x in communication)),
        "selected_round_candidate":best,"test_rows_read":0,"test_read":False}
    (out/"resource_report.json").write_text(json.dumps(resource,indent=2),encoding="utf-8")
    return {"method":method,"parameter":parameter,"best_round":best["round"],"selection_score":best["score"],"resource":resource}


def candidate_name(method: str, value: float) -> str:
    return f"{method}_{value:.12g}".replace(".","p").replace("-","m")


def provenance(out: Path, config: dict, method: str, parameter: dict, cache_manifest: dict) -> None:
    initial=ROOT/config["initial_checkpoint"]
    code_files=[ROOT/"scripts/run_stage7c.py",ROOT/"scripts/stage7b_core.py"]
    payload={"stage":"7C","method":method,"parameter":parameter,"config_sha256":json_hash(config),"initialization_sha256":sha256_file(initial),
        "code_sha256":{str(p.relative_to(ROOT)):sha256_file(p) for p in code_files},"cache_manifest_sha256":json_hash(cache_manifest),
        "test_read":False,"allowed_splits":["train","validation"],"seed":2026,"clients":134,"full_participation":True,"rounds":40,"local_epochs":1,
        "platform":platform.platform(),"torch":torch.__version__}
    (out/"provenance.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    (out/"config.json").write_text(json.dumps({**config,"candidate_method":method,"candidate_parameter":parameter},indent=2),encoding="utf-8")
    pd.DataFrame([{"client_id":cid,"initialization_sha256":payload["initialization_sha256"],"parameter_count":13_833,"match":True} for cid in ALL_CLIENTS]).to_csv(out/"baseline_initialization_audit.csv",index=False)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--method",choices=["apple","fedamp"],required=True);parser.add_argument("--value",type=float,required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/stage7c.json")
    parser.add_argument("--resume-from",type=Path);parser.add_argument("--max-round",type=int);args=parser.parse_args()
    config=read_config(args.config);set_reproducible_seed(config["seed"],config["torch_num_threads"])
    global ALL_CLIENTS
    ALL_CLIENTS=list(range(1,135)) if config["client_ids"]=="all_134" else list(config["client_ids"])
    if args.resume_from and args.method!="apple": raise RuntimeError("--resume-from is currently valid only for APPLE")
    if args.method=="apple" and args.value not in config["apple"]["core_learning_rate_candidates"]: raise ValueError("Unauthorized APPLE candidate")
    allowed=[config["fedamp"]["proximal_coefficient_default"]*x for x in config["fedamp"]["proximal_coefficient_multipliers"]]
    if args.method=="fedamp" and not any(math.isclose(args.value,x,rel_tol=1e-12) for x in allowed): raise ValueError("Unauthorized FedAMP candidate")
    store=TrainValidationStore(config);manifest=store.prepare();init=initial_state(config)
    out=ROOT/config["output_root"]/args.method/candidate_name(args.method,args.value);out.mkdir(parents=True,exist_ok=True)
    parameter={"core_learning_rate":args.value} if args.method=="apple" else {"proximal_coefficient":args.value}
    provenance(out,config,args.method,parameter,manifest)
    summary=run_apple_candidate(config,store,init,args.value,out,args.resume_from,args.max_round) if args.method=="apple" else run_fedamp_candidate(config,store,init,args.value,out)
    (out/"candidate_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
