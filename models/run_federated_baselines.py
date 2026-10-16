"""Run Stage-4 FedAvg/FedProx pilots and full-participation baselines."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
sys.path.insert(0, str(ROOT / "scripts"))

from federated_baselines import (  # noqa: E402
    cosine_matrix,
    cosine_statistics,
    load_client_cache,
    model_vector,
    state_vector,
    train_one_client,
    weighted_average_states,
)
from local_baseline import (  # noqa: E402
    forecast_metrics,
    initial_model,
    parameter_count,
    predict,
    save_json,
    set_reproducible_seed,
    sha256_file,
)


def rel(value: str) -> Path:
    return ROOT / value


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def initial_state(config: dict, seed: int, seed_dir: Path) -> tuple[dict[str, torch.Tensor], Path]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / f"initial_model_seed_{seed}.pt"
    if seed == 2026:
        source = rel(config["stage3_results_dir"]) / "initial_model_seed_2026.pt"
        if not path.exists():
            shutil.copy2(source, path)
    elif not path.exists():
        set_reproducible_seed(seed, config["torch_num_threads"])
        model = initial_model(config)
        torch.save({
            "state_dict": model.state_dict(), "seed": seed,
            "parameter_count": parameter_count(model),
            "architecture": "76-64-ReLU-64-ReLU-64-ReLU-9",
        }, path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return {key: value.detach().cpu().clone() for key, value in saved["state_dict"].items()}, path


def evaluate_validation(state: dict[str, torch.Tensor], cache: dict, config: dict, round_number: int) -> tuple[float, float, list[dict]]:
    model = initial_model(config)
    model.load_state_dict(state)
    rows = []
    weighted_pinball = 0.0
    weighted_crossing = 0.0
    total = 0
    for client_id, data in cache.items():
        predictions, _ = predict(model, data.validation_x, config["batch_size"])
        metrics = forecast_metrics(data.validation_y, predictions, np.asarray(config["quantiles"]), config["pinaw_normalizer_pu"])
        count = len(data.validation_y)
        rows.append({
            "round": round_number, "client_id": client_id,
            "validation_samples": count, "validation_Pinball": metrics["Pinball"],
            "validation_crossing_rate": metrics["quantile_crossing_rate"],
        })
        weighted_pinball += metrics["Pinball"] * count
        weighted_crossing += metrics["quantile_crossing_rate"] * count
        total += count
    return weighted_pinball / total, weighted_crossing / total, rows


def run_method(
    method: str,
    mu: float,
    client_ids: list[int],
    cache: dict,
    config: dict,
    seed: int,
    method_dir: Path,
    store_updates: bool,
) -> dict:
    method_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir = method_dir / "rounds"
    rounds_dir.mkdir(exist_ok=True)
    update_dir = method_dir / "updates"
    cosine_dir = method_dir / "cosine_matrices"
    if store_updates:
        update_dir.mkdir(exist_ok=True)
        cosine_dir.mkdir(exist_ok=True)
    seed_dir = method_dir.parent
    global_state, initialization_path = initial_state(config, seed, seed_dir)
    initial_hash = sha256_file(initialization_path)
    initial_validation, initial_crossing, validation_rows = evaluate_validation(global_state, cache, config, 0)
    best_validation = initial_validation
    best_round = 0
    best_state = {key: value.clone() for key, value in global_state.items()}
    stale = 0
    summaries = []
    client_train_rows = []
    total_local_steps = 0
    all_finite = True
    max_sync_error = 0.0
    max_manual_aggregation_error = 0.0
    started = time.perf_counter()
    early = config["round_early_stopping"]

    for round_number in range(1, config["max_communication_rounds"] + 1):
        round_started = time.perf_counter()
        before_vector = state_vector(global_state).numpy()
        local_states = []
        counts = []
        updates = []
        round_rows = []
        for client_id in client_ids:
            state, diagnostics = train_one_client(global_state, cache[client_id], config, round_number, seed, mu)
            local_states.append(state)
            counts.append(diagnostics["train_samples"])
            updates.append((state_vector(state).numpy() - before_vector).astype(np.float32))
            diagnostics.update({"round": round_number, "method": method, "mu": mu})
            round_rows.append(diagnostics)
            total_local_steps += int(diagnostics["local_steps"])
            all_finite = all_finite and bool(diagnostics["finite"])
            max_sync_error = max(max_sync_error, diagnostics["synchronization_max_abs_error"])
        aggregated = weighted_average_states(local_states, counts)
        aggregate_vector = state_vector(aggregated).numpy()
        stacked_states = np.stack([state_vector(state).numpy().astype(np.float64) for state in local_states])
        manual_vector = np.average(stacked_states, axis=0, weights=np.asarray(counts, dtype=np.float64))
        manual_error = float(np.max(np.abs(aggregate_vector.astype(np.float64) - manual_vector)))
        max_manual_aggregation_error = max(max_manual_aggregation_error, manual_error)
        global_state = aggregated
        updates_array = np.stack(updates)
        cosines = cosine_matrix(updates_array)
        cos_stats = cosine_statistics(cosines)
        update_norms = np.linalg.norm(updates_array, axis=1)
        validation_pinball, validation_crossing, current_validation_rows = evaluate_validation(global_state, cache, config, round_number)
        validation_rows.extend(current_validation_rows)
        round_seconds = time.perf_counter() - round_started
        summary = {
            "round": round_number,
            "method": method,
            "mu": mu,
            "participating_clients": len(client_ids),
            "train_samples": int(sum(counts)),
            "local_steps": int(sum(row["local_steps"] for row in round_rows)),
            "sample_weighted_train_pinball": float(np.average([row["mean_train_pinball"] for row in round_rows], weights=counts)),
            "sample_weighted_validation_pinball": validation_pinball,
            "sample_weighted_validation_crossing_rate": validation_crossing,
            "global_parameter_norm": float(np.linalg.norm(aggregate_vector)),
            "global_round_update_norm": float(np.linalg.norm(aggregate_vector - before_vector)),
            "average_client_update_norm": float(np.mean(update_norms)),
            "median_client_update_norm": float(np.median(update_norms)),
            **cos_stats,
            "synchronization_max_abs_error": max(row["synchronization_max_abs_error"] for row in round_rows),
            "manual_aggregation_max_abs_error": manual_error,
            "round_seconds": round_seconds,
        }
        summaries.append(summary)
        client_train_rows.extend(round_rows)
        torch.save({"state_dict": global_state, "round": round_number, "method": method, "mu": mu}, rounds_dir / f"round_{round_number:03d}.pt")
        if store_updates:
            np.save(update_dir / f"updates_round_{round_number:03d}.npy", updates_array)
            np.save(cosine_dir / f"cosine_round_{round_number:03d}.npy", cosines.astype(np.float32))
        print(
            f"{method} mu={mu:g} round={round_number:02d} val={validation_pinball:.6f} "
            f"cos<0={cos_stats['cosine_fraction_negative']:.3f} sec={round_seconds:.1f}",
            flush=True,
        )
        if validation_pinball < best_validation - early["min_delta"]:
            best_validation = validation_pinball
            best_round = round_number
            best_state = {key: value.clone() for key, value in global_state.items()}
            stale = 0
        else:
            stale += 1
            if stale >= early["patience"]:
                break

    runtime = time.perf_counter() - started
    torch.save({
        "state_dict": best_state, "best_round": best_round, "method": method,
        "mu": mu, "seed": seed, "initial_checkpoint_sha256": initial_hash,
    }, method_dir / "selected_checkpoint.pt")
    pd.DataFrame(summaries).to_csv(method_dir / "table_round_summary.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(method_dir / "table_client_validation_by_round.csv", index=False)
    pd.DataFrame(client_train_rows).to_csv(method_dir / "table_client_train_by_round.csv", index=False)
    result = {
        "method": method, "mu": mu, "seed": seed, "clients": client_ids,
        "initial_validation_pinball": initial_validation,
        "initial_validation_crossing_rate": initial_crossing,
        "best_validation_pinball": best_validation, "best_round": best_round,
        "rounds_executed": len(summaries), "local_epochs": config["local_epochs"],
        "total_local_steps": total_local_steps, "runtime_seconds": runtime,
        "average_round_seconds": runtime / len(summaries),
        "all_finite": all_finite, "max_synchronization_abs_error": max_sync_error,
        "max_manual_aggregation_abs_error": max_manual_aggregation_error,
        "optimizer_state_policy": config["optimizer_state_policy"],
        "initial_checkpoint_sha256": initial_hash,
        "store_updates": store_updates,
    }
    save_json(method_dir / "training_summary.json", result)
    return result


def evaluate_test(method_dir: Path, cache: dict, config: dict, method: str, mu: float, seed: int) -> pd.DataFrame:
    checkpoint = torch.load(method_dir / "selected_checkpoint.pt", map_location="cpu", weights_only=True)
    model = initial_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    predictions_dir = method_dir / "test_predictions_by_client"
    predictions_dir.mkdir(exist_ok=True)
    rows = []
    for client_id, data in cache.items():
        predictions, inference_seconds = predict(model, data.test_x, config["batch_size"])
        metrics = forecast_metrics(data.test_y, predictions, np.asarray(config["quantiles"]), config["pinaw_normalizer_pu"])
        rows.append({
            "client_id": client_id, "method": method, "mu": mu, "seed": seed,
            "best_round": checkpoint["best_round"], "test_samples": len(data.test_y),
            **metrics, "CRPS": metrics["CRPS_approx"], "inference_time": inference_seconds,
        })
        frame = data.arrays.frame.iloc[data.test_indices][["issue_time", "target_time", "target_patv_pu"]].reset_index(drop=True)
        for index, quantile in enumerate(config["quantiles"]):
            frame[f"q{quantile:.1f}"] = predictions[:, index]
        frame.to_parquet(predictions_dir / f"client_{client_id:03d}.parquet", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(method_dir / "table_client_results.csv", index=False)
    return result


def pilot(config: dict) -> dict:
    seed = config["primary_seed"]
    seed_dir = rel(config["output_root"]) / f"seed_{seed}"
    pilot_dir = seed_dir / "pilot_12"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    scaler = load_json(rel(config["processed_dir"]) / "scaler.json")
    cache = load_client_cache(rel(config["processed_dir"]), config["pilot_clients"], scaler, config["lookback_steps"])
    runs = []
    fedavg_dir = pilot_dir / "fedavg"
    if (fedavg_dir / "training_summary.json").exists():
        runs.append(load_json(fedavg_dir / "training_summary.json"))
    else:
        runs.append(run_method("FedAvg", 0.0, config["pilot_clients"], cache, config, seed, fedavg_dir, True))
    for mu in config["fedprox_mu_candidates"]:
        method_dir = pilot_dir / f"fedprox_mu_{mu:g}"
        if (method_dir / "training_summary.json").exists():
            runs.append(load_json(method_dir / "training_summary.json"))
        else:
            runs.append(run_method("FedProx", mu, config["pilot_clients"], cache, config, seed, method_dir, True))
    frame = pd.DataFrame(runs)
    frame.to_csv(pilot_dir / "table_pilot_method_selection.csv", index=False)
    prox = frame.loc[frame.method.eq("FedProx")].sort_values(["best_validation_pinball", "mu"])
    selected_mu = float(prox.iloc[0]["mu"])
    thresholds = config["pilot_pass"]
    checks = {
        "all_finite": bool(frame.all_finite.all()),
        "all_synchronized": bool((frame.max_synchronization_abs_error <= thresholds["all_sync_max_abs_error_max"]).all()),
        "manual_aggregation_matches": bool((frame.max_manual_aggregation_abs_error <= thresholds["aggregation_manual_max_abs_error_max"]).all()),
        "all_validation_improves": bool((frame.best_validation_pinball < frame.initial_validation_pinball).all()),
        "cosine_recorded": all((pilot_dir / ("fedavg" if row.method == "FedAvg" else f"fedprox_mu_{row.mu:g}") / "table_round_summary.csv").exists() for row in frame.itertuples()),
    }
    passed = all(checks.values())
    decision = {"passed": passed, "selected_mu": selected_mu, "checks": checks, "selection_metric": "minimum best sample-weighted validation Pinball"}
    save_json(pilot_dir / "pilot_decision.json", decision)
    report = "# Federated Pilot Report\n\n" + f"Decision: **{'PASS' if passed else 'FAIL'}**\n\nSelected FedProx mu: **{selected_mu:g}**, using validation only.\n\n" + frame.to_csv(index=False)
    (pilot_dir / "federated_pilot_report.md").write_text(report, encoding="utf-8")
    del cache
    gc.collect()
    return decision


def full_seed(config: dict, seed: int, evaluate: bool = True) -> dict:
    seed_dir = rel(config["output_root"]) / f"seed_{seed}"
    pilot_decision = load_json(rel(config["output_root"]) / f"seed_{config['primary_seed']}" / "pilot_12/pilot_decision.json")
    if not pilot_decision["passed"]:
        raise RuntimeError("Pilot failed; full training blocked")
    mu = float(pilot_decision["selected_mu"])
    scaler = load_json(rel(config["processed_dir"]) / "scaler.json")
    client_ids = list(range(1, 135))
    print(f"Loading 134-client cache for seed {seed}", flush=True)
    cache = load_client_cache(rel(config["processed_dir"]), client_ids, scaler, config["lookback_steps"])
    outputs = {}
    for method, method_mu, directory_name in (("FedAvg", 0.0, "fedavg"), ("FedProx", mu, "fedprox")):
        method_dir = seed_dir / directory_name
        summary_path = method_dir / "training_summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            print(f"Reusing {method} seed {seed}", flush=True)
        else:
            store = seed == config["store_full_updates_seed"]
            summary = run_method(method, method_mu, client_ids, cache, config, seed, method_dir, store)
        outputs[method] = summary
        table_path = method_dir / "table_client_results.csv"
        if evaluate and not table_path.exists():
            evaluate_test(method_dir, cache, config, method, method_mu, seed)
    save_json(seed_dir / "seed_training_summary.json", outputs)
    del cache
    gc.collect()
    return outputs


def write_manifest(config_path: Path, config: dict, status: str) -> None:
    root = rel(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "run_manifest.json", {
        "stage": 4, "status": status, "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path), "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__)),
        "component_sha256": sha256_file(ROOT / "scripts/federated_baselines.py"),
        "stage2_manifest_sha256": sha256_file(rel("results/stage2/dataset_manifest.csv")),
        "stage3_local_table_sha256": sha256_file(rel(config["stage3_results_dir"]) / "table_local_client_results.csv"),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "device": "cpu", "cuda_available": torch.cuda.is_available(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/federated_baselines.json")
    parser.add_argument("--mode", choices=("pilot", "full-primary", "additional-seeds"), required=True)
    args = parser.parse_args()
    config_path = rel(args.config)
    config = load_json(config_path)
    set_reproducible_seed(config["primary_seed"], config["torch_num_threads"])
    if args.mode == "pilot":
        decision = pilot(config)
        write_manifest(config_path, config, "PILOT_PASSED" if decision["passed"] else "PILOT_FAILED")
        if not decision["passed"]:
            raise SystemExit(1)
    elif args.mode == "full-primary":
        full_seed(config, config["primary_seed"], True)
        write_manifest(config_path, config, "PRIMARY_COMPLETE")
    else:
        for seed in config["additional_seeds"]:
            full_seed(config, seed, True)
        write_manifest(config_path, config, "THREE_SEEDS_COMPLETE")


if __name__ == "__main__":
    main()
