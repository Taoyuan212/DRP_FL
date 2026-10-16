"""Run Stage 3: local probabilistic forecasting baseline and diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Keep Matplotlib's cache inside the writable research workspace.  This avoids
# host-profile permission warnings without changing any numerical computation.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parents[1] / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from local_baseline import (
    INPUT_SOURCE_COLUMNS,
    QuantileMLP,
    empirical_quantile_predictions,
    forecast_metrics,
    initial_model,
    load_client_arrays,
    mean_loss,
    parameter_count,
    predict,
    save_json,
    set_reproducible_seed,
    sha256_file,
    train_model,
)


ROOT = Path(__file__).parents[1]


def rel(path: str) -> Path:
    return ROOT / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def client_path(processed_dir: Path, client_id: int) -> Path:
    return processed_dir / f"client_{client_id:03d}_windows.parquet"


def write_markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns.astype(str)) + " |"
    divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def run_pytest(output_dir: Path) -> tuple[bool, str]:
    command = [str(ROOT / ".venv/Scripts/python.exe"), "-m", "pytest", "tests/test_local_baseline.py", "-q"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    log = completed.stdout + completed.stderr
    (output_dir / "metric_unit_tests.log").write_text(log, encoding="utf-8")
    return completed.returncode == 0, log.strip()


def audit_dataset(config: dict, output_dir: Path) -> dict:
    processed = rel(config["processed_dir"])
    manifest = pd.read_csv(rel(config["stage2_results_dir"]) / "dataset_manifest.csv")
    scaler_path = processed / "scaler.json"
    scaler = load_json(scaler_path)
    split_definition = load_json(rel(config["stage2_results_dir"]) / "split_definition.json")
    rng = np.random.default_rng(config["seed"])
    audit_rows: list[dict] = []
    client_rows: list[dict] = []
    target_min = math.inf
    target_max = -math.inf
    below_zero = 0
    above_one = 0
    target_count = 0
    failures: list[str] = []
    sample_candidates: list[tuple[int, int]] = []

    for client_id in manifest["client_id"].astype(int):
        path = client_path(processed, client_id)
        if not path.exists():
            failures.append(f"missing processed window file: {path.name}")
            continue
        arrays = load_client_arrays(path, scaler, config["lookback_steps"])
        frame = arrays.frame
        if arrays.inputs.shape != (len(frame), config["input_dimension"]):
            failures.append(f"client {client_id}: input dimension mismatch")
        if not np.isfinite(arrays.inputs).all() or not np.isfinite(arrays.targets).all():
            failures.append(f"client {client_id}: non-finite model tensors")
        if not frame["is_valid"].all() or not frame["horizon_minutes"].eq(config["horizon_minutes"]).all():
            failures.append(f"client {client_id}: invalid/horizon-mismatched row in window file")
        expected_issue = frame["history_start"] + pd.to_timedelta((config["lookback_steps"] - 1) * config["frequency_minutes"], unit="min")
        if not expected_issue.equals(frame["issue_time"]):
            failures.append(f"client {client_id}: history order/cadence endpoint mismatch")
        if not (frame["issue_time"] < frame["target_time"]).all() or not (frame["history_start"] < frame["target_time"]).all():
            failures.append(f"client {client_id}: non-causal timestamp")
        split_sets = {name: set(frame.loc[frame["split"].eq(name), "issue_time"]) for name in ("train", "validation", "test")}
        if any(split_sets[a].intersection(split_sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            failures.append(f"client {client_id}: issue-time split overlap")
        manifest_row = manifest.loc[manifest["client_id"].eq(client_id)].iloc[0]
        actual_counts = frame["split"].value_counts()
        count_ok = (
            int(actual_counts.get("train", 0)) == int(manifest_row["train_sample_count"])
            and int(actual_counts.get("validation", 0)) == int(manifest_row["validation_sample_count"])
            and int(actual_counts.get("test", 0)) == int(manifest_row["test_sample_count"])
        )
        if not count_ok:
            failures.append(f"client {client_id}: manifest sample count mismatch")
        values = arrays.targets.astype(np.float64)
        target_min = min(target_min, float(values.min()))
        target_max = max(target_max, float(values.max()))
        below_zero += int(np.sum(values < 0))
        above_one += int(np.sum(values > 1))
        target_count += len(values)
        client_rows.append({
            "client_id": client_id,
            "rows": len(frame),
            "input_dimension": arrays.inputs.shape[1],
            "manifest_counts_match": count_ok,
            "causal_timestamps": bool((frame["issue_time"] < frame["target_time"]).all()),
            "target_min_pu": float(values.min()),
            "target_max_pu": float(values.max()),
        })
        chosen = rng.choice(len(frame), size=min(2, len(frame)), replace=False)
        sample_candidates.extend((client_id, int(index)) for index in chosen)

    # Fixed-seed random backcheck over distinct clients, supplemented to exactly 100 rows.
    chosen_candidates = rng.choice(len(sample_candidates), size=config["audit_random_samples"], replace=False)
    for audit_number, candidate_index in enumerate(chosen_candidates, start=1):
        client_id, row_index = sample_candidates[int(candidate_index)]
        arrays = load_client_arrays(client_path(processed, client_id), scaler, config["lookback_steps"])
        row = arrays.frame.iloc[row_index]
        patv = np.asarray(row["input_patv_clean_kw"], dtype=np.float32)
        wspd = np.asarray(row["input_wspd_ms"], dtype=np.float32)
        direct_history = np.stack((
            (patv - scaler["Patv_clean_kw"]["mean"]) / scaler["Patv_clean_kw"]["std_population"],
            (wspd - scaler["Wspd"]["mean"]) / scaler["Wspd"]["std_population"],
        ), axis=1).reshape(-1)
        direct = np.concatenate((direct_history, row[list(INPUT_SOURCE_COLUMNS[2:])].to_numpy(dtype=np.float32)))
        input_match = bool(np.allclose(direct, arrays.inputs[row_index], rtol=1e-6, atol=1e-6))
        target_match = bool(np.isclose(float(row["target_patv_pu"]), arrays.targets[row_index]))
        causal = bool(row["history_start"] + pd.Timedelta(minutes=350) == row["issue_time"] < row["target_time"])
        if not (input_match and target_match and causal):
            failures.append(f"random backcheck {audit_number} failed")
        audit_rows.append({
            "audit_number": audit_number,
            "client_id": client_id,
            "processed_row_index": row_index,
            "issue_time": row["issue_time"],
            "target_time": row["target_time"],
            "split": row["split"],
            "input_match": input_match,
            "target_match": target_match,
            "causal": causal,
        })

    scaler_ok = (
        scaler["method"] == "shared_training_only_scaling"
        and scaler["fit_start"] == split_definition["splits"]["train"][0]
        and scaler["fit_end_exclusive"] == split_definition["splits"]["train"][1]
    )
    if not scaler_ok:
        failures.append("scaler provenance is not confined to frozen training interval")
    pd.DataFrame(client_rows).to_csv(output_dir / "table_dataset_audit_by_client.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output_dir / "table_dataloader_backcheck_100.csv", index=False)
    unit_ok, unit_log = run_pytest(output_dir)
    if not unit_ok:
        failures.append("metric/model unit tests failed")
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "clients_checked": len(client_rows),
        "samples_checked": target_count,
        "input_source_columns": list(INPUT_SOURCE_COLUMNS),
        "input_dimension": config["input_dimension"],
        "target_min_pu": target_min,
        "target_max_pu": target_max,
        "target_below_zero_count": below_zero,
        "target_above_one_count": above_one,
        "target_was_clipped_in_stage3": False,
        "scaler_training_only": scaler_ok,
        "scaler_sha256": sha256_file(scaler_path),
        "random_backchecks_passed": int(sum(row["input_match"] and row["target_match"] and row["causal"] for row in audit_rows)),
        "metric_unit_tests_passed": unit_ok,
        "failures": failures,
    }
    save_json(output_dir / "dataset_audit_summary.json", summary)
    (output_dir / "metric_validation_report.md").write_text(
        "# Metric Validation Report\n\n"
        f"Status: **{'PASS' if unit_ok else 'FAIL'}**\n\n"
        "The tests use hand-checkable synthetic values for MAE, RMSE, mean Pinball Loss, "
        "discrete-quantile CRPS approximation, PICP, signed PINAW, and sample-level quantile "
        "crossing rate. CRPS is defined as twice the mean quantile score over q=0.1,...,0.9; "
        "it is an approximation based on the available quantiles, not exact continuous-distribution CRPS. "
        "Predicted quantiles are never sorted. PINAW uses raw q-upper minus q-lower width so crossings are not hidden.\n\n"
        "```text\n" + unit_log + "\n```\n",
        encoding="utf-8",
    )
    return summary


def initialize_checkpoint(config: dict, output_dir: Path) -> Path:
    path = output_dir / f"initial_model_seed_{config['seed']}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("seed") == config["seed"] and saved.get("parameter_count") == 13833:
            return path
    set_reproducible_seed(config["seed"], config["torch_num_threads"])
    model = initial_model(config)
    torch.save({
        "state_dict": model.state_dict(),
        "seed": config["seed"],
        "parameter_count": parameter_count(model),
        "architecture": "76-64-ReLU-64-ReLU-64-ReLU-9",
    }, path)
    return path


def fresh_model(config: dict, initial_path: Path) -> QuantileMLP:
    model = initial_model(config)
    saved = torch.load(initial_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    return model


def train_client(client_id: int, config: dict, output_dir: Path, initial_path: Path, write_artifacts: bool = True) -> dict:
    scaler = load_json(rel(config["processed_dir"]) / "scaler.json")
    arrays = load_client_arrays(client_path(rel(config["processed_dir"]), client_id), scaler, config["lookback_steps"])
    train_x, train_y, _ = arrays.select("train")
    val_x, val_y, _ = arrays.select("validation")
    test_x, test_y, test_indices = arrays.select("test")
    model = fresh_model(config, initial_path)
    quantiles_tensor = torch.tensor(config["quantiles"], dtype=torch.float32)
    initial_validation_loss = mean_loss(model, val_x, val_y, quantiles_tensor, config["batch_size"])
    model, history, best_epoch, train_seconds, nonfinite_training = train_model(
        model, train_x, train_y, val_x, val_y, config
    )
    predictions, inference_seconds = predict(model, test_x, config["batch_size"])
    quantiles = np.asarray(config["quantiles"])
    metrics = forecast_metrics(test_y, predictions, quantiles, config["pinaw_normalizer_pu"])
    # Persistence uses the last issue-time Patv, converted back from the frozen shared scaler.
    last_patv_pu = (
        test_x[:, 70] * scaler["Patv_clean_kw"]["std_population"] + scaler["Patv_clean_kw"]["mean"]
    ) / config["target_scale_kw"]
    persistence_mae = float(np.mean(np.abs(last_patv_pu - test_y)))
    persistence_rmse = float(np.sqrt(np.mean((last_patv_pu - test_y) ** 2)))
    climatology = empirical_quantile_predictions(train_y, len(test_y), quantiles)
    climatology_metrics = forecast_metrics(test_y, climatology, quantiles, config["pinaw_normalizer_pu"])
    result = {
        "client_id": client_id,
        "train_samples": len(train_y),
        "validation_samples": len(val_y),
        "test_samples": len(test_y),
        "best_epoch": best_epoch,
        **{key: value for key, value in metrics.items() if isinstance(value, (int, float, bool, np.bool_))},
        "training_time": train_seconds,
        "inference_time": inference_seconds,
        "inference_ms_per_sample": inference_seconds * 1000 / len(test_y),
        "initial_validation_loss": initial_validation_loss,
        "first_train_loss": history[0]["train_loss"] if history else float("nan"),
        "last_train_loss": history[-1]["train_loss"] if history else float("nan"),
        "best_validation_loss": min(row["validation_loss"] for row in history) if history else float("nan"),
        "max_gradient_norm": max(row["max_gradient_norm"] for row in history) if history else float("nan"),
        "nonfinite_training": nonfinite_training,
        "epochs_run": len(history),
        "persistence_MAE": persistence_mae,
        "persistence_RMSE": persistence_rmse,
        "persistence_MAE_improvement_fraction": (persistence_mae - metrics["MAE"]) / persistence_mae,
        "persistence_RMSE_improvement_fraction": (persistence_rmse - metrics["RMSE"]) / persistence_rmse,
        "climatology_Pinball": climatology_metrics["Pinball"],
        "climatology_CRPS_approx": climatology_metrics["CRPS_approx"],
        "initial_checkpoint_sha256": sha256_file(initial_path),
    }
    if write_artifacts:
        client_dir = output_dir / f"client_{client_id:03d}"
        client_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "client_id": client_id, "seed": config["seed"], "best_epoch": best_epoch}, client_dir / "checkpoint.pt")
        pd.DataFrame(history).to_csv(client_dir / "training_history.csv", index=False)
        save_json(client_dir / "best_epoch.json", {"best_epoch": best_epoch, "best_validation_loss": result["best_validation_loss"]})
        save_json(client_dir / "metrics.json", result)
        save_json(client_dir / "runtime.json", {"training_seconds": train_seconds, "inference_seconds": inference_seconds, "device": "cpu"})
        prediction_frame = arrays.frame.iloc[test_indices][["issue_time", "target_time", "target_patv_pu"]].reset_index(drop=True)
        for index, quantile in enumerate(quantiles):
            prediction_frame[f"q{quantile:.1f}"] = predictions[:, index]
        prediction_frame.to_parquet(client_dir / "test_predictions.parquet", index=False)
    del arrays, train_x, val_x, test_x, model
    return result


def pilot_pass(results: pd.DataFrame, config: dict, metric_tests_passed: bool) -> tuple[bool, dict]:
    thresholds = config["pilot"]["pass_thresholds"]
    checks = {
        "all_training_losses_decline": bool((results["last_train_loss"] < results["first_train_loss"]).all()),
        "all_validation_losses_improve": bool((results["best_validation_loss"] < results["initial_validation_loss"]).all()),
        "finite_predictions_all_clients": bool(results["finite_predictions"].all() and (~results["nonfinite_training"]).all()),
        "median_persistence_mae_improvement": bool(results["persistence_MAE_improvement_fraction"].median() >= thresholds["median_persistence_mae_improvement_fraction_min"]),
        "median_persistence_rmse_improvement": bool(results["persistence_RMSE_improvement_fraction"].median() >= thresholds["median_persistence_rmse_improvement_fraction_min"]),
        "enough_clients_improve_both_persistence_metrics": bool(((results["persistence_MAE_improvement_fraction"] > 0) & (results["persistence_RMSE_improvement_fraction"] > 0)).sum() >= thresholds["clients_improving_both_persistence_mae_and_rmse_min"]),
        "mean_pinball_better_than_training_only_climatology": bool(results["Pinball"].mean() < results["climatology_Pinball"].mean()),
        "median_q50_pearson": bool(results["q50_pearson"].median() >= thresholds["median_q50_pearson_min"]),
        "training_time_acceptable": bool(results["training_time"].max() <= thresholds["single_client_training_time_seconds_max"]),
        "metric_tests_pass": bool(metric_tests_passed),
        "crossing_rate_recorded": bool(results["quantile_crossing_rate"].notna().all()),
    }
    return all(checks.values()), checks


def run_pilot(config: dict, output_dir: Path, initial_path: Path, audit: dict) -> tuple[bool, pd.DataFrame]:
    existing_table = output_dir / "table_pilot_client_results.csv"
    existing_decision = output_dir / "pilot_decision.json"
    expected_clients = set(config["pilot"]["client_ids"])
    if existing_table.exists() and existing_decision.exists():
        saved_results = pd.read_csv(existing_table)
        decision = load_json(existing_decision)
        artifact_clients = {
            client_id for client_id in expected_clients
            if (output_dir / f"client_{client_id:03d}" / "checkpoint.pt").exists()
            and (output_dir / f"client_{client_id:03d}" / "metrics.json").exists()
        }
        if decision.get("passed") and set(saved_results["client_id"].astype(int)) == expected_clients and artifact_clients == expected_clients:
            print("PILOT reusing previously passed seed-2026 artifacts", flush=True)
            return True, saved_results
    rows = []
    for client_id in config["pilot"]["client_ids"]:
        print(f"PILOT client={client_id:03d}", flush=True)
        rows.append(train_client(client_id, config, output_dir, initial_path, True))
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "table_pilot_client_results.csv", index=False)
    passed, checks = pilot_pass(results, config, audit["metric_unit_tests_passed"])
    selection = pd.DataFrame({
        "client_id": config["pilot"]["client_ids"],
        "selection_role": config["pilot"]["selection_roles"],
    })
    selection.to_csv(output_dir / "table_pilot_selection.csv", index=False)
    report = (
        "# Pilot Training Report\n\n"
        f"Overall decision: **{'PASS' if passed else 'FAIL — full Local-134 training blocked'}**\n\n"
        "Selection was frozen from Stage-1/2 statistics before training:\n\n"
        + write_markdown_table(selection) + "\n\n"
        "Pass checks:\n\n"
        + write_markdown_table(pd.DataFrame([{"check": key, "pass": value} for key, value in checks.items()])) + "\n\n"
        "Client diagnostics (per-unit metrics):\n\n"
        + write_markdown_table(results[["client_id", "best_epoch", "Pinball", "MAE", "RMSE", "persistence_MAE_improvement_fraction", "persistence_RMSE_improvement_fraction", "q50_pearson", "quantile_crossing_rate", "training_time", "max_gradient_norm"]].round(6))
        + "\n\nDevice: CPU only; GPU memory is not applicable. No NaN/Inf repair, output clipping, quantile sorting, or test-based tuning was performed.\n"
    )
    (output_dir / "pilot_training_report.md").write_text(report, encoding="utf-8")
    save_json(output_dir / "pilot_decision.json", {"passed": passed, "checks": checks, "thresholds": config["pilot"]["pass_thresholds"]})
    return passed, results


def train_all_local(config: dict, output_dir: Path, initial_path: Path) -> pd.DataFrame:
    existing_path = output_dir / "table_local_client_results.csv"
    completed: dict[int, dict] = {}
    if existing_path.exists():
        completed = {int(row["client_id"]): row for row in pd.read_csv(existing_path).to_dict("records")}
    for client_id in range(1, 135):
        metrics_path = output_dir / f"client_{client_id:03d}" / "metrics.json"
        checkpoint_path = output_dir / f"client_{client_id:03d}" / "checkpoint.pt"
        if client_id not in completed and metrics_path.exists() and checkpoint_path.exists():
            saved = load_json(metrics_path)
            if saved.get("initial_checkpoint_sha256") == sha256_file(initial_path):
                completed[client_id] = saved
    if completed:
        pd.DataFrame([completed[key] for key in sorted(completed)]).to_csv(existing_path, index=False)
    for client_id in range(1, 135):
        if client_id not in completed:
            print(f"LOCAL client={client_id:03d}/134", flush=True)
            completed[client_id] = train_client(client_id, config, output_dir, initial_path, True)
            pd.DataFrame([completed[key] for key in sorted(completed)]).to_csv(existing_path, index=False)
    return pd.DataFrame([completed[key] for key in sorted(completed)])


def load_pooled(config: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    scaler = load_json(rel(config["processed_dir"]) / "scaler.json")
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for client_id in range(1, 135):
        arrays = load_client_arrays(client_path(rel(config["processed_dir"]), client_id), scaler, config["lookback_steps"])
        inputs, targets, _ = arrays.select(split)
        x_parts.append(inputs)
        y_parts.append(targets)
    return np.concatenate(x_parts), np.concatenate(y_parts)


def train_centralized(config: dict, output_dir: Path, initial_path: Path) -> pd.DataFrame:
    central_dir = output_dir / "centralized_reference"
    central_dir.mkdir(exist_ok=True)
    table_path = output_dir / "table_centralized_client_results.csv"
    checkpoint_path = central_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        print("CENTRALIZED loading pooled train/validation arrays", flush=True)
        train_x, train_y = load_pooled(config, "train")
        val_x, val_y = load_pooled(config, "validation")
        model = fresh_model(config, initial_path)
        started = time.perf_counter()
        model, history, best_epoch, train_seconds, nonfinite = train_model(model, train_x, train_y, val_x, val_y, config)
        torch.save({"state_dict": model.state_dict(), "seed": config["seed"], "best_epoch": best_epoch}, checkpoint_path)
        pd.DataFrame(history).to_csv(central_dir / "training_history.csv", index=False)
        save_json(central_dir / "training_summary.json", {
            "label": "Centralized Oracle / Reference", "best_epoch": best_epoch,
            "training_seconds": train_seconds, "wall_seconds_including_array_loading_excluded": time.perf_counter() - started,
            "nonfinite": nonfinite, "train_samples": len(train_y), "validation_samples": len(val_y),
        })
        del train_x, train_y, val_x, val_y
    model = fresh_model(config, checkpoint_path)
    scaler = load_json(rel(config["processed_dir"]) / "scaler.json")
    rows: list[dict] = []
    prediction_dir = central_dir / "test_predictions_by_client"
    prediction_dir.mkdir(exist_ok=True)
    for client_id in range(1, 135):
        print(f"CENTRALIZED EVAL client={client_id:03d}/134", flush=True)
        arrays = load_client_arrays(client_path(rel(config["processed_dir"]), client_id), scaler, config["lookback_steps"])
        inputs, targets, indices = arrays.select("test")
        predictions, inference_time = predict(model, inputs, config["batch_size"])
        metrics = forecast_metrics(targets, predictions, np.asarray(config["quantiles"]), config["pinaw_normalizer_pu"])
        rows.append({"client_id": client_id, "test_samples": len(targets), **metrics, "inference_time": inference_time})
        frame = arrays.frame.iloc[indices][["issue_time", "target_time", "target_patv_pu"]].reset_index(drop=True)
        for index, quantile in enumerate(config["quantiles"]):
            frame[f"q{quantile:.1f}"] = predictions[:, index]
        frame.to_parquet(prediction_dir / f"client_{client_id:03d}.parquet", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(table_path, index=False)
    return result


def select_cross_clients(config: dict, output_dir: Path) -> tuple[list[int], np.ndarray]:
    matrices = []
    names = ["matrix_clean_patv_distance.csv", "matrix_clean_wspd_distance.csv", "matrix_clean_conditional_powercurve_distance.csv"]
    for name in names:
        matrix = pd.read_csv(rel(config["stage2_results_dir"]) / name, index_col=0).to_numpy(dtype=float)
        upper = matrix[np.triu_indices_from(matrix, 1)]
        scale = np.nanmedian(upper[upper > 0])
        matrices.append(matrix / scale)
    composite = np.nanmean(np.stack(matrices), axis=0)
    chosen = list(dict.fromkeys(config["pilot"]["client_ids"]))
    while len(chosen) < config["cross_client_count"]:
        candidates = [client for client in range(1, 135) if client not in chosen]
        min_distances = {client: float(np.nanmin(composite[client - 1, np.asarray(chosen) - 1])) for client in candidates}
        chosen.append(sorted(candidates, key=lambda client: (-min_distances[client], client))[0])
    table = pd.DataFrame({"selection_order": range(1, len(chosen) + 1), "client_id": chosen, "source": ["pilot anchor"] * 6 + ["Stage-2 composite-distance maximin"] * (len(chosen) - 6)})
    table.to_csv(output_dir / "table_cross_client_selection.csv", index=False)
    return chosen, composite


def cross_client_evaluation(config: dict, output_dir: Path, clients: list[int], composite: np.ndarray) -> tuple[pd.DataFrame, dict]:
    scaler = load_json(rel(config["processed_dir"]) / "scaler.json")
    test_data = {}
    for client_id in clients:
        arrays = load_client_arrays(client_path(rel(config["processed_dir"]), client_id), scaler, config["lookback_steps"])
        test_data[client_id] = arrays.select("test")[:2]
    rows = []
    for source in clients:
        checkpoint = output_dir / f"client_{source:03d}" / "checkpoint.pt"
        model = fresh_model(config, checkpoint)
        for target in clients:
            inputs, targets = test_data[target]
            predictions, inference_time = predict(model, inputs, config["batch_size"])
            metrics = forecast_metrics(targets, predictions, np.asarray(config["quantiles"]), config["pinaw_normalizer_pu"])
            rows.append({
                "source_client_id": source, "target_client_id": target,
                "Pinball": metrics["Pinball"], "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"], "q50_pearson": metrics["q50_pearson"],
                "distribution_distance_composite": composite[source - 1, target - 1],
                "inference_time": inference_time,
            })
    frame = pd.DataFrame(rows)
    diagonal = frame.loc[frame["source_client_id"].eq(frame["target_client_id"])].set_index("target_client_id")["Pinball"]
    frame["target_local_diagonal_Pinball"] = frame["target_client_id"].map(diagonal)
    frame["Pinball_penalty_vs_target_local"] = frame["Pinball"] - frame["target_local_diagonal_Pinball"]
    frame.to_csv(output_dir / "table_cross_client_pilot_results.csv", index=False)
    off = frame.loc[frame["source_client_id"].ne(frame["target_client_id"])].copy()
    spearman = stats.spearmanr(off["distribution_distance_composite"], off["Pinball_penalty_vs_target_local"])
    asymmetries = []
    for i, source in enumerate(clients):
        for target in clients[i + 1:]:
            gij = float(frame.loc[frame["source_client_id"].eq(source) & frame["target_client_id"].eq(target), "Pinball"].iloc[0])
            gji = float(frame.loc[frame["source_client_id"].eq(target) & frame["target_client_id"].eq(source), "Pinball"].iloc[0])
            asymmetries.append(abs(gij - gji))
    diagnostics = {
        "clients": clients,
        "off_diagonal_pairs": len(off),
        "target_local_diagonal_is_best_count": int(sum(
            diagonal[target] <= frame.loc[frame["target_client_id"].eq(target), "Pinball"].min() + 1e-12 for target in clients
        )),
        "off_diagonal_beats_target_local_count": int((off["Pinball_penalty_vs_target_local"] < 0).sum()),
        "off_diagonal_penalty_median": float(off["Pinball_penalty_vs_target_local"].median()),
        "off_diagonal_penalty_p95": float(off["Pinball_penalty_vs_target_local"].quantile(0.95)),
        "median_absolute_directional_asymmetry_pinball": float(np.median(asymmetries)),
        "distance_penalty_spearman_rho": float(spearman.statistic),
        "distance_penalty_spearman_p": float(spearman.pvalue),
    }
    save_json(output_dir / "cross_client_diagnostics.json", diagnostics)
    return frame, diagnostics


def plot_heatmap(matrix: pd.DataFrame, path: Path, title: str, colorbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Target client")
    ax.set_ylabel("Source model client")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(local: pd.DataFrame, central: pd.DataFrame, cross: pd.DataFrame, config: dict, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    for column, filename, xlabel in (
        ("Pinball", "figure_local_pinball_distribution.png", "Mean Pinball Loss (p.u.)"),
        ("RMSE", "figure_local_rmse_distribution.png", "q0.5 RMSE (p.u.)"),
        ("PICP_80", "figure_local_picp80_distribution.png", "80% PICP"),
        ("quantile_crossing_rate", "figure_local_crossing_distribution.png", "Sample-level crossing rate"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(local[column], bins=20, color="#377eb8", edgecolor="white")
        ax.set_xlabel(xlabel); ax.set_ylabel("Clients"); ax.set_title(f"Local client distribution: {xlabel}")
        fig.tight_layout(); fig.savefig(figure_dir / filename, dpi=180); plt.close(fig)
    manifest = pd.read_csv(rel(config["stage2_results_dir"]) / "dataset_manifest.csv")
    quality = pd.read_csv(rel(config["eda_results_dir"]) / "table_client_quality.csv")
    merged = local.merge(manifest[["client_id", "train_sample_count", "target_unknown_rate"]], on="client_id").merge(quality[["client_id", "missing_rate"]], on="client_id")
    for x, filename, xlabel in (
        ("train_sample_count", "figure_performance_vs_sample_count.png", "Training samples"),
        ("target_unknown_rate", "figure_performance_vs_unknown_ratio.png", "Target unknown rate"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4.5)); ax.scatter(merged[x], merged["Pinball"], s=18, alpha=0.7)
        ax.set_xlabel(xlabel); ax.set_ylabel("Mean Pinball Loss (p.u.)"); fig.tight_layout(); fig.savefig(figure_dir / filename, dpi=180); plt.close(fig)
    representatives = config["pilot"]["client_ids"][:3]
    fig, axes = plt.subplots(len(representatives), 1, figsize=(10, 7), sharex=False)
    for ax, client_id in zip(axes, representatives):
        prediction = pd.read_parquet(output_dir / f"client_{client_id:03d}" / "test_predictions.parquet").head(288)
        x = np.arange(len(prediction)); ax.fill_between(x, prediction["q0.1"], prediction["q0.9"], alpha=0.25, label="q0.1–q0.9")
        ax.plot(x, prediction["q0.5"], linewidth=1, label="q0.5"); ax.plot(x, prediction["target_patv_pu"], linewidth=1, label="Observed")
        ax.set_title(f"Client {client_id}"); ax.set_ylabel("Power (p.u.)")
    axes[0].legend(ncol=3); axes[-1].set_xlabel("Consecutive test samples (first 288 valid windows)")
    fig.tight_layout(); fig.savefig(figure_dir / "figure_representative_probabilistic_forecasts.png", dpi=180); plt.close(fig)
    pinball_matrix = cross.pivot(index="source_client_id", columns="target_client_id", values="Pinball")
    rmse_matrix = cross.pivot(index="source_client_id", columns="target_client_id", values="RMSE")
    plot_heatmap(pinball_matrix, figure_dir / "figure_cross_client_pinball_heatmap.png", "Cross-client mean Pinball Loss", "Pinball (p.u.)")
    plot_heatmap(rmse_matrix, figure_dir / "figure_cross_client_rmse_heatmap.png", "Cross-client q0.5 RMSE", "RMSE (p.u.)")
    comparison = central[["client_id", "Pinball"]].merge(local[["client_id", "Pinball"]], on="client_id", suffixes=("_central", "_local"))
    fig, ax = plt.subplots(figsize=(5.5, 5)); ax.scatter(comparison["Pinball_local"], comparison["Pinball_central"], s=18, alpha=0.7)
    limits = [min(comparison.filter(like="Pinball").min()), max(comparison.filter(like="Pinball").max())]; ax.plot(limits, limits, "k--", linewidth=1)
    ax.set_xlabel("Local Pinball"); ax.set_ylabel("Centralized Reference Pinball"); fig.tight_layout(); fig.savefig(figure_dir / "figure_centralized_vs_local.png", dpi=180); plt.close(fig)


def summarize_results(local: pd.DataFrame, central: pd.DataFrame, cross_diag: dict, config: dict, output_dir: Path) -> dict:
    metrics = ["MAE", "RMSE", "Pinball", "CRPS_approx", "PICP_80", "PINAW_80", "quantile_crossing_rate"]
    rows = []
    for metric in metrics:
        series = local[metric]
        rows.append({
            "metric": metric, "mean": series.mean(), "median": series.median(), "std": series.std(ddof=1),
            "min": series.min(), "max": series.max(), "p5": series.quantile(0.05), "p95": series.quantile(0.95),
        })
    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(output_dir / "table_local_metric_distribution_summary.csv", index=False)
    comparison = local[["client_id", "Pinball"]].merge(central[["client_id", "Pinball"]], on="client_id", suffixes=("_local", "_central"))
    comparison["central_minus_local"] = comparison["Pinball_central"] - comparison["Pinball_local"]
    comparison.to_csv(output_dir / "table_centralized_vs_local.csv", index=False)
    summary = {
        "local_clients": len(local),
        "best_local_client_by_pinball": int(local.loc[local["Pinball"].idxmin(), "client_id"]),
        "worst_local_client_by_pinball": int(local.loc[local["Pinball"].idxmax(), "client_id"]),
        "centralized_lower_pinball_client_count": int((comparison["central_minus_local"] < 0).sum()),
        "centralized_higher_pinball_client_count": int((comparison["central_minus_local"] > 0).sum()),
        "centralized_mean_pinball": float(central["Pinball"].mean()),
        "local_mean_pinball": float(local["Pinball"].mean()),
        "local_median_pinball": float(local["Pinball"].median()),
        **cross_diag,
    }
    save_json(output_dir / "stage3_summary.json", summary)
    client83 = local.loc[local["client_id"].eq(83)].iloc[0]
    report = (
        "# Local Model Summary\n\n"
        "All results use the frozen Stage-2 Main-134/v1 windows, the common training-only scaler, seed 2026, and the 1-hour task. Metrics are in per-unit power unless stated otherwise.\n\n"
        "## Local distribution\n\n" + write_markdown_table(summary_table.round(6)) + "\n\n"
        f"Best/worst clients by Pinball: {summary['best_local_client_by_pinball']} / {summary['worst_local_client_by_pinball']}. "
        f"Client 83: Pinball={client83['Pinball']:.6f}, RMSE={client83['RMSE']:.6f}, PICP80={client83['PICP_80']:.6f}.\n\n"
        "## Centralized Oracle / Reference\n\n"
        f"Mean Pinball: centralized={central['Pinball'].mean():.6f}, local={local['Pinball'].mean():.6f}. "
        f"Centralized is lower for {summary['centralized_lower_pinball_client_count']}/134 clients and higher for {summary['centralized_higher_pinball_client_count']}/134.\n\n"
        "## Cross-client diagnostic\n\n"
        f"Composite Stage-2 distance versus cross-client Pinball penalty: Spearman rho={cross_diag['distance_penalty_spearman_rho']:.4f}, p={cross_diag['distance_penalty_spearman_p']:.4g}. "
        f"Off-diagonal models beat the target-local diagonal in {cross_diag['off_diagonal_beats_target_local_count']}/{cross_diag['off_diagonal_pairs']} directed cases. "
        "This is observational evidence only and does not define a relation model.\n"
    )
    (output_dir / "local_model_summary.md").write_text(report, encoding="utf-8")
    return summary


def write_decision_log(config: dict, output_dir: Path, cross_clients: list[int] | None = None) -> None:
    rows = [
        ("D-S3-01", "Plain 76-64-64-64-9 ReLU MLP", "Isolate forecasting feasibility from later federation; 13,833 parameters"),
        ("D-S3-02", "Adam, lr=0.001", "Single frozen optimizer setting for every client and centralized reference"),
        ("D-S3-03", "Batch size 2048", "Same fixed logic across clients and centralized reference"),
        ("D-S3-04", "Max 50 epochs; patience 6; min_delta 1e-5; restore best validation", "Validation-only early stopping; test never controls training"),
        ("D-S3-05", "Seed 2026; identical saved initialization", "Comparable Local parameters; seeds 2027/2028 deferred"),
        ("D-S3-06", "Mean Pinball over q=0.1,...,0.9", "Frozen probabilistic objective"),
        ("D-S3-07", "CRPS approximation = 2×mean discrete quantile score", "Explicitly not exact continuous CRPS"),
        ("D-S3-08", "Raw outputs; signed PINAW; sample-level any-adjacent crossing", "Do not conceal crossing via sorting or monotonic layers"),
        ("D-S3-09", "Centralized Oracle / Reference uses identical MLP", "Diagnostic pooled-data upper/reference condition, not deployable FL"),
        ("D-S3-10", f"Pilot clients {config['pilot']['client_ids']}", "Selected only from Stage-1/2 statistics before outcomes"),
        ("D-S3-11", f"Cross-client clients {cross_clients or 'pending'}", "Pilot anchors plus deterministic maximin coverage using frozen Stage-2 distances"),
    ]
    frame = pd.DataFrame(rows, columns=["decision_id", "decision", "rationale"])
    (output_dir / "Decision_Log.md").write_text("# Stage 3 Decision Log\n\n" + write_markdown_table(frame) + "\n", encoding="utf-8")


def write_run_manifest(config_path: Path, output_dir: Path, status: str) -> None:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "table_artifact_provenance.csv":
            artifacts.append({"artifact": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})
    pd.DataFrame(artifacts).to_csv(output_dir / "table_artifact_provenance.csv", index=False)
    save_json(output_dir / "run_manifest.json", {
        "stage": 3, "status": status, "config": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
        "runner": str(Path(__file__).resolve()), "runner_sha256": sha256_file(Path(__file__)),
        "component_sha256": sha256_file(ROOT / "scripts/local_baseline.py"),
        "stage2_manifest_sha256": sha256_file(rel("results/stage2/dataset_manifest.csv")),
        "stage2_scaler_sha256": sha256_file(rel("data/processed/v1/scaler.json")),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available(),
        "training_performed": status not in ("AUDIT_ONLY", "AUDIT_FAILED"),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_baseline.json")
    parser.add_argument("--mode", choices=("audit", "pilot", "full"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = rel(args.config)
    config = load_json(config_path)
    output_dir = rel(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(config["seed"], config["torch_num_threads"])
    audit = audit_dataset(config, output_dir)
    write_decision_log(config, output_dir)
    if audit["status"] != "PASS":
        write_run_manifest(config_path, output_dir, "AUDIT_FAILED")
        raise SystemExit("Stage-3 dataset audit failed; training blocked")
    if args.mode == "audit":
        write_run_manifest(config_path, output_dir, "AUDIT_ONLY")
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    initial_path = initialize_checkpoint(config, output_dir)
    pilot_passed, pilot = run_pilot(config, output_dir, initial_path, audit)
    if not pilot_passed:
        write_run_manifest(config_path, output_dir, "PILOT_FAILED")
        print("Pilot failed; Local-134 and centralized training blocked", flush=True)
        return
    if args.mode == "pilot":
        write_run_manifest(config_path, output_dir, "PILOT_PASSED")
        return
    local = train_all_local(config, output_dir, initial_path)
    central = train_centralized(config, output_dir, initial_path)
    cross_clients, composite = select_cross_clients(config, output_dir)
    cross, diagnostics = cross_client_evaluation(config, output_dir, cross_clients, composite)
    make_figures(local, central, cross, config, output_dir)
    summarize_results(local, central, diagnostics, config, output_dir)
    write_decision_log(config, output_dir, cross_clients)
    write_run_manifest(config_path, output_dir, "COMPLETE")
    print("Stage 3 complete", flush=True)


if __name__ == "__main__":
    main()
