"""Execute the pre-registered shared-anchor DRP-FL V1 mechanism test."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
sys.path.insert(0, str(ROOT / "scripts"))

from drp_v1 import (  # noqa: E402
    ema_relation, file_sha256, personalized_vectors, relation_statistics,
    relation_weights, select_validation_candidate, vector_to_state,
)
from federated_baselines import (  # noqa: E402
    cosine_matrix, load_client_cache, state_vector, train_one_client,
    weighted_average_states,
)
from local_baseline import (  # noqa: E402
    forecast_metrics, initial_model, mean_loss, predict, set_reproducible_seed,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda item: item.item() if isinstance(item, np.generic) else str(item)), encoding="utf-8")


def resolve(value: str) -> Path:
    return ROOT / value


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def setup(config_path: Path) -> tuple[dict, dict, Path]:
    config = load_json(config_path)
    stage4 = load_json(resolve(config["stage4_config"]))
    out = resolve(config["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    return config, stage4, out


def stage4_paths(config: dict, seed: int) -> tuple[Path, Path]:
    seed_dir = resolve(config["stage4_root"]) / f"seed_{seed}"
    return seed_dir, seed_dir / "fedavg"


def load_sample_weights(fedavg_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(fedavg_dir / "table_client_train_by_round.csv")
    first = frame.loc[frame["round"] == 1].sort_values("client_id")
    ids = first["client_id"].to_numpy(dtype=int)
    counts = first["train_samples"].to_numpy(dtype=np.float64)
    return ids, counts / counts.sum()


def anchor_and_locals(config: dict, seed: int, round_number: int) -> tuple[dict, np.ndarray, np.ndarray]:
    seed_dir, fedavg_dir = stage4_paths(config, seed)
    if round_number == 1:
        anchor_state = load_state(seed_dir / f"initial_model_seed_{seed}.pt")
    else:
        anchor_state = load_state(fedavg_dir / "rounds" / f"round_{round_number - 1:03d}.pt")
    anchor = state_vector(anchor_state).numpy().astype(np.float64)
    updates = np.load(fedavg_dir / "updates" / f"updates_round_{round_number:03d}.npy").astype(np.float64)
    return anchor_state, anchor, anchor[None, :] + updates


def audit_seed_artifacts(config: dict, stage4: dict, seed: int, out: Path) -> dict:
    seed_dir, fedavg_dir = stage4_paths(config, seed)
    ids, p = load_sample_weights(fedavg_dir)
    rows = []
    for round_number in range(1, stage4["max_communication_rounds"] + 1):
        update_path = fedavg_dir / "updates" / f"updates_round_{round_number:03d}.npy"
        cosine_path = fedavg_dir / "cosine_matrices" / f"cosine_round_{round_number:03d}.npy"
        checkpoint = fedavg_dir / "rounds" / f"round_{round_number:03d}.pt"
        if not all(path.exists() for path in (update_path, cosine_path, checkpoint)):
            raise FileNotFoundError(f"Incomplete frozen artifacts at seed {seed}, round {round_number}")
        _, _, locals_ = anchor_and_locals(config, seed, round_number)
        manual = p @ locals_
        frozen = state_vector(load_state(checkpoint)).numpy().astype(np.float64)
        rows.append({
            "seed": seed, "round": round_number,
            "clients": len(ids), "parameters": locals_.shape[1],
            "max_abs_anchor_reconstruction_error": float(np.max(np.abs(manual - frozen))),
            "updates_sha256": file_sha256(update_path),
            "cosine_sha256": file_sha256(cosine_path),
            "checkpoint_sha256": file_sha256(checkpoint),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(out / f"table_seed_{seed}_shared_anchor_audit.csv", index=False)
    max_error = float(audit["max_abs_anchor_reconstruction_error"].max())
    result = {
        "seed": seed, "rounds": len(rows), "clients": int(len(ids)),
        "max_abs_anchor_reconstruction_error": max_error,
        "tolerance": config["artifact_tolerance"],
        "passed": max_error <= config["artifact_tolerance"],
        "interpretation": "The non-zero error is the frozen Stage-4 float32-update/manual-aggregation roundoff already recorded in Stage 4.",
    }
    save_json(out / f"seed_{seed}_shared_anchor_audit.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Shared-anchor audit failed: {max_error:.3g}")
    return result


def load_cache(stage4: dict, client_ids: list[int]):
    scaler = load_json(resolve(stage4["processed_dir"]) / "scaler.json")
    return load_client_cache(resolve(stage4["processed_dir"]), client_ids, scaler, stage4["lookback_steps"])


def local_validation_losses(config: dict, stage4: dict, seed: int, cache: dict, out: Path) -> pd.DataFrame:
    path = out / f"table_seed_{seed}_local_validation.csv"
    if path.exists():
        return pd.read_csv(path)
    q = torch.tensor(stage4["quantiles"], dtype=torch.float32)
    rows = []
    for client_id, data in cache.items():
        model = initial_model(stage4)
        checkpoint = resolve(config["local_root"]) / f"seed_{seed}" / f"client_{client_id:03d}" / "checkpoint.pt"
        model.load_state_dict(load_state(checkpoint))
        loss = mean_loss(model, data.validation_x, data.validation_y, q, stage4["batch_size"])
        rows.append({"seed": seed, "client_id": client_id, "local_validation_pinball": loss})
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def evaluate_vectors(vectors: np.ndarray, template: dict, cache: dict, stage4: dict, split: str, prediction_dir: Path | None = None) -> pd.DataFrame:
    if split == "validation":
        return evaluate_validation_vectors_batched(vectors, template, cache, stage4)
    q = torch.tensor(stage4["quantiles"], dtype=torch.float32)
    model = initial_model(stage4)
    rows = []
    for row_index, (client_id, data) in enumerate(cache.items()):
        model.load_state_dict(vector_to_state(vectors[row_index], template))
        predictions, inference_time = predict(model, data.test_x, stage4["batch_size"])
        metrics = forecast_metrics(data.test_y, predictions, np.asarray(stage4["quantiles"]), stage4["pinaw_normalizer_pu"])
        rows.append({"client_id": client_id, "samples": len(data.test_y), **metrics, "CRPS": metrics["CRPS_approx"], "inference_time": inference_time})
        if prediction_dir is not None:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            payload = {"y_true": data.test_y}
            for index, quantile in enumerate(stage4["quantiles"]):
                payload[f"q{quantile:.1f}"] = predictions[:, index]
            pd.DataFrame(payload).to_parquet(prediction_dir / f"client_{client_id:03d}.parquet", index=False)
    return pd.DataFrame(rows)


@torch.inference_mode()
def evaluate_validation_vectors_batched(vectors: np.ndarray, template: dict, cache: dict, stage4: dict) -> pd.DataFrame:
    """Evaluate one personalized model per client using a batched functional call.

    Padding is masked before aggregation; this is exactly the same per-sample
    Pinball objective as sequential evaluation, only the execution layout differs.
    """
    from torch.func import functional_call, vmap

    model = initial_model(stage4)
    flat = torch.from_numpy(np.asarray(vectors, dtype=np.float32))
    batched_parameters = {}
    offset = 0
    for name, tensor in template.items():
        size = tensor.numel()
        batched_parameters[name] = flat[:, offset:offset + size].reshape(len(flat), *tensor.shape)
        offset += size
    if offset != flat.shape[1]:
        raise ValueError("Personalized vectors do not match model parameters")

    client_items = list(cache.items())
    lengths = torch.tensor([len(data.validation_y) for _, data in client_items], dtype=torch.long)
    total_loss = torch.zeros(len(client_items), dtype=torch.float64)
    quantiles = torch.tensor(stage4["quantiles"], dtype=torch.float32)
    batch_size = int(stage4["batch_size"])
    max_length = int(lengths.max())

    def one_model(parameters, inputs):
        return functional_call(model, parameters, (inputs,))

    vmapped = vmap(one_model, in_dims=(0, 0))
    for start in range(0, max_length, batch_size):
        width = min(batch_size, max_length - start)
        inputs = torch.zeros((len(client_items), width, stage4["input_dimension"]), dtype=torch.float32)
        targets = torch.zeros((len(client_items), width), dtype=torch.float32)
        mask = torch.zeros((len(client_items), width), dtype=torch.bool)
        for client_index, (_, data) in enumerate(client_items):
            stop = min(start + width, len(data.validation_y))
            count = max(0, stop - start)
            if count:
                inputs[client_index, :count] = torch.from_numpy(data.validation_x[start:stop])
                targets[client_index, :count] = torch.from_numpy(data.validation_y[start:stop])
                mask[client_index, :count] = True
        predictions = vmapped(batched_parameters, inputs)
        errors = targets[:, :, None] - predictions
        losses = torch.maximum(quantiles[None, None, :] * errors, (quantiles[None, None, :] - 1.0) * errors).mean(dim=2)
        total_loss += (losses * mask).sum(dim=1).to(torch.float64)
    means = total_loss / lengths.to(torch.float64)
    return pd.DataFrame({
        "client_id": [client_id for client_id, _ in client_items],
        "Pinball": means.numpy(), "samples": lengths.numpy(),
    })


def candidate_summary(client_metrics: pd.DataFrame, local_validation: pd.DataFrame) -> dict:
    merged = client_metrics.merge(local_validation, on="client_id", validate="one_to_one")
    degradation = merged["Pinball"] - merged["local_validation_pinball"]
    return {
        "macro_validation_pinball": float(merged["Pinball"].mean()),
        "median_validation_pinball": float(merged["Pinball"].median()),
        "validation_degraded_clients": int((degradation > 0).sum()),
        "validation_improved_clients": int((degradation < 0).sum()),
        "validation_nt_rate": float((degradation > 0).mean()),
        "worst_validation_degradation": float(degradation.max()),
    }


def run_grid(config: dict, stage4: dict, out: Path) -> dict:
    seed = config["primary_seed"]
    audit_seed_artifacts(config, stage4, seed, out)
    _, fedavg_dir = stage4_paths(config, seed)
    client_ids, p = load_sample_weights(fedavg_dir)
    cache = load_cache(stage4, client_ids.tolist())
    local_val = local_validation_losses(config, stage4, seed, cache, out)
    result_path = out / "table_drp_v1_hyperparameter_validation.csv"
    completed = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    done = set(zip(completed.get("beta", []), completed.get("tau", []), completed.get("round", [])))
    rows = completed.to_dict("records")
    ema = {float(beta): None for beta in config["betas"]}
    for round_number in range(1, stage4["max_communication_rounds"] + 1):
        template, _, locals_ = anchor_and_locals(config, seed, round_number)
        raw = np.load(fedavg_dir / "cosine_matrices" / f"cosine_round_{round_number:03d}.npy")
        for beta in config["betas"]:
            beta = float(beta)
            ema[beta] = ema_relation(ema[beta], raw, beta)
            for tau in config["temperatures"]:
                tau = float(tau)
                if (beta, tau, round_number) in done:
                    continue
                weights = relation_weights(ema[beta], p, tau)
                vectors = personalized_vectors(locals_, weights)
                metrics = evaluate_vectors(vectors, template, cache, stage4, "validation")
                row = {"seed": seed, "variant": "dynamic", "beta": beta, "tau": tau, "round": round_number,
                       **candidate_summary(metrics, local_val), **relation_statistics(weights)}
                rows.append(row)
                pd.DataFrame(rows).sort_values(["round", "beta", "tau"]).to_csv(result_path, index=False)
                print(f"GRID round={round_number:02d} beta={beta:g} tau={tau:g} val={row['macro_validation_pinball']:.6f} NT={row['validation_nt_rate']:.3f}", flush=True)
    dynamic = [row for row in rows if row.get("variant") == "dynamic"]
    selected = select_validation_candidate(dynamic, tolerance=1e-12)
    save_json(out / "selected_hyperparameters_seed_2026.json", {
        "selection_data": "seed 2026 validation only", "selection_order": ["macro_validation_pinball", "validation_nt_rate", "worst_validation_degradation"],
        "beta": selected["beta"], "tau": selected["tau"], "round": selected["round"], "selected_row": selected,
        "test_metrics_consulted": False,
    })
    return selected


def smoothed_relation_at(config: dict, seed: int, beta: float, round_number: int) -> tuple[np.ndarray, np.ndarray]:
    _, fedavg_dir = stage4_paths(config, seed)
    smoothed = None
    raw = None
    for current_round in range(1, round_number + 1):
        raw = np.load(fedavg_dir / "cosine_matrices" / f"cosine_round_{current_round:03d}.npy")
        smoothed = ema_relation(smoothed, raw, beta)
    return np.asarray(raw), np.asarray(smoothed)


def add_nt_columns(frame: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(local[["client_id", "Pinball"]].rename(columns={"Pinball": "Local_Pinball"}), on="client_id", how="left", validate="many_to_one")
    merged["absolute_gain_vs_local"] = merged["Local_Pinball"] - merged["Pinball"]
    merged["relative_degradation_vs_local"] = (merged["Pinball"] - merged["Local_Pinball"]) / merged["Local_Pinball"]
    merged["negative_transfer"] = merged["Pinball"] > merged["Local_Pinball"]
    return merged


def summarize_test(frame: pd.DataFrame, local: pd.DataFrame) -> dict:
    merged = add_nt_columns(frame, local)
    rel = merged["relative_degradation_vs_local"]
    gain = merged["absolute_gain_vs_local"]
    return {
        "mean_Pinball": float(merged["Pinball"].mean()), "median_Pinball": float(merged["Pinball"].median()),
        "mean_MAE": float(merged["MAE"].mean()), "mean_RMSE": float(merged["RMSE"].mean()), "mean_CRPS": float(merged["CRPS"].mean()),
        "mean_PICP80": float(merged["PICP_80"].mean()), "mean_PINAW80": float(merged["PINAW_80"].mean()),
        "mean_crossing_rate": float(merged["quantile_crossing_rate"].mean()),
        "improved_clients": int((gain > 0).sum()), "degraded_clients": int((gain < 0).sum()),
        "negative_transfer_rate": float((gain < 0).mean()), "relative_degradation_gt_1pct": int((rel > 0.01).sum()),
        "relative_degradation_gt_5pct": int((rel > 0.05).sum()), "relative_degradation_gt_10pct": int((rel > 0.10).sum()),
        "worst_degradation": float((-gain).max()), "best_gain": float(gain.max()),
    }


def existing_test_results(config: dict, seed: int, method: str) -> pd.DataFrame:
    if method == "Local":
        path = resolve(config["local_root"]) / f"seed_{seed}" / "table_local_client_results.csv"
    else:
        path = resolve(config["stage4_root"]) / f"seed_{seed}" / method.lower() / "table_client_results.csv"
    frame = pd.read_csv(path)
    if "CRPS" not in frame:
        frame["CRPS"] = frame["CRPS_approx"]
    return frame


def plot_primary_outputs(out: Path, client_ids: np.ndarray, raw: np.ndarray, smoothed: np.ndarray, weights: np.ndarray, controls: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    for name, matrix, lower, upper in (
        ("raw_cosine", raw, -1, 1), ("smoothed_cosine", smoothed, -1, 1), ("personalized_weights", weights, 0, float(np.quantile(weights, 0.995))),
    ):
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm" if "cosine" in name else "viridis", vmin=lower, vmax=upper)
        ax.set_title(f"{name.replace('_', ' ').title()} (ascending client ID)")
        ax.set_xlabel("Source client index"); ax.set_ylabel("Target client index")
        fig.colorbar(image, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(figures / f"figure_{name}_heatmap.png", dpi=180); plt.close(fig)
    self_weight = np.diag(weights)
    entropy = -np.sum(np.where(weights > 0, weights * np.log(weights), 0.0), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(self_weight, bins=20); axes[0].set_title("Selected self-weight")
    axes[1].hist(np.exp(entropy), bins=20); axes[1].set_title("Effective collaborators")
    fig.tight_layout(); fig.savefig(figures / "figure_collaboration_distributions.png", dpi=180); plt.close(fig)
    if not controls.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        grouped = controls.groupby("variant")["mean_Pinball"].agg(["mean", "std"]).reset_index()
        ax.bar(grouped["variant"], grouped["mean"], yerr=grouped["std"].fillna(0), capsize=3)
        ax.tick_params(axis="x", rotation=35); ax.set_ylabel("Mean client Pinball")
        fig.tight_layout(); fig.savefig(figures / "figure_control_comparison.png", dpi=180); plt.close(fig)


def run_primary_test(config: dict, stage4: dict, out: Path) -> dict:
    seed = config["primary_seed"]
    selected = load_json(out / "selected_hyperparameters_seed_2026.json")
    beta, tau, selected_round = float(selected["beta"]), float(selected["tau"]), int(selected["round"])
    _, fedavg_dir = stage4_paths(config, seed)
    client_ids, p = load_sample_weights(fedavg_dir)
    cache = load_cache(stage4, client_ids.tolist())
    local = existing_test_results(config, seed, "Local")
    fedavg = existing_test_results(config, seed, "FedAvg")
    fedprox = existing_test_results(config, seed, "FedProx")
    template, _, locals_ = anchor_and_locals(config, seed, selected_round)
    raw, smoothed = smoothed_relation_at(config, seed, beta, selected_round)
    weights = relation_weights(smoothed, p, tau)
    vectors = personalized_vectors(locals_, weights)
    drp = evaluate_vectors(vectors, template, cache, stage4, "test", out / "seed_2026" / "test_predictions_by_client")
    drp.insert(1, "method", "DRP-dynamic-EMA")
    drp.insert(2, "seed", seed); drp.insert(3, "beta", beta); drp.insert(4, "tau", tau); drp.insert(5, "selected_round", selected_round)

    # Static relation: first five rounds only; tau remains frozen. Its checkpoint is selected on validation.
    static_matrix = np.mean([np.load(fedavg_dir / "cosine_matrices" / f"cosine_round_{r:03d}.npy").astype(np.float64) for r in range(1, config["static_relation_rounds"] + 1)], axis=0)
    np.fill_diagonal(static_matrix, 1.0)
    static_weights = relation_weights(static_matrix, p, tau)
    local_val = pd.read_csv(out / f"table_seed_{seed}_local_validation.csv")
    static_validation_rows = []
    for round_number in range(config["static_relation_rounds"], stage4["max_communication_rounds"] + 1):
        round_template, _, round_locals = anchor_and_locals(config, seed, round_number)
        metrics = evaluate_vectors(personalized_vectors(round_locals, static_weights), round_template, cache, stage4, "validation")
        static_validation_rows.append({"variant": "static", "beta": np.nan, "tau": tau, "round": round_number, **candidate_summary(metrics, local_val), **relation_statistics(static_weights)})
    static_selected = select_validation_candidate(static_validation_rows)
    pd.DataFrame(static_validation_rows).to_csv(out / "table_drp_v1_static_validation.csv", index=False)
    static_template, _, static_locals = anchor_and_locals(config, seed, int(static_selected["round"]))
    static_test = evaluate_vectors(personalized_vectors(static_locals, static_weights), static_template, cache, stage4, "test")
    static_test.insert(1, "method", "DRP-static")

    # No-EMA uses beta=0, frozen tau, and validation-selected round from the completed grid.
    grid = pd.read_csv(out / "table_drp_v1_hyperparameter_validation.csv")
    noema_selected = select_validation_candidate(grid[(grid["beta"] == 0.0) & (grid["tau"] == tau)].to_dict("records"))
    noema_round = int(noema_selected["round"])
    noema_template, _, noema_locals = anchor_and_locals(config, seed, noema_round)
    _, noema_similarity = smoothed_relation_at(config, seed, 0.0, noema_round)
    noema_weights = relation_weights(noema_similarity, p, tau)
    noema_test = evaluate_vectors(personalized_vectors(noema_locals, noema_weights), noema_template, cache, stage4, "test")
    noema_test.insert(1, "method", "DRP-dynamic-noEMA")

    control_rows = []
    main_summary = summarize_test(drp, local)
    control_rows.append({"variant": "DRP-dynamic-EMA", "repeat": 0, "selected_round": selected_round, **main_summary, **relation_statistics(weights)})
    control_rows.append({"variant": "DRP-dynamic-noEMA", "repeat": 0, "selected_round": noema_round, **summarize_test(noema_test, local), **relation_statistics(noema_weights)})
    control_rows.append({"variant": "DRP-static", "repeat": 0, "selected_round": int(static_selected["round"]), **summarize_test(static_test, local), **relation_statistics(static_weights)})
    control_rows.append({"variant": "FedAvg", "repeat": 0, "selected_round": int(fedavg["best_round"].iloc[0]), **summarize_test(fedavg, local), **relation_statistics(np.tile(p, (len(p), 1)))})
    rng = np.random.default_rng(config["random_relation_seed"])
    for repeat in range(1, config["random_relation_repeats"] + 1):
        permutation = rng.permutation(len(client_ids))
        randomized = smoothed[np.ix_(permutation, permutation)]
        random_weights = relation_weights(randomized, p, tau)
        random_test = evaluate_vectors(personalized_vectors(locals_, random_weights), template, cache, stage4, "test")
        control_rows.append({"variant": "Random-relation", "repeat": repeat, "selected_round": selected_round, **summarize_test(random_test, local), **relation_statistics(random_weights)})
        target_permutation = np.roll(np.arange(len(client_ids)), repeat)
        shuffled = smoothed[target_permutation, :]
        shuffled_weights = relation_weights(shuffled, p, tau)
        shuffled_test = evaluate_vectors(personalized_vectors(locals_, shuffled_weights), template, cache, stage4, "test")
        control_rows.append({"variant": "Shuffled-target", "repeat": repeat, "selected_round": selected_round, **summarize_test(shuffled_test, local), **relation_statistics(shuffled_weights)})
        print(f"CONTROL repeat={repeat:02d} random={control_rows[-2]['mean_Pinball']:.6f} shuffled={control_rows[-1]['mean_Pinball']:.6f}", flush=True)
    controls = pd.DataFrame(control_rows)
    controls.to_csv(out / "table_drp_v1_control_experiments.csv", index=False)

    all_clients = []
    for method, frame in (("Local", local), ("FedAvg", fedavg), ("FedProx", fedprox), ("DRP-dynamic-EMA", drp), ("DRP-dynamic-noEMA", noema_test), ("DRP-static", static_test)):
        current = frame.copy(); current["method"] = method; current["seed"] = seed
        all_clients.append(add_nt_columns(current, local))
    client_results = pd.concat(all_clients, ignore_index=True)
    client_results.to_csv(out / "table_drp_v1_client_results.csv", index=False)
    nt_rows = [{"seed": seed, "method": method, **summarize_test(frame, local)} for method, frame in (("Local", local), ("FedAvg", fedavg), ("FedProx", fedprox), ("DRP-dynamic-EMA", drp), ("DRP-dynamic-noEMA", noema_test), ("DRP-static", static_test))]
    pd.DataFrame(nt_rows).to_csv(out / "table_drp_v1_negative_transfer.csv", index=False)
    grid.to_csv(out / "table_drp_v1_relation_statistics.csv", index=False)
    np.save(out / "selected_raw_cosine.npy", raw.astype(np.float32)); np.save(out / "selected_smoothed_cosine.npy", smoothed.astype(np.float32)); np.save(out / "selected_aggregation_weights.npy", weights.astype(np.float32))
    pd.DataFrame(weights, index=client_ids, columns=client_ids).to_csv(out / "matrix_selected_aggregation_weights.csv")

    random_rows = controls[controls["variant"] == "Random-relation"]
    shuffled_rows = controls[controls["variant"] == "Shuffled-target"]
    fedavg_summary = summarize_test(fedavg, local)
    gate_def = config["pre_registered_seed_2026_continuation_gate"]
    checks = {
        "mean_not_worse_than_fedavg": main_summary["mean_Pinball"] <= fedavg_summary["mean_Pinball"] + gate_def["drp_mean_pinball_not_worse_than_fedavg_tolerance"],
        "fewer_degraded_than_fedavg": main_summary["degraded_clients"] < fedavg_summary["degraded_clients"],
        "worst_not_worse_than_fedavg": main_summary["worst_degradation"] <= fedavg_summary["worst_degradation"] + gate_def["worst_degradation_not_worse_than_fedavg_tolerance"],
        "beats_random_mean": main_summary["mean_Pinball"] <= random_rows["mean_Pinball"].mean() - gate_def["drp_must_beat_random_mean_by"],
        "beats_random_repeats": int((main_summary["mean_Pinball"] < random_rows["mean_Pinball"]).sum()) >= gate_def["drp_must_beat_at_least_random_repeats"],
        "beats_shuffled_mean": main_summary["mean_Pinball"] <= shuffled_rows["mean_Pinball"].mean() - gate_def["drp_must_beat_shuffled_mean_by"],
        "nonlocal_median_self_weight": relation_statistics(weights)["median_self_weight"] <= gate_def["median_self_weight_max"],
        "external_mass": relation_statistics(weights)["mean_external_mass"] >= gate_def["mean_external_mass_min"],
    }
    gate = {"pre_registered_checks": checks, "all_passed": bool(all(checks.values())), "main": main_summary, "fedavg": fedavg_summary,
            "random_mean": float(random_rows["mean_Pinball"].mean()), "random_std": float(random_rows["mean_Pinball"].std(ddof=1)),
            "shuffled_mean": float(shuffled_rows["mean_Pinball"].mean()), "shuffled_std": float(shuffled_rows["mean_Pinball"].std(ddof=1)),
            "static": summarize_test(static_test, local), "noema": summarize_test(noema_test, local),
            "static_selected_round": int(static_selected["round"]), "noema_selected_round": noema_round}
    save_json(out / "seed_2026_continuation_gate.json", gate)
    plot_primary_outputs(out, client_ids, raw, smoothed, weights, controls)
    return gate


def regenerate_updates(config: dict, stage4: dict, out: Path, seed: int) -> dict:
    """Recreate missing updates from each frozen FedAvg anchor and audit equality."""
    seed_dir, fedavg_dir = stage4_paths(config, seed)
    client_ids, _ = load_sample_weights(fedavg_dir)
    cache = load_cache(stage4, client_ids.tolist())
    regen = out / "regenerated_updates" / f"seed_{seed}"
    updates_dir = regen / "updates"; cosine_dir = regen / "cosine_matrices"
    updates_dir.mkdir(parents=True, exist_ok=True); cosine_dir.mkdir(parents=True, exist_ok=True)
    initialization = seed_dir / f"initial_model_seed_{seed}.pt"
    frozen_summary = load_json(fedavg_dir / "training_summary.json")
    if file_sha256(initialization) != frozen_summary["initial_checkpoint_sha256"]:
        raise RuntimeError(f"Seed {seed} initialization hash mismatch before regeneration")
    audit_rows = []
    for round_number in range(1, stage4["max_communication_rounds"] + 1):
        update_path = updates_dir / f"updates_round_{round_number:03d}.npy"
        cosine_path = cosine_dir / f"cosine_round_{round_number:03d}.npy"
        if update_path.exists() and cosine_path.exists():
            updates = np.load(update_path)
            anchor_state = load_state(initialization if round_number == 1 else fedavg_dir / "rounds" / f"round_{round_number - 1:03d}.pt")
            anchor_vector = state_vector(anchor_state).numpy()
            local_states = [vector_to_state(anchor_vector + updates[index], anchor_state) for index in range(len(client_ids))]
            counts = [len(cache[int(client_id)].train_y) for client_id in client_ids]
            reused = True
        else:
            anchor_state = load_state(initialization if round_number == 1 else fedavg_dir / "rounds" / f"round_{round_number - 1:03d}.pt")
            anchor_vector = state_vector(anchor_state).numpy()
            local_states, updates, counts = [], [], []
            for client_id in client_ids:
                state, diagnostics = train_one_client(anchor_state, cache[int(client_id)], stage4, round_number, seed, 0.0)
                if not diagnostics["finite"] or diagnostics["synchronization_max_abs_error"] != 0.0:
                    raise RuntimeError(f"Non-finite or unsynchronized client {client_id}, round {round_number}")
                local_states.append(state); counts.append(diagnostics["train_samples"])
                updates.append((state_vector(state).numpy() - anchor_vector).astype(np.float32))
            updates = np.stack(updates)
            np.save(update_path, updates)
            np.save(cosine_path, cosine_matrix(updates).astype(np.float32))
            reused = False
        aggregated = weighted_average_states(local_states, counts)
        frozen = load_state(fedavg_dir / "rounds" / f"round_{round_number:03d}.pt")
        error = float(np.max(np.abs(state_vector(aggregated).numpy().astype(np.float64) - state_vector(frozen).numpy().astype(np.float64))))
        audit_rows.append({"seed": seed, "round": round_number, "reused_regenerated_artifact": reused, "max_abs_checkpoint_error": error,
                           "updates_sha256": file_sha256(update_path), "cosine_sha256": file_sha256(cosine_path)})
        pd.DataFrame(audit_rows).to_csv(regen / "table_regeneration_audit.csv", index=False)
        print(f"REGEN seed={seed} round={round_number:02d} checkpoint_error={error:.3g} reused={reused}", flush=True)
        if error > config["artifact_tolerance"]:
            raise RuntimeError(f"Seed {seed} round {round_number} fails frozen checkpoint reproduction: {error}")
    result = {"seed": seed, "initialization_sha256": file_sha256(initialization), "rounds": len(audit_rows),
              "max_abs_checkpoint_error": float(max(row["max_abs_checkpoint_error"] for row in audit_rows)),
              "validation_trajectory_reproduced_by_checkpoint_identity": True,
              "selected_round": int(frozen_summary["best_round"]), "passed": True}
    save_json(regen / "regeneration_summary.json", result)
    return result


def relation_artifact_dir(config: dict, out: Path, seed: int) -> Path:
    _, fedavg = stage4_paths(config, seed)
    if (fedavg / "updates" / "updates_round_001.npy").exists():
        return fedavg
    return out / "regenerated_updates" / f"seed_{seed}"


def offline_locals_for_seed(config: dict, out: Path, seed: int, round_number: int) -> tuple[dict, np.ndarray]:
    seed_dir, fedavg = stage4_paths(config, seed)
    anchor_state = load_state(seed_dir / f"initial_model_seed_{seed}.pt" if round_number == 1 else fedavg / "rounds" / f"round_{round_number - 1:03d}.pt")
    anchor = state_vector(anchor_state).numpy().astype(np.float64)
    updates = np.load(relation_artifact_dir(config, out, seed) / "updates" / f"updates_round_{round_number:03d}.npy").astype(np.float64)
    return anchor_state, anchor[None, :] + updates


def offline_smoothed(config: dict, out: Path, seed: int, beta: float, round_number: int) -> np.ndarray:
    base = relation_artifact_dir(config, out, seed) / "cosine_matrices"
    smoothed = None
    for current in range(1, round_number + 1):
        smoothed = ema_relation(smoothed, np.load(base / f"cosine_round_{current:03d}.npy"), beta)
    return np.asarray(smoothed)


def evaluate_additional_seed(config: dict, stage4: dict, out: Path, seed: int) -> dict:
    frozen = load_json(out / "selected_hyperparameters_seed_2026.json")
    beta, tau = float(frozen["beta"]), float(frozen["tau"])
    _, fedavg_dir = stage4_paths(config, seed)
    client_ids, p = load_sample_weights(fedavg_dir)
    cache = load_cache(stage4, client_ids.tolist())
    local_val = local_validation_losses(config, stage4, seed, cache, out)
    rows = []
    smoothed = None
    relation_dir = relation_artifact_dir(config, out, seed) / "cosine_matrices"
    for round_number in range(1, stage4["max_communication_rounds"] + 1):
        template, locals_ = offline_locals_for_seed(config, out, seed, round_number)
        smoothed = ema_relation(smoothed, np.load(relation_dir / f"cosine_round_{round_number:03d}.npy"), beta)
        weights = relation_weights(smoothed, p, tau)
        metrics = evaluate_vectors(personalized_vectors(locals_, weights), template, cache, stage4, "validation")
        rows.append({"seed": seed, "variant": "dynamic_frozen_beta_tau", "beta": beta, "tau": tau, "round": round_number,
                     **candidate_summary(metrics, local_val), **relation_statistics(weights)})
        print(f"ADDITIONAL-VAL seed={seed} round={round_number:02d} val={rows[-1]['macro_validation_pinball']:.6f}", flush=True)
    validation = pd.DataFrame(rows)
    validation.to_csv(out / f"table_seed_{seed}_drp_validation.csv", index=False)
    selected = select_validation_candidate(rows)
    round_number = int(selected["round"])
    template, locals_ = offline_locals_for_seed(config, out, seed, round_number)
    similarity = offline_smoothed(config, out, seed, beta, round_number)
    weights = relation_weights(similarity, p, tau)
    drp = evaluate_vectors(personalized_vectors(locals_, weights), template, cache, stage4, "test", out / f"seed_{seed}" / "test_predictions_by_client")
    drp["method"] = "DRP-dynamic-EMA"; drp["seed"] = seed; drp["beta"] = beta; drp["tau"] = tau; drp["selected_round"] = round_number
    drp.to_csv(out / f"table_seed_{seed}_drp_client_results.csv", index=False)
    local = existing_test_results(config, seed, "Local")
    result = {"seed": seed, "beta": beta, "tau": tau, "selected_round": round_number, **summarize_test(drp, local), **relation_statistics(weights)}
    save_json(out / f"seed_{seed}_drp_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/drp_v1.json")
    parser.add_argument("--phase", choices=("audit", "grid", "primary-test", "regenerate", "additional-test"), default="grid")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config, stage4, out = setup(config_path)
    set_reproducible_seed(config["primary_seed"], stage4["torch_num_threads"])
    manifest = {
        "started_utc": pd.Timestamp.utcnow().isoformat(), "phase": args.phase,
        "config": str(config_path.resolve()), "config_sha256": file_sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": file_sha256(Path(__file__)),
        "stage4_config_sha256": file_sha256(resolve(config["stage4_config"])),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
        "training_performed": False, "test_used_for_hyperparameter_selection": False,
    }
    save_json(out / f"run_manifest_{args.phase}.json", manifest)
    if args.phase == "audit":
        result = audit_seed_artifacts(config, stage4, config["primary_seed"], out)
    elif args.phase == "grid":
        result = run_grid(config, stage4, out)
    elif args.phase == "primary-test":
        result = run_primary_test(config, stage4, out)
    elif args.phase == "regenerate":
        if args.seed not in config["additional_seeds"]:
            raise ValueError("--seed must be one of the pre-registered additional seeds")
        result = regenerate_updates(config, stage4, out, args.seed)
    else:
        if args.seed not in config["additional_seeds"]:
            raise ValueError("--seed must be one of the pre-registered additional seeds")
        result = evaluate_additional_seed(config, stage4, out, args.seed)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
