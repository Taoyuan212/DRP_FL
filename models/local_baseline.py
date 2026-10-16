"""Frozen Stage-3 local probabilistic forecasting components.

This module intentionally contains only a plain MLP, quantile loss, deterministic
data construction from Stage-2 windows, and diagnostic metrics.  It contains no
federated-learning or client-relation code.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SPLIT_NAMES = ("train", "validation", "test")
INPUT_SOURCE_COLUMNS = (
    "input_patv_clean_kw",
    "input_wspd_ms",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def set_reproducible_seed(seed: int, torch_num_threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(torch_num_threads)
    torch.use_deterministic_algorithms(True)


class QuantileMLP(nn.Module):
    def __init__(self, input_dim: int = 76, hidden_dims: Iterable[int] = (64, 64, 64), output_dim: int = 9):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def pinball_loss(predictions: torch.Tensor, targets: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        targets = targets[:, None]
    errors = targets - predictions
    return torch.maximum(quantiles * errors, (quantiles - 1.0) * errors).mean()


def pinball_by_sample(y_true: np.ndarray, predictions: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    errors = y_true[:, None] - predictions
    return np.maximum(quantiles[None, :] * errors, (quantiles[None, :] - 1.0) * errors).mean(axis=1)


def forecast_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    quantiles: np.ndarray,
    pinaw_normalizer: float = 1.0,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if predictions.shape != (len(y_true), len(quantiles)):
        raise ValueError("Prediction shape does not match targets and quantiles")
    q_index = {round(float(q), 6): index for index, q in enumerate(quantiles)}
    q50 = predictions[:, q_index[0.5]]
    errors = q50 - y_true
    sample_pinball = pinball_by_sample(y_true, predictions, quantiles)
    crossing = np.any(predictions[:, :-1] > predictions[:, 1:], axis=1)

    result = {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "Pinball": float(np.mean(sample_pinball)),
        # Discrete quantile-score approximation, not exact continuous-distribution CRPS.
        "CRPS_approx": float(2.0 * np.mean(sample_pinball)),
        "quantile_crossing_rate": float(np.mean(crossing)),
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "finite_predictions": bool(np.isfinite(predictions).all()),
    }
    for coverage, lower_q, upper_q in ((80, 0.1, 0.9), (60, 0.2, 0.8)):
        lower = predictions[:, q_index[lower_q]]
        upper = predictions[:, q_index[upper_q]]
        result[f"PICP_{coverage}"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        # Kept signed: negative widths expose, rather than conceal, interval crossing.
        result[f"PINAW_{coverage}"] = float(np.mean(upper - lower) / pinaw_normalizer)
    if len(y_true) > 1 and np.std(q50) > 0 and np.std(y_true) > 0:
        result["q50_pearson"] = float(np.corrcoef(q50, y_true)[0, 1])
    else:
        result["q50_pearson"] = float("nan")
    return result


def _stack_list_column(series: pd.Series, expected_steps: int) -> np.ndarray:
    array = np.stack(series.to_numpy()).astype(np.float32, copy=False)
    if array.shape[1:] != (expected_steps,):
        raise ValueError(f"Expected fixed history length {expected_steps}; got {array.shape}")
    return array


@dataclass
class ClientArrays:
    frame: pd.DataFrame
    inputs: np.ndarray
    targets: np.ndarray
    splits: np.ndarray

    def select(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = self.splits == split
        return self.inputs[mask], self.targets[mask], np.flatnonzero(mask)


def load_client_arrays(path: Path, scaler: dict, expected_steps: int = 36) -> ClientArrays:
    columns = [
        "client_id", "issue_time", "history_start", "target_time", "horizon_minutes", "split",
        "is_valid", "target_patv_pu", *INPUT_SOURCE_COLUMNS,
    ]
    frame = pd.read_parquet(path, columns=columns)
    patv = _stack_list_column(frame["input_patv_clean_kw"], expected_steps)
    wspd = _stack_list_column(frame["input_wspd_ms"], expected_steps)
    patv = (patv - scaler["Patv_clean_kw"]["mean"]) / scaler["Patv_clean_kw"]["std_population"]
    wspd = (wspd - scaler["Wspd"]["mean"]) / scaler["Wspd"]["std_population"]
    # Time-major ordering preserves the temporal sequence: Patv_t, Wspd_t, then t+1.
    history = np.stack((patv, wspd), axis=2).reshape(len(frame), expected_steps * 2)
    calendar = frame[list(INPUT_SOURCE_COLUMNS[2:])].to_numpy(dtype=np.float32, copy=True)
    inputs = np.concatenate((history, calendar), axis=1).astype(np.float32, copy=False)
    targets = frame["target_patv_pu"].to_numpy(dtype=np.float32, copy=True)
    return ClientArrays(frame, inputs, targets, frame["split"].astype(str).to_numpy())


def make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


@torch.inference_mode()
def predict(model: nn.Module, inputs: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    model.eval()
    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    dummy_targets = np.zeros(len(inputs), dtype=np.float32)
    for batch_inputs, _ in make_loader(inputs, dummy_targets, batch_size, False, 0):
        outputs.append(model(batch_inputs).cpu().numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(outputs, axis=0), elapsed


@torch.inference_mode()
def mean_loss(model: nn.Module, inputs: np.ndarray, targets: np.ndarray, quantiles: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch_inputs, batch_targets in make_loader(inputs, targets, batch_size, False, 0):
        batch_loss = pinball_loss(model(batch_inputs), batch_targets, quantiles)
        total += float(batch_loss) * len(batch_targets)
        count += len(batch_targets)
    return total / count


def train_model(
    model: nn.Module,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    config: dict,
) -> tuple[nn.Module, list[dict], int, float, bool]:
    quantiles = torch.tensor(config["quantiles"], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    early = config["early_stopping"]
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict] = []
    nonfinite = False
    started = time.perf_counter()

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        max_gradient_norm = 0.0
        loader = make_loader(train_inputs, train_targets, config["batch_size"], True, config["seed"] + epoch)
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(batch_inputs), batch_targets, quantiles)
            if not torch.isfinite(loss):
                nonfinite = True
                break
            loss.backward()
            squared_norm = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    squared_norm += float(torch.sum(parameter.grad.detach() ** 2))
            max_gradient_norm = max(max_gradient_norm, math.sqrt(squared_norm))
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_targets)
            sample_count += len(batch_targets)
        if nonfinite:
            break
        validation_loss = mean_loss(
            model, validation_inputs, validation_targets, quantiles, config["batch_size"]
        )
        train_loss = total_loss / sample_count
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "max_gradient_norm": max_gradient_norm,
        })
        if validation_loss < best_loss - early["min_delta"]:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early["patience"]:
                break

    runtime = time.perf_counter() - started
    if best_state is not None and early["restore_best"]:
        model.load_state_dict(best_state)
    return model, history, best_epoch, runtime, nonfinite


def persistence_metrics(arrays: ClientArrays, split: str = "test") -> dict[str, float]:
    inputs, targets, _ = arrays.select(split)
    # Last Patv is at index 70 in time-major 36x2 history and is standardized.
    scaler_note = "caller converts standardized last Patv separately"
    return {"_last_patv_standardized_mean": float(np.mean(inputs[:, -6])), "_note": scaler_note}


def empirical_quantile_predictions(train_targets: np.ndarray, count: int, quantiles: np.ndarray) -> np.ndarray:
    values = np.quantile(train_targets, quantiles)
    return np.repeat(values[None, :], count, axis=0)


def initial_model(config: dict) -> QuantileMLP:
    return QuantileMLP(config["input_dimension"], config["hidden_dimensions"], len(config["quantiles"]))


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
