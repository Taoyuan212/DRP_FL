"""FedAvg/FedProx primitives for the frozen Stage-4 protocol.

No personalized aggregation or relation-based weighting is implemented here.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from local_baseline import (
    ClientArrays,
    initial_model,
    load_client_arrays,
    make_loader,
    pinball_loss,
)


def model_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])


def state_vector(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1).cpu() for tensor in state.values()])


def weighted_average_states(
    states: list[dict[str, torch.Tensor]],
    sample_counts: Iterable[int],
) -> dict[str, torch.Tensor]:
    counts = np.asarray(list(sample_counts), dtype=np.float64)
    if len(states) != len(counts) or len(states) == 0:
        raise ValueError("States and positive sample counts must have equal non-zero length")
    if np.any(counts <= 0):
        raise ValueError("Aggregation sample counts must be positive")
    weights = counts / counts.sum()
    averaged: dict[str, torch.Tensor] = {}
    for key in states[0]:
        accumulator = torch.zeros_like(states[0][key], dtype=torch.float64)
        for weight, state in zip(weights, states):
            accumulator.add_(state[key].detach().to(torch.float64), alpha=float(weight))
        averaged[key] = accumulator.to(states[0][key].dtype)
    return averaged


def cosine_matrix(updates: np.ndarray) -> np.ndarray:
    updates = np.asarray(updates, dtype=np.float64)
    norms = np.linalg.norm(updates, axis=1)
    normalized = np.divide(updates, norms[:, None], out=np.zeros_like(updates), where=norms[:, None] > 0)
    matrix = normalized @ normalized.T
    np.fill_diagonal(matrix, 1.0)
    return np.clip(matrix, -1.0, 1.0)


def cosine_statistics(matrix: np.ndarray) -> dict[str, float]:
    values = matrix[np.triu_indices_from(matrix, k=1)]
    return {
        "cosine_mean": float(np.mean(values)),
        "cosine_median": float(np.median(values)),
        "cosine_fraction_negative": float(np.mean(values < 0)),
        "cosine_p5": float(np.quantile(values, 0.05)),
        "cosine_p95": float(np.quantile(values, 0.95)),
    }


def add_fedprox_gradient(
    parameters: Iterable[torch.nn.Parameter],
    anchors: Iterable[torch.Tensor],
    mu: float,
) -> torch.Tensor:
    """Add the exact FedProx gradient in-place and return its detached loss value.

    Constructing a second autograd graph for a parameter-wide squared norm caused
    pathological CPU overhead in the full 134-client loop.  The derivative of
    (mu/2)||theta-theta_global||^2 is exactly mu(theta-theta_global), so adding it
    after the base-loss backward pass is mathematically equivalent.
    """
    squared_norm = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for parameter, anchor in zip(parameters, anchors):
            difference = parameter.detach() - anchor
            squared_norm += torch.sum(difference.to(torch.float64) ** 2)
            if parameter.grad is None:
                parameter.grad = mu * difference
            else:
                parameter.grad.add_(difference, alpha=mu)
    return 0.5 * mu * squared_norm


@dataclass
class FederatedClientData:
    client_id: int
    arrays: ClientArrays
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_indices: np.ndarray


def load_client_cache(processed_dir: Path, client_ids: Iterable[int], scaler: dict, lookback_steps: int) -> dict[int, FederatedClientData]:
    cache = {}
    for client_id in client_ids:
        arrays = load_client_arrays(processed_dir / f"client_{client_id:03d}_windows.parquet", scaler, lookback_steps)
        train_x, train_y, _ = arrays.select("train")
        validation_x, validation_y, _ = arrays.select("validation")
        test_x, test_y, test_indices = arrays.select("test")
        cache[client_id] = FederatedClientData(
            client_id, arrays, train_x, train_y, validation_x, validation_y,
            test_x, test_y, test_indices,
        )
    return cache


def train_one_client(
    global_state: dict[str, torch.Tensor],
    data: FederatedClientData,
    config: dict,
    round_number: int,
    seed: int,
    mu: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    model = initial_model(config)
    model.load_state_dict(global_state)
    start_vector = model_vector(model)
    global_vector = state_vector(global_state)
    synchronization_error = float(torch.max(torch.abs(start_vector - global_vector)))
    reference = [tensor.detach().clone() for tensor in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    quantiles = torch.tensor(config["quantiles"], dtype=torch.float32)
    total_pinball = 0.0
    total_objective = 0.0
    samples = 0
    steps = 0
    finite = True
    for local_epoch in range(config["local_epochs"]):
        loader_seed = seed + round_number * 100_000 + data.client_id * 100 + local_epoch
        loader = make_loader(data.train_x, data.train_y, config["batch_size"], True, loader_seed)
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            base_loss = pinball_loss(predictions, targets, quantiles)
            if not torch.isfinite(base_loss):
                finite = False
                break
            base_loss.backward()
            proximal = add_fedprox_gradient(model.parameters(), reference, mu) if mu > 0 else torch.zeros((), dtype=torch.float64)
            objective_value = float(base_loss.detach()) + float(proximal)
            if not math.isfinite(objective_value):
                finite = False
                break
            optimizer.step()
            total_pinball += float(base_loss.detach()) * len(targets)
            total_objective += objective_value * len(targets)
            samples += len(targets)
            steps += 1
        if not finite:
            break
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    delta = state_vector(state) - global_vector
    return state, {
        "client_id": data.client_id,
        "train_samples": len(data.train_y),
        "local_steps": steps,
        "mean_train_pinball": total_pinball / samples if samples else math.nan,
        "mean_train_objective": total_objective / samples if samples else math.nan,
        "update_norm": float(torch.linalg.vector_norm(delta)),
        "synchronization_max_abs_error": synchronization_error,
        "finite": finite,
    }
