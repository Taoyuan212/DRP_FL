"""Pure DRP-FL V1 aggregation primitives.

The functions here deliberately do not train models. V1 reuses the frozen
FedAvg local updates and changes only the prediction-time aggregation matrix.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def ema_relation(previous: np.ndarray | None, current: np.ndarray, beta: float) -> np.ndarray:
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    current = np.asarray(current, dtype=np.float64)
    if previous is None:
        result = current.copy()
    else:
        result = beta * np.asarray(previous, dtype=np.float64) + (1.0 - beta) * current
    np.fill_diagonal(result, 1.0)
    return result


def relation_weights(similarity: np.ndarray, sample_weights: np.ndarray, tau: float) -> np.ndarray:
    """Compute stable p-weighted softmax rows.

    If every similarity in a row is equal, the row is exactly the FedAvg
    sample-weight vector up to floating-point roundoff.
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    p = np.asarray(sample_weights, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    if p.shape != (similarity.shape[1],) or np.any(p <= 0):
        raise ValueError("sample_weights must be positive and match similarity")
    if tau <= 0:
        raise ValueError("tau must be positive")
    p = p / p.sum()
    logits = similarity / tau + np.log(p)[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def relation_statistics(weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=np.float64)
    self_weight = np.diag(weights)
    entropy = -np.sum(np.where(weights > 0, weights * np.log(weights), 0.0), axis=1)
    effective = np.exp(entropy)
    return {
        "mean_self_weight": float(np.mean(self_weight)),
        "median_self_weight": float(np.median(self_weight)),
        "p5_self_weight": float(np.quantile(self_weight, 0.05)),
        "p95_self_weight": float(np.quantile(self_weight, 0.95)),
        "mean_external_mass": float(np.mean(1.0 - self_weight)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_effective_collaborators": float(np.mean(effective)),
        "median_effective_collaborators": float(np.median(effective)),
    }


def personalized_vectors(local_vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    local_vectors = np.asarray(local_vectors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(local_vectors), len(local_vectors)):
        raise ValueError("weight matrix and client vectors do not align")
    return (weights @ local_vectors).astype(np.float32)


def vector_to_state(vector: np.ndarray, template: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    flat = np.asarray(vector)
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for key, tensor in template.items():
        size = tensor.numel()
        values = torch.from_numpy(flat[offset:offset + size].copy()).reshape(tensor.shape).to(tensor.dtype)
        state[key] = values
        offset += size
    if offset != len(flat):
        raise ValueError("vector length does not match template")
    return state


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_validation_candidate(rows: list[dict], tolerance: float = 1e-12) -> dict:
    """Primary macro Pinball, then NT rate, then worst degradation."""
    if not rows:
        raise ValueError("no validation candidates")
    ordered = sorted(rows, key=lambda row: (
        round(float(row["macro_validation_pinball"]) / tolerance) if tolerance > 0 else float(row["macro_validation_pinball"]),
        float(row["validation_nt_rate"]),
        float(row["worst_validation_degradation"]),
        float(row.get("beta", 0.0)),
        float(row.get("tau", 0.0)),
        int(row["round"]),
    ))
    return ordered[0]


def permute_relation(similarity: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Relabel both targets and sources, preserving the full value distribution."""
    similarity = np.asarray(similarity)
    permutation = np.asarray(permutation)
    return similarity[np.ix_(permutation, permutation)]


def shuffle_target_rows(similarity: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return np.asarray(similarity)[np.asarray(permutation), :]
