"""Auditable primitives for the Stage-7B strong-PFL pilot.

Only train and official-validation rows are accepted by the loader.  Test rows,
quantile sorting, crossing correction, and proposed-method code are absent.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call

from local_baseline import INPUT_SOURCE_COLUMNS, QuantileMLP, forecast_metrics, make_loader, pinball_loss, predict


ALLOWED_SPLITS = ("train", "validation")


@dataclass
class PilotClient:
    client_id: int
    train_x: np.ndarray
    train_y: np.ndarray
    train_issue_time: np.ndarray
    train_target_time: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_train_validation_only(path: Path, scaler: dict, expected_steps: int = 36) -> PilotClient:
    columns = ["client_id", "issue_time", "target_time", "split", "is_valid", "target_patv_pu", *INPUT_SOURCE_COLUMNS]
    frame = pd.read_parquet(path, columns=columns, filters=[("split", "in", list(ALLOWED_SPLITS))])
    seen = set(frame["split"].astype(str).unique())
    if not seen.issubset(ALLOWED_SPLITS) or "test" in seen:
        raise RuntimeError(f"Test boundary violation in {path}: {sorted(seen)}")
    if not bool(frame["is_valid"].all()):
        raise RuntimeError(f"Invalid frozen window in {path}")
    patv = np.stack(frame["input_patv_clean_kw"].to_numpy()).astype(np.float32)
    wspd = np.stack(frame["input_wspd_ms"].to_numpy()).astype(np.float32)
    if patv.shape[1] != expected_steps or wspd.shape[1] != expected_steps:
        raise RuntimeError("Frozen lookback mismatch")
    patv = (patv - scaler["Patv_clean_kw"]["mean"]) / scaler["Patv_clean_kw"]["std_population"]
    wspd = (wspd - scaler["Wspd"]["mean"]) / scaler["Wspd"]["std_population"]
    history = np.stack((patv, wspd), axis=2).reshape(len(frame), expected_steps * 2)
    calendar = frame[list(INPUT_SOURCE_COLUMNS[2:])].to_numpy(dtype=np.float32)
    x = np.concatenate((history, calendar), axis=1).astype(np.float32)
    y = frame["target_patv_pu"].to_numpy(dtype=np.float32)
    split = frame["split"].astype(str).to_numpy()
    tr, va = split == "train", split == "validation"
    return PilotClient(
        int(frame["client_id"].iloc[0]), x[tr], y[tr],
        frame.loc[tr, "issue_time"].to_numpy(), frame.loc[tr, "target_time"].to_numpy(),
        x[va], y[va],
    )


def load_pilot_data(config: dict) -> dict[int, PilotClient]:
    root = Path(config["processed_dir"])
    scaler = json.loads((root / "scaler.json").read_text(encoding="utf-8"))
    return {cid: read_train_validation_only(root / f"client_{cid:03d}_windows.parquet", scaler, config["lookback_steps"])
            for cid in config["client_ids"]}


def clone_state(state: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in state.items())


def state_vector(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.detach().reshape(-1).cpu() for v in state.values()])


def vector_state(vector: torch.Tensor, template: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    out, offset = OrderedDict(), 0
    for key, value in template.items():
        n = value.numel()
        out[key] = vector[offset:offset+n].reshape_as(value).to(value.dtype).clone()
        offset += n
    if offset != vector.numel():
        raise ValueError("Parameter vector length mismatch")
    return out


def weighted_state(states: list[dict[str, torch.Tensor]], weights: Iterable[float]) -> OrderedDict[str, torch.Tensor]:
    w = np.asarray(list(weights), dtype=np.float64)
    if len(states) != len(w) or not np.isfinite(w).all() or abs(w.sum() - 1.0) > 1e-6:
        raise ValueError("Weights must be finite and normalized")
    return vector_state(sum(float(a) * state_vector(s).double() for a, s in zip(w, states)), states[0])


def model_from_state(state: dict[str, torch.Tensor]) -> QuantileMLP:
    model = QuantileMLP()
    model.load_state_dict(state)
    return model


def state_loss(state: dict[str, torch.Tensor], x: np.ndarray, y: np.ndarray, quantiles: torch.Tensor, batch_size: int) -> float:
    model = model_from_state(state)
    total = 0.0
    with torch.inference_mode():
        for bx, by in make_loader(x, y, batch_size, False, 0):
            total += float(pinball_loss(model(bx), by, quantiles)) * len(by)
    return total / len(y)


def validate_state(state: dict[str, torch.Tensor], data: PilotClient, quantiles: np.ndarray, batch_size: int) -> dict[str, float]:
    pred, elapsed = predict(model_from_state(state), data.validation_x, batch_size)
    metrics = forecast_metrics(data.validation_y, pred, quantiles, 1.0)
    metrics["inference_seconds"] = elapsed
    return metrics


def train_state(start: dict[str, torch.Tensor], x: np.ndarray, y: np.ndarray, config: dict,
                seed: int, lr: float | None = None, optimizer_name: str = "Adam",
                prox_center: dict[str, torch.Tensor] | None = None, prox_mu: float = 0.0,
                extra_gradient: torch.Tensor | None = None,
                cosine_center: torch.Tensor | None = None, cosine_lambda: float = 0.0) -> tuple[OrderedDict, dict]:
    model = model_from_state(start)
    if optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr or config["learning_rate"], momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr or config["learning_rate"])
    q = torch.tensor(config["quantiles"], dtype=torch.float32)
    prox = state_vector(prox_center).float() if prox_center is not None else None
    forward = backward = steps = 0
    total, seen = 0.0, 0
    started = time.perf_counter()
    for bx, by in make_loader(x, y, config["batch_size"], True, seed):
        opt.zero_grad(set_to_none=True)
        loss = pinball_loss(model(bx), by, q); forward += 1
        if prox is not None and prox_mu:
            current = torch.cat([p.reshape(-1) for p in model.parameters()])
            loss = loss + 0.5 * prox_mu * torch.sum((current - prox) ** 2)
        if cosine_center is not None and cosine_lambda:
            current = torch.cat([p.reshape(-1) for p in model.parameters()])
            loss = loss + cosine_lambda * torch.dot(cosine_center, current) / torch.clamp(torch.linalg.vector_norm(current), min=1e-12)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite local objective")
        loss.backward(); backward += 1
        if extra_gradient is not None:
            off = 0
            for p in model.parameters():
                n = p.numel(); p.grad.add_(extra_gradient[off:off+n].reshape_as(p)); off += n
        opt.step(); steps += 1
        total += float(loss.detach()) * len(by); seen += len(by)
    return clone_state(model.state_dict()), {"forward_passes": forward, "backward_passes": backward,
        "optimizer_steps": steps, "mean_objective": total / seen, "training_seconds": time.perf_counter() - started}


def fomo_inner_split(data: PilotClient, fraction: float = 0.1, purge_hours: int = 7) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(data.train_issue_time)
    n_val = max(1, int(math.ceil(len(order) * fraction)))
    val = order[-n_val:]
    val_start = data.train_issue_time[val].min()
    fit_limit = val_start - np.timedelta64(purge_hours, "h")
    fit = order[data.train_target_time[order] <= fit_limit]
    if len(fit) == 0 or data.train_target_time[fit].max() > fit_limit:
        raise RuntimeError("FedFomo purge failed")
    return fit, val


def fomo_candidate_indices(preference: np.ndarray, target: int, m: int, epsilon: float, rng: np.random.Generator) -> list[int]:
    possible = [j for j in range(len(preference)) if j != target]
    tie = rng.random(len(possible))
    ranked = [possible[k] for k in np.lexsort((tie, preference[possible]))[::-1]]
    random_order = list(rng.permutation(possible))
    selected = []
    for _ in range(m):
        pool = random_order if rng.random() < max(epsilon, 0.0) else ranked
        while pool:
            candidate = pool.pop(0)
            if candidate not in selected:
                selected.append(candidate); break
    return selected


def fomo_update(reference: dict[str, torch.Tensor], candidates: list[dict[str, torch.Tensor]],
                reference_loss: float, candidate_losses: list[float]) -> tuple[OrderedDict, np.ndarray]:
    ref = state_vector(reference).double()
    raw = np.asarray([(reference_loss - loss) / max(float(torch.linalg.vector_norm(state_vector(s).double() - ref)), 1e-12)
                      for s, loss in zip(candidates, candidate_losses)])
    positive = np.maximum(raw, 0.0)
    if positive.sum() <= 0:
        return clone_state(reference), raw
    weights = positive / positive.sum()
    out = ref.clone()
    for weight, state in zip(weights, candidates):
        out += float(weight) * (state_vector(state).double() - ref)
    return vector_state(out, reference), raw


def simplex_projection(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho_candidates = np.flatnonzero(u - cssv / (np.arange(len(u)) + 1) > 0)
    rho = int(rho_candidates[-1])
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def pfedgraph_weights(states: list[dict[str, torch.Tensor]], initial: dict[str, torch.Tensor], sample_prior: np.ndarray,
                       graph_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    base = state_vector(initial).double().numpy()
    updates = np.stack([state_vector(s).double().numpy() - base for s in states])
    norms = np.linalg.norm(updates, axis=1, keepdims=True)
    normalized = np.divide(updates, norms, out=np.zeros_like(updates), where=norms > 0)
    cosine = normalized @ normalized.T
    cosine[np.diag_indices_from(cosine)] = 1.0
    # Official QP: lambda||w-p||^2 - cosine_i^T w over the simplex.
    weights = np.stack([simplex_projection(sample_prior + cosine[i] / (2.0 * graph_lambda)) for i in range(len(states))])
    return weights, cosine


def apple_reg_coef(mu: float, round_index: int, rounds: int, fraction: float = 0.1) -> float:
    dynamic = int(rounds * fraction)
    if not mu or round_index >= dynamic:
        return 0.0
    return float((0.001 ** (round_index / dynamic)) * mu)


def apple_train_client(core_states: list[dict[str, torch.Tensor]], relation: np.ndarray, own: int,
                       p0: np.ndarray, data: PilotClient, config: dict, method: dict, round_index: int) -> tuple[OrderedDict, np.ndarray, dict]:
    template = QuantileMLP()
    own_params = OrderedDict((k, torch.nn.Parameter(v.clone())) for k, v in core_states[own].items())
    p = torch.nn.Parameter(torch.tensor(relation, dtype=torch.float32))
    optimizer = torch.optim.SGD([{"params": list(own_params.values()), "lr": method["lr_core"]},
                                 {"params": [p], "lr": method["lr_relation"]}], momentum=0.9)
    q = torch.tensor(config["quantiles"], dtype=torch.float32)
    fixed = [{k: v.detach() for k, v in s.items()} for s in core_states]
    coef = apple_reg_coef(method["mu"], round_index, config["rounds"], method["dynamic_fraction"])
    fp = bp = steps = 0; total = seen = 0
    started = time.perf_counter()
    for bx, by in make_loader(data.train_x, data.train_y, config["batch_size"], True,
                              config["seed"] + (round_index + 1) * 100000 + data.client_id):
        optimizer.zero_grad(set_to_none=True)
        mixed = OrderedDict()
        for key in core_states[0]:
            mixed[key] = sum(p[j] * (own_params[key] if j == own else fixed[j][key]) for j in range(len(core_states)))
        prediction = functional_call(template, mixed, (bx,)); fp += 1
        loss = pinball_loss(prediction, by, q) + coef * torch.sum((p - torch.tensor(p0, dtype=p.dtype)) ** 2)
        if not torch.isfinite(loss): raise FloatingPointError("APPLE non-finite")
        loss.backward(); bp += 1; optimizer.step(); steps += 1
        total += float(loss.detach()) * len(by); seen += len(by)
    return clone_state(own_params), p.detach().cpu().numpy().copy(), {"forward_passes": fp, "backward_passes": bp,
        "optimizer_steps": steps, "mean_objective": total / seen, "training_seconds": time.perf_counter()-started, "prox_coef": coef}


def apple_personalized_state(core_states: list[dict[str, torch.Tensor]], relation: np.ndarray) -> OrderedDict:
    # APPLE DRs are intentionally unconstrained in the official implementation.
    vector = sum(float(w) * state_vector(s).double() for w, s in zip(relation, core_states))
    return vector_state(vector, core_states[0])


def heur_fedamp_weights(states: list[dict[str, torch.Tensor]], self_weight: float, sigma: float) -> np.ndarray:
    vectors = torch.stack([state_vector(s).double() for s in states])
    vectors = vectors / torch.clamp(torch.linalg.vector_norm(vectors, dim=1, keepdim=True), min=1e-12)
    cosine = (vectors @ vectors.T).numpy(); n = len(states)
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mask = np.arange(n) != i
        z = sigma * cosine[i, mask]; z -= z.max()
        result[i, mask] = (1.0 - self_weight) * np.exp(z) / np.exp(z).sum()
        result[i, i] = self_weight
    return result


def pgfed_alpha_update(alpha: np.ndarray, loss_minuses: np.ndarray, mean_grad_dot_model: float, lr: float) -> np.ndarray:
    return np.maximum(alpha - lr * (loss_minuses + mean_grad_dot_model), 0.0)


def pinball_kink_audit(quantiles: Iterable[float]) -> list[float]:
    pred = torch.zeros((1, len(list(quantiles))), requires_grad=True)
    target = torch.zeros(1)
    q = torch.tensor(list(quantiles))
    pinball_loss(pred, target, q).backward()
    return pred.grad.detach().reshape(-1).tolist()


def full_gradient_and_loss_minus(state: dict[str, torch.Tensor], x: np.ndarray, y: np.ndarray,
                                 config: dict) -> tuple[torch.Tensor, float, dict]:
    """PGFed's uploaded gradient and first-order risk intercept on local Train."""
    model = model_from_state(state)
    q = torch.tensor(config["quantiles"], dtype=torch.float32)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    loss_sum, seen, batches = 0.0, 0, 0
    loaders = make_loader(x, y, config["batch_size"], False, 0)
    for bx, by in loaders:
        loss = pinball_loss(model(bx), by, q)
        # Official PGFed accumulates per-batch mean gradients, then divides by
        # the number of batches (not by the number of samples).
        loss.backward()
        loss_sum += float(loss.detach()) * len(by); seen += len(by); batches += 1
    grad = torch.cat([(p.grad.detach() / batches).reshape(-1).cpu() for p in model.parameters()])
    vector = state_vector(state).float()
    mean_loss = loss_sum / seen
    return grad, float(mean_loss - torch.dot(grad, vector)), {
        "forward_passes": batches, "backward_passes": batches, "optimizer_steps": 0,
    }


def pgfed_train_client(start: dict[str, torch.Tensor], x: np.ndarray, y: np.ndarray, config: dict,
                       seed: int, method: dict, alpha: np.ndarray,
                       prev_grads: np.ndarray | None, prev_loss_minuses: np.ndarray | None,
                       optimizer_state: dict | None) -> tuple[OrderedDict, np.ndarray, dict, dict]:
    """Official PGFed batch semantics with persistent SGD momentum state."""
    model = model_from_state(start)
    optimizer = torch.optim.SGD(model.parameters(), lr=method["predictor_learning_rate"], momentum=0.9)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    q = torch.tensor(config["quantiles"], dtype=torch.float32)
    alpha = np.asarray(alpha, dtype=np.float64).copy()
    extra = mean_grad = None
    if prev_grads is not None:
        extra = torch.tensor(method["mu"] * np.sum(alpha[:, None] * prev_grads, axis=0), dtype=torch.float32)
        mean_grad = torch.tensor(method["mu"] * np.mean(prev_grads, axis=0), dtype=torch.float32)
    fp = bp = steps = 0; total = seen = 0; started = time.perf_counter()
    for bx, by in make_loader(x, y, config["batch_size"], True, seed):
        optimizer.zero_grad(set_to_none=True)
        loss = pinball_loss(model(bx), by, q); fp += 1
        if not torch.isfinite(loss): raise FloatingPointError("PGFed non-finite")
        loss.backward(); bp += 1
        if extra is not None:
            off = 0
            for p in model.parameters():
                n = p.numel(); p.grad.add_(extra[off:off+n].reshape_as(p)); off += n
            dot = float(torch.dot(torch.cat([p.detach().reshape(-1) for p in model.parameters()]), mean_grad))
            alpha = pgfed_alpha_update(alpha, method["mu"] * prev_loss_minuses, dot,
                                        method["alpha_learning_rate"])
        optimizer.step(); steps += 1
        total += float(loss.detach()) * len(by); seen += len(by)
    return clone_state(model.state_dict()), alpha, copy.deepcopy(optimizer.state_dict()), {
        "forward_passes": fp, "backward_passes": bp, "optimizer_steps": steps,
        "mean_objective": total / seen, "training_seconds": time.perf_counter()-started,
        "nonlocal_gradient_norm": 0.0 if extra is None else float(torch.linalg.vector_norm(extra)),
    }
