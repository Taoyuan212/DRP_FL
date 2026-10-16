"""Deterministic implementation audits for Stage 7C calibration."""
from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np
import torch

import scripts.run_stage7c as stage7c
from scripts.run_stage7c import apple_personalized_state_fast, candidate_name, fedamp_centers_fast, make_fixed_bank, score_round, vector_mapping
from scripts.stage7b_core import QuantileMLP, heur_fedamp_weights, state_vector, vector_state, weighted_state


def _state(value: float) -> dict[str, torch.Tensor]:
    template = QuantileMLP(input_dim=3).state_dict()
    return {key: torch.full_like(tensor, value) for key, tensor in template.items()}


def test_apple_weighted_aggregation_matches_reference() -> None:
    states = [_state(value) for value in (1.0, 2.0, 4.0)]
    weights = np.array([0.2, 0.3, 0.5])
    actual = apple_personalized_state_fast(states, weights)
    expected = weighted_state(states, weights)
    for key in expected:
        assert torch.allclose(actual[key], expected[key], atol=1e-12, rtol=0)


def test_vectorized_apple_rows_match_individual_aggregation() -> None:
    states = [_state(value) for value in (1.0, 2.0, 4.0)]
    relations = np.array([[.7,.2,.1],[.2,.6,.2],[.1,.2,.7]])
    bank = torch.stack([state_vector(s).double() for s in states])
    actual = [vector_state(row, states[0]) for row in torch.as_tensor(relations) @ bank]
    for i, state in enumerate(actual):
        expected = apple_personalized_state_fast(states, relations[i])
        for key in expected:
            assert torch.allclose(state[key], expected[key], atol=1e-12, rtol=0)


def test_flat_parameter_mapping_retains_exact_parameter_values() -> None:
    state = _state(2.0)
    vector = state_vector(state).float().requires_grad_(True)
    mapping = vector_mapping(vector, state)
    for key in state:
        assert torch.allclose(mapping[key], state[key])
    sum(value.sum() for value in mapping.values()).backward()
    assert torch.all(vector.grad == 1)
    bank = make_fixed_bank([_state(1.0), state])
    assert bank.shape == (2, vector.numel())


def test_fedamp_vectorized_centers_match_reference() -> None:
    states = [_state(value) for value in (1.0, 2.0, 4.0)]
    weights, centers = fedamp_centers_fast(states, self_weight=.5, sigma=1.)
    reference_weights = heur_fedamp_weights(states, self_weight=.5, sigma=1.)
    assert np.allclose(weights, reference_weights, atol=1e-12, rtol=0)
    for i, center in enumerate(centers):
        expected = weighted_state(states, reference_weights[i])
        for key in expected:
            assert torch.allclose(center[key], expected[key], atol=1e-11, rtol=0)


def test_score_order_and_locked_grid() -> None:
    better = [{"Pinball": .1, "quantile_crossing_rate": .01}, {"Pinball": .2, "quantile_crossing_rate": .02}]
    worse_primary = [{"Pinball": .100001, "quantile_crossing_rate": 0}, {"Pinball": .200001, "quantile_crossing_rate": 0}]
    assert score_round(better) < score_round(worse_primary)
    assert candidate_name("apple", .005) == "apple_0p005"
    config = json.loads(Path("configs/stage7c.json").read_text(encoding="utf-8"))
    assert config["test_read"] is False
    assert config["apple"]["core_learning_rate_candidates"] == [.005,.01,.02]
    assert config["fedamp"]["proximal_coefficient_multipliers"] == [.5,1.,2.]


def test_resume_guard_hard_fails_wrong_lr_seed_and_client_order() -> None:
    config = stage7c.read_config(Path("configs/stage7ca_resume_b.json"))
    stage7c.ALL_CLIENTS = list(config["client_ids"])
    checkpoint = torch.load(
        "results/stage7ca/production_b_v2/apple/apple_0p01/checkpoint_round_001.pt",
        map_location="cpu", weights_only=False)
    for kind in ("lr", "seed", "clients"):
        changed = copy.deepcopy(config)
        value = 0.01
        if kind == "lr":
            value = 0.02
        elif kind == "seed":
            changed["seed"] = 2027
        else:
            changed["client_ids"] = changed["client_ids"][::-1]
        stage7c.ALL_CLIENTS = list(changed["client_ids"])
        try:
            stage7c._guard_resume(checkpoint, changed, value)
        except RuntimeError as exc:
            assert "HARD FAIL" in str(exc)
        else:
            raise AssertionError(f"{kind} mismatch did not hard fail")
