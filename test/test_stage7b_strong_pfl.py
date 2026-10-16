from collections import OrderedDict
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from local_baseline import QuantileMLP
from stage7b_core import (
    apple_personalized_state, clone_state, fomo_candidate_indices, fomo_update,
    heur_fedamp_weights, pgfed_alpha_update, pfedgraph_weights, pinball_kink_audit,
    simplex_projection, state_vector, weighted_state,
)


def states(n=3):
    torch.manual_seed(7)
    base = clone_state(QuantileMLP().state_dict())
    return [clone_state(base) for _ in range(n)], base


def test_predictor_output_and_parameter_count():
    model = QuantileMLP()
    assert sum(p.numel() for p in model.parameters()) == 13833
    assert model(torch.zeros(2, 76)).shape == (2, 9)


def test_fedfomo_normalization_identical_and_persistent_selection():
    ss, base = states()
    out, raw = fomo_update(base, ss[1:], 0.2, [0.2, 0.2])
    assert torch.equal(state_vector(out), state_vector(base))
    assert np.isfinite(raw).all()
    pref = np.array([0.0, 2.0, 1.0])
    a = fomo_candidate_indices(pref, 0, 2, 0, np.random.default_rng(4))
    b = fomo_candidate_indices(pref, 0, 2, 0, np.random.default_rng(4))
    assert a == b == [1, 2]


def test_apple_identical_core_models_are_permutation_consistent():
    ss, base = states()
    a = apple_personalized_state(ss, np.array([0.2, 0.3, 0.5]))
    b = apple_personalized_state([ss[2], ss[0], ss[1]], np.array([0.5, 0.2, 0.3]))
    assert torch.allclose(state_vector(a), state_vector(b), atol=0, rtol=0)
    assert torch.allclose(state_vector(a), state_vector(base), atol=1e-7)


def test_pfedgraph_simplex_objective_and_permutation():
    ss, base = states()
    # Give non-zero, distinct update vectors.
    for i, s in enumerate(ss):
        first = next(iter(s)); s[first] = s[first] + (i + 1) * 1e-4
    prior = np.array([0.2, 0.3, 0.5])
    w, cos = pfedgraph_weights(ss, base, prior, 0.8)
    assert np.min(w) >= 0 and np.max(np.abs(w.sum(1)-1)) <= 1e-12
    for i in range(3):
        objective = 0.8*np.sum((w[i]-prior)**2)-np.dot(cos[i], w[i])
        initial = -np.dot(cos[i], prior)
        assert objective <= initial + 1e-10
    perm = np.array([2, 0, 1])
    wp, _ = pfedgraph_weights([ss[i] for i in perm], base, prior[perm], 0.8)
    assert np.allclose(w[np.ix_(perm,perm)], wp, atol=1e-12)


def test_fedamp_manual_two_path_toy_and_identical_behavior():
    ss, _ = states()
    w = heur_fedamp_weights(ss, 0.5, 1.0)
    manual = np.full((3,3), 0.25); np.fill_diagonal(manual, 0.5)
    assert np.allclose(w, manual, atol=1e-12)
    assert np.allclose(w.sum(1), 1)
    aggregates = [state_vector(weighted_state(ss,row)) for row in w]
    assert all(torch.allclose(aggregates[0], x, atol=1e-7) for x in aggregates[1:])


def test_pgfed_alpha_changes_with_nonlocal_risk_and_kink_is_finite():
    a = np.ones(2)/2
    left = pgfed_alpha_update(a, np.array([1.0, 2.0]), 0.2, 0.01)
    right = pgfed_alpha_update(a, np.array([2.0, 1.0]), 0.2, 0.01)
    assert not np.allclose(left, right)
    kink = np.asarray(pinball_kink_audit([.1,.2,.3,.4,.5,.6,.7,.8,.9]))
    assert np.isfinite(kink).all() and kink.shape == (9,)


def test_fedamp_and_pgfed_permutation_consistency():
    ss, _ = states()
    for i, state in enumerate(ss):
        key = next(iter(state)); state[key] = state[key] + i * 1e-3
    w = heur_fedamp_weights(ss, 0.5, 1.0)
    perm = np.array([2, 0, 1])
    wp = heur_fedamp_weights([ss[i] for i in perm], 0.5, 1.0)
    assert np.allclose(w[np.ix_(perm, perm)], wp, atol=1e-12)
    alpha = np.array([0.2, 0.3, 0.5]); risks = np.array([1.0, 3.0, 2.0])
    updated = pgfed_alpha_update(alpha, risks, 0.4, 0.01)
    updated_perm = pgfed_alpha_update(alpha[perm], risks[perm], 0.4, 0.01)
    assert np.allclose(updated[perm], updated_perm)
    # PGFed's global-model aggregation is sample-weight normalized.
    aggregate = weighted_state(ss, alpha)
    aggregate_perm = weighted_state([ss[i] for i in perm], alpha[perm])
    assert torch.allclose(state_vector(aggregate), state_vector(aggregate_perm), atol=1e-7)


def test_checkpoint_roundtrip(tmp_path):
    ss, _ = states()
    path = tmp_path / "state.pt"
    torch.save(ss, path)
    loaded = torch.load(path, weights_only=True)
    assert all(torch.equal(state_vector(a), state_vector(b)) for a,b in zip(ss,loaded))
