from collections import OrderedDict

import numpy as np
import torch

from scripts.drp_v1 import (
    ema_relation,
    personalized_vectors,
    relation_statistics,
    relation_weights,
    select_validation_candidate,
    vector_to_state,
)


def test_equal_relation_reduces_to_fedavg():
    p = np.array([2.0, 3.0, 5.0])
    similarity = np.full((3, 3), 0.37)
    weights = relation_weights(similarity, p, tau=0.05)
    np.testing.assert_allclose(weights, np.tile(p / p.sum(), (3, 1)), atol=1e-14)


def test_weights_valid_and_statistics_nontrivial():
    similarity = np.eye(3)
    weights = relation_weights(similarity, np.ones(3), tau=0.5)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0)
    stats = relation_statistics(weights)
    assert 0 < stats["mean_external_mass"] < 1
    assert 1 < stats["mean_effective_collaborators"] <= 3


def test_ema_initial_and_recurrence():
    first = np.array([[1.0, 0.2], [0.2, 1.0]])
    second = np.array([[1.0, -0.2], [-0.2, 1.0]])
    smoothed = ema_relation(ema_relation(None, first, 0.9), second, 0.5)
    np.testing.assert_allclose(smoothed, np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_personalized_vectors_and_state_roundtrip():
    local = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    weights = np.array([[1, 0], [0.25, 0.75]])
    vectors = personalized_vectors(local, weights)
    np.testing.assert_allclose(vectors[1], [3.25, 4.25, 5.25])
    template = OrderedDict(a=torch.zeros(2), b=torch.zeros(1))
    state = vector_to_state(vectors[1], template)
    np.testing.assert_allclose(torch.cat(list(state.values())).numpy(), vectors[1])


def test_validation_selection_uses_registered_order():
    rows = [
        {"macro_validation_pinball": 0.1, "validation_nt_rate": 0.3, "worst_validation_degradation": 0.2, "round": 2},
        {"macro_validation_pinball": 0.1, "validation_nt_rate": 0.2, "worst_validation_degradation": 0.3, "round": 3},
        {"macro_validation_pinball": 0.11, "validation_nt_rate": 0.0, "worst_validation_degradation": 0.0, "round": 1},
    ]
    assert select_validation_candidate(rows)["round"] == 3
