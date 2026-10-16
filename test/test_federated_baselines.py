import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("federated_baselines", ROOT / "scripts/federated_baselines.py")
fed = importlib.util.module_from_spec(spec)
sys.modules["federated_baselines"] = fed
spec.loader.exec_module(fed)


def test_weighted_average_matches_manual_two_client_calculation():
    states = [
        {"weight": torch.tensor([1.0, 3.0]), "bias": torch.tensor([2.0])},
        {"weight": torch.tensor([5.0, 7.0]), "bias": torch.tensor([6.0])},
    ]
    result = fed.weighted_average_states(states, [1, 3])
    assert torch.allclose(result["weight"], torch.tensor([4.0, 6.0]))
    assert torch.allclose(result["bias"], torch.tensor([5.0]))


def test_weighted_average_rejects_nonpositive_counts():
    try:
        fed.weighted_average_states([{"x": torch.tensor([1.0])}], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("zero sample count must fail")


def test_cosine_matrix_and_statistics_are_manual():
    updates = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    matrix = fed.cosine_matrix(updates)
    assert np.allclose(matrix, [[1, -1, 0], [-1, 1, 0], [0, 0, 1]])
    stats = fed.cosine_statistics(matrix)
    assert np.isclose(stats["cosine_mean"], -1 / 3)
    assert np.isclose(stats["cosine_fraction_negative"], 1 / 3)


def test_proximal_term_is_zero_at_global_and_has_expected_gradient():
    parameter = torch.tensor([2.0, -1.0], requires_grad=True)
    anchor = torch.tensor([1.0, 1.0])
    mu = 0.1
    proximal = 0.5 * mu * torch.sum((parameter - anchor) ** 2)
    proximal.backward()
    assert torch.allclose(parameter.grad, mu * (parameter.detach() - anchor))
    same = 0.5 * mu * torch.sum((anchor - anchor) ** 2)
    assert same.item() == 0.0


def test_inplace_fedprox_gradient_matches_autograd_objective():
    mu = 0.01
    direct = torch.tensor([2.0, -1.0], requires_grad=True)
    anchor = torch.tensor([1.0, 1.0])
    base_direct = torch.sum(direct**2)
    objective = base_direct + 0.5 * mu * torch.sum((direct - anchor) ** 2)
    objective.backward()
    expected_gradient = direct.grad.clone()

    inplace = torch.tensor([2.0, -1.0], requires_grad=True)
    torch.sum(inplace**2).backward()
    prox_value = fed.add_fedprox_gradient([inplace], [anchor], mu)
    assert torch.allclose(inplace.grad, expected_gradient)
    assert torch.isclose(prox_value, torch.tensor(0.5 * mu * 5.0, dtype=torch.float64))


def test_state_and_model_vectors_use_same_parameter_order():
    config = {"input_dimension": 76, "hidden_dimensions": [64, 64, 64], "quantiles": list(np.arange(0.1, 1.0, 0.1))}
    model = fed.initial_model(config)
    assert torch.equal(fed.model_vector(model), fed.state_vector(model.state_dict()))
