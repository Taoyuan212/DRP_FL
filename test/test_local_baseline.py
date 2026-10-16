import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("local_baseline", ROOT / "scripts/local_baseline.py")
baseline = importlib.util.module_from_spec(spec)
sys.modules["local_baseline"] = baseline
spec.loader.exec_module(baseline)


def test_mlp_shape_parameter_count_and_forbidden_modules():
    model = baseline.QuantileMLP()
    assert model(torch.zeros((3, 76))).shape == (3, 9)
    assert baseline.parameter_count(model) == 13833
    names = {type(module).__name__ for module in model.modules()}
    assert not names.intersection({"Transformer", "GRU", "LSTM", "Conv1d", "BatchNorm1d", "Dropout", "MultiheadAttention"})


def test_pinball_zero_for_exact_prediction():
    y = torch.tensor([0.2, 0.8])
    pred = y[:, None].repeat(1, 2)
    assert baseline.pinball_loss(pred, y, torch.tensor([0.1, 0.9])).item() == 0.0


def test_q90_penalizes_underestimate_more_than_q10():
    y = torch.tensor([1.0])
    pred = torch.tensor([[0.0]])
    low = baseline.pinball_loss(pred, y, torch.tensor([0.1]))
    high = baseline.pinball_loss(pred, y, torch.tensor([0.9]))
    assert torch.isclose(low, torch.tensor(0.1))
    assert torch.isclose(high, torch.tensor(0.9))


def test_batch_pinball_matches_manual_calculation():
    y = torch.tensor([0.0, 1.0])
    pred = torch.tensor([[0.2, 0.4], [0.6, 1.2]])
    q = torch.tensor([0.1, 0.9])
    manual = (0.9 * 0.2 + 0.1 * 0.4 + 0.1 * 0.4 + 0.1 * 0.2) / 4
    assert torch.isclose(baseline.pinball_loss(pred, y, q), torch.tensor(manual))


def test_metrics_are_manually_verifiable():
    q = np.arange(0.1, 1.0, 0.1)
    y = np.array([0.0, 1.0])
    predictions = np.array([
        [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    ])
    metrics = baseline.forecast_metrics(y, predictions, q)
    expected_pinball = baseline.pinball_by_sample(y, predictions, q).mean()
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert np.isclose(metrics["Pinball"], expected_pinball)
    assert np.isclose(metrics["CRPS_approx"], 2 * expected_pinball)
    assert metrics["PICP_80"] == 1.0
    assert np.isclose(metrics["PINAW_80"], 0.8)
    assert metrics["PICP_60"] == 1.0
    assert np.isclose(metrics["PINAW_60"], 0.6)
    assert metrics["quantile_crossing_rate"] == 0.0


def test_crossing_rate_is_sample_fraction_and_outputs_are_not_sorted():
    q = np.arange(0.1, 1.0, 0.1)
    y = np.array([0.5, 0.5])
    predictions = np.tile(q, (2, 1))
    predictions[1, 3], predictions[1, 4] = 0.6, 0.4
    original = predictions.copy()
    metrics = baseline.forecast_metrics(y, predictions, q)
    assert metrics["quantile_crossing_rate"] == 0.5
    assert np.array_equal(predictions, original)
