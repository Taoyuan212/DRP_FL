import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local_baseline import initial_model
from stage6e import head_relations_and_weights, parameter_layout, qa_head_vectors


def frozen_layout():
    stage4 = json.loads((ROOT / "configs/federated_baselines.json").read_text())
    config = json.loads((ROOT / "configs/stage6e.json").read_text())
    return parameter_layout(initial_model(stage4).state_dict(), config["parameter_whitelist"], config["expected_parameter_counts"])


def test_named_layout_exactly_partitions_13833_parameters():
    layout = frozen_layout()
    assert len(layout.body_indices) == 13248
    assert [len(x) for x in layout.head_indices] == [65] * 9
    assert layout.total == 13833


def test_quantile_weights_sum_to_one_and_single_model_is_formed():
    layout = frozen_layout()
    rng = np.random.default_rng(6)
    updates = rng.normal(size=(4, layout.total))
    local = rng.normal(size=(4, layout.total))
    p = np.asarray([1, 2, 3, 4], dtype=float)
    relations, heads = head_relations_and_weights(updates, p, 0.2, layout)
    body = np.tile(p / p.sum(), (4, 1))
    output = qa_head_vectors(local, body, heads, layout)
    assert relations.shape == (9, 4, 4)
    assert np.allclose(heads.sum(axis=2), 1.0)
    assert output.shape == (4, 13833)
