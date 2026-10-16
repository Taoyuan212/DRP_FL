import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_sdwpf_eda.py"
spec = importlib.util.spec_from_file_location("run_sdwpf_eda", MODULE_PATH)
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)


def test_official_flags_are_separate():
    df = pd.DataFrame(
        {
            "Patv": [0, 100, 100, 100],
            "Wspd": [3, 3, 3, 3],
            "Pab1": [0, 90, 0, 0],
            "Pab2": [0, 0, 0, 0],
            "Pab3": [0, 0, 0, 0],
            "Ndir": [0, 0, 721, 0],
            "Wdir": [0, 0, 0, 181],
        }
    )
    flags = eda.official_flags(df)
    assert flags["official_unknown_stopped"].tolist() == [True, False, False, False]
    assert flags["official_unknown_pitch"].tolist() == [False, True, False, False]
    assert flags["official_abnormal_ndir"].tolist() == [False, False, True, False]
    assert flags["official_abnormal_wdir"].tolist() == [False, False, False, True]


def test_longest_true_run():
    assert eda.longest_true_run(np.array([False, True, True, False, True])) == (2, 2)
    assert eda.longest_true_run(np.zeros(4, dtype=bool)) == (0, 0)


def test_quantile_distance_is_symmetric():
    m = eda.robust_quantile_distance([np.array([0, 1, 2]), np.array([1, 2, 3])], 11)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0)
    assert m[0, 1] > 0


def test_month_boundary_falls_back_when_no_internal_month_end():
    idx = pd.date_range("2020-01-01", periods=10 * 24, freq="h")
    target = idx[167]
    assert eda.internal_month_boundary(idx, target, idx[0], idx[-1]) == target


def test_month_boundary_selects_valid_internal_month_end():
    idx = pd.date_range("2020-01-01", "2020-06-30 23:00", freq="h")
    target = pd.Timestamp("2020-04-10")
    boundary = eda.internal_month_boundary(idx, target, idx[0], idx[-1])
    assert boundary == pd.Timestamp("2020-03-31 23:00")
