import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


windows = load_module("stage2_windows", "scripts/stage2_windows.py")


def test_invalid_reason_taxonomy_is_complete():
    assert set(windows.INVALID_REASONS) == {
        "global_gap",
        "local_missing",
        "sampling_regime_boundary",
        "split_boundary",
        "target_missing",
        "insufficient_history",
        "other",
    }


def test_causal_fill_never_uses_future_or_long_run():
    missing = np.array([True, False, True, True, True, False])
    assert windows.causal_fill_mask(missing, 2).tolist() == [False, True, True, True, False, True]


def test_valid_window_and_target_are_strictly_causal():
    idx = pd.date_range("2021-01-01", periods=10, freq="10min")
    frame = pd.DataFrame({"Patv": 1.0, "Wspd": 2.0, "target_valid": True}, index=idx)
    context = windows.WindowContext(10, idx[0], idx[-1] + pd.Timedelta(minutes=10))
    valid, reason = windows.assess_window(
        frame,
        idx[5],
        pd.Timedelta(minutes=30),
        pd.Timedelta(minutes=20),
        context,
    )
    assert valid and reason is None
    assert windows.is_valid_window(1, idx[5], pd.Timedelta(minutes=30), pd.Timedelta(minutes=20), {1: frame}, context)


def test_window_rejects_missing_history_and_regime_boundary():
    idx = pd.date_range("2021-01-01", periods=10, freq="10min")
    frame = pd.DataFrame({"Patv": 1.0, "Wspd": 2.0, "target_valid": True}, index=idx)
    frame.loc[idx[4], "Patv"] = np.nan
    context = windows.WindowContext(10, idx[0], idx[-1] + pd.Timedelta(minutes=10))
    assert windows.assess_window(frame, idx[5], pd.Timedelta(minutes=30), pd.Timedelta(minutes=10), context) == (False, "local_missing")
    assert windows.assess_window(frame, idx[8], pd.Timedelta(minutes=30), pd.Timedelta(minutes=20), context) == (False, "sampling_regime_boundary")


def test_window_rejects_global_gap():
    idx = pd.date_range("2021-01-01", periods=10, freq="10min")
    frame = pd.DataFrame({"Patv": 1.0, "Wspd": 2.0, "target_valid": True}, index=idx)
    context = windows.WindowContext(10, idx[0], idx[-1] + pd.Timedelta(minutes=10), ((idx[4], idx[4]),))
    assert windows.assess_window(frame, idx[5], pd.Timedelta(minutes=30), pd.Timedelta(minutes=10), context) == (False, "global_gap")
