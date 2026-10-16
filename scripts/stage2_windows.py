"""Reusable causal window-validity helpers for SDWPF Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


INVALID_REASONS = (
    "global_gap",
    "local_missing",
    "sampling_regime_boundary",
    "split_boundary",
    "target_missing",
    "insufficient_history",
    "other",
)


@dataclass(frozen=True)
class WindowContext:
    frequency_minutes: int
    regime_start: pd.Timestamp
    regime_end_exclusive: pd.Timestamp
    global_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()


def _crosses_interval(
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
) -> bool:
    return any(start <= gap_end and end >= gap_start for gap_start, gap_end in intervals)


def assess_window(
    client_frame: pd.DataFrame,
    issue_time: pd.Timestamp,
    lookback: pd.Timedelta,
    horizon: pd.Timedelta,
    context: WindowContext,
    *,
    target_valid_column: str = "target_valid",
    core_columns: tuple[str, ...] = ("Patv", "Wspd"),
) -> tuple[bool, str | None]:
    """Assess one direct-horizon sample without interpolation or future input.

    The history includes ``issue_time`` and spans exactly ``lookback`` on the
    configured regular grid.  The direct target is at ``issue_time+horizon``.
    Reasons are assigned in a stable priority order for reproducible counts.
    """
    issue_time = pd.Timestamp(issue_time)
    target_time = issue_time + horizon
    step = pd.Timedelta(minutes=context.frequency_minutes)
    history_start = issue_time - lookback + step

    if history_start < context.regime_start:
        return False, "insufficient_history"
    if target_time >= context.regime_end_exclusive:
        return False, "sampling_regime_boundary"
    if _crosses_interval(history_start, target_time, context.global_gaps):
        return False, "global_gap"

    expected = pd.date_range(history_start, issue_time, freq=step)
    history = client_frame.reindex(expected)
    if len(history) != int(lookback / step) or history[list(core_columns)].isna().any(axis=None):
        return False, "local_missing"
    if target_time not in client_frame.index:
        return False, "target_missing"
    target = client_frame.loc[target_time]
    if isinstance(target, pd.DataFrame):
        return False, "other"
    if not bool(target.get(target_valid_column, False)):
        return False, "target_missing"
    return True, None


def is_valid_window(
    client_id: int,
    issue_time: pd.Timestamp,
    lookback: pd.Timedelta,
    horizon: pd.Timedelta,
    client_frames: Mapping[int, pd.DataFrame],
    context: WindowContext,
) -> bool:
    """Public boolean API requested by the Stage 2 protocol."""
    valid, _ = assess_window(client_frames[client_id], issue_time, lookback, horizon, context)
    return valid


def causal_fill_mask(missing: np.ndarray, max_steps: int) -> np.ndarray:
    """Return values usable after a past-only limited forward fill.

    The first value of a series and runs longer than ``max_steps`` remain
    unusable.  This helper never looks at a future observation.
    """
    missing = np.asarray(missing, dtype=bool)
    usable = ~missing.copy()
    last_observed = -1
    for i, is_missing in enumerate(missing):
        if not is_missing:
            last_observed = i
        elif last_observed >= 0 and i - last_observed <= max_steps:
            usable[i] = True
    return usable
