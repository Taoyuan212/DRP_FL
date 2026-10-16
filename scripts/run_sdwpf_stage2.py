#!/usr/bin/env python3
"""Freeze SDWPF Stage 2 cleaning and forecasting-task definitions.

No predictive or federated model is trained.  Raw data are read-only.  The
processed V1 files are sample-index tables with causal source values and flags;
model-ready tensors are deliberately deferred to the baseline stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage2_windows import INVALID_REASONS


OFFICIAL_SOURCE = "Zhou et al., Scientific Data 11, 649 (2024), DOI:10.1038/s41597-024-03427-5"
OFFICIAL_URL = "https://www.nature.com/articles/s41597-024-03427-5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2.json"))
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def official_flags(frame: pd.DataFrame) -> pd.DataFrame:
    patv, wspd = frame["Patv"], frame["Wspd"]
    pitch = frame[["Pab1", "Pab2", "Pab3"]]
    return pd.DataFrame(
        {
            "official_unknown": ((patv <= 0) & (wspd > 2.5)) | pitch.gt(89).any(axis=1),
            "official_abnormal": frame["Ndir"].abs().gt(720) | frame["Wdir"].abs().gt(180),
        },
        index=frame.index,
    ).fillna(False)


def sampling_regimes(timeline: pd.DatetimeIndex) -> pd.DataFrame:
    """Return contiguous regimes separated by interval class changes or gaps."""
    deltas = pd.Series(timeline).diff().dt.total_seconds().div(60)
    interval_class = deltas.where(deltas.isin([10.0, 15.0]), -1.0)
    records = []
    regime_start = 0
    current = interval_class.iloc[1]
    for pos in range(2, len(timeline)):
        if interval_class.iloc[pos] != current:
            end = pos - 1
            segment_deltas = deltas.iloc[regime_start + 1 : end + 1]
            records.append((regime_start, end, current, segment_deltas))
            regime_start = pos - 1
            current = interval_class.iloc[pos]
    records.append((regime_start, len(timeline) - 1, current, deltas.iloc[regime_start + 1 :]))

    rows = []
    for idx, (start, end, dominant, segment_deltas) in enumerate(records, 1):
        if dominant == -1:
            prior_regular = deltas.iloc[: start + 1]
            prior_regular = prior_regular[prior_regular.isin([10.0, 15.0])]
            dominant_value = float(prior_regular.iloc[-1]) if len(prior_regular) else np.nan
        else:
            dominant_value = float(dominant)
        output_start = timeline[start]
        output_end = timeline[end]
        output_records = end - start + 1
        if dominant == -1:
            step = pd.Timedelta(minutes=dominant_value)
            output_start = timeline[start] + step
            output_end = timeline[end] - step
            output_records = 0
        rows.append(
            {
                "regime_id": f"R{idx}",
                "start_time": output_start,
                "end_time": output_end,
                "dominant_interval_minutes": dominant_value,
                "number_of_records": output_records,
                "number_of_transitions": len(segment_deltas),
                "abnormal_interval_ratio": 1.0 if dominant == -1 else (float((segment_deltas != dominant_value).mean()) if len(segment_deltas) else 0.0),
                "interval_class": "regular" if dominant in (10.0, 15.0) else "gap_transition",
            }
        )
    return pd.DataFrame(rows)


def longest_missing_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return int((changes[1::2] - changes[::2]).max(initial=0))


def rolling_complete(mask: np.ndarray, window: int) -> np.ndarray:
    """Whether every point in each trailing window is usable."""
    values = pd.Series(np.asarray(mask, dtype=np.int8)).rolling(window, min_periods=window).sum()
    return values.eq(window).to_numpy()


def direct_target(values: np.ndarray, steps: int, fill_value=False) -> np.ndarray:
    out = np.full(len(values), fill_value, dtype=np.asarray(values).dtype)
    if steps < len(values):
        out[:-steps] = values[steps:]
    return out


def rolling_past_only_fillable(series: pd.Series, max_steps: int) -> np.ndarray:
    """Usable mask after causal extrapolation from two prior observations.

    This is a count-only sensitivity rule.  Actual imputation is deliberately
    not materialized in Stage 2.  Runs at the beginning or without two finite
    predecessors remain unusable.
    """
    values = series.to_numpy(float)
    usable = np.isfinite(values)
    finite_positions = []
    run_length = 0
    for pos, value in enumerate(values):
        if np.isfinite(value):
            finite_positions.append(pos)
            run_length = 0
        else:
            run_length += 1
            if run_length <= max_steps and len(finite_positions) >= 2:
                usable[pos] = True
    return usable


def assign_split(target_times: pd.Series, splits: dict) -> pd.Series:
    result = pd.Series(pd.NA, index=target_times.index, dtype="string")
    for name, (start, end) in splits.items():
        result.loc[(target_times >= pd.Timestamp(start)) & (target_times < pd.Timestamp(end))] = name
    return result


def classify_invalid(
    history_ready: np.ndarray,
    target_in_regime: np.ndarray,
    target_exists: np.ndarray,
    target_valid: np.ndarray,
) -> np.ndarray:
    reason = np.full(len(history_ready), "", dtype=object)
    reason[~history_ready] = "local_missing"
    reason[~target_valid] = "target_missing"
    reason[~target_exists] = "target_missing"
    reason[~target_in_regime] = "sampling_regime_boundary"
    return reason


def quantile_distance(values_by_client: list[np.ndarray], n_quantiles: int = 101) -> np.ndarray:
    quantile_grid = np.linspace(0, 1, n_quantiles)
    summaries = []
    for values in values_by_client:
        finite = values[np.isfinite(values)]
        summaries.append(np.quantile(finite, quantile_grid))
    summaries = np.asarray(summaries)
    return np.mean(np.abs(summaries[:, None, :] - summaries[None, :, :]), axis=2)


def curve_distance(curves: np.ndarray) -> np.ndarray:
    out = np.full((len(curves), len(curves)), np.nan)
    for i in range(len(curves)):
        for j in range(i, len(curves)):
            valid = np.isfinite(curves[i]) & np.isfinite(curves[j])
            value = float(np.sqrt(np.mean((curves[i, valid] - curves[j, valid]) ** 2))) if valid.any() else np.nan
            out[i, j] = out[j, i] = value
    return out


def levinson_pacf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """PACF from one finite segment using Durbin-Levinson recursion."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values - values.mean()
    if len(values) <= max_lag or np.allclose(values, 0):
        return np.full(max_lag + 1, np.nan)
    gamma = np.array([np.dot(values[lag:], values[: len(values) - lag]) / len(values) for lag in range(max_lag + 1)])
    phi = np.zeros((max_lag + 1, max_lag + 1))
    pacf = np.full(max_lag + 1, np.nan)
    pacf[0] = 1.0
    for order in range(1, max_lag + 1):
        denominator = gamma[0] - sum(phi[order - 1, j] * gamma[j] for j in range(1, order))
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            break
        numerator = gamma[order] - sum(phi[order - 1, j] * gamma[order - j] for j in range(1, order))
        phi[order, order] = numerator / denominator
        for j in range(1, order):
            phi[order, j] = phi[order - 1, j] - phi[order, order] * phi[order - 1, order - j]
        pacf[order] = phi[order, order]
    return pacf


def longest_finite_segment(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    padded = np.r_[False, finite, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    if not len(changes):
        return np.array([], dtype=float)
    starts, ends = changes[::2], changes[1::2]
    position = int(np.argmax(ends - starts))
    return values[starts[position] : ends[position]]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_documents(out: Path, cfg: dict, totals: dict, non_iid: dict, regimes: pd.DataFrame) -> None:
    sampling = f"""# Sampling strategy

## Decision

Use the native 10-minute records from `2021-01-01 00:10:00` through `2021-12-31 23:50:00` (analysis end is exclusive at `2022-01-01 00:00:00`). Do not interpolate across either system-wide gap and do not resample the main task.

## Evidence

- The file contains {int(((regimes.dominant_interval_minutes == 15) & (regimes.interval_class == 'regular')).sum())} regular 15-minute regimes and {int(((regimes.dominant_interval_minutes == 10) & (regimes.interval_class == 'regular')).sum())} regular 10-minute regime, separated by two gap transitions; the exact boundaries are in `table_sampling_regimes.csv`.
| Criterion | A: aggregate both years to 30 min | B: 2021-only native 10 min | C: separate yearly models |
|---|---|---|---|
| Data fidelity | No upsampling, but replaces native records with unequal 2-point/3-point aggregates | Preserves released 2021 values exactly | Preserves both native regimes |
| Continuity | Keeps the April gap and year-boundary gap; neither may be bridged | Complete regular grid inside the selected year | 2020 remains split by the April gap |
| FL synchronization | Common grid, but aggregation semantics differ by source regime | Direct common grid for all clients | Separate synchronization schedules/models |
| Sample size | Largest duration, lowest resolution | One full year and {int((pd.Timestamp('2022-01-01')-pd.Timestamp('2021-01-01')).total_seconds()/600):,} nominal time points per client | Uses both years, but not in one comparable task |
| Paper explanation | Requires defending mixed averaging supports | Simple and auditable, while disclosing reduced span | Difficult: year, cadence and model state are confounded |
| Horizon meaning | Clear after aggregation, but 10-min dynamics are irreversibly lost | Clear direct 10/60/180-min targets | Same clock horizons, different step counts and models |

Strategy B is selected for V1. Strategy A is not selected merely for retaining more dates because it changes the temporal support of the labels. Strategy C is retained only as a future sensitivity design.

The paper's claim that the full file is 10-minute data conflicts with the observed file: 2020 transitions are predominantly 15 minutes. The project follows the observed timestamps and reports this discrepancy rather than silently imposing a grid.
"""
    (out / "sampling_strategy.md").write_text(sampling, encoding="utf-8")

    cleaning = f"""# Data cleaning policy V1

Official source: [{OFFICIAL_SOURCE}]({OFFICIAL_URL}). Project rules below are explicitly separated from official rules.

## Main task policy

| Class | Input history | Target label | Delete raw row | Interpolate | Indicator | Robustness branch |
|---|---|---|---|---|---|---|
| Missing Patv/Wspd | Invalidates the complete V1 window | Mask | Never | Main: none | Flags retained in index | Causal fill up to 1 or 2 steps is count-only sensitivity |
| Official unknown power | Keep observed historical Patv/Wspd; it is causally known operating history | Mask from Task B | Never | Never | `history_unknown_count`, `target_official_unknown` | Task A later, not in Stage 2 data |
| Official abnormal record (Wdir/Ndir rule) | Core Patv/Wspd are retained only for history because excluded fields are not V1 inputs | Mask | Never | Never | `target_official_abnormal` | Retain flags for audit |
| Project physical/sensor flag | If it affects Wspd, invalidates V1 history; Etmp/Itmp are excluded from V1 | Mask only when the Core Wspd is flagged; an unused temperature sensor alone does not invalidate Patv | Never | Never | Flags retained | Threshold sensitivity required before adding those fields |
| Negative Patv not official-unknown | Historical/target value is clipped to zero in the processed view, raw is unchanged | Valid after clipping | Never | Not applicable | Raw and clipped value retained | Unclipped-score sensitivity later |

Project physical thresholds are audit rules, not official SDWPF definitions: Wspd outside [{cfg['physical_sensor_flags']['wspd_min_ms']}, {cfg['physical_sensor_flags']['wspd_max_ms']}] m/s, and Etmp/Itmp outside [{cfg['physical_sensor_flags']['temperature_min_c']}, {cfg['physical_sensor_flags']['temperature_max_c']}] °C. Temperature fields are excluded from V1, so their flags neither remove a Core-feature history window nor invalidate an otherwise valid Patv target. This avoids manufacturing target selection from an unused sensor; the flags are retained for audit.

## Boundary rules

A sample is invalid if history or direct target crosses a global gap, a sampling-regime boundary, or a split boundary; if history lacks any Core value; or if the target is missing/unknown/abnormal. V1 uses no interpolation. The reusable boolean interface is `is_valid_window(...)` in `scripts/stage2_windows.py`.
"""
    (out / "data_cleaning_policy.md").write_text(cleaning, encoding="utf-8")

    task = f"""# Forecasting task definition V1

## Frozen primary estimand

For each of 134 turbine clients, estimate quantiles q={{0.1,...,0.9}} of a single future active-power observation under a normal-operation target mask, conditional only on locally observed information through issue time. The primary direct horizon is {cfg['primary_horizon_minutes']} minutes; secondary direct horizons are {cfg['secondary_horizons_minutes'][0]} and {cfg['secondary_horizons_minutes'][1]} minutes. The history length is {cfg['lookback_hours']} hours.

Task B is the main experiment because official unknown/missing/abnormal targets are explicitly described as unsuitable for model evaluation. Task A (realized operational power including state outcomes) would require predicting exogenous scheduling/maintenance states that are not identified in this dataset; it remains an extension, not the main estimand.

Unknown-power observations may appear in history because they are known at issue time, but they are never scored as future labels. This avoids discarding most otherwise observable histories while preserving the official target mask. The target uses per-unit power (`Patv / 1500 kW`); processed files retain kW as well.

Main-134 is frozen for the primary experiment. Sensitivity-133 excludes client 83 only after all primary results are produced.

## Evidence adequacy

The selected 6-hour history is supported by the empirical audit in `table_lookback_diagnostics.csv`: the cross-client median Patv ACF is materially higher at 6 hours than at 12 or 24 hours, while PACF is near zero at all three long lags. Six hours is therefore the least costly defensible V1 context, not a proven optimum; 12 and 24 hours remain predeclared sensitivities. No model was trained in Stage 2.
"""
    (out / "task_definition.md").write_text(task, encoding="utf-8")

    decision_log = f"""# Decision Log

| ID | Date | Decision | Alternatives | Evidence | Consequence / risk | Status | Revisit trigger |
|---|---|---|---|---|---|---|---|
| D-S2-01 | 2026-08-13 | 2021-only native 10-minute V1 | 30-min both years; separate-year models | Observed cadence regimes and gaps | One year only; avoids changed temporal support | Frozen V1 | Multi-year robustness phase |
| D-S2-02 | 2026-08-13 | Task B normal-operation target mask | Task A operational power | Official usage notes; retained sample audit | Cannot claim outage/scheduling forecast | Frozen V1 | Explicit status labels become available |
| D-S2-03 | 2026-08-13 | No interpolation in main data | Causal fill 1/2 steps | Complete-window counts remain adequate | Lower retained rate, highest traceability | Frozen V1 | A client lacks adequate split samples |
| D-S2-04 | 2026-08-13 | Patv, Wspd, issue-time cyclic calendar V1 | ERA5 and SCADA state fields | Stage 1 availability/leakage audit | Deliberately limited predictive ceiling | Frozen V1 | Baseline failure diagnoses missing exogenous signal |
| D-S2-05 | 2026-08-13 | Primary 60 min; secondary 10/180 min | 30 min, 6 h | Operational range, difficulty spread, no NWP | 3 h may expose lack of future wind input | Frozen V1 | Pilot shows degenerate persistence comparison |
| D-S2-06 | 2026-08-13 | 6 h lookback; 12/24 h sensitivities | Single arbitrary window | ACF/PACF audit and cost | Not an optimized choice | Frozen V1 | Baseline sensitivity is material |
| D-S2-07 | 2026-08-13 | Shared training-only scaling for Patv/Wspd; target per-unit | Local scalers | Relation uses parameter-update cosine similarity | Shares aggregate scaler statistics; local-vs-shared remains required ablation | Frozen V1 | Relation conclusions change under local scaling |
| D-S2-08 | 2026-08-13 | Main-134; Sensitivity-133 excludes 83 | Remove 83 initially | Stage 1 quality audit | High-missing client may stress FL | Frozen V1 | Client 83 makes training numerically infeasible |

## Normalization-specific analysis

Local standardization is natural for private clients, but client-specific affine transforms alter input-to-hidden gradients and hence `Δθ_i`; cosine similarity can then reflect scaler differences rather than collaborative value. V1 therefore uses a common scaler estimated only from pooled training partitions (share only sufficient statistics in a deployment implementation), and a common 1500 kW physical target scale. A later relation-robustness ablation must compare local scaling and layer-wise update normalization. This is not evidence that shared scaling is always superior.
"""
    (out / "Decision_Log.md").write_text(decision_log, encoding="utf-8")

    summary = {
        "primary_valid_samples": totals["primary_valid_samples"],
        "non_iid_after_cleaning": non_iid,
        "status": "Stage 2 complete; training not started",
    }
    write_json(out / "stage2_summary.json", summary)


def main() -> int:
    args = parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    raw_path = (root / cfg["input_path"]).resolve()
    out = (root / cfg["output_dir"]).resolve()
    processed = (root / cfg["processed_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    columns = ["TurbID", "Tmstamp", "Patv", "Wspd", "Wdir", "Ndir", "Etmp", "Itmp", "Pab1", "Pab2", "Pab3"]
    frame = pd.read_parquet(raw_path, columns=columns)
    frame["Tmstamp"] = pd.to_datetime(frame["Tmstamp"], errors="raise")
    frame = frame.sort_values(["TurbID", "Tmstamp"], kind="mergesort").reset_index(drop=True)
    timeline = pd.DatetimeIndex(frame["Tmstamp"].drop_duplicates().sort_values())
    regimes = sampling_regimes(timeline)
    regimes.to_csv(out / "table_sampling_regimes.csv", index=False, encoding="utf-8-sig")

    start = pd.Timestamp(cfg["analysis_start"])
    end = pd.Timestamp(cfg["analysis_end_exclusive"])
    model = frame.loc[(frame["Tmstamp"] >= start) & (frame["Tmstamp"] < end)].copy()
    flags = official_flags(model)
    model = pd.concat([model, flags], axis=1)
    physical = cfg["physical_sensor_flags"]
    model["project_wspd_abnormal"] = ~model["Wspd"].between(physical["wspd_min_ms"], physical["wspd_max_ms"])
    model["project_temperature_abnormal"] = (~model["Etmp"].between(physical["temperature_min_c"], physical["temperature_max_c"])) | (~model["Itmp"].between(physical["temperature_min_c"], physical["temperature_max_c"]))
    model["target_valid"] = model["Patv"].notna() & ~model["official_unknown"] & ~model["official_abnormal"] & ~model["project_wspd_abnormal"]
    model["Patv_clean_kw"] = model["Patv"].clip(lower=0)
    model["Patv_pu"] = model["Patv_clean_kw"] / float(cfg["normalization"]["rated_power_kw"])

    freq = int(cfg["frequency_minutes"])
    lookback_steps = int(cfg["lookback_hours"] * 60 / freq)
    primary_steps = int(cfg["primary_horizon_minutes"] / freq)
    splits = cfg["splits"]
    horizons = cfg["all_materialized_horizons_minutes"]
    rng = np.random.default_rng(int(cfg["random_seed"]))
    manifest_rows, strategy_rows, invalid_counts = [], [], []
    audit_rows, client_curves, patv_values, wspd_values = [], [], [], []
    acf_rows = []
    all_index_parts = []
    wind_bins = np.asarray(cfg["non_iid"]["wind_bins_ms"], dtype=float)
    bin_centers = (wind_bins[:-1] + wind_bins[1:]) / 2

    for client_id, group in model.groupby("TurbID", sort=True):
        group = group.sort_values("Tmstamp").reset_index(drop=True)
        n = len(group)
        expected = pd.date_range(start, end - pd.Timedelta(minutes=freq), freq=f"{freq}min")
        if n != len(expected) or not np.array_equal(group["Tmstamp"].to_numpy(), expected.to_numpy()):
            raise ValueError(f"Client {client_id}: 2021 timeline is not a complete {freq}-minute grid")

        observed_core = group[["Patv", "Wspd"]].notna().all(axis=1).to_numpy() & ~group["project_wspd_abnormal"].to_numpy()
        target_valid_now = group["target_valid"].to_numpy(bool)
        target_exists_now = group["Patv"].notna().to_numpy()
        history_unknown_count = pd.Series(group["official_unknown"].to_numpy(np.int16)).rolling(lookback_steps, min_periods=lookback_steps).sum().fillna(-1).to_numpy(np.int16)
        base_history_start = group["Tmstamp"] - pd.Timedelta(hours=cfg["lookback_hours"]) + pd.Timedelta(minutes=freq)

        patv_series = group["Patv_clean_kw"].to_numpy(float)
        finite_segment = longest_finite_segment(patv_series)
        # Cap the deterministic segment for PACF cost while preserving the same start.
        pacf_segment = finite_segment[: min(len(finite_segment), 5000)]
        max_lag = max(int(hours * 60 / freq) for hours in cfg["lookback_candidates_hours"])
        pacf_values = levinson_pacf(pacf_segment, max_lag)
        for hours in cfg["lookback_candidates_hours"]:
            lag = int(hours * 60 / freq)
            left, right = patv_series[:-lag], patv_series[lag:]
            finite_pair = np.isfinite(left) & np.isfinite(right)
            acf_value = float(np.corrcoef(left[finite_pair], right[finite_pair])[0, 1]) if finite_pair.sum() > 2 else np.nan
            acf_rows.append(
                {
                    "client_id": client_id,
                    "lookback_hours": hours,
                    "lag_steps": lag,
                    "acf": acf_value,
                    "pacf": float(pacf_values[lag]),
                    "acf_pair_count": int(finite_pair.sum()),
                    "pacf_segment_length": int(len(pacf_segment)),
                }
            )

        strategy_masks = {
            "A_no_interpolation": observed_core,
            "B_past_only_linear_extrapolation_1": rolling_past_only_fillable(group["Patv"], 1) & rolling_past_only_fillable(group["Wspd"], 1) & ~group["project_wspd_abnormal"].to_numpy(),
            "B_past_only_linear_extrapolation_2": rolling_past_only_fillable(group["Patv"], 2) & rolling_past_only_fillable(group["Wspd"], 2) & ~group["project_wspd_abnormal"].to_numpy(),
        }
        primary_target_times = group["Tmstamp"] + pd.Timedelta(minutes=cfg["primary_horizon_minutes"])
        primary_split = assign_split(primary_target_times, splits)
        primary_boundary_ok = np.zeros(n, dtype=bool)
        for split_name, (split_start, _) in splits.items():
            in_split = primary_split.eq(split_name).fillna(False).to_numpy()
            primary_boundary_ok[in_split] = (base_history_start.loc[in_split] >= pd.Timestamp(split_start)).to_numpy()
        for strategy, usable_points in strategy_masks.items():
            ready = rolling_complete(usable_points, lookback_steps)
            tv = direct_target(target_valid_now, primary_steps)
            candidate = np.arange(n) + primary_steps < n
            valid = ready & tv & candidate & primary_split.notna().to_numpy() & primary_boundary_ok
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "client_id": client_id,
                    "candidate_issue_times": int(candidate.sum()),
                    "valid_samples": int(valid.sum()),
                    "discarded_samples": int(candidate.sum() - valid.sum()),
                    "retained_rate": float(valid.sum() / candidate.sum()),
                    "longest_original_core_missing_run_steps": longest_missing_run(~observed_core),
                }
            )

        ready = rolling_complete(observed_core, lookback_steps)
        horizon_valid_counts = {}
        for horizon_minutes in horizons:
            steps = int(horizon_minutes / freq)
            candidate = np.arange(n) + steps < n
            tv = direct_target(target_valid_now, steps)
            horizon_target_times = group["Tmstamp"] + pd.Timedelta(minutes=horizon_minutes)
            horizon_split = assign_split(horizon_target_times, splits)
            horizon_boundary_ok = np.zeros(n, dtype=bool)
            for split_name, (split_start, _) in splits.items():
                in_split = horizon_split.eq(split_name).fillna(False).to_numpy()
                horizon_boundary_ok[in_split] = (base_history_start.loc[in_split] >= pd.Timestamp(split_start)).to_numpy()
            valid = ready & tv & candidate & horizon_split.notna().to_numpy() & horizon_boundary_ok
            horizon_valid_counts[str(horizon_minutes)] = int(valid.sum())

        steps = primary_steps
        target_times = group["Tmstamp"] + pd.Timedelta(minutes=cfg["primary_horizon_minutes"])
        target_in_regime = (target_times < end).to_numpy()
        target_exists = direct_target(target_exists_now, steps)
        target_valid = direct_target(target_valid_now, steps)
        reasons = classify_invalid(ready, target_in_regime, target_exists, target_valid)
        # Initial history is a separate protocol reason, not local missingness.
        reasons[np.arange(n) < lookback_steps - 1] = "insufficient_history"
        valid = reasons == ""
        split = assign_split(target_times, splits)
        # No history may originate before the start of its target split.
        history_start = base_history_start
        for split_name, (split_start, split_end) in splits.items():
            in_split = split.eq(split_name).fillna(False).to_numpy()
            crosses_split = in_split & (history_start < pd.Timestamp(split_start)).to_numpy()
            reasons[crosses_split & (reasons == "")] = "split_boundary"
        valid = reasons == ""

        target_patv_kw = direct_target(group["Patv_clean_kw"].to_numpy(float), steps, np.nan)
        target_patv_pu = direct_target(group["Patv_pu"].to_numpy(float), steps, np.nan)
        target_unknown = direct_target(group["official_unknown"].to_numpy(bool), steps)
        target_abnormal = direct_target((group["official_abnormal"] | group["project_wspd_abnormal"]).to_numpy(bool), steps)
        index = pd.DataFrame(
            {
                "client_id": int(client_id),
                "issue_time": group["Tmstamp"],
                "history_start": history_start,
                "target_time": target_times,
                "horizon_minutes": cfg["primary_horizon_minutes"],
                "split": split,
                "is_valid": valid,
                "invalid_reason": np.where(valid, pd.NA, reasons),
                "history_unknown_count": history_unknown_count,
                "target_official_unknown": target_unknown,
                "target_abnormal": target_abnormal,
                "patv_issue_raw_kw": group["Patv"],
                "patv_issue_clean_kw": group["Patv_clean_kw"],
                "wspd_issue_ms": group["Wspd"],
                "target_patv_kw": target_patv_kw,
                "target_patv_pu": target_patv_pu,
                "hour_sin": np.sin(2 * np.pi * (group["Tmstamp"].dt.hour * 60 + group["Tmstamp"].dt.minute) / 1440),
                "hour_cos": np.cos(2 * np.pi * (group["Tmstamp"].dt.hour * 60 + group["Tmstamp"].dt.minute) / 1440),
                "day_of_year_sin": np.sin(2 * np.pi * (group["Tmstamp"].dt.dayofyear - 1) / 365),
                "day_of_year_cos": np.cos(2 * np.pi * (group["Tmstamp"].dt.dayofyear - 1) / 365),
            }
        )
        index.to_parquet(processed / f"client_{int(client_id):03d}.parquet", index=False)
        valid_positions = np.flatnonzero(valid)
        patv_windows = np.lib.stride_tricks.sliding_window_view(group["Patv_clean_kw"].to_numpy(float), lookback_steps)
        wspd_windows = np.lib.stride_tricks.sliding_window_view(group["Wspd"].to_numpy(float), lookback_steps)
        aligned_positions = valid_positions - (lookback_steps - 1)
        if (aligned_positions < 0).any():
            raise AssertionError(f"Client {client_id}: valid sample lacks complete history")
        scalar_columns = index.loc[valid, [
            "client_id", "issue_time", "history_start", "target_time", "horizon_minutes", "split", "is_valid",
            "history_unknown_count", "target_official_unknown", "target_abnormal", "target_patv_kw", "target_patv_pu",
            "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos",
        ]].reset_index(drop=True)
        window_table = pa.Table.from_pandas(scalar_columns, preserve_index=False)
        patv_list = pa.FixedSizeListArray.from_arrays(pa.array(patv_windows[aligned_positions].reshape(-1), type=pa.float32()), lookback_steps)
        wspd_list = pa.FixedSizeListArray.from_arrays(pa.array(wspd_windows[aligned_positions].reshape(-1), type=pa.float32()), lookback_steps)
        window_table = window_table.append_column("input_patv_clean_kw", patv_list).append_column("input_wspd_ms", wspd_list)
        pq.write_table(window_table, processed / f"client_{int(client_id):03d}_windows.parquet", compression="zstd")
        all_index_parts.append(index[["client_id", "issue_time", "target_time", "split", "is_valid", "invalid_reason"]])

        for reason in INVALID_REASONS:
            invalid_counts.append({"client_id": client_id, "reason": reason, "count": int((reasons == reason).sum())})
        split_counts = index.loc[index["is_valid"], "split"].value_counts()
        invalid_n = int((~index["is_valid"]).sum())
        manifest_rows.append(
            {
                "dataset_version": "Main-134",
                "client_id": client_id,
                "train_sample_count": int(split_counts.get("train", 0)),
                "validation_sample_count": int(split_counts.get("validation", 0)),
                "test_sample_count": int(split_counts.get("test", 0)),
                "valid_sample_count": int(index["is_valid"].sum()),
                "invalid_sample_count": invalid_n,
                "retained_rate": float(index["is_valid"].mean()),
                "valid_samples_by_horizon_minutes": json.dumps(horizon_valid_counts, sort_keys=True),
                "target_unknown_rate": float(group["official_unknown"].mean()),
                "core_missing_rate": float(group[["Patv", "Wspd"]].isna().any(axis=1).mean()),
            }
        )

        cleaned_valid = group["target_valid"]
        patv_values.append(group.loc[cleaned_valid, "Patv_clean_kw"].to_numpy(float))
        wspd_values.append(group.loc[cleaned_valid, "Wspd"].to_numpy(float))
        bins = pd.cut(group["Wspd"], wind_bins, right=False, labels=False)
        curve = []
        for bin_idx in range(len(bin_centers)):
            values = group.loc[cleaned_valid & bins.eq(bin_idx), "Patv_clean_kw"]
            curve.append(float(values.mean()) if values.notna().sum() >= cfg["non_iid"]["minimum_curve_bin_count"] else np.nan)
        client_curves.append(curve)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(strategy_rows).to_csv(out / "table_cleaning_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(invalid_counts).to_csv(out / "table_invalid_window_reason_counts.csv", index=False, encoding="utf-8-sig")
    lookback = pd.DataFrame(acf_rows)
    lookback.to_csv(out / "table_lookback_diagnostics_by_client.csv", index=False, encoding="utf-8-sig")
    lookback.groupby("lookback_hours")[["acf", "pacf", "acf_pair_count", "pacf_segment_length"]].agg(["min", "median", "max"]).to_csv(out / "table_lookback_diagnostics.csv", encoding="utf-8-sig")

    all_index = pd.concat(all_index_parts, ignore_index=True)
    valid_index = all_index[all_index["is_valid"]]
    if len(valid_index) < cfg["manual_timestamp_check_count"]:
        raise ValueError("Not enough valid samples for the manual timestamp audit")
    sample_idx = rng.choice(valid_index.index.to_numpy(), cfg["manual_timestamp_check_count"], replace=False)
    audit = valid_index.loc[sample_idx].sort_values(["client_id", "issue_time"]).copy()
    audit["input_before_target"] = audit["issue_time"] < audit["target_time"]
    audit["manual_review_status"] = "pending_human_review"
    audit.to_csv(out / "table_manual_timestamp_audit_100.csv", index=False, encoding="utf-8-sig")

    # Shared scaler is calculated only from records whose timestamp is in the training interval.
    train_start, train_end = map(pd.Timestamp, splits["train"])
    scaler_fit_mask = (
        (model["Tmstamp"] >= train_start)
        & (model["Tmstamp"] < train_end)
        & model[["Patv", "Wspd"]].notna().all(axis=1)
        & ~model["project_wspd_abnormal"]
    )
    train_core = model.loc[scaler_fit_mask, ["Patv_clean_kw", "Wspd"]]
    scaler = {
        "method": cfg["normalization"]["input"],
        "fit_start": train_start,
        "fit_end_exclusive": train_end,
        "fit_rows": int(len(train_core)),
        "fit_record_rule": "training timestamps only; Patv and Wspd observed; Wspd inside frozen project range",
        "Patv_clean_kw": {"mean": float(train_core["Patv_clean_kw"].mean()), "std_population": float(train_core["Patv_clean_kw"].std(ddof=0))},
        "Wspd": {"mean": float(train_core["Wspd"].mean()), "std_population": float(train_core["Wspd"].std(ddof=0))},
        "target_scale_kw": float(cfg["normalization"]["rated_power_kw"]),
    }
    write_json(processed / "scaler.json", scaler)

    rated = float(cfg["normalization"]["rated_power_kw"])
    rated_validation = pd.DataFrame(
        {
            "client_id": [client for client, _ in model.groupby("TurbID", sort=True)],
            "patv_q99_kw": [float(group["Patv_clean_kw"].quantile(0.99)) for _, group in model.groupby("TurbID", sort=True)],
            "patv_q999_kw": [float(group["Patv_clean_kw"].quantile(0.999)) for _, group in model.groupby("TurbID", sort=True)],
            "patv_max_kw": [float(group["Patv_clean_kw"].max()) for _, group in model.groupby("TurbID", sort=True)],
        }
    )
    rated_validation["declared_rated_power_kw"] = rated
    rated_validation["same_documented_turbine_type"] = True
    rated_validation["evidence_note"] = "SDWPF paper: all 134 turbines are SL1500/82; independent technical specification: SL1500/82 rated power 1500 kW"
    rated_validation.to_csv(out / "table_rated_power_validation.csv", index=False, encoding="utf-8-sig")

    feature_set = {
        "version": "v1",
        "historical_features": ["Patv_clean_kw", "Wspd"],
        "issue_time_features": ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"],
        "excluded": ["ERA5", "Wdir", "Ndir", "Pab1", "Pab2", "Pab3", "Etmp", "Itmp", "Prtv", "operating_states"],
        "causality": "All history ends at issue_time; calendar features use issue_time only.",
    }
    write_json(out / "feature_set_v1.json", feature_set)
    dataset_versions = {
        "Main-134": {"included_clients": "all 134 clients", "role": "primary experiments"},
        "Sensitivity-133": {"included_clients": "all except client 83", "excluded_clients": [83], "role": "predeclared data-quality sensitivity only"},
        "partition_layout": {"client_###.parquet": "all candidate issue-time audit index", "client_###_windows.parquet": "valid primary-horizon samples with fixed-size Core-feature input windows"},
        "no_training_in_stage2": True,
    }
    write_json(out / "dataset_versions.json", dataset_versions)
    split_definition = {
        "scheme": "fixed_date_chronological_half_open_intervals",
        "timezone": "UTC+08:00 per SDWPF documentation; timestamps stored as naive local time",
        "splits": splits,
        "purge_rule": f"A {cfg['lookback_hours']} h history window must start no earlier than its target split start; boundaries are purged per client.",
        "normalization_fit_split": "train only",
    }
    write_json(out / "split_definition.json", split_definition)

    clients = manifest["client_id"].astype(int).tolist()
    p_matrix = quantile_distance(patv_values)
    w_matrix = quantile_distance(wspd_values)
    c_matrix = curve_distance(np.asarray(client_curves))
    for matrix, filename in [(p_matrix, "matrix_clean_patv_distance.csv"), (w_matrix, "matrix_clean_wspd_distance.csv"), (c_matrix, "matrix_clean_conditional_powercurve_distance.csv")]:
        pd.DataFrame(matrix, index=clients, columns=clients).to_csv(out / filename, encoding="utf-8-sig")
    offdiag = ~np.eye(len(clients), dtype=bool)
    non_iid = {
        "patv_quantile_distance_median_kw": float(np.nanmedian(p_matrix[offdiag])),
        "wspd_quantile_distance_median_ms": float(np.nanmedian(w_matrix[offdiag])),
        "conditional_powercurve_rmse_median_kw": float(np.nanmedian(c_matrix[offdiag])),
    }
    quality_rank = manifest[["client_id", "retained_rate"]].copy()
    stage1_quality_path = root / "results" / "eda" / "table_client_quality.csv"
    if stage1_quality_path.exists():
        stage1_quality = pd.read_csv(stage1_quality_path)
        stage1_quality["stage1_quality_rank"] = stage1_quality["missing_rate"].rank(method="average", ascending=True)
        quality_rank = quality_rank.merge(stage1_quality[["client_id", "missing_rate", "stage1_quality_rank"]], on="client_id", how="left")
        quality_rank["stage2_retained_rank"] = quality_rank["retained_rate"].rank(method="average", ascending=False)
        quality_rank["rank_shift"] = quality_rank["stage2_retained_rank"] - quality_rank["stage1_quality_rank"]
        spearman = float(quality_rank[["stage1_quality_rank", "stage2_retained_rank"]].corr(method="spearman").iloc[0, 1])
    else:
        quality_rank["stage2_retained_rank"] = quality_rank["retained_rate"].rank(method="average", ascending=False)
        spearman = np.nan
    quality_rank.to_csv(out / "table_cleaning_client_rank_sensitivity.csv", index=False, encoding="utf-8-sig")
    non_iid["stage1_quality_vs_stage2_retained_rank_spearman"] = spearman
    stage1_sensitivity_path = root / "results" / "eda" / "table_non_iid_quality_sensitivity.csv"
    if stage1_sensitivity_path.exists():
        stage1_sensitivity = pd.read_csv(stage1_sensitivity_path)
        comparison_rows = []
        for matrix_name, stage2_value in [
            ("Patv distribution", non_iid["patv_quantile_distance_median_kw"]),
            ("Wspd distribution", non_iid["wspd_quantile_distance_median_ms"]),
            ("Conditional power curve", non_iid["conditional_powercurve_rmse_median_kw"]),
        ]:
            prior = stage1_sensitivity.loc[stage1_sensitivity["matrix"].eq(matrix_name)].iloc[0]
            comparison_rows.append(
                {
                    "matrix": matrix_name,
                    "stage1_official_valid_median": prior["official_valid_offdiagonal_median"],
                    "stage2_clean_2021_median": stage2_value,
                    "change": stage2_value - prior["official_valid_offdiagonal_median"],
                    "interpretation": "Not causal: Stage 2 also changes calendar span and quality policy",
                }
            )
        pd.DataFrame(comparison_rows).to_csv(out / "table_non_iid_before_after_cleaning.csv", index=False, encoding="utf-8-sig")
    totals = {"primary_valid_samples": int(manifest["valid_sample_count"].sum())}
    write_documents(out, cfg, totals, non_iid, regimes)

    # Validation assertions are deliberately duplicated in the run, not only tests.
    assert (audit["issue_time"] < audit["target_time"]).all()
    assert (manifest[["train_sample_count", "validation_sample_count", "test_sample_count"]].min(axis=1) > 0).all()
    assert scaler["fit_end_exclusive"] == train_end
    assert np.allclose(p_matrix, p_matrix.T, equal_nan=True) and np.allclose(np.diag(p_matrix), 0)
    assert np.allclose(c_matrix, c_matrix.T, equal_nan=True) and np.allclose(np.diag(c_matrix), 0)

    script_path = Path(__file__).resolve()
    manifest_json = {
        "stage": 2,
        "raw_path": str(raw_path),
        "raw_sha256": sha256(raw_path),
        "script_path": str(script_path),
        "script_sha256": sha256(script_path),
        "window_helper_sha256": sha256(script_path.parent / "stage2_windows.py"),
        "config_path": str(cfg_path),
        "config_sha256": sha256(cfg_path),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "official_source": OFFICIAL_SOURCE,
        "training_performed": False,
    }
    write_json(out / "run_manifest.json", manifest_json)
    print(json.dumps({"status": "ok", "valid_samples": totals["primary_valid_samples"], "non_iid": non_iid}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
