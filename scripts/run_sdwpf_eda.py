#!/usr/bin/env python3
"""Reproducible, non-modeling SDWPF asset audit and EDA.

The script is deliberately read-only with respect to raw data. It does not
interpolate, delete, replace, or train predictive/federated models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).parents[1] / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


OFFICIAL_SOURCE = "Zhou et al., Scientific Data 11, 649 (2024), DOI:10.1038/s41597-024-03427-5, Usage Notes"
OFFICIAL_URL = "https://doi.org/10.1038/s41597-024-03427-5"

FIELD_METADATA = {
    "TurbID": ("Turbine identifier", "Data-quality / status feature"),
    "Day": ("Day of record (KDD release); chronology component", "Time feature"),
    "Tmstamp": ("Record creation time; official timezone UTC+08:00", "Time feature"),
    "Wspd": ("Wind speed at turbine top, mechanical anemometer (m/s)", "Meteorological feature"),
    "Wdir": ("Relative wind direction: wind direction minus nacelle direction (degree)", "Directional feature"),
    "Etmp": ("Environmental temperature measured on outer nacelle surface (degree C)", "Meteorological feature"),
    "Itmp": ("Temperature inside nacelle (degree C)", "Turbine operating feature"),
    "Ndir": ("Nacelle yaw direction from true north (degree)", "Directional feature"),
    "Pab1": ("Blade 1 pitch angle (degree)", "Turbine operating feature"),
    "Pab2": ("Blade 2 pitch angle (degree)", "Turbine operating feature"),
    "Pab3": ("Blade 3 pitch angle (degree)", "Turbine operating feature"),
    "Prtv": ("Reactive power (kW)", "Turbine operating feature"),
    "T2m": ("ERA5 temperature at 2 m (degree C)", "Meteorological feature"),
    "Sp": ("ERA5 surface pressure (Pa)", "Meteorological feature"),
    "RelH": ("ERA5-derived relative humidity", "Meteorological feature"),
    "Wspd_w": ("ERA5 wind speed at 10 m (m/s)", "Meteorological feature"),
    "Wdir_w": ("ERA5 wind direction at 10 m (degree)", "Directional feature"),
    "Tp": ("ERA5 total precipitation (m)", "Meteorological feature"),
    "Patv": ("Active power produced at timestamp (kW)", "Target candidate"),
}

FEATURE_DECISIONS = {
    "Patv": ("Core", "Historical target carries persistence and operating-regime information", "Official missing/unknown and negative values require masking/sensitivity policy", "Direct physical target", "Only lagged values available before forecast origin; contemporaneous/future Patv leaks", "Same unit and turbine type across clients"),
    "Wspd": ("Core", "Primary physical driver of power and basis of empirical power curve", "Audit missing and extremes; official source does not define a numeric abnormal range", "Directly interpretable m/s", "Use observations no later than forecast origin", "Same SCADA concept across clients, but sensors may differ"),
    "Wdir": ("Optional", "Relative inflow direction may affect wake and power response", "Official abnormal range is outside [-180,180]", "Circular variable; encode sin/cos if retained", "Historical values only", "Common definition; angle handling must be identical"),
    "Etmp": ("Optional", "Air temperature may affect density and equipment state", "Audit missing/extreme sensor values", "Physical environmental temperature", "Historical values only", "Same SCADA concept across clients"),
    "Itmp": ("Optional", "Internal temperature may indicate turbine operating condition", "May contain sensor/maintenance artifacts", "Operational-state proxy", "Historical values only", "Same field but client calibration may vary"),
    "Ndir": ("Optional", "Yaw orientation may help interpret relative wind and wake exposure", "Official abnormal range is outside [-720,720]", "Circular/yaw state; wrapping needs explicit decision", "Historical values only", "Same definition across clients"),
    "Pab1": ("Optional", "Blade pitch is strongly related to operating state", "Values >89 degrees define official unknown-power state", "Physical control-state variable", "Historical values only; target-time pitch would reveal future state", "Same turbine type; three channels may be redundant"),
    "Pab2": ("Optional", "Same rationale as Pab1", "Official paper states three pitch values are generally identical", "Physical control-state variable", "Historical values only", "Likely redundant with Pab1"),
    "Pab3": ("Optional", "Same rationale as Pab1", "Official paper states three pitch values are generally identical", "Physical control-state variable", "Historical values only", "Likely redundant with Pab1"),
    "Prtv": ("Optional", "Reactive power can indicate electrical/operating state", "May be slightly negative due to component consumption", "Interpretable electrical state", "Historical values only", "Same unit across clients"),
    "T2m": ("Excluded initially", "Weather covariate may add information beyond nacelle sensors", "ERA5 quality is spatially smoother than SCADA", "Physical reanalysis field", "Future reanalysis at target time is not an operational forecast and can leak unavailable information", "Common grid product; client mapping must be verified"),
    "Sp": ("Excluded initially", "Pressure may proxy air-density variation", "ERA5 field; low client variation is possible", "Physical pressure", "Future reanalysis availability problem", "Likely shared/coarse across clients"),
    "RelH": ("Excluded initially", "Humidity may affect air conditions", "Derived ERA5 field", "Physical but secondary", "Future reanalysis availability problem", "Likely shared/coarse across clients"),
    "Wspd_w": ("Excluded initially", "Large-scale wind can complement turbine-top Wspd", "ERA5 10 m wind differs from hub-height SCADA", "Physical reanalysis wind", "Future reanalysis is not a true forecast", "Mapping from grid to clients must be verified"),
    "Wdir_w": ("Excluded initially", "Large-scale wind direction may contextualize wakes", "ERA5 circular field", "Physical direction", "Future reanalysis is not a true forecast", "Mapping from grid to clients must be verified"),
    "Tp": ("Excluded initially", "Precipitation may indicate weather regime", "Sparse/skewed ERA5 field", "Physical precipitation", "Future reanalysis is not a true forecast", "Likely shared/coarse across clients"),
    "TurbID": ("Excluded initially", "Identifier is needed for client partitioning, not a physical predictor", "Complete identifier expected", "Administrative ID", "No temporal leakage but can enable memorization", "Values are client-specific by definition"),
    "Day": ("Excluded initially", "Source chronology component", "Parsing must be validated", "Administrative time component", "Raw index/date is replaced by causal calendar encodings", "Shared calendar"),
    "Tmstamp": ("Excluded initially", "Source chronology component", "Parsing must be validated", "Administrative time component", "Derive hour/day cyclic features using forecast-origin time only", "Shared clock; official timezone UTC+08:00"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/eda.json"))
    p.add_argument("--input", type=Path, help="Override input measurement file")
    p.add_argument("--location", type=Path, help="Override turbine location file")
    p.add_argument("--output", type=Path, help="Override output directory")
    return p.parse_args()


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise RuntimeError("Parquet input requires pyarrow; install requirements-eda.txt") from e
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input format: {suffix}")


def infer_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    ts_col = infer_column(df, ["Datetime", "Timestamp", "DateTime", "date_time"])
    day_col = infer_column(df, ["Day", "Date"])
    time_col = infer_column(df, ["Tmstamp", "Time", "time"])
    if ts_col:
        return pd.to_datetime(df[ts_col], errors="coerce")
    if day_col and time_col:
        day_raw = df[day_col].astype(str).str.strip()
        time_raw = df[time_col].astype(str).str.strip()
        # Full release commonly provides a date plus a time; KDD provides day index plus time.
        if day_raw.str.contains(r"[-/]", regex=True, na=False).mean() > 0.5:
            return pd.to_datetime(day_raw + " " + time_raw, errors="coerce")
        day_num = pd.to_numeric(df[day_col], errors="coerce")
        base = pd.Timestamp("1970-01-01")
        return base + pd.to_timedelta(day_num - day_num.min(), unit="D") + pd.to_timedelta(time_raw, errors="coerce")
    if time_col:
        return pd.to_datetime(df[time_col], errors="coerce")
    raise ValueError("No official time columns found (expected Datetime/Timestamp or Day+Tmstamp)")


def official_flags(df: pd.DataFrame) -> pd.DataFrame:
    def num(name: str) -> pd.Series:
        return pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series(np.nan, index=df.index)

    patv, wspd, wdir, ndir = num("Patv"), num("Wspd"), num("Wdir"), num("Ndir")
    pitches = pd.concat([num("Pab1"), num("Pab2"), num("Pab3")], axis=1)
    missing_any = df.isna().any(axis=1)
    unknown_stopped = (patv <= 0) & (wspd > 2.5)
    unknown_pitch = pitches.gt(89).any(axis=1)
    abnormal_ndir = (ndir < -720) | (ndir > 720)
    abnormal_wdir = (wdir < -180) | (wdir > 180)
    return pd.DataFrame(
        {
            "official_missing_any": missing_any,
            "official_unknown_stopped": unknown_stopped.fillna(False),
            "official_unknown_pitch": unknown_pitch.fillna(False),
            "official_abnormal_ndir": abnormal_ndir.fillna(False),
            "official_abnormal_wdir": abnormal_wdir.fillna(False),
            "official_abnormal_any": (abnormal_ndir | abnormal_wdir).fillna(False),
            "official_unknown_any": (unknown_stopped | unknown_pitch).fillna(False),
        },
        index=df.index,
    )


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0, 0
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = changes[1::2] - changes[::2]
    return int(lengths.max()), int(len(lengths))


def robust_quantile_distance(values_by_client: list[np.ndarray], n_q: int) -> np.ndarray:
    q = np.linspace(0, 1, n_q)
    quantiles = []
    for values in values_by_client:
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        quantiles.append(np.quantile(x, q) if x.size else np.full(n_q, np.nan))
    quantiles = np.asarray(quantiles)
    n = len(quantiles)
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            valid = np.isfinite(quantiles[i]) & np.isfinite(quantiles[j])
            d = float(np.mean(np.abs(quantiles[i, valid] - quantiles[j, valid]))) if valid.any() else np.nan
            out[i, j] = out[j, i] = d
    return out


def conditional_curve_distance(curves: np.ndarray) -> np.ndarray:
    n = curves.shape[0]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            valid = np.isfinite(curves[i]) & np.isfinite(curves[j])
            d = float(np.sqrt(np.mean((curves[i, valid] - curves[j, valid]) ** 2))) if valid.any() else np.nan
            out[i, j] = out[j, i] = d
    return out


def internal_month_boundary(expected: pd.DatetimeIndex, target: pd.Timestamp, lower: pd.Timestamp, upper: pd.Timestamp) -> pd.Timestamp:
    """Return nearest valid month-end timestamp, or the target when none exists."""
    if len(expected) < 2:
        return target
    months = expected.to_period("M")
    change = np.flatnonzero(months[1:] != months[:-1])
    candidates = expected[change]
    candidates = candidates[(candidates > lower) & (candidates < upper)]
    if len(candidates) == 0:
        return target
    return candidates[int(np.argmin(np.abs(candidates.asi8 - target.value)))]


def save_matrix(matrix: np.ndarray, clients: list, path: Path) -> None:
    pd.DataFrame(matrix, index=clients, columns=clients).to_csv(path, encoding="utf-8-sig")


def plot_heatmap(matrix: np.ndarray, title: str, path: Path, unit: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set(title=title, xlabel="Client index", ylabel="Client index")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(unit)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_rules(path: Path) -> None:
    text = f"""# SDWPF data-quality rules

This document separates **official dataset rules** from **project-level descriptive heuristics**. Stage 1 only counts flags; it does not delete, interpolate, replace, or relabel raw observations.

Official source: [{OFFICIAL_SOURCE}]({OFFICIAL_URL}). The authors state that the released data were not corrected and that their processing suggestions are not mandatory preprocessing decisions.

## Official rule O1 — negative active power

- Condition: `Patv < 0`.
- Meaning: small negative active/reactive readings may reflect control-system or sensor consumption while the turbine is not generating.
- Official source: Usage Notes, “Zero values”.
- Possible modeling treatment: consider mapping negative `Patv` to zero, but only after task definition and sensitivity analysis.
- Risk if untreated: target tails and zero-mass estimates may be distorted.

## Official rule O2 — missing values

- Condition: a sensor or target field is missing.
- Meaning: observations were not collected, potentially because of maintenance or equipment failure.
- Official source: Usage Notes, “Missing values”.
- Possible modeling treatment: mask targets; decide feature-window eligibility and any causal imputation after chronological splitting.
- Risk if untreated: missingness may create invalid windows and client-specific pseudo-heterogeneity.

## Official rule O3 — unknown power under stoppage-like state

- Condition: `Patv <= 0` and `Wspd > 2.5 m/s`.
- Meaning: official document treats actual active power as unknown because the turbine is stopped although wind is sufficient for generation.
- Official source: Usage Notes, “Unknown values”.
- Possible modeling treatment: exclude from target scoring or treat through an explicit operational-state task; do not relabel as ordinary zero generation.
- Risk if untreated: a forecasting model is asked to predict externally scheduled/maintenance stoppage as meteorological generation.

## Official rule O4 — unknown power under feathered blade state

- Condition: `Pab1 > 89°` or `Pab2 > 89°` or `Pab3 > 89°`.
- Meaning: the turbine is at rest; active power is treated as unknown.
- Official source: Usage Notes, “Unknown values”.
- Possible modeling treatment: mask target or model operating state separately after task definition.
- Risk if untreated: conditional power curves mix normal generation with stopped operation.

## Official rule O5 — abnormal nacelle direction

- Condition: `Ndir < -720°` or `Ndir > 720°`.
- Meaning: outside the turbine system’s documented reasonable range; recording-system outlier.
- Official source: Usage Notes, “Abnormal values”.
- Possible modeling treatment: flag the entire record for later exclusion or field-specific repair, decided after audit.
- Risk if untreated: directional features and client distance estimates may be corrupted.

## Official rule O6 — abnormal relative wind direction

- Condition: `Wdir < -180°` or `Wdir > 180°`.
- Meaning: outside the documented reasonable range; recording-system outlier.
- Official source: Usage Notes, “Abnormal values”.
- Possible modeling treatment: flag the record; do not silently wrap angles until semantics and raw-error mechanism are confirmed.
- Risk if untreated: input heterogeneity may be dominated by sensor errors.

## Project descriptive flags — not official labels

- `low_or_zero_power`: `Patv <= 0`; descriptive only, overlaps official unknown-power cases.
- `possible_external_stop_or_curtailment_proxy`: identical to official rule O3 for counting purposes. The official paper states that unknown power can arise from renovation and/or active scheduling, but the dataset does not provide a confirmed curtailment label. It must therefore not be called measured curtailment.
- `system_level_missing_timestamp`: at least the configured fraction (default 80%) of clients are absent at a common 10-minute timestamp. This distinguishes synchronous collection gaps from single-client gaps but does not prove their cause.

These project flags must not be cited as official SDWPF definitions or treated as confirmed physical states without further evidence.
"""
    path.write_text(text, encoding="utf-8")


def choose_representatives(quality: pd.DataFrame) -> tuple[list, dict]:
    candidates = {}
    complete = quality.sort_values(["coverage_completeness", "missing_rate"], ascending=[False, True]).iloc[0]
    candidates[complete.name] = "highest coverage / low missingness"
    if "patv_mean" in quality:
        candidates[quality["patv_mean"].idxmax()] = "highest mean Patv"
        candidates[quality["patv_mean"].idxmin()] = "lowest mean Patv"
    candidates[quality["official_flag_rate"].idxmax()] = "highest official flag rate"
    if "wspd_mean" in quality:
        candidates[quality["wspd_mean"].idxmax()] = "distinct high mean Wspd"
    return list(candidates.keys())[:5], candidates


def main() -> int:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    input_path = (args.input or Path(cfg["input_path"])).resolve()
    location_path = (args.location or Path(cfg["location_path"])).resolve()
    out = (args.output or Path(cfg["output_dir"])).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Official SDWPF file not found: {input_path}\n"
            "Place an unmodified official file in data/raw/ or pass --input. No results were fabricated."
        )

    df = read_table(input_path)
    df.columns = [str(c).strip() for c in df.columns]
    client_col = infer_column(df, ["TurbID", "TurbineID", "turbine_id"])
    if not client_col:
        raise ValueError("No turbine ID column found")
    if "Patv" not in df or "Wspd" not in df:
        raise ValueError("Required official fields Patv and Wspd are absent")
    df["_timestamp"] = build_timestamp(df)
    flags = official_flags(df.drop(columns=["_timestamp"]))
    df = pd.concat([df, flags], axis=1)
    try:
        df[client_col] = pd.to_numeric(df[client_col])
    except (TypeError, ValueError):
        pass
    clients = sorted(df[client_col].dropna().unique().tolist())
    resolution = pd.Timedelta(minutes=cfg["time_resolution_minutes"])
    valid_ts = df["_timestamp"].dropna()
    global_start, global_end = valid_ts.min(), valid_ts.max()
    nominal_grid = pd.date_range(global_start, global_end, freq=resolution)
    actual_timeline = pd.DatetimeIndex(sorted(valid_ts.unique()))
    client_groupby = df.groupby(client_col, sort=True, observed=True)

    # Audit the actual cadence before constructing any missing-timestamp grid.
    timeline_deltas = pd.Series(actual_timeline[1:] - actual_timeline[:-1])
    delta_counts = timeline_deltas.value_counts().sort_index()
    regular_cadences = delta_counts[delta_counts >= max(10, int(0.01 * len(timeline_deltas)))].index.tolist()
    if not regular_cadences:
        regular_cadences = [timeline_deltas.mode().iloc[0]]
    max_regular_cadence = max(regular_cadences)
    system_gap_rows = []
    system_gap_blocks = []
    for idx, delta in enumerate(timeline_deltas, start=1):
        if delta <= max_regular_cadence:
            continue
        prior = timeline_deltas.iloc[: idx - 1]
        prior_regular = prior[prior.isin(regular_cadences)]
        cadence = prior_regular.iloc[-1] if len(prior_regular) else min(regular_cadences)
        inferred = pd.date_range(actual_timeline[idx - 1] + cadence, actual_timeline[idx] - cadence, freq=cadence)
        system_gap_blocks.append(
            {
                "gap_start": inferred.min() if len(inferred) else pd.NaT,
                "gap_end": inferred.max() if len(inferred) else pd.NaT,
                "previous_observed": actual_timeline[idx - 1],
                "next_observed": actual_timeline[idx],
                "preceding_cadence_minutes": cadence.total_seconds() / 60,
                "missing_timestamps_inferred": len(inferred),
                "gap_duration_hours_between_observations": delta.total_seconds() / 3600,
                "affected_client_fraction": 1.0,
                "interpretation": "System-wide gap inferred from the preceding dominant cadence; no interpolation performed",
            }
        )
        system_gap_rows.extend(
            {"timestamp": ts, "missing_client_fraction": 1.0, "system_level_gap_flag": True, "inference_cadence_minutes": cadence.total_seconds() / 60}
            for ts in inferred
        )
    system_gap_table = pd.DataFrame(system_gap_blocks)
    system_gap_table.to_csv(out / "table_system_gap_blocks.csv", index=False, encoding="utf-8-sig")
    cadence_rows = [
        {"interval_minutes": td.total_seconds() / 60, "transition_count": int(count), "is_dominant_regular_cadence": td in regular_cadences}
        for td, count in delta_counts.items()
    ]
    pd.DataFrame(cadence_rows).to_csv(out / "table_time_cadence_audit.csv", index=False, encoding="utf-8-sig")
    total_system_missing = int(sum(r["missing_timestamps_inferred"] for r in system_gap_blocks))
    longest_system_gap_hours = float(max((r["gap_duration_hours_between_observations"] for r in system_gap_blocks), default=0.0))

    # Data asset overview: one dataset row plus one row per client.
    duplicated_timestamp = int(df["_timestamp"].duplicated(keep=False).sum())
    duplicated_pair = int(df.duplicated([client_col, "_timestamp"], keep=False).sum())
    overview_rows = [
        {
            "scope": "dataset",
            "client_id": "ALL",
            "file_name": input_path.name,
            "file_format": input_path.suffix.lower().lstrip("."),
            "sha256": sha256(input_path),
            "total_rows": len(df),
            "total_fields_raw": len([c for c in df.columns if not c.startswith("official_") and c != "_timestamp"]),
            "field_names": "|".join([c for c in df.columns if not c.startswith("official_") and c != "_timestamp"]),
            "field_dtypes": "|".join(f"{c}:{df[c].dtype}" for c in df.columns if not c.startswith("official_") and c != "_timestamp"),
            "time_start": global_start,
            "time_end": global_end,
            "resolution_minutes": cfg["time_resolution_minutes"],
            "observed_regular_cadences_minutes": "|".join(str(int(td.total_seconds() / 60)) for td in regular_cadences),
            "turbine_count": len(clients),
            "record_count": len(df),
            "duplicate_timestamp_rows": duplicated_timestamp,
            "has_duplicate_timestamps_across_rows": bool(duplicated_timestamp),
            "duplicate_turbine_time_rows": duplicated_pair,
            "has_duplicate_turbine_time_records": bool(duplicated_pair),
        }
    ]
    for client, g in client_groupby:
        overview_rows.append(
            {
                "scope": "client",
                "client_id": client,
                "file_name": input_path.name,
                "file_format": input_path.suffix.lower().lstrip("."),
                "sha256": "",
                "total_rows": len(df),
                "total_fields_raw": overview_rows[0]["total_fields_raw"],
                "field_names": "",
                "field_dtypes": "",
                "time_start": g["_timestamp"].min(),
                "time_end": g["_timestamp"].max(),
                "resolution_minutes": cfg["time_resolution_minutes"],
                "observed_regular_cadences_minutes": "|".join(str(int(td.total_seconds() / 60)) for td in regular_cadences),
                "turbine_count": len(clients),
                "record_count": len(g),
                "duplicate_timestamp_rows": int(g["_timestamp"].duplicated(keep=False).sum()),
                "has_duplicate_timestamps_across_rows": bool(g["_timestamp"].duplicated(keep=False).any()),
                "duplicate_turbine_time_rows": int(g.duplicated([client_col, "_timestamp"], keep=False).sum()),
                "has_duplicate_turbine_time_records": bool(g.duplicated([client_col, "_timestamp"], keep=False).any()),
            }
        )
    pd.DataFrame(overview_rows).to_csv(out / "table_data_overview.csv", index=False, encoding="utf-8-sig")

    # Variable audit, excluding derived audit flags.
    audit_rows = []
    raw_cols = [c for c in df.columns if not c.startswith("official_") and c != "_timestamp"]
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    for col in raw_cols:
        s = df[col]
        num_s = pd.to_numeric(s, errors="coerce")
        numeric_fraction = num_s.notna().mean()
        stats = num_s.describe(percentiles=quantiles) if numeric_fraction > 0.5 else pd.Series(dtype=float)
        description, category = FIELD_METADATA.get(col, ("Not defined in the verified official field table", "Not recommended for initial model"))
        row = {
            "field": col,
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "min": stats.get("min", np.nan),
            "max": stats.get("max", np.nan),
            "mean": stats.get("mean", np.nan),
            "std": stats.get("std", np.nan),
            "unique_count": int(s.nunique(dropna=True)),
            "official_definition": description,
            "candidate_category": category,
            "definition_source": OFFICIAL_URL if col in FIELD_METADATA else "unverified",
        }
        for q in quantiles:
            row[f"q{int(q * 100):02d}"] = stats.get(f"{int(q * 100)}%", np.nan)
        audit_rows.append(row)
    pd.DataFrame(audit_rows).to_csv(out / "table_variable_audit.csv", index=False, encoding="utf-8-sig")
    feature_rows = []
    for col in raw_cols:
        decision = FEATURE_DECISIONS.get(col, ("Excluded initially", "No verified task rationale", "Unknown", "Unverified field", "Availability not established", "Cross-client semantics unverified"))
        feature_rows.append(
            {
                "field": col,
                "candidate_group": decision[0],
                "target_relationship": decision[1],
                "data_quality": decision[2],
                "interpretability": decision[3],
                "future_leakage_risk": decision[4],
                "cross_client_consistency": decision[5],
                "status": "Candidate only; not frozen",
            }
        )
    pd.DataFrame(feature_rows).to_csv(out / "table_feature_candidates.csv", index=False, encoding="utf-8-sig")

    # Client availability is evaluated against the union of timestamps actually
    # present in the file. System-wide gaps are reported separately above.
    present = pd.crosstab(df["_timestamp"], df[client_col]).reindex(index=actual_timeline, columns=clients, fill_value=0) > 0
    pd.DataFrame(system_gap_rows, columns=["timestamp", "missing_client_fraction", "system_level_gap_flag", "inference_cadence_minutes"]).to_csv(
        out / "table_synchronous_missingness.csv", index=False, encoding="utf-8-sig"
    )

    temporal_rows, quality_rows = [], []
    rated = float(cfg["rated_power_kw"])
    for client in clients:
        g = client_groupby.get_group(client).sort_values("_timestamp")
        observed_unique = g["_timestamp"].dropna().drop_duplicates()
        client_missing_mask = ~actual_timeline.isin(observed_unique)
        longest_gap, blocks = longest_true_run(client_missing_mask)
        p = pd.to_numeric(g["Patv"], errors="coerce")
        w = pd.to_numeric(g["Wspd"], errors="coerce")
        raw_fields = g[raw_cols]
        official_flag = g["official_abnormal_any"] | g["official_unknown_any"]
        quality_rows.append(
            {
                "client_id": client,
                "sample_count": len(g),
                "missing_rate": float(raw_fields.isna().sum().sum() / raw_fields.size),
                "patv_missing_rate": float(p.isna().mean()),
                "wspd_missing_rate": float(w.isna().mean()),
                "official_abnormal_rate": float(g["official_abnormal_any"].mean()),
                "official_unknown_power_rate": float(g["official_unknown_any"].mean()),
                "official_flag_rate": float(official_flag.mean()),
                "longest_missing_block_intervals": longest_gap,
                "longest_missing_block_hours": longest_gap * min(td.total_seconds() / 60 for td in regular_cadences) / 60,
                "low_or_zero_power_rate": float((p <= 0).mean()),
                "possible_external_stop_or_curtailment_proxy_rate": float(g["official_unknown_stopped"].mean()),
                "proxy_note": "Official unknown-power condition; may include renovation, stoppage or active scheduling; not a confirmed curtailment label",
                "zero_power_rate": float((p == 0).mean()),
                "normal_wind_abnormal_power_rate_official": float(((w > 2.5) & (p <= 0)).mean()),
                "coverage_completeness": float(len(observed_unique) / len(actual_timeline)),
                "system_timeline_coverage_cadence_aware": float(len(actual_timeline) / (len(actual_timeline) + total_system_missing)),
                "patv_mean": float(p.mean()),
                "patv_variance": float(p.var()),
                "patv_min": float(p.min()),
                "patv_max": float(p.max()),
                "negative_power_rate": float((p < 0).mean()),
                "patv_extreme_high_rate_gt_rated": float((p > rated).mean()),
                "wspd_mean": float(w.mean()),
                "wspd_std": float(w.std()),
                "wspd_min": float(w.min()),
                "wspd_max": float(w.max()),
                "wspd_low_rate_lt4": float((w < 4).mean()),
                "wspd_mid_rate_4to12": float(((w >= 4) & (w < 12)).mean()),
                "wspd_high_rate_ge12": float((w >= 12).mean()),
            }
        )
        temporal_rows.append(
            {
                "client_id": client,
                "expected_timestamps_cadence_aware": len(actual_timeline) + total_system_missing,
                "observed_unique_timestamps": len(observed_unique),
                "system_missing_timestamps_inferred": total_system_missing,
                "client_specific_missing_timestamps": int(client_missing_mask.sum()),
                "longest_client_specific_missing_block_intervals": longest_gap,
                "longest_client_specific_missing_block_hours": longest_gap * min(td.total_seconds() / 60 for td in regular_cadences) / 60,
                "number_of_client_specific_missing_blocks": blocks,
                "duplicate_timestamps": int(g["_timestamp"].duplicated(keep=False).sum()),
                "nominal_10min_expected_timestamps": len(nominal_grid),
                "nominal_10min_grid_missing": int((~nominal_grid.isin(observed_unique)).sum()),
                "timestamps_off_nominal_10min_grid": int((~observed_unique.isin(nominal_grid)).sum()),
                "dominant_regular_cadences_minutes": "|".join(str(int(td.total_seconds() / 60)) for td in regular_cadences),
            }
        )
    quality = pd.DataFrame(quality_rows).set_index("client_id")
    temporal = pd.DataFrame(temporal_rows).set_index("client_id")
    quality.to_csv(out / "table_client_quality.csv", encoding="utf-8-sig")
    temporal.to_csv(out / "table_temporal_integrity.csv", encoding="utf-8-sig")

    # Distribution and conditional heterogeneity matrices.
    pvals, wvals, pvals_valid, wvals_valid = [], [], [], []
    for c in clients:
        cg = client_groupby.get_group(c)
        p_all = pd.to_numeric(cg["Patv"], errors="coerce")
        w_all = pd.to_numeric(cg["Wspd"], errors="coerce")
        official_valid = ~(cg["official_abnormal_any"] | cg["official_unknown_any"] | cg["official_missing_any"])
        pvals.append(p_all.to_numpy())
        wvals.append(w_all.to_numpy())
        pvals_valid.append(p_all[official_valid].to_numpy())
        wvals_valid.append(w_all[official_valid].to_numpy())
    p_dist = robust_quantile_distance(pvals, cfg["distribution_quantiles"])
    w_dist = robust_quantile_distance(wvals, cfg["distribution_quantiles"])
    p_dist_valid = robust_quantile_distance(pvals_valid, cfg["distribution_quantiles"])
    w_dist_valid = robust_quantile_distance(wvals_valid, cfg["distribution_quantiles"])
    save_matrix(p_dist, clients, out / "matrix_power_distribution_distance.csv")
    save_matrix(w_dist, clients, out / "matrix_windspeed_distribution_distance.csv")
    save_matrix(p_dist_valid, clients, out / "matrix_power_distribution_distance_official_valid.csv")
    save_matrix(w_dist_valid, clients, out / "matrix_windspeed_distribution_distance_official_valid.csv")

    wind_bins = np.asarray(cfg["wind_bins_ms"], dtype=float)
    bin_centers = (wind_bins[:-1] + wind_bins[1:]) / 2
    curves = np.full((len(clients), len(bin_centers)), np.nan)
    curves_valid = np.full((len(clients), len(bin_centers)), np.nan)
    curve_counts = np.zeros_like(curves)
    curve_rows = []
    for ci, client in enumerate(clients):
        g = client_groupby.get_group(client)
        w, p = pd.to_numeric(g["Wspd"], errors="coerce"), pd.to_numeric(g["Patv"], errors="coerce")
        b = pd.cut(w, wind_bins, labels=False, include_lowest=True)
        official_valid_curve = ~(g["official_abnormal_any"] | g["official_unknown_any"] | g["official_missing_any"])
        for bi in range(len(bin_centers)):
            vals = p[b == bi]
            vals_valid = p[(b == bi) & official_valid_curve]
            curve_counts[ci, bi] = vals.notna().sum()
            if curve_counts[ci, bi] >= cfg["minimum_curve_bin_count"]:
                curves[ci, bi] = vals.mean()
            if vals_valid.notna().sum() >= cfg["minimum_curve_bin_count"]:
                curves_valid[ci, bi] = vals_valid.mean()
            curve_rows.append(
                {"client_id": client, "wind_bin_left": wind_bins[bi], "wind_bin_right": wind_bins[bi + 1], "wind_bin_center": bin_centers[bi], "count": int(curve_counts[ci, bi]), "mean_patv": curves[ci, bi], "official_valid_count": int(vals_valid.notna().sum()), "official_valid_mean_patv": curves_valid[ci, bi]}
            )
    pd.DataFrame(curve_rows).to_csv(out / "table_empirical_power_curves.csv", index=False, encoding="utf-8-sig")
    cond_dist = conditional_curve_distance(curves)
    cond_dist_valid = conditional_curve_distance(curves_valid)
    save_matrix(cond_dist, clients, out / "matrix_conditional_powercurve_distance.csv")
    save_matrix(cond_dist_valid, clients, out / "matrix_conditional_powercurve_distance_official_valid.csv")

    # Spatial information, if an official matching file is present.
    spatial_note = "Location file unavailable; spatial outputs not computed."
    if location_path.exists():
        loc = read_table(location_path)
        loc.columns = [str(c).strip() for c in loc.columns]
        loc_client = infer_column(loc, ["TurbID", "TurbineID", "turbine_id"])
        xcol, ycol = infer_column(loc, ["x", "X"]), infer_column(loc, ["y", "Y"])
        if loc_client and xcol and ycol:
            loc = loc.drop_duplicates(loc_client).set_index(loc_client).reindex(clients)
            xy = loc[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").to_numpy()
            spatial = cdist(xy, xy)
            save_matrix(spatial, clients, out / "matrix_spatial_distance.csv")
            nn = np.where(np.eye(len(clients), dtype=bool), np.nan, spatial)
            loc["nearest_neighbor_distance_m"] = np.nanmin(nn, axis=1)
            loc.to_csv(out / "table_spatial_audit.csv", encoding="utf-8-sig")
            spatial_note = f"Spatial distance matrix computed from official {location_path.name}."
            fig, ax = plt.subplots(figsize=(7, 9), constrained_layout=True)
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="viridis", s=28)
            ax.set(title="SDWPF turbine layout", xlabel="Relative x (m)", ylabel="Relative y (m)", aspect="equal")
            fig.colorbar(sc, ax=ax, label="Client order")
            fig.savefig(out / "figure_turbine_layout.png", dpi=180)
            plt.close(fig)
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
            elevation_col = infer_column(loc.reset_index(), ["elevation", "Elevation", "elev", "Ele", "height"])
            if elevation_col:
                axes[0].hist(pd.to_numeric(loc[elevation_col], errors="coerce").dropna(), bins=20)
                axes[0].set(title="Elevation distribution", xlabel="Elevation (m)", ylabel="Turbines")
            else:
                axes[0].text(0.5, 0.5, "Elevation field unavailable", ha="center", va="center")
                axes[0].set_axis_off()
            axes[1].hist(loc["nearest_neighbor_distance_m"].dropna(), bins=20)
            axes[1].set(title="Nearest-neighbor distance", xlabel="Distance (m)", ylabel="Turbines")
            pair_values = spatial[np.triu_indices(len(spatial), k=1)]
            axes[2].hist(pair_values[np.isfinite(pair_values)], bins=25)
            axes[2].set(title="Pairwise spatial distance", xlabel="Distance (m)", ylabel="Pairs")
            fig.savefig(out / "figure_spatial_distributions.png", dpi=180)
            plt.close(fig)

    # Figures.
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    quality["sample_count"].plot.bar(ax=ax, width=0.8)
    ax.set(title="Client sample counts", xlabel="Client", ylabel="Records")
    ax.tick_params(axis="x", labelsize=5)
    fig.savefig(out / "figure_client_sample_count.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(quality["missing_rate"], bins=25, edgecolor="black", linewidth=0.4)
    ax.set(title="Client missing-rate distribution", xlabel="Missing cell rate", ylabel="Clients")
    fig.savefig(out / "figure_missing_rate_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(quality["official_flag_rate"], bins=25, edgecolor="black", linewidth=0.4)
    ax.set(title="Client official abnormal/unknown flag distribution", xlabel="Flagged record rate", ylabel="Clients")
    fig.savefig(out / "figure_client_abnormal_ratio.png", dpi=180)
    plt.close(fig)

    # Daily counts relative to the paper's nominal 10-minute resolution make
    # the 2020 15-minute and 2021 10-minute regimes visible without resampling.
    daily_counts = pd.crosstab(df["_timestamp"].dt.floor("D"), df[client_col]).reindex(columns=clients, fill_value=0)
    full_days = pd.date_range(global_start.floor("D"), global_end.floor("D"), freq="1D")
    avail_daily = (daily_counts.reindex(full_days, fill_value=0) / (24 * 60 / cfg["time_resolution_minutes"])).T
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    im = ax.imshow(avail_daily, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax.set(title="Turbine × day records relative to nominal 10-minute grid", xlabel="Day index", ylabel="Client index")
    fig.colorbar(im, ax=ax, label="Observed / 144 nominal daily records")
    fig.savefig(out / "figure_data_availability_heatmap.png", dpi=180)
    plt.close(fig)

    for col, label, fname in [("Patv", "Active power Patv (kW)", "figure_patv_distribution.png"), ("Wspd", "Wind speed Wspd (m/s)", "figure_wspd_distribution.png")]:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        lo, hi = values.quantile([0.001, 0.999])
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.hist(values.clip(lo, hi), bins=100)
        ax.set(title=f"Overall {col} distribution (0.1–99.9% clipped for display)", xlabel=label, ylabel="Records")
        fig.savefig(out / fname, dpi=180)
        plt.close(fig)

    representatives, reasons = choose_representatives(quality)
    pd.DataFrame({"client_id": representatives, "selection_reason": [reasons[c] for c in representatives]}).to_csv(out / "table_representative_clients.csv", index=False, encoding="utf-8-sig")
    for col, fname, xlabel in [("Patv", "figure_representative_patv_ecdf.png", "Patv (kW)"), ("Wspd", "figure_representative_wspd_ecdf.png", "Wspd (m/s)")]:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for client in representatives:
            x = np.sort(pd.to_numeric(client_groupby.get_group(client)[col], errors="coerce").dropna().to_numpy())
            if len(x):
                idx = np.linspace(0, len(x) - 1, min(len(x), 2000)).astype(int)
                ax.plot(x[idx], (idx + 1) / len(x), label=f"{client}: {reasons[client]}")
        ax.set(title=f"Representative-client {col} ECDF", xlabel=xlabel, ylabel="Empirical CDF")
        ax.legend(fontsize=7)
        fig.savefig(out / fname, dpi=180)
        plt.close(fig)

    for col, fname, xlabel in [("Patv", "figure_all_client_patv_ecdf.png", "Patv (kW)"), ("Wspd", "figure_all_client_wspd_ecdf.png", "Wspd (m/s)")]:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for client in clients:
            x_ecdf = np.sort(pd.to_numeric(client_groupby.get_group(client)[col], errors="coerce").dropna().to_numpy())
            if len(x_ecdf):
                ecdf_idx = np.linspace(0, len(x_ecdf) - 1, min(len(x_ecdf), 400)).astype(int)
                ax.plot(x_ecdf[ecdf_idx], (ecdf_idx + 1) / len(x_ecdf), color="tab:blue", alpha=0.12, linewidth=0.6)
        ax.set(title=f"All-client {col} ECDF overlay", xlabel=xlabel, ylabel="Empirical CDF")
        fig.savefig(out / fname, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for client in representatives:
        ci = clients.index(client)
        ax.plot(bin_centers, curves[ci], marker="o", ms=3, label=f"{client}: {reasons[client]}")
    ax.set(title="Representative empirical power curves", xlabel="Wspd bin center (m/s)", ylabel="Mean Patv (kW)")
    ax.legend(fontsize=7)
    fig.savefig(out / "figure_empirical_power_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(representatives), figsize=(4 * len(representatives), 4), constrained_layout=True, squeeze=False)
    for ax, client in zip(axes[0], representatives):
        rep = client_groupby.get_group(client)
        rep_x = pd.to_numeric(rep["Wspd"], errors="coerce").to_numpy()
        rep_y = pd.to_numeric(rep["Patv"], errors="coerce").to_numpy()
        rep_valid = np.isfinite(rep_x) & np.isfinite(rep_y)
        ax.hexbin(rep_x[rep_valid], rep_y[rep_valid], gridsize=45, bins="log", mincnt=1, cmap="viridis")
        ax.set(title=f"Client {client}\n{reasons[client]}", xlabel="Wspd (m/s)", ylabel="Patv (kW)")
    fig.savefig(out / "figure_representative_powercurve_hexbin.png", dpi=180)
    plt.close(fig)

    # Overall power-curve hexbin, bounded for readability but based on raw observations.
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = pd.to_numeric(df["Wspd"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["Patv"], errors="coerce").to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    plot_idx = np.flatnonzero(valid)
    if len(plot_idx) > 500_000:
        plot_idx = plot_idx[np.linspace(0, len(plot_idx) - 1, 500_000).astype(int)]
    ax.hexbin(x[plot_idx], y[plot_idx], gridsize=90, bins="log", mincnt=1, cmap="viridis")
    ax.set(title=f"Overall Wspd → Patv density (deterministic display sample n={len(plot_idx):,})", xlabel="Wspd (m/s)", ylabel="Patv (kW)")
    fig.savefig(out / "figure_powercurve_hexbin.png", dpi=180)
    plt.close(fig)

    plot_heatmap(p_dist, "Patv distribution distance (quantile Wasserstein approximation)", out / "figure_power_distribution_distance_heatmap.png", "kW")
    plot_heatmap(w_dist, "Wspd distribution distance (quantile Wasserstein approximation)", out / "figure_windspeed_distribution_distance_heatmap.png", "m/s")
    plot_heatmap(cond_dist, "Conditional empirical power-curve distance", out / "figure_conditional_powercurve_distance_heatmap.png", "kW RMSE")
    quality_impact = pd.DataFrame(
        {
            "matrix": ["Patv distribution", "Wspd distribution", "Conditional power curve"],
            "raw_offdiagonal_median": [np.nanmedian(p_dist[~np.eye(len(clients), dtype=bool)]), np.nanmedian(w_dist[~np.eye(len(clients), dtype=bool)]), np.nanmedian(cond_dist[~np.eye(len(clients), dtype=bool)])],
            "official_valid_offdiagonal_median": [np.nanmedian(p_dist_valid[~np.eye(len(clients), dtype=bool)]), np.nanmedian(w_dist_valid[~np.eye(len(clients), dtype=bool)]), np.nanmedian(cond_dist_valid[~np.eye(len(clients), dtype=bool)])],
        }
    )
    quality_impact["absolute_change_after_official_filter"] = quality_impact["official_valid_offdiagonal_median"] - quality_impact["raw_offdiagonal_median"]
    quality_impact["interpretation"] = "Descriptive sensitivity only; filtering is not yet a frozen modeling policy"
    quality_impact.to_csv(out / "table_non_iid_quality_sensitivity.csv", index=False, encoding="utf-8-sig")

    # Transparent recommendation rules; never removes a client.
    rec = cfg["recommendation"]
    recommendation_rows = []
    for client, q in quality.iterrows():
        severe_reasons = []
        if q["coverage_completeness"] < rec["caution_min_coverage"]:
            severe_reasons.append("client-specific coverage below caution threshold")
        if q["missing_rate"] > rec["caution_max_missing_rate"]:
            severe_reasons.append("high cell missing rate")
        if q["official_flag_rate"] > rec["caution_max_official_flag_rate"]:
            severe_reasons.append("high official abnormal/unknown rate")
        if q["longest_missing_block_hours"] > 24 * rec["not_recommended_longest_gap_days"]:
            severe_reasons.append("very long client-specific disconnection")
        if severe_reasons:
            category, reasons_out = "Not recommended", severe_reasons
        elif (
            q["coverage_completeness"] >= rec["recommended_min_coverage"]
            and q["missing_rate"] <= rec["recommended_max_missing_rate"]
            and q["patv_missing_rate"] <= rec["recommended_max_missing_rate"]
            and q["wspd_missing_rate"] <= rec["recommended_max_missing_rate"]
            and q["official_flag_rate"] <= rec["recommended_max_official_flag_rate"]
        ):
            category, reasons_out = "Recommended", ["meets configured client-specific coverage, target/input missingness, and official-flag thresholds"]
        else:
            caution = []
            if q["patv_missing_rate"] > rec["recommended_max_missing_rate"]:
                caution.append("Patv missing rate exceeds recommended threshold")
            if q["wspd_missing_rate"] > rec["recommended_max_missing_rate"]:
                caution.append("Wspd missing rate exceeds recommended threshold")
            if q["missing_rate"] > rec["recommended_max_missing_rate"]:
                caution.append("cell missing rate exceeds recommended threshold")
            if q["official_flag_rate"] > rec["recommended_max_official_flag_rate"]:
                caution.append("official abnormal/unknown rate exceeds recommended threshold")
            category, reasons_out = "Usable with caution", caution or ["at least one recommended threshold is not met"]
        recommendation_rows.append({"client_id": client, "recommendation": category, "reason": "; ".join(reasons_out), "threshold_source": "Project EDA configuration; not an official SDWPF rule"})
    recommendations = pd.DataFrame(recommendation_rows)
    recommendations.to_csv(out / "table_client_recommendation.csv", index=False, encoding="utf-8-sig")

    write_rules(out / "data_quality_rules.md")
    # Machine-readable run manifest makes exact reruns auditable.
    manifest = {
        "input_path": str(input_path),
        "input_sha256": overview_rows[0]["sha256"],
        "location_path": str(location_path),
        "location_sha256": sha256(location_path) if location_path.exists() else None,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "configuration": cfg,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "official_source": OFFICIAL_URL,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Candidate chronological boundaries are derived from the actual common coverage.
    n_time = len(actual_timeline)
    exact_train_end = actual_timeline[max(0, math.floor(n_time * 0.70) - 1)]
    exact_val_end = actual_timeline[max(0, math.floor(n_time * 0.80) - 1)]
    calendar_train_end = internal_month_boundary(actual_timeline, exact_train_end, global_start, exact_val_end - resolution)
    calendar_val_end = internal_month_boundary(actual_timeline, exact_val_end, calendar_train_end + resolution, global_end - resolution)
    split_rows = [
        {"scheme": "Fixed calendar boundaries (recommended candidate)", "train_start": global_start, "train_end": calendar_train_end, "validation_start": calendar_train_end + resolution, "validation_end": calendar_val_end, "test_start": calendar_val_end + resolution, "test_end": global_end, "advantages": "Same interpretable month boundaries for all clients", "limitations": "Ratios deviate from exact 70/10/20; final dates require task/window audit"},
        {"scheme": "Exact 70/10/20 timestamps", "train_start": global_start, "train_end": exact_train_end, "validation_start": exact_train_end + resolution, "validation_end": exact_val_end, "test_start": exact_val_end + resolution, "test_end": global_end, "advantages": "Exact common-time proportions", "limitations": "May cut months/seasons and boundary windows require purging"},
        {"scheme": "Rolling-origin evaluation", "train_start": global_start, "train_end": "Expanding window", "validation_start": "Multiple origins", "validation_end": "Fixed validation window", "test_start": "After each origin", "test_end": global_end, "advantages": "Tests temporal robustness and regime dependence", "limitations": "Higher computation; fold dates depend on forecast horizon and history window"},
    ]
    pd.DataFrame(split_rows).to_csv(out / "table_time_split_candidates.csv", index=False, encoding="utf-8-sig")

    rec_counts = recommendations["recommendation"].value_counts().to_dict()
    offdiag = ~np.eye(len(clients), dtype=bool)
    report = f"""# SDWPF Stage 1 EDA report

## 1. Data facts

- Audited file: `{input_path.name}` (SHA-256 `{overview_rows[0]['sha256']}`).
- Raw shape: {len(df):,} rows × {overview_rows[0]['total_fields_raw']} fields; clients: {len(clients)}.
- Parsed coverage: {global_start} to {global_end}; target resolution: {cfg['time_resolution_minutes']} minutes.
- Duplicate turbine-time rows: {duplicated_pair:,}.
- {spatial_note}
- Official rules and source are documented in `data_quality_rules.md`; raw observations were not corrected.

## 2. Analysis results

- Median client coverage completeness: {quality['coverage_completeness'].median():.4f}; range {quality['coverage_completeness'].min():.4f}–{quality['coverage_completeness'].max():.4f}.
- Median client cell-missing rate: {quality['missing_rate'].median():.4f}; maximum {quality['missing_rate'].max():.4f}.
- Median official abnormal/unknown flag rate: {quality['official_flag_rate'].median():.4f}; maximum {quality['official_flag_rate'].max():.4f}.
- Cadence-aware system-level missing timestamps inferred: {total_system_missing:,}.
- Pairwise Patv distribution distance median: {np.nanmedian(p_dist[offdiag]):.3f} kW.
- Pairwise Wspd distribution distance median: {np.nanmedian(w_dist[offdiag]):.3f} m/s.
- Pairwise conditional power-curve distance median: {np.nanmedian(cond_dist[offdiag]):.3f} kW RMSE.
- After excluding records carrying any official missing/unknown/abnormal flag for descriptive sensitivity only, the Patv distance median is {np.nanmedian(p_dist_valid[offdiag]):.3f} kW and Wspd distance median is {np.nanmedian(w_dist_valid[offdiag]):.3f} m/s. The delta from raw distances is recorded in `table_non_iid_quality_sensitivity.csv` and is not a frozen cleaning policy.
- Recommendation counts under transparent project thresholds: {rec_counts}. No client was removed.

## 3. Temporary judgments

- SDWPF is provisionally suitable for turbine-level multi-client mechanism studies if client filtering, operational-state handling, and common chronological boundaries are frozen before modeling.
- Input/target distribution differences are descriptive Non-IID evidence. Conditional power-curve distance is the more task-relevant Stage 1 evidence because it compares Patv response at matched Wspd bins; it is not yet transfer-value evidence.
- Official missing/unknown/abnormal rates vary by client and can manufacture pseudo-Non-IID. Any later claim must report results both before and after a task-justified quality policy.
- “Possible curtailment-like” is a heuristic screen only and must not be interpreted as confirmed curtailment.

## 4. Unresolved questions

- Freeze the forecasting horizon, history window, and whether the estimand is available power or realized power.
- Decide whether official unknown-power intervals are target-masked, modeled as an operational state, or included in a sensitivity branch.
- Validate whether ERA5 values are observation/reanalysis values aligned at the same timestamp or truly available forecasts for the intended deployment; future-timestamp reanalysis features may leak unavailable information.
- Freeze client recommendation thresholds through a Decision Log before baseline training.
- Confirm candidate time split using common calendar boundaries and calculate all normalization parameters from training data only.

## Candidate features (not frozen)

- Core: historical `Patv`, `Wspd`; causal calendar encodings derived from timestamps after defining horizon.
- Optional: `Wdir`, `Etmp`, `Itmp`, `Ndir`, `Pab1`–`Pab3`, `Prtv`; ERA5 `T2m`, `Sp`, `RelH`, `Wspd_w`, `Wdir_w`, `Tp` only after availability/leakage audit.
- Excluded initially: `TurbID` as a numeric predictor; raw future observed/reanalysis weather; audit flags as ordinary continuous features; duplicate blade-pitch channels without redundancy analysis.

## Candidate chronological splits

1. Fixed calendar boundary (recommended candidate): train {global_start}–{calendar_train_end}, validation {calendar_train_end + resolution}–{calendar_val_end}, test {calendar_val_end + resolution}–{global_end}. Same boundaries for all clients; final dates require window/horizon audit.
2. Exact 70/10/20 timestamp split: train ends {exact_train_end}, validation ends {exact_val_end}; maximizes sample proportions but boundaries may cut seasons/months.
3. Rolling-origin evaluation: strongest temporal robustness evidence but higher computation; use as later robustness analysis, not the first development split.

Normalization and any imputation parameters must be estimated on training data only; validation/test cannot participate.

## Stage 1 Decision Summary

1. SDWPF 是否适合继续？暂时适合，用于 turbine-level 多客户端机制验证；不能替代 multi-wind-farm 工程验证。
2. 是否推荐全部 134 台？不默认推荐全部；本次阈值分类为 {rec_counts}，但未删除任何客户端。
3. 最明显的数据质量问题是什么？客户端缺失、长缺口和官方 unknown-power 状态并存，且比例不均。
4. 当前最可信的 Non-IID 证据是什么？匹配 Wspd 分箱后的经验功率曲线仍存在差异；其距离中位数为 {np.nanmedian(cond_dist[offdiag]):.3f} kW RMSE。
5. 是否有“伪 Non-IID”？有此风险：客户端级缺失与官方异常/未知状态比例异质。
6. 建议保留哪些变量？核心候选为历史 Patv、Wspd 和因果时间编码；其余为待消融或待可用性审计候选。
7. 推荐哪种时间划分？所有客户端共享固定日历边界的约 70/10/20 chronological split，最终日期尚未冻结。
8. 建模前还必须解决什么？任务时距/窗口、unknown-power estimand、ERA5 可用性、客户端阈值和训练期归一化规则。
9. 是否需要修改 Charter？不需要修改核心问题；应在 Decision Log 中补充 SDWPF 版本、任务 estimand 与数据质量分支。
10. 下一阶段是否进入清洗与任务定义？可以，但只能在上述未决项形成显式决策后进入；不能直接训练联邦模型。
"""
    cadence_text = "、".join(f"{int(td.total_seconds() / 60)} 分钟" for td in regular_cadences)
    report = f"""# SDWPF Stage 1 EDA report

## 1. 数据事实

- 审计文件：`{input_path.name}`，SHA-256 为 `{overview_rows[0]['sha256']}`。
- 原始规模：{len(df):,} 行、{overview_rows[0]['total_fields_raw']} 个字段、{len(clients)} 台风机。
- 实际时间范围：{global_start} 至 {global_end}；无 turbine-time 重复记录。
- 文件实际存在 {cadence_text} 两种主要采样节奏，而非全程固定 {cfg['time_resolution_minutes']} 分钟。
- 基于实际分段节奏识别到 {len(system_gap_blocks)} 个全场共同缺段，共推断缺少 {total_system_missing:,} 个时间点；最长相邻观测间隔为 {longest_system_gap_hours:.2f} 小时。
- {spatial_note}
- 原始数据未被插值、删除、替换或重排；官方规则见 `data_quality_rules.md`。

## 2. 分析结果

- 客户端在文件实际时间戳集合上的覆盖完整度中位数为 {quality['coverage_completeness'].median():.4f}，范围 {quality['coverage_completeness'].min():.4f}–{quality['coverage_completeness'].max():.4f}。
- 客户端字段缺失率中位数为 {quality['missing_rate'].median():.4f}，最大为 {quality['missing_rate'].max():.4f}。
- 客户端官方 abnormal/unknown-power 标记率中位数为 {quality['official_flag_rate'].median():.4f}，范围 {quality['official_flag_rate'].min():.4f}–{quality['official_flag_rate'].max():.4f}。
- Patv、Wspd 边际分布距离中位数分别为 {np.nanmedian(p_dist[offdiag]):.3f} kW 和 {np.nanmedian(w_dist[offdiag]):.3f} m/s。
- 同风速分箱条件下的经验功率曲线距离中位数为 {np.nanmedian(cond_dist[offdiag]):.3f} kW RMSE；排除官方 missing/unknown/abnormal 标记记录后的敏感性结果为 {np.nanmedian(cond_dist_valid[offdiag]):.3f} kW RMSE。
- 客户端建议分类为 {rec_counts}。该分类不删除任何风机，也不把全场共同缺段重复归因于单台风机。

## 3. 暂时性判断

- SDWPF 可继续用于 turbine-level 多客户端机制验证，但 2020 年 15 分钟与 2021 年 10 分钟节奏切换必须在任务冻结前处理，不能直接跨节奏拼接固定步数窗口。
- 当前最强的任务相关 Non-IID 描述性证据，是相同 Wspd 分箱下仍存在客户端功率响应差异；这不等价于已经证明客户端间迁移价值不同。
- 数据质量会影响异质性估计：官方有效记录敏感性分支明显改变距离。因此后续必须预先冻结质量处理分支，并同时报告原始与质量控制结果。
- `Patv <= 0 且 Wspd > 2.5` 仅按官方定义记为 unknown power；可能涉及停机、检修或外部调度，不能称为已确认限功率。

## 4. 尚未解决的问题

- 冻结预测时距、历史窗口，以及预测 realized power 还是 available power。
- 决定 2020 年 15 分钟与 2021 年 10 分钟数据是分阶段使用、因果重采样，还是主实验仅采用同一节奏区间。
- 决定官方 unknown-power 记录的目标掩码及敏感性分析方案。
- 核验 ERA5 再分析字段在目标工程部署中的可获得性，避免使用目标时刻未来不可得信息。
- 在 Decision Log 中冻结客户端阈值、时间边界、训练期归一化和窗口跨缺口规则。

## 特征候选（未冻结）

- Core：仅使用预测起点之前的历史 `Patv`、`Wspd`，以及由预测起点构造的因果时间编码。
- Optional：`Wdir`、`Etmp`、`Itmp`、`Ndir`、`Pab1`–`Pab3`、`Prtv`，后续以消融决定。
- 初始排除：`TurbID` 数值编码、目标时刻未来 SCADA、未经可用性核验的未来 ERA5 再分析值、将质量标记当普通连续特征。

## 候选时间划分

1. 固定日历边界候选：train {global_start}–{calendar_train_end}，validation {calendar_train_end + resolution}–{calendar_val_end}，test {calendar_val_end + resolution}–{global_end}。最终需结合节奏切换与预测窗口重新冻结。
2. 按实际时间戳数量的 70/10/20：train 截止 {exact_train_end}，validation 截止 {exact_val_end}；比例精确，但可能切断月份和运行状态。
3. Rolling-origin：适合作为后续稳健性验证，计算成本更高。

所有归一化、插值或其他统计参数只能由训练集估计。

## Stage 1 Decision Summary

1. SDWPF 暂时适合继续用于 turbine-level 多客户端机制研究，但不能替代 multi-wind-farm 工程验证。
2. 不应默认删除或保留全部 134 台；当前分类为 {rec_counts}，无风机被实际删除。
3. 最明显问题是采样节奏由 15 分钟切换为 10 分钟、存在全场共同缺段、SCADA 字段缺失、unknown-power 比例高且跨客户端差异明显、若干传感器极值异常。
4. 当前最可信的 Non-IID 证据是同 Wspd 分箱的经验功率曲线差异；官方有效记录分支距离中位数为 {np.nanmedian(cond_dist_valid[offdiag]):.3f} kW RMSE。
5. 存在数据质量导致伪 Non-IID 的风险，因为质量控制显著改变边际与条件距离。
6. Core 候选为历史 Patv、历史 Wspd 和因果时间编码。
7. 推荐所有客户端共享固定日历边界，但具体边界必须结合统一采样节奏方案重新冻结。
8. 建模前必须解决采样节奏、窗口跨缺口、unknown-power estimand、ERA5 可用性和训练期统计规则。
9. 不修改 Research Project Charter；仅需用 Decision Log 记录新增的数据事实和处理决策。
10. 已具备进入“数据清洗与预测任务冻结”的条件，但尚不具备直接训练模型的条件。
"""
    (out / "EDA_report.md").write_text(report, encoding="utf-8")
    provenance_rows = []
    for artifact in sorted(p for p in out.iterdir() if p.is_file() and p.name not in {".gitkeep", "table_artifact_provenance.csv", "STAGE1_BLOCKED.md"}):
        provenance_rows.append(
            {
                "artifact": artifact.name,
                "artifact_sha256": sha256(artifact),
                "source_data": input_path.name,
                "source_data_sha256": overview_rows[0]["sha256"],
                "location_data": location_path.name if location_path.exists() else "",
                "location_data_sha256": manifest["location_sha256"] or "",
                "script": Path(__file__).name,
                "script_sha256": manifest["script_sha256"],
                "config": args.config.name,
                "config_sha256": manifest["config_sha256"],
            }
        )
    pd.DataFrame(provenance_rows).to_csv(out / "table_artifact_provenance.csv", index=False, encoding="utf-8-sig")
    print(f"EDA complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
