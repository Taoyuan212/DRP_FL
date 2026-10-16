#!/usr/bin/env python3
"""Independent acceptance checks for the SDWPF Stage 2 artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "stage2.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
RAW = ROOT / CFG["input_path"]
OUT = ROOT / CFG["output_dir"]
PROCESSED = ROOT / CFG["processed_dir"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_check(rows: list[dict], name: str, passed: bool, evidence: str) -> None:
    rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})


def main() -> int:
    files = sorted(path for path in PROCESSED.glob("client_[0-9][0-9][0-9].parquet"))
    window_files = sorted(PROCESSED.glob("client_[0-9][0-9][0-9]_windows.parquet"))
    checks: list[dict] = []
    add_check(checks, "exactly_134_client_partitions", len(files) == 134, f"found={len(files)}")
    add_check(checks, "exactly_134_materialized_window_partitions", len(window_files) == 134, f"found={len(window_files)}")

    cfg_splits = {key: tuple(map(pd.Timestamp, value)) for key, value in CFG["splits"].items()}
    all_frames, task_rows = [], []
    min_counts = {"train": np.inf, "validation": np.inf, "test": np.inf}
    temporal_ok = split_ok = no_overlap = source_order_ok = True
    target_min, target_max = np.inf, -np.inf
    for file in files:
        frame = pd.read_parquet(file)
        all_frames.append(frame[["client_id", "issue_time", "history_start", "target_time", "split", "is_valid", "invalid_reason"]])
        valid = frame[frame["is_valid"]].copy()
        temporal_ok &= bool((valid["history_start"] <= valid["issue_time"]).all() and (valid["issue_time"] < valid["target_time"]).all())
        source_order_ok &= bool((valid["target_time"] - valid["issue_time"] == pd.Timedelta(minutes=CFG["primary_horizon_minutes"])).all())
        for name, (start, end) in cfg_splits.items():
            subset = valid[valid["split"].eq(name)]
            min_counts[name] = min(min_counts[name], len(subset))
            split_ok &= bool(((subset["target_time"] >= start) & (subset["target_time"] < end) & (subset["history_start"] >= start)).all())
        sets = {name: set(valid.loc[valid["split"].eq(name), "target_time"]) for name in cfg_splits}
        no_overlap &= not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
        target_min = min(target_min, float(valid["target_patv_pu"].min()))
        target_max = max(target_max, float(valid["target_patv_pu"].max()))

        observed_core = frame[["patv_issue_raw_kw", "wspd_issue_ms"]].notna().all(axis=1).to_numpy()
        observed_core &= frame["wspd_issue_ms"].between(CFG["physical_sensor_flags"]["wspd_min_ms"], CFG["physical_sensor_flags"]["wspd_max_ms"]).to_numpy()
        lookback_steps = int(CFG["lookback_hours"] * 60 / CFG["frequency_minutes"])
        ready = pd.Series(observed_core.astype(np.int8)).rolling(lookback_steps, min_periods=lookback_steps).sum().eq(lookback_steps).to_numpy()
        assigned = frame["split"].notna().to_numpy()
        boundary_ok = np.ones(len(frame), dtype=bool)
        for name, (start, _) in cfg_splits.items():
            in_split = frame["split"].eq(name).fillna(False).to_numpy()
            boundary_ok[in_split] = (frame.loc[in_split, "history_start"] >= start).to_numpy()
        common = ready & assigned & boundary_ok & frame["target_patv_kw"].notna().to_numpy() & ~frame["target_abnormal"].to_numpy()
        task_a = common
        task_b = common & ~frame["target_official_unknown"].to_numpy()
        task_rows.append(
            {
                "client_id": int(frame["client_id"].iloc[0]),
                "task_a_operational_valid_samples": int(task_a.sum()),
                "task_b_normal_operation_valid_samples": int(task_b.sum()),
                "task_b_fraction_of_task_a": float(task_b.sum() / task_a.sum()),
                "official_unknown_targets_excluded": int((task_a & frame["target_official_unknown"].to_numpy()).sum()),
            }
        )

    add_check(checks, "all_inputs_precede_targets", temporal_ok and source_order_ok, f"primary_horizon={CFG['primary_horizon_minutes']}min")
    add_check(checks, "no_split_timestamp_overlap_per_client", no_overlap, "target timestamp sets are disjoint")
    add_check(checks, "windows_purged_at_split_boundaries", split_ok, "history_start and target_time stay inside target split")
    add_check(checks, "all_clients_have_each_split", all(value > 0 for value in min_counts.values()), json.dumps(min_counts))
    add_check(checks, "target_physical_range_audited", np.isfinite(target_min) and target_min >= 0 and target_max <= 1.1, f"per_unit_range=[{target_min:.6f},{target_max:.6f}]")

    scaler = json.loads((PROCESSED / "scaler.json").read_text(encoding="utf-8"))
    scaler_end = pd.Timestamp(scaler["fit_end_exclusive"])
    add_check(checks, "scaler_training_only", scaler_end == cfg_splits["train"][1], f"fit_end_exclusive={scaler_end}")

    audit = pd.read_csv(OUT / "table_manual_timestamp_audit_100.csv", parse_dates=["issue_time", "target_time"])
    audit["raw_history_count"] = 0
    audit["expected_history_count"] = int(CFG["lookback_hours"] * 60 / CFG["frequency_minutes"])
    audit["history_grid_complete"] = False
    audit["history_core_complete"] = False
    audit["target_rule_valid"] = False
    audit["target_matches_processed"] = False
    audit["materialized_window_matches_raw"] = False
    raw_columns = ["TurbID", "Tmstamp", "Patv", "Wspd", "Wdir", "Ndir", "Pab1", "Pab2", "Pab3"]
    for client_id, positions in audit.groupby("client_id").groups.items():
        raw = pd.read_parquet(RAW, columns=raw_columns, filters=[("TurbID", "=", int(client_id))]).set_index("Tmstamp").sort_index()
        processed = pd.read_parquet(PROCESSED / f"client_{int(client_id):03d}.parquet").set_index("issue_time")
        windows = pd.read_parquet(PROCESSED / f"client_{int(client_id):03d}_windows.parquet").set_index("issue_time")
        for position in positions:
            row = audit.loc[position]
            history_start = row["issue_time"] - pd.Timedelta(hours=CFG["lookback_hours"]) + pd.Timedelta(minutes=CFG["frequency_minutes"])
            expected = pd.date_range(history_start, row["issue_time"], freq=f"{CFG['frequency_minutes']}min")
            history = raw.reindex(expected)
            target = raw.loc[row["target_time"]]
            unknown = bool(((target["Patv"] <= 0) and (target["Wspd"] > 2.5)) or (target[["Pab1", "Pab2", "Pab3"]] > 89).any())
            abnormal = bool(abs(target["Wdir"]) > 180 or abs(target["Ndir"]) > 720 or not (CFG["physical_sensor_flags"]["wspd_min_ms"] <= target["Wspd"] <= CFG["physical_sensor_flags"]["wspd_max_ms"]))
            process_row = processed.loc[row["issue_time"]]
            clean_target = max(float(target["Patv"]), 0.0)
            audit.loc[position, "raw_history_count"] = history.index.isin(raw.index).sum()
            audit.loc[position, "history_grid_complete"] = bool(history.index.equals(expected))
            audit.loc[position, "history_core_complete"] = bool(history[["Patv", "Wspd"]].notna().all(axis=None) and history["Wspd"].between(CFG["physical_sensor_flags"]["wspd_min_ms"], CFG["physical_sensor_flags"]["wspd_max_ms"]).all())
            audit.loc[position, "target_rule_valid"] = bool(pd.notna(target["Patv"]) and not unknown and not abnormal)
            audit.loc[position, "target_matches_processed"] = bool(np.isclose(clean_target, process_row["target_patv_kw"]) and np.isclose(clean_target / CFG["normalization"]["rated_power_kw"], process_row["target_patv_pu"]))
            window_row = windows.loc[row["issue_time"]]
            raw_patv = history["Patv"].clip(lower=0).to_numpy(float)
            raw_wspd = history["Wspd"].to_numpy(float)
            audit.loc[position, "materialized_window_matches_raw"] = bool(np.allclose(np.asarray(window_row["input_patv_clean_kw"], dtype=float), raw_patv) and np.allclose(np.asarray(window_row["input_wspd_ms"], dtype=float), raw_wspd))
    audit_ok = bool(audit[["history_grid_complete", "history_core_complete", "target_rule_valid", "target_matches_processed", "materialized_window_matches_raw"]].all(axis=None) and (audit["raw_history_count"] == audit["expected_history_count"]).all())
    audit["manual_review_status"] = np.where(audit_ok, "verified_by_raw_backcheck", "failed_raw_backcheck")
    audit.to_csv(OUT / "table_manual_timestamp_audit_100.csv", index=False, encoding="utf-8-sig")
    add_check(checks, "random_100_raw_window_backcheck", audit_ok, f"verified={int((audit['manual_review_status'] == 'verified_by_raw_backcheck').sum())}/100")

    for filename in ["matrix_clean_patv_distance.csv", "matrix_clean_wspd_distance.csv", "matrix_clean_conditional_powercurve_distance.csv"]:
        matrix = pd.read_csv(OUT / filename, index_col=0).to_numpy(float)
        ok = matrix.shape == (134, 134) and np.isfinite(matrix).all() and np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0)
        add_check(checks, f"valid_{filename}", ok, f"shape={matrix.shape}")

    task_comparison = pd.DataFrame(task_rows)
    task_comparison.to_csv(OUT / "table_task_definition_comparison.csv", index=False, encoding="utf-8-sig")
    total_a = int(task_comparison["task_a_operational_valid_samples"].sum())
    total_b = int(task_comparison["task_b_normal_operation_valid_samples"].sum())
    manifest = pd.read_csv(OUT / "dataset_manifest.csv")
    add_check(checks, "task_b_count_matches_manifest", total_b == int(manifest["valid_sample_count"].sum()), f"task_b={total_b}, manifest={int(manifest['valid_sample_count'].sum())}")

    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(OUT / "table_stage2_validation_checks.csv", index=False, encoding="utf-8-sig")
    if not check_frame["status"].eq("PASS").all():
        print(check_frame.to_string(index=False))
        return 1

    validation_manifest = {
        "validation_script": str(Path(__file__).resolve()),
        "validation_script_sha256": digest(Path(__file__).resolve()),
        "raw_sha256": digest(RAW),
        "config_sha256": digest(CFG_PATH),
        "checks_passed": int(len(check_frame)),
        "task_a_valid_samples": total_a,
        "task_b_valid_samples": total_b,
    }
    (OUT / "validation_manifest.json").write_text(json.dumps(validation_manifest, indent=2) + "\n", encoding="utf-8")

    provenance_rows = []
    artifact_paths = [path for path in OUT.iterdir() if path.is_file() and path.name != "table_artifact_provenance.csv" and path.suffix.lower() != ".log"]
    artifact_paths.extend(path for path in PROCESSED.iterdir() if path.is_file())
    for artifact in sorted(artifact_paths):
        provenance_rows.append(
            {
                "artifact": str(artifact.relative_to(ROOT)),
                "artifact_sha256": digest(artifact),
                "raw_source": str(RAW),
                "raw_sha256": validation_manifest["raw_sha256"],
                "config": str(CFG_PATH),
                "config_sha256": validation_manifest["config_sha256"],
                "generator": "run_sdwpf_stage2.py or validate_sdwpf_stage2.py",
                "training_performed": False,
            }
        )
    pd.DataFrame(provenance_rows).to_csv(OUT / "table_artifact_provenance.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(validation_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
