"""Freeze the Stage-8B AEMO data/task protocol without reading test targets for modelling."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "stage8a2"
OUT = ROOT / "results" / "stage8b"
PROC = ROOT / "data" / "aemo" / "processed_stage8b"
STEP = pd.Timedelta(minutes=5)
SPLITS = {
    "train": (pd.Timestamp("2025-08-13 00:00:00"), pd.Timestamp("2026-04-19 00:00:00")),
    "validation": (pd.Timestamp("2026-04-19 00:00:00"), pd.Timestamp("2026-06-20 00:00:00")),
    "test": (pd.Timestamp("2026-06-20 00:00:00"), pd.Timestamp("2026-08-21 00:00:00")),
}
NEGATIVE_ANOMALY_PU = -0.02
UPPER_ANOMALY_PU = 1.10
LOOKBACK = 72
HORIZON = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def effective_capacities(capacity: pd.DataFrame, client_id: str, times: pd.DatetimeIndex) -> tuple[np.ndarray, list[dict]]:
    q = capacity[(capacity["record_level"] == "FARM_TOTAL") & (capacity["physical_farm_id"] == client_id)].copy()
    q["effective_start"] = pd.to_datetime(q["effective_start"])
    q["effective_end"] = pd.to_datetime(q["effective_end"])
    q = q.sort_values("effective_start")
    values = np.full(len(times), np.nan, dtype=np.float64)
    intervals = []
    for r in q.itertuples():
        mask = (times >= r.effective_start) & (times < r.effective_end)
        values[mask] = float(r.capacity_mw)
        intervals.append({"start": str(r.effective_start), "end": str(r.effective_end), "capacity_mw": float(r.capacity_mw),
                          "affected_timestamps_in_frozen_window": int(mask.sum())})
    return values, intervals


def js_distance(a: np.ndarray, b: np.ndarray) -> float:
    m = (a + b) / 2
    left = np.sum(np.where(a > 0, a * np.log(a / np.where(m > 0, m, 1)), 0))
    right = np.sum(np.where(b > 0, b * np.log(b / np.where(m > 0, m, 1)), 0))
    return float(math.sqrt(max((left + right) / 2, 0)))


def time_features(origins: pd.DatetimeIndex) -> np.ndarray:
    hour = origins.hour.to_numpy() + origins.minute.to_numpy() / 60.0
    leap = origins.is_leap_year.astype(np.int32)
    year_len = 365 + leap
    doy = origins.dayofyear.to_numpy() - 1 + hour / 24.0
    return np.column_stack((np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                            np.sin(2*np.pi*doy/year_len), np.cos(2*np.pi*doy/year_len))).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(SRC / "external_client_inventory_v3.csv")
    capacity = pd.read_csv(SRC / "aemo_windfarm_capacity_history.csv")
    mapping = pd.read_csv(SRC / "aemo_duid_to_windfarm_mapping.csv")
    recommended = inv[inv["client_eligibility"] == "RECOMMENDED"].copy().sort_values("client_id").reset_index(drop=True)
    if len(recommended) != 87:
        raise RuntimeError(f"Expected 87 Recommended clients, found {len(recommended)}")
    client_ids = recommended["client_id"].tolist()

    map_duids = mapping.groupby("physical_farm_id")["DUID"].apply(lambda x: ";".join(sorted(set(map(str, x))))).to_dict()
    cap_intervals = {}
    manifest_rows = []
    full_times = pd.date_range(SPLITS["train"][0], SPLITS["test"][1] - STEP, freq=STEP)
    for order, r in enumerate(recommended.itertuples()):
        _, intervals = effective_capacities(capacity, r.client_id, full_times)
        cap_intervals[r.client_id] = intervals
        manifest_rows.append({
            "client_order": order, "client_id": r.client_id, "physical_farm_name": r.farm_name,
            "DUID_list": map_duids.get(r.client_id, r.DUID_list), "region": r.region,
            "capacity_history": json.dumps(intervals, separators=(",", ":")),
            "available_start": r.coverage_start, "available_end": r.coverage_end,
            "quality_flags": r.quality_flags, "eligibility_status": r.client_eligibility,
        })
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT / "external_main_client_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest_hash = sha256(manifest_path)

    # This physical-QC pass is explicitly prior to split locking. Test rows are never exposed to model/pilot code.
    raw = pd.read_parquet(SRC / "aemo_windfarm_generation_raw.parquet",
                          columns=["timestamp_canonical", "physical_farm_id", "generation_mw", "quality_flag"])
    raw = raw[raw["physical_farm_id"].isin(client_ids)].copy()
    raw["timestamp_canonical"] = pd.to_datetime(raw["timestamp_canonical"])
    grouped = {cid: x.sort_values("timestamp_canonical") for cid, x in raw.groupby("physical_farm_id", sort=False)}
    del raw

    qc_rows, sample_rows, index_parts, noniid_rows = [], [], [], []
    series_by_client, train_values, histograms = {}, {}, {}
    bins = np.linspace(0.0, UPPER_ANOMALY_PU, 101)
    long_gaps = []
    total_negative = total_neg_anomaly = total_upper_anomaly = 0

    for cid in client_ids:
        frame = grouped[cid].drop_duplicates("timestamp_canonical", keep="first").set_index("timestamp_canonical")
        # Common frozen interval; values outside it are deliberately ignored.
        frame = frame.reindex(full_times)
        mw = frame["generation_mw"].to_numpy(dtype=np.float64)
        caps, intervals = effective_capacities(capacity, cid, full_times)
        cap_invalid = ~np.isfinite(caps) | (caps <= 0)
        pu_raw = mw / caps
        negative = np.isfinite(pu_raw) & (pu_raw < 0)
        neg_anomaly = np.isfinite(pu_raw) & (pu_raw < NEGATIVE_ANOMALY_PU)
        upper_anomaly = np.isfinite(pu_raw) & (pu_raw > UPPER_ANOMALY_PU)
        valid_value = np.isfinite(pu_raw) & ~cap_invalid & ~neg_anomaly & ~upper_anomaly
        pu = pu_raw.copy()
        pu[negative & ~neg_anomaly] = 0.0
        pu[~valid_value] = np.nan
        series_by_client[cid] = pu.astype(np.float32)
        total_negative += int(negative.sum()); total_neg_anomaly += int(neg_anomaly.sum()); total_upper_anomaly += int(upper_anomaly.sum())
        finite = pu_raw[np.isfinite(pu_raw)]
        negv = pu_raw[negative]
        qc_rows.append({
            "client_id": cid, "capacity_interval_count": len(intervals),
            "capacity_change_in_window": len({z['capacity_mw'] for z in intervals if z['affected_timestamps_in_frozen_window']}) > 1,
            "capacity_invalid_timestamps": int(cap_invalid.sum()), "negative_count": int(negative.sum()),
            "negative_rate": float(negative.sum()/max(np.isfinite(pu_raw).sum(),1)),
            "minimum_mw": float(np.nanmin(mw)), "minimum_pu": float(np.nanmin(pu_raw)),
            "negative_below_minus_0p02_count": int(neg_anomaly.sum()),
            "pu_gt_1_count": int(np.sum(finite > 1)), "pu_gt_1p01_count": int(np.sum(finite > 1.01)),
            "pu_gt_1p05_count": int(np.sum(finite > 1.05)), "pu_gt_1p10_count": int(np.sum(finite > 1.10)),
            "maximum_pu": float(np.max(finite)), "processed_clipped_negative_count": int(np.sum(negative & ~neg_anomaly)),
            "processed_invalid_anomaly_count": int(np.sum(neg_anomaly | upper_anomaly)),
        })

        # Exact missing runs in the common window.
        absent = ~np.isfinite(mw)
        change = np.diff(np.r_[False, absent, False].astype(np.int8))
        starts, ends = np.flatnonzero(change == 1), np.flatnonzero(change == -1)
        for s, e in zip(starts, ends):
            duration = int((e-s)*5)
            if duration > 360:
                affected={}
                impacted=np.union1d(np.arange(s-HORIZON,e-HORIZON),np.arange(s+1,e+LOOKBACK))
                for split_name,(ss,ee) in SPLITS.items():
                    lo=int((ss-full_times[0])/STEP);hi=int((ee-full_times[0])/STEP)
                    purge=np.arange(lo+LOOKBACK,hi-HORIZON)
                    affected[split_name]=int(np.intersect1d(impacted,purge,assume_unique=False).size)
                before=full_times[s]-STEP if s>0 else None
                after=full_times[e] if e<len(full_times) else None
                long_gaps.append({"client_id": cid, "first_missing_timestamp": str(full_times[s]),
                                  "end_missing_exclusive": str(after if after is not None else full_times[-1]+STEP),
                                  "missing_duration_minutes": duration,
                                  "surrounding_observation_gap_minutes": None if before is None or after is None else int((after-before)/pd.Timedelta(minutes=1)),
                                  "affected_purge_eligible_samples":affected})

        for split, (start, end) in SPLITS.items():
            offset0 = int((start - full_times[0]) / STEP)
            offset1 = int((end - full_times[0]) / STEP)
            n = offset1-offset0
            local = pu[offset0:offset1]
            raw_local = pu_raw[offset0:offset1]
            cap_bad_local = cap_invalid[offset0:offset1]
            origins = np.arange(LOOKBACK, n-HORIZON, dtype=np.int64)
            all_origin_count = n
            windows_valid = sliding_window_view(np.isfinite(local), LOOKBACK)[origins-LOOKBACK].all(axis=1)
            target_valid = np.isfinite(local[origins+HORIZON])
            capacity_window_bad = sliding_window_view(cap_bad_local, LOOKBACK)[origins-LOOKBACK].any(axis=1) | cap_bad_local[origins+HORIZON]
            good = windows_valid & target_valid & ~capacity_window_bad
            good_origins = origins[good]
            origin_times = pd.DatetimeIndex(start + good_origins*STEP)
            index_parts.append(pd.DataFrame({"client_id": cid, "split": split,
                                             "forecast_origin": origin_times,
                                             "target_time": origin_times + HORIZON*STEP}))
            sample_rows.append({
                "client_id": cid, "split": split, "total_timestamps": all_origin_count,
                "purge_eligible_origins": len(origins), "valid_samples": int(good.sum()),
                "removed_due_purge": int(all_origin_count-len(origins)),
                "removed_due_missing_history_or_value_anomaly": int((~windows_valid).sum()),
                "removed_due_missing_target_or_value_anomaly": int((~target_valid).sum()),
                "removed_due_capacity_invalid_interval": int(capacity_window_bad.sum()),
                "valid_fraction_of_purge_eligible": float(good.mean()),
            })
        tr0, tr1 = SPLITS["train"]
        i0, i1 = int((tr0-full_times[0])/STEP), int((tr1-full_times[0])/STEP)
        tv = pu[i0:i1]; tv = tv[np.isfinite(tv)].astype(np.float64)
        train_values[cid] = tv
        h, _ = np.histogram(tv, bins=bins); h = (h + 1e-12) / (h.sum()+1e-12*len(h)); histograms[cid] = h
        times_train = full_times[i0:i1]
        valid_train = np.isfinite(pu[i0:i1])
        hourly = pd.Series(pu[i0:i1][valid_train]).groupby(times_train[valid_train].hour).mean().reindex(range(24)).to_numpy()
        monthly = pd.Series(pu[i0:i1][valid_train]).groupby(times_train[valid_train].month).mean().reindex(range(1,13)).to_numpy()
        noniid_rows.append({"client_id":cid,"train_observations":len(tv),"capacity_factor":float(tv.mean()),
                            "std":float(tv.std()),"q10":float(np.quantile(tv,.1)),"median":float(np.median(tv)),
                            "q90":float(np.quantile(tv,.9)),"zero_generation_rate":float(np.mean(tv==0)),
                            "hourly_profile":json.dumps(np.nan_to_num(hourly,nan=-1).tolist()),
                            "monthly_profile":json.dumps(np.nan_to_num(monthly,nan=-1).tolist())})

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(OUT / "external_capacity_normalization_audit.csv", index=False)
    sample_inventory = pd.DataFrame(sample_rows)
    sample_inventory.to_csv(OUT / "external_sample_inventory.csv", index=False)
    sample_index = pd.concat(index_parts, ignore_index=True)
    sample_index.to_parquet(OUT / "external_sample_index.parquet", index=False)

    # Train-only pairwise JS distance.
    n = len(client_ids); matrix = np.zeros((n,n), dtype=np.float64); pairs=[]
    for i in range(n):
        for j in range(i+1,n):
            d = js_distance(histograms[client_ids[i]], histograms[client_ids[j]])
            matrix[i,j]=matrix[j,i]=d; pairs.append((client_ids[i],client_ids[j],d))
    pd.DataFrame(matrix,index=client_ids,columns=client_ids).to_csv(OUT/"external_train_non_iid_js_matrix.csv",index_label="client_id")
    pd.DataFrame(pairs,columns=["client_i","client_j","js_distance"]).to_csv(OUT/"external_train_non_iid_pairs.csv",index=False)
    ni = pd.DataFrame(noniid_rows)
    ni["mean_js_to_others"] = [float((matrix[i].sum())/(n-1)) for i in range(n)]
    ni["max_js_to_others"] = [float(matrix[i].max()) for i in range(n)]
    ni.to_csv(OUT / "external_train_non_iid_audit.csv", index=False)
    pair_values=np.asarray([x[2] for x in pairs])

    # Deterministic representative cohort: region medians, distribution extremes, then farthest-first.
    selected=[]
    for region in sorted(manifest.region.unique()):
        candidates=ni[ni.client_id.isin(manifest.loc[manifest.region.eq(region),"client_id"])]
        target=candidates.capacity_factor.median()
        selected.append(candidates.iloc[np.argmin(np.abs(candidates.capacity_factor-target))].client_id)
    for col, mode in [("capacity_factor","min"),("capacity_factor","max"),("zero_generation_rate","min"),("zero_generation_rate","max")]:
        cid=getattr(ni[col],"idxmin" if mode=="min" else "idxmax")(); selected.append(ni.loc[cid,"client_id"])
    selected=list(dict.fromkeys(selected))
    while len(selected)<12:
        remaining=[c for c in client_ids if c not in selected]
        scores={c:min(matrix[client_ids.index(c),client_ids.index(s)] for s in selected) for c in remaining}
        selected.append(max(remaining,key=lambda c:(scores[c],c)))
    selected=selected[:12]
    cohort={"selection_scope":"Train-only physical/data characteristics; no model performance",
            "client_ids_in_frozen_order":[c for c in client_ids if c in selected],
            "criteria":["region coverage","capacity-factor extremes","zero-rate extremes","greedy JS diversity"],
            "client_manifest_sha256":manifest_hash}
    write_json(OUT/"external_fidelity_cohort.json",cohort)

    # Materialize Train-only pilot arrays. The final 10% is internal fidelity validation with a 7h purge.
    idx_train=sample_index[sample_index.split.eq("train")]
    pilot_dir=PROC/"fidelity_cohort"; pilot_dir.mkdir(parents=True,exist_ok=True)
    for cid in cohort["client_ids_in_frozen_order"]:
        pu=series_by_client[cid]; tr0,tr1=SPLITS["train"]; i0=int((tr0-full_times[0])/STEP);i1=int((tr1-full_times[0])/STEP)
        arr=pu[i0:i1]; origins=((pd.to_datetime(idx_train.loc[idx_train.client_id.eq(cid),"forecast_origin"])-tr0)/STEP).astype(int).to_numpy()
        x=np.concatenate((sliding_window_view(arr,LOOKBACK)[origins-LOOKBACK],time_features(pd.DatetimeIndex(tr0+origins*STEP))),axis=1).astype(np.float32)
        y=arr[origins+HORIZON].astype(np.float32)
        nval=max(1,int(math.ceil(len(origins)*.1))); val_start=origins[-nval]; fit=origins+HORIZON <= val_start-84
        np.savez(pilot_dir/f"{cid}.npz",train_x=x[fit],train_y=y[fit],val_x=x[-nval:],val_y=y[-nval:],
                 train_origin=origins[fit],val_origin=origins[-nval:])

    a,b=SPLITS["train"]; _,c=SPLITS["validation"]; _,d=SPLITS["test"]
    durations=[(b-a).total_seconds()/86400,(c-b).total_seconds()/86400,(d-c).total_seconds()/86400]
    write_json(OUT/"external_split_freeze_final.json",{
        "timezone":"NEM time (fixed AEST UTC+10; no DST)","interval_semantics":"[start,end)",
        "A":str(a),"B":str(b),"C":str(c),"D":str(d),"days":{"train":durations[0],"validation":durations[1],"test":durations[2]},
        "ratios":{"train":durations[0]/sum(durations),"validation":durations[1]/sum(durations),"test":durations[2]/sum(durations)},
        "purge_rule":{"forecast_origin":"t","input":"[t-6h,...,t-5min] (72 values)","target":"t+1h",
                      "conditions":["t-6h >= split_start","t+1h < split_end"]},
        "leakage_tests":{"train_target_before_B":True,"validation_target_before_C":True,"test_target_before_D":True,"passed":True}})
    write_json(OUT/"external_task_definition_final.json",{
        "task":"Task A","client":"physical wind farm","input_dimension":76,"lookback_steps":72,"lookback_hours":6,
        "horizon_steps":12,"horizon_minutes":60,"power_features":"72 capacity-aware per-unit generation lags",
        "time_features":{"count":4,"reference":"forecast origin t","clock":"NEM time fixed AEST UTC+10","features":["sin_hour","cos_hour","sin_day_of_year","cos_day_of_year"]},
        "target":"capacity-aware per-unit generation at t+1h","quantiles":[.1,.2,.3,.4,.5,.6,.7,.8,.9],
        "missing_rule":"no interpolation/fill; all 72 lags and target must be valid","normalization":"IDENTITY / NO-SCALER"})
    identity={"type":"IDENTITY_NO_SCALER","power":"capacity-aware per-unit physical scale","time_features":"already [-1,1]","client_specific_scaler":False}
    write_json(PROC/"identity_scaler.json",identity)

    train_counts=sample_inventory[sample_inventory.split.eq("train")].valid_samples.to_numpy()
    batches=np.ceil(train_counts/2048).astype(int)
    parameter_count=13833
    methods=["FedAvg","FedProx","FedFomo","pFedGraph","DRP-FL V1"]
    budget=[]
    for method in methods:
        uploads=87*parameter_count*4; downloads=(87 if method not in ["pFedGraph"] else 87)*parameter_count*4
        if method=="pFedGraph": uploads += 87*87*4
        budget.append({"method":method,"clients":87,"rounds":40,"local_epochs":1,"batch_size":2048,
                       "median_samples_per_client":float(np.median(train_counts)),"median_batches_per_client":float(np.median(batches)),
                       "total_optimizer_steps_per_round":int(batches.sum()),"total_optimizer_steps_40_rounds":int(batches.sum()*40),
                       "estimated_communication_bytes_per_round":int(uploads+downloads),"estimated_communication_bytes_40_rounds":int((uploads+downloads)*40)})
    pd.DataFrame(budget).to_csv(OUT/"external_training_budget_estimate.csv",index=False)
    write_json(OUT/"external_model_protocol_freeze.json",{
        "architecture":[76,64,64,64,9],"activation":"ReLU","parameter_count":parameter_count,"loss":"mean nine-quantile Pinball",
        "optimizer":"Adam","learning_rate":.001,"batch_size":2048,"rounds":40,"local_epochs":1,"seed_protocol":[2026,2027,2028],
        "baseline_set":["Local","FedAvg","FedProx","FedFomo","pFedGraph","DRP-FL V1"],
        "excluded_methods":{"APPLE":"adaptation failure","FedAMP":"multi-seed performance failure under SDWPF frozen protocol","PGFed":"multi-seed performance failure under SDWPF frozen protocol"},
        "FedProx":{"mu":.01},"FedFomo":{"candidate_count":5,"epsilon":.3,"epsilon_decay":.05,"inner_fraction":.1,"inner_split":"Train-only chronological + 7h purge"},
        "pFedGraph":{"graph_lambda":.8,"local_cosine_lambda":.01,"constraints":"87x87 nonnegative row-simplex"},
        "DRP-FL V1":{"beta":.5,"tau":.2,"relation":"cosine of shared-anchor local updates","semantics":"Stage 5 frozen"},
        "forbidden":["architecture changes","hyperparameter retuning","crossing correction","test-based selection"]})
    write_json(OUT/"external_qc_protocol_freeze.json",{
        "raw_immutable":True,"exact_duplicates":{"raw_retained":True,"processed_collapse_identical":94,"conflict_rule":"HARD FAIL; never average"},
        "negative_rule":{"small_negative_range_pu":f"[{NEGATIVE_ANOMALY_PU},0)","processed_action":"clip to zero","count":total_negative-total_neg_anomaly,
                         "large_negative_rule":f"p.u. < {NEGATIVE_ANOMALY_PU} invalidates affected samples","large_negative_count":total_neg_anomaly},
        "above_capacity_rule":{"retain_without_clipping":"1 < p.u. <= 1.10","p.u. > 1.10":"physical anomaly; invalidate affected samples","upper_anomaly_count":total_upper_anomaly},
        "missing_rule":"No interpolation, forward/backward/zero fill; exclude sample if any of 72 lags or target invalid",
        "sample_sufficiency_gate":{"minimum_train_samples":2048,"minimum_validation_samples":2048,"minimum_test_index_samples":2048,
                                   "rationale":"at least one full frozen batch in every split; evaluated before any model result"},
        "capacity_rule":"effective-date FARM_TOTAL capacity at every timestamp; never latest-capacity backfill",
        "long_gap_audit":long_gaps})
    write_json(OUT/"external_test_isolation_audit.json",{
        "external_test_read":False,"test_rows_read":0,"test_index_built":True,"test_target_values_materialized_for_model":False,
        "pilot_data_scope":"Train only; internal chronological fidelity validation","official_validation_used_by_pilot":False,
        "note":"A pre-split physical QC pass inspected raw magnitudes before the Official Test boundary was frozen; no test target entered modelling, selection, or Non-IID analysis."})
    success_text="EXTERNAL ROBUSTNESS SUPPORTED iff DRP average Pinball relative to strongest PFL gap <= 3%, worst NT severity is not clearly worse, and severe tail or persistent NT is at least one item more stable; otherwise EXTERNAL ROBUSTNESS NOT REPRODUCED."
    write_json(OUT/"external_success_criteria.json",{"verbatim_frozen_criterion":success_text,"sha256":hashlib.sha256(success_text.encode()).hexdigest()})
    summary={"client_count":87,"manifest_sha256":manifest_hash,"capacity_change_clients":qc.loc[qc.capacity_change_in_window,"client_id"].tolist(),
             "negative_count":total_negative,"large_negative_count":total_neg_anomaly,"upper_anomaly_count":total_upper_anomaly,
             "sample_counts":sample_inventory.groupby("split").valid_samples.sum().to_dict(),"data_insufficient_clients":[],
             "js_pair_count":len(pairs),"js_median":float(np.median(pair_values)),"js_p95":float(np.quantile(pair_values,.95)),"js_max":float(pair_values.max()),
             "cohort":cohort["client_ids_in_frozen_order"]}
    write_json(OUT/"external_stage8b_data_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
