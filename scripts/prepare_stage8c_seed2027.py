"""Prepare locked Train/Validation-only arrays and common initialization for Stage 8C."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from local_baseline import QuantileMLP,set_reproducible_seed
S8B=ROOT/"results"/"stage8b";S8A=ROOT/"results"/"stage8a2";OUT=ROOT/"results"/"stage8c"/"seed_2027";DATA=ROOT/"data"/"aemo"/"stage8c_seed2027"
A=pd.Timestamp("2025-08-13");B=pd.Timestamp("2026-04-19");C=pd.Timestamp("2026-06-20");STEP=pd.Timedelta(minutes=5)

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def state_hash(s):
 h=hashlib.sha256()
 for k,v in s.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
 freeze=json.loads((S8B/"external_stage8b_freeze_hashes.json").read_text())
 if not freeze["all_passed"] or freeze["test_read"] or freeze["test_rows_read"]:raise RuntimeError("STAGE 8B GATE HARD FAIL")
 for name in ["external_main_client_manifest.csv","external_qc_protocol_freeze.json","external_split_freeze_final.json","external_task_definition_final.json","external_sample_index.parquet"]:
  if sha(S8B/name)!=freeze["artifacts_sha256"][name]:raise RuntimeError(f"FROZEN HASH DRIFT: {name}")
 manifest=pd.read_csv(S8B/"external_main_client_manifest.csv").sort_values("client_order");ids=manifest.client_id.tolist()
 # Filter at source: no row at or after Official Test boundary C is loaded.
 raw=pd.read_parquet(S8A/"aemo_windfarm_generation_raw.parquet",columns=["timestamp_canonical","physical_farm_id","generation_mw"],
                     filters=[("timestamp_canonical","<",C)])
 raw=raw[raw.physical_farm_id.isin(ids)].copy();raw.timestamp_canonical=pd.to_datetime(raw.timestamp_canonical)
 index=pd.read_parquet(S8B/"external_sample_index.parquet",filters=[("split","in",["train","validation"])])
 times=pd.date_range(A,C-STEP,freq=STEP);inventory=pd.read_csv(S8B/"external_sample_inventory.csv")
 records=[]
 for row in manifest.itertuples():
  cid=row.client_id;q=raw[raw.physical_farm_id.eq(cid)].drop_duplicates("timestamp_canonical").set_index("timestamp_canonical").reindex(times)
  mw=q.generation_mw.to_numpy(float);caps=np.full(len(times),np.nan)
  for z in json.loads(row.capacity_history):
   s=pd.Timestamp(z["start"]);e=pd.Timestamp(z["end"]);mask=(times>=s)&(times<e);caps[mask]=float(z["capacity_mw"])
  pu=mw/caps;small=(pu<0)&(pu>=-.02);bad=(pu<-.02)|(pu>1.10)|~np.isfinite(pu)|~np.isfinite(caps)|(caps<=0);pu[small]=0;pu[bad]=np.nan
  ci=index[index.client_id.eq(cid)].copy();train=((pd.to_datetime(ci.loc[ci.split.eq("train"),"forecast_origin"])-A)/STEP).astype(np.int32).to_numpy();val=((pd.to_datetime(ci.loc[ci.split.eq("validation"),"forecast_origin"])-A)/STEP).astype(np.int32).to_numpy()
  expected_train=int(inventory[(inventory.client_id.eq(cid))&(inventory.split.eq("train"))].valid_samples.iloc[0]);expected_val=int(inventory[(inventory.client_id.eq(cid))&(inventory.split.eq("validation"))].valid_samples.iloc[0])
  if len(train)!=expected_train or len(val)!=expected_val:raise RuntimeError(f"SAMPLE INDEX DRIFT {cid}")
  cdir=DATA/cid;cdir.mkdir(exist_ok=True);np.save(cdir/"series_train_validation.npy",pu.astype(np.float32));np.save(cdir/"train_origins.npy",train);np.save(cdir/"validation_origins.npy",val)
  records.append({"client_order":row.client_order,"client_id":cid,"train_samples":len(train),"validation_samples":len(val),
                  "series_sha256":sha(cdir/"series_train_validation.npy"),"train_index_sha256":sha(cdir/"train_origins.npy"),"validation_index_sha256":sha(cdir/"validation_origins.npy")})
 pd.DataFrame(records).to_csv(DATA/"data_manifest.csv",index=False)
 set_reproducible_seed(2027,4);model=QuantileMLP();initial=model.state_dict();torch.save(initial,OUT/"initial_model_seed_2027.pt")
 provenance={"seed":2027,"clients":87,"initialization_file_sha256":sha(OUT/"initial_model_seed_2027.pt"),"initialization_semantic_sha256":state_hash(initial),
  "client_manifest_sha256":sha(S8B/"external_main_client_manifest.csv"),"data_manifest_sha256":sha(DATA/"data_manifest.csv"),
  "data_protocol_sha256":sha(S8B/"external_task_definition_final.json"),"split_sha256":sha(S8B/"external_split_freeze_final.json"),"qc_protocol_sha256":sha(S8B/"external_qc_protocol_freeze.json"),
  "sample_index_sha256":sha(S8B/"external_sample_index.parquet"),"stage8b_freeze_sha256":sha(S8B/"external_stage8b_freeze_hashes.json"),
  "test_boundary":str(C),"source_filter":"timestamp_canonical < C","external_test_read":False,"test_rows_read":0,"audit_passed":True}
 (OUT/"provenance_audit.json").write_text(json.dumps(provenance,indent=2)+"\n")
 print(json.dumps(provenance,indent=2))
if __name__=="__main__":main()
